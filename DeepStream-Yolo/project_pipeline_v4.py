import cv2
import numpy as np
import onnxruntime as ort
import time
from scipy.optimize import linear_sum_assignment

# ==========================================
# 1. TRACKER RAPID IOU (IN-MEMORY)
# ==========================================
class IOUTracker:
    def __init__(self, max_lost=10, iou_threshold=0.3):
        self.next_id = 1
        self.tracks = {} # {track_id: {"bbox": [x1,y1,x2,y2], "lost": int, "name": str, "emb_done": bool}}
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold

    def _compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(1e-5, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = max(1e-5, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        return interArea / float(boxAArea + boxBArea - interArea)

    def update(self, detections):
        # detections: list of [x1, y1, x2, y2, score]
        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid]["bbox"] for tid in track_ids]

        if len(track_boxes) == 0:
            for det in detections:
                self.tracks[self.next_id] = {"bbox": det[:4], "lost": 0, "name": "Necunoscut", "emb_done": False}
                self.next_id += 1
            return self.tracks

        # Matrice IOU pentru Hungarian Matching
        iou_matrix = np.zeros((len(track_boxes), len(detections)), dtype=np.float32)
        for t_idx, t_box in enumerate(track_boxes):
            for d_idx, d_box in enumerate(detections):
                iou_matrix[t_idx, d_idx] = self._compute_iou(t_box, d_box[:4])

        cost_matrix = 1.0 - iou_matrix
        row_inds, col_inds = linear_sum_assignment(cost_matrix)

        assigned_tracks = set()
        assigned_dets = set()

        for r, c in zip(row_inds, col_inds):
            if iou_matrix[r, c] >= self.iou_threshold:
                t_id = track_ids[r]
                self.tracks[t_id]["bbox"] = detections[c][:4]
                self.tracks[t_id]["lost"] = 0
                assigned_tracks.add(r)
                assigned_dets.add(c)

        # Tratează trackerele pierdute
        for t_idx, t_id in enumerate(track_ids):
            if t_idx not in assigned_tracks:
                self.tracks[t_id]["lost"] += 1
                if self.tracks[t_id]["lost"] > self.max_lost:
                    del self.tracks[t_id]

        # Adaugă detecțiile noi
        for d_idx, det in enumerate(detections):
            if d_idx not in assigned_dets:
                self.tracks[self.next_id] = {"bbox": det[:4], "lost": 0, "name": "Necunoscut", "emb_done": False}
                self.next_id += 1

        return self.tracks


# ==========================================
# 2. PIPELINE ONNX RUNTIME (YOLO + LANDMARKS + REC)
# ==========================================
class JetsonFacePipeline:
    def __init__(self, yolo_path, landmarks_path, rec_path):
        # Prioritizează TensorRT, apoi CUDA, apoi CPU
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        
        print("[INFO] Încărcare modele ONNX în GPU...")
        self.session_yolo = ort.InferenceSession(yolo_path, providers=providers)
        self.session_lmk = ort.InferenceSession(landmarks_path, providers=providers)
        self.session_rec = ort.InferenceSession(rec_path, providers=providers)

        # Baza de date (Nume -> vector embedding 512-d normalizat L2)
        # Poți adăuga persoane prin metoda add_person_to_db()
        self.face_db = {}
        self.similarity_threshold = 0.45 # Prag standard ArcFace cos_sim

    def add_person_to_db(self, name, embedding):
        norm_emb = embedding / np.linalg.norm(embedding)
        self.face_db[name] = norm_emb
        print(f"[DB] Persoana '{name}' a fost înregistrată în baza de date.")

    def preprocess_yolo(self, frame, input_size=(640, 640)):
        h, w, _ = frame.shape
        img_resized = cv2.resize(frame, input_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(img_tensor, axis=0), (w / input_size[0], h / input_size[1])

    def detect_faces(self, frame):
        tensor, (scale_x, scale_y) = self.preprocess_yolo(frame)
        input_name = self.session_yolo.get_inputs()[0].name
        outputs = self.session_yolo.run(None, {input_name: tensor})[0]
        
        # Parsare generică pentru YOLOv8-Face [batch, 5+landmarks, anchors]
        # (Ajustează post-procesarea dacă folosești alt format de ieșire YOLO)
        detections = []
        preds = np.squeeze(outputs)
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T

        for row in preds:
            conf = row[4] # confidence
            if conf > 0.5:
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                x1 = int((cx - w / 2) * scale_x)
                y1 = int((cy - h / 2) * scale_y)
                x2 = int((cx + w / 2) * scale_x)
                y2 = int((cy + h / 2) * scale_y)
                detections.append([max(0, x1), max(0, y1), x2, y2, float(conf)])
        return detections

    def get_landmarks_2d106(self, face_crop):
        # 2d106fdet necesită de obicei input 192x192 RGB normalizat
        resized = cv2.resize(face_crop, (192, 192))
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)
        
        input_name = self.session_lmk.get_inputs()[0].name
        lmks = self.session_lmk.run(None, {input_name: tensor})[0]
        return lmks # 106 coordonate puncte faciale

    def get_face_embedding(self, face_crop):
        # w600k_mbf (ArcFace/MobileFaceNet) necesită 112x112 RGB, normalizat [-1, 1]
        resized = cv2.resize(face_crop, (112, 112))
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_norm = (img_rgb.astype(np.float32) - 127.5) / 128.0
        tensor = img_norm.transpose(2, 0, 1)
        tensor = np.expand_dims(tensor, axis=0)

        input_name = self.session_rec.get_inputs()[0].name
        embedding = self.session_rec.run(None, {input_name: tensor})[0][0]
        return embedding / np.linalg.norm(embedding)

    def identify_face(self, embedding):
        best_name = "Necunoscut"
        max_sim = -1.0
        for name, db_emb in self.face_db.items():
            sim = np.dot(embedding, db_emb) # Cosine similarity pe vectori normalizați
            if sim > max_sim:
                max_sim = sim
                best_name = name
        
        if max_sim >= self.similarity_threshold:
            return best_name, max_sim
        return "Necunoscut", max_sim


# ==========================================
# 3. BUCLA PRINCIPALĂ (WEB CAM MJPEG -> PIPELINE)
# ==========================================
def main():
    # Inițializare pipeline cu căile către modele
    pipeline = JetsonFacePipeline(
        yolo_path="yolo_face.onnx",
        landmarks_path="2d106fdet.onnx",
        rec_path="w600k_mbf.onnx"
    )

    # EXEMPLU: Înregistrare vectori în baza de date in-memory
    # Poți genera un vector de start rulând pipeline.get_face_embedding(poza_crop)
    # pipeline.add_person_to_db("Stefan", np.random.rand(512))
    # pipeline.add_person_to_db("Vlad", np.random.rand(512))

    # Configurare Captură WebCam - MJPEG Explicit pe V4L2
    print("[INFO] Deschiem camera video în format MJPEG...")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[EROARE] Nu se poate deschide camera auto pe /dev/video0!")
        return

    tracker = IOUTracker(max_lost=15, iou_threshold=0.3)

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detecție fețe (YOLO)
        detections = pipeline.detect_faces(frame)

        # 2. Tracking ID continuu
        tracks = tracker.update(detections)

        # 3. Procesare selectivă pentru economie de resurse pe Jetson
        for tid, tdata in list(tracks.items()):
            x1, y1, x2, y2 = [int(v) for v in tdata["bbox"]]
            
            # Verifică granițele cadrului
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Rulăm inferența grea DOAR DACĂ ID-ul nu a fost recunoscut anterior
            if not tdata["emb_done"] and (x2 - x1) > 40 and (y2 - y1) > 40:
                face_crop = frame[y1:y2, x1:x2]
                
                # Excepțional: Extracție 106 puncte faciale (folosite pentru aliniere/afisare)
                _ = pipeline.get_landmarks_2d106(face_crop)
                
                # Recunoaștere facială (512-d ArcFace embedding)
                emb = pipeline.get_face_embedding(face_crop)
                nume_recunoscut, sim = pipeline.identify_face(emb)
                
                # Salvăm rezultatul în tracker (cache in-memory)
                tracker.tracks[tid]["name"] = nume_recunoscut
                tracker.tracks[tid]["emb_done"] = True # Marcăm ca procesat

            # Desenare Bounding Box & Nume persisenta din Tracker
            name_label = tdata["name"]
            color = (0, 255, 0) if name_label != "Necunoscut" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{tid} | {name_label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Afișare FPS pentru monitorizare performanță Jetson
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Jetson Xavier - Face Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()