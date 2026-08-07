import cv2
import numpy as np
import onnxruntime as ort
from scipy.spatial.distance import cosine
from sort import Sort

# 1. Baza de date (Dictionar)
# Pentru a-i recunoaște când revin în cadru.
# Aici vei înlocui np.random.randn cu embedding-urile reale generate prima dată.
known_faces = {
    "Stefan": np.random.randn(512), 
    "Vlad": np.random.randn(512),
    "Persoana_3": np.random.randn(512)
}

THRESHOLD_RECUNOASTERE = 0.55 # Ajustează în funcție de teste

def align_face(img, landmarks):
    """
    Funcție simplificată pentru alinierea feței pe baza a 106 puncte.
    În practică, se extrag 5 puncte de referință (ochi, nas, gură) și se face warpAffine.
    Aici returnăm un crop standard de 112x112 necesar pentru w600k_mbf.
    """
    # Placeholder pentru logica matematică de aliniere cv2.warpAffine
    resized = cv2.resize(img, (112, 112))
    return resized

def main():
    # 2. Inițializare modele ONNX (Asigură-te că folosești CUDA/TensorRT)
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    session_yolo = ort.InferenceSession("/workspace/v2/best.onnx", providers=providers)
    session_106 = ort.InferenceSession("/workspace/_landmark/2d106det.onnx", providers=providers)
    session_w600k = ort.InferenceSession("/workspace/_landmark/w600k_mbf.onnx", providers=providers)
    
    # 3. Inițializare Tracker Python (înlocuitorul logic pentru nvtracker aici)
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    
    # 4. GStreamer Pipeline pentru captura MJPEG (Hardware Accelerated)
    # nvjpegdec va decoda stream-ul folosind procesorul dedicat de imagine al Jetson-ului.
    gst_pipeline = (
        "v4l2src device=/dev/video0 ! "
        "image/jpeg, width=1280, height=720, framerate=30/1 ! "
        "nvjpegdec ! nvvidconv ! "
        "video/x-raw, format=BGRx ! videoconvert ! "
        "video/x-raw, format=BGR ! appsink"
    )
    
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("Eroare: Nu s-a putut deschide camera web.")
        return

    print("Pipeline pornit! Apasă 'q' pentru ieșire.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # --- A. DETECȚIE YOLO ---
        # (Preprocesare sumară: depinde de input-ul cerut de YOLO-ul tău - ex: 640x640)
        input_yolo = cv2.resize(frame, (960, 960))
        input_yolo = input_yolo.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_yolo = np.expand_dims(input_yolo, axis=0)
        
        # Rulare inferență YOLO
        # boxes_out va conține [x1, y1, x2, y2, scor, clasa]
        boxes_out = session_yolo.run(None, {session_yolo.get_inputs()[0].name: input_yolo})[0]
        
        # Filtrare sumară (presupunem că modelul scoate direct matricea utilă)
        detections = []
        for box in boxes_out[0]:
            if box[4] > 0.5: # Confidence threshold
                detections.append([box[0], box[1], box[2], box[3], box[4]])
                
        detections = np.array(detections) if len(detections) > 0 else np.empty((0, 5))
        
        # --- B. TRACKING ---
        # Pasăm cutiile detectate către tracker
        tracked_objects = tracker.update(detections)
        
        # --- C. LANDMARKS & RECUNOAȘTERE ---
        for track in tracked_objects:
            x1, y1, x2, y2, obj_id = [int(v) for v in track]
            
            # Asigurare limite (pentru a nu depăși rezoluția cadrului)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
                
            # 1. Landmarks (2d106fdet.onnx)
            # Modelul necesită un crop de dimensiune specifică (ex: 192x192)
            face_192 = cv2.resize(face_crop, (192, 192))
            input_106 = face_192.transpose(2, 0, 1).astype(np.float32) / 255.0
            input_106 = np.expand_dims(input_106, axis=0)
            
            landmarks_out = session_106.run(None, {session_106.get_inputs()[0].name: input_106})[0]
            
            # 2. Aliniere (Warp Affine)
            aligned_face = align_face(face_crop, landmarks_out)
            
            # 3. Recunoaștere (w600k_mbf.onnx)
            # Modelul buffalo_l de obicei ia format RGB (112, 112)
            input_w600k = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
            input_w600k = input_w600k.transpose(2, 0, 1).astype(np.float32) / 255.0
            input_w600k = np.expand_dims(input_w600k, axis=0)
            
            embedding = session_w600k.run(None, {session_w600k.get_inputs()[0].name: input_w600k})[0][0]
            
            # --- D. POTRIVIREA CU BAZA DE DATE ---
            nume_recunoscut = "Necunoscut"
            min_dist = float('inf')
            
            for nume, db_emb in known_faces.items():
                # Distanța cosinus (0.0 înseamnă potrivire perfectă)
                dist = cosine(embedding, db_emb) 
                if dist < min_dist:
                    min_dist = dist
                    if dist < THRESHOLD_RECUNOASTERE:
                        nume_recunoscut = nume

            # Desenare pe cadru
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{nume_recunoscut} (ID:{obj_id}) - {min_dist:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Afișare (notă: necesită X11 forwarding activat pe container)
        cv2.imshow("Jetson Face Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
