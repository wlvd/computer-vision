"""Varianta pentru fisiere video a pipeline-ului de recunoastere faciala.

Fata de complete_pipeline_tRT.py (camera live + fereastra pe ecran), aici:
  - sursa e un fisier video dat ca parametru;
  - iesirea nu e o fereastra, ci un folder numit dupa video, care contine
    videoclipul adnotat, baza de date rezultata si un log pe fiecare cadru;
  - landmark-urile se calculeaza pe FIECARE cadru, pentru toate fetele vizibile,
    ca sa poata fi desenate. Deciziile de recunoastere raman insa pe aceeasi
    cadenta ca la varianta live (fereastra de calitate + cel mai bun cadru),
    deci ce se vede in log e ce s-ar fi intamplat si pe camera.

Utilizare:
    python3 complete_pipeline_tRT_media.py sample_vid.mp4
    python3 complete_pipeline_tRT_media.py sample/sample_vid.mp4 --database baza.json

Numele videoclipului se cauta, in ordine: asa cum a fost dat, langa script, si
in subfolderul sample/ de langa script.
"""

import argparse
import json
import os
import re
import signal
import sys
import time
from collections import deque, Counter

import numpy as np
import cv2
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds
import tensorrt as trt
import pycuda.driver as cuda

# ============================================================
# CONFIGURARE
# ============================================================

if not hasattr(np, "bool"):
    np.bool = bool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_CONFIG_PATH = "config_infer_best_v2.txt"
TRACKER_CONFIG_PATH = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"

PFLD_MODEL_PATH = "/workspace/DeepStream-Yolo/pfld.engine"
RECOGNITION_MODEL_PATH = "/workspace/DeepStream-Yolo/w600k_mbf.engine"
FACE_DATABASE_PATH = "/workspace/DeepStream-Yolo/face_database.json"

MIN_CONFIDENCE = 0.5          # sub asta, nici nu incercam sa procesam fata
MIN_FACE_SIZE = 60            # px, latura minima a bbox-ului
MIN_BLUR = 100.0              # varianta Laplacianului; sub asta poza e miscata

VERIFY_INTERVAL_FRAMES = 15   # la cat timp cel mult re-verificam un track activ
RETRY_INTERVAL_ON_FAIL = 5    # daca poarta de calitate a picat, reincercam mai repede
LABEL_HISTORY_SIZE = 5        # cate decizii recente pastram pentru vot majoritar
TRACK_TIMEOUT_FRAMES = 300    # dupa cate cadre de absenta stergem un track din memorie
PRUNE_CHECK_INTERVAL = 90     # la cate cadre verificam track-uri "moarte"

QUALITY_WINDOW_FRAMES = 6     # cu cate cadre inainte de termen strangem probe
CANDIDATE_INTERVAL_FRAMES = 2 # la cate cadre luam o proba in fereastra
QUALITY_GOOD_ENOUGH = 0.65    # poza asa buna incat nu mai are rost sa asteptam
QUALITY_MIN = 0.30            # sub asta nu merita consumat modelul de recunoastere

QUALITY_WEIGHTS = {"yaw": 0.30, "pitch": 0.20, "roll": 0.10, "sharp": 0.20, "size": 0.20}

VERIFY_THRESHOLD = 0.42       # prag empiric, ajustat pe baza testelor offline

AUTO_ENROLL = True
ENROLL_MARGIN = 0.10          # banda de incertitudine sub pragul de recunoastere
ENROLL_MAX_SCORE = VERIFY_THRESHOLD - ENROLL_MARGIN
ENROLL_MIN_CHECKS = 4
ENROLL_MIN_FACE = 90
ENROLL_MIN_BLUR = 110.0
ENROLL_MIN_QUALITY = 0.55

LABEL_UNKNOWN = "necunoscut"
LABEL_UNCERTAIN = "incert"

ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)
ALIGN_SIZE = 112

# --- desen ---
# Cele 5 puncte ArcFace se deseneaza prin display meta (nvdsosd le randeaza
# garantat). Setul complet de landmark-uri se deseneaza direct pe suprafata, cu
# OpenCV -- merge pe Jetson (memorie unificata); daca suprafata nu e scriptibila
# se renunta la el si raman cele 5 puncte.
DRAW_ALL_LANDMARKS = True
LANDMARK_RADIUS = 2
POINT_COLORS = [           # ochi stang, ochi drept, nas, gura stanga, gura dreapta
    (0.0, 0.8, 1.0),
    (0.0, 0.8, 1.0),
    (0.2, 1.0, 0.2),
    (1.0, 0.4, 0.4),
    (1.0, 0.4, 0.4),
]
BOX_COLOR_KNOWN = (0.0, 1.0, 0.0, 1.0)
BOX_COLOR_UNCERTAIN = (1.0, 0.8, 0.0, 1.0)
BOX_COLOR_UNKNOWN = (1.0, 0.3, 0.3, 1.0)
BOX_COLOR_PENDING = (0.6, 0.6, 0.6, 1.0)

PROGRESS_EVERY_FRAMES = 100

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

cuda.init()
CUDA_CONTEXT = cuda.Device(0).retain_primary_context()

# Se completeaza in load_models(), nu la import: caile depind de argumente.
landmark_model = None
recognition_model = None
face_database = None
LANDMARK_POINTS = 0

# ============================================================
# MODELE TENSORRT
# ============================================================

class TrtModel:
    """Un .engine TensorRT, cu bufferele alocate o singura data, la batch maxim.

    Identic cu cel din complete_pipeline_tRT.py: engine-urile au batch 8, deci
    intr-un singur apel intra toate fetele unui cadru.
    """

    def __init__(self, engine_path):
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(
                f"Nu gasesc engine-ul: {engine_path}\n"
                f"Construieste-l din .onnx cu trtexec (vezi instructiunile)."
            )

        CUDA_CONTEXT.push()
        try:
            with open(engine_path, "rb") as f:
                runtime = trt.Runtime(TRT_LOGGER)
                self.engine = runtime.deserialize_cuda_engine(f.read())

            if self.engine is None:
                raise RuntimeError(
                    f"Engine-ul {engine_path} nu a putut fi deserializat: de obicei "
                    f"inseamna alta versiune de TensorRT sau alta placa."
                )

            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()

            self.dynamic = False
            self.max_batch = None
            for index in range(self.engine.num_bindings):
                if not self.engine.binding_is_input(index):
                    continue
                shape = tuple(self.engine.get_binding_shape(index))
                if shape[0] == -1:
                    self.dynamic = True
                    limit = int(self.engine.get_profile_shape(0, index)[2][0])
                else:
                    limit = int(shape[0])
                self.max_batch = limit if self.max_batch is None else min(self.max_batch, limit)
            self.max_batch = max(1, self.max_batch or 1)

            for index in range(self.engine.num_bindings):
                if not self.engine.binding_is_input(index):
                    continue
                shape = tuple(self.engine.get_binding_shape(index))
                if -1 in shape:
                    self.context.set_binding_shape(
                        index,
                        (self.max_batch,) + tuple(1 if d == -1 else d for d in shape[1:]),
                    )

            self.bindings = [0] * self.engine.num_bindings
            self.inputs, self.outputs = [], []

            for index in range(self.engine.num_bindings):
                shape = tuple(self.context.get_binding_shape(index))
                dtype = trt.nptype(self.engine.get_binding_dtype(index))
                host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
                host[:] = 0
                device = cuda.mem_alloc(host.nbytes)

                self.bindings[index] = int(device)
                entry = {
                    "index": index,
                    "name": self.engine.get_binding_name(index),
                    "shape": shape,
                    "sample_shape": shape[1:],
                    "sample_elems": int(np.prod(shape[1:])),
                    "dtype": dtype,
                    "host": host,
                    "device": device,
                }
                if self.engine.binding_is_input(index):
                    self.inputs.append(entry)
                else:
                    self.outputs.append(entry)

            self._current_batch = self.max_batch
        finally:
            CUDA_CONTEXT.pop()

        print(f"  {os.path.basename(engine_path)}: "
              f"intrare {self.inputs[0]['shape']} -> iesire {self.outputs[0]['shape']} "
              f"(batch max {self.max_batch}, {'dinamic' if self.dynamic else 'fix'})")

    def _set_batch(self, batch):
        if not self.dynamic or batch == self._current_batch:
            return
        for entry in self.inputs:
            self.context.set_binding_shape(entry["index"], (batch,) + entry["sample_shape"])
        self._current_batch = batch

    def infer_batch(self, array):
        """Ruleaza engine-ul pe n esantioane deodata (n <= max_batch)."""
        source = self.inputs[0]
        data = np.ascontiguousarray(array, dtype=source["dtype"])

        if data.ndim < 2:
            raise ValueError("infer_batch asteapta un tensor cu dimensiune de batch in fata.")
        count = int(data.shape[0])

        if not 1 <= count <= self.max_batch:
            raise ValueError(
                f"Batch de {count}, engine-ul suporta intre 1 si {self.max_batch}."
            )
        if data.size != count * source["sample_elems"]:
            raise ValueError(
                f"Intrare de {data.size} valori pentru {count} esantioane, "
                f"engine-ul asteapta {count * source['sample_elems']}."
            )

        CUDA_CONTEXT.push()
        try:
            self._set_batch(count)
            flat = data.ravel()
            source["host"][:flat.size] = flat
            cuda.memcpy_htod_async(source["device"], source["host"], self.stream)
            self.context.execute_async_v2(
                bindings=self.bindings, stream_handle=self.stream.handle
            )
            for entry in self.outputs:
                cuda.memcpy_dtoh_async(entry["host"], entry["device"], self.stream)
            self.stream.synchronize()
        finally:
            CUDA_CONTEXT.pop()

        return [
            entry["host"][: count * entry["sample_elems"]]
                 .reshape((count,) + entry["sample_shape"]).copy()
            for entry in self.outputs
        ]

    def infer(self, array):
        """Un singur esantion. Intoarce iesirile FARA dimensiunea de batch."""
        source = self.inputs[0]
        data = np.asarray(array).reshape((1,) + source["sample_shape"])
        return [out[0] for out in self.infer_batch(data)]


# Cate fete au trecut prin fiecare model si in cate apeluri -- pentru raportul final.
GPU_STATS = {"landmark_faces": 0, "landmark_calls": 0,
             "recognition_faces": 0, "recognition_calls": 0}


def run_batched(model, tensors, counter):
    """Ruleaza modelul in transe de cel mult max_batch, pastrand ordinea."""
    results = []
    for start in range(0, len(tensors), model.max_batch):
        chunk = np.stack(tensors[start:start + model.max_batch])
        results.extend(model.infer_batch(chunk)[0])
        GPU_STATS[counter + "_calls"] += 1
    GPU_STATS[counter + "_faces"] += len(tensors)
    return results


def l2(vec):
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-12 else vec

# ============================================================
# BAZA DE DATE
# ============================================================

class FaceDatabase:
    """Prototipurile, tinute ca o matrice (N, 512): cautarea e un singur matmul.

    Citeste si scrie acelasi JSON {nume: [512 float]} ca celelalte scripturi din
    W7, deci baza produsa aici poate fi folosita direct de pipeline-ul live.
    """

    def __init__(self, path, save_path=None):
        self.path = path
        self.save_path = save_path or path
        self.labels = []
        self.matrix = np.zeros((0, 0), dtype=np.float32)
        self.dirty = False
        self.source_count = 0

        if not path or not os.path.isfile(path):
            print(f"Baza de date de pornire nu exista ({path}); incep cu una goala.")
            return

        with open(path, "r") as f:
            raw = json.load(f)
        if raw:
            self.labels = list(raw)
            self.matrix = np.stack([l2(raw[label]) for label in self.labels])
        self.source_count = len(self.labels)

    def __len__(self):
        return len(self.labels)

    def add(self, label, embedding):
        vector = l2(embedding)[np.newaxis, :]
        self.matrix = vector if len(self.labels) == 0 else np.vstack([self.matrix, vector])
        self.labels.append(label)
        self.dirty = True

    def verify(self, embedding, threshold, margin):
        """(eticheta, scor). Intre prag-margine si prag raspunsul e LABEL_UNCERTAIN."""
        if not self.labels:
            return LABEL_UNKNOWN, -1.0

        scores = self.matrix @ np.asarray(embedding, dtype=np.float32)
        best = int(np.argmax(scores))
        best_score = float(scores[best])

        if best_score >= threshold:
            return self.labels[best], best_score
        if best_score >= threshold - margin:
            return LABEL_UNCERTAIN, best_score
        return LABEL_UNKNOWN, best_score

    def save(self):
        """Scrie in save_path (folderul rulcarii), nu peste baza de pornire."""
        tmp = self.save_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(
                {label: self.matrix[i].tolist() for i, label in enumerate(self.labels)}, f
            )
        os.replace(tmp, self.save_path)
        self.dirty = False


def load_models(landmark_path, recognition_path, database_path, database_out):
    """Incarca engine-urile si baza de date. Chemata din main, dupa argumente."""
    global landmark_model, recognition_model, face_database, LANDMARK_POINTS

    print("Incarc engine-urile TensorRT...")
    landmark_model = TrtModel(landmark_path)
    recognition_model = TrtModel(recognition_path)

    LANDMARK_POINTS = landmark_model.outputs[0]["sample_elems"] // 2
    print(f"Model landmark-uri: {LANDMARK_POINTS} puncte")

    face_database = FaceDatabase(database_path, database_out)
    print(f"Baza de date de pornire: {len(face_database)} identitati "
          f"{face_database.labels if len(face_database) <= 12 else ''}")

# ============================================================
# STARE PER TRACK
# ============================================================

class TrackState:
    def __init__(self, first_frame=0):
        self.last_checked_frame = -999999
        self.last_check_failed = False
        self.last_seen_frame = 0
        self.history = deque(maxlen=LABEL_HISTORY_SIZE)
        self.current_label = None
        self.current_score = 0.0
        self.unknown_streak = 0
        self.enrolled = False

        self.last_candidate_frame = -999999
        self.best_quality = -1.0
        self.best_aligned = None
        self.best_blur = 0.0
        self.best_size = 0

        # doar pentru raportul final
        self.first_frame = first_frame
        self.checks = 0

    @property
    def deadline_gap(self):
        return RETRY_INTERVAL_ON_FAIL if self.last_check_failed else VERIFY_INTERVAL_FRAMES

    def clear_best(self):
        self.best_quality = -1.0
        self.best_aligned = None
        self.best_blur = 0.0
        self.best_size = 0

    def offer(self, aligned, quality, blur, size):
        if quality > self.best_quality:
            self.best_quality = quality
            self.best_aligned = aligned
            self.best_blur = blur
            self.best_size = size


track_states = {}

# Istoricul complet al track-urilor, pentru summary.json: track_states se curata
# periodic, aici nu stergem nimic.
track_reports = {}


class FaceSample:
    """O fata vizibila in cadrul curent, cu tot ce s-a calculat pentru ea."""

    __slots__ = ("track_id", "state", "obj_meta", "crop", "blur", "size",
                 "box", "landmarks", "five_points", "aligned", "quality",
                 "is_candidate", "action")

    def __init__(self, track_id, state, obj_meta, crop, blur, size, box):
        self.track_id = track_id
        self.state = state
        self.obj_meta = obj_meta
        self.crop = crop
        self.blur = blur
        self.size = size
        self.box = box              # (x1, y1, x2, y2) in cadru
        self.landmarks = None       # toate punctele, in coordonate de cadru
        self.five_points = None     # cele 5 puncte ArcFace, in coordonate de cadru
        self.aligned = None
        self.quality = 0.0
        self.is_candidate = False
        self.action = "vazut"

# ============================================================
# PROCESARE
# ============================================================

def blur_score(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def preprocess_landmark(crop_bgr):
    img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (112, 112))
    tensor = img_resized.astype(np.float32) / 255.0
    return np.transpose(tensor, (2, 0, 1))


def postprocess_landmark(raw, crop_shape):
    h, w = crop_shape[:2]
    landmark_pixels = np.asarray(raw, dtype=np.float32).reshape(-1, 2).copy()
    landmark_pixels[:, 0] *= w
    landmark_pixels[:, 1] *= h
    return landmark_pixels


LANDMARK_LAYOUTS = {
    68:  {"left_eye": range(36, 42), "right_eye": range(42, 48), "nose": 30, "mouth": (48, 54)},
    98:  {"left_eye": range(60, 68), "right_eye": range(68, 76), "nose": 54, "mouth": (76, 82)},
    106: {"left_eye": 38, "right_eye": 88, "nose": 86, "mouth": (52, 61)},
}


def get_5_points(landmark):
    """Cele 5 puncte ArcFace, indiferent de markup-ul modelului de landmark-uri."""
    count = landmark.shape[0]
    layout = LANDMARK_LAYOUTS.get(count)
    if layout is None:
        raise ValueError(
            f"Modelul de landmark-uri scoate {count} puncte, iar maparea catre "
            f"cele 5 puncte ArcFace nu e definita. Markup-uri cunoscute: "
            f"{sorted(LANDMARK_LAYOUTS)}."
        )

    def take(index):
        return landmark[index].mean(axis=0) if isinstance(index, range) else landmark[index]

    left_mouth, right_mouth = layout["mouth"]
    return np.array(
        [take(layout["left_eye"]), take(layout["right_eye"]), take(layout["nose"]),
         landmark[left_mouth], landmark[right_mouth]],
        dtype=np.float32,
    )


def umeyama_similarity(src, dst):
    """Transformarea (rotatie + scalare + translatie) care duce src peste dst."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    num, dim = src.shape

    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_demean, dst_demean = src - src_mean, dst - dst_mean

    covariance = dst_demean.T @ src_demean / num
    signs = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(covariance) < 0:
        signs[dim - 1] = -1.0

    u, singular, vt = np.linalg.svd(covariance)
    matrix = np.eye(dim + 1, dtype=np.float64)
    matrix[:dim, :dim] = u @ np.diag(signs) @ vt

    variance = src_demean.var(axis=0).sum()
    if variance < 1e-12:
        return None

    scale = float(singular @ signs) / variance
    matrix[:dim, dim] = dst_mean - scale * (matrix[:dim, :dim] @ src_mean)
    matrix[:dim, :dim] *= scale
    return matrix[:dim, :]


def align_face(crop_bgr, five_points):
    transform_matrix = umeyama_similarity(five_points, ARCFACE_TEMPLATE)
    if transform_matrix is None or not np.all(np.isfinite(transform_matrix)):
        return None
    return cv2.warpAffine(crop_bgr, transform_matrix, (ALIGN_SIZE, ALIGN_SIZE), borderValue=0)


def preprocess_recognition(aligned_bgr):
    img_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    img_norm = (img_rgb.astype(np.float32) - 127.5) / 127.5
    return np.transpose(img_norm, (2, 0, 1))


def _nose_ratio(points):
    left_eye, right_eye, nose, left_mouth, right_mouth = points
    eye_vec = right_eye - left_eye
    interocular = float(np.linalg.norm(eye_vec))
    unit = eye_vec / interocular
    normal = np.array([-unit[1], unit[0]], dtype=np.float64)
    eye_center = (left_eye + right_eye) / 2.0
    mouth_center = (left_mouth + right_mouth) / 2.0
    height = float(np.dot(mouth_center - eye_center, normal))
    return float(np.dot(nose - eye_center, normal)) / height


NOSE_RATIO_FRONTAL = _nose_ratio(ARCFACE_TEMPLATE.astype(np.float64))


def face_quality(five_points, blur, min_side):
    """Cat de utilizabila e poza, in [0, 1]: frontalitate + claritate + marime."""
    left_eye, right_eye, nose, left_mouth, right_mouth = np.asarray(
        five_points, dtype=np.float64
    )

    eye_vec = right_eye - left_eye
    interocular = float(np.linalg.norm(eye_vec))
    if interocular < 1e-3:
        return 0.0

    unit = eye_vec / interocular
    normal = np.array([-unit[1], unit[0]], dtype=np.float64)
    eye_center = (left_eye + right_eye) / 2.0
    mouth_center = (left_mouth + right_mouth) / 2.0

    height = float(np.dot(mouth_center - eye_center, normal))
    if height <= 1e-3:
        return 0.0

    def clamp01(value):
        return float(min(1.0, max(0.0, value)))

    lateral = abs(float(np.dot(nose - eye_center, unit))) / interocular
    yaw = clamp01(1.0 - lateral / 0.25)

    ratio = float(np.dot(nose - eye_center, normal)) / height
    pitch = clamp01(1.0 - abs(ratio - NOSE_RATIO_FRONTAL) / 0.25)

    roll_deg = abs(np.degrees(np.arctan2(float(eye_vec[1]), float(eye_vec[0]))))
    roll = clamp01(1.0 - min(roll_deg, 180.0 - roll_deg) / 30.0)

    sharp = clamp01(blur / (2.0 * MIN_BLUR))
    size = clamp01(min_side / float(ENROLL_MIN_FACE))

    w = QUALITY_WEIGHTS
    return (w["yaw"] * yaw + w["pitch"] * pitch + w["roll"] * roll
            + w["sharp"] * sharp + w["size"] * size)


def analyse_faces(faces):
    """Landmark-uri (un singur apel GPU) + aliniere + calitate, pentru tot cadrul.

    Completeaza direct campurile din FaceSample. Se ruleaza pe toate fetele
    vizibile, nu doar pe candidati, fiindca landmark-urile se si deseneaza.
    """
    if not faces:
        return

    raw_landmarks = run_batched(
        landmark_model, [preprocess_landmark(f.crop) for f in faces], "landmark"
    )

    for face, raw in zip(faces, raw_landmarks):
        landmark = postprocess_landmark(raw, face.crop.shape)
        five = get_5_points(landmark)

        offset = np.array([face.box[0], face.box[1]], dtype=np.float32)
        face.landmarks = landmark + offset
        face.five_points = five + offset

        face.aligned = align_face(face.crop, five)
        if face.aligned is not None:
            face.quality = face_quality(five, face.blur, face.size)


def embed_aligned(aligned_faces):
    if not aligned_faces:
        return []
    raw = run_batched(
        recognition_model, [preprocess_recognition(a) for a in aligned_faces], "recognition"
    )
    return [l2(emb) for emb in raw]


def verify_embedding(embedding):
    return face_database.verify(embedding, VERIFY_THRESHOLD, ENROLL_MARGIN)


def enroll(embedding):
    used = {int(m.group(1)) for m in
            (re.match(r"persoana_(\d+)$", label) for label in face_database.labels) if m}
    name = f"persoana_{max(used, default=0) + 1}"
    face_database.add(name, embedding)
    print(f"[INROLARE] identitate noua: {name} (total {len(face_database)})")
    return name

# ============================================================
# DESEN
# ============================================================

_surface_draw_warned = [False]


def draw_five_points(batch_meta, frame_meta, faces):
    """Cele 5 puncte ArcFace, ca cercuri in display meta (le randeaza nvdsosd).

    Un display meta duce cel mult MAX_ELEMENTS_IN_DISPLAY_META cercuri, deci
    pentru mai multe fete se cer mai multe din pool.
    """
    limit = pyds.MAX_ELEMENTS_IN_DISPLAY_META
    pending = []
    for face in faces:
        if face.five_points is None:
            continue
        for index, (x, y) in enumerate(face.five_points):
            pending.append((int(x), int(y), POINT_COLORS[index]))

    for start in range(0, len(pending), limit):
        chunk = pending[start:start + limit]
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_circles = len(chunk)
        for i, (x, y, color) in enumerate(chunk):
            circle = display_meta.circle_params[i]
            circle.xc = x
            circle.yc = y
            circle.radius = LANDMARK_RADIUS + 1
            circle.circle_color.set(color[0], color[1], color[2], 1.0)
            circle.has_bg_color = 1
            circle.bg_color.set(color[0], color[1], color[2], 1.0)
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)


def draw_all_landmarks(surface_rgba, faces):
    """Setul complet de puncte, direct pe suprafata (OpenCV).

    Merge pe Jetson, unde suprafata mapata e chiar memoria pe care o citeste
    encoder-ul. Daca nu se poate scrie in ea, renuntam: cele 5 puncte desenate
    prin display meta raman oricum vizibile.
    """
    try:
        for face in faces:
            if face.landmarks is None:
                continue
            for x, y in face.landmarks:
                cv2.circle(surface_rgba, (int(x), int(y)), LANDMARK_RADIUS,
                           (255, 255, 0, 255), -1)
    except Exception as error:      # suprafata nescriptibila / alt layout
        if not _surface_draw_warned[0]:
            _surface_draw_warned[0] = True
            print(f"[AVERTISMENT] nu pot desena pe suprafata ({error}); "
                  f"raman doar cele 5 puncte ArcFace.")


def box_color(state):
    if state.current_label is None:
        return BOX_COLOR_PENDING
    if state.current_label == LABEL_UNKNOWN:
        return BOX_COLOR_UNKNOWN
    if state.current_label == LABEL_UNCERTAIN:
        return BOX_COLOR_UNCERTAIN
    return BOX_COLOR_KNOWN


def label_object(obj_meta, state):
    """Scrie eticheta si culoarea casetei peste ce deseneaza nvdsosd implicit."""
    if state.current_label:
        text = f"{state.current_label} ({state.current_score:.2f})"
    else:
        text = "verificare..."

    obj_meta.text_params.display_text = text
    obj_meta.text_params.set_bg_clr = 1
    obj_meta.text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)
    obj_meta.text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
    obj_meta.text_params.font_params.font_size = 12

    color = box_color(state)
    obj_meta.rect_params.border_color.set(*color)
    obj_meta.rect_params.border_width = 2

# ============================================================
# LOG
# ============================================================

def r(value, digits=3):
    return round(float(value), digits)


class FrameLogger:
    """Un rand JSON pe cadru, in frames.jsonl."""

    def __init__(self, path):
        self.path = path
        self.handle = open(path, "w", encoding="utf-8")
        self.frames = 0

    def write(self, record):
        self.handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        self.handle.write("\n")
        self.frames += 1
        if self.frames % 50 == 0:
            self.handle.flush()

    def close(self):
        self.handle.flush()
        self.handle.close()


class Run:
    """Starea rularii curente: unde scriem, ce s-a intamplat pana acum."""

    def __init__(self, args, output_dir, video_path):
        self.args = args
        self.output_dir = output_dir
        self.video_path = video_path
        self.logger = FrameLogger(os.path.join(output_dir, "frames.jsonl"))
        self.frames = 0
        self.faces_seen = 0
        self.recognitions = 0
        self.enrollments = []
        self.probe_ms = []
        self.started = time.time()


RUN = None

# ============================================================
# PROBE PRINCIPAL
# ============================================================

def media_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        started = time.perf_counter()
        frame_number = frame_meta.frame_num

        surface = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        try:
            # Copie pentru crop-uri: pe suprafata originala desenam la final, iar
            # crop-urile trebuie sa fie curate cand ajung la modele.
            frame_image = cv2.cvtColor(
                np.array(surface, copy=True, order='C'), cv2.COLOR_RGBA2BGR
            )
            frame_h, frame_w = frame_image.shape[:2]

            record = process_frame(batch_meta, frame_meta, frame_number,
                                  frame_image, frame_w, frame_h, surface)
        finally:
            pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        RUN.probe_ms.append(elapsed_ms)
        record["ms"] = r(elapsed_ms, 2)
        RUN.logger.write(record)
        RUN.frames += 1

        if PROGRESS_EVERY_FRAMES and frame_number % PROGRESS_EVERY_FRAMES == 0:
            print(f"  cadru {frame_number}: {len(record['faces'])} fete, "
                  f"{len(face_database)} identitati, {elapsed_ms:.1f} ms")

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def process_frame(batch_meta, frame_meta, frame_number, frame_image,
                  frame_w, frame_h, surface):
    """Toata logica pe un cadru. Intoarce inregistrarea pentru log."""
    pts_seconds = frame_meta.buf_pts / 1e9 if frame_meta.buf_pts else 0.0

    if frame_number % PRUNE_CHECK_INTERVAL == 0:
        stale = [tid for tid, st in track_states.items()
                 if frame_number - st.last_seen_frame > TRACK_TIMEOUT_FRAMES]
        for tid in stale:
            del track_states[tid]

    # --- Pasul 1: strangem fetele vizibile, fara sa atingem GPU-ul ---
    faces = []
    rejected = []
    active = {}

    l_obj = frame_meta.obj_meta_list
    while l_obj is not None:
        try:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
        except StopIteration:
            break

        track_id = int(obj_meta.object_id)
        confidence = float(obj_meta.confidence)
        rect = obj_meta.rect_params

        x1 = max(0, int(rect.left))
        y1 = max(0, int(rect.top))
        x2 = min(frame_w, int(rect.left + rect.width))
        y2 = min(frame_h, int(rect.top + rect.height))
        w, h = x2 - x1, y2 - y1

        if confidence < MIN_CONFIDENCE:
            rejected.append({"track": track_id, "box": [x1, y1, x2, y2],
                             "det": r(confidence), "poarta": "incredere_mica"})
        elif w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            rejected.append({"track": track_id, "box": [x1, y1, x2, y2],
                             "det": r(confidence), "poarta": "prea_mica"})
        else:
            state = track_states.setdefault(track_id, TrackState(frame_number))
            state.last_seen_frame = frame_number
            active[track_id] = state
            report = track_reports.setdefault(
                track_id, {"primul_cadru": frame_number, "verificari": 0,
                           "eticheta": None, "scor": 0.0, "inrolat": False}
            )
            report["ultimul_cadru"] = frame_number

            crop = frame_image[y1:y2, x1:x2]
            faces.append(FaceSample(track_id, state, obj_meta, crop,
                                    blur_score(crop), min(w, h), (x1, y1, x2, y2)))

        try:
            l_obj = l_obj.next
        except StopIteration:
            break

    RUN.faces_seen += len(faces)

    # --- Pasul 2: landmark-uri + aliniere + calitate, intr-un singur apel GPU ---
    analyse_faces(faces)

    # --- Pasul 3: care fete intra in concursul pentru "cel mai bun cadru" ---
    # Aceleasi reguli ca la varianta live, ca deciziile sa fie comparabile:
    # fereastra dinaintea termenului, o proba la cateva cadre, poarta de blur.
    for face in faces:
        state = face.state
        frames_since_check = frame_number - state.last_checked_frame
        window_open = frames_since_check >= state.deadline_gap - QUALITY_WINDOW_FRAMES
        due_for_sample = frame_number - state.last_candidate_frame >= CANDIDATE_INTERVAL_FRAMES

        if not (window_open and due_for_sample):
            continue
        if face.blur < MIN_BLUR:
            face.action = "sarit_blur"
            state.last_candidate_frame = frame_number
            continue
        if face.aligned is None:
            face.action = "aliniere_esuata"
            state.last_candidate_frame = frame_number
            continue

        state.last_candidate_frame = frame_number
        face.is_candidate = True
        face.action = "proba"
        state.offer(face.aligned, face.quality, face.blur, face.size)

    # --- Pasul 4: pentru cine rulam recunoasterea in acest cadru ---
    to_recognize = []
    for track_id, state in active.items():
        due = frame_number - state.last_checked_frame >= state.deadline_gap
        usable = state.best_aligned is not None and state.best_quality >= QUALITY_MIN
        good_enough = state.best_quality >= QUALITY_GOOD_ENOUGH

        if usable and (good_enough or due):
            state.last_checked_frame = frame_number
            state.last_check_failed = False
            to_recognize.append((track_id, state))
        elif due:
            state.last_checked_frame = frame_number
            state.last_check_failed = True
            state.clear_best()

    # --- Pasul 5: un singur apel GPU pentru toate fetele alese ---
    embeddings = embed_aligned([state.best_aligned for _, state in to_recognize])

    # --- Pasul 6: verificare si inrolare, secvential ---
    decisions = {}
    for (track_id, state), embedding in zip(to_recognize, embeddings):
        label, score = verify_embedding(embedding)
        quality, blur, size = state.best_quality, state.best_blur, state.best_size
        state.clear_best()
        state.checks += 1
        RUN.recognitions += 1

        enrolled_now = False
        if AUTO_ENROLL and label == LABEL_UNKNOWN and not state.enrolled:
            state.unknown_streak += 1
            if (state.unknown_streak >= ENROLL_MIN_CHECKS
                    and score < ENROLL_MAX_SCORE
                    and size >= ENROLL_MIN_FACE
                    and blur >= ENROLL_MIN_BLUR
                    and quality >= ENROLL_MIN_QUALITY):
                label = enroll(embedding)
                score = 1.0
                enrolled_now = True
                state.enrolled = True
                state.history.clear()
                RUN.enrollments.append({"nume": label, "cadru": frame_number,
                                        "track": track_id, "calitate": r(quality)})
        else:
            state.unknown_streak = 0

        state.history.append(label)
        state.current_label = Counter(state.history).most_common(1)[0][0]
        state.current_score = score

        report = track_reports[track_id]
        report["verificari"] = state.checks
        report["eticheta"] = state.current_label
        report["scor"] = r(score)
        report["inrolat"] = report["inrolat"] or enrolled_now

        decisions[track_id] = {
            "eticheta_bruta": label, "scor": r(score), "calitate": r(quality),
            "blur": r(blur, 1), "inrolat": enrolled_now,
        }

        if state.current_label not in (LABEL_UNKNOWN, LABEL_UNCERTAIN):
            print(f"[ALERTA] track {track_id} -> {state.current_label} "
                  f"(scor={score:.3f}, calitate={quality:.2f}, cadru={frame_number})")

    # --- Pasul 7: desen ---
    for face in faces:
        if face.track_id in decisions:
            face.action = "recunoscut"
        label_object(face.obj_meta, face.state)

    draw_five_points(batch_meta, frame_meta, faces)
    if DRAW_ALL_LANDMARKS:
        draw_all_landmarks(surface, faces)

    # --- Pasul 8: inregistrarea pentru log ---
    face_records = []
    for face in faces:
        state = face.state
        entry = {
            "track": face.track_id,
            "box": [int(v) for v in face.box],
            "det": r(float(face.obj_meta.confidence)),
            "blur": r(face.blur, 1),
            "calitate": r(face.quality),
            "actiune": face.action,
            "eticheta": state.current_label,
            "scor": r(state.current_score),
        }
        if face.five_points is not None:
            entry["puncte5"] = [[r(x, 1), r(y, 1)] for x, y in face.five_points]
        if face.track_id in decisions:
            entry["decizie"] = decisions[face.track_id]
        face_records.append(entry)

    return {
        "cadru": frame_number,
        "timp": r(pts_seconds),
        "faces": face_records,
        "respinse": rejected,
        "gpu": {"landmark": len(faces), "recunoastere": len(to_recognize)},
        "identitati": len(face_database),
    }

# ============================================================
# PIPELINE GSTREAMER
# ============================================================

def make_element(factory_names, name):
    """Primul element disponibil din lista (Jetson vs. desktop au alte encodere)."""
    if isinstance(factory_names, str):
        factory_names = [factory_names]
    for factory in factory_names:
        element = Gst.ElementFactory.make(factory, name)
        if element:
            if len(factory_names) > 1:
                print(f"  {name}: {factory}")
            return element
    raise RuntimeError(
        f"Niciunul dintre elementele {factory_names} nu e disponibil "
        f"(lipseste un plugin GStreamer?)."
    )


def on_pad_added(decodebin, pad, target):
    """Legam doar pad-ul video al lui uridecodebin."""
    caps = pad.get_current_caps() or pad.query_caps()
    name = caps.to_string()
    if not name.startswith("video/"):
        return

    sinkpad = target.get_static_pad("sink")
    if sinkpad.is_linked():
        return
    if pad.link(sinkpad) != Gst.PadLinkReturn.OK:
        print(f"Eroare: nu pot lega sursa la conversie ({name.split(',')[0]}).")


def build_pipeline(video_path, output_video, args):
    pipeline = Gst.Pipeline()
    if not pipeline:
        raise RuntimeError("Nu s-a putut crea pipeline-ul.")

    print("Creare elemente pipeline...")

    source = make_element("uridecodebin", "sursa")
    source.set_property("uri", Gst.filename_to_uri(video_path))

    # Convertorul de dupa sursa accepta si memorie de sistem (decodare software),
    # si NVMM (decodare hardware), deci merge indiferent ce alege decodebin.
    vidconv_in = make_element("nvvideoconvert", "conversie-intrare")
    caps_in = make_element("capsfilter", "caps-intrare")
    caps_in.set_property("caps", Gst.Caps.from_string(
        "video/x-raw(memory:NVMM), format=NV12"))

    streammux = make_element("nvstreammux", "stream-muxer")
    streammux.set_property('width', args.width)
    streammux.set_property('height', args.height)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 40000)
    streammux.set_property('live-source', 0)

    pgie = make_element("nvinfer", "detector-fete")
    pgie.set_property('config-file-path', args.pgie_config)

    tracker = make_element("nvtracker", "tracker")
    tracker.set_property('tracker-width', 640)
    tracker.set_property('tracker-height', 384)
    tracker.set_property('gpu-id', 0)
    tracker.set_property('ll-lib-file',
                         "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property('ll-config-file', args.tracker_config)

    vidconv_osd = make_element("nvvideoconvert", "conversie-osd")
    caps_rgba = make_element("capsfilter", "caps-rgba")
    caps_rgba.set_property('caps', Gst.Caps.from_string(
        "video/x-raw(memory:NVMM), format=RGBA"))

    nvosd = make_element("nvdsosd", "osd")

    vidconv_out = make_element("nvvideoconvert", "conversie-iesire")
    caps_out = make_element("capsfilter", "caps-iesire")

    encoder = make_element(["nvv4l2h264enc", "x264enc"], "encoder")
    if encoder.get_factory().get_name() == "nvv4l2h264enc":
        encoder.set_property('bitrate', args.bitrate)
        caps_out.set_property('caps', Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), format=NV12"))
    else:
        # cale de rezerva pe desktop: encoder software, deci memorie de sistem
        encoder.set_property('bitrate', max(1, args.bitrate // 1000))
        encoder.set_property('speed-preset', 'ultrafast')
        caps_out.set_property('caps', Gst.Caps.from_string(
            "video/x-raw, format=I420"))

    parser = make_element("h264parse", "parser")
    muxer = make_element("qtmux", "muxer")
    sink = make_element("filesink", "filesink")
    sink.set_property('location', output_video)
    sink.set_property('sync', False)
    sink.set_property('async', False)

    elements = [source, vidconv_in, caps_in, streammux, pgie, tracker,
                vidconv_osd, caps_rgba, nvosd, vidconv_out, caps_out,
                encoder, parser, muxer, sink]
    for element in elements:
        pipeline.add(element)

    print("Legare elemente pipeline...")
    # uridecodebin isi creeaza pad-ul abia cand afla ce contine fisierul
    source.connect("pad-added", on_pad_added, vidconv_in)

    vidconv_in.link(caps_in)
    sinkpad = streammux.get_request_pad("sink_0")
    caps_in.get_static_pad("src").link(sinkpad)

    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(vidconv_osd)
    vidconv_osd.link(caps_rgba)
    caps_rgba.link(nvosd)
    nvosd.link(vidconv_out)
    vidconv_out.link(caps_out)
    caps_out.link(encoder)
    encoder.link(parser)
    parser.link(muxer)
    muxer.link(sink)

    caps_rgba.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, media_probe, 0)
    return pipeline

# ============================================================
# RAPORT FINAL
# ============================================================

def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def write_summary(run, args, output_dir):
    durations = run.probe_ms
    identities = {}
    for track_id, report in track_reports.items():
        label = report["eticheta"] or "fara_decizie"
        identities.setdefault(label, []).append(track_id)

    summary = {
        "video": run.video_path,
        "folder": output_dir,
        "cadre": run.frames,
        "fete_procesate": run.faces_seen,
        "recunoasteri": run.recognitions,
        "durata_rulare_s": r(time.time() - run.started, 1),
        "baza_de_date": {
            "pornire": args.database,
            "identitati_initiale": face_database.source_count,
            "identitati_finale": len(face_database),
            "inrolari": run.enrollments,
        },
        "gpu": dict(GPU_STATS),
        "apeluri_economisite": {
            "landmark": GPU_STATS["landmark_faces"] - GPU_STATS["landmark_calls"],
            "recunoastere": GPU_STATS["recognition_faces"] - GPU_STATS["recognition_calls"],
        },
        "timp_probe_ms": {
            "mediu": r(sum(durations) / len(durations), 2) if durations else 0.0,
            "p50": r(percentile(durations, 0.50), 2),
            "p95": r(percentile(durations, 0.95), 2),
            "maxim": r(max(durations), 2) if durations else 0.0,
        },
        "track_uri": track_reports,
        "identitati": {label: sorted(tracks) for label, tracks in identities.items()},
        "praguri": {
            "MIN_CONFIDENCE": MIN_CONFIDENCE, "MIN_FACE_SIZE": MIN_FACE_SIZE,
            "MIN_BLUR": MIN_BLUR, "VERIFY_THRESHOLD": VERIFY_THRESHOLD,
            "ENROLL_MARGIN": ENROLL_MARGIN, "ENROLL_MAX_SCORE": ENROLL_MAX_SCORE,
            "QUALITY_GOOD_ENOUGH": QUALITY_GOOD_ENOUGH, "QUALITY_MIN": QUALITY_MIN,
            "ENROLL_MIN_QUALITY": ENROLL_MIN_QUALITY,
            "VERIFY_INTERVAL_FRAMES": VERIFY_INTERVAL_FRAMES,
        },
    }

    path = os.path.join(output_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary

# ============================================================
# MAIN
# ============================================================

def resolve_video(name):
    """Cauta fisierul: asa cum a fost dat, langa script, apoi in sample/."""
    candidates = [
        name,
        os.path.join(SCRIPT_DIR, name),
        os.path.join(SCRIPT_DIR, "sample", name),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        "Nu gasesc videoclipul. Am cautat:\n  " + "\n  ".join(candidates)
    )


def resolve_config(path, what):
    """Configurile se cauta in folderul curent, apoi langa script.

    Pipeline-ul live foloseste cai relative (se ruleaza din /workspace/DeepStream-Yolo),
    dar scriptul asta poate fi pornit de oriunde. Verificam existenta acum, nu cand
    porneste nvinfer: acolo eroarea e mult mai greu de citit.
    """
    candidates = [path, os.path.join(SCRIPT_DIR, os.path.basename(path))]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        f"Nu gasesc {what}. Am cautat:\n  " + "\n  ".join(candidates)
    )


def prepare_output_dir(video_path, args):
    """Folderul cu numele videoclipului. Nu suprascrie o rulare anterioara."""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    base = args.output or os.path.join(SCRIPT_DIR, stem)

    if args.force or not os.path.exists(base):
        os.makedirs(base, exist_ok=True)
        return base

    index = 2
    while os.path.exists(f"{base}_{index}"):
        index += 1
    target = f"{base}_{index}"
    os.makedirs(target)
    print(f"{base} exista deja; scriu in {target} (foloseste --force ca sa suprascrii).")
    return target


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Ruleaza pipeline-ul de recunoastere faciala pe un fisier video "
                    "si scrie video adnotat + baza de date + log pe cadru.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video",
                        help="numele fisierului video (langa script sau in sample/)")
    parser.add_argument("--output", default=None,
                        help="folderul de iesire (implicit: numele videoclipului)")
    parser.add_argument("--database", default=FACE_DATABASE_PATH,
                        help="baza de date de pornire; nu e modificata, rezultatul "
                             "se scrie in folderul de iesire")
    parser.add_argument("--pgie-config", default=YOLO_CONFIG_PATH,
                        help="configul nvinfer pentru detectorul de fete")
    parser.add_argument("--tracker-config", default=TRACKER_CONFIG_PATH,
                        help="configul low-level pentru nvtracker")
    parser.add_argument("--landmark-engine", default=PFLD_MODEL_PATH)
    parser.add_argument("--recognition-engine", default=RECOGNITION_MODEL_PATH)
    parser.add_argument("--width", type=int, default=1920,
                        help="latimea de lucru; pragurile de blur si de marime "
                             "sunt calibrate pe 1080p")
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--bitrate", type=int, default=4000000,
                        help="bitrate-ul videoclipului de iesire, in biti/s")
    parser.add_argument("--no-enroll", action="store_true",
                        help="nu adauga identitati noi in baza de date")
    parser.add_argument("--no-landmarks", action="store_true",
                        help="deseneaza doar cele 5 puncte ArcFace, nu tot setul")
    parser.add_argument("--force", action="store_true",
                        help="scrie peste folderul de iesire daca exista deja")
    return parser.parse_args(argv)


def main():
    global AUTO_ENROLL, DRAW_ALL_LANDMARKS, RUN

    args = parse_args()
    AUTO_ENROLL = not args.no_enroll
    DRAW_ALL_LANDMARKS = not args.no_landmarks

    video_path = resolve_video(args.video)
    args.pgie_config = resolve_config(args.pgie_config, "configul nvinfer (detectorul de fete)")
    args.tracker_config = resolve_config(args.tracker_config, "configul nvtracker")

    output_dir = prepare_output_dir(video_path, args)
    output_video = os.path.join(
        output_dir, os.path.splitext(os.path.basename(video_path))[0] + "_adnotat.mp4"
    )

    print(f"Video:  {video_path}")
    print(f"Iesire: {output_dir}")

    load_models(args.landmark_engine, args.recognition_engine,
                args.database, os.path.join(output_dir, "face_database.json"))

    Gst.init(None)
    RUN = Run(args, output_dir, video_path)

    pipeline = build_pipeline(video_path, output_video, args)
    loop = GLib.MainLoop()

    def bus_call(bus, message, loop):
        message_type = message.type
        if message_type == Gst.MessageType.EOS:
            print("Sfarsit de fisier.")
            loop.quit()
        elif message_type == Gst.MessageType.WARNING:
            warning, debug = message.parse_warning()
            print(f"Avertisment GStreamer: {warning}: {debug}")
        elif message_type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            print(f"Eroare GStreamer: {error}: {debug}")
            loop.quit()
        return True

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    def sigint_handler(sig, frame):
        # EOS, nu oprire brutala: altfel qtmux nu apuca sa inchida fisierul mp4
        # si videoclipul ramane necitibil.
        print("\nOprire ceruta; trimit EOS ca sa se inchida corect fisierul...")
        pipeline.send_event(Gst.Event.new_eos())

    signal.signal(signal.SIGINT, sigint_handler)

    print("Pornire procesare. Ctrl+C pentru oprire controlata.")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        RUN.logger.close()
        face_database.save()
        summary = write_summary(RUN, args, output_dir)

        print("\n--- gata ---")
        print(f"  cadre procesate:   {summary['cadre']}")
        print(f"  fete procesate:    {summary['fete_procesate']}")
        print(f"  recunoasteri:      {summary['recunoasteri']}")
        print(f"  identitati:        {summary['baza_de_date']['identitati_initiale']} "
              f"-> {summary['baza_de_date']['identitati_finale']}")
        print(f"  timp probe/cadru:  {summary['timp_probe_ms']['mediu']} ms "
              f"(p95 {summary['timp_probe_ms']['p95']} ms)")
        print(f"\n  {output_video}")
        print(f"  {os.path.join(output_dir, 'face_database.json')}")
        print(f"  {os.path.join(output_dir, 'frames.jsonl')}")
        print(f"  {os.path.join(output_dir, 'summary.json')}")


if __name__ == '__main__':
    main()
