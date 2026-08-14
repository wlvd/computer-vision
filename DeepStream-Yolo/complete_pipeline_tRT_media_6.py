"""Varianta pentru fisiere video a pipeline-ului de recunoastere faciala.

Fata de complete_pipeline_tRT.py (camera live + fereastra pe ecran), aici:
  - sursa e un fisier video dat ca parametru;
  - iesirea nu e o fereastra, ci un folder numit dupa video, care contine
    videoclipul adnotat, baza de date rezultata si un log pe fiecare cadru;
  - landmark-urile se calculeaza pe FIECARE cadru, pentru toate fetele vizibile,
    ca sa poata fi desenate. Deciziile de recunoastere raman insa pe aceeasi
    cadenta ca la varianta live (fereastra de calitate + cel mai bun cadru),
    deci ce se vede in log e ce s-ar fi intamplat si pe camera.

Rezolutia de lucru se ia din fisier, nu e fixa: merg si filmarile verticale
(1080x1920) sau cele cu eticheta de rotatie pusa de telefon. Cu --width/--height
se poate cere alta, iar --max-side micsoreaza sursele foarte mari.

Folderul de iesire e REFOLOSIT la rulari succesive pe acelasi video: baza de
date din el se incarca la pornire si se scrie inapoi tot acolo. Prima rulare
inroleaza identitatile, a doua le recunoaste, fara sa se creeze alt folder.
Fisierele unei rulari (video adnotat, frames, summary) sunt numerotate, asa ca
rularile vechi raman pentru comparatie.

Utilizare:
    python3 complete_pipeline_tRT_media_6.py sample_vid.mp4
    python3 complete_pipeline_tRT_media_6.py sample_vid.mp4        # a doua oara: recunoaste
    python3 complete_pipeline_tRT_media_6.py sample/sample_vid.mp4 --database baza.json
    python3 complete_pipeline_tRT_media_6.py sample_vid.mp4 --reset-db --overwrite

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
import warnings
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

# numpy a scos np.bool, iar unele bindings vechi il cer inca. Pe numpy nou, chiar
# si verificarea emite FutureWarning, de aceea o facem tacut: la pornire ies deja
# zeci de linii de la GStreamer si TensorRT, iar peste ele o eroare adevarata
# trece neobservata.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    if not hasattr(np, "bool"):
        np.bool = bool

# API-ul cu bindings indexate (binding_is_input, get_binding_shape, ...) e cel
# din TensorRT 8.5, adica ce vine cu JetPack 5.x, si e deprecat de la 8.5 incolo.
# Merge, dar scoate zece randuri de avertismente la fiecare pornire. Le taiem:
# in TensorRT 10 API-ul chiar dispare, si atunci vom primi AttributeError, nu un
# avertisment ascuns (vezi si comentariul din TrtModel).
warnings.filterwarnings("ignore", category=DeprecationWarning, module="__main__")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_CONFIG_PATH = "config_infer_best_v2.txt"
TRACKER_CONFIG_PATH = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"

PFLD_MODEL_PATH = "/workspace/DeepStream-Yolo/pfld.engine"
RECOGNITION_MODEL_PATH = "/workspace/DeepStream-Yolo/w600k_mbf.engine"
FACE_DATABASE_PATH = "/workspace/DeepStream-Yolo/face_database.json"

MIN_CONFIDENCE = 0.5          # sub asta, nici nu incercam sa procesam fata
MIN_FACE_SIZE = 30            # px, latura minima a bbox-ului
MIN_BLUR = 60.0              # varianta Laplacianului la care consideram poza clara

# Varianta live arunca proba daca blur < MIN_BLUR. Aici nu: pe fisier, varianta
# Laplacianului depinde de codec si de scalarea la rezolutia de lucru (un clip
# 720p urcat la 1080p are varianta de ~2 ori mai mica pe aceeasi fata), asa ca un
# prag absolut poate bloca un track la nesfarsit -- fara proba nu se face nicio
# verificare, deci nici streak-ul de inrolare nu creste si rularea se termina cu
# zero identitati si zero explicatii. Claritatea intra oricum in scorul de
# calitate (termenul "sharp"), iar QUALITY_MIN filtreaza; aici pastram doar un
# prag de siguranta sub care poza chiar nu are informatie.
BLUR_REJECT_FACTOR = 0.25     # probe cu blur < MIN_BLUR * factor se arunca

# Masurat pe sample_vid.mp4: track-urile care trec de porti traiesc intre 2 si 25
# de cadre. La o verificare pe 15 cadre, majoritatea apuca una singura, deci nu
# ajung niciodata la ENROLL_MIN_CHECKS. Costul e mic (recunoasterea a insemnat 15
# apeluri in tot clipul), asa ca verificam mai des.
VERIFY_INTERVAL_FRAMES = 10   # la cat timp cel mult re-verificam un track activ
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

# Pragurile de inrolare sunt mai relaxate decat in varianta live, din doua motive.
# Intai, aici inrolarea se face din cea mai buna proba a intregului track (vezi
# TrackState.best_unknown), nu din proba ferestrei curente: prototipul e ales, nu
# nimerit, deci nu mai e nevoie ca fiecare fereastra sa fie perfecta. Apoi,
# marimea si claritatea depind de rezolutia sursei si de scalare, iar pragurile
# calibrate pe camera 1080p taiau tot pe fisierele de test. Toate se pot schimba
# din linia de comanda (--enroll-min-face, --enroll-min-blur, ...).
ENROLL_MIN_CHECKS = 2         # 3 cerea ~45 de cadre de track neintrerupt; nu exista
ENROLL_MIN_FACE = 35

# Cea mai importanta poarta pentru ce ajunge in baza, si singura care nu se poate
# compensa. Masurat pe sample_vid.mp4: prototipurile cu frontalitate 0 au dat
# autosimilaritate 0.089 fata de alte poze ale aceleiasi persoane -- adica baza
# se umple cu intrari care nu vor recunoaste pe nimeni, niciodata. Cele cu 0.29
# si 0.52 au dat 0.80 si 0.85. Pragul taie exact profilurile, fara sa ceara poze
# de buletin: 40% din fetele clipului au frontalitate 0.
ENROLL_MIN_FRONTALITY = 0.15
ENROLL_MIN_BLUR = 60.0
ENROLL_MIN_QUALITY = 0.35

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
DRAW_OVERLAY = True        # se stinge cu --no-video: nu mai are cine sa vada desenul
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

# ============================================================
# COMPATIBILITATE pyds
# ============================================================
# Bindings-urile Python nu expun aceleasi simboluri in toate versiunile de
# DeepStream: MAX_ELEMENTS_IN_DISPLAY_META, de pilda, exista in antetul C
# (nvdsmeta.h, valoarea 16) dar nu e exportata in pyds peste tot. Rezolvam o
# singura data, aici, si nu in probe -- altfel un simbol lipsa arunca acelasi
# AttributeError pe fiecare cadru, iar pipeline-ul merge mai departe si umple
# consola cu acelasi traceback.

MAX_DISPLAY_META_ELEMENTS = getattr(pyds, "MAX_ELEMENTS_IN_DISPLAY_META", 16)
_unmap_surface = getattr(pyds, "unmap_nvds_buf_surface", None)
_acquire_display_meta = getattr(pyds, "nvds_acquire_display_meta_from_pool", None)
_add_display_meta = getattr(pyds, "nvds_add_display_meta_to_frame", None)

_warned = set()


def warn_once(key, message):
    """Un avertisment o singura data, nu pe fiecare cadru."""
    if key not in _warned:
        _warned.add(key)
        print(f"[AVERTISMENT] {message}")


def check_pyds_api():
    """Verifica la pornire ce ofera pyds-ul instalat.

    Ce lipseste si e obligatoriu opreste rularea acum, cu un mesaj clar, in loc
    sa strice fiecare cadru. Ce lipseste si e optional doar dezactiveaza o
    functie (desen, eliberarea suprafetei), cu avertisment.
    """
    required = ["gst_buffer_get_nvds_batch_meta", "get_nvds_buf_surface",
                "NvDsFrameMeta", "NvDsObjectMeta"]
    missing = [name for name in required if not hasattr(pyds, name)]
    if missing:
        raise RuntimeError(
            "Bindings-urile pyds nu au ce trebuie: " + ", ".join(missing) +
            ".\nVerifica versiunea de deepstream_python_apps fata de DeepStream."
        )

    if _unmap_surface is None:
        warn_once("unmap", "pyds nu are unmap_nvds_buf_surface; suprafetele nu se "
                           "elibereaza explicit (verifica memoria la rulari lungi).")
    if _acquire_display_meta is None or _add_display_meta is None:
        warn_once("display_meta", "pyds nu are functiile de display meta; cele 5 "
                                  "puncte ArcFace nu se pot desena prin nvdsosd.")
    if not hasattr(pyds, "MAX_ELEMENTS_IN_DISPLAY_META"):
        print(f"pyds nu exporta MAX_ELEMENTS_IN_DISPLAY_META; "
              f"folosesc {MAX_DISPLAY_META_ELEMENTS} (valoarea din nvdsmeta.h).")


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
        self.best_frontality = 0.0

        # Cea mai buna proba iesita "necunoscut" de cand exista track-ul, cu
        # embedding-ul ei cu tot. Fara asta, inrolarea se judeca pe proba
        # ferestrei curente: un track poate avea 10 verificari, una dintre ele
        # dintr-un cadru foarte bun, si sa nu se inroleze niciodata pentru ca
        # exact la a patra verificare persoana era intoarsa. Prototipul care
        # ajunge in baza e ales, nu nimerit.
        self.best_unknown = None    # dict: aligned, quality, blur, size, frame
        self.last_unknown_score = -1.0

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
        self.best_frontality = 0.0

    def offer(self, aligned, quality, front, blur, size):
        if quality > self.best_quality:
            self.best_quality = quality
            self.best_aligned = aligned
            self.best_blur = blur
            self.best_size = size
            self.best_frontality = front

    def offer_unknown(self, aligned, quality, front, blur, size, frame_number):
        """Cea mai buna poza a track-ului DINTRE CELE care pot fi prototip.

        Filtram intai, alegem dupa. Invers -- cea mai buna dupa calitate, si apoi
        vedem daca trece pragurile -- nu merge: calitatea pune pe marime doar
        0.20, asa ca o fata mica si frontala bate una mare si putin intoarsa.
        Proba castigatoare pica apoi la poarta de marime, iar track-ul ramane
        neinrolat desi avusese o proba perfect buna (masurat: track 50 avea la
        cadrul 197 o proba de 121 px, dar prototipul ramasese unul de 61 px).

        Alegerea se face dupa FRONTALITATE, nu dupa calitate. Calitatea amesteca
        unghiurile cu claritatea si marimea, asa ca o fata din profil, dar mare si
        clara, iese "mai buna" decat una ceva mai mica privita din fata -- si chiar
        asa s-a intamplat: track 23 a intrat in baza cu o poza din profil (yaw 0),
        desi avea la dispozitie una cu yaw 0.26. Aici nu cautam poza cea mai
        frumoasa, ci pe cea din care se poate recunoaste persoana mai tarziu.

        Se pastreaza poza aliniata, nu embedding-ul: asa poate fi oferita ORICE
        fata vazuta, nu doar cea pe care s-a nimerit sa cada o verificare, iar
        modelul de recunoastere se plateste o singura data, la inrolare.
        """
        if sample_blockers(size, blur, quality, front):
            return
        best = self.best_unknown
        if best is None or (front, quality) > (best["frontality"], best["quality"]):
            self.best_unknown = {"aligned": aligned, "quality": quality,
                                 "frontality": front, "blur": blur, "size": size,
                                 "frame": frame_number}


track_states = {}

# Istoricul complet al track-urilor, pentru summary.json: track_states se curata
# periodic, aici nu stergem nimic.
track_reports = {}


class FaceSample:
    """O fata vizibila in cadrul curent, cu tot ce s-a calculat pentru ea."""

    __slots__ = ("track_id", "state", "obj_meta", "crop", "blur", "size",
                 "box", "landmarks", "five_points", "aligned", "quality",
                 "frontality", "wants_sample", "is_candidate", "action")

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
        self.frontality = 0.0       # min(yaw, pitch): cat de din fata e privita
        self.wants_sample = False   # e in fereastra de probe? (se afla fara GPU)
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


def quality_parts(five_points, blur, min_side):
    """Componentele calitatii, fiecare in [0, 1]. None daca punctele sunt degenerate."""
    left_eye, right_eye, nose, left_mouth, right_mouth = np.asarray(
        five_points, dtype=np.float64
    )

    eye_vec = right_eye - left_eye
    interocular = float(np.linalg.norm(eye_vec))
    if interocular < 1e-3:
        return None

    unit = eye_vec / interocular
    normal = np.array([-unit[1], unit[0]], dtype=np.float64)
    eye_center = (left_eye + right_eye) / 2.0
    mouth_center = (left_mouth + right_mouth) / 2.0

    height = float(np.dot(mouth_center - eye_center, normal))
    if height <= 1e-3:
        return None

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

    return {"yaw": yaw, "pitch": pitch, "roll": roll, "sharp": sharp, "size": size}


def face_quality(parts):
    """Cat de utilizabila e poza, in [0, 1]: frontalitate + claritate + marime."""
    return sum(QUALITY_WEIGHTS[name] * parts[name] for name in QUALITY_WEIGHTS)


def frontality(parts):
    """Cat de din fata e privita persoana, in [0, 1]: min(yaw, pitch).

    Roll-ul nu intra: alinierea Umeyama roteste poza in plan, deci o fata inclinata
    ajunge oricum dreapta in crop-ul de 112x112. Yaw si pitch sunt rotatii in afara
    planului si nu se pot corecta -- o fata din profil, aliniata la un sablon
    frontal, iese intinsa, iar embedding-ul ei nu seamana nici macar cu alte poze
    ale aceleiasi persoane.

    Se ia minimul, nu media: un cap intors la 90 de grade ramane inutilizabil
    oricat de bine ar sta pe verticala.

    Masurat pe sample_vid.mp4: prototipurile cu yaw 0 au dat autosimilaritate 0.089
    (practic zgomot), iar cele cu 0.29 si 0.52 au dat 0.80 si 0.85.
    """
    return min(parts["yaw"], parts["pitch"])


def analyse_faces(faces):
    """Landmark-uri (un singur apel GPU) + aliniere + calitate, pentru tot cadrul.

    Completeaza direct campurile din FaceSample. Se ruleaza pe toate fetele
    vizibile, nu doar pe candidati: landmark-urile se deseneaza, si tot ele decid
    care poza ajunge prototip in baza de date.
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
            parts = quality_parts(five, face.blur, face.size)
            if parts is not None:
                face.quality = face_quality(parts)
                face.frontality = frontality(parts)


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


def majority_label(history):
    """Eticheta cu cele mai multe voturi; la egalitate, cea mai recenta.

    Counter.most_common ar da-o pe prima aparuta, ceea ce dupa o inrolare inseamna
    ca numele proaspat inrolat bate un "necunoscut" venit dupa el, si in video
    ramane afisata o identitate cu scor aproape zero.
    """
    counts = Counter(history)
    top = max(counts.values())
    for label in reversed(history):
        if counts[label] == top:
            return label


def prototype_record(best):
    """Prototipul, pentru log. None cand track-ul inca n-a prins nicio poza buna."""
    if not best:
        return None
    return {"calitate": r(best["quality"]), "frontalitate": r(best["frontality"]),
            "blur": r(best["blur"], 1), "marime": best["size"],
            "cadru": best["frame"]}


def ready_to_enroll(state):
    """Track-ul are si dovada ca nu e in baza, si o poza buna din care sa-l inrolam."""
    return (AUTO_ENROLL and not state.enrolled
            and state.best_unknown is not None
            and state.unknown_streak >= ENROLL_MIN_CHECKS
            and state.last_unknown_score < ENROLL_MAX_SCORE)


def sample_blockers(size, blur, quality, front):
    """Ce opreste o poza sa poata fi prototip. Lista goala = poate fi."""
    blockers = []
    if size < ENROLL_MIN_FACE:
        blockers.append("marime")
    if blur < ENROLL_MIN_BLUR:
        blockers.append("blur")
    if quality < ENROLL_MIN_QUALITY:
        blockers.append("calitate")
    if front < ENROLL_MIN_FRONTALITY:
        blockers.append("profil")
    return blockers


def enroll_blockers(state, score, size, blur, quality, front):
    """Ce anume opreste inrolarea track-ului acum. Lista goala = se inroleaza.

    Fara asta, o rulare care se termina cu zero identitati nu spune nimic: nu se
    stie daca fetele erau prea mici, prea neclare, prea din profil, sau daca pur
    si simplu niciun track n-a trait destul cat sa adune ENROLL_MIN_CHECKS
    verificari. Se numara pe toata rularea si ajunge in summary.json.

    Pragurile de proba (marime, blur, calitate) sunt deja aplicate cand se strange
    prototipul, in offer_unknown; aici se raporteaza doar cand track-ul inca n-are
    niciuna buna, ca sa se vada care poarta o taie.
    """
    blockers = []
    if state.unknown_streak < ENROLL_MIN_CHECKS:
        blockers.append("streak")
    if score >= ENROLL_MAX_SCORE:
        blockers.append("scor")
    if state.best_unknown is None:
        # Track-ul n-a prins inca nicio poza buna. Spunem de ce nu e buna cea de
        # acum -- e cel mai apropiat lucru de "ce prag il taie".
        blockers.extend(sample_blockers(size, blur, quality, front)
                        or ["fara_poza_buna"])
    return blockers

# ============================================================
# DESEN
# ============================================================

def draw_five_points(batch_meta, frame_meta, faces):
    """Cele 5 puncte ArcFace, ca cercuri in display meta (le randeaza nvdsosd).

    Un display meta duce cel mult MAX_DISPLAY_META_ELEMENTS cercuri, deci pentru
    mai multe fete se cer mai multe din pool.

    Desenul nu are voie sa strice procesarea: orice problema de aici (simbol
    lipsa in pyds, camp redenumit intre versiuni) se raporteaza o singura data
    si se merge mai departe -- recunoasterea si logul sunt oricum treaba
    importanta, adnotarea e doar ca sa se vada.
    """
    if _acquire_display_meta is None or _add_display_meta is None:
        return

    pending = []
    for face in faces:
        if face.five_points is None:
            continue
        for index, (x, y) in enumerate(face.five_points):
            pending.append((int(x), int(y), POINT_COLORS[index]))

    limit = MAX_DISPLAY_META_ELEMENTS
    try:
        for start in range(0, len(pending), limit):
            chunk = pending[start:start + limit]
            display_meta = _acquire_display_meta(batch_meta)
            display_meta.num_circles = len(chunk)
            for i, (x, y, color) in enumerate(chunk):
                circle = display_meta.circle_params[i]
                circle.xc = x
                circle.yc = y
                circle.radius = LANDMARK_RADIUS + 1
                circle.circle_color.set(color[0], color[1], color[2], 1.0)
                circle.has_bg_color = 1
                circle.bg_color.set(color[0], color[1], color[2], 1.0)
            _add_display_meta(frame_meta, display_meta)
    except Exception as error:
        warn_once("five_points",
                  f"nu pot desena cele 5 puncte prin display meta ({error}); "
                  f"continui fara ele.")


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
        warn_once("surface", f"nu pot desena pe suprafata ({error}); "
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
    """Un rand JSON pe cadru, in frames_<rulare>.jsonl."""

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

    def __init__(self, args, output_dir, video_path, paths, run_index,
                 database_in, database_out):
        self.args = args
        self.output_dir = output_dir
        self.video_path = video_path
        self.paths = paths
        self.run_index = run_index
        self.database_in = database_in
        self.database_out = database_out
        self.logger = FrameLogger(paths["frames"])
        self.frames = 0
        self.faces_seen = 0
        self.recognitions = 0
        self.enrollments = []
        self.enroll_attempts = 0
        self.enroll_blockers = Counter()
        # De ce nu s-a facut verificarea cand era programata: fara asta, o rulare
        # in care nicio proba nu trece de calitate arata identic cu una in care
        # nu s-a vazut nicio fata (zero incercari de inrolare, zero blocaje).
        self.checks_skipped = Counter()
        self.deadline_qualities = []
        self.probe_ms = []
        # distributiile marimii, claritatii si calitatii fetelor din tot clipul:
        # fara ele, un "marime x35" in blocaje nu spune daca pragul e cu putin
        # prea sus sau cu totul nepotrivit pentru filmarea asta
        self.face_sizes = []
        self.face_blurs = []
        self.face_qualities = []
        self.face_frontalities = []
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
            if _unmap_surface is not None:
                _unmap_surface(hash(gst_buffer), frame_meta.batch_id)

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

    # --- Pasul 2: cine cere o proba in cadrul asta ---
    # Poarta asta -- fereastra dinaintea termenului si ritmul probelor -- nu are
    # nevoie de landmark-uri, deci se poate calcula inainte de ele. Aceleasi
    # reguli ca la varianta live, ca deciziile sa ramana comparabile.
    for face in faces:
        state = face.state
        frames_since_check = frame_number - state.last_checked_frame
        window_open = frames_since_check >= state.deadline_gap - QUALITY_WINDOW_FRAMES
        due_for_sample = frame_number - state.last_candidate_frame >= CANDIDATE_INTERVAL_FRAMES
        face.wants_sample = window_open and due_for_sample

    # --- Pasul 3: landmark-uri + aliniere + calitate, intr-un singur apel GPU ---
    # Pe toate fetele, chiar si fara desen. Am incercat sa le cer doar pentru
    # candidati, dar premisa s-a schimbat: landmark-urile nu mai servesc doar
    # desenului, ci si alegerii prototipului (mai jos), iar un track poate deveni
    # bun exact intre doua verificari -- masurat pe sample_vid.mp4, track 47 avea
    # un singur cadru bun, la 161, intre verificarile de la 152 si 162.
    # Costul e mic: modelul e 112x112, cu batch, si ruleaza doar pe fetele care au
    # trecut deja de porti (149 fete in tot clipul, adica 0.3 pe cadru).
    analysed = faces
    analyse_faces(analysed)

    for face in faces:
        RUN.face_sizes.append(face.size)
        RUN.face_blurs.append(face.blur)
        RUN.face_qualities.append(face.quality)
        RUN.face_frontalities.append(face.frontality)

    # --- Pasul 4: strangem prototipul, indiferent de cadenta verificarilor ---
    for face in faces:
        if face.aligned is not None:
            face.state.offer_unknown(face.aligned, face.quality, face.frontality,
                                     face.blur, face.size, frame_number)

    # --- Pasul 5: care fete intra in concursul pentru "cel mai bun cadru" ---
    for face in faces:
        state = face.state
        if not face.wants_sample:
            continue
        if face.blur < MIN_BLUR * BLUR_REJECT_FACTOR:
            face.action = "sarit_blur"
            state.last_candidate_frame = frame_number
            continue
        if face.aligned is None:
            face.action = "aliniere_esuata"
            state.last_candidate_frame = frame_number
            continue

        state.last_candidate_frame = frame_number
        face.is_candidate = True
        # Probele sub MIN_BLUR intra in concurs, dar se vad in log: daca in toata
        # rularea nu apare decat "proba_neclara", pragul de claritate e nepotrivit
        # pentru filmarea asta, nu fetele sunt de vina.
        face.action = "proba" if face.blur >= MIN_BLUR else "proba_neclara"
        state.offer(face.aligned, face.quality, face.frontality, face.blur, face.size)

    # --- Pasul 6: pentru cine rulam recunoasterea in acest cadru ---
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
            if state.best_aligned is None:
                RUN.checks_skipped["fara_proba"] += 1
            else:
                RUN.checks_skipped["calitate_sub_QUALITY_MIN"] += 1
                RUN.deadline_qualities.append(state.best_quality)
            state.clear_best()

    # --- Pasul 6: un singur apel GPU pentru toate fetele alese ---
    embeddings = embed_aligned([state.best_aligned for _, state in to_recognize])

    # --- Pasul 7: verificare (cine e), fara inrolare ---
    decisions = {}
    for (track_id, state), embedding in zip(to_recognize, embeddings):
        label, score = verify_embedding(embedding)
        quality, blur, size = state.best_quality, state.best_blur, state.best_size
        front = state.best_frontality
        state.clear_best()
        state.checks += 1
        RUN.recognitions += 1

        blockers = None
        if AUTO_ENROLL and label == LABEL_UNKNOWN and not state.enrolled:
            state.unknown_streak += 1
            state.last_unknown_score = score
            blockers = enroll_blockers(state, score, size, blur, quality, front)
            if blockers:
                for blocker in blockers:
                    RUN.enroll_blockers[blocker] += 1
                RUN.enroll_attempts += 1
        else:
            if AUTO_ENROLL and label == LABEL_UNCERTAIN and not state.enrolled:
                # A semanat destul cu cineva din baza cat sa cada in banda de
                # incertitudine: nu inrolam (am dubla o identitate existenta), dar
                # se numara. Daca grosul blocajelor e aici, in cale nu sta o
                # poarta de calitate, ci VERIFY_THRESHOLD/ENROLL_MARGIN fata de
                # baza cu care s-a pornit.
                RUN.enroll_blockers["incert"] += 1
                RUN.enroll_attempts += 1
            state.unknown_streak = 0

        state.history.append(label)
        state.current_label = majority_label(state.history)
        state.current_score = score

        report = track_reports[track_id]
        report["verificari"] = state.checks
        report["eticheta"] = state.current_label
        report["scor"] = r(score)

        decisions[track_id] = {
            "eticheta_bruta": label, "scor": r(score), "calitate": r(quality),
            "frontalitate": r(front), "blur": r(blur, 1), "marime": size,
            "streak": state.unknown_streak, "inrolat": False,
        }
        best = state.best_unknown
        if blockers:
            decisions[track_id]["blocaje"] = blockers
            # prototipul strans pana acum, ca sa se vada daca track-ul are deja o
            # poza buna si asteapta doar streak-ul, sau inca n-a prins niciuna
            decisions[track_id]["cea_mai_buna_proba"] = prototype_record(best)

        if state.current_label not in (LABEL_UNKNOWN, LABEL_UNCERTAIN):
            print(f"[ALERTA] track {track_id} -> {state.current_label} "
                  f"(scor={score:.3f}, calitate={quality:.2f}, cadru={frame_number})")

    # --- Pasul 8: inrolarea, decuplata de cadenta verificarilor ---
    # Un track se inroleaza cand are si dovada ca nu e in baza (streak-ul de
    # verificari "necunoscut"), si o poza buna -- in ORICE ordine ar veni cele
    # doua. Legate de momentul verificarii, se pierdeau track-urile care devin
    # bune imediat dupa o verificare si dispar inainte de urmatoarea (masurat:
    # track 65 avea cadre bune la 223-226, cu verificari la 212 si 222).
    ready = [(track_id, state) for track_id, state in active.items()
             if ready_to_enroll(state)]
    for (track_id, state), embedding in zip(ready, embed_aligned(
            [state.best_unknown["aligned"] for _, state in ready])):
        best = state.best_unknown
        name = enroll(embedding)
        state.enrolled = True
        state.current_label = name
        state.current_score = 1.0
        state.history.clear()
        state.history.append(name)

        track_reports[track_id]["eticheta"] = name
        track_reports[track_id]["inrolat"] = True
        RUN.enrollments.append(dict({"nume": name, "cadru": frame_number,
                                     "track": track_id}, **prototype_record(best)))
        decisions.setdefault(track_id, {}).update(
            {"eticheta_bruta": name, "inrolat": True, "scor": 1.0,
             "streak": state.unknown_streak,
             "cea_mai_buna_proba": prototype_record(best)}
        )
        print(f"[INROLARE] track {track_id} -> {name} din cadrul {best['frame']} "
              f"({best['size']}px, frontalitate {best['frontality']:.2f}, "
              f"calitate {best['quality']:.2f})")

    # --- Pasul 9: desen ---
    for face in faces:
        if face.track_id in decisions:
            face.action = "recunoscut"

    # Fara video de iesire nu are cine sa vada desenul, deci nu il mai facem:
    # scutim si conversiile, si scrisul in suprafata.
    if DRAW_OVERLAY:
        for face in faces:
            label_object(face.obj_meta, face.state)
        draw_five_points(batch_meta, frame_meta, faces)
        if DRAW_ALL_LANDMARKS:
            draw_all_landmarks(surface, faces)

    # --- Pasul 10: inregistrarea pentru log ---
    face_records = []
    for face in faces:
        state = face.state
        entry = {
            "track": face.track_id,
            "box": [int(v) for v in face.box],
            "det": r(float(face.obj_meta.confidence)),
            "blur": r(face.blur, 1),
            "actiune": face.action,
            "eticheta": state.current_label,
            "scor": r(state.current_score),
        }
        if face.landmarks is not None:
            entry["calitate"] = r(face.quality)
            entry["frontalitate"] = r(face.frontality)
            entry["puncte5"] = [[r(x, 1), r(y, 1)] for x, y in face.five_points]
        if face.track_id in decisions:
            entry["decizie"] = decisions[face.track_id]
        face_records.append(entry)

    return {
        "cadru": frame_number,
        "timp": r(pts_seconds),
        "faces": face_records,
        "respinse": rejected,
        "gpu": {"landmark": len(analysed), "recunoastere": len(to_recognize)},
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


def on_child_added(child_proxy, element, name, args):
    """Coboara prin decodebin pana la decodorul hardware si ii taie pool-ul.

    Elementele apar pe rand, pe masura ce decodebin afla ce contine fisierul,
    deci trebuie urmarite si bin-urile create intre timp.
    """
    if "decodebin" in name:
        element.connect("child-added", on_child_added, args)
        return
    if "nvv4l2decoder" in name and element.find_property("num-extra-surfaces"):
        element.set_property("num-extra-surfaces", args.decoder_surfaces)
        print(f"  {name}: num-extra-surfaces={args.decoder_surfaces}")


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


# ------------------------------------------------------------
# Sursa: rezolutie si orientare
# ------------------------------------------------------------
# Eticheta image-orientation spune cu cat trebuie rotita imaginea IN SENSUL
# ACELOR DE CEAS ca sa se vada corect ("rotate-90" = 90 grade CW). Un telefon
# scrie asa: fluxul ramane 1920x1080, iar faptul ca filmarea e verticala e doar
# o eticheta in container. Decodorul nu roteste nimic, deci fara corectia de aici
# fetele ajung culcate la detector, care nu le mai gaseste.
TAG_TO_ROTATION = {"rotate-0": 0, "rotate-90": 90, "rotate-180": 180, "rotate-270": 270}

# nvvideoconvert: 1 = 90 grade trigonometric, 2 = 180, 3 = 90 in sensul acelor
# de ceas.
ROTATION_TO_FLIP_METHOD = {90: 3, 180: 2, 270: 1}


def _rotation_from_tags(tags):
    """Rotatia ceruta de eticheta, in grade. Necunoscut / lipsa = 0."""
    if tags is None:
        return 0
    found, value = tags.get_string("image-orientation")
    if not found:
        return 0
    # "flip-rotate-*" (oglindire + rotatie) se ignora: apar foarte rar si
    # nvvideoconvert n-are o singura metoda care sa le faca pe amandoua.
    return TAG_TO_ROTATION.get(value, 0)


# ------------------------------------------------------------
# Citirea antetului MP4/MOV
# ------------------------------------------------------------
# Sondarea prin GstPbutils sau OpenCV inseamna sa pornesti un decodor ca sa afli
# doua numere. Pe Jetson, la un fisier 4K, asta deschide blocul NVDEC si un pool
# de suprafete de 3840x2160 inainte ca pipeline-ul adevarat sa-si ceara si el
# memoria -- exact felul in care se ajunge la "NvMapMemAllocInternalTagged ...
# error 12" (adica ENOMEM) si, imediat dupa, la CUDNN_STATUS_INTERNAL_ERROR in
# convolutiile detectorului. Antetul MP4/MOV ne da si rezolutia, si rotatia,
# citind cateva zeci de octeti, fara decodor si fara memorie video.

MP4_CONTAINER_BOXES = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}
VISUAL_SAMPLE_FORMATS = {b"avc1", b"avc3", b"hev1", b"hvc1", b"hvc2",
                         b"mp4v", b"av01", b"vp08", b"vp09", b"jpeg"}


def _iter_boxes(handle, end):
    """(tip, inceput continut, sfarsit) pentru fiecare box de la pozitia curenta."""
    while handle.tell() + 8 <= end:
        start = handle.tell()
        header = handle.read(8)
        if len(header) < 8:
            return
        size = int.from_bytes(header[:4], "big")
        box_type = header[4:8]

        if size == 1:                          # dimensiune pe 64 de biti
            extended = handle.read(8)
            if len(extended) < 8:
                return
            size = int.from_bytes(extended, "big")
        elif size == 0:                        # boxul tine pana la capatul fisierului
            size = end - start
        if size < 8 or start + size > end:
            return

        yield box_type, handle.tell(), start + size
        handle.seek(start + size)              # consumatorul a mutat cursorul


def _read_tkhd(data, track):
    """Dimensiunile de afisare si rotatia din matricea de transformare."""
    if len(data) < 4:
        return
    version = data[0]
    base = 4 + (32 if version == 1 else 20)    # creare, modificare, id, durata...
    matrix = base + 16                         # ...rezervat, layer, grup, volum
    if len(data) < matrix + 44:
        return

    def fixed(offset, shift=65536.0):
        return int.from_bytes(data[offset:offset + 4], "big", signed=True) / shift

    # Matricea QuickTime [a b u; c d v; x y w]: unghiul iese din primii doi termeni.
    angle = float(np.degrees(np.arctan2(fixed(matrix + 4), fixed(matrix)))) % 360.0
    track["rotation"] = int(round(angle / 90.0) * 90) % 360
    track["display"] = (int(round(fixed(matrix + 36))), int(round(fixed(matrix + 40))))


def _read_stsd(data, track):
    """Dimensiunile codate, din prima intrare vizuala (avc1, hvc1, ...)."""
    entry = data[8:]                           # versiune, flag-uri, numar de intrari
    if len(entry) >= 36 and entry[4:8] in VISUAL_SAMPLE_FORMATS:
        track["coded"] = (int.from_bytes(entry[32:34], "big"),
                          int.from_bytes(entry[34:36], "big"))


def _fill_mp4_track(handle, end, track):
    for box_type, content, box_end in _iter_boxes(handle, end):
        if box_type in MP4_CONTAINER_BOXES:
            handle.seek(content)
            _fill_mp4_track(handle, box_end, track)
        elif box_type in (b"tkhd", b"stsd"):
            handle.seek(content)
            data = handle.read(box_end - content)
            (_read_tkhd if box_type == b"tkhd" else _read_stsd)(data, track)


def probe_mp4_header(video_path):
    """(latime, inaltime, rotatie) din antet, sau None daca nu e un MP4/MOV citibil."""
    try:
        size = os.path.getsize(video_path)
        with open(video_path, "rb") as handle:
            for box_type, content, box_end in _iter_boxes(handle, size):
                if box_type != b"moov":
                    continue
                handle.seek(content)
                for sub_type, sub_content, sub_end in _iter_boxes(handle, box_end):
                    if sub_type != b"trak":
                        continue
                    track = {}
                    handle.seek(sub_content)
                    _fill_mp4_track(handle, sub_end, track)
                    # Pistele audio n-au intrare vizuala in stsd, deci se sar
                    # singure. Dimensiunile codate bat cele de afisare: decodorul
                    # le scoate pe primele.
                    width, height = track.get("coded") or track.get("display") or (0, 0)
                    if width > 0 and height > 0:
                        return {"width": int(width), "height": int(height),
                                "rotation": track.get("rotation", 0),
                                "citit_cu": "antet MP4"}
    except Exception as error:
        warn_once("mp4_header", f"nu pot citi antetul MP4 ({error}); incerc altfel.")
    return None


def probe_video_info(video_path, timeout_s=10):
    """Rezolutia si orientarea fisierului, citite inainte de a construi pipeline-ul.

    Intai din antet (fara decodor, vezi mai sus). Pentru alte containere se cade
    pe GstPbutils, care cere Gst.init() inainte, si apoi pe OpenCV.
    """
    header = probe_mp4_header(video_path)
    if header is not None:
        return header

    try:
        gi.require_version("GstPbutils", "1.0")
        from gi.repository import GstPbutils

        discoverer = GstPbutils.Discoverer.new(timeout_s * Gst.SECOND)
        info = discoverer.discover_uri(Gst.filename_to_uri(video_path))
        streams = info.get_video_streams()
        if streams:
            stream = streams[0]
            rotation = _rotation_from_tags(stream.get_tags())
            if rotation == 0:
                try:
                    rotation = _rotation_from_tags(info.get_tags())
                except Exception:      # get_tags e depreciat pe versiuni noi
                    pass
            return {"width": int(stream.get_width()), "height": int(stream.get_height()),
                    "rotation": rotation, "citit_cu": "GstPbutils"}
        warn_once("discoverer_video",
                  "GstPbutils nu vede niciun flux video in fisier; incerc cu OpenCV.")
    except Exception as error:
        warn_once("discoverer",
                  f"nu pot interoga fisierul cu GstPbutils ({error}); incerc cu OpenCV.")

    capture = cv2.VideoCapture(video_path)
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orientation_prop = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
        rotation = (int(capture.get(orientation_prop)) % 360
                    if orientation_prop is not None else 0)
    finally:
        capture.release()

    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"Nu pot afla rezolutia lui {video_path}. Da-o pe loc, cu "
            f"--width si --height."
        )
    return {"width": width, "height": height,
            "rotation": rotation if rotation in TAG_TO_ROTATION.values() else 0,
            "citit_cu": "OpenCV"}


def rotation_supported():
    """nvvideoconvert stie sa roteasca? (flip-method nu exista pe toate platformele)"""
    element = Gst.ElementFactory.make("nvvideoconvert", None)
    return element is not None and element.find_property("flip-method") is not None


def even(value):
    """Latimi/inaltimi impare strica NV12 si encoder-ul; rotunjim la par."""
    return max(2, int(round(value / 2.0)) * 2)


def auto_bitrate(width, height):
    """Bitrate potrivit rezolutiei, in biti/s.

    4 Mbit/s fix insemna ~0.06 biti pe pixel pe cadru la 1080p si de patru ori
    mai putin la 4K -- de aici imaginea moale, cu blocuri in scenele cu miscare.
    Aproximativ 4 biti pe pixel pe secunda inseamna 8 Mbit/s la 1080p si 33 la 4K.
    """
    return int(width * height * 4)


def working_resolution(source, args):
    """Rezolutia la care ruleaza pipeline-ul, pornind de la cea a sursei.

    Fix 1920x1080 mergea doar pe filmari orizontale: capsfilter-ul de dupa decodor
    nu pastreaza raportul, deci un clip 1080x1920 era turtit de nvvideoconvert la
    16:9. Pe fete turtite pe orizontala, detectorul si landmark-urile nu mai au ce
    cauta -- de aici "unele video-uri nu merg".

    Implicit pastram exact rezolutia sursei (dupa rotire) si o micsoram doar daca
    depaseste --max-side; motivul micsorarii ramane cel din build_pipeline: la 4K,
    bufferele dintre decodor si streammux mananca degeaba memoria NVMM.

    Inversarea laturilor se face dupa rotatia CHIAR APLICATA (args.rotation), nu
    dupa eticheta din fisier: cu --rotate 0, sau daca nvvideoconvert nu stie sa
    roteasca, cadrele vin nerotite si niste laturi inversate le-ar turti.
    """
    width, height = source["width"], source["height"]
    if args.rotation in (90, 270):
        width, height = height, width

    if args.width and args.height:
        given = args.width / float(args.height)
        actual = width / float(height)
        if abs(given - actual) / actual > 0.01:
            print(f"[AVERTISMENT] --width/--height ({args.width}x{args.height}) au alt "
                  f"raport decat sursa ({width}x{height}); imaginea va fi deformata, "
                  f"iar detectia are de suferit.")
        return even(args.width), even(args.height)
    if args.width:
        return even(args.width), even(args.width * height / float(width))
    if args.height:
        return even(args.height * width / float(height)), even(args.height)

    longest = max(width, height)
    if args.max_side and longest > args.max_side:
        factor = args.max_side / float(longest)
        return even(width * factor), even(height * factor)
    return even(width), even(height)


def build_pipeline(video_path, output_video, args):
    pipeline = Gst.Pipeline()
    if not pipeline:
        raise RuntimeError("Nu s-a putut crea pipeline-ul.")

    print("Creare elemente pipeline...")

    source = make_element("uridecodebin", "sursa")
    source.set_property("uri", Gst.filename_to_uri(video_path))

    # Fara astea doua, uridecodebin incearca sa decodeze si pista audio: pe
    # fisierele cu AAC iese "No decoder available for type audio/mpeg", plus un
    # decodor pornit degeaba. Noi legam oricum doar pad-ul video (on_pad_added).
    source.set_property("caps", Gst.Caps.from_string("video/x-raw(ANY)"))
    source.set_property("expose-all-streams", False)

    # Suprafetele decodorului sunt la rezolutia SURSEI, nu la cea de lucru: la 4K,
    # fiecare inseamna ~12 MB de NVMM. Scalarea de mai jos nu le poate micsora,
    # asa ca macar nu cerem mai multe decat minimul cerut de driver.
    source.connect("child-added", on_child_added, args)

    # Convertorul de dupa sursa accepta si memorie de sistem (decodare software),
    # si NVMM (decodare hardware), deci merge indiferent ce alege decodebin.
    #
    # Scalarea la rezolutia de lucru se cere AICI, nu se lasa pe seama lui
    # nvstreammux: altfel tot ce e intre decodor si streammux isi aloca bufferele
    # la rezolutia sursei. La un clip 4K inseamna surse de 12 MB per buffer, ori
    # marimea pool-ului, degeaba -- si pe Jetson memoria NVMM se termina exact
    # asa ("NvMapMemAllocInternalTagged ... error 12").
    vidconv_in = make_element("nvvideoconvert", "conversie-intrare")

    # Rotatia se face tot aici, inaintea capsfilter-ului: dupa ea, latimea si
    # inaltimea sunt deja cele din caps (vezi working_resolution, care le-a
    # inversat pentru 90/270).
    if args.rotation:
        vidconv_in.set_property("flip-method", ROTATION_TO_FLIP_METHOD[args.rotation])
        print(f"  conversie-intrare: rotesc {args.rotation} grade "
              f"(flip-method={ROTATION_TO_FLIP_METHOD[args.rotation]})")

    caps_in = make_element("capsfilter", "caps-intrare")
    caps_in.set_property("caps", Gst.Caps.from_string(
        f"video/x-raw(memory:NVMM), format=NV12, "
        f"width={args.width}, height={args.height}"))

    streammux = make_element("nvstreammux", "stream-muxer")
    streammux.set_property('width', args.width)
    streammux.set_property('height', args.height)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 40000)
    streammux.set_property('live-source', 0)

    pgie = make_element("nvinfer", "detector-fete")
    pgie.set_property('config-file-path', args.pgie_config)

    tracker = make_element("nvtracker", "tracker")
    # nvtracker scaleaza cadrul la dimensiunile astea; pe o filmare verticala,
    # 640x384 (16:9) inseamna ca tracker-ul lucreaza pe o imagine turtita, cu
    # fete deformate altfel decat cele pe care le-a dat detectorul. Le intoarcem
    # dupa cadru. Raman multipli de 32, cum cere DeepStream.
    tracker_width, tracker_height = (640, 384) if args.width >= args.height else (384, 640)
    tracker.set_property('tracker-width', tracker_width)
    tracker.set_property('tracker-height', tracker_height)
    tracker.set_property('gpu-id', 0)
    tracker.set_property('ll-lib-file',
                         "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property('ll-config-file', args.tracker_config)

    vidconv_osd = make_element("nvvideoconvert", "conversie-osd")
    caps_rgba = make_element("capsfilter", "caps-rgba")
    caps_rgba.set_property('caps', Gst.Caps.from_string(
        "video/x-raw(memory:NVMM), format=RGBA"))

    # Fara video de iesire, tot ce urmeaza dupa desen dispare: encoderul hardware
    # (NVENC) cu pool-urile lui, cele doua conversii si muxer-ul. Pe o placa la
    # limita memoriei, asta e cel mai mare lucru pe care il putem taia fara sa
    # atingem recunoasterea -- baza de date si logul ies la fel.
    if args.no_video:
        sink = make_element("fakesink", "iesire-nula")
        sink.set_property('sync', False)
        sink.set_property('async', False)
        sink.set_property('enable-last-sample', False)

        for element in [source, vidconv_in, caps_in, streammux, pgie, tracker,
                        vidconv_osd, caps_rgba, sink]:
            pipeline.add(element)

        print("Legare elemente pipeline (fara video de iesire)...")
        source.connect("pad-added", on_pad_added, vidconv_in)
        vidconv_in.link(caps_in)
        caps_in.get_static_pad("src").link(streammux.get_request_pad("sink_0"))
        streammux.link(pgie)
        pgie.link(tracker)
        tracker.link(vidconv_osd)
        vidconv_osd.link(caps_rgba)
        caps_rgba.link(sink)

        caps_rgba.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, media_probe, 0)
        return pipeline

    nvosd = make_element("nvdsosd", "osd")

    vidconv_out = make_element("nvvideoconvert", "conversie-iesire")
    caps_out = make_element("capsfilter", "caps-iesire")

    encoder = make_element(["nvv4l2h264enc", "x264enc"], "encoder")
    bitrate = args.bitrate or auto_bitrate(args.width, args.height)
    print(f"  bitrate: {bitrate / 1e6:.1f} Mbit/s"
          + ("" if args.bitrate else " (calculat din rezolutie)"))

    if encoder.get_factory().get_name() == "nvv4l2h264enc":
        encoder.set_property('bitrate', bitrate)
        # Implicit, nvv4l2h264enc encodeaza in Baseline (profilul 66 din log):
        # fara CABAC si fara cadre B, adica exact ce se vede ca "imagine slaba"
        # la acelasi bitrate. High costa la fel de mult de encodat pe NVENC.
        if encoder.find_property("profile"):
            encoder.set_property('profile', 4)          # 0 Baseline, 2 Main, 4 High
        # Bitrate variabil: scenele simple consuma putin, iar cele cu miscare --
        # exact acolo unde se pierd fetele -- primesc cat le trebuie.
        if encoder.find_property("control-rate"):
            encoder.set_property('control-rate', 0)     # 0 variabil, 1 constant
        if encoder.find_property("peak-bitrate"):
            encoder.set_property('peak-bitrate', int(bitrate * 1.5))
        caps_out.set_property('caps', Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), format=NV12"))
    else:
        # cale de rezerva pe desktop: encoder software, deci memorie de sistem
        encoder.set_property('bitrate', max(1, bitrate // 1000))
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


def distribution(values, digits=1):
    """min / p10 / median / p90 / max, ca sa se vada daca un prag e realist."""
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "min": r(ordered[0], digits),
        "p10": r(percentile(ordered, 0.10), digits),
        "median": r(percentile(ordered, 0.50), digits),
        "p90": r(percentile(ordered, 0.90), digits),
        "max": r(ordered[-1], digits),
        "n": len(ordered),
    }


def write_summary(run, args, output_dir):
    durations = run.probe_ms
    identities = {}
    for track_id, report in track_reports.items():
        label = report["eticheta"] or "fara_decizie"
        identities.setdefault(label, []).append(track_id)

    source = getattr(args, "source_info", {})
    summary = {
        "video": run.video_path,
        "folder": output_dir,
        "rulare": run.run_index,
        # None = rularea a mers pana la capatul fisierului.
        "eroare": getattr(run, "failure", None),
        # Rezolutia de lucru nu mai e fixa, deci fara ea nu se pot citi nici
        # distributiile de mai jos: pragurile sunt in pixeli la rezolutia asta.
        "sursa": {
            "rezolutie": f"{source.get('width')}x{source.get('height')}",
            "rotatie_eticheta": source.get("rotation"),
            "rotatie_aplicata": getattr(args, "rotation", 0),
            "rezolutie_lucru": f"{args.width}x{args.height}",
            "citit_cu": source.get("citit_cu"),
            "video_adnotat": None if args.no_video else run.paths["video"],
            "memorie_libera_mb": {"pornire": getattr(args, "free_memory_mb", None),
                                  "final": available_memory_mb()[0]},
        },
        "cadre": run.frames,
        "fete_procesate": run.faces_seen,
        "recunoasteri": run.recognitions,
        "durata_rulare_s": r(time.time() - run.started, 1),
        "baza_de_date": {
            "pornire": run.database_in,
            "scrisa_in": run.database_out,
            "continuata": run.database_in == run.database_out,
            "identitati_initiale": face_database.source_count,
            "identitati_finale": len(face_database),
            "inrolari": run.enrollments,
        },
        # De cate ori a picat fiecare conditie de inrolare. Daca rularea se
        # termina cu zero identitati, aici scrie de ce: pragul cu numarul cel
        # mai mare e cel care blocheaza.
        "blocaje_inrolare": {
            "incercari": run.enroll_attempts,
            "cauze": dict(run.enroll_blockers.most_common()),
        },
        # Verificari programate care nu s-au facut deloc. Daca aici sunt numere
        # mari si "blocaje_inrolare" e gol, problema nu e la pragurile de
        # inrolare: fetele nici n-au ajuns la modelul de recunoastere.
        "verificari_sarite": {
            "cauze": dict(run.checks_skipped.most_common()),
            "calitate_la_termen": distribution(run.deadline_qualities, 3),
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
        # Ce s-a masurat efectiv pe fetele din clip. Pragurile sunt in pixeli la
        # rezolutia de lucru, deci fara distributiile astea nu se poate spune
        # daca un prag e cu putin prea sus sau complet nepotrivit filmarii.
        "distributii": {
            "marime_px": distribution(run.face_sizes, 0),
            "blur": distribution(run.face_blurs, 1),
            "calitate": distribution(run.face_qualities, 3),
            # Cea mai utila dintre toate cand baza iese proasta: daca mediana e
            # langa zero, filmarea e din profil si degeaba se relaxeaza restul.
            "frontalitate": distribution(run.face_frontalities, 3),
        },
        "praguri": {
            "rezolutie_lucru": f"{args.width}x{args.height}",
            "MIN_CONFIDENCE": MIN_CONFIDENCE, "MIN_FACE_SIZE": MIN_FACE_SIZE,
            "MIN_BLUR": MIN_BLUR, "BLUR_REJECT_FACTOR": BLUR_REJECT_FACTOR,
            "VERIFY_THRESHOLD": VERIFY_THRESHOLD,
            "ENROLL_MARGIN": ENROLL_MARGIN, "ENROLL_MAX_SCORE": ENROLL_MAX_SCORE,
            "ENROLL_MIN_FACE": ENROLL_MIN_FACE, "ENROLL_MIN_BLUR": ENROLL_MIN_BLUR,
            "ENROLL_MIN_CHECKS": ENROLL_MIN_CHECKS,
            "ENROLL_MIN_FRONTALITY": ENROLL_MIN_FRONTALITY,
            "QUALITY_GOOD_ENOUGH": QUALITY_GOOD_ENOUGH, "QUALITY_MIN": QUALITY_MIN,
            "ENROLL_MIN_QUALITY": ENROLL_MIN_QUALITY,
            "VERIFY_INTERVAL_FRAMES": VERIFY_INTERVAL_FRAMES,
        },
    }

    with open(run.paths["summary"], "w", encoding="utf-8") as f:
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


RUN_INDEX_RE = re.compile(r"^summary_(\d+)\.json$")


def prepare_output_dir(video_path, args):
    """Folderul cu numele videoclipului, REFOLOSIT la rulari succesive.

    Asta e ce face posibila secventa "prima rulare inroleaza, a doua recunoaste":
    folderul e memoria rularilor pe videoclipul asta, iar baza de date din el se
    incarca la pornire (vezi resolve_database). Un folder nou pe rulare ar
    insemna ca fiecare rulare porneste iar de la zero identitati.

    Ce se pierde prin refolosire -- suprascrierea rularii anterioare -- se evita
    numerotand fisierele rularii (vezi run_paths), nu folderul.
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    base = args.output or os.path.join(SCRIPT_DIR, stem)

    if args.new_dir and os.path.exists(base):
        index = 2
        while os.path.exists(f"{base}_{index}"):
            index += 1
        base = f"{base}_{index}"
        print(f"--new-dir: pornesc curat, in {base}.")

    existed = os.path.isdir(base)
    os.makedirs(base, exist_ok=True)
    return base, existed


def run_paths(output_dir, stem, args):
    """Caile fisierelor rularii curente, numerotate dupa rularile deja existente."""
    used = sorted(int(match.group(1)) for match in
                  (RUN_INDEX_RE.match(name) for name in os.listdir(output_dir)) if match)
    if args.overwrite:
        index = used[-1] if used else 1
    else:
        index = (used[-1] if used else 0) + 1

    tag = f"{index:03d}"
    return index, {
        "video": os.path.join(output_dir, f"{stem}_adnotat_{tag}.mp4"),
        "frames": os.path.join(output_dir, f"frames_{tag}.jsonl"),
        "summary": os.path.join(output_dir, f"summary_{tag}.json"),
    }


def resolve_database(output_dir, args):
    """(de unde se incarca, unde se scrie) baza de date.

    Baza din folderul de iesire e si intrare, si iesire: prima rulare o creeaza
    (pornind, daca exista, de la cea data cu --database), rularile urmatoare o
    gasesc si recunosc ce a inrolat prima. --database ramane doar samanta pentru
    prima rulare si nu e niciodata modificata; --reset-db o ia de la capat.
    """
    folder_db = os.path.join(output_dir, "face_database.json")

    if os.path.isfile(folder_db) and not args.reset_db:
        return folder_db, folder_db

    if os.path.isfile(folder_db) and args.reset_db:
        print(f"--reset-db: ignor {folder_db} si o iau de la capat.")

    seed = args.database if args.database and os.path.isfile(args.database) else None
    return seed, folder_db


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Ruleaza pipeline-ul de recunoastere faciala pe un fisier video "
                    "si scrie video adnotat + baza de date + log pe cadru.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video",
                        help="numele fisierului video (langa script sau in sample/)")
    parser.add_argument("--output", default=None,
                        help="folderul de iesire (implicit: numele videoclipului); "
                             "daca exista, e refolosit impreuna cu baza lui de date")
    parser.add_argument("--database", default=FACE_DATABASE_PATH,
                        help="baza de date samanta, folosita doar cand folderul de "
                             "iesire nu are inca una; nu e modificata")
    parser.add_argument("--pgie-config", default=YOLO_CONFIG_PATH,
                        help="configul nvinfer pentru detectorul de fete")
    parser.add_argument("--tracker-config", default=TRACKER_CONFIG_PATH,
                        help="configul low-level pentru nvtracker")
    parser.add_argument("--landmark-engine", default=PFLD_MODEL_PATH)
    parser.add_argument("--recognition-engine", default=RECOGNITION_MODEL_PATH)
    parser.add_argument("--width", type=int, default=None,
                        help="latimea de lucru; implicit se ia din fisier, ca sa "
                             "mearga si filmarile verticale. Data singura, "
                             "inaltimea se calculeaza pastrand raportul")
    parser.add_argument("--height", type=int, default=None,
                        help="inaltimea de lucru; implicit din fisier")
    parser.add_argument("--max-side", type=int, default=1920,
                        help="latura maxima a rezolutiei de lucru; peste ea sursa "
                             "se micsoreaza pastrand raportul (la 4K, bufferele "
                             "dintre decodor si streammux epuizeaza memoria NVMM). "
                             "0 = fara limita")
    parser.add_argument("--decoder-surfaces", type=int, default=0,
                        help="suprafete in plus fata de minimul cerut de decodor; "
                             "la surse 4K fiecare costa ~12 MB de memorie NVMM")
    parser.add_argument("--rotate", default="auto", choices=["auto", "0", "90", "180", "270"],
                        help="rotatia aplicata sursei, in grade, in sensul acelor de "
                             "ceas; 'auto' urmeaza eticheta din fisier")
    parser.add_argument("--bitrate", type=int, default=0,
                        help="bitrate-ul videoclipului de iesire, in biti/s; "
                             "0 = calculat din rezolutie (~8 Mbit/s la 1080p)")
    parser.add_argument("--no-video", action="store_true",
                        help="nu scrie videoclipul adnotat: scoate encoderul si "
                             "desenul din pipeline. Prima solutie cand placa ramane "
                             "fara memorie; baza de date si logul ies neschimbate")
    parser.add_argument("--no-enroll", action="store_true",
                        help="nu adauga identitati noi in baza de date")
    parser.add_argument("--no-landmarks", action="store_true",
                        help="deseneaza doar cele 5 puncte ArcFace, nu tot setul")
    parser.add_argument("--new-dir", action="store_true",
                        help="porneste intr-un folder nou (fara identitatile "
                             "inrolate la rularile anterioare)")
    parser.add_argument("--overwrite", action="store_true",
                        help="scrie peste fisierele ultimei rulari din folder, in "
                             "loc sa le numeroteze mai departe")
    parser.add_argument("--reset-db", action="store_true",
                        help="ignora baza de date din folder si o ia de la samanta "
                             "--database (cu --database '' porneste complet goala)")

    # Pragurile care decid ce ajunge in baza. Sunt in linia de comanda pentru ca
    # depind de filmare (rezolutie, codec, distanta fata de camera), iar valorile
    # masurate efectiv sunt in summary.json, la "distributii": se citesc de acolo
    # si se dau aici, fara sa fie nevoie de modificat scriptul.
    # Implicit None peste tot, ca sa ramana in vigoare valorile din script pentru
    # ce nu e dat explicit. Toate marimile sunt in pixeli la rezolutia de lucru.
    tuning = parser.add_argument_group("praguri (vezi 'distributii' din summary.json)")
    tuning.add_argument("--min-face", type=int, default=None,
                        help=f"latura minima a bbox-ului ca fata sa fie procesata "
                             f"(implicit {MIN_FACE_SIZE})")
    tuning.add_argument("--verify-interval", type=int, default=None,
                        help=f"la cate cadre cel mult se re-verifica un track; mai "
                             f"des inseamna mai multe sanse ca un track scurt sa "
                             f"apuce ENROLL_MIN_CHECKS (implicit {VERIFY_INTERVAL_FRAMES})")
    tuning.add_argument("--min-blur", type=float, default=None,
                        help=f"varianta Laplacianului considerata 'clar' "
                             f"(implicit {MIN_BLUR})")
    tuning.add_argument("--quality-min", type=float, default=None,
                        help=f"calitatea minima ca sa merite rulat modelul de "
                             f"recunoastere (implicit {QUALITY_MIN})")
    tuning.add_argument("--verify-threshold", type=float, default=None,
                        help=f"scorul cosinus de la care o fata e recunoscuta "
                             f"(implicit {VERIFY_THRESHOLD})")
    tuning.add_argument("--enroll-min-checks", type=int, default=None,
                        help=f"cate verificari 'necunoscut' la rand cere o inrolare "
                             f"(implicit {ENROLL_MIN_CHECKS})")
    tuning.add_argument("--enroll-min-face", type=int, default=None,
                        help=f"implicit {ENROLL_MIN_FACE}")
    tuning.add_argument("--enroll-min-blur", type=float, default=None)
    tuning.add_argument("--enroll-min-quality", type=float, default=None)
    tuning.add_argument("--enroll-min-frontality", type=float, default=None,
                        help=f"cat de din fata trebuie privita persoana ca poza ei "
                             f"sa ajunga prototip, min(yaw, pitch) in [0,1] "
                             f"(implicit {ENROLL_MIN_FRONTALITY})")
    return parser.parse_args(argv)


def apply_thresholds(args):
    """Muta pragurile date in linia de comanda peste constantele modulului."""
    global MIN_FACE_SIZE, MIN_BLUR, QUALITY_MIN, VERIFY_THRESHOLD
    global ENROLL_MIN_CHECKS, ENROLL_MIN_FACE, ENROLL_MIN_BLUR
    global ENROLL_MIN_QUALITY, ENROLL_MAX_SCORE, VERIFY_INTERVAL_FRAMES
    global ENROLL_MIN_FRONTALITY

    if args.min_face is not None:
        MIN_FACE_SIZE = args.min_face
    if args.verify_interval is not None:
        VERIFY_INTERVAL_FRAMES = args.verify_interval
    if args.min_blur is not None:
        MIN_BLUR = args.min_blur
    if args.quality_min is not None:
        QUALITY_MIN = args.quality_min
    if args.verify_threshold is not None:
        VERIFY_THRESHOLD = args.verify_threshold
    if args.enroll_min_checks is not None:
        ENROLL_MIN_CHECKS = args.enroll_min_checks
    if args.enroll_min_face is not None:
        ENROLL_MIN_FACE = args.enroll_min_face
    if args.enroll_min_blur is not None:
        ENROLL_MIN_BLUR = args.enroll_min_blur
    if args.enroll_min_quality is not None:
        ENROLL_MIN_QUALITY = args.enroll_min_quality
    if args.enroll_min_frontality is not None:
        ENROLL_MIN_FRONTALITY = args.enroll_min_frontality
    # derivat, deci trebuie recalculat dupa ce se schimba pragul de verificare
    ENROLL_MAX_SCORE = VERIFY_THRESHOLD - ENROLL_MARGIN


# Pragurile in pixeli NU se scaleaza cu rezolutia de lucru, desi la un moment dat
# faceam asta. Motivul: ce conteaza pentru recunoastere e cati pixeli are efectiv
# fata, nu ce fractiune din cadru ocupa. Aceeasi persoana filmata 4K si redusa la
# 1080p chiar are jumatate din detaliu, iar un prag care se scaleaza odata cu
# cadrul ar declara cele doua cazuri echivalente -- si ar anula tocmai castigul
# pentru care ai creste --max-side.


def available_memory_mb():
    """(liber, total) in MB, din /proc/meminfo. None acolo unde nu exista.

    Pe Jetson memoria e unificata: pool-urile NVMM, workspace-ul cuDNN si
    procesele obisnuite trag din acelasi loc, deci numarul asta e chiar cel care
    decide daca rularea trece sau pica cu ENOMEM.
    """
    try:
        values = {}
        with open("/proc/meminfo", "r") as handle:
            for line in handle:
                name, _, rest = line.partition(":")
                if name in ("MemTotal", "MemAvailable"):
                    values[name] = int(rest.split()[0]) // 1024
        return values.get("MemAvailable"), values.get("MemTotal")
    except Exception:
        return None, None


def explain_inference_error(text, args):
    """Traduce esecul lui nvinfer in ce se poate face concret.

    "Failed to queue input batch" nu spune nimic singur, iar cauza adevarata e cu
    zeci de randuri mai sus, printre mesajele TensorRT.
    """
    if "queue input batch" not in text and "NVDSINFER_TENSORRT_ERROR" not in text:
        return

    free, total = available_memory_mb()
    source = args.source_info
    print("\n  Detectorul a picat la inferenta. Aproape sigur memoria: "
          "CUDNN_STATUS_INTERNAL_ERROR")
    print("  inseamna de obicei ca nu mai e loc de workspace, nu ca modelul e gresit.")
    if free is not None:
        print(f"  Liber acum: {free} MB din {total} MB.")
    print(f"  Sursa se decodeaza la rezolutia ei ({source['width']}x{source['height']}), "
          f"oricat de mic ar fi\n  cadrul de lucru, iar peste asta vin detectorul, "
          f"cele doua engine-uri si encoderul.")
    print("  De incercat, in ordine:")
    print("  1. --no-video: scoate NVENC si conversiile de iesire. Baza de date si "
          "logul ies la fel,\n     doar videoclipul adnotat nu se mai scrie.")
    print("  2. --max-side 1280: micsoreaza tot ce e dupa decodor.")
    print("  3. Elibereaza memorie: opreste interfata grafica, alte procese CUDA, "
          "si urmareste\n     cu tegrastats cat ramane liber in timpul rularii.")


def main():
    global AUTO_ENROLL, DRAW_ALL_LANDMARKS, DRAW_OVERLAY, RUN

    args = parse_args()
    AUTO_ENROLL = not args.no_enroll
    DRAW_OVERLAY = not args.no_video
    DRAW_ALL_LANDMARKS = not args.no_landmarks and DRAW_OVERLAY
    apply_thresholds(args)

    check_pyds_api()

    video_path = resolve_video(args.video)
    args.pgie_config = resolve_config(args.pgie_config, "configul nvinfer (detectorul de fete)")
    args.tracker_config = resolve_config(args.tracker_config, "configul nvtracker")

    # Gst.init inainte de sondare: GstPbutils are nevoie de el.
    Gst.init(None)
    source = probe_video_info(video_path)
    args.rotation = source["rotation"] if args.rotate == "auto" else int(args.rotate)
    if args.rotation and not rotation_supported():
        print(f"[AVERTISMENT] nvvideoconvert nu are flip-method pe platforma asta; "
              f"nu pot roti sursa cu {args.rotation} grade. Fetele raman culcate, "
              f"deci detectorul le va rata.")
        args.rotation = 0
    args.width, args.height = working_resolution(source, args)
    args.source_info = source

    free, total = available_memory_mb()
    args.free_memory_mb = free
    orientation = " (verticala)" if args.height > args.width else ""
    print(f"Sursa:  {source['width']}x{source['height']}, rotatie "
          f"{source['rotation']} grade [{source['citit_cu']}]"
          + (f"; memorie libera {free} MB din {total} MB" if free is not None else ""))
    print(f"Lucru:  {args.width}x{args.height}{orientation}"
          + (f", rotesc cu {args.rotation} grade" if args.rotation else ""))
    # Micsorarea sursei injumatateste si fetele: pragurile sunt in pixeli la
    # rezolutia de lucru, deci merita spus pe loc cat s-a pierdut.
    if args.width < source["width"]:
        factor = source["width"] / float(args.width)
        print(f"        sursa e micsorata de {factor:.1f} ori, deci si fetele: "
              f"cu --max-side {source['width']} raman la marimea lor")

    stem = os.path.splitext(os.path.basename(video_path))[0]
    output_dir, reused = prepare_output_dir(video_path, args)
    run_index, paths = run_paths(output_dir, stem, args)
    database_in, database_out = resolve_database(output_dir, args)
    output_video = paths["video"]

    print(f"Video:  {video_path}")
    print(f"Iesire: {output_dir} ({'refolosit' if reused else 'nou'}, rularea {run_index})")

    load_models(args.landmark_engine, args.recognition_engine,
                database_in, database_out)
    if database_in == database_out:
        print("Continui baza de date a folderului: ce s-a inrolat la rularile "
              "anterioare se recunoaste acum.")

    RUN = Run(args, output_dir, video_path, paths, run_index,
              database_in, database_out)

    pipeline = build_pipeline(video_path, output_video, args)
    loop = GLib.MainLoop()

    # Retinem esecul ca sa iasa si in codul de retur: pana acum o rulare cazuta
    # in prima secunda si una terminata cu bine ieseau amandoua cu 0, deci un
    # script care le porneste in serie n-avea de unde sa stie.
    failure = []

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
            failure.append(str(error))
            explain_inference_error(f"{error} {debug}", args)
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
        RUN.failure = failure[0] if failure else None
        summary = write_summary(RUN, args, output_dir)

        print("\n--- gata ---")
        print(f"  cadre procesate:   {summary['cadre']}")
        print(f"  fete procesate:    {summary['fete_procesate']}")
        print(f"  recunoasteri:      {summary['recunoasteri']}")
        print(f"  identitati:        {summary['baza_de_date']['identitati_initiale']} "
              f"-> {summary['baza_de_date']['identitati_finale']}")

        # Daca n-a intrat nimeni in baza, spunem pe loc care prag a stat in cale.
        blocaje = summary["blocaje_inrolare"]
        sarite = summary["verificari_sarite"]["cauze"]
        if blocaje["cauze"]:
            cauze = ", ".join(f"{name} x{count}" for name, count in blocaje["cauze"].items())
            print(f"  inrolari blocate:  {blocaje['incercari']} incercari -> {cauze}")
            for name, key, prag in (("marime", "marime_px", ENROLL_MIN_FACE),
                                    ("blur", "blur", ENROLL_MIN_BLUR),
                                    ("calitate", "calitate", ENROLL_MIN_QUALITY),
                                    ("profil", "frontalitate", ENROLL_MIN_FRONTALITY)):
                if name not in blocaje["cauze"]:
                    continue
                masurat = summary["distributii"][key]
                if masurat:
                    print(f"    {name}: masurat {masurat['min']}-{masurat['max']} "
                          f"(median {masurat['median']}), prag {prag}")
        elif summary["fete_procesate"] == 0:
            print("  (nicio fata nu a trecut de porti: vezi 'respinse' in log)")
        if sarite:
            cauze = ", ".join(f"{name} x{count}" for name, count in sarite.items())
            print(f"  verificari sarite: {cauze}")
        print(f"  timp probe/cadru:  {summary['timp_probe_ms']['mediu']} ms "
              f"(p95 {summary['timp_probe_ms']['p95']} ms)")
        free_now, _ = available_memory_mb()
        if free_now is not None:
            print(f"  memorie libera:    {free_now} MB")
        print("")
        if not args.no_video:
            print(f"  {output_video}")
        print(f"  {database_out}")
        print(f"  {paths['frames']}")
        print(f"  {paths['summary']}")

    return 1 if failure else 0


if __name__ == '__main__':
    sys.exit(main())
