import sys
import os
import re
import json
import signal
import time
from collections import defaultdict, deque, Counter

import numpy as np
import cv2
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib
import pyds
import tensorrt as trt
import pycuda.driver as cuda

# ============================================================
# CONFIGURARE
# ============================================================

if not hasattr(np, "bool"):
    np.bool = bool

YOLO_CONFIG_PATH = "config_infer_best_v2.txt"
TRACKER_CONFIG_PATH = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"

PFLD_MODEL_PATH = "/workspace/DeepStream-Yolo/pfld.engine"
RECOGNITION_MODEL_PATH = "/workspace/DeepStream-Yolo/w600k_mbf.engine"
FACE_DATABASE_PATH = "/workspace/DeepStream-Yolo/face_database.json"

MIN_CONFIDENCE = 0.5          # sub asta, nici nu incercam sa procesam fata
MIN_FACE_SIZE = 60            # px, latura minima a bbox-ului

# Varianta Laplacianului creste cu dimensiunea si contrastul crop-ului. 800 e o
# valoare de poza statica, sharp; pe camera live, cu miscare si compresie MJPEG,
# aproape niciun cadru nu trece de ea, deci poarta pica mereu si eticheta ramane
# "verificare..." la infinit. 100 lasa sa treaca fetele utilizabile.
MIN_BLUR = 100.0

VERIFY_INTERVAL_FRAMES = 15   # la cat timp cel mult re-verificam un track activ
RETRY_INTERVAL_ON_FAIL = 5    # daca poarta de calitate a picat, reincercam mai repede
LABEL_HISTORY_SIZE = 5        # cate decizii recente pastram pentru vot majoritar
TRACK_TIMEOUT_FRAMES = 300    # dupa cate cadre de absenta stergem un track din memorie
PRUNE_CHECK_INTERVAL = 90     # la cate cadre verificam track-uri "moarte"

# --- alegerea cadrului: cea mai buna poza, nu prima care pica pe interval ---
#
# Inainte, recunoasterea rula pe cadrul care se nimerea la multiplu de
# VERIFY_INTERVAL_FRAMES, indiferent daca omul era intors sau prins in miscare.
# Acum, in ultimele QUALITY_WINDOW_FRAMES cadre dinaintea termenului se ia cate
# o proba la fiecare CANDIDATE_INTERVAL_FRAMES cadre, se ruleaza doar modelul de
# landmark-uri (ieftin si batch-uit) si se pastreaza cea mai buna poza. Modelul
# de recunoastere, care e partea scumpa, ruleaza o singura data, pe castigator.
QUALITY_WINDOW_FRAMES = 6     # cu cate cadre inainte de termen incepem sa strangem probe
CANDIDATE_INTERVAL_FRAMES = 2 # la cate cadre luam o proba in fereastra
QUALITY_GOOD_ENOUGH = 0.65    # poza asa buna incat nu mai are rost sa asteptam altele
QUALITY_MIN = 0.30            # sub asta nu merita consumat modelul de recunoastere

# Ponderile scorului de calitate; insumeaza 1. Frontalitatea (yaw) cantareste
# cel mai mult: un profil strica embedding-ul mai rau decat o poza usor neclara.
QUALITY_WEIGHTS = {"yaw": 0.30, "pitch": 0.20, "roll": 0.10, "sharp": 0.20, "size": 0.20}

VERIFY_THRESHOLD = 0.42       # prag empiric, ajustat pe baza testelor offline anterioare

# --- inrolare live: fete noi intra singure in baza de date ---
AUTO_ENROLL = True
# Zona dintre ENROLL_MAX_SCORE si VERIFY_THRESHOLD NU e o gaura moarta, e banda
# de incertitudine: seamana prea mult cu cineva din baza ca sa declaram o
# identitate noua (am face un duplicat), dar prea putin ca sa dam un nume.
# Inainte, un track blocat in banda ramanea acolo la nesfarsit -- crestea streak-ul
# de "necunoscut" fara sa poata inrola vreodata. Acum banda are o semnificatie
# explicita: rezultatul e "incert", streak-ul de inrolare se reseteaza (exista
# indicii ca persoana e deja in baza) si se reincearca pe o poza mai buna.
# Cu ENROLL_MARGIN = 0.0 banda dispare si orice fata nerecunoscuta devine
# inrolabila -- mai putine "incert", dar mai multe identitati duplicate.
ENROLL_MARGIN = 0.10
ENROLL_MAX_SCORE = VERIFY_THRESHOLD - ENROLL_MARGIN   # 0.32
ENROLL_MIN_CHECKS = 4         # de cate ori la rand trebuie sa iasa "necunoscut"
ENROLL_MIN_FACE = 90          # px, mai strict decat MIN_FACE_SIZE: inrolam doar fete mari
ENROLL_MIN_BLUR = 110.0       # la fel, mai strict decat poarta de recunoastere
ENROLL_MIN_QUALITY = 0.55     # inrolam doar poze frontale: prototipul ramane in baza
DB_SAVE_INTERVAL_FRAMES = 300 # cat de des scriem baza de date pe disc

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

# ============================================================
# MODELE TENSORRT (incarcate o singura data, globale)
# ============================================================

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Contextul CUDA se creeaza explicit, NU prin pycuda.autoinit.
#
# Motivul: autoinit creeaza un context legat de firul care face importul, adica
# firul principal. Probe-ul GStreamer ruleaza insa pe firul de streaming, iar de
# acolo contextul acela nu e valid -- primesti "invalid device context" sau, mai
# rau, rezultate aiurea. Contextul primar e in plus si cel folosit de DeepStream,
# deci il imprumutam pe acelasi in loc sa cream unul concurent.
cuda.init()
CUDA_CONTEXT = cuda.Device(0).retain_primary_context()


class TrtModel:
    """Un .engine TensorRT, cu bufferele alocate o singura data.

    Alocarea se face in constructor, nu la fiecare inferenta: altfel cudaMalloc
    ar domina timpul de rulare si s-ar pierde tot castigul fata de CPU.
    Bufferele se aloca la batch-ul MAXIM suportat de engine, ca sa poata fi
    refolosite indiferent de cate fete are cadrul curent.

    API-ul cu bindings indexate (execute_async_v2) e cel din TensorRT 8.x, adica
    ce vine cu JetPack 5.x. In TensorRT 10 e scos, si trebuie inlocuit cu
    set_tensor_address + execute_async_v3.
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
                    f"Engine-ul {engine_path} nu a putut fi deserializat. De obicei "
                    f"inseamna ca a fost construit cu alta versiune de TensorRT sau "
                    f"pe alta placa: engine-urile nu sunt portabile."
                )

            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()

            # Cat de multe esantioane intra intr-un singur apel. Doua cazuri:
            #   - batch fix (forma (8, 3, 112, 112)): 8, si trebuie mereu umplut
            #     cu 8 randuri, chiar daca doar primele n ne intereseaza;
            #   - batch dinamic (prima dimensiune -1): maximul din profilul de
            #     optimizare, iar forma reala se fixeaza inainte de fiecare apel.
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

            # Formele iesirilor nu se pot calcula cat timp intrarile sunt -1,
            # deci le concretizam intai la batch maxim.
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
                # Coada bufferului de intrare ramane necompletata cand cadrul are
                # mai putine fete decat batch-ul; o zerorizam ca engine-ul sa nu
                # macine memorie neinitializata (NaN-uri) pe randurile de umplutura.
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
        """Fixeaza forma intrarilor pentru engine-urile cu batch dinamic.

        La batch fix nu avem ce face: engine-ul calculeaza mereu toate randurile,
        iar apelantul ignora ce iese peste cele n cerute.
        """
        if not self.dynamic or batch == self._current_batch:
            return
        for entry in self.inputs:
            self.context.set_binding_shape(entry["index"], (batch,) + entry["sample_shape"])
        self._current_batch = batch

    def infer_batch(self, array):
        """Ruleaza engine-ul pe n esantioane deodata (n <= max_batch).

        `array` are dimensiunea de batch in fata: (n, 3, 112, 112). Intoarce
        lista de iesiri, fiecare de forma (n, ...) -- deci indexabile cu acelasi
        i ca intrarea.

        push/pop la fiecare apel pentru ca metoda e chemata din probe, adica de pe
        firul de streaming al GStreamer. Costa microsecunde.
        """
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
                f"engine-ul asteapta {count * source['sample_elems']} "
                f"(forma pe esantion {source['sample_shape']})."
            )

        CUDA_CONTEXT.push()
        try:
            self._set_batch(count)
            flat = data.ravel()
            source["host"][:flat.size] = flat
            # Se copiaza tot bufferul, nu doar primele count esantioane: e o
            # felie contigua de ~1 MB, iar asa evitam sa dam pycuda un view din
            # memoria pagelocked pe calea asincrona.
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


def run_batched(model, tensors):
    """Ruleaza modelul pe o lista de intrari, in transe de cel mult max_batch.

    Aici se castiga: in loc de un apel GPU pe fata, avem unul pe cadru (sau pe
    fiecare grup de max_batch fete). Se intoarce doar prima iesire a modelului,
    ca lista de tensori per esantion, in aceeasi ordine ca la intrare -- ambele
    modele folosite aici au o singura iesire relevanta.
    """
    results = []
    for start in range(0, len(tensors), model.max_batch):
        chunk = np.stack(tensors[start:start + model.max_batch])
        results.extend(model.infer_batch(chunk)[0])
    return results


print("Incarc engine-urile TensorRT...")
landmark_model = TrtModel(PFLD_MODEL_PATH)
recognition_model = TrtModel(RECOGNITION_MODEL_PATH)

# Cate puncte scoate modelul decide maparea catre cele 5 puncte ArcFace.
# PFLD e antrenat pe WFLW si scoate de obicei 98; scripturile anterioare
# presupuneau 68 (iBUG). Verifica ce scrie aici la pornire.
# Se numara pe forma unui singur esantion, nu pe tot bufferul: la batch 8,
# iesirea are 8x mai multe valori, iar impartirea la 2 ar da 8x mai multe puncte.
_landmark_points = landmark_model.outputs[0]["sample_elems"] // 2
print(f"Model landmark-uri: {_landmark_points} puncte")

def l2(vec):
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-12 else vec


class FaceDatabase:
    """Prototipurile, tinute ca o matrice (N, 512), nu ca un dict de vectori.

    De ce nu pgvector / FAISS / alt motor de vectori: pe Jetson avem ordinul
    zecilor-sutelor de identitati, iar cautarea exacta e un singur produs
    matrice-vector -- sub 0.1 ms la N=1000. Un Postgres cu pgvector ar adauga un
    serviciu, un socket si un index ANN aproximativ ca sa rezolve o problema pe
    care nu o avem; ar incepe sa merite abia la ~10^5 vectori sau daca mai multe
    placi ar trebui sa imparta aceeasi baza. Ce s-a schimbat fata de varianta
    anterioara e bucla Python peste dict, inlocuita cu un np.dot pe toata
    matricea: la 100 de identitati e de ordinul a 50x mai rapid, si conteaza,
    fiindca verificarea se face in probe, pe firul de streaming.

    Prototipurile TREBUIE sa aiba lungime 1: comparatia e produs scalar, care e
    cosinus doar intre vectori normalizati. Daca scriptul de inrolare nu a
    normalizat, scorurile ies scalate aiurea si pragul nu mai are sens -- de
    aceea normalizam la incarcare, nu presupunem.
    """

    def __init__(self, path):
        self.path = path
        self.labels = []
        self.matrix = np.zeros((0, 0), dtype=np.float32)
        self.dirty = False

        try:
            with open(path, "r") as f:
                raw = json.load(f)
        except FileNotFoundError:
            print(f"Nu exista {path}, pornesc cu baza de date goala.")
            return

        if raw:
            self.labels = list(raw)
            self.matrix = np.stack([l2(raw[label]) for label in self.labels])

    def __len__(self):
        return len(self.labels)

    def add(self, label, embedding):
        vector = l2(embedding)[np.newaxis, :]
        self.matrix = vector if len(self.labels) == 0 else np.vstack([self.matrix, vector])
        self.labels.append(label)
        self.dirty = True

    def verify(self, embedding, threshold, margin):
        """Compara cu toate prototipurile deodata.

        Intoarce (eticheta, scor). Eticheta e numele identitatii daca scorul
        trece de prag, LABEL_UNCERTAIN daca pica in banda de sub prag dar peste
        (prag - margin), si LABEL_UNKNOWN daca e clar sub. Banda din mijloc e
        raspunsul cinstit "seamana cu cineva, dar nu destul cat sa spun cine":
        acolo nu dam nume, dar nici nu inrolam, ca sa nu facem duplicate.
        """
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

    def save(self, force=False):
        if not (self.dirty or force):
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(
                {label: self.matrix[i].tolist() for i, label in enumerate(self.labels)}, f
            )
        os.replace(tmp, self.path)
        self.dirty = False


face_database = FaceDatabase(FACE_DATABASE_PATH)
print(f"Baza de date incarcata: {face_database.labels}")


def save_database():
    face_database.save()

# ============================================================
# STARE PER TRACK
# ============================================================

class TrackState:
    def __init__(self):
        self.last_checked_frame = -999999
        self.last_check_failed = False
        self.last_seen_frame = 0
        self.history = deque(maxlen=LABEL_HISTORY_SIZE)
        self.current_label = None
        self.current_score = 0.0
        self.unknown_streak = 0    # verificari consecutive iesite "necunoscut"
        self.enrolled = False      # ca sa nu inrolam de doua ori acelasi track

        # cea mai buna poza vazuta in fereastra curenta (vezi best-frame in config)
        self.last_candidate_frame = -999999
        self.best_quality = -1.0
        self.best_aligned = None   # fata deja aliniata 112x112, gata de recunoastere
        self.best_blur = 0.0
        self.best_size = 0

    @property
    def deadline_gap(self):
        """Peste cate cadre de la ultima verificare expira fereastra curenta."""
        return RETRY_INTERVAL_ON_FAIL if self.last_check_failed else VERIFY_INTERVAL_FRAMES

    def clear_best(self):
        self.best_quality = -1.0
        self.best_aligned = None
        self.best_blur = 0.0
        self.best_size = 0

    def offer(self, aligned, quality, blur, size):
        """Retine poza daca e mai buna decat ce aveam in fereastra asta."""
        if quality > self.best_quality:
            self.best_quality = quality
            self.best_aligned = aligned
            self.best_blur = blur
            self.best_size = size

track_states = {}


class PendingFace:
    """O proba din cadrul curent, care asteapta sa fie trimisa pe GPU.

    Tine crop-ul plus ce mai trebuie dupa inferenta (blur si latura minima,
    pentru scorul de calitate si pentru poarta de inrolare), fiindca decizia se
    ia abia dupa ce s-a rulat batch-ul, cand bucla peste obiecte s-a terminat.
    """

    __slots__ = ("track_id", "state", "crop", "blur", "size")

    def __init__(self, track_id, state, crop, blur, size):
        self.track_id = track_id
        self.state = state
        self.crop = crop
        self.blur = blur
        self.size = size

# ============================================================
# FUNCTII DE PROCESARE (identice cu formulele validate offline)
# ============================================================

def blur_score(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def preprocess_landmark(crop_bgr):
    """Crop -> tensorul (3, 112, 112) cerut de PFLD, fara dimensiune de batch.

    Preprocesarea e separata de inferenta ca sa se poata aduna crop-urile mai
    multor fete intr-un singur apel GPU.
    """
    img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (112, 112))
    tensor = img_resized.astype(np.float32) / 255.0
    return np.transpose(tensor, (2, 0, 1))


def postprocess_landmark(raw, crop_shape):
    """Iesirea normalizata a PFLD -> coordonate in pixelii crop-ului."""
    h, w = crop_shape[:2]
    landmark_pixels = np.asarray(raw, dtype=np.float32).reshape(-1, 2).copy()
    landmark_pixels[:, 0] *= w
    landmark_pixels[:, 1] *= h
    return landmark_pixels


def run_landmark(crop_bgr):
    """Varianta pe o singura fata (folosita in afara probe-ului batched)."""
    raw = landmark_model.infer(preprocess_landmark(crop_bgr))[0]
    return postprocess_landmark(raw, crop_bgr.shape)


# Indicii celor 5 puncte ArcFace, pe fiecare markup posibil:
# ochi stang, ochi drept, varf nas, colt gura stanga, colt gura dreapta.
# Cheia e numarul de puncte pe care il scoate modelul.
#   68  = iBUG (300W)
#   98  = WFLW, markup-ul pe care e antrenat PFLD
#   106 = InsightFace 2d106det
LANDMARK_LAYOUTS = {
    68:  {"left_eye": range(36, 42), "right_eye": range(42, 48), "nose": 30, "mouth": (48, 54)},
    98:  {"left_eye": range(60, 68), "right_eye": range(68, 76), "nose": 54, "mouth": (76, 82)},
    106: {"left_eye": 38, "right_eye": 88, "nose": 86, "mouth": (52, 61)},
}


def get_5_points(landmark):
    """Cele 5 puncte ArcFace, indiferent de markup-ul modelului de landmark-uri.

    Versiunea anterioara folosea indicii iBUG-68 (36-41 pentru ochi) pe orice
    iesire. PFLD e antrenat pe WFLW si scoate 98 de puncte, unde 36-41 cad pe
    sprancene, nu pe ochi. Rezultatul: aliniere gresita, deci embedding valid
    doar pentru exact aceeasi imagine -- de aici recunoasterea pozei din baza de
    date si esecul pe orice alta poza sau pe camera live.
    """
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
    """Transformarea (rotatie + scalare + translatie) care duce src peste dst.

    Inlocuieste estimateAffinePartial2D cu LMEDS: acela e un estimator robust,
    gandit sa arunce outlieri dintr-un set mare de corespondente. Cu doar 5
    puncte, toate valide, alege subseturi minimale si poate da transformari
    degenerate sau None. Umeyama e potrivirea in sensul celor mai mici patrate
    pe toate cele 5 puncte, exact ce foloseste ArcFace la aliniere.
    """
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
    """Fata aliniata -> tensorul (3, 112, 112) cerut de w600k_mbf."""
    img_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    img_norm = (img_rgb.astype(np.float32) - 127.5) / 127.5
    return np.transpose(img_norm, (2, 0, 1))


def get_embedding(aligned_bgr):
    """Varianta pe o singura fata (folosita in afara probe-ului batched)."""
    emb = recognition_model.infer(preprocess_recognition(aligned_bgr))[0]
    return l2(emb)


def verify_embedding(embedding, db=None, threshold=VERIFY_THRESHOLD, margin=ENROLL_MARGIN):
    db = face_database if db is None else db
    return db.verify(embedding, threshold, margin)


# Cat de jos sta nasul intre linia ochilor si cea a gurii pe o fata frontala.
# Se calculeaza din chiar sablonul ArcFace (~0.495), ca referinta si formula sa
# nu poata ajunge sa spuna lucruri diferite.
def _nose_ratio(points):
    left_eye, right_eye, nose, left_mouth, right_mouth = points
    eye_vec = right_eye - left_eye
    interocular = float(np.linalg.norm(eye_vec))
    unit = eye_vec / interocular
    normal = np.array([-unit[1], unit[0]], dtype=np.float64)   # perpendiculara, in jos
    eye_center = (left_eye + right_eye) / 2.0
    mouth_center = (left_mouth + right_mouth) / 2.0
    height = float(np.dot(mouth_center - eye_center, normal))
    return float(np.dot(nose - eye_center, normal)) / height


NOSE_RATIO_FRONTAL = _nose_ratio(ARCFACE_TEMPLATE.astype(np.float64))


def face_quality(five_points, blur, min_side):
    """Cat de utilizabila e poza asta, in [0, 1]. Ieftin: doar geometrie pe 5 puncte.

    Nu e nevoie de niciun model in plus. Landmark-urile se calculeaza oricum
    inainte de aliniere, deci frontalitatea vine practic gratis:

      yaw   -- nasul, proiectat pe linia ochilor, cat de departe e de mijloc.
               Pe un profil ajunge langa un ochi; pe frontal sta la mijloc.
      pitch -- nasul, pe verticala, intre linia ochilor si cea a gurii. Capul
               ridicat sau coborat il impinge spre una dintre ele.
      roll  -- inclinarea liniei ochilor. Alinierea corecteaza rotatia, dar o
               interpolare mare tot pierde detaliu, deci penalizare mica.
      sharp -- varianta Laplacianului, deja calculata pentru poarta de blur.
      size  -- latura bbox-ului; fetele mici au mai putina informatie reala.
    """
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
        return 0.0    # gura peste ochi: landmark-uri aiurea, nu ne bazam pe ele

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

    weights = QUALITY_WEIGHTS
    return (weights["yaw"] * yaw + weights["pitch"] * pitch + weights["roll"] * roll
            + weights["sharp"] * sharp + weights["size"] * size)


def align_and_score(samples):
    """Etapa ieftina, batch-uita: landmark-uri -> aliniere -> scor de calitate.

    `samples` e o lista de PendingFace. Intoarce o lista de (aligned, quality)
    in aceeasi ordine, cu (None, 0.0) acolo unde alinierea a esuat.

    Se ruleaza pe toate probele, tocmai ca sa se poata alege ulterior cea mai
    buna. Modelul de recunoastere, care e partea scumpa, ruleaza separat si doar
    pe castigator (vezi embed_aligned).
    """
    if not samples:
        return []

    raw_landmarks = run_batched(
        landmark_model, [preprocess_landmark(s.crop) for s in samples]
    )

    scored = []
    for sample, raw in zip(samples, raw_landmarks):
        landmark = postprocess_landmark(raw, sample.crop.shape)
        five_points = get_5_points(landmark)
        aligned = align_face(sample.crop, five_points)
        if aligned is None:
            scored.append((None, 0.0))
        else:
            scored.append((aligned, face_quality(five_points, sample.blur, sample.size)))
    return scored


def embed_aligned(aligned_faces):
    """Embedding-uri pentru fete deja aliniate, intr-un singur apel GPU."""
    if not aligned_faces:
        return []
    raw = run_batched(recognition_model, [preprocess_recognition(a) for a in aligned_faces])
    return [l2(emb) for emb in raw]


def embed_faces(crops):
    """landmark -> align -> embedding pentru o lista de crop-uri.

    Intoarce cate un embedding pe crop, in aceeasi ordine, cu None unde
    alinierea a esuat. Folosita in afara probe-ului (teste, inrolare offline);
    probe-ul merge pe cele doua etape separate, ca sa nu cheltuie modelul de
    recunoastere pe probe pe care oricum le arunca.

    Verificarea si inrolarea NU se fac aici, ci raman secventiale in probe.
    Motivul: daca o fata din cadru se inroleaza, urmatoarea fata din acelasi
    cadru trebuie comparata cu baza de date care o contine deja, exact ca in
    varianta care procesa fetele una cate una. Doar partea de GPU se bateaza.
    """
    if not crops:
        return []

    samples = [PendingFace(None, None, crop, MIN_BLUR, MIN_FACE_SIZE) for crop in crops]
    scored = align_and_score(samples)

    index_map = [i for i, (aligned, _) in enumerate(scored) if aligned is not None]
    embeddings = [None] * len(crops)
    for i, emb in zip(index_map, embed_aligned([scored[i][0] for i in index_map])):
        embeddings[i] = emb
    return embeddings


def process_face(crop_bgr):
    """landmark -> align -> embedding -> verificare, pe o singura fata.

    Returneaza (label, score, embedding), sau None daca alinierea esueaza.
    Embedding-ul e intors si el, ca sa poata fi folosit la inrolarea live.
    """
    embedding = embed_faces([crop_bgr])[0]
    if embedding is None:
        return None
    label, score = verify_embedding(embedding)
    return label, score, embedding


def enroll(embedding):
    """Adauga o identitate noua si intoarce numele primit."""
    used = {int(m.group(1)) for m in
            (re.match(r"persoana_(\d+)$", label) for label in face_database.labels) if m}
    name = f"persoana_{max(used, default=0) + 1}"
    face_database.add(name, embedding)
    print(f"[INROLARE] identitate noua: {name} (total {len(face_database)})")
    return name

# ============================================================
# PROBE PRINCIPAL
# ============================================================

def full_pipeline_probe(pad, info, u_data):
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

        frame_number = frame_meta.frame_num

        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        frame_image = np.array(n_frame, copy=True, order='C')
        frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGBA2BGR)
        frame_h, frame_w = frame_image.shape[:2]

        # curatare periodica a track-urilor "moarte" (nu mai apar in cadru)
        if frame_number % PRUNE_CHECK_INTERVAL == 0:
            stale = [tid for tid, st in track_states.items()
                     if frame_number - st.last_seen_frame > TRACK_TIMEOUT_FRAMES]
            for tid in stale:
                del track_states[tid]

        if AUTO_ENROLL and frame_number % DB_SAVE_INTERVAL_FRAMES == 0 and frame_number > 0:
            save_database()

        # Pasul 1: parcurgem obiectele si doar STRANGEM probe. Nimic nu ajunge pe
        # GPU aici -- altfel s-ar face un apel pe fata, exact ce vrem sa evitam.
        visible = []   # (obj_meta, state) pentru tot ce are eticheta de desenat
        active = {}    # track_id -> state, deduplicat, pentru decizia de la pasul 3
        samples = []   # probele din cadrul curent, candidate la "cea mai buna poza"

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            track_id = obj_meta.object_id
            confidence = obj_meta.confidence
            rect = obj_meta.rect_params

            x1 = max(0, int(rect.left))
            y1 = max(0, int(rect.top))
            x2 = min(frame_w, int(rect.left + rect.width))
            y2 = min(frame_h, int(rect.top + rect.height))
            w, h = x2 - x1, y2 - y1

            if confidence >= MIN_CONFIDENCE and w >= MIN_FACE_SIZE and h >= MIN_FACE_SIZE:
                state = track_states.setdefault(track_id, TrackState())
                state.last_seen_frame = frame_number
                visible.append((obj_meta, state))
                active[track_id] = state

                # Fereastra de colectare se deschide cu QUALITY_WINDOW_FRAMES
                # inainte de termen; in ea luam o proba la cateva cadre.
                frames_since_check = frame_number - state.last_checked_frame
                window_open = frames_since_check >= state.deadline_gap - QUALITY_WINDOW_FRAMES
                due_for_sample = (frame_number - state.last_candidate_frame
                                  >= CANDIDATE_INTERVAL_FRAMES)

                if window_open and due_for_sample:
                    state.last_candidate_frame = frame_number
                    crop = frame_image[y1:y2, x1:x2]
                    b_score = blur_score(crop)
                    # Poarta de blur ramane prima filtrare, ieftina: o poza
                    # miscata nu merita nici macar landmark-urile.
                    if b_score >= MIN_BLUR:
                        samples.append(
                            PendingFace(track_id, state, crop, b_score, min(w, h))
                        )

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # Pasul 2: landmark-uri + aliniere + scor de calitate pentru toate
        # probele, intr-un singur apel GPU. Fiecare track pastreaza cea mai buna
        # poza vazuta in fereastra lui.
        for sample, (aligned, quality) in zip(samples, align_and_score(samples)):
            if aligned is not None:
                sample.state.offer(aligned, quality, sample.blur, sample.size)

        # Pasul 3: alegem pentru ce track-uri rulam recunoasterea acum. Fie
        # poza e deja foarte buna si nu are rost sa mai asteptam, fie a expirat
        # termenul si mergem cu ce am strans.
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
                # Fereastra s-a inchis fara nicio poza utilizabila: reincercam
                # mai repede, ca la poarta de calitate picata.
                state.last_checked_frame = frame_number
                state.last_check_failed = True
                state.clear_best()

        # Pasul 4: un singur apel GPU pentru toate fetele alese in acest cadru.
        embeddings = embed_aligned([state.best_aligned for _, state in to_recognize])

        # Pasul 5: verificare, inrolare si vot majoritar -- secvential, in
        # ordinea fetelor din cadru, ca fiecare fata sa fie comparata cu baza de
        # date asa cum arata ea in momentul respectiv.
        for (track_id, state), embedding in zip(to_recognize, embeddings):
            label, score = verify_embedding(embedding)
            quality, blur, size = state.best_quality, state.best_blur, state.best_size
            state.clear_best()

            # Inrolare live: fata nu seamana cu nimeni din baza, a iesit
            # "necunoscut" de mai multe ori la rand si poza e destul de buna cat
            # sa merite sa devina prototip permanent.
            if AUTO_ENROLL and label == LABEL_UNKNOWN and not state.enrolled:
                state.unknown_streak += 1
                print(f"[DEBUG ENROLL] track {track_id}: "
                      f"streak={state.unknown_streak}/{ENROLL_MIN_CHECKS} "
                      f"score={score:.3f} (nevoie <{ENROLL_MAX_SCORE:.2f}) "
                      f"size={size}px (nevoie >={ENROLL_MIN_FACE}) "
                      f"blur={blur:.1f} (nevoie >={ENROLL_MIN_BLUR}) "
                      f"calitate={quality:.2f} (nevoie >={ENROLL_MIN_QUALITY})")
                if (state.unknown_streak >= ENROLL_MIN_CHECKS
                        and score < ENROLL_MAX_SCORE
                        and size >= ENROLL_MIN_FACE
                        and blur >= ENROLL_MIN_BLUR
                        and quality >= ENROLL_MIN_QUALITY):
                    label = enroll(embedding)
                    score = 1.0
                    state.enrolled = True
                    state.history.clear()
            else:
                # Si un nume, si "incert" inseamna ca exista deja cineva
                # asemanator in baza: nu mai numaram spre o identitate noua.
                state.unknown_streak = 0

            state.history.append(label)
            state.current_label = Counter(state.history).most_common(1)[0][0]
            state.current_score = score

            if state.current_label not in (LABEL_UNKNOWN, LABEL_UNCERTAIN):
                print(f"[ALERTA] track {track_id} -> {state.current_label} "
                      f"(scor={score:.3f}, calitate={quality:.2f}, frame={frame_number})")

        # Pasul 4: scriem eticheta curenta peste text-ul implicit desenat de
        # nvdsosd. Vine dupa pasul 3, ca sa arate deciziile din cadrul curent.
        for obj_meta, state in visible:
            label_text = state.current_label if state.current_label else "..."
            display_text = f"{label_text} ({state.current_score:.2f})" if state.current_label else "verificare..."
            obj_meta.text_params.display_text = display_text
            obj_meta.text_params.set_bg_clr = 1
            obj_meta.text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)
            obj_meta.text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            obj_meta.text_params.font_params.font_size = 12

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

# ============================================================
# PIPELINE GSTREAMER
# ============================================================

def main():
    Gst.init(None)
    pipeline = Gst.Pipeline()
    if not pipeline:
        print("Eroare: Nu s-a putut crea pipeline-ul.")
        sys.exit(1)

    print("Creare elemente pipeline...")

    source = Gst.ElementFactory.make("v4l2src", "usb-cam")
    source.set_property('device', '/dev/video0')

    caps_v4l2 = Gst.ElementFactory.make("capsfilter", "v4l2-caps")
    caps_v4l2.set_property('caps', Gst.Caps.from_string("image/jpeg, width=1920, height=1080, framerate=30/1"))

    jpegdec = Gst.ElementFactory.make("jpegdec", "jpeg-decoder")

    vidconv1 = Gst.ElementFactory.make("nvvideoconvert", "converter1")

    caps_vidconv = Gst.ElementFactory.make("capsfilter", "nvmm-caps")
    caps_vidconv.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"))

    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    # 1920x1080, ca in run_yolo_tracker_crop.py, scriptul care a produs crop-urile
    # din care s-a construit baza de date. La 1280x720 fetele sunt cu 1.5x mai
    # mici, varianta Laplacianului scade sub pragul de blur calibrat pe 1080p, si
    # nicio fata nu mai trece de poarta de calitate: eticheta ramane "verificare...".
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 40000)

    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property('config-file-path', YOLO_CONFIG_PATH)

    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    tracker.set_property('tracker-width', 640)
    tracker.set_property('tracker-height', 384)
    tracker.set_property('gpu-id', 0)
    tracker.set_property('ll-lib-file', "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property('ll-config-file', TRACKER_CONFIG_PATH)

    vidconv2 = Gst.ElementFactory.make("nvvideoconvert", "converter2")

    caps_rgba = Gst.ElementFactory.make("capsfilter", "rgba-caps")
    caps_rgba.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))

    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    sink.set_property('sync', False)

    elements = [source, caps_v4l2, jpegdec, vidconv1, caps_vidconv, streammux,
                pgie, tracker, vidconv2, caps_rgba, nvosd, transform, sink]
    for el in elements:
        pipeline.add(el)

    print("Legare elemente pipeline...")
    source.link(caps_v4l2)
    caps_v4l2.link(jpegdec)
    jpegdec.link(vidconv1)
    vidconv1.link(caps_vidconv)

    sinkpad = streammux.get_request_pad("sink_0")
    srcpad = caps_vidconv.get_static_pad("src")
    srcpad.link(sinkpad)

    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(vidconv2)
    vidconv2.link(caps_rgba)
    caps_rgba.link(nvosd)
    nvosd.link(transform)
    transform.link(sink)

    rgba_src_pad = caps_rgba.get_static_pad("src")
    rgba_src_pad.add_probe(Gst.PadProbeType.BUFFER, full_pipeline_probe, 0)

    loop = GLib.MainLoop()

    def bus_call(bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End-of-stream primit.")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Eroare GStreamer: {err}: {debug}")
            loop.quit()
        return True

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    def sigint_handler(sig, frame):
        print("\nOprire ceruta de utilizator, trimit EOS...")
        pipeline.send_event(Gst.Event.new_eos())

    signal.signal(signal.SIGINT, sigint_handler)

    print("Pornire procesare. Apasa Ctrl+C pentru a opri.")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        if AUTO_ENROLL:
            save_database()
            print(f"Baza de date salvata: {len(face_database)} identitati.")
        print("Pipeline oprit cu succes!")


if __name__ == '__main__':
    main()
