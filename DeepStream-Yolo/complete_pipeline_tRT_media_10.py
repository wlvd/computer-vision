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
    python3 complete_pipeline_tRT_media_10.py sample_vid.mp4
    python3 complete_pipeline_tRT_media_10.py sample_vid.mp4       # a doua oara: recunoaste
    python3 complete_pipeline_tRT_media_10.py sample/sample_vid.mp4 --database baza.json
    python3 complete_pipeline_tRT_media_10.py sample_vid.mp4 --reset-db --overwrite

Numele videoclipului se cauta, in ordine: asa cum a fost dat, langa script, si
in subfolderul sample/ de langa script.

Despre viteza. Masurat pe videotest3.mp4 (1080x1920, cadru larg): 711 cadre in
71.6 s, adica 9.9 FPS, cu 43.3 ms in probe (fix 24.8 ms/cadru + 4.3 ms/fata, la
4 fete pe cadru) si ~57 ms in restul pipeline-ului. Detectorul scotea 42 de
casete pe cadru, din care 38 sub pragul de marime -- si toate 38 erau urmarite
de tracker si scrise in log. Ce s-a facut, in ordinea castigului:

  1. queue-uri intre elemente. Fara ele tot pipeline-ul rula pe UN SINGUR fir,
     deci cele 43 ms de probe se adunau la cele 57 ms de decodare/detectie/
     encodare in loc sa se suprapuna cu ele.
  2. casetele care nu pot trece de porti se sterg INAINTE de tracker, nu dupa:
     NvDCF filtra 42 de tinte pe cadru ca sa foloseasca 4.
  3. din suprafata se copiaza doar crop-urile fetelor, nu tot cadrul: 4 bucati
     de ~5 KB in loc de 14 MB de copiat si convertit pe fiecare cadru.
  4. punctele se deseneaza vectorizat (o scriere numpy), nu cu 424 de apeluri
     cv2.circle pe cadru.
  5. respinsele se numara, nu se scriu una cate una: 69% din log erau ele.

Primele cinci nu ating deciziile de recunoastere: aceleasi cadre, aceleasi probe,
aceleasi etichete. A sasea, in schimb, e o alegere, nu o optimizare curata:

  6. RACIREA. Dupa ce un track are identitate confirmata, nu i se mai cer
     landmark-uri pe fiecare cadru, ci doar in fereastra dinaintea urmatoarei
     verificari, iar verificarile lui se rarese de la 10 la
     IDENTIFIED_INTERVAL_FRAMES cadre. Landmark-urile erau 63% din timpul de
     probe, si pe un clip cu oameni deja in baza aproape toate se calculau pentru
     oameni pe care ii stiam deja. Simulat pe videotest3.mp4, la o a doua rulare
     (baza plina): 4.0 -> 0.5 fete cu landmark pe cadru, 86% economisit.

     Ce se plateste: daca tracker-ul schimba persoana sub acelasi id in timpul
     racirii, eticheta gresita ramane pana la urmatoarea verificare. Si nu se mai
     deseneaza punctele pe oamenii recunoscuti -- caseta si eticheta raman, ele
     merg prin metadate. Inrolarea NU e atinsa: un track fara identitate primeste
     landmark-uri pe fiecare cadru, ca inainte, pentru ca din ele se alege
     prototipul. Se dezactiveaza cu --landmark-all.

Ce se masoara efectiv se vede in summary.json, la "viteza", "timp_etape_ms" si
"apeluri_economisite".

URMARIREA. Masurat pe videotest3.mp4: zero schimbari de id, dar 120 de goluri in
13 track-uri, unul de 233 de cadre -- oamenii isi pierdeau caseta cand intorceau
capul si o recapatau dupa. Cauzele si ce s-a facut:

  - filtrul dinaintea tracker-ului taia si pe incredere, nu doar pe marime. Cand
    cineva intoarce capul, detectorul nu rateaza fata, ii scade increderea; caseta
    aia slaba e tot ce are NvDCF ca sa lege track-ul. Acum se filtreaza DOAR pe
    marime, si dupa latura mare, nu dupa cea mica (o fata din profil se ingusteaza
    fara sa piarda din inaltime).
  - minDetectorConfidence si maxShadowTrackAge din configul nvtracker se ajusteaza
    la pornire, intr-o copie scrisa in folderul rularii (vezi patch_tracker_config).
  - implicit se cere NvDCF_accuracy, nu NvDCF_perf: costa mai mult per tinta, dar
    filtrul de mai sus a taiat tintele de la 42.8 la ~13 pe cadru.
  - starea unui track nu se mai sterge cat timp tracker-ul inca il vede, chiar daca
    fata nu trece de porti: altfel un om intors 300 de cadre revenea ca necunoscut
    si putea fi inrolat a doua oara, ca identitate dublata.

ReID la nivel de tracker NU se foloseste, desi NvDCF il are: modelele livrate sunt
pentru corpuri (Market-1501, crop-uri de ~128x256), iar aici obiectele sunt fete de
40-66 px. Re-identificarea adevarata o face oricum baza de date, prin embedding-ul
ArcFace: un track rupt isi recapata numele la prima verificare.
"""

import argparse
import json
import os
import re
import signal
import subprocess
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

# NvDCF_accuracy inaintea lui NvDCF_perf: are estimare de stare mai buna si tine
# track-urile in umbra mai mult, adica exact ce lipseste cand cineva intoarce
# capul. Costa mai mult per tinta, dar filtrul dinaintea tracker-ului a taiat
# tintele de la 42.8 la ~13 pe cadru, deci acum se poate plati. Daca fisierul nu
# exista pe versiunea instalata, se cade pe perf.
_DS_CONFIGS = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app"
TRACKER_CONFIG_PATH = [
   # f"{_DS_CONFIGS}/config_tracker_NvDCF_accuracy.yml",
    f"{_DS_CONFIGS}/config_tracker_NvDCF_perf.yml"
]

# Ce se schimba in configul low-level al tracker-ului inainte de rulare. Nu se
# editeaza fisierul din DeepStream: se scrie o copie in folderul rularii si
# nvtracker primeste copia, deci rularea ramane reproductibila si originalul
# neatins. None = se lasa cum e in fisier.
#
# minDetectorConfidence: 0.0, adica tracker-ul primeste TOATE casetele. Fetele
#   slabe (cap intors, ocluzie partiala) sunt tocmai cele care tin track-ul legat;
#   filtrarea lor pe calitate se face mai tarziu, in probe, unde nu strica decat
#   fetei respective. Vezi si detection_filter_probe.
# maxShadowTrackAge: cate cadre supravietuieste un track fara nicio detectie.
#   Masurat pe videotest3.mp4: 120 de goluri in 13 track-uri, cel mai lung de 233
#   de cadre. Implicitul NvDCF (~30) le rupe pe aproape toate.
TRACKER_OVERRIDES = {
    "minDetectorConfidence": 0.0,
    "maxShadowTrackAge": 90,
}

PFLD_MODEL_PATH = "/workspace/DeepStream-Yolo/pfld.engine"
RECOGNITION_MODEL_PATH = "/workspace/DeepStream-Yolo/w600k_mbf.engine"
FACE_DATABASE_PATH = "/workspace/DeepStream-Yolo/face_database.json"

MIN_CONFIDENCE = 0.5          # sub asta, nici nu incercam sa procesam fata
MIN_FACE_SIZE = 40            # px, latura minima a bbox-ului
MIN_BLUR = 80.0              # varianta Laplacianului la care consideram poza clara

# Casetele prea mici ca sa fie vreodata fete utilizabile se sterg din metadate
# imediat dupa detector, deci tracker-ul nici nu le vede. Masurat pe
# videotest3.mp4: 42 de casete pe cadru, din care 4 treceau -- NvDCF facea
# urmarire prin corelatie pentru 38 de casete degeaba, pe fiecare cadru.
#
# Pragul de aici e mai jos decat MIN_FACE_SIZE, cu factorul de mai jos, si asta
# e intentionat: o fata care oscileaza in jurul pragului trebuie sa-si pastreze
# track-ul si in cadrele in care e cu un pixel prea mica, altfel primeste alt id
# la fiecare oscilatie si nu aduna niciodata ENROLL_MIN_CHECKS verificari.
#
# Filtrul NU se uita la incredere: vezi detection_filter_probe pentru de ce.
PRETRACK_FILTER = True
PRETRACK_SIZE_FACTOR = 0.8
PRETRACK_MIN_SIZE = MIN_FACE_SIZE * PRETRACK_SIZE_FACTOR   # recalculat in apply_thresholds

# Detaliul fiecarei casete respinse in log. Implicit doar numarate pe poarta:
# vezi comentariul din process_frame.
LOG_REJECTED = False

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
VERIFY_INTERVAL_FRAMES = 15   # la cat timp cel mult re-verificam un track activ
RETRY_INTERVAL_ON_FAIL = 7    # daca poarta de calitate a picat, reincercam mai repede

# Cadenta pentru track-urile care AU deja o identitate confirmata. Odata ce stim
# cine e cineva, tot ce mai facem cu el e sa ne asiguram ca tracker-ul nu ne-a
# schimbat omul sub acelasi id -- si asta nu cere o verificare la 10 cadre.
# Masurat pe videotest3.mp4: 4 fete pe cadru, landmark-urile 18.1 ms din 28.9 ms
# de probe, adica 63% din tot ce se calculeaza pe cadru mergea pe oameni deja
# recunoscuti. La 30 de cadre (~1.2 s la 25 FPS) ramane destul de des cat sa
# prindem un id schimbat, dar de trei ori mai rar.
#
# Riscul, explicit: daca tracker-ul schimba persoana sub acelasi id in timpul
# racirii, eticheta gresita ramane afisata pana la urmatoarea verificare.
IDENTIFIED_INTERVAL_FRAMES = 30
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
ENROLL_MIN_FACE = 40

# Cea mai importanta poarta pentru ce ajunge in baza, si singura care nu se poate
# compensa. Masurat pe sample_vid.mp4: prototipurile cu frontalitate 0 au dat
# autosimilaritate 0.089 fata de alte poze ale aceleiasi persoane -- adica baza
# se umple cu intrari care nu vor recunoaste pe nimeni, niciodata. Cele cu 0.29
# si 0.52 au dat 0.80 si 0.85. Pragul taie exact profilurile, fara sa ceara poze
# de buletin: 40% din fetele clipului au frontalitate 0.
ENROLL_MIN_FRONTALITY = 0.15
ENROLL_MIN_BLUR = 80.0
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
# Toate punctele se deseneaza cu OpenCV, direct pe suprafata mapata. Pe Jetson
# memoria e unificata, deci suprafata pe care o primim de la pyds e chiar cea pe
# care o citesc nvdsosd si encoder-ul: ce scriem aici ajunge in fisier. Cele 5
# puncte treceau inainte prin display meta (nvdsosd), adica doua cai de desen
# pentru acelasi lucru, cu limita de MAX_ELEMENTS_IN_DISPLAY_META si cu un simbol
# pyds care nu exista in toate versiunile. Daca suprafata nu e scriptibila
# (dGPU), se pierde tot desenul de puncte -- casetele si etichetele raman, ele
# merg prin metadate, nu prin pixeli.
# Landmark-urile se cer doar pentru fetele care chiar au nevoie de ele: cele care
# dau o proba in cadrul asta, si cele ale caror track-uri inca n-au identitate
# (acolo landmark-urile aleg prototipul, deci se cer pe fiecare cadru, ca pana
# acum). Restul -- oameni deja recunoscuti, intre doua verificari -- raman doar
# urmariti de tracker. Se pierde desenul punctelor pe ei; caseta si eticheta nu,
# ele merg prin metadate.
LANDMARK_ONLY_WHEN_NEEDED = True

DRAW_ALL_LANDMARKS = True
DRAW_OVERLAY = True        # se stinge cu --no-video: nu mai are cine sa vada desenul
LANDMARK_RADIUS = 2

# Culori in ordinea canalelor suprafetei (RGBA), nu normalizate: aici scriem
# pixeli, nu completam structuri nvdsosd.
LANDMARK_COLOR = (255, 255, 0, 255)     # setul complet, galben
POINT_COLORS = np.array([  # ochi stang, ochi drept, nas, gura stanga, gura dreapta
    (0, 204, 255, 255),
    (0, 204, 255, 255),
    (51, 255, 51, 255),
    (255, 102, 102, 255),
    (255, 102, 102, 255),
], dtype=np.uint8)


def disc_offsets(radius):
    """Deplasarile (dy, dx) ale pixelilor dintr-un disc de raza data.

    Se calculeaza o data, la import: un punct desenat inseamna apoi o singura
    scriere numpy indexata, nu un apel cv2.circle. La 106 puncte pe fata si 4
    fete pe cadru, diferenta e intre o operatie si 424.
    """
    span = np.arange(-radius, radius + 1)
    dy, dx = np.meshgrid(span, span, indexing="ij")
    inside = dy * dy + dx * dx <= radius * radius
    return dy[inside].ravel(), dx[inside].ravel()


LANDMARK_DISC = disc_offsets(LANDMARK_RADIUS)
FIVE_POINT_DISC = disc_offsets(LANDMARK_RADIUS + 1)
BOX_COLOR_KNOWN = (0.0, 1.0, 0.0, 1.0)
BOX_COLOR_UNCERTAIN = (1.0, 0.8, 0.0, 1.0)
BOX_COLOR_UNKNOWN = (1.0, 0.3, 0.3, 1.0)
BOX_COLOR_PENDING = (0.6, 0.6, 0.6, 1.0)

PROGRESS_EVERY_FRAMES = 100

# ============================================================
# COMPATIBILITATE pyds
# ============================================================
# Bindings-urile Python nu expun aceleasi simboluri in toate versiunile de
# DeepStream. Ce e optional il rezolvam o singura data, aici, si nu in probe --
# altfel un simbol lipsa arunca acelasi AttributeError pe fiecare cadru, iar
# pipeline-ul merge mai departe si umple consola cu acelasi traceback.

_unmap_surface = getattr(pyds, "unmap_nvds_buf_surface", None)
_remove_obj_meta = getattr(pyds, "nvds_remove_obj_meta_from_frame", None)

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

    global PRETRACK_FILTER
    if PRETRACK_FILTER and _remove_obj_meta is None:
        PRETRACK_FILTER = False
        warn_once("remove_obj", "pyds nu are nvds_remove_obj_meta_from_frame; "
                                "tracker-ul va primi si casetele prea mici, deci "
                                "rularea va fi mai lenta (rezultatele, aceleasi).")


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

            # Se copiaza doar esantioanele cerute, nu tot bufferul alocat la batch
            # maxim. Bufferele sunt dimensionate o data, pentru cazul cel mai rau,
            # dar un cadru obisnuit are 4 fete si engine-urile au batch 8: fara
            # feliile de aici se mutau de doua ori mai multi octeti decat trebuia,
            # de doua ori pe apel (dus si intors), pe fiecare cadru.
            cuda.memcpy_htod_async(source["device"], source["host"][:flat.size],
                                   self.stream)
            self.context.execute_async_v2(
                bindings=self.bindings, stream_handle=self.stream.handle
            )
            results = []
            for entry in self.outputs:
                view = entry["host"][: count * entry["sample_elems"]]
                cuda.memcpy_dtoh_async(view, entry["device"], self.stream)
                results.append((entry, view))
            self.stream.synchronize()
        finally:
            CUDA_CONTEXT.pop()

        return [view.reshape((count,) + entry["sample_shape"]).copy()
                for entry, view in results]


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
    def __init__(self):
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

        self.checks = 0             # doar pentru raportul final

    @property
    def resolved(self):
        """Track-ul are deja o identitate, nu doar o banuiala.

        LABEL_UNKNOWN si LABEL_UNCERTAIN nu se pun: primul inca vaneaza un
        prototip pentru inrolare, al doilea inca poate deveni o potrivire adevarata
        la o proba mai buna. Amandoua au nevoie de landmark-uri pe fiecare cadru.
        """
        return (self.current_label is not None
                and self.current_label not in (LABEL_UNKNOWN, LABEL_UNCERTAIN))

    @property
    def deadline_gap(self):
        if self.last_check_failed:
            return RETRY_INTERVAL_ON_FAIL
        return IDENTIFIED_INTERVAL_FRAMES if self.resolved else VERIFY_INTERVAL_FRAMES

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
                 "frontality", "wants_sample", "action")

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
        self.action = "vazut"       # ce s-a intamplat cu fata, pentru log

# ============================================================
# PROCESARE
# ============================================================

def crop_bgr(surface_rgba, box):
    """Fata decupata din suprafata mapata, convertita in BGR.

    Se converteste DOAR dreptunghiul fetei, nu tot cadrul. Inainte se copia
    suprafata intreaga si se convertea RGBA->BGR pe ea (la 1080x1920: 8 MB de
    copiat plus 14 MB de citit-scris in conversie, pe fiecare cadru), desi din ea
    se foloseau patru bucati de cate ~43x43 px. Asta era grosul celor 24.8 ms
    fixe pe cadru din masuratoare.

    Rezultatul e un tablou nou, deci desenul de la finalul cadrului nu ajunge in
    crop-urile deja luate -- fetele intra curate in modele.
    """
    x1, y1, x2, y2 = box
    return cv2.cvtColor(surface_rgba[y1:y2, x1:x2], cv2.COLOR_RGBA2BGR)


def blur_score(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # CV_32F, nu CV_64F: varianta iese aceeasi pe crop-uri de zeci de pixeli, dar
    # se muta jumatate din octeti.
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


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

def stamp_points(surface_rgba, points, disc, color):
    """Deseneaza TOATE punctele deodata, printr-o singura scriere indexata.

    points e (n, 2) in coordonate de cadru, color e ori o culoare RGBA, ori un
    tablou (n, 4) cu cate una de fiecare punct. Discul e precalculat, deci aici
    nu ramane decat o adunare cu broadcast si o masca.

    Pixelii din afara cadrului se ARUNCA, nu se prind de margine: modelul de
    landmark-uri poate scoate puncte in afara crop-ului, iar o taiere cu clip
    le-ar aduna pe toate pe chenarul imaginii, unde n-are ce cauta niciun punct
    facial. Asa se comporta si cv2.circle, deci desenul iese la fel ca inainte.
    """
    if len(points) == 0:
        return
    height, width = surface_rgba.shape[:2]
    dy, dx = disc
    centres = np.rint(points).astype(np.int32)
    ys = centres[:, 1, None] + dy
    xs = centres[:, 0, None] + dx
    inside = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
    if isinstance(color, np.ndarray) and color.ndim == 2:
        # o culoare pe punct, repetata peste tot discul lui
        color = np.broadcast_to(color[:, None, :], ys.shape + (4,))[inside]
    surface_rgba[ys[inside], xs[inside]] = color


def draw_landmarks(surface_rgba, faces):
    """Punctele faciale, direct pe suprafata mapata.

    Cele 5 puncte ArcFace se deseneaza ultimele si cu raza mai mare, ca sa se
    vada peste setul complet. Punctele tuturor fetelor se aduna intr-un singur
    tablou: inainte fiecare punct insemna un apel cv2.circle, adica 106 puncte x
    4 fete = 424 de treceri prin bindings pe fiecare cadru, pentru cateva mii de
    pixeli scrisi in total.

    Desenul nu are voie sa strice procesarea: daca suprafata nu e scriptibila, se
    raporteaza o data si se merge mai departe -- recunoasterea si logul sunt
    oricum treaba importanta, adnotarea e doar ca sa se vada. Casetele si
    etichetele nu se pierd: ele merg prin metadate.
    """
    drawn = [face for face in faces if face.landmarks is not None]
    if not drawn:
        return

    try:
        if DRAW_ALL_LANDMARKS:
            stamp_points(surface_rgba,
                         np.concatenate([face.landmarks for face in drawn]),
                         LANDMARK_DISC, LANDMARK_COLOR)
        stamp_points(surface_rgba,
                     np.concatenate([face.five_points for face in drawn]),
                     FIVE_POINT_DISC, np.tile(POINT_COLORS, (len(drawn), 1)))
    except Exception as error:      # suprafata nescriptibila / alt layout
        warn_once("surface", f"nu pot desena pe suprafata ({error}); "
                             f"raman doar casetele si etichetele.")


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
        # None = rularea a mers pana la capatul fisierului; altfel, eroarea de pe
        # magistrala GStreamer care a oprit-o.
        self.failure = None

        # Cate casete a scos detectorul si cate au ajuns la tracker. Diferenta e
        # munca economisita de filtrul dinaintea lui.
        self.detections = 0
        self.detections_kept = 0
        # Marimea casetelor taiate de poarta de marime, oriunde ar fi fost taiate
        # (inainte de tracker sau in probe). Fara ele nu se poate raspunde la
        # singura intrebare care conteaza cand baza iese goala: "cat as castiga
        # daca as cobori pragul?" -- detaliul fiecarei casete nu se mai scrie in
        # log, dar distributia costa o adaugare la lista si spune acelasi lucru.
        self.rejected_sizes = []
        # Unde se duc milisecundele din probe, insumate pe toata rularea. Fara
        # ele, "43 ms pe cadru" nu spune ce anume sa optimizezi in continuare;
        # perf_counter costa ~50 ns pe apel, deci masuram mereu, nu sub un flag.
        self.stage_s = Counter()


RUN = None

# ============================================================
# PROBE PRINCIPAL
# ============================================================

def iter_frames(gst_buffer):
    """Cadrele din batch. Aceeasi plimbare prin lista inlantuita, o singura data."""
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            return
        yield frame_meta
        try:
            l_frame = l_frame.next
        except StopIteration:
            return


def iter_objects(frame_meta):
    """Obiectele unui cadru.

    Nodul urmator se citeste DUPA yield, deci consumatorul nu are voie sa stearga
    metadate in timp ce se plimba: intai se strange ce trebuie sters, abia apoi
    se sterge (vezi detection_filter_probe).
    """
    l_obj = frame_meta.obj_meta_list
    while l_obj is not None:
        try:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
        except StopIteration:
            return
        yield obj_meta
        try:
            l_obj = l_obj.next
        except StopIteration:
            return


def detection_filter_probe(pad, info, u_data):
    """Sterge, chiar dupa detector, casetele prea mici ca sa poata fi vreodata fete.

    Tracker-ul e cel mai scump element de dupa nvinfer si costa proportional cu
    numarul de tinte: NvDCF tine cate un filtru prin corelatie pentru fiecare.
    Masurat pe videotest3.mp4, detectorul scotea 42.2 casete pe cadru si doar 4.0
    treceau de porti -- restul erau urmarite fara sa poata ajunge la un model.

    SE FILTREAZA DOAR PE MARIME, niciodata pe incredere. O versiune anterioara
    arunca aici si casetele sub MIN_CONFIDENCE, si asta strica urmarirea exact
    cand conteaza mai mult: cand cineva intoarce capul, detectorul nu rateaza
    fata, ci ii scade increderea la 0.3-0.4. Caseta aia slaba e tot ce are NvDCF
    ca sa lege track-ul mai departe -- e chiar mecanismul pe care se bazeaza
    asocierea in doi pasi din ByteTrack. Aruncata aici, tracker-ul ramane fara
    nicio observatie si trebuie sa mearga pe predictie pana expira track-ul.
    Masurat: 120 de goluri in 13 track-uri, unul de 233 de cadre. Fetele slabe
    sunt oricum respinse mai tarziu, in probe, unde nu strica decat lor.

    Marimea se ia dupa latura MARE, nu dupa cea mica: o fata intoarsa din profil
    se ingusteaza pe orizontala fara sa piarda din inaltime, deci min(w, h) ar
    taia-o taman in cadrele in care tracker-ul are mai multa nevoie de ea. Poarta
    adevarata de marime, cu min(w, h), ramane in probe.

    Stergerea se face DUPA ce s-a terminat plimbarea prin lista: metadatele sunt
    o lista inlantuita, iar nvds_remove_obj_meta_from_frame elibereaza nodul, deci
    un l_obj.next luat dupa stergere ar citi memorie eliberata.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    for frame_meta in iter_frames(gst_buffer):
        doomed = []
        for obj_meta in iter_objects(frame_meta):
            rect = obj_meta.rect_params
            if max(rect.width, rect.height) < PRETRACK_MIN_SIZE:
                doomed.append(obj_meta)
                RUN.rejected_sizes.append(min(rect.width, rect.height))

        RUN.detections += frame_meta.num_obj_meta
        for obj_meta in doomed:
            _remove_obj_meta(frame_meta, obj_meta)
        RUN.detections_kept += frame_meta.num_obj_meta

    return Gst.PadProbeReturn.OK


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

        # Suprafata se foloseste ca atare, fara copie: din ea se decupeaza doar
        # fetele (crop_bgr), iar tot pe ea se deseneaza la final. Copia intreaga
        # de dinainte era cel mai scump lucru din probe si servea doar ca sa fie
        # crop-urile curate -- dar crop_bgr scoate oricum tablouri noi, luate
        # inainte de pasul de desen.
        surface = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        try:
            frame_h, frame_w = surface.shape[:2]
            record = process_frame(frame_meta, frame_number,
                                   frame_w, frame_h, surface)
        finally:
            if _unmap_surface is not None:
                _unmap_surface(hash(gst_buffer), frame_meta.batch_id)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        record["ms"] = r(elapsed_ms, 2)

        logged = time.perf_counter()
        RUN.logger.write(record)
        log_s = time.perf_counter() - logged
        RUN.stage_s["log"] += log_s

        # In probe_ms intra si scrisul in log: ruleaza tot pe firul asta, deci tot
        # el tine cadrul pe loc. "ms" din log ramane doar partea de procesare, ca
        # sa se poata compara cu rularile de dinainte.
        RUN.probe_ms.append(elapsed_ms + log_s * 1000.0)
        RUN.frames += 1

        if PROGRESS_EVERY_FRAMES and frame_number % PROGRESS_EVERY_FRAMES == 0:
            print(f"  cadru {frame_number}: {len(record['faces'])} fete, "
                  f"{len(face_database)} identitati, {elapsed_ms:.1f} ms")

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def process_frame(frame_meta, frame_number, frame_w, frame_h, surface):
    """Toata logica pe un cadru. Intoarce inregistrarea pentru log."""
    pts_seconds = frame_meta.buf_pts / 1e9 if frame_meta.buf_pts else 0.0
    started = time.perf_counter()

    if frame_number % PRUNE_CHECK_INTERVAL == 0:
        stale = [tid for tid, st in track_states.items()
                 if frame_number - st.last_seen_frame > TRACK_TIMEOUT_FRAMES]
        for tid in stale:
            del track_states[tid]

    # --- Pasul 1: strangem fetele vizibile, fara sa atingem GPU-ul ---
    # Respinsele se NUMARA pe poarta, nu se scriu una cate una. Pe un cadru larg
    # detectorul scoate zeci de casete prea mici, iar detaliul lor era 69% din
    # log (1.9 MB din 2.7 MB pe o rulare de 711 cadre) fara sa fie citit vreodata:
    # ce se cauta acolo e "cate si de ce", si aia ramane. Cu --log-respinse se
    # intoarce lista intreaga, pentru cand chiar se vaneaza o caseta anume.
    faces = []
    rejected = Counter()
    rejected_detail = [] if LOG_REJECTED else None
    active = {}

    for obj_meta in iter_objects(frame_meta):
        track_id = int(obj_meta.object_id)
        confidence = float(obj_meta.confidence)
        rect = obj_meta.rect_params

        x1 = max(0, int(rect.left))
        y1 = max(0, int(rect.top))
        x2 = min(frame_w, int(rect.left + rect.width))
        y2 = min(frame_h, int(rect.top + rect.height))
        w, h = x2 - x1, y2 - y1

        if confidence < MIN_CONFIDENCE:
            gate = "incredere_mica"
        elif w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            gate = "prea_mica"
        else:
            gate = None

        if gate is not None:
            rejected[gate] += 1
            if gate == "prea_mica":
                RUN.rejected_sizes.append(min(w, h))
            if rejected_detail is not None:
                rejected_detail.append({"track": track_id, "box": [x1, y1, x2, y2],
                                        "det": r(confidence), "poarta": gate})

            # Fata n-a trecut de porti, dar track-ul e viu si tracker-ul inca il
            # tine: il marcam ca vazut, ca sa nu-i stergem starea. Altfel un om
            # care sta intors 300 de cadre (masurat: goluri de pana la 233) isi
            # pierde eticheta, streak-ul si prototipul, revine ca necunoscut si
            # poate fi inrolat a doua oara, ca identitate dublata.
            #
            # NU se creeaza stare noua aici: ar insemna cate un TrackState pentru
            # fiecare dintre zecile de casete mici de pe cadru, care n-au trecut
            # niciodata de porti si nici n-o sa treaca.
            known = track_states.get(track_id)
            if known is not None:
                known.last_seen_frame = frame_number
            continue

        state = track_states.setdefault(track_id, TrackState())
        state.last_seen_frame = frame_number
        active[track_id] = state
        report = track_reports.setdefault(
            track_id, {"primul_cadru": frame_number, "verificari": 0,
                       "eticheta": None, "scor": 0.0, "inrolat": False,
                       "marime_maxima": 0}
        )
        report["ultimul_cadru"] = frame_number
        # Cat de mare a ajuns fata track-ului, oricand in viata lui. Asta arata pe
        # loc care track-uri au fost taiate DOAR de ENROLL_MIN_FACE: unul cu
        # marime_maxima 48 si prag 50 n-a avut nicio sansa, oricat de bine ar fi
        # fost privit.
        report["marime_maxima"] = max(report["marime_maxima"], min(w, h))

        crop = crop_bgr(surface, (x1, y1, x2, y2))
        faces.append(FaceSample(track_id, state, obj_meta, crop,
                                blur_score(crop), min(w, h), (x1, y1, x2, y2)))

    RUN.faces_seen += len(faces)
    RUN.stage_s["citire"] += time.perf_counter() - started

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
    # NU pe toate fetele. Un track fara identitate primeste landmark-uri pe fiecare
    # cadru, pentru ca din ele se alege prototipul si un track poate deveni bun
    # exact intre doua verificari (masurat pe sample_vid.mp4: track 47 avea un
    # singur cadru bun, la 161, intre verificarile de la 152 si 162). Dar un om
    # deja recunoscut nu mai are ce prototip sa caute: pentru el se calculeaza
    # ceva doar cand da o proba, adica in fereastra dinaintea urmatoarei
    # verificari. Intre timp ramane doar urmarit de tracker.
    #
    # Aici era grosul costului: 63% din timpul de probe se ducea pe landmark-uri,
    # iar pe un clip cu oameni deja in baza aproape toate erau pe ei.
    started = time.perf_counter()
    if LANDMARK_ONLY_WHEN_NEEDED:
        analysed = [face for face in faces
                    if face.wants_sample or not face.state.resolved]
    else:
        analysed = faces
    analyse_faces(analysed)
    RUN.stage_s["landmark"] += time.perf_counter() - started

    # --- Pasul 4: statistici + prototipul, indiferent de cadenta verificarilor ---
    for face in faces:
        # Marimea si claritatea se stiu fara GPU, deci se numara pentru toti.
        RUN.face_sizes.append(face.size)
        RUN.face_blurs.append(face.blur)
        if face.landmarks is None:
            # Fara landmark-uri nu exista nici calitate, nici frontalitate: ele NU
            # se pun in distributii ca zerouri, altfel raportul ar arata o filmare
            # din profil acolo unde de fapt n-am masurat nimic.
            face.action = "urmarit"
            continue
        RUN.face_qualities.append(face.quality)
        RUN.face_frontalities.append(face.frontality)
        if face.aligned is not None:
            face.state.offer_unknown(face.aligned, face.quality, face.frontality,
                                     face.blur, face.size, frame_number)

    # --- Pasul 5: care fete intra in concursul pentru "cel mai bun cadru" ---
    for face in faces:
        state = face.state
        if not face.wants_sample:
            continue

        # Ritmul probelor se respecta si cand proba e aruncata: altfel o fata
        # neclara ar fi reincercata pe fiecare cadru din fereastra.
        state.last_candidate_frame = frame_number

        if face.blur < MIN_BLUR * BLUR_REJECT_FACTOR:
            face.action = "sarit_blur"
        elif face.aligned is None:
            face.action = "aliniere_esuata"
        else:
            # Probele sub MIN_BLUR intra in concurs, dar se vad in log: daca in
            # toata rularea nu apare decat "proba_neclara", pragul de claritate e
            # nepotrivit pentru filmarea asta, nu fetele sunt de vina.
            face.action = "proba" if face.blur >= MIN_BLUR else "proba_neclara"
            state.offer(face.aligned, face.quality, face.frontality,
                        face.blur, face.size)

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

    # --- Pasul 7: verificare (cine e), fara inrolare ---
    # Un singur apel GPU pentru toate fetele alese la pasul anterior.
    started = time.perf_counter()
    embeddings = embed_aligned([state.best_aligned for _, state in to_recognize])
    RUN.stage_s["recunoastere"] += time.perf_counter() - started
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
        started = time.perf_counter()
        for face in faces:
            label_object(face.obj_meta, face.state)
        draw_landmarks(surface, faces)
        RUN.stage_s["desen"] += time.perf_counter() - started

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

    record = {
        "cadru": frame_number,
        "timp": r(pts_seconds),
        "faces": face_records,
        "respinse": dict(rejected),
        "gpu": {"landmark": len(faces), "recunoastere": len(to_recognize)},
        "identitati": len(face_database),
    }
    if rejected_detail:
        record["respinse_detaliu"] = rejected_detail
    return record

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


def make_queue(name, depth):
    """Un queue cu limite doar pe numarul de buffere, niciodata cu pierderi.

    Fara queue-uri, TOT pipeline-ul ruleaza pe un singur fir: fiecare element il
    apeleaza direct pe urmatorul, deci cele ~43 ms petrecute in probe se adunau
    la cele ~57 ms de decodare, detectie, urmarire si encodare in loc sa se
    suprapuna cu ele. Un queue rupe lantul in doua fire, iar cele doua bucati
    merg in paralel pe cadre diferite.

    Limitele pe octeti si pe timp se scot, altfel se ating primele si queue-ul
    blocheaza mai devreme decat vrem; adancimea se tine mica pentru ca fiecare
    buffer in asteptare e o suprafata NVMM (la 1080x1920 RGBA, ~8 MB bucata).
    Fara pierderi (leaky ramane 0): se logheaza fiecare cadru, deci un cadru
    aruncat ar fi o gaura in raport, nu doar o imagine lipsa.
    """
    queue = make_element("queue", name)
    queue.set_property("max-size-buffers", depth)
    queue.set_property("max-size-bytes", 0)
    queue.set_property("max-size-time", 0)
    return queue


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
# Un telefon care filmeaza vertical nu roteste pixelii: fluxul ramane 1920x1080,
# iar orientarea e doar o eticheta in container. Decodorul o ignora, deci fara
# corectia de aici fetele ajung culcate la detector, care nu le mai gaseste.
# Peste tot in script, "rotatie" inseamna cate grade IN SENSUL ACELOR DE CEAS
# trebuie rotita imaginea ca sa se vada corect.

# nvvideoconvert: 1 = 90 grade trigonometric, 2 = 180, 3 = 90 in sensul acelor
# de ceas.
ROTATION_TO_FLIP_METHOD = {90: 3, 180: 2, 270: 1}


def normalise_rotation(degrees):
    """Rotunjeste la cel mai apropiat multiplu de 90 si aduce in [0, 360)."""
    try:
        return int(round(float(degrees) / 90.0) * 90) % 360
    except (TypeError, ValueError):
        return 0


# ------------------------------------------------------------
# Sondarea fisierului: rezolutie si rotatie
# ------------------------------------------------------------
# Cele doua numere se citesc din antet, cu ffprobe, si niciodata cu un decodor.
# Motivul e memoria, nu viteza: GstPbutils.Discoverer -- si orice altceva care
# porneste un decodebin -- ajunge pe Jetson la nvv4l2decoder, care isi deschide
# pool-ul de suprafete la rezolutia SURSEI. La un fisier 4K asta inseamna zeci de
# MB de NVMM ocupati inainte ca pipeline-ul adevarat sa-si ceara memoria lui, si
# de acolo vine "NvMapMemAllocInternalTagged ... error 12" (adica ENOMEM), urmat
# imediat de CUDNN_STATUS_INTERNAL_ERROR in convolutiile detectorului.
#
# ffprobe citeste doar antetul, strict pe CPU, si merge pe orice container --
# spre deosebire de varianta anterioara, care parsa manual box-urile MP4/MOV si
# nu stia nimic despre MKV sau AVI. Rezerva e OpenCV, tot pe CPU: decodarea lui
# software nu atinge NVDEC, deci nu strica nimic daca ffprobe lipseste.


def _ffprobe_rotation(stream):
    """Rotatia de aplicat, in grade in sensul acelor de ceas.

    Doua conventii, cu semne diferite. Eticheta veche (tags.rotate) e deja
    exprimata ca "roteste cu atat CW". Matricea de afisare, in schimb, e
    raportata de ffmpeg cu semn invers -- un clip filmat vertical, care trebuie
    rotit 90 CW, apare acolo ca rotation: -90 -- deci i se schimba semnul.
    """
    tag = (stream.get("tags") or {}).get("rotate")
    if tag is not None:
        return normalise_rotation(tag)
    for side_data in stream.get("side_data_list") or []:
        if "rotation" in side_data:
            return normalise_rotation(-float(side_data["rotation"]))
    return 0


def _probe_with_ffprobe(video_path, timeout_s):
    """Dict cu rezolutia si rotatia, sau None daca ffprobe nu poate raspunde."""
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_streams", "-of", "json", video_path]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=timeout_s)
    except (OSError, subprocess.SubprocessError) as error:
        warn_once("ffprobe", f"nu pot rula ffprobe ({error}); incerc cu OpenCV.")
        return None

    if result.returncode != 0:
        # stderr, nu doar codul. Un cod 127 singur nu spune nimic; randul din
        # stderr spune exact ce lipseste ("error while loading shared libraries:
        # libavfilter.so.7"), adica diferenta dintre "reinstaleaza ffmpeg" si o
        # ora de cautat. Prima linie ajunge, restul sunt de obicei ecouri.
        detail = (result.stderr or b"").decode("utf-8", "replace").strip()
        first_line = detail.splitlines()[0] if detail else "fara mesaj pe stderr"
        warn_once("ffprobe", f"ffprobe a iesit cu codul {result.returncode} "
                             f"({first_line}); incerc cu OpenCV.")
        return None

    try:
        streams = json.loads(result.stdout or b"{}").get("streams") or []
    except ValueError as error:
        warn_once("ffprobe", f"nu inteleg raspunsul lui ffprobe ({error}); "
                             f"incerc cu OpenCV.")
        return None

    for stream in streams:
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        if width > 0 and height > 0:
            return {"width": width, "height": height,
                    "rotation": _ffprobe_rotation(stream),
                    "citit_cu": "ffprobe"}
    return None


def probe_video_info(video_path, timeout_s=10):
    """Rezolutia si orientarea fisierului, citite inainte de a construi pipeline-ul."""
    info = _probe_with_ffprobe(video_path, timeout_s)
    if info is not None:
        return info

    capture = cv2.VideoCapture(video_path)
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orientation_prop = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
        rotation = (normalise_rotation(capture.get(orientation_prop))
                    if orientation_prop is not None else 0)
    finally:
        capture.release()

    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"Nu pot afla rezolutia lui {video_path}. Da-o pe loc, cu "
            f"--width si --height."
        )
    return {"width": width, "height": height, "rotation": rotation,
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

    # Scalarea cadrelor pentru tracker se face pe GPU, nu pe VIC. Pe Jetson,
    # implicit ajunge la VIC, iar la rezolutiile care nu-i convin scoate "NvVic
    # handle" sau erori 12 (memorie CMA) si pipeline-ul cade. Proprietatea nu
    # exista pe toate versiunile de DeepStream, deci o punem doar daca e acolo.
    if tracker.find_property("compute-hw"):
        tracker.set_property('compute-hw', 1)       # 0 implicit, 1 GPU, 2 VIC
        print("  tracker: compute-hw=1 (GPU)")

    vidconv_osd = make_element("nvvideoconvert", "conversie-osd")
    # Un queue nu se poate umple peste cate buffere are pool-ul elementului de
    # dinaintea lui: cu pool-ul implicit (4), cele trei fire s-ar bloca unul pe
    # altul si suprapunerea s-ar pierde. Proprietatea nu exista pe toate
    # versiunile de DeepStream, deci se pune doar daca e acolo.
    for converter in (vidconv_in, vidconv_osd):
        if converter.find_property("output-buffers"):
            converter.set_property("output-buffers", args.queue_size + 4)

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
        tail = [sink]
    else:
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
                encoder.set_property('profile', 4)      # 0 Baseline, 2 Main, 4 High
            # Bitrate variabil: scenele simple consuma putin, iar cele cu miscare --
            # exact acolo unde se pierd fetele -- primesc cat le trebuie.
            if encoder.find_property("control-rate"):
                encoder.set_property('control-rate', 0)  # 0 variabil, 1 constant
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
        tail = [nvosd, vidconv_out, caps_out, encoder, parser, muxer, sink]

    # Firele de executie. Un queue inainte de conversia pentru probe si unul dupa
    # ea taie lantul in trei bucati care merg in paralel, pe cadre diferite:
    #
    #   [decodare -> streammux -> detector -> tracker] | [probe] | [osd -> encoder]
    #
    # Probe-ul ruleaza pe firul care impinge in queue-ul de dupa caps_rgba, deci
    # nici detectorul dinaintea lui, nici encoderul de dupa nu il mai asteapta.
    # Asta e singura schimbare care nu atinge deloc ce se calculeaza: aceleasi
    # cadre, aceleasi decizii, doar nu una dupa alta.
    queue_pre = make_queue("coada-detectie", args.queue_size)
    queue_probe = make_queue("coada-probe", args.queue_size)
    queue_post = make_queue("coada-iesire", args.queue_size)

    # Lantul dupa streammux e liniar, deci se leaga in bucla: singurele legaturi
    # speciale sunt sursa (pad creat tarziu) si intrarea in streammux (pad cerut).
    chain = ([streammux, pgie, tracker, queue_probe, vidconv_osd, caps_rgba,
              queue_post] + tail)
    for element in [source, vidconv_in, caps_in, queue_pre] + chain:
        pipeline.add(element)

    print("Legare elemente pipeline"
          + (" (fara video de iesire)..." if args.no_video else "..."))

    # uridecodebin isi creeaza pad-ul abia cand afla ce contine fisierul
    source.connect("pad-added", on_pad_added, vidconv_in)
    vidconv_in.link(caps_in)
    caps_in.link(queue_pre)
    queue_pre.get_static_pad("src").link(streammux.get_request_pad("sink_0"))

    for upstream, downstream in zip(chain, chain[1:]):
        if not upstream.link(downstream):
            raise RuntimeError(
                f"Nu pot lega {upstream.get_name()} de {downstream.get_name()} "
                f"(caps incompatibile?)."
            )

    # Filtrul de casete se pune pe IESIREA detectorului, deci inainte de tracker.
    if PRETRACK_FILTER:
        pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER,
                                             detection_filter_probe, 0)
        print(f"  filtru inainte de tracker: casete cu ambele laturi sub "
              f"{PRETRACK_MIN_SIZE:.0f} px (increderea NU se filtreaza aici)")

    caps_rgba.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, media_probe, 0)
    return pipeline

# ============================================================
# RAPORT FINAL
# ============================================================

def percentile(ordered, fraction):
    """Percentila dintr-o lista DEJA sortata (apelantul sorteaza o singura data)."""
    if not ordered:
        return 0.0
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


def tracks_above_size():
    """Cate track-uri ating fiecare prag de marime, in jurul lui ENROLL_MIN_FACE.

    Cand "marime" e blocajul principal, distributia pe fete nu ajunge: 25% din
    fete peste prag pot fi toate ale aceluiasi om. Ce conteaza e cati OAMENI
    diferiti au avut macar un cadru destul de mare, si asta se vede doar numarand
    track-uri. Masurat pe videotest3.mp4: la prag 50 -> 3 track-uri, la 42 -> 9,
    la 40 -> 13. Aceeasi filmare, acelasi model, alt prag.
    """
    reached = [report.get("marime_maxima", 0) for report in track_reports.values()]
    if not reached:
        return {}
    steps = sorted({max(MIN_FACE_SIZE, ENROLL_MIN_FACE + delta)
                    for delta in (-10, -8, -5, -2, 0, 5, 10)})
    return {str(step): sum(1 for size in reached if size >= step) for step in steps}


def write_summary(run):
    args, output_dir = run.args, run.output_dir
    elapsed = time.time() - run.started
    durations = sorted(run.probe_ms)
    identities = {}
    for track_id, report in track_reports.items():
        label = report["eticheta"] or "fara_decizie"
        identities.setdefault(label, []).append(track_id)

    source = args.source_info
    summary = {
        "video": run.video_path,
        "folder": output_dir,
        "rulare": run.run_index,
        # None = rularea a mers pana la capatul fisierului.
        "eroare": run.failure,
        # Rezolutia de lucru nu mai e fixa, deci fara ea nu se pot citi nici
        # distributiile de mai jos: pragurile sunt in pixeli la rezolutia asta.
        "sursa": {
            "rezolutie": f"{source['width']}x{source['height']}",
            "rotatie_eticheta": source["rotation"],
            "rotatie_aplicata": args.rotation,
            "rezolutie_lucru": f"{args.width}x{args.height}",
            "citit_cu": source["citit_cu"],
            "video_adnotat": None if args.no_video else run.paths["video"],
            "memorie_libera_mb": {"pornire": args.free_memory_mb,
                                  "final": available_memory_mb()[0]},
        },
        "cadre": run.frames,
        "fete_procesate": run.faces_seen,
        "recunoasteri": run.recognitions,
        "durata_rulare_s": r(elapsed, 1),
        # Cat s-a mers, si cat din asta a stat pipeline-ul dupa probe. Cu
        # queue-uri, probe-ul ruleaza in paralel cu restul, deci "probe_din_total"
        # poate trece bine peste 100%: inseamna doar ca probe-ul e acum partea
        # lunga si ca de el trebuie sa te legi mai departe.
        "viteza": {
            "fps": r(run.frames / elapsed, 2) if elapsed > 0 else 0.0,
            "ms_pe_cadru": r(1000.0 * elapsed / run.frames, 2) if run.frames else 0.0,
            "probe_din_total_%": (r(100.0 * sum(run.probe_ms) / (1000.0 * elapsed), 1)
                                  if elapsed > 0 else 0.0),
            "queue_size": args.queue_size,
        },
        # Cate casete a scos detectorul si cate au ajuns la tracker. Raportul
        # dintre ele arata cat de mult ajuta filtrul de dinaintea tracker-ului --
        # pe un cadru larg cu multa lume e diferenta intre 42 si 4 tinte urmarite.
        "detectii": {
            "brute": run.detections,
            "la_tracker": run.detections_kept,
            "aruncate": run.detections - run.detections_kept,
            "filtru_activ": PRETRACK_FILTER,
            "prag_px": r(PRETRACK_MIN_SIZE, 1),
            "filtreaza_increderea": False,   # intentionat: vezi detection_filter_probe
        },
        # Cu ce a fost urmarit. Fara asta nu se poate compara o rulare cu alta
        # cand se umbla la parametrii tracker-ului -- si ei sunt cei care decid
        # daca un om care intoarce capul isi pastreaza id-ul.
        "tracker": {
            "config_sursa": os.path.basename(args.tracker_source),
            "config_folosit": os.path.basename(args.tracker_config),
            "ajustari": args.tracker_overrides,
        },
        # Unde se duc milisecundele din probe, mediu pe cadru. Aici se citeste ce
        # merita optimizat la runda urmatoare, fara sa mai fie nevoie de profiler.
        "timp_etape_ms": {name: r(1000.0 * total / run.frames, 2)
                          for name, total in run.stage_s.most_common()} if run.frames else {},
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
            # Cate track-uri (adica oameni, nu cadre) ar ajunge la ENROLL_MIN_FACE
            # daca pragul ar fi altul. Se citeste cand "marime" e cauza de sus.
            "track_uri_peste_prag_marime": tracks_above_size(),
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
            # Fete vazute pe care NU s-a mai cerut modelul de landmark-uri:
            # track-ul lor avea deja identitate si nu dadea proba in cadrul acela.
            # Raportat la "fete_procesate", asta arata cat a lucrat racirea.
            "fete_doar_urmarite": run.faces_seen - GPU_STATS["landmark_faces"],
        },
        "timp_probe_ms": {
            "mediu": r(sum(durations) / len(durations), 2) if durations else 0.0,
            "p50": r(percentile(durations, 0.50), 2),
            "p95": r(percentile(durations, 0.95), 2),
            "maxim": r(durations[-1], 2) if durations else 0.0,
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
            # Casetele taiate de poarta de marime. Citita impreuna cu "marime_px"
            # de mai sus, spune cat s-ar castiga coborand MIN_FACE_SIZE: daca p90
            # de aici e mult sub prag, nu se castiga nimic, oamenii sunt pur si
            # simplu prea departe de camera.
            "marime_respinse_px": distribution(run.rejected_sizes, 0),
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
            "IDENTIFIED_INTERVAL_FRAMES": IDENTIFIED_INTERVAL_FRAMES,
            "LANDMARK_ONLY_WHEN_NEEDED": LANDMARK_ONLY_WHEN_NEEDED,
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

    path poate fi si o lista de variante, in ordinea preferintei: asa configul de
    tracker cere intai NvDCF_accuracy si cade pe NvDCF_perf daca versiunea
    instalata nu-l are.
    """
    wanted = [path] if isinstance(path, str) else list(path)
    candidates = []
    for option in wanted:
        candidates += [option, os.path.join(SCRIPT_DIR, os.path.basename(option))]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        f"Nu gasesc {what}. Am cautat:\n  " + "\n  ".join(candidates)
    )


# Chei de forma "  nume: valoare   # comentariu", oriunde in fisier. Se inlocuieste
# doar valoarea, deci indentarea, sectiunile si comentariile raman neatinse.
TRACKER_KEY_RE = "^([ \t]*{key}[ \t]*:[ \t]*)([^\\s#]+)(.*)$"


def patch_tracker_config(source, overrides, destination):
    """Scrie o copie a configului de tracker, cu cheile date rescrise.

    Se lucreaza pe linii, cu regex, si NU cu un parser YAML. Doua motive: nu
    depindem de PyYAML, care nu e garantat in containerul DeepStream, si un
    round-trip prin parser ar rescrie tot fisierul -- ar pierde comentariile si ar
    putea reordona chei, iar configurile astea sunt lungi si pline de explicatii
    utile. Aici se schimba exact ce cerem si nimic altceva.

    Intoarce dict cu ce s-a schimbat efectiv. Cheile negasite se raporteaza, nu se
    adauga: daca versiunea instalata numeste altfel un parametru, vrem sa aflam,
    nu sa scriem o cheie pe care tracker-ul o va ignora tacut.
    """
    with open(source, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()

    applied, missing = {}, []
    for key, value in overrides.items():
        if value is None:
            continue
        pattern = re.compile(TRACKER_KEY_RE.format(key=re.escape(key)), re.MULTILINE)
        text, count = pattern.subn(lambda m: f"{m.group(1)}{value}{m.group(3)}", text)
        if count:
            applied[key] = value
        else:
            missing.append(key)

    if missing:
        warn_once("tracker_keys",
                  f"configul de tracker nu are cheile {', '.join(missing)}; "
                  f"raman valorile lui. Verifica numele cu: "
                  f"grep -n 'ShadowTrack\\|minDetectorConfidence' {source}")

    with open(destination, "w", encoding="utf-8") as handle:
        handle.write(text)
    return applied


RUN_INDEX_RE = re.compile(r"^summary_(\d+)\.json$")


def prepare_output_dir(stem, args):
    """Folderul cu numele videoclipului, REFOLOSIT la rulari succesive.

    Asta e ce face posibila secventa "prima rulare inroleaza, a doua recunoaste":
    folderul e memoria rularilor pe videoclipul asta, iar baza de date din el se
    incarca la pornire (vezi resolve_database). Un folder nou pe rulare ar
    insemna ca fiecare rulare porneste iar de la zero identitati.

    Ce se pierde prin refolosire -- suprascrierea rularii anterioare -- se evita
    numerotand fisierele rularii (vezi run_paths), nu folderul.
    """
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
                        help="configul low-level pentru nvtracker; implicit se "
                             "incearca NvDCF_accuracy, apoi NvDCF_perf")
    parser.add_argument("--shadow-track-age", type=int, default=None,
                        help=f"cate cadre traieste un track fara nicio detectie "
                             f"(maxShadowTrackAge). Asta decide daca cineva care "
                             f"intoarce capul isi pastreaza id-ul; masurat pe "
                             f"videotest3.mp4, golurile ajung la 233 de cadre "
                             f"(implicit {TRACKER_OVERRIDES['maxShadowTrackAge']})")
    parser.add_argument("--min-detector-confidence", type=float, default=None,
                        help=f"increderea de la care nvtracker accepta o caseta. "
                             f"Se tine jos intentionat: casetele slabe sunt cele "
                             f"care leaga track-ul peste ocluzii, iar calitatea se "
                             f"filtreaza oricum mai tarziu "
                             f"(implicit {TRACKER_OVERRIDES['minDetectorConfidence']})")
    parser.add_argument("--no-tracker-patch", action="store_true",
                        help="foloseste configul de tracker exact cum e pe disc, "
                             "fara ajustarile de mai sus")
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
    parser.add_argument("--queue-size", type=int, default=4,
                        help="cate cadre pot astepta in fiecare queue; asta decide "
                             "cat se suprapun decodarea/detectia, probe-ul si "
                             "encodarea. Fiecare cadru in asteptare e o suprafata "
                             "NVMM (~8 MB la 1080p RGBA), deci mai mult nu e "
                             "gratis. 1 = practic fara suprapunere")
    parser.add_argument("--no-pretrack-filter", action="store_true",
                        help="trimite la tracker toate casetele detectorului, nu "
                             "doar pe cele care pot trece de porti. Mult mai lent "
                             "pe cadre largi; de folosit doar ca sa se compare")
    parser.add_argument("--log-respinse", action="store_true",
                        help="scrie in log fiecare caseta respinsa, nu doar cate "
                             "au fost pe fiecare poarta (logul creste de ~4 ori)")
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
    tuning.add_argument("--identified-interval", type=int, default=None,
                        help=f"racirea: la cate cadre se re-verifica un track care "
                             f"ARE deja identitate. Intre verificari nu i se mai cer "
                             f"landmark-uri, deci asta e pargia principala de viteza "
                             f"pe un clip cu oameni deja in baza. Mai mare = mai "
                             f"rapid, dar un id schimbat de tracker se prinde mai "
                             f"tarziu (implicit {IDENTIFIED_INTERVAL_FRAMES})")
    tuning.add_argument("--landmark-all", action="store_true",
                        help="cere landmark-uri pe toate fetele, in fiecare cadru, "
                             "ca inainte de racire; de folosit doar ca sa se compare")
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
    global ENROLL_MIN_FRONTALITY, PRETRACK_MIN_SIZE, IDENTIFIED_INTERVAL_FRAMES

    if args.identified_interval is not None:
        IDENTIFIED_INTERVAL_FRAMES = args.identified_interval
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
    # derivate, deci trebuie recalculate dupa pragurile de mai sus
    ENROLL_MAX_SCORE = VERIFY_THRESHOLD - ENROLL_MARGIN
    PRETRACK_MIN_SIZE = MIN_FACE_SIZE * PRETRACK_SIZE_FACTOR


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


MEMORY_ERROR_MARKERS = (
    "queue input batch",            # nvinfer n-a putut trimite batch-ul
    "NVDSINFER_TENSORRT_ERROR",     # de obicei CUDNN_STATUS_INTERNAL_ERROR sub el
    "bufferpool",                   # "failed to activate bufferpool" la nvvideoconvert
    "NvMapMemAllocInternalTagged",  # alocatorul NVMM, "error 12" = ENOMEM
    "Error in allocating buffer",
    "gst-resource-error-quark",
)


def explain_inference_error(text, args):
    """Traduce un esec de alocare in ce se poate face concret.

    Niciunul dintre mesajele de mai sus nu spune nimic singur, iar cauza adevarata
    e aceeasi pentru toate: memoria unificata s-a terminat. "failed to activate
    bufferpool" la conversie-osd si "CUDNN_STATUS_INTERNAL_ERROR" in detector sunt
    acelasi ENOMEM, prins in doua locuri diferite.
    """
    if not any(marker in text for marker in MEMORY_ERROR_MARKERS):
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
    global PRETRACK_FILTER, LOG_REJECTED, LANDMARK_ONLY_WHEN_NEEDED

    args = parse_args()
    AUTO_ENROLL = not args.no_enroll
    LANDMARK_ONLY_WHEN_NEEDED = not args.landmark_all
    DRAW_OVERLAY = not args.no_video
    DRAW_ALL_LANDMARKS = not args.no_landmarks and DRAW_OVERLAY
    PRETRACK_FILTER = not args.no_pretrack_filter
    LOG_REJECTED = args.log_respinse
    args.queue_size = max(1, args.queue_size)
    apply_thresholds(args)

    check_pyds_api()

    video_path = resolve_video(args.video)
    args.pgie_config = resolve_config(args.pgie_config, "configul nvinfer (detectorul de fete)")
    args.tracker_config = resolve_config(args.tracker_config, "configul nvtracker")

    # Gst.init inainte de rotation_supported(): fabricile de elemente au nevoie
    # de el. Sondarea fisierului nu, ea trece prin ffprobe/OpenCV.
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
    output_dir, reused = prepare_output_dir(stem, args)
    run_index, paths = run_paths(output_dir, stem, args)
    database_in, database_out = resolve_database(output_dir, args)
    output_video = paths["video"]

    # Configul de tracker se ajusteaza acum, cand stim unde scriem: copia patchuita
    # ramane in folderul rularii, langa log si summary, deci se vede mai tarziu
    # exact cu ce parametri s-a urmarit.
    args.tracker_overrides = {}
    args.tracker_source = args.tracker_config
    if not args.no_tracker_patch:
        overrides = dict(TRACKER_OVERRIDES)
        if args.shadow_track_age is not None:
            overrides["maxShadowTrackAge"] = args.shadow_track_age
        if args.min_detector_confidence is not None:
            overrides["minDetectorConfidence"] = args.min_detector_confidence
        patched = os.path.join(output_dir, f"tracker_{run_index:03d}.yml")
        args.tracker_overrides = patch_tracker_config(
            args.tracker_config, overrides, patched
        )
        if args.tracker_overrides:
            valori = ", ".join(f"{k}={v}" for k, v in args.tracker_overrides.items())
            print(f"Tracker: {os.path.basename(args.tracker_config)} -> {valori}")
        args.tracker_config = patched

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
        summary = write_summary(RUN)

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
            # Cate persoane s-ar castiga coborand pragul de marime. Fara asta,
            # "marime x453" spune ca pragul taie, dar nu si daca sub el mai e
            # cineva de prins.
            scara = summary["blocaje_inrolare"]["track_uri_peste_prag_marime"]
            if "marime" in blocaje["cauze"] and scara:
                puncte = ", ".join(f"{prag}px -> {count}" for prag, count in scara.items())
                print(f"    track-uri care ating pragul: {puncte} "
                      f"(--enroll-min-face)")
        elif summary["fete_procesate"] == 0:
            print("  (nicio fata nu a trecut de porti: vezi 'respinse' in log)")
        if sarite:
            cauze = ", ".join(f"{name} x{count}" for name, count in sarite.items())
            print(f"  verificari sarite: {cauze}")
        viteza = summary["viteza"]
        print(f"  viteza:            {viteza['fps']} FPS "
              f"({viteza['ms_pe_cadru']} ms/cadru)")
        print(f"  timp probe/cadru:  {summary['timp_probe_ms']['mediu']} ms "
              f"(p95 {summary['timp_probe_ms']['p95']} ms) = "
              f"{viteza['probe_din_total_%']}% din rulare")
        if summary["timp_etape_ms"]:
            etape = ", ".join(f"{name} {value}"
                              for name, value in summary["timp_etape_ms"].items())
            print(f"    din care:        {etape} (ms/cadru)")
        urmarite = summary["apeluri_economisite"]["fete_doar_urmarite"]
        if summary["fete_procesate"]:
            print(f"  landmark-uri:      {summary['gpu']['landmark_faces']} fete din "
                  f"{summary['fete_procesate']} ({urmarite} doar urmarite, "
                  f"{100 * urmarite / summary['fete_procesate']:.0f}% economisit)")
        # Detectorul poate sa fi rulat pe cadre care n-au ajuns niciodata la probe
        # (pipeline cazut intre tracker si conversie), deci "cadre" poate fi 0 cu
        # detectii nenule -- de aici trebuie impartit cu grija.
        detectii = summary["detectii"]
        if detectii["brute"] and summary["cadre"]:
            print(f"  detectii:          {detectii['brute'] / summary['cadre']:.1f}/cadru, "
                  f"la tracker {detectii['la_tracker'] / summary['cadre']:.1f}/cadru")
        elif detectii["brute"]:
            print(f"  detectii:          {detectii['brute']} brute, "
                  f"{detectii['la_tracker']} la tracker (niciun cadru n-a ajuns la probe)")
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
