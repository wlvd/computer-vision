#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recunoastere faciala pe Jetson Xavier: YOLO -> 2d106det -> MobileFaceNet.

Aceeasi logica si acelasi format de iesire ca yolo_2d106.py, dar scris numai
pentru Jetson. Ce s-a schimbat fata de varianta de PC:

    - fara nimic de Windows (fara preincarcare de DLL-uri CUDA, fara MSMF/DSHOW)
    - GStreamer implicit, pentru decodare si encodare pe NVDEC / NVENC
    - TensorRT implicit (--no-trt il opreste), cu motoarele pastrate in cache
    - modelele .onnx sunt luate direct de pe placa, fara sa le mai caute
    - scris pentru Python 3.8 (JetPack 5.x)

PIPELINE, pe fiecare cadru (vezi FacePipeline.process_frame):

    1. DETECTIE     YOLO da casetele fetelor din cadru
    2. TRACKING     casetele primesc un ID stabil intre cadre
    3. LANDMARKS    2d106det da 106 puncte pe fata, din care alegem 5
    4. ALINIERE     fata e rotita/scalata la 112x112 dupa cele 5 puncte
    5. EMBEDDING    MobileFaceNet da un vector de 512 numere
    6. IDENTITATE   vectorul e comparat cu galeria: cine e, sau persoana noua

DEPENDENTE:  numpy, opencv-python (compilat cu GStreamer), onnxruntime-gpu

    JetPack 5.x nu are onnxruntime-gpu pe PyPI pentru aarch64:
        pip install --extra-index-url https://pypi.jetson-ai-lab.dev/jp5/cu114 \
            onnxruntime-gpu

    Verifica totul dintr-o comanda:
        python3 yolo_jetson.py --check

MODELE (langa script; se pot schimba cu --yolo / --landmark / --embedding):
    _detection/best.onnx        detector YOLO, export cu end2end=True
    _landmark/2d106det.onnx     landmark-uri, 106 puncte
    _embedding/w600k_mbf.onnx   MobileFaceNet (InsightFace buffalo_s)

RULARE:
    python3 yolo_jetson.py video.mp4              # decodare pe NVDEC, TensorRT
    python3 yolo_jetson.py video.mp4 --no-trt     # doar CUDA, porneste imediat
    python3 yolo_jetson.py 0                      # camera USB /dev/video0
    python3 yolo_jetson.py 0 --csi                # camera CSI (nvarguscamerasrc)
    python3 yolo_jetson.py rtsp://... --show      # camera IP, cu fereastra
    python3 yolo_jetson.py --list-cameras
    python3 yolo_jetson.py --check

INAINTE DE O MASURATOARE, pune placa la putere maxima:
    sudo nvpmodel -m 0
    sudo jetson_clocks
"""

# ATENTIE: importul de mai jos trebuie sa ramana. Face ca adnotarile de tip sa
# fie doar text, niciodata evaluate, deci putem scrie `int | None` si
# `list[Track]` chiar si pe Python 3.8, care nu le suporta la rulare.
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    sys.exit(
        "EROARE: onnxruntime nu este instalat.\n"
        "  JetPack 5.x:\n"
        "    pip install --extra-index-url "
        "https://pypi.jetson-ai-lab.dev/jp5/cu114 onnxruntime-gpu\n"
        "  Pachetul de pe PyPI standard NU are build de aarch64 cu CUDA."
    )


# ============================================================
# 1. CONFIGURARE
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_VIDEO = "nightsample_3.mp4"
RESULTS_ROOT = "results"

# Modelele sunt deja pe placa: le luam direct, fara lista de candidati.
YOLO_MODEL = SCRIPT_DIR / "_detection" / "best.onnx"
LANDMARK_MODEL = SCRIPT_DIR / "_landmark" / "2d106det.onnx"
EMBEDDING_MODEL = SCRIPT_DIR / "_embedding" / "w600k_mbf.onnx"

# --- detectie ---
YOLO_CONF = 0.25
# Sub acest scor detectia e folosita doar de tracker, nu si pentru
# recunoastere (idee preluata din ByteTrack).
YOLO_LOW_CONF = 0.10

# --- normalizarea intrarii in modele ---
# Valorile pe care le deduce InsightFace inspectand graful ONNX. Le fixam aici
# ca sa nu depindem de pachetul `onnx` pe placa. Verificate pe 2d106det.onnx
# si w600k_mbf.onnx din pachetele buffalo.
LANDMARK_INPUT_MEAN = 0.0
LANDMARK_INPUT_STD = 1.0
EMBEDDING_INPUT_MEAN = 127.5
EMBEDDING_INPUT_STD = 127.5

# --- tracker ---
TRACK_IOU_THRESHOLD = 0.25
TRACK_BUFFER_FRAMES = 30   # cate cadre supravietuieste un track fara detectie
TRACK_MIN_HITS = 2         # cate detectii pana cand track-ul e considerat real

# --- galerie / recunoastere ---
UPDATE_THRESHOLD = 0.60      # peste asta, potrivirea rafineaza galeria
GALLERY_UPDATE_ALPHA = 0.05  # cat de mult trage o potrivire buna centroidul
AUTO_ENROLL = True

MIN_SAMPLES_FOR_DECISION = 3    # cate embedding-uri inainte de prima decizie
MIN_SAMPLES_FOR_ENROLL = 5      # cate inainte de a inregistra o persoana noua
REDECIDE_EVERY_N_SAMPLES = 15   # cat de des se reia decizia pentru un track

MIN_ENROLL_YOLO_CONF = 0.40

# Praguri de calitate: 1.0 = fata mare si clara.
QUALITY_REF_SIZE = 60.0
QUALITY_REF_SHARPNESS = 60.0

CONFLICT_WINDOW_FRAMES = 45   # doua track-uri simultane nu pot fi aceeasi persoana
TRACK_MEMORY_FRAMES = 300     # cat timp tinem minte un track dispărut

SAVE_GALLERY_EVERY_N_FRAMES = 250
PROGRESS_EVERY_N_FRAMES = 50

DRAW_ALL_106_LANDMARKS = False
DRAW_FIVE_ARCFACE_POINTS = True
PREVIEW_MAX_HEIGHT = 720
TXT_PRECISION = 8

EMBEDDING_DIM = 512

# --- GStreamer / NVENC ---
DEFAULT_ENCODE_BITRATE = 8000000   # 8 Mbit/s pentru videoul de iesire
RTSP_LATENCY_MS = 100

# Din cele 106 puncte InsightFace: ochi stang, ochi drept, nas, gura stanga/dreapta.
ARCFACE_5_INDICES_106 = np.array([38, 88, 86, 52, 61], dtype=np.int32)

# Sablonul ArcFace de 112x112 catre care se aliniaza fiecare fata.
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass
class Thresholds:
    """Pragurile care se pot schimba din linia de comanda."""

    conf: float = YOLO_CONF                # peste asta detectia e "sigura"
    match: float = 0.42                    # similaritate minima pentru "aceeasi persoana"
    min_face: int = 12                     # sub asta nici nu incercam landmarks
    min_enroll_face: int = 26              # fata minima pentru a inregistra pe cineva nou
    min_enroll_quality: float = 0.06       # calitate minima pentru inregistrare

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> Thresholds:
        return cls(
            conf=args.conf,
            match=args.match_threshold,
            min_face=args.min_face,
            min_enroll_face=args.min_enroll_face,
            min_enroll_quality=args.min_enroll_quality,
        )


# MIN_FACE_SIZE = 12: masurat pe videotest1/car_sample, coborarea de la 20 la 12
# aduce de 2.8x mai multe detectii recunoscute. Sub 12, 2d106det incepe sa
# respinga masiv (esec landmarks 1153 -> 5902), deci costul creste degeaba.


# ============================================================
# 2. UNELTE MICI
# ============================================================


def resolve_path(value: str) -> Path:
    """Cale absoluta; cele relative se citesc fata de folderul scriptului."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Aduce vectorul la lungime 1, ca sa putem folosi produsul scalar ca similaritate."""

    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))

    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Embedding cu norma zero sau invalida.")

    return vector / norm


def clamp_box(
    box: Sequence[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """Taie caseta la marginile cadrului. None daca ce ramane e gol."""

    x1, y1, x2, y2 = (float(v) for v in box)

    if not all(np.isfinite(v) for v in (x1, y1, x2, y2)):
        return None

    x1_i = int(max(0, min(width - 1, round(x1))))
    y1_i = int(max(0, min(height - 1, round(y1))))
    x2_i = int(max(0, min(width, round(x2))))
    y2_i = int(max(0, min(height, round(y2))))

    if x2_i <= x1_i or y2_i <= y1_i:
        return None

    return x1_i, y1_i, x2_i, y2_i


def draw_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    font_scale: float = 0.45,
) -> None:
    """Text pe fundal colorat, mutat in cadru daca ar iesi afara."""

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1

    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    box_height = text_height + baseline + 6
    box_width = text_width + 8

    frame_height, frame_width = frame.shape[:2]

    left = int(max(0, min(x, frame_width - box_width - 1)))
    top = int(y) - box_height

    if top < 0:                      # nu incape deasupra casetei -> il punem dedesubt
        top = int(y) + 2

    top = int(max(0, min(top, frame_height - box_height - 1)))

    cv2.rectangle(frame, (left, top), (left + box_width, top + box_height), color, -1)
    cv2.putText(
        frame,
        text,
        (left + 4, top + text_height + 3),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def percentiles(values: Sequence[float]) -> str:
    """Rezumat statistic pentru raportul de calibrare."""

    if not values:
        return "n/a"

    array = np.asarray(values, dtype=np.float64)
    p10, p50, p90 = np.percentile(array, [10, 50, 90])

    return (
        f"min={array.min():.3f} p10={p10:.3f} median={p50:.3f} "
        f"p90={p90:.3f} max={array.max():.3f}"
    )


# ============================================================
# 3. GEOMETRIE  (ce foloseam din skimage)
# ============================================================


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Transformarea (rotatie + scalare + translatie) care duce src peste dst.

    Echivalentul lui skimage.transform.SimilarityTransform().estimate().
    Verificat numeric fata de skimage: diferenta maxima ~2.5e-4 px.
    """

    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    num, dim = src.shape

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    covariance = dst_demean.T @ src_demean / num

    signs = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(covariance) < 0:
        signs[dim - 1] = -1.0

    matrix = np.eye(dim + 1, dtype=np.float64)

    u, singular, vt = np.linalg.svd(covariance)
    rank = np.linalg.matrix_rank(covariance)

    if rank == 0:
        return np.full((dim + 1, dim + 1), np.nan)

    if rank == dim - 1:
        if np.linalg.det(u) * np.linalg.det(vt) > 0:
            matrix[:dim, :dim] = u @ vt
        else:
            saved = signs[dim - 1]
            signs[dim - 1] = -1.0
            matrix[:dim, :dim] = u @ np.diag(signs) @ vt
            signs[dim - 1] = saved
    else:
        matrix[:dim, :dim] = u @ np.diag(signs) @ vt

    variance = src_demean.var(axis=0).sum()
    if variance < 1e-12:
        return np.full((dim + 1, dim + 1), np.nan)

    scale = float(singular @ signs) / variance

    matrix[:dim, dim] = dst_mean - scale * (matrix[:dim, :dim] @ src_mean)
    matrix[:dim, :dim] *= scale

    return matrix


def norm_crop(
    image: np.ndarray,
    landmarks: np.ndarray,
    image_size: int = 112,
) -> np.ndarray:
    """PASUL 4: alinierea ArcFace — cele 5 puncte sunt duse peste sablonul fix.

    Rezultatul e o fata dreapta, centrata, la aceeasi scara ca la antrenare.
    Fara asta, embedding-urile aceleiasi persoane nu s-ar mai potrivi.
    """

    ratio = float(image_size) / 112.0
    destination = ARCFACE_TEMPLATE * ratio

    matrix = umeyama_similarity(landmarks, destination)

    if not np.all(np.isfinite(matrix)):
        raise ValueError("Transformare de aliniere degenerata.")

    return cv2.warpAffine(image, matrix[0:2, :], (image_size, image_size), borderValue=0.0)


def landmark_crop_matrix(bbox: Sequence[float], output_size: int) -> np.ndarray:
    """Matricea de decupare pe care o asteapta 2d106det.

    Modelul NU primeste caseta redimensionata, ci un patrat centrat pe caseta,
    cu latura max(w, h) * 1.5 — are nevoie de context in jurul fetei. Reproduce
    insightface.utils.face_align.transform() cu rotatie zero.
    """

    x1, y1, x2, y2 = (float(v) for v in bbox)

    width = x2 - x1
    height = y2 - y1
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    scale = output_size / (max(width, height) * 1.5)
    half = output_size / 2.0

    return np.array(
        [
            [scale, 0.0, half - scale * center_x],
            [0.0, scale, half - scale * center_y],
        ],
        dtype=np.float64,
    )


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Aplica o matrice affine 2x3 unui set de puncte (N, 2)."""

    points = np.asarray(points, dtype=np.float32)
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float32)])
    return (homogeneous @ np.asarray(matrix, dtype=np.float32).T).astype(np.float32)


# ============================================================
# 4. CELE TREI MODELE ONNX
# ============================================================


def build_providers(use_trt: bool, trt_cache: Path, force_cpu: bool) -> list:
    """Lista de backend-uri, in ordinea in care ONNX Runtime le incearca.

    Pe Xavier ordinea conteaza: TensorRT preia ce poate din graf, restul cade pe
    CUDA. Daca un model nu e suportat integral de TRT (de exemplu NMS-ul din
    exportul end2end), ONNX Runtime imparte singur graful — nu e o eroare.
    """

    available = ort.get_available_providers()

    if force_cpu:
        print("  ATENTIE: --cpu pe Xavier inseamna cateva secunde per cadru.")
        return ["CPUExecutionProvider"]

    providers: list = []

    if use_trt:
        if "TensorrtExecutionProvider" not in available:
            print(
                "  ATENTIE: TensorrtExecutionProvider indisponibil.\n"
                "           Wheel-ul de onnxruntime-gpu nu e cel de la\n"
                "           pypi.jetson-ai-lab.dev? Se continua doar cu CUDA."
            )
        else:
            trt_cache.mkdir(parents=True, exist_ok=True)
            providers.append(
                (
                    "TensorrtExecutionProvider",
                    {
                        "device_id": 0,
                        # Xavier are unitati fp16 rapide: aici e cel mai mare
                        # castig, fara pierdere vizibila de precizie.
                        "trt_fp16_enable": True,
                        # Fara cache, fiecare pornire recompileaza motoarele.
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": str(trt_cache),
                        "trt_timing_cache_enable": True,
                    },
                )
            )

    if "CUDAExecutionProvider" in available:
        providers.append(("CUDAExecutionProvider", {"device_id": 0}))
    else:
        print(
            "  ATENTIE: CUDAExecutionProvider indisponibil.\n"
            "           Ai instalat onnxruntime in loc de onnxruntime-gpu?\n"
            "           Pe Jetson foloseste wheel-ul de la pypi.jetson-ai-lab.dev."
        )

    providers.append("CPUExecutionProvider")   # ultima plasa de siguranta
    return providers


def make_session(model_path: Path, providers: list) -> ort.InferenceSession:
    """Deschide un model .onnx si spune pe ce backend a ajuns si in cat timp."""

    if not model_path.is_file():
        raise FileNotFoundError(f"Modelul nu exista pe placa: {model_path}")

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3

    started = time.perf_counter()
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=providers
    )
    elapsed = time.perf_counter() - started

    # O pornire lunga inseamna aproape sigur ca TensorRT a compilat un motor.
    note = "  (compilare TensorRT)" if elapsed > 20.0 else ""
    print(f"  {model_path.name:20s} -> {session.get_providers()[0]:26s} "
          f"{elapsed:5.1f}s{note}")

    return session


class FaceDetector:
    """PASUL 1: YOLO exportat cu end2end=True — NMS e deja in graful ONNX.

    Adica modelul intoarce direct casetele finale, (1, N, 6), unde fiecare rand
    e [x1, y1, x2, y2, scor, clasa]. Nu mai decodam grile si nu mai facem NMS.
    """

    def __init__(self, model_path: Path, providers: list):
        self.session = make_session(model_path, providers)

        input_cfg = self.session.get_inputs()[0]
        self.input_name = input_cfg.name
        self.output_names = [o.name for o in self.session.get_outputs()]

        shape = input_cfg.shape
        height = shape[2] if isinstance(shape[2], int) else 640
        width = shape[3] if isinstance(shape[3], int) else 640
        self.input_size = (int(width), int(height))

        output_shape = self.session.get_outputs()[0].shape
        if not (len(output_shape) == 3 and output_shape[-1] == 6):
            raise RuntimeError(
                f"Modelul {model_path.name} are output {output_shape}, dar acest "
                "script asteapta un export cu end2end=True, adica (1, N, 6).\n"
                "Reexporta din ultralytics:  model.export(format='onnx', nms=True)"
            )

    def letterbox(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Redimensioneaza pastrand proportiile, pe o panza gri.

        Intoarce si factorul de scara si marginile, ca sa putem duce casetele
        inapoi in coordonatele cadrului original.
        """

        target_width, target_height = self.input_size
        height, width = image.shape[:2]

        ratio = min(target_width / width, target_height / height)
        new_width = int(round(width * ratio))
        new_height = int(round(height * ratio))

        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
        left = (target_width - new_width) // 2
        top = (target_height - new_height) // 2
        canvas[top:top + new_height, left:left + new_width] = resized

        return canvas, ratio, left, top

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Casetele fetelor din cadru: boxes (N, 4) in pixeli, scores (N,)."""

        canvas, ratio, pad_x, pad_y = self.letterbox(image)

        # BGR -> RGB, HWC -> CHW, adaugam dimensiunea de batch, scalam la [0, 1].
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None]
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0

        detections = self.session.run(self.output_names, {self.input_name: blob})[0][0]

        detections = detections[detections[:, 4] >= conf_threshold]

        if detections.shape[0] == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        # Din coordonate de letterbox inapoi in coordonatele cadrului.
        boxes = detections[:, :4].copy()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio

        return boxes.astype(np.float32), detections[:, 4].astype(np.float32)


class LandmarkModel:
    """PASUL 3: 2d106det — 106 puncte pe fata, dintr-un crop de 192x192."""

    def __init__(self, model_path: Path, providers: list):
        self.session = make_session(model_path, providers)

        input_cfg = self.session.get_inputs()[0]
        self.input_name = input_cfg.name
        self.input_size = int(input_cfg.shape[2])
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.num_points = int(self.session.get_outputs()[0].shape[1]) // 2
        if self.num_points < 106:
            raise RuntimeError(
                f"{model_path.name} produce {self.num_points} puncte, nu 106."
            )

    def get(self, image: np.ndarray, bbox: Sequence[float]) -> np.ndarray:
        """Punctele pentru caseta data, in coordonatele cadrului original."""

        # Decupam direct din cadru catre 192x192, cu o singura transformare.
        matrix = landmark_crop_matrix(bbox, self.input_size)
        cropped = cv2.warpAffine(
            image, matrix, (self.input_size, self.input_size), borderValue=0.0
        )

        blob = cv2.dnn.blobFromImage(
            cropped,
            1.0 / LANDMARK_INPUT_STD,
            (self.input_size, self.input_size),
            (LANDMARK_INPUT_MEAN,) * 3,
            swapRB=True,
        )

        prediction = self.session.run(self.output_names, {self.input_name: blob})[0][0]

        points = prediction.reshape((-1, 2))
        if self.num_points < points.shape[0]:
            points = points[-self.num_points:, :]

        # Modelul da coordonate in [-1, 1] fata de centrul crop-ului:
        # le aducem in pixeli de crop, apoi in pixeli de cadru.
        points = (points + 1.0) * (self.input_size // 2)

        return transform_points(points, cv2.invertAffineTransform(matrix))


class FaceEncoder:
    """PASUL 5: MobileFaceNet — o fata aliniata 112x112 devine 512 numere."""

    def __init__(self, model_path: Path, providers: list):
        self.session = make_session(model_path, providers)

        input_cfg = self.session.get_inputs()[0]
        self.input_name = input_cfg.name
        self.input_size = int(input_cfg.shape[2])
        self.output_names = [o.name for o in self.session.get_outputs()]

    def get_feat(self, aligned: np.ndarray) -> np.ndarray:
        blob = cv2.dnn.blobFromImage(
            aligned,
            1.0 / EMBEDDING_INPUT_STD,
            (self.input_size, self.input_size),
            (EMBEDDING_INPUT_MEAN,) * 3,
            swapRB=True,
        )
        return self.session.run(self.output_names, {self.input_name: blob})[0]


@dataclass
class Models:
    """Cele trei modele, incarcate o singura data."""

    detector: FaceDetector
    landmark: LandmarkModel
    encoder: FaceEncoder


def load_models(args: argparse.Namespace) -> Models:
    """Deschide cele trei .onnx de pe placa si spune pe ce ruleaza fiecare."""

    print("=" * 62)
    print("MODELE")
    print("=" * 62)

    yolo_path = resolve_path(args.yolo) if args.yolo else YOLO_MODEL
    landmark_path = resolve_path(args.landmark) if args.landmark else LANDMARK_MODEL
    embedding_path = resolve_path(args.embedding) if args.embedding else EMBEDDING_MODEL

    for label, path in (
        ("YOLO", yolo_path),
        ("2d106det", landmark_path),
        ("embedding", embedding_path),
    ):
        print(f"  {label:12s}: {path}")

    use_trt = not args.no_trt
    trt_cache = resolve_path(args.trt_cache)

    if use_trt:
        cached = list(trt_cache.glob("*.engine")) if trt_cache.is_dir() else []
        if cached:
            print(f"  TensorRT: {len(cached)} motoare gasite in {trt_cache}")
        else:
            print(
                f"  TensorRT: cache gol ({trt_cache}).\n"
                "            Prima rulare compileaza motoarele — pe Xavier poate\n"
                "            dura 5-15 minute. Rularile urmatoare pornesc imediat."
            )

    providers = build_providers(use_trt, trt_cache, args.cpu)

    models = Models(
        detector=FaceDetector(yolo_path, providers),
        landmark=LandmarkModel(landmark_path, providers),
        encoder=FaceEncoder(embedding_path, providers),
    )

    print(f"  intrare YOLO  : "
          f"{models.detector.input_size[0]}x{models.detector.input_size[1]}")

    # ONNX Runtime listeaza TensorRT ca disponibil chiar si cand bibliotecile
    # lipsesc sau au alta versiune. In cazul ala arunca un "EP Error" lung si
    # cade singur pe CUDA — deci verificam ce s-a intamplat de fapt.
    sessions = (models.detector.session, models.landmark.session, models.encoder.session)
    if use_trt and not any(
        "TensorrtExecutionProvider" in s.get_providers() for s in sessions
    ):
        print(
            "\n  ATENTIE: TensorRT a fost cerut, dar niciun model nu ruleaza pe el\n"
            "           (vezi 'EP Error' mai sus). Merge pe CUDA, doar mai incet.\n"
            "           Pe Jetson asta inseamna de obicei ca versiunea de\n"
            "           onnxruntime-gpu nu se potriveste cu TensorRT din JetPack."
        )

    on_cpu = models.detector.session.get_providers()[0] == "CPUExecutionProvider"
    if on_cpu and not args.cpu:
        print(
            "\n  ATENTIE: totul ruleaza pe CPU. Pe Xavier asta e inutilizabil.\n"
            "           Verifica instalarea onnxruntime-gpu."
        )
    print()

    return models


# ============================================================
# 5. TRACKER  (PASUL 2)
# ============================================================


@dataclass
class Track:
    """O fata urmarita in timp: unde e, cat de sigur, cum se misca."""

    track_id: int
    box: np.ndarray
    score: float
    hits: int = 1                 # de cate ori a fost confirmat de o detectie
    age: int = 0
    time_since_update: int = 0    # de cate cadre nu a mai fost vazut
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))

    def predict(self) -> np.ndarray:
        """Unde ar trebui sa fie in cadrul urmator, daca isi tine viteza."""

        return self.box + self.velocity

    def update(self, box: np.ndarray, score: float) -> None:
        # Viteza e o medie intre cea veche si deplasarea noua (netezire simpla).
        self.velocity = 0.5 * self.velocity + 0.5 * (box - self.box)
        self.box = box.astype(np.float32)
        self.score = score
        self.hits += 1
        self.time_since_update = 0


def iou_matrix(tracks: np.ndarray, detections: np.ndarray) -> np.ndarray:
    """Cat se suprapun toate track-urile cu toate detectiile: matrice (T, D)."""

    if tracks.size == 0 or detections.size == 0:
        return np.zeros((tracks.shape[0], detections.shape[0]), dtype=np.float32)

    # Dreptunghiul de intersectie pentru fiecare pereche.
    tl_x = np.maximum(tracks[:, None, 0], detections[None, :, 0])
    tl_y = np.maximum(tracks[:, None, 1], detections[None, :, 1])
    br_x = np.minimum(tracks[:, None, 2], detections[None, :, 2])
    br_y = np.minimum(tracks[:, None, 3], detections[None, :, 3])

    inter = np.clip(br_x - tl_x, 0, None) * np.clip(br_y - tl_y, 0, None)

    area_t = (tracks[:, 2] - tracks[:, 0]) * (tracks[:, 3] - tracks[:, 1])
    area_d = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])

    union = area_t[:, None] + area_d[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def greedy_match(
    scores: np.ndarray,
    threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Potriveste rand cu coloana, luand mereu cel mai bun scor ramas.

    Alternativa la algoritmul Hungarian din scipy: mai simpla, aproape la fel
    de buna cand casetele nu se suprapun mult. Si scapam de scipy pe placa.
    Intoarce: potrivirile, randurile nepotrivite, coloanele nepotrivite.
    """

    matches: list[tuple[int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    if scores.size:
        # Toate perechile, sortate descrescator dupa scor.
        rows, cols = np.unravel_index(np.argsort(-scores, axis=None), scores.shape)

        for row, col in zip(rows, cols):
            row, col = int(row), int(col)

            if scores[row, col] < threshold:
                break                       # de aici in jos, nimic nu mai e valid
            if row in used_rows or col in used_cols:
                continue                    # deja luate de o potrivire mai buna

            matches.append((row, col))
            used_rows.add(row)
            used_cols.add(col)

    unmatched_rows = [r for r in range(scores.shape[0]) if r not in used_rows]
    unmatched_cols = [c for c in range(scores.shape[1]) if c not in used_cols]

    return matches, unmatched_rows, unmatched_cols


class IouTracker:
    """Tracker prin IoU, cu asociere in doua etape (in stil ByteTrack).

    Aici nu avem nici BoT-SORT din ultralytics, nici nvtracker din DeepStream
    (asta e varianta pe ONNX Runtime), deci tracker-ul e scris in fisier.
    Ideea din ByteTrack: detectiile cu scor mare creeaza track-uri noi, cele cu
    scor mic doar sustin track-uri existente. Ajuta mult cand fata se intuneca
    sau se misca repede.
    """

    def __init__(
        self,
        high_threshold: float,
        low_threshold: float,
        iou_threshold: float = TRACK_IOU_THRESHOLD,
        buffer_frames: int = TRACK_BUFFER_FRAMES,
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.iou_threshold = iou_threshold
        self.buffer_frames = buffer_frames
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
    ) -> list[tuple[int, np.ndarray, float]]:
        """Da detectiile cadrului, primeste (track_id, box, score) pentru cele vizibile."""

        for track in self.tracks:
            track.age += 1
            track.time_since_update += 1

        is_high = scores >= self.high_threshold
        is_low = (~is_high) & (scores >= self.low_threshold)

        predicted = (
            np.stack([t.predict() for t in self.tracks])
            if self.tracks
            else np.empty((0, 4), dtype=np.float32)
        )

        # --- Etapa 1: detectiile sigure, cu toate track-urile ---
        high_indices = np.flatnonzero(is_high)
        matches, unmatched_tracks, unmatched_high = greedy_match(
            iou_matrix(predicted, boxes[is_high]), self.iou_threshold
        )

        for track_index, det_index in matches:
            index = high_indices[det_index]
            self.tracks[track_index].update(boxes[index], float(scores[index]))

        # --- Etapa 2: detectiile slabe, doar pentru track-urile ramase ---
        low_indices = np.flatnonzero(is_low)
        if unmatched_tracks and low_indices.size:
            remaining = np.stack([self.tracks[i].predict() for i in unmatched_tracks])
            second_matches, still_unmatched, _ = greedy_match(
                iou_matrix(remaining, boxes[low_indices]), self.iou_threshold
            )
            for local_index, det_index in second_matches:
                track_index = unmatched_tracks[local_index]
                index = low_indices[det_index]
                self.tracks[track_index].update(boxes[index], float(scores[index]))

            unmatched_tracks = [unmatched_tracks[i] for i in still_unmatched]

        # --- Track-uri noi: doar din detectii sigure ---
        for det_index in unmatched_high:
            index = high_indices[det_index]
            self.tracks.append(
                Track(
                    track_id=self._next_id,
                    box=boxes[index].astype(np.float32),
                    score=float(scores[index]),
                )
            )
            self._next_id += 1

        # Uitam track-urile nevazute prea mult timp.
        self.tracks = [
            t for t in self.tracks if t.time_since_update <= self.buffer_frames
        ]

        # Raportam doar ce am vazut in cadrul asta, si doar track-urile
        # confirmate (sau cele cu scor mare, care merita crezute imediat).
        return [
            (t.track_id, t.box, t.score)
            for t in self.tracks
            if t.time_since_update == 0
            and (t.hits >= TRACK_MIN_HITS or t.score >= self.high_threshold)
        ]


# ============================================================
# 6. DIN 106 PUNCTE IN 5, SI VERIFICAREA LOR
# ============================================================


def landmarks_106_to_five(landmarks: np.ndarray) -> np.ndarray:
    """Alege cele 5 puncte de care are nevoie alinierea ArcFace."""

    points = np.asarray(landmarks, dtype=np.float32)

    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"Forma landmarks invalida: {points.shape}")
    if points.shape[0] < 106:
        raise ValueError(f"Modelul a intors {points.shape[0]} puncte, nu 106.")

    five = points[ARCFACE_5_INDICES_106, :2].copy()

    # Sablonul asteapta stanga inainte de dreapta; la fete rotite se pot inversa.
    if five[0, 0] > five[1, 0]:
        five[[0, 1]] = five[[1, 0]]      # ochii
    if five[3, 0] > five[4, 0]:
        five[[3, 4]] = five[[4, 3]]      # colturile gurii

    return np.ascontiguousarray(five, dtype=np.float32)


def check_five_landmarks(five: np.ndarray, box: Sequence[int]) -> str | None:
    """Are sens fata asta? Intoarce motivul respingerii, sau None daca e buna.

    Pe fete mici sau intoarse, 2d106det da uneori puncte imposibile. Daca le
    aliniem oricum, iese un embedding fals care strica galeria. E mai bine sa
    aruncam detectia. Motivele ajung in raportul final.
    """

    if five.shape != (5, 2) or not np.all(np.isfinite(five)):
        return "puncte invalide"

    x1, y1, x2, y2 = (float(v) for v in box)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)

    left_eye, right_eye, nose, mouth_left, mouth_right = five

    eye_distance = float(np.linalg.norm(right_eye - left_eye))
    mouth_distance = float(np.linalg.norm(mouth_right - mouth_left))
    eye_y = float((left_eye[1] + right_eye[1]) / 2.0)
    mouth_y = float((mouth_left[1] + mouth_right[1]) / 2.0)

    if eye_distance < max(4.0, width * 0.07):
        return "ochi prea apropiati"
    if mouth_distance < max(3.0, width * 0.06):
        return "gura degenerata"
    if mouth_y <= eye_y:
        return "gura peste ochi"
    if nose[1] < eye_y - 0.20 * height:
        return "nas deasupra ochilor"
    if nose[1] > mouth_y + 0.25 * height:
        return "nas sub gura"

    # Punctele pot iesi puțin din caseta, dar nu oricat.
    margin_x = width * 0.45
    margin_y = height * 0.45

    if np.any(five[:, 0] < x1 - margin_x) or np.any(five[:, 0] > x2 + margin_x):
        return "puncte in afara casetei (x)"
    if np.any(five[:, 1] < y1 - margin_y) or np.any(five[:, 1] > y2 + margin_y):
        return "puncte in afara casetei (y)"

    return None


def face_quality(aligned: np.ndarray, face_size: float, conf: float) -> float:
    """Cat de mult merita crezut embedding-ul acestei fete, intre 0 si 1.

    Produsul a patru lucruri: cat e de mare, cat e de clara, daca nu e prea
    intunecata/arsa, si cat de sigur a fost detectorul.
    """

    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())

    size_score = min(1.0, face_size / QUALITY_REF_SIZE)
    sharp_score = min(1.0, sharpness / QUALITY_REF_SHARPNESS)
    bright_score = 0.25 if (brightness < 18.0 or brightness > 240.0) else 1.0

    return float(size_score * sharp_score * bright_score * conf)


# ============================================================
# 7. GALERIA DE IDENTITATI
# ============================================================


class Gallery:
    """Un embedding "mediu" per persoana, salvat pe disc in .npz.

    Toate embedding-urile au lungime 1, deci similaritatea dintre doua fete e
    pur si simplu produsul lor scalar (cosinus): 1.0 = identice, 0.0 = fara
    legatura.
    """

    def __init__(self, path: Path):
        self.path = path
        self.ids: list[int] = []
        self.embeddings: list[np.ndarray] = []
        self.counts: list[int] = []          # din cate potriviri s-a format fiecare
        self.next_person_id = 1
        self._matrix: np.ndarray | None = None   # cache pentru comparatii

    # --- disc ---

    def load(self) -> None:
        if not self.path.exists():
            print(f"  Galerie noua (nu exista {self.path.name}).")
            return

        try:
            data = np.load(self.path, allow_pickle=False)
        except Exception as error:
            raise RuntimeError(f"Nu am putut citi galeria {self.path}: {error}") from error

        for key in ("ids", "embeddings"):
            if key not in data.files:
                raise RuntimeError(f"Galeria nu contine campul '{key}'.")

        ids = np.asarray(data["ids"], dtype=np.int64).reshape(-1)
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)

        if embeddings.ndim != 2:
            raise RuntimeError(f"Galerie invalida: embeddings au forma {embeddings.shape}.")
        if ids.shape[0] != embeddings.shape[0]:
            raise RuntimeError("Numar de ID-uri diferit de numarul de embeddings.")

        if "counts" in data.files:
            counts = np.asarray(data["counts"], dtype=np.int64).reshape(-1)
            if counts.shape[0] != ids.shape[0]:
                counts = np.ones_like(ids)
        else:
            counts = np.ones_like(ids)

        self.ids = [int(v) for v in ids]
        self.counts = [int(v) for v in counts]
        self.embeddings = []

        for index, embedding in enumerate(embeddings):
            try:
                self.embeddings.append(l2_normalize(embedding))
            except ValueError:
                print(f"  Avertisment: embedding {index} din galerie e degenerat.")
                self.embeddings.append(np.zeros(embeddings.shape[1], dtype=np.float32))

        self.next_person_id = max(self.ids, default=0) + 1
        self._matrix = None

        print(
            f"  Galerie incarcata: {len(self.ids)} identitati, "
            f"urmatorul ID={self.next_person_id}"
        )

    def save(self) -> None:
        """Scrie intr-un fisier temporar, apoi il mutam — ca sa nu pierdem tot
        daca placa se reseteaza in timpul scrierii."""

        self.path.parent.mkdir(parents=True, exist_ok=True)

        matrix = (
            np.stack(self.embeddings, axis=0).astype(np.float32)
            if self.embeddings
            else np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        )

        temp_path = self.path.with_name(self.path.name + ".tmp")

        # savez_compressed adauga ".npz" la un nume de fisier, deci ii dam un
        # obiect fisier deja deschis.
        with temp_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                ids=np.asarray(self.ids, dtype=np.int64),
                embeddings=matrix,
                counts=np.asarray(self.counts, dtype=np.int64),
            )

        os.replace(str(temp_path), str(self.path))

    # --- comparare si actualizare ---

    @property
    def matrix(self) -> np.ndarray | None:
        """Toate embedding-urile ca o matrice (P, 512), refacuta doar la nevoie."""

        if not self.embeddings:
            return None
        if self._matrix is None:
            self._matrix = np.stack(self.embeddings, axis=0).astype(np.float32)
        return self._matrix

    def similarities(self, embedding: np.ndarray) -> np.ndarray | None:
        """Similaritatea cu fiecare persoana din galerie, dintr-o inmultire."""

        matrix = self.matrix
        if matrix is None or matrix.shape[1] != embedding.shape[0]:
            return None
        return matrix @ embedding

    def add(self, embedding: np.ndarray) -> int:
        """Persoana noua. Intoarce ID-ul primit."""

        person_id = self.next_person_id
        self.next_person_id += 1

        self.ids.append(person_id)
        self.embeddings.append(embedding.astype(np.float32).copy())
        self.counts.append(1)
        self._matrix = None

        return person_id

    def update(self, index: int, embedding: np.ndarray) -> None:
        """Trage usor embedding-ul salvat spre cel nou (medie exponentiala).

        Alpha mic inseamna ca o singura potrivire buna schimba putin: galeria se
        adapteaza la lumina si unghi, dar nu deraiaza dintr-un cadru nefericit.
        """

        blended = (
            (1.0 - GALLERY_UPDATE_ALPHA) * self.embeddings[index]
            + GALLERY_UPDATE_ALPHA * embedding
        )
        try:
            self.embeddings[index] = l2_normalize(blended)
        except ValueError:
            return

        self.counts[index] += 1
        self._matrix = None


# ============================================================
# 8. IDENTITATEA UNUI TRACK  (PASUL 6)
# ============================================================


@dataclass
class TrackState:
    """Ce am adunat despre un track: media embedding-urilor si cine credem ca e.

    Decizia NU se ia dintr-un singur cadru. Adunam embedding-uri ponderate cu
    calitatea, si abia dupa cateva incercam sa spunem cine e.
    """

    track_id: int
    embedding_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(EMBEDDING_DIM, dtype=np.float64)
    )
    sample_count: int = 0
    person_id: int | None = None
    similarity: float = -1.0
    is_new_identity: bool = False
    best_quality: float = 0.0
    last_frame: int = -1
    samples_at_last_decision: int = 0

    def add_sample(self, embedding: np.ndarray, quality: float) -> None:
        if self.embedding_sum.shape[0] != embedding.shape[0]:
            self.embedding_sum = np.zeros(embedding.shape[0], dtype=np.float64)
            self.sample_count = 0

        self.embedding_sum += embedding.astype(np.float64) * max(quality, 1e-3)
        self.sample_count += 1
        self.best_quality = max(self.best_quality, quality)

    def mean_embedding(self) -> np.ndarray:
        return l2_normalize(self.embedding_sum.astype(np.float32))

    def needs_decision(self) -> bool:
        """Prima decizie dupa 3 mostre; apoi o reluam la fiecare 15 mostre noi."""

        if self.sample_count < MIN_SAMPLES_FOR_DECISION:
            return False
        if self.person_id is None:
            return True
        return self.sample_count - self.samples_at_last_decision >= REDECIDE_EVERY_N_SAMPLES


def prune_tracks(tracks: dict[int, TrackState], frame_index: int) -> None:
    """Sterge track-urile nevazute de mult, ca sa nu creasca memoria la infinit.

    Pe Xavier memoria e partajata cu GPU-ul, deci chiar conteaza.
    """

    horizon = max(CONFLICT_WINDOW_FRAMES, TRACK_MEMORY_FRAMES)
    stale = [
        track_id
        for track_id, state in tracks.items()
        if frame_index - state.last_frame > horizon
    ]
    for track_id in stale:
        del tracks[track_id]


def person_ids_in_use(
    tracks: dict[int, TrackState],
    current_track_id: int,
    frame_index: int,
) -> set[int]:
    """ID-urile ocupate chiar acum de alte track-uri.

    Doua fete vizibile in acelasi timp nu pot fi aceeasi persoana, deci le
    interzicem sa primeasca acelasi ID.
    """

    in_use: set[int] = set()

    for track_id, state in tracks.items():
        if track_id == current_track_id or state.person_id is None:
            continue
        if frame_index - state.last_frame <= CONFLICT_WINDOW_FRAMES:
            in_use.add(state.person_id)

    return in_use


def decide_identity(
    state: TrackState,
    gallery: Gallery,
    blocked_ids: set[int],
    allow_enroll: bool,
    thresholds: Thresholds,
) -> None:
    """Cine e track-ul asta? Modifica state.person_id / similarity pe loc.

    Trei rezultate posibile:
      - seamana destul cu cineva din galerie   -> recunoscut
      - nu seamana, dar avem date bune         -> persoana noua, inregistrata
      - nu seamana si datele-s slabe           -> lasam in asteptare
    """

    try:
        mean = state.mean_embedding()
    except ValueError:
        return

    state.samples_at_last_decision = state.sample_count

    # Cel mai bun candidat care nu e deja folosit de alt track vizibil.
    similarities = gallery.similarities(mean)
    best_index: int | None = None
    best_similarity = -1.0

    if similarities is not None and similarities.size:
        for index in np.argsort(-similarities):
            index = int(index)
            candidate_id = gallery.ids[index]
            if candidate_id in blocked_ids and candidate_id != state.person_id:
                continue
            best_index = index
            best_similarity = float(similarities[index])
            break

    state.similarity = best_similarity

    # 1. Recunoscut.
    if best_index is not None and best_similarity >= thresholds.match:
        state.person_id = gallery.ids[best_index]
        state.is_new_identity = False
        if best_similarity >= UPDATE_THRESHOLD:
            gallery.update(best_index, mean)
        return

    # 2. Avea deja un ID si acum nu se mai potriveste: il pastram, nu ne
    #    razgandim dintr-o serie slaba de cadre.
    if state.person_id is not None:
        return

    # 3. Persoana noua — dar numai daca merita inregistrata.
    if not (AUTO_ENROLL and allow_enroll):
        return
    if state.sample_count < MIN_SAMPLES_FOR_ENROLL:
        return
    if state.best_quality < thresholds.min_enroll_quality:
        return

    state.person_id = gallery.add(mean)
    state.is_new_identity = True


# ============================================================
# 9. VIDEO IN / OUT PRIN GSTREAMER
# ============================================================
#
# Pe Xavier videoul nu se decodeaza pe CPU. Cele opt nuclee Carmel se termina
# repede daca le pui sa despacheteze H.264, si atunci degeaba ai GPU liber.
# Toate pipeline-urile de mai jos trec prin aceleasi trei elemente:
#
#   nvv4l2decoder   decodare pe NVDEC (blocul hardware dedicat)
#   nvvidconv       conversie de format pe hardware, scoate din memoria NVMM
#   appsink         de aici citeste OpenCV, cadre BGR
#
# OpenCV are nevoie de cadre BGR in memoria de sistem, deci ultimul
# videoconvert ramane pe CPU — dar doar converteste BGRx -> BGR, nu decodeaza.


def has_gstreamer_support() -> bool:
    """OpenCV-ul de pe placa e compilat cu GStreamer?

    Cel din JetPack (apt install python3-opencv, sau build-ul din SDK Manager)
    este. Cel instalat cu `pip install opencv-python` de obicei NU este — si
    atunci toate pipeline-urile de aici cad, fara un mesaj clar.
    """

    for line in cv2.getBuildInformation().splitlines():
        if "GStreamer" in line:
            return "YES" in line.upper()
    return False


def gst_file_pipelines(path: str, flip: int) -> list[str]:
    """Variante de decodare pentru un fisier, in ordinea in care le incercam."""

    tail = (
        f"nvvidconv flip-method={flip} ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=false sync=false max-buffers=2"
    )

    # Calea se pune in ghilimele: altfel un director cu spatiu in nume rupe
    # pipeline-ul, iar mesajul de eroare nu spune de ce.
    location = f'location="{path}"'

    return [
        # decodebin alege singur decodorul; pe Jetson nvv4l2decoder are rank
        # mare, deci il ia el. Merge pentru mp4, mkv, ts, H.264 sau H.265.
        f"filesrc {location} ! decodebin ! {tail}",
        # Daca negocierea automata da peste cap: mp4 + H.264 explicit.
        f"filesrc {location} ! qtdemux ! h264parse ! nvv4l2decoder ! {tail}",
        # Acelasi lucru pentru H.265.
        f"filesrc {location} ! qtdemux ! h265parse ! nvv4l2decoder ! {tail}",
    ]


def gst_usb_pipelines(
    device: int,
    width: int,
    height: int,
    fps: float,
    flip: int,
) -> list[str]:
    """Camera USB. Intai MJPEG (decodat pe NVDEC), apoi format brut."""

    tail = (
        f"nvvidconv flip-method={flip} ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true sync=false max-buffers=1"
    )

    width = width or 1280
    height = height or 720
    framerate = int(fps) or 30

    return [
        # MJPEG: singurul mod in care majoritatea camerelor USB dau rezolutie
        # mare la 30 fps fara sa sufoce magistrala USB.
        f"v4l2src device=/dev/video{device} io-mode=2 ! "
        f"image/jpeg,width={width},height={height},framerate={framerate}/1 ! "
        f"nvv4l2decoder mjpeg=1 ! {tail}",
        # Brut: sigur, dar limitat de banda USB.
        f"v4l2src device=/dev/video{device} ! "
        f"video/x-raw,width={width},height={height},framerate={framerate}/1 ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true sync=false max-buffers=1",
    ]


def gst_csi_pipelines(
    sensor_id: int,
    width: int,
    height: int,
    fps: float,
    flip: int,
) -> list[str]:
    """Camera CSI (conectorul de pe placa), prin ISP-ul Argus."""

    width = width or 1920
    height = height or 1080
    framerate = int(fps) or 30

    return [
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},"
        f"framerate={framerate}/1,format=NV12 ! "
        f"nvvidconv flip-method={flip} ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true sync=false max-buffers=1",
    ]


def gst_network_pipelines(url: str, flip: int) -> list[str]:
    """Camera IP / RTSP, cu decodare pe NVDEC."""

    tail = (
        f"nvvidconv flip-method={flip} ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true sync=false max-buffers=1"
    )

    return [
        # uridecodebin se descurca si cu H.264 si cu H.265.
        f"uridecodebin uri={url} source::latency={RTSP_LATENCY_MS} ! {tail}",
        # Varianta explicita, cand negocierea automata nu merge.
        f"rtspsrc location={url} latency={RTSP_LATENCY_MS} protocols=udp ! "
        f"rtph264depay ! h264parse ! nvv4l2decoder ! {tail}",
    ]


def gst_encode_pipeline(path: Path, bitrate: int, codec: str) -> str:
    """Encodare pe NVENC. La 1080p, pe CPU nu ai cum sa tii pasul."""

    encoder = "nvv4l2h265enc" if codec == "h265" else "nvv4l2h264enc"
    parser = "h265parse" if codec == "h265" else "h264parse"

    return (
        "appsrc ! video/x-raw,format=BGR ! videoconvert ! "
        "video/x-raw,format=I420 ! nvvidconv ! "
        "video/x-raw(memory:NVMM),format=NV12 ! "
        f"{encoder} bitrate={bitrate} insert-sps-pps=1 ! {parser} ! qtmux ! "
        f'filesink location="{path}"'   # ghilimele: calea poate avea spatii
    )


def try_pipelines(pipelines: Sequence[str], label: str):
    """Incearca pipeline-urile pe rand si intoarce primul care se deschide."""

    for index, pipeline in enumerate(pipelines):
        capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if capture.isOpened():
            suffix = "" if index == 0 else f" (varianta {index + 1})"
            return capture, f"GStreamer / {label}{suffix}"
        capture.release()

    return None, ""


def is_live_source(source: str) -> bool:
    """Camera sau flux de retea — surse care nu pot fi derulate."""

    return source.isdigit() or source.split("://")[0] in (
        "rtsp", "rtmp", "http", "https", "udp", "tcp"
    )


def probe_video_file(path: str) -> tuple[float, int]:
    """Citeste fps si numarul de cadre cu ffmpeg, inainte de a porni GStreamer.

    appsink nu raspunde la CAP_PROP_FRAME_COUNT, deci fara asta nu am avea nici
    procent de progres, nici ETA, iar videoul de iesire ar primi fps greșit.
    """

    probe = cv2.VideoCapture(path)
    if not probe.isOpened():
        return 0.0, 0

    fps = float(probe.get(cv2.CAP_PROP_FPS))
    frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()

    if not np.isfinite(fps) or fps <= 0:
        fps = 0.0

    return fps, max(0, frames)


def open_writer(path: Path, fps: float, width: int, height: int, args):
    """Videoul de iesire: NVENC prin GStreamer, cu mp4v ca rezerva."""

    if not args.no_gst:
        pipeline = gst_encode_pipeline(path, args.bitrate, args.encoder)
        writer = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, fps, (width, height))
        if writer.isOpened():
            return writer, f"GStreamer / {args.encoder} pe NVENC"
        print("  GStreamer a esuat la encodare, se revine la mp4v (pe CPU).")

    fourcc = getattr(cv2, "VideoWriter_fourcc", None) or cv2.VideoWriter.fourcc

    writer = cv2.VideoWriter(str(path), int(fourcc(*"mp4v")), fps, (width, height))
    return (writer, "mp4v (software)") if writer.isOpened() else (None, "esuat")


class LiveFrameGrabber:
    """Citeste camera pe un fir separat si pastreaza doar cadrul cel mai nou.

    Fara asta nu exista timp real: camera livreaza 30 cadre/secunda indiferent
    cat de repede procesezi. Daca pipeline-ul face 8 fps pe Xavier, restul se
    aduna intr-o coada si imaginea ramane tot mai mult in urma.

    Aici pierdem cadre, dar ramanem in prezent — compromisul corect pentru
    supraveghere.
    """

    def __init__(self, capture: cv2.VideoCapture):
        self.capture = capture
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.dropped = 0
        self.running = True
        self.error = None

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while self.running:
            ok, frame = self.capture.read()

            if not ok:
                self.error = "sursa nu mai livreaza cadre"   # camera deconectata
                self.running = False
                break

            with self.lock:
                if self.frame is not None:
                    self.dropped += 1      # cel vechi n-a fost citit: il aruncam
                self.frame = frame
                self.frame_id += 1

    def read(self, timeout: float = 5.0) -> tuple[bool, np.ndarray | None]:
        """Cel mai nou cadru necitit; asteapta pana la timeout daca nu exista."""

        deadline = time.perf_counter() + timeout

        while True:
            with self.lock:
                if self.frame is not None:
                    frame = self.frame
                    self.frame = None
                    return True, frame

            if not self.running:
                return False, None

            if time.perf_counter() > deadline:
                self.error = f"niciun cadru in {timeout:.0f}s"
                return False, None

            time.sleep(0.002)

    def stop(self) -> None:
        self.running = False
        self.thread.join(timeout=1.0)


@dataclass
class VideoSource:
    """Sursa video, cu sau fara fir de citire — bucla principala nu vede diferenta."""

    capture: cv2.VideoCapture
    backend: str
    fps: float
    total_frames: int
    live: bool
    grabber: LiveFrameGrabber | None = None
    first_frame: np.ndarray | None = None   # cadrul cu care am validat pipeline-ul

    def read(self) -> tuple[bool, np.ndarray | None]:
        # Primul cadru a fost deja citit ca sa verificam ca pipeline-ul livreaza
        # date; il dam acum, ca sa nu lipseasca din rezultate.
        if self.first_frame is not None:
            frame = self.first_frame
            self.first_frame = None
            return True, frame

        if self.grabber is not None:
            return self.grabber.read()

        return self.capture.read()

    @property
    def dropped(self) -> int:
        return self.grabber.dropped if self.grabber else 0

    def stop_reason(self) -> str:
        if self.grabber is not None and self.grabber.error:
            return self.grabber.error
        return "sfarsit de flux"

    def close(self) -> None:
        if self.grabber is not None:
            self.grabber.stop()
        self.capture.release()


def open_source(args: argparse.Namespace) -> VideoSource | None:
    """Deschide fisier / camera USB / camera CSI / RTSP. None daca nu reuseste."""

    source = args.video
    live = is_live_source(source)

    capture = None
    backend = ""
    fps = 0.0
    total_frames = 0

    # --- fisier ---
    if not live:
        path = str(resolve_path(source))
        if not Path(path).is_file():
            print(f"EROARE: video inexistent: {path}")
            return None

        fps, total_frames = probe_video_file(path)

        if not args.no_gst:
            capture, backend = try_pipelines(
                gst_file_pipelines(path, args.flip), "fisier pe NVDEC"
            )
            if capture is None:
                print("  GStreamer nu a putut deschide fisierul, se revine la OpenCV.")

        if capture is None:
            capture = cv2.VideoCapture(path)
            backend = "OpenCV / ffmpeg (decodare pe CPU)"

    # --- camera CSI ---
    elif args.csi and source.isdigit():
        capture, backend = try_pipelines(
            gst_csi_pipelines(
                int(source), args.cam_width, args.cam_height, args.cam_fps, args.flip
            ),
            f"camera CSI sensor-id={source}",
        )
        if capture is None:
            print(
                f"EROARE: nu pot deschide camera CSI {source}.\n"
                "  Verifica:  ls /dev/video*\n"
                "             sudo systemctl status nvargus-daemon"
            )
            return None

    # --- camera USB ---
    elif source.isdigit():
        if not args.no_gst:
            capture, backend = try_pipelines(
                gst_usb_pipelines(
                    int(source), args.cam_width, args.cam_height, args.cam_fps, args.flip
                ),
                f"camera USB /dev/video{source}",
            )
        if capture is None:
            capture = cv2.VideoCapture(int(source))
            backend = "OpenCV / V4L2"

    # --- flux de retea ---
    else:
        if not args.no_gst:
            capture, backend = try_pipelines(
                gst_network_pipelines(source, args.flip), "RTSP pe NVDEC"
            )
        if capture is None:
            os.environ.setdefault(
                "OPENCV_FFMPEG_CAPTURE_OPTIONS",
                "rtsp_transport;udp|buffer_size;102400|max_delay;500000",
            )
            capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            backend = "OpenCV / ffmpeg"
            if capture.isOpened():
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if capture is None or not capture.isOpened():
        print(f"EROARE: nu pot deschide sursa: {source}")
        if source.isdigit():
            print(
                "  Camera e folosita de alt proces? --list-cameras arata ce e\n"
                "  disponibil. Pentru camera de pe conectorul CSI foloseste --csi."
            )
        return None

    # Citim primul cadru ca sa fim siguri ca pipeline-ul chiar livreaza date:
    # cu GStreamer, isOpened() poate fi True si totusi sa nu vina nimic.
    # Cadrul nu se pierde — il pastram in VideoSource si il dam la prima citire.
    ok, first_frame = capture.read()
    if not ok or first_frame is None:
        print(
            f"EROARE: sursa s-a deschis dar nu livreaza cadre ({backend}).\n"
            "  Incearca --no-gst ca sa vezi daca problema e in pipeline."
        )
        capture.release()
        return None

    if fps <= 0:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = float(args.cam_fps) if live else 30.0

    if live:
        total_frames = 0
    if args.max_frames > 0:
        total_frames = (
            min(total_frames, args.max_frames) if total_frames > 0 else args.max_frames
        )

    height, width = first_frame.shape[:2]

    video_source = VideoSource(
        capture=capture,
        backend=backend,
        fps=fps,
        total_frames=total_frames,
        live=live,
        first_frame=first_frame,
    )

    if live:
        print("=" * 62)
        print("SURSA LIVE")
        print("=" * 62)
        print(f"  rezolutie primita   : {width}x{height} @ {fps:.0f} fps")
        print("  cadrele vechi sunt aruncate ca sa ramana in timp real")
        if source.isdigit() and not args.csi:
            print(
                "  daca fps-ul real e mult sub cel cerut, camera nu da MJPEG:\n"
                "  scade --cam-width/--cam-height sau muta-o pe un port USB 3"
            )
        print()

        # Firul de citire porneste abia acum, dupa ce sursa a fost validata.
        video_source.grabber = LiveFrameGrabber(capture)

    return video_source


def list_cameras() -> None:
    """Ce camere vede placa: /dev/video* plus senzorii CSI."""

    print("Camere disponibile:")

    devices = sorted(Path("/dev").glob("video*"))
    if not devices:
        print("  niciun /dev/video* (camera nu e conectata?)")

    for device in devices:
        name_file = Path("/sys/class/video4linux") / device.name / "name"
        try:
            name = name_file.read_text().strip()
        except OSError:
            name = "necunoscut"
        print(f"  {str(device):14s} {name}")

    # Senzorii CSI merg prin demonul Argus, nu ca dispozitiv V4L2 obisnuit.
    print()
    print("  Senzori CSI (pentru --csi):")
    try:
        probe = subprocess.run(
            ["gst-inspect-1.0", "nvarguscamerasrc"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
        )
        if probe.returncode == 0:
            print("    nvarguscamerasrc exista; incearca --csi cu sensor-id 0 sau 1")
        else:
            print("    nvarguscamerasrc lipseste (placa nu are camera CSI configurata)")
    except (OSError, subprocess.SubprocessError):
        print("    gst-inspect-1.0 nu e disponibil")


def can_show_windows() -> bool:
    """Xavier-ul ruleaza deseori headless; nu vrem sa crape acolo."""

    if not os.environ.get("DISPLAY"):
        return False
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__probe__")
        return True
    except cv2.error:
        return False


# ============================================================
# 10. REZULTATUL PENTRU O FATA
# ============================================================

# Ce s-a intamplat cu o fata din cadru:
FACE_TOO_SMALL = "too_small"        # sub min_face, nici nu am incercat
FACE_LANDMARK_FAIL = "lmk_fail"     # 2d106det a dat puncte fara sens
FACE_EMBEDDING_FAIL = "emb_fail"    # alinierea sau encoderul au picat
FACE_OK = "ok"                      # avem embedding si o decizie


@dataclass
class FaceResult:
    """Ce am aflat despre o fata dintr-un cadru. Desenarea si scrierea in
    fisiere se fac din asta, nu din interiorul pipeline-ului."""

    box: tuple[int, int, int, int]
    kind: str
    detection_index: int = -1
    track_id: int = -1
    label: str = ""
    color: tuple[int, int, int] = (110, 110, 110)

    # Doar pentru kind == FACE_OK:
    person_id: int = -1
    status: str = ""
    similarity: float = -1.0
    quality: float = 0.0
    yolo_conf: float = 0.0
    embedding: np.ndarray | None = None
    landmarks: np.ndarray | None = None
    five: np.ndarray | None = None


@dataclass
class Stats:
    """Numaratori pentru raportul final."""

    detections: int = 0
    too_small: int = 0
    landmark_fail: int = 0
    embedding_fail: int = 0
    recognized: int = 0
    new: int = 0
    pending: int = 0
    embeddings_written: int = 0

    reject_reasons: dict[str, int] = field(default_factory=dict)
    seen_track_ids: set[int] = field(default_factory=set)
    face_sizes: list[float] = field(default_factory=list)
    qualities: list[float] = field(default_factory=list)
    frame_times: list[float] = field(default_factory=list)


# ============================================================
# 11. PIPELINE-UL  (aici e toata logica, pas cu pas)
# ============================================================


class FacePipeline:
    """Leaga cele sase etape. Nu deseneaza si nu scrie fisiere — doar decide."""

    def __init__(
        self,
        models: Models,
        gallery: Gallery,
        thresholds: Thresholds,
        allow_enroll: bool = True,
    ):
        self.models = models
        self.gallery = gallery
        self.thresholds = thresholds
        self.allow_enroll = allow_enroll

        self.tracker = IouTracker(thresholds.conf, YOLO_LOW_CONF)
        self.tracks: dict[int, TrackState] = {}
        self.stats = Stats()

    # ------------------------------------------------------------
    # Un cadru intreg
    # ------------------------------------------------------------

    def process_frame(self, frame: np.ndarray, frame_index: int) -> list[FaceResult]:
        """PASII 1-2 pentru cadru, apoi 3-6 pentru fiecare fata gasita."""

        # PASUL 1: detectie. Pragul e cel mic — detectiile slabe sunt utile
        # tracker-ului chiar daca nu le folosim la recunoastere.
        boxes, scores = self.models.detector.detect(frame, YOLO_LOW_CONF)

        # PASUL 2: tracking. De aici incolo lucram cu track-uri, nu cu detectii.
        tracked = self.tracker.update(boxes, scores)
        self.stats.detections += len(tracked)

        results: list[FaceResult] = []
        for detection_index, (track_id, box, score) in enumerate(tracked):
            result = self._process_face(
                frame, frame_index, detection_index, track_id, box, float(score)
            )
            if result is not None:
                results.append(result)

        return results

    # ------------------------------------------------------------
    # O singura fata
    # ------------------------------------------------------------

    def _process_face(
        self,
        frame: np.ndarray,
        frame_index: int,
        detection_index: int,
        track_id: int,
        box: np.ndarray,
        score: float,
    ) -> FaceResult | None:
        stats = self.stats

        # --- caseta in limitele cadrului ---
        frame_height, frame_width = frame.shape[:2]
        clamped = clamp_box(box, frame_width, frame_height)
        if clamped is None:
            return None

        face_size = float(min(clamped[2] - clamped[0], clamped[3] - clamped[1]))
        stats.face_sizes.append(face_size)

        if face_size < self.thresholds.min_face:
            stats.too_small += 1
            return FaceResult(box=clamped, kind=FACE_TOO_SMALL, track_id=track_id)

        # --- PASUL 3: landmarks (si verificarea lor) ---
        try:
            landmarks = self.models.landmark.get(frame, clamped)
            five = landmarks_106_to_five(landmarks)

            reason = check_five_landmarks(five, clamped)
            if reason is not None:
                raise ValueError(reason)
        except Exception as error:
            stats.landmark_fail += 1
            reason = str(error) or type(error).__name__
            stats.reject_reasons[reason] = stats.reject_reasons.get(reason, 0) + 1
            return FaceResult(
                box=clamped,
                kind=FACE_LANDMARK_FAIL,
                track_id=track_id,
                label="LMK FAIL",
                color=(0, 0, 255),
            )

        # --- PASII 4 si 5: aliniere -> embedding ---
        try:
            aligned = norm_crop(frame, five, self.models.encoder.input_size)
            quality = face_quality(aligned, face_size, score)
            embedding = l2_normalize(self.models.encoder.get_feat(aligned).flatten())
        except Exception as error:
            stats.embedding_fail += 1
            if stats.embedding_fail <= 5:
                print(f"  Embedding fail cadru={frame_index}: {error}")
            return FaceResult(
                box=clamped,
                kind=FACE_EMBEDDING_FAIL,
                track_id=track_id,
                label="EMB FAIL",
                color=(0, 0, 255),
            )

        stats.qualities.append(quality)
        stats.seen_track_ids.add(track_id)

        # --- PASUL 6: identitatea, decisa pe track, nu pe cadru ---
        state = self.tracks.get(track_id)
        if state is None:
            state = TrackState(track_id=track_id)
            self.tracks[track_id] = state

        state.add_sample(embedding, quality)
        state.last_frame = frame_index

        # Nu inregistram persoane noi din fete mici sau detectii nesigure —
        # asa apar identitatile fantoma.
        allow_enroll = (
            self.allow_enroll
            and face_size >= self.thresholds.min_enroll_face
            and score >= MIN_ENROLL_YOLO_CONF
        )

        if state.needs_decision():
            decide_identity(
                state,
                self.gallery,
                person_ids_in_use(self.tracks, track_id, frame_index),
                allow_enroll,
                self.thresholds,
            )

        return self._describe(
            state, clamped, detection_index, quality, score, embedding, landmarks, five
        )

    def _describe(
        self,
        state: TrackState,
        box: tuple[int, int, int, int],
        detection_index: int,
        quality: float,
        score: float,
        embedding: np.ndarray,
        landmarks: np.ndarray,
        five: np.ndarray,
    ) -> FaceResult:
        """Traduce starea track-ului in eticheta si culoare."""

        if state.person_id is None:
            status, person_id = "pending", -1
            label, color = f"T{state.track_id} ?", (0, 165, 255)       # portocaliu
            self.stats.pending += 1

        elif state.is_new_identity:
            status, person_id = "new", state.person_id
            label, color = f"ID {state.person_id:03d} NEW", (255, 128, 0)
            self.stats.new += 1
            state.is_new_identity = False       # "NEW" se arata o singura data

        else:
            status, person_id = "recognized", state.person_id
            label = f"ID {state.person_id:03d} {state.similarity:.2f}"
            color = (0, 200, 0)                                        # verde
            self.stats.recognized += 1

        return FaceResult(
            box=box,
            kind=FACE_OK,
            detection_index=detection_index,
            track_id=state.track_id,
            label=label,
            color=color,
            person_id=person_id,
            status=status,
            similarity=state.similarity,
            quality=quality,
            yolo_conf=score,
            embedding=embedding,
            landmarks=landmarks,
            five=five,
        )


# ============================================================
# 12. DESENARE SI SCRIERE
# ============================================================


def draw_result(frame: np.ndarray, result: FaceResult) -> None:
    """Deseneaza o fata: gri = prea mica, rosu = eroare, verde = recunoscuta."""

    x1, y1, x2, y2 = result.box

    if result.kind == FACE_TOO_SMALL:
        cv2.rectangle(frame, (x1, y1), (x2, y2), result.color, 1)
        return

    if result.kind in (FACE_LANDMARK_FAIL, FACE_EMBEDDING_FAIL):
        cv2.rectangle(frame, (x1, y1), (x2, y2), result.color, 1)
        draw_label(frame, result.label, x1, y1, result.color, 0.42)
        return

    cv2.rectangle(frame, (x1, y1), (x2, y2), result.color, 2)
    draw_label(frame, result.label, x1, y1, result.color)

    if DRAW_ALL_106_LANDMARKS and result.landmarks is not None:
        for point in result.landmarks[:, :2]:
            cv2.circle(frame, _to_int_point(point), 1, (255, 0, 255), -1)

    if DRAW_FIVE_ARCFACE_POINTS and result.five is not None:
        # ochi, ochi, nas, gura, gura
        colors = ((0, 255, 255), (0, 255, 255), (255, 255, 0), (255, 0, 255), (255, 0, 255))
        for point, color in zip(result.five, colors):
            cv2.circle(frame, _to_int_point(point), 2, color, -1)


def _to_int_point(point: np.ndarray) -> tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


def draw_hud(frame: np.ndarray, text: str) -> None:
    cv2.putText(
        frame, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
    )


class ResultWriter:
    """Scrie doua fisiere care merg mana in mana:

    arcface_embeddings.txt       linia N = cele 512 numere
    arcface_embedding_index.txt  linia N = de unde vine (cadru, track, ID, ...)
    """

    HEADER = (
        "embedding_line frame_index detection_index track_id person_id "
        "status similarity quality x1 y1 x2 y2 yolo_confidence\n"
    )

    def __init__(self, embeddings_path: Path, index_path: Path):
        self.embeddings_path = embeddings_path
        self.index_path = index_path
        self.lines_written = 0

    def __enter__(self) -> ResultWriter:
        self._emb_file = self.embeddings_path.open("w", encoding="utf-8")
        self._idx_file = self.index_path.open("w", encoding="utf-8")
        self._idx_file.write(self.HEADER)
        return self

    def __exit__(self, *exc_info) -> None:
        self._emb_file.close()
        self._idx_file.close()

    def write(self, result: FaceResult, frame_index: int) -> None:
        self.lines_written += 1

        values = " ".join(f"{v:.{TXT_PRECISION}f}" for v in result.embedding)
        self._emb_file.write(values + "\n")

        x1, y1, x2, y2 = result.box
        self._idx_file.write(
            f"{self.lines_written} {frame_index} {result.detection_index} "
            f"{result.track_id} {result.person_id} {result.status} "
            f"{result.similarity:.6f} {result.quality:.6f} "
            f"{x1} {y1} {x2} {y2} {result.yolo_conf:.6f}\n"
        )


@dataclass
class OutputPaths:
    video: Path
    embeddings: Path
    index: Path
    gallery: Path


def build_output_paths(args: argparse.Namespace, live: bool) -> OutputPaths:
    if args.name:
        name = args.name
    elif live:
        name = "camera"
    else:
        name = Path(args.video).stem

    directory = resolve_path(args.results_root) / name
    directory.mkdir(parents=True, exist_ok=True)

    return OutputPaths(
        video=directory / f"{name}_ids.mp4",
        embeddings=directory / "arcface_embeddings.txt",
        index=directory / "arcface_embedding_index.txt",
        gallery=(
            resolve_path(args.gallery) if args.gallery
            else directory / "face_gallery_2d106.npz"
        ),
    )


# ============================================================
# 13. BUCLA PRINCIPALA
# ============================================================


@dataclass
class RunOutcome:
    frames_done: int
    elapsed: float
    interrupted: bool
    embeddings_written: int


def process_video(
    pipeline: FacePipeline,
    source: VideoSource,
    paths: OutputPaths,
    args: argparse.Namespace,
    show_video: bool,
) -> RunOutcome:
    """Citeste cadru cu cadru: proceseaza, deseneaza, scrie, afiseaza."""

    stats = pipeline.stats
    window_name = "YOLO + 2d106det + MobileFaceNet"

    writer: cv2.VideoWriter | None = None
    window_created = False

    frames_done = 0
    frame_index = -1
    interrupted = False
    start_time = time.perf_counter()

    result_writer = ResultWriter(paths.embeddings, paths.index)

    try:
        with result_writer:
            while True:
                ok, frame = source.read()
                if not ok:
                    if source.live:
                        print(f"\n  Sursa live s-a oprit: {source.stop_reason()}")
                    break

                frame_index += 1
                frames_done += 1

                if args.max_frames > 0 and frames_done > args.max_frames:
                    frames_done -= 1
                    break

                frame_height, frame_width = frame.shape[:2]

                # Videoul de iesire are nevoie de dimensiunile primului cadru.
                if writer is None and not args.no_video_out:
                    writer, backend = open_writer(
                        paths.video, source.fps, frame_width, frame_height, args
                    )
                    if writer is None:
                        print("  ATENTIE: nu pot crea videoul de iesire; continui fara.")
                        args.no_video_out = True
                    else:
                        print(f"  iesire  : {backend}\n")

                # --- tot lucrul pe cadru ---
                frame_start = time.perf_counter()

                results = pipeline.process_frame(frame, frame_index)
                for result in results:
                    draw_result(frame, result)
                    if result.kind == FACE_OK:
                        result_writer.write(result, frame_index)

                stats.frame_times.append(time.perf_counter() - frame_start)

                # --- afisare si salvare ---
                elapsed = time.perf_counter() - start_time
                speed = frames_done / elapsed if elapsed > 0 else 0.0

                hud = (
                    f"cadru {frame_index}  "
                    f"identitati {len(pipeline.gallery.ids)}  {speed:.1f} fps"
                )
                if source.live:
                    # Daca "sarite" creste repede, procesarea nu tine pasul:
                    # scade rezolutia camerei sau creste --min-face.
                    hud += f"  sarite {source.dropped}"
                draw_hud(frame, hud)

                if writer is not None:
                    writer.write(frame)

                if show_video:
                    if not window_created:
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        scale = min(1.0, PREVIEW_MAX_HEIGHT / frame_height)
                        cv2.resizeWindow(
                            window_name,
                            int(frame_width * scale),
                            int(frame_height * scale),
                        )
                        window_created = True

                    cv2.imshow(window_name, frame)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        print("  Oprit de utilizator (q).")
                        interrupted = True
                        break

                if frame_index > 0 and frame_index % PROGRESS_EVERY_N_FRAMES == 0:
                    print_progress(
                        frame_index,
                        frames_done,
                        source.total_frames,
                        speed,
                        len(pipeline.gallery.ids),
                        result_writer.lines_written,
                    )

                if frame_index > 0 and frame_index % SAVE_GALLERY_EVERY_N_FRAMES == 0:
                    pipeline.gallery.save()
                    prune_tracks(pipeline.tracks, frame_index)

    except KeyboardInterrupt:
        print("\n  Intrerupt de la tastatura (Ctrl+C).")
        interrupted = True

    finally:
        source.close()
        if writer is not None:
            writer.release()
        if window_created:
            cv2.destroyAllWindows()
        pipeline.gallery.save()

    stats.embeddings_written = result_writer.lines_written

    return RunOutcome(
        frames_done=frames_done,
        elapsed=time.perf_counter() - start_time,
        interrupted=interrupted,
        embeddings_written=result_writer.lines_written,
    )


def print_progress(
    frame_index: int,
    frames_done: int,
    total_frames: int,
    speed: float,
    identities: int,
    embeddings: int,
) -> None:
    if total_frames > 0:
        remaining = max(0, total_frames - frames_done)
        eta = remaining / speed if speed > 0 else 0.0
        print(
            f"  {100.0 * frames_done / total_frames:5.1f}% "
            f"cadru {frame_index}/{total_frames} | {speed:5.1f} fps | "
            f"ETA {eta:6.1f}s | identitati={identities} | embeddings={embeddings}"
        )
    else:
        print(
            f"  cadru {frame_index} | {speed:5.1f} fps | "
            f"identitati={identities} | embeddings={embeddings}"
        )


# ============================================================
# 14. RAPOARTE
# ============================================================


def read_text_file(path: str) -> str | None:
    """Citeste un fisier mic din /proc sau /etc, sau None daca nu exista."""

    try:
        return Path(path).read_text(errors="replace").strip().strip(chr(0))
    except OSError:
        return None


def print_environment() -> None:
    """Ce placa, ce JetPack, ce backend-uri — primul lucru de verificat."""

    print("=" * 62)
    print("MEDIU")
    print("=" * 62)
    print(f"  Python        : {sys.version.split()[0]}")
    print(f"  OpenCV        : {cv2.__version__}")
    print(f"  ONNX Runtime  : {ort.__version__}")
    print(f"  Provideri     : {ort.get_available_providers()}")

    gstreamer = has_gstreamer_support()
    print(f"  GStreamer     : {'da' if gstreamer else 'NU'}")

    board = read_text_file("/proc/device-tree/model")
    if board:
        print(f"  Placa         : {board}")

    release = read_text_file("/etc/nv_tegra_release")
    if release:
        print(f"  L4T           : {release.splitlines()[0].strip('# ')}")

    # Modul de putere: pe Xavier, modurile mici folosesc doar o parte din
    # nuclee si scad ceasurile. Conteaza mult intr-o masuratoare de fps.
    power_mode = read_nvpmodel()
    if power_mode:
        print(f"  Mod de putere : {power_mode}")
        print("                  maxim: sudo nvpmodel -m 0 && sudo jetson_clocks")

    if not gstreamer:
        print(
            "\n  ATENTIE: OpenCV-ul asta nu are GStreamer, deci decodarea pe\n"
            "           NVDEC nu functioneaza. Foloseste OpenCV din JetPack\n"
            "           (sudo apt install python3-opencv) sau ruleaza --no-gst."
        )
    print()


def read_nvpmodel() -> str | None:
    """Modul de putere curent, din `nvpmodel -q`."""

    try:
        query = subprocess.run(
            ["nvpmodel", "-q"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if query.returncode != 0:
        return None

    lines = [
        line.strip()
        for line in query.stdout.decode("utf-8", "replace").splitlines()
        if line.strip()
    ]
    return " / ".join(lines[-2:]) if lines else None


def print_report(
    pipeline: FacePipeline,
    outcome: RunOutcome,
    source: VideoSource,
    paths: OutputPaths,
    initial_identities: int,
    no_video_out: bool,
) -> None:
    stats = pipeline.stats
    gallery = pipeline.gallery

    mean_frame_ms = float(np.mean(stats.frame_times) * 1000) if stats.frame_times else 0.0
    fps = outcome.frames_done / outcome.elapsed if outcome.elapsed > 0 else 0.0

    print()
    print("=" * 62)
    print("REZULTAT" + ("  (intrerupt)" if outcome.interrupted else ""))
    print("=" * 62)
    print(f"  cadre procesate     : {outcome.frames_done}")
    print(f"  timp                : {outcome.elapsed:.1f}s ({fps:.1f} fps)")
    print(f"  inferenta / cadru   : {mean_frame_ms:.1f} ms")

    if source.live:
        delivered = outcome.frames_done + source.dropped
        share = 100.0 * outcome.frames_done / delivered if delivered else 0.0
        print(
            f"  cadre de la camera  : {delivered} "
            f"(procesate {outcome.frames_done} = {share:.0f}%, sarite {source.dropped})"
        )

    print(f"  detectii urmarite   : {stats.detections}")
    print(f"  respinse (prea mici): {stats.too_small}")
    print(f"  esec landmarks      : {stats.landmark_fail}")
    print(f"  esec embedding      : {stats.embedding_fail}")
    print(f"  embeddings scrise   : {outcome.embeddings_written}")
    print(f"  detectii recunoscute: {stats.recognized}")
    print(f"  detectii ID nou     : {stats.new}")
    print(f"  detectii fara ID    : {stats.pending}")
    print(f"  track-uri vazute    : {len(stats.seen_track_ids)}")
    print(
        f"  identitati in galerie: {len(gallery.ids)} "
        f"(+{len(gallery.ids) - initial_identities} fata de start)"
    )

    if stats.reject_reasons:
        print()
        print("  Motive respingere landmarks:")
        for reason, count in sorted(stats.reject_reasons.items(), key=lambda kv: -kv[1])[:8]:
            print(f"    {count:6d}  {reason}")

    print()
    print("  CALIBRARE:")
    print(f"    marime fata (px) : {percentiles(stats.face_sizes)}")
    print(f"    scor calitate    : {percentiles(stats.qualities)}")

    if stats.face_sizes:
        below = sum(1 for s in stats.face_sizes if s < pipeline.thresholds.min_enroll_face)
        share = 100.0 * below / len(stats.face_sizes)
        if share > 60.0:
            print(
                f"    ATENTIE: {share:.0f}% din fete sunt sub --min-enroll-face; "
                "scade pragul."
            )

    print()
    print("  Fisiere:")
    if not no_video_out:
        print(f"    video      : {paths.video}")
    print(f"    embeddings : {paths.embeddings}")
    print(f"    index      : {paths.index}")
    print(f"    galerie    : {paths.gallery}")
    print()


# ============================================================
# 15. LINIA DE COMANDA
# ============================================================


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = Thresholds()

    parser = argparse.ArgumentParser(
        description="YOLO + 2d106det + MobileFaceNet pe Jetson Xavier, prin ONNX Runtime.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("video", nargs="?", default=DEFAULT_VIDEO,
                        help="Fisier video, index de camera (ex: 0), sau rtsp://...")

    # --- modele si iesiri ---
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    parser.add_argument("--yolo", default=None,
                        help="Alt best.onnx decat cel implicit de langa script.")
    parser.add_argument("--landmark", default=None, help="Alt 2d106det.onnx.")
    parser.add_argument("--embedding", default=None, help="Alt w600k_mbf.onnx.")
    parser.add_argument("--gallery", default=None)
    parser.add_argument("--name", default=None,
                        help="Numele folderului de rezultate (implicit: numele sursei).")

    # --- praguri ---
    parser.add_argument("--conf", type=float, default=defaults.conf)
    parser.add_argument("--match-threshold", type=float, default=defaults.match)
    parser.add_argument("--min-face", type=int, default=defaults.min_face)
    parser.add_argument("--min-enroll-face", type=int, default=defaults.min_enroll_face)
    parser.add_argument("--min-enroll-quality", type=float,
                        default=defaults.min_enroll_quality)

    # --- accelerare ---
    parser.add_argument("--no-trt", action="store_true",
                        help="Fara TensorRT: porneste imediat, dar merge mai incet.")
    parser.add_argument("--trt-cache", default="trt_cache",
                        help="Director pentru motoarele TensorRT compilate.")
    parser.add_argument("--no-gst", action="store_true",
                        help="Fara GStreamer: video pe CPU. Doar pentru depanare.")
    parser.add_argument("--cpu", action="store_true",
                        help="Forteaza CPU. Pe Xavier e inutilizabil, doar pentru test.")

    # --- video de iesire ---
    parser.add_argument("--encoder", choices=("h264", "h265"), default="h264",
                        help="Codecul NVENC pentru videoul de iesire.")
    parser.add_argument("--bitrate", type=int, default=DEFAULT_ENCODE_BITRATE,
                        help="Bitrate-ul videoului de iesire, in biti/s.")
    parser.add_argument("--flip", type=int, default=0, choices=range(0, 8),
                        help="nvvidconv flip-method (2 = rotit 180, util pe CSI).")

    # --- afisare ---
    # Fereastra e implicit oprita: placa ruleaza de obicei fara monitor.
    parser.add_argument("--show", dest="show", action="store_true", default=False,
                        help="Afiseaza fereastra de preview (necesita DISPLAY).")
    parser.add_argument("--no-show", dest="show", action="store_false",
                        help="Nu afisa fereastra (implicit).")
    parser.add_argument("--no-video-out", action="store_true",
                        help="Nu scrie videoul de iesire.")
    parser.add_argument("--save-video", action="store_true",
                        help="Scrie videoul si pentru surse live (creste la nesfarsit).")

    # --- galerie ---
    parser.add_argument("--no-enroll", action="store_true")
    parser.add_argument("--reset-gallery", action="store_true")

    # --- camera ---
    parser.add_argument("--csi", action="store_true",
                        help="Indexul dat e un senzor CSI, nu /dev/video*.")
    parser.add_argument("--cam-width", type=int, default=1280)
    parser.add_argument("--cam-height", type=int, default=720)
    parser.add_argument("--cam-fps", type=float, default=30.0)
    parser.add_argument("--list-cameras", action="store_true",
                        help="Afiseaza camerele disponibile si iese.")

    # --- diverse ---
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--check", action="store_true",
                        help="Verifica placa, GStreamer si modelele, apoi iese.")

    return parser.parse_args(argv)


def load_gallery(path: Path, reset: bool) -> Gallery:
    print("=" * 62)
    print("GALERIE")
    print("=" * 62)

    gallery = Gallery(path)
    if reset and path.exists():
        print(f"  --reset-gallery: se ignora {path.name}")
    else:
        gallery.load()
    print()

    return gallery


# ============================================================
# 16. MAIN
# ============================================================


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list_cameras:
        list_cameras()
        return 0

    print_environment()

    # 1. Modelele, direct de pe placa.
    try:
        models = load_models(args)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"\nEROARE: {error}")
        return 1

    if args.check:
        print("Verificare terminata: modelele se incarca si sesiunile pornesc.")
        return 0

    # 2. Sursa video.
    source = open_source(args)
    if source is None:
        return 1

    # 3. Unde scriem, si ce identitati stim deja.
    paths = build_output_paths(args, source.live)
    gallery = load_gallery(paths.gallery, args.reset_gallery)
    initial_identities = len(gallery.ids)

    # 4. Pipeline-ul.
    thresholds = Thresholds.from_args(args)
    pipeline = FacePipeline(
        models=models,
        gallery=gallery,
        thresholds=thresholds,
        allow_enroll=not args.no_enroll,
    )

    # 5. Fereastra, doar daca exista DISPLAY.
    show_video = args.show and can_show_windows()
    if args.show and not show_video:
        print("  Fara DISPLAY; fereastra dezactivata.\n")

    # Pe camera nu scriem video implicit: un flux live nu se opreste singur,
    # iar fisierul ar creste la nesfarsit.
    if source.live and not args.save_video:
        args.no_video_out = True

    print("=" * 62)
    print("PROCESARE")
    print("=" * 62)
    print(f"  sursa   : {source.backend}")
    total = source.total_frames if source.total_frames > 0 else "necunoscut"
    print(f"  fps={source.fps:.2f} cadre={total}")
    print(f"  match_threshold={thresholds.match}  min_face={thresholds.min_face}px")
    print()

    # 6. Rulam si raportam.
    outcome = process_video(pipeline, source, paths, args, show_video)
    print_report(pipeline, outcome, source, paths, initial_identities, args.no_video_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
