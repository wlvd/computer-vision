"""Video-file variant of the face recognition pipeline -- runtime only.

Same recognition as complete_pipeline_tRT_media_15.py, gate for gate: the same detection
thresholds, the same quality window and best-frame choice on frontality, the same
cooldown, the same identity continuity and the same automatic enrollment. What is gone is
the tuning scaffolding, which had nothing to do with recognising a face:

  - summary_*.json and frames_*.jsonl, with the counters and distributions behind them;
  - the enrollment-blocker diagnostics and the memory-error explanations;
  - 31 of the 41 command-line options. The thresholds are now plain constants below --
  edit them here instead of passing flags. Use _15 when you need to measure why a
  threshold blocks, then copy the values you settled on into this file.

The output folder is REUSED across runs on the same video: the database inside it is
loaded at startup and written back there, so the first run enrolls and the second
recognises. The annotated clips are numbered, so older runs stay.

Usage:
    python3 complete_pipeline_tRT_media_17.py sample_vid.mp4
    python3 complete_pipeline_tRT_media_17.py sample_vid.mp4      # second time: recognises
    python3 complete_pipeline_tRT_media_17.py sample_vid.mp4 --no-video --reset-db

The video name is looked up as given, next to the script, then in sample/.

Two behaviours worth knowing before reading the code:

  COOLDOWN. Once a track has a confirmed identity it is re-checked every
  IDENTIFIED_INTERVAL_FRAMES instead of VERIFY_INTERVAL_FRAMES, and in between it gets no
  landmarks at all (they were 63% of the probe time). The cost: if the tracker swaps the
  person under the same id the wrong label stays until the next check, and the points are
  not drawn on recognised people. A track without an identity still gets landmarks every
  frame, so enrollment is not affected.

  LABEL CONTINUITY. What we know about a person (TrackState.identity) is kept apart from
  what the last check produced (the vote in history): a conclusion versus a measurement.
  An identity falls only after IDENTITY_FLIP_CHECKS consecutive contradicting checks, and
  boxes that fail the gates are still labelled with it (dim green with "~"). Without both,
  one bad window or one turned head turned "person_13" back into nvdsosd's "face 1".
"""

import argparse
import os
import json
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
# CONFIGURATION -- edit here; there are no flags for these
# ============================================================

# Some old bindings still ask for np.bool; on a recent numpy even the check warns.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    if not hasattr(np, "bool"):
        np.bool = bool
# The indexed-binding API is the TensorRT 8.5 one (JetPack 5.x) and is deprecated there.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="__main__")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_CONFIG_PATH = "config_infer_best_v2.txt"
PFLD_MODEL_PATH = "/workspace/DeepStream-Yolo/pfld.engine"
RECOGNITION_MODEL_PATH = "/workspace/DeepStream-Yolo/w600k_mbf.engine"
FACE_DATABASE_PATH = "/workspace/DeepStream-Yolo/face_database.json"

# NvDCF_accuracy keeps tracks in the shadow longer, which is what is missing when somebody
# turns their head; if it is not installed, fall back to perf.
_DS_CONFIGS = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app"
TRACKER_CONFIG_PATH = [
   # f"{_DS_CONFIGS}/config_tracker_NvDCF_accuracy.yml",
    f"{_DS_CONFIGS}/config_tracker_NvDCF_perf.yml"]

# Rewritten into a copy kept in the run folder, so the DeepStream original stays untouched.
# minDetectorConfidence 0.0 = the tracker gets every box (weak boxes are what keep a track
# linked); maxShadowTrackAge is how many frames a track survives with no detection, and the
# NvDCF default (~30) breaks most gaps, which reach 233 frames in our measurements.
TRACKER_OVERRIDES = {"minDetectorConfidence": 0.0, "maxShadowTrackAge": 90}

MIN_CONFIDENCE, MIN_FACE_SIZE, MIN_BLUR = 0.5, 40, 80.0   # detection gate: score, px, sharpness
# Confidence gate for a box on a track we already know. Lower on purpose: the tracker has
# already linked it to a history, and a turned head (0.3-0.4) must not fall out entirely.
TRACKED_MIN_CONFIDENCE = 0.30
# Boxes too small to ever be usable faces are dropped right after the detector, so the
# tracker never sees them (measured: 42 boxes per frame, 4 usable). The threshold is below
# MIN_FACE_SIZE on purpose, so a face oscillating around the gate keeps its track id, and
# it does NOT look at confidence: see detection_filter_probe.
PRETRACK_FILTER = True
PRETRACK_MIN_SIZE = MIN_FACE_SIZE * 0.8
# Samples below MIN_BLUR still compete: on a file the Laplacian variance depends on the
# codec and on scaling, so an absolute threshold can block a track forever. Only this
# safety floor is dropped outright.
BLUR_REJECT_FACTOR = 0.25

VERIFY_INTERVAL_FRAMES = 15   # at most this many frames between checks of a track
RETRY_INTERVAL_ON_FAIL = 7    # sooner retry when the quality gate failed
IDENTIFIED_INTERVAL_FRAMES = 30   # the cooldown; see the header
LABEL_HISTORY_SIZE = 5        # recent decisions kept for the majority vote
TRACK_TIMEOUT_FRAMES = 300    # absent frames after which a track is forgotten
PRUNE_CHECK_INTERVAL = 90     # how often we look for dead tracks
IDENTITY_FLIP_CHECKS = 2      # consecutive contradicting checks needed to drop an identity
IDENTITY_STALE_FRAMES = 150   # after this long without confirmation: yellow, with "?"

QUALITY_WINDOW_FRAMES = 6     # frames before the deadline in which samples are collected
CANDIDATE_INTERVAL_FRAMES = 2 # frames between samples inside that window
QUALITY_GOOD_ENOUGH = 0.65    # good enough to check immediately
QUALITY_MIN = 0.30            # below this the recognition model is not worth running
QUALITY_WEIGHTS = {"yaw": 0.30, "pitch": 0.20, "roll": 0.10, "sharp": 0.20, "size": 0.20}

VERIFY_THRESHOLD = 0.42       # cosine score from which a face is recognised
ENROLL_MARGIN = 0.10          # uncertainty band below the recognition threshold
ENROLL_MAX_SCORE = VERIFY_THRESHOLD - ENROLL_MARGIN
AUTO_ENROLL, ENROLL_MIN_CHECKS = True, 2
ENROLL_MIN_FACE, ENROLL_MIN_BLUR, ENROLL_MIN_QUALITY = 40, 80.0, 0.35
# Frontality is the one gate that cannot be compensated for: prototypes at frontality 0
# gave 0.089 self-similarity against other shots of the same person, those at 0.29 and
# 0.52 gave 0.80 and 0.85.
ENROLL_MIN_FRONTALITY = 0.15

LABEL_UNKNOWN, LABEL_UNCERTAIN = "unknown", "uncertain"
# The old Romanian prefix is still recognised when numbering, so a database from an
# earlier run keeps counting instead of restarting at 1.
ENROLL_NAME_PREFIX = "person"
ENROLL_NAME_RE = re.compile(r"^(?:person|persoana)_(\d+)$")

ARCFACE_TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                             [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
ALIGN_SIZE = 112

QUEUE_SIZE = 4                # frames waiting in each queue; every one is an NVMM surface
PROGRESS_EVERY_FRAMES = 100
# Points are written straight into the mapped surface (unified memory on Jetson); boxes and
# labels go through metadata, so they survive even if the surface is not writable.
# Landmarks are only asked for where they are needed: faces giving a sample now, and tracks
# with no identity yet.
LANDMARK_ONLY_WHEN_NEEDED = DRAW_ALL_LANDMARKS = True
DRAW_OVERLAY = True           # off with --no-video: nobody is left to see it
DRAW_REJECTED_BOXES = True    # stateless detections: thin grey box, no text
LANDMARK_RADIUS = 2
# RGBA, in the surface's channel order (these are pixels, not nvdsosd structs).
LANDMARK_COLOR = (255, 255, 0, 255)
POINT_COLORS = np.array([  # left eye, right eye, nose, left mouth, right mouth
    (0, 204, 255, 255), (0, 204, 255, 255), (51, 255, 51, 255),
    (255, 102, 102, 255), (255, 102, 102, 255)], dtype=np.uint8)

def disc_offsets(radius):
    """(dy, dx) offsets of the pixels inside a disc, precomputed once."""
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
BOX_COLOR_HELD = (0.3, 0.85, 0.5, 1.0)    # known identity, not confirmed in this frame
BOX_COLOR_STALE = (0.9, 0.75, 0.2, 1.0)   # not confirmed for IDENTITY_STALE_FRAMES
BOX_WIDTH_CONFIRMED, BOX_WIDTH_HELD, BOX_WIDTH_REJECTED = 2, 1, 1

# Optional pyds symbols, resolved once so a missing one does not raise every frame.
_unmap_surface = getattr(pyds, "unmap_nvds_buf_surface", None)
_remove_obj_meta = getattr(pyds, "nvds_remove_obj_meta_from_frame", None)
_warned = set()

def warn_once(key, message):
    if key not in _warned:
        _warned.add(key)
        print(f"[WARNING] {message}")

def check_pyds_api():
    """Fail at startup on missing mandatory symbols; only warn on optional ones."""
    missing = [name for name in ("gst_buffer_get_nvds_batch_meta", "get_nvds_buf_surface",
                                 "NvDsFrameMeta", "NvDsObjectMeta")
               if not hasattr(pyds, name)]
    if missing:
        raise RuntimeError("The pyds bindings are missing: " + ", ".join(missing) +
                           ".\nCheck the deepstream_python_apps version against DeepStream.")
    if _unmap_surface is None:
        warn_once("unmap", "pyds has no unmap_nvds_buf_surface; surfaces are not released "
                           "explicitly (watch the memory on long runs).")
    global PRETRACK_FILTER
    if PRETRACK_FILTER and _remove_obj_meta is None:
        PRETRACK_FILTER = False
        warn_once("remove_obj", "pyds has no nvds_remove_obj_meta_from_frame; the tracker "
                                "will also get the boxes that are too small (slower, same "
                                "results).")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
cuda.init()
CUDA_CONTEXT = cuda.Device(0).retain_primary_context()

landmark_model = None       # filled in by load_models(): the paths depend on the arguments
recognition_model = None
face_database = None
track_states = {}
STATS = Counter()           # frames / faces, only for the line printed at the end

# ============================================================
# TENSORRT MODELS
# ============================================================

class TrtModel:
    """A TensorRT .engine with its buffers allocated once, at the maximum batch."""

    def __init__(self, engine_path):
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f"Cannot find the engine: {engine_path}\nBuild it from "
                                    f"the .onnx with trtexec (see the instructions).")
        CUDA_CONTEXT.push()
        try:
            with open(engine_path, "rb") as f:
                self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
            if self.engine is None:
                raise RuntimeError(f"The engine {engine_path} could not be deserialized: "
                                   f"usually a different TensorRT version or a different GPU.")
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()

            indices = list(range(self.engine.num_bindings))
            ins = [i for i in indices if self.engine.binding_is_input(i)]
            shapes = {i: tuple(self.engine.get_binding_shape(i)) for i in ins}
            self.dynamic = any(shapes[i][0] == -1 for i in ins)
            # On a dynamic input the ceiling is the profile's maximum; the smallest wins.
            self.max_batch = max(1, min([int(self.engine.get_profile_shape(0, i)[2][0])
                                         if shapes[i][0] == -1 else int(shapes[i][0])
                                         for i in ins] or [1]))
            # The buffers can only be sized once every binding has a concrete shape.
            for i in ins:
                if -1 in shapes[i]:
                    self.context.set_binding_shape(i, (self.max_batch,) + tuple(
                        1 if d == -1 else d for d in shapes[i][1:]))

            self.bindings, self.inputs, self.outputs = [0] * len(indices), [], []
            for i in indices:
                shape = tuple(self.context.get_binding_shape(i))
                host = cuda.pagelocked_empty(int(np.prod(shape)),
                                             trt.nptype(self.engine.get_binding_dtype(i)))
                host[:] = 0
                device = cuda.mem_alloc(host.nbytes)
                self.bindings[i] = int(device)
                target = self.inputs if self.engine.binding_is_input(i) else self.outputs
                target.append({"index": i, "shape": shape, "sample_shape": shape[1:],
                               "sample_elems": int(np.prod(shape[1:])), "dtype": host.dtype,
                               "host": host, "device": device})
            self._current_batch = self.max_batch
        finally:
            CUDA_CONTEXT.pop()
        print(f"  {os.path.basename(engine_path)}: input {self.inputs[0]['shape']} -> output "
              f"{self.outputs[0]['shape']} (max batch {self.max_batch}, "
              f"{'dynamic' if self.dynamic else 'fixed'})")

    def _set_batch(self, batch):
        if not self.dynamic or batch == self._current_batch:
            return
        for entry in self.inputs:
            self.context.set_binding_shape(entry["index"], (batch,) + entry["sample_shape"])
        self._current_batch = batch

    def infer_batch(self, array):
        """Run the engine on n samples at once (n <= max_batch)."""
        source = self.inputs[0]
        data = np.ascontiguousarray(array, dtype=source["dtype"])
        count = int(data.shape[0])
        if not 1 <= count <= self.max_batch:
            raise ValueError(f"Batch of {count}, the engine supports 1..{self.max_batch}.")

        CUDA_CONTEXT.push()
        try:
            self._set_batch(count)
            flat = data.ravel()
            source["host"][:flat.size] = flat
            # Copy only the requested samples: the buffers are sized for the worst case
            # (batch 8) but a frame usually has 4 faces.
            cuda.memcpy_htod_async(source["device"], source["host"][:flat.size], self.stream)
            self.context.execute_async_v2(bindings=self.bindings,
                                          stream_handle=self.stream.handle)
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

def run_batched(model, tensors):
    """Run the model in chunks of at most max_batch, preserving the order."""
    results = []
    for start in range(0, len(tensors), model.max_batch):
        results.extend(model.infer_batch(np.stack(tensors[start:start + model.max_batch]))[0])
    return results

def l2(vec):
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-12 else vec

# ============================================================
# DATABASE
# ============================================================

class FaceDatabase:
    """Prototypes kept as an (N, 512) matrix: a lookup is a single matmul.

    Same {name: [512 floats]} JSON as the other W7/W8 scripts.
    """

    def __init__(self, path, save_path=None):
        self.save_path = save_path or path
        self.labels = []
        self.matrix = np.zeros((0, 0), dtype=np.float32)
        self.source_count = 0
        if not path or not os.path.isfile(path):
            print(f"The seed database does not exist ({path}); starting empty.")
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
        self.matrix = vector if not self.labels else np.vstack([self.matrix, vector])
        self.labels.append(label)

    def verify(self, embedding):
        """(label, score); inside ENROLL_MARGIN below the threshold: LABEL_UNCERTAIN."""
        if not self.labels:
            return LABEL_UNKNOWN, -1.0
        scores = self.matrix @ np.asarray(embedding, dtype=np.float32)
        best = int(np.argmax(scores))
        best_score = float(scores[best])
        if best_score >= VERIFY_THRESHOLD:
            return self.labels[best], best_score
        return (LABEL_UNCERTAIN if best_score >= VERIFY_THRESHOLD - ENROLL_MARGIN
                else LABEL_UNKNOWN), best_score

    def save(self):
        """Atomic write into save_path (the run folder), never over the seed."""
        tmp = self.save_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({label: self.matrix[i].tolist()
                       for i, label in enumerate(self.labels)}, f)
        os.replace(tmp, self.save_path)

def load_models(landmark_path, recognition_path, database_path, database_out):
    global landmark_model, recognition_model, face_database
    print("Loading the TensorRT engines...")
    landmark_model = TrtModel(landmark_path)
    recognition_model = TrtModel(recognition_path)
    print(f"Landmark model: {landmark_model.outputs[0]['sample_elems'] // 2} points")
    face_database = FaceDatabase(database_path, database_out)
    print(f"Seed database: {len(face_database)} identities "
          f"{face_database.labels if len(face_database) <= 12 else ''}")

def enroll(embedding):
    used = {int(m.group(1)) for m in
            (ENROLL_NAME_RE.match(label) for label in face_database.labels) if m}
    name = f"{ENROLL_NAME_PREFIX}_{max(used, default=0) + 1}"
    face_database.add(name, embedding)
    return name

# ============================================================
# PER-TRACK STATE
# ============================================================

class TrackState:
    """What we know about a track.

    current_label is the result of the last check (a measurement, with bad frames in it);
    identity is who we believe the person is (a conclusion, which falls only after
    IDENTITY_FLIP_CHECKS contradicting checks). identity is what gets displayed, so a
    single bad window does not erase a name we already found.
    """

    def __init__(self):
        self.last_checked_frame = self.last_candidate_frame = -999999
        self.last_check_failed, self.enrolled = False, False
        self.last_seen_frame, self.unknown_streak = 0, 0
        self.history = deque(maxlen=LABEL_HISTORY_SIZE)
        self.current_label, self.current_score = None, 0.0
        self.identity = None            # the name we believe in
        self.identity_score = 0.0       # score of the check that last confirmed it
        self.identity_frame = -999999   # when it was last confirmed
        self.contra_checks = 0          # consecutive checks contradicting it
        self.checks_confirmed = 0       # checks that supported it
        self.clear_best()
        # Best "unknown" shot of the whole track, so the enrollment prototype is chosen
        # rather than whatever the current window happened to catch.
        self.best_unknown, self.last_unknown_score = None, -1.0

    @property
    def resolved(self):
        """Has an identity. LABEL_UNKNOWN/LABEL_UNCERTAIN never get there."""
        return self.identity is not None

    @property
    def doubted(self):
        return self.identity is not None and self.contra_checks > 0

    @property
    def needs_landmarks(self):
        """Tracks without an identity (prototype hunting) and doubted ones."""
        return not self.resolved or self.doubted

    @property
    def deadline_gap(self):
        """How long this track may go unchecked. A doubted identity goes back to the normal
        pace: no point waiting the cooled interval to find out whether the tracker swapped
        our person."""
        if self.last_check_failed:
            return RETRY_INTERVAL_ON_FAIL
        if self.doubted or not self.resolved:
            return VERIFY_INTERVAL_FRAMES
        return IDENTIFIED_INTERVAL_FRAMES

    def _adopt(self, name, score, frame_number, confirmed=1):
        self.identity, self.identity_score, self.identity_frame = name, score, frame_number
        self.contra_checks, self.checks_confirmed = 0, confirmed

    def apply_decision(self, voted, score, frame_number):
        """Move a check result into what is displayed, with hysteresis.

        Returns "confirmed", "unknown", "held", "switched" or "lost".
        """
        real = voted not in (LABEL_UNKNOWN, LABEL_UNCERTAIN)

        if self.identity is None:
            self.current_label, self.current_score = voted, score
            if not real:
                return "unknown"
            self._adopt(voted, score, frame_number)
            return "confirmed"

        if real and voted == self.identity:
            self._adopt(voted, score, frame_number, self.checks_confirmed + 1)
            self.current_label, self.current_score = voted, score
            return "confirmed"

        self.contra_checks += 1
        if self.contra_checks < IDENTITY_FLIP_CHECKS:
            # Keep the supporting score too, otherwise the box would read a correct name
            # with a score contradicting it.
            self.current_label, self.current_score = self.identity, self.identity_score
            return "held"

        self.checks_confirmed = 0
        self.current_label, self.current_score = voted, score
        if real:
            self._adopt(voted, score, frame_number)
            return "switched"

        # history is cleared so the old vote does not drag over the new decisions; from
        # here the track is a candidate for enrollment again.
        self.identity, self.identity_score, self.identity_frame = None, 0.0, -999999
        self.contra_checks = 0
        self.history.clear()
        self.history.append(voted)
        return "lost"

    def confirm_enrolled(self, name, frame_number):
        """After an enrollment the identity is certain: no hysteresis."""
        self.enrolled = True
        self._adopt(name, 1.0, frame_number)
        self.current_label, self.current_score = name, 1.0
        self.history.clear()
        self.history.append(name)

    def clear_best(self):
        self.best_quality, self.best_aligned = -1.0, None
        self.best_blur, self.best_size, self.best_frontality = 0.0, 0, 0.0

    def offer(self, aligned, quality, front, blur, size):
        if quality > self.best_quality:
            self.best_quality, self.best_aligned = quality, aligned
            self.best_blur, self.best_size, self.best_frontality = blur, size, front

    def offer_unknown(self, aligned, quality, front, blur, size, frame_number):
        """The track's best shot AMONG those that can be a prototype.

        Filter first, choose after -- otherwise the winner fails the size gate later. The
        choice is on frontality, not quality: quality mixes in sharpness and size, so a
        large sharp profile would beat a smaller frontal shot, and profiles are useless for
        recognition.
        """
        if not can_be_prototype(size, blur, quality, front):
            return
        best = self.best_unknown
        if best is None or (front, quality) > (best["frontality"], best["quality"]):
            self.best_unknown = {"aligned": aligned, "quality": quality, "frontality": front,
                                 "blur": blur, "size": size, "frame": frame_number}

class FaceSample:
    """A face visible in the current frame, with everything computed for it."""

    __slots__ = ("track_id", "state", "obj_meta", "crop", "blur", "size", "box",
                 "landmarks", "five_points", "aligned", "quality", "frontality",
                 "wants_sample")

    def __init__(self, track_id, state, obj_meta, crop, blur, size, box):
        self.track_id, self.state, self.obj_meta = track_id, state, obj_meta
        self.crop, self.blur, self.size = crop, blur, size
        self.box = box                  # (x1, y1, x2, y2) in frame coordinates
        self.landmarks = self.five_points = self.aligned = None
        self.quality, self.frontality = 0.0, 0.0    # frontality = min(yaw, pitch)
        self.wants_sample = False

# ============================================================
# PROCESSING
# ============================================================

def crop_bgr(surface_rgba, box):
    """The face cut out of the mapped surface, converted to BGR.

    Only the face rectangle is converted, not the whole frame, and the result is a new
    array, so the drawing at the end of the frame never reaches the crops.
    """
    x1, y1, x2, y2 = box
    return cv2.cvtColor(surface_rgba[y1:y2, x1:x2], cv2.COLOR_RGBA2BGR)

def blur_score(img_bgr):
    return float(cv2.Laplacian(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())

def preprocess_landmark(crop):
    resized = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (112, 112))
    return np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))

def preprocess_recognition(aligned_bgr):
    rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.transpose((rgb - 127.5) / 127.5, (2, 0, 1))

LANDMARK_LAYOUTS = {
    68:  {"left_eye": range(36, 42), "right_eye": range(42, 48), "nose": 30, "mouth": (48, 54)},
    98:  {"left_eye": range(60, 68), "right_eye": range(68, 76), "nose": 54, "mouth": (76, 82)},
    106: {"left_eye": 38, "right_eye": 88, "nose": 86, "mouth": (52, 61)},
}

def get_5_points(landmark):
    """The 5 ArcFace points, whatever markup the landmark model uses."""
    layout = LANDMARK_LAYOUTS.get(landmark.shape[0])
    if layout is None:
        raise ValueError(f"The landmark model outputs {landmark.shape[0]} points and the "
                         f"mapping to the 5 ArcFace points is not defined for it. Known "
                         f"markups: {sorted(LANDMARK_LAYOUTS)}.")

    def take(index):
        return landmark[index].mean(axis=0) if isinstance(index, range) else landmark[index]

    left_mouth, right_mouth = layout["mouth"]
    return np.array([take(layout["left_eye"]), take(layout["right_eye"]),
                     take(layout["nose"]), landmark[left_mouth], landmark[right_mouth]],
                    dtype=np.float32)

def umeyama_similarity(src, dst):
    """The rotation + scale + translation taking src onto dst."""
    src, dst = np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64)
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

def align_face(crop, five_points):
    matrix = umeyama_similarity(five_points, ARCFACE_TEMPLATE)
    if matrix is None or not np.all(np.isfinite(matrix)):
        return None
    return cv2.warpAffine(crop, matrix, (ALIGN_SIZE, ALIGN_SIZE), borderValue=0)

def face_geometry(five_points):
    """The frame the pose measurements live in, or None if the eyes coincide.

    (eye vector, its unit, its normal, eye centre, interocular distance, eye->mouth
    height); the last two are what everything else divides by.
    """
    left_eye, right_eye, _, left_mouth, right_mouth = np.asarray(five_points, dtype=np.float64)
    eye_vec = right_eye - left_eye
    interocular = float(np.linalg.norm(eye_vec))
    if interocular < 1e-3:
        return None
    unit = eye_vec / interocular
    normal = np.array([-unit[1], unit[0]], dtype=np.float64)
    eye_center = (left_eye + right_eye) / 2.0
    height = float(np.dot((left_mouth + right_mouth) / 2.0 - eye_center, normal))
    return eye_vec, unit, normal, eye_center, interocular, height

def nose_ratio(five_points):
    """How far down the nose sits between the eyes and the mouth."""
    _, _, normal, eye_center, _, height = face_geometry(five_points)
    return float(np.dot(np.asarray(five_points, dtype=np.float64)[2] - eye_center,
                        normal)) / height

NOSE_RATIO_FRONTAL = nose_ratio(ARCFACE_TEMPLATE)

def clamp01(value):
    return float(min(1.0, max(0.0, value)))

def quality_and_frontality(five_points, blur, min_side):
    """(quality, frontality), both in [0, 1], or None if the points are degenerate.

    frontality = min(yaw, pitch). Roll is excluded because the alignment corrects it; yaw
    and pitch cannot be corrected. The minimum, not the average: a head turned 90 degrees
    is useless however well it sits vertically.
    """
    geometry = face_geometry(five_points)
    if geometry is None:
        return None
    eye_vec, unit, normal, eye_center, interocular, height = geometry
    if height <= 1e-3:
        return None

    nose = np.asarray(five_points, dtype=np.float64)[2] - eye_center
    roll_deg = abs(np.degrees(np.arctan2(float(eye_vec[1]), float(eye_vec[0]))))
    parts = {
        "yaw": clamp01(1.0 - abs(float(np.dot(nose, unit))) / interocular / 0.25),
        "pitch": clamp01(1.0 - abs(float(np.dot(nose, normal)) / height
                                   - NOSE_RATIO_FRONTAL) / 0.25),
        "roll": clamp01(1.0 - min(roll_deg, 180.0 - roll_deg) / 30.0),
        "sharp": clamp01(blur / (2.0 * MIN_BLUR)),
        "size": clamp01(min_side / float(ENROLL_MIN_FACE)),
    }
    return (sum(QUALITY_WEIGHTS[name] * parts[name] for name in QUALITY_WEIGHTS),
            min(parts["yaw"], parts["pitch"]))

def analyse_faces(faces):
    """Landmarks (a single GPU call) + alignment + quality, filled into the samples."""
    if not faces:
        return
    for face, raw in zip(faces, run_batched(landmark_model,
                                            [preprocess_landmark(f.crop) for f in faces])):
        h, w = face.crop.shape[:2]
        landmark = np.asarray(raw, dtype=np.float32).reshape(-1, 2) * np.array([w, h],
                                                                               dtype=np.float32)
        five = get_5_points(landmark)
        offset = np.array([face.box[0], face.box[1]], dtype=np.float32)
        face.landmarks, face.five_points = landmark + offset, five + offset

        face.aligned = align_face(face.crop, five)
        if face.aligned is not None:
            measured = quality_and_frontality(five, face.blur, face.size)
            if measured is not None:
                face.quality, face.frontality = measured

def embed_aligned(aligned_faces):
    if not aligned_faces:
        return []
    return [l2(emb) for emb in run_batched(
        recognition_model, [preprocess_recognition(a) for a in aligned_faces])]

def majority_label(history):
    """The label with the most votes; on a tie, the most recent one."""
    counts = Counter(history)
    top = max(counts.values())
    for label in reversed(history):
        if counts[label] == top:
            return label

def can_be_prototype(size, blur, quality, front):
    """Whether a shot is good enough to become somebody's database entry."""
    return (size >= ENROLL_MIN_FACE and blur >= ENROLL_MIN_BLUR
            and quality >= ENROLL_MIN_QUALITY and front >= ENROLL_MIN_FRONTALITY)

def ready_to_enroll(state):
    """Proof it is not in the database, plus a good shot to enroll it from.

    "identity is None" keeps a held identity from being enrolled a second time under a new
    name after one bad window.
    """
    return (AUTO_ENROLL and not state.enrolled and state.identity is None
            and state.best_unknown is not None
            and state.unknown_streak >= ENROLL_MIN_CHECKS
            and state.last_unknown_score < ENROLL_MAX_SCORE)

# ============================================================
# DRAWING
# ============================================================

def stamp_points(surface_rgba, points, disc, color):
    """Draw all the points at once, through a single indexed write.

    Pixels outside the frame are dropped, not clamped: the landmark model can output points
    outside the crop and clipping would pile them on the border.
    """
    if len(points) == 0:
        return
    height, width = surface_rgba.shape[:2]
    dy, dx = disc
    centres = np.rint(points).astype(np.int32)
    ys, xs = centres[:, 1, None] + dy, centres[:, 0, None] + dx
    inside = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
    if isinstance(color, np.ndarray) and color.ndim == 2:
        color = np.broadcast_to(color[:, None, :], ys.shape + (4,))[inside]
    surface_rgba[ys[inside], xs[inside]] = color

def draw_landmarks(surface_rgba, faces):
    """The facial points, straight onto the mapped surface.

    A non-writable surface is reported once and the run continues: the boxes and labels go
    through metadata and are not lost.
    """
    drawn = [face for face in faces if face.landmarks is not None]
    if not drawn:
        return
    try:
        if DRAW_ALL_LANDMARKS:
            stamp_points(surface_rgba, np.concatenate([f.landmarks for f in drawn]),
                         LANDMARK_DISC, LANDMARK_COLOR)
        stamp_points(surface_rgba, np.concatenate([f.five_points for f in drawn]),
                     FIVE_POINT_DISC, np.tile(POINT_COLORS, (len(drawn), 1)))
    except Exception as error:
        warn_once("surface", f"cannot draw on the surface ({error}); only the boxes and the "
                             f"labels remain.")

def display_for(state, frame_number, measured):
    """(text, colour, border width) for a track's box.

    measured = something was actually measured on the face in this frame. While the track
    has an identity it stays displayed; only how strongly we assert it changes: solid green
    when just confirmed, dim green with "~" from memory, yellow with "?" when stale.
    """
    if state.identity is not None:
        text = f"{state.identity} ({state.identity_score:.2f})"
        if frame_number - state.identity_frame > IDENTITY_STALE_FRAMES:
            return text + " ?", BOX_COLOR_STALE, BOX_WIDTH_HELD
        if state.doubted or not measured:
            return text + " ~", BOX_COLOR_HELD, BOX_WIDTH_HELD
        return text, BOX_COLOR_KNOWN, BOX_WIDTH_CONFIRMED

    if state.current_label is None:
        return "checking...", BOX_COLOR_PENDING, BOX_WIDTH_CONFIRMED
    color = {LABEL_UNKNOWN: BOX_COLOR_UNKNOWN,
             LABEL_UNCERTAIN: BOX_COLOR_UNCERTAIN}.get(state.current_label, BOX_COLOR_KNOWN)
    return f"{state.current_label} ({state.current_score:.2f})", color, BOX_WIDTH_CONFIRMED

def label_object(obj_meta, state, frame_number, measured=True):
    """Write the label and box colour over nvdsosd's default drawing.

    Called for every object that has state, including those that did not pass the gates
    this frame -- otherwise nvdsosd draws nvinfer's default "face <id>".
    """
    text, color, width = display_for(state, frame_number, measured)
    obj_meta.text_params.display_text = text
    obj_meta.text_params.set_bg_clr = 1
    obj_meta.text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)
    obj_meta.text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
    obj_meta.text_params.font_params.font_size = 12
    obj_meta.rect_params.border_color.set(*color)
    obj_meta.rect_params.border_width = width

def blank_object(obj_meta):
    """A detection with no state: thin grey box, no text.

    A tracker id is not an identity and has no business on screen next to real names; the
    box stays so the detection is still visible.
    """
    obj_meta.text_params.display_text = ""
    obj_meta.text_params.set_bg_clr = 0
    obj_meta.rect_params.border_color.set(*BOX_COLOR_PENDING)
    obj_meta.rect_params.border_width = BOX_WIDTH_REJECTED if DRAW_REJECTED_BOXES else 0

# ============================================================
# PROBES
# ============================================================

def iter_frames(gst_buffer):
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
    """The objects of one frame.

    The next node is read after the yield, so the caller must collect what has to be
    deleted and delete it only after the walk.
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
    """Drop boxes too small to ever be faces, right after the detector.

    NvDCF costs one correlation filter per target and most boxes on a wide frame can never
    pass the gates. Only SIZE is filtered, never confidence: a turned head keeps being
    detected at 0.3-0.4 and that weak box is all the tracker has to keep the track linked.
    Size uses the LARGER side, because a profile face narrows without losing height; the
    real gate, on min(w, h), stays in the probe.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK
    for frame_meta in iter_frames(gst_buffer):
        doomed = [obj for obj in iter_objects(frame_meta)
                  if max(obj.rect_params.width, obj.rect_params.height) < PRETRACK_MIN_SIZE]
        # After the walk: removing a node frees it, so l_obj.next would read freed memory.
        for obj_meta in doomed:
            _remove_obj_meta(frame_meta, obj_meta)
    return Gst.PadProbeReturn.OK

def media_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    for frame_meta in iter_frames(gst_buffer):
        started = time.perf_counter()
        # The surface is used in place: only the faces are cut out of it, and the drawing
        # goes back onto the same one.
        surface = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        try:
            frame_h, frame_w = surface.shape[:2]
            seen = process_frame(frame_meta, frame_meta.frame_num, frame_w, frame_h, surface)
        finally:
            if _unmap_surface is not None:
                _unmap_surface(hash(gst_buffer), frame_meta.batch_id)

        STATS["frames"] += 1
        STATS["faces"] += seen
        if PROGRESS_EVERY_FRAMES and frame_meta.frame_num % PROGRESS_EVERY_FRAMES == 0:
            print(f"  frame {frame_meta.frame_num}: {seen} faces, {len(face_database)} "
                  f"identities, {(time.perf_counter() - started) * 1000.0:.1f} ms")
    return Gst.PadProbeReturn.OK

def process_frame(frame_meta, frame_number, frame_w, frame_h, surface):
    """All the logic for one frame. Returns how many faces passed the gates."""
    if frame_number % PRUNE_CHECK_INTERVAL == 0:
        for tid in [t for t, s in track_states.items()
                    if frame_number - s.last_seen_frame > TRACK_TIMEOUT_FRAMES]:
            del track_states[tid]

    # --- Step 1: collect the visible faces, without touching the GPU ---
    # held_objects = failed the gates but belong to a track with state, so they are still
    # labelled with what we know; blank_objects = no state at all.
    faces, active, held_objects, blank_objects = [], {}, [], []
    for obj_meta in iter_objects(frame_meta):
        track_id, confidence = int(obj_meta.object_id), float(obj_meta.confidence)
        rect = obj_meta.rect_params
        x1, y1 = max(0, int(rect.left)), max(0, int(rect.top))
        x2, y2 = (min(frame_w, int(rect.left + rect.width)),
                  min(frame_h, int(rect.top + rect.height)))
        w, h = x2 - x1, y2 - y1

        known = track_states.get(track_id)
        min_confidence = TRACKED_MIN_CONFIDENCE if known is not None else MIN_CONFIDENCE
        if confidence < min_confidence or w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            # Mark a known track seen so its state (label, streak, prototype) is not
            # dropped. No new state is created here -- that would mean one TrackState per
            # background box.
            if known is not None:
                known.last_seen_frame = frame_number
                held_objects.append((known, obj_meta))
            else:
                blank_objects.append(obj_meta)
            continue

        state = known
        if state is None:
            # A new track starts from nothing; re-identification is the database's job, at
            # the first check through the recognition model.
            state = TrackState()
            track_states[track_id] = state
        state.last_seen_frame = frame_number
        active[track_id] = state

        crop = crop_bgr(surface, (x1, y1, x2, y2))
        faces.append(FaceSample(track_id, state, obj_meta, crop, blur_score(crop),
                                min(w, h), (x1, y1, x2, y2)))

    # --- Step 2: who asks for a sample in this frame (no GPU needed) ---
    for face in faces:
        state = face.state
        face.wants_sample = (
            frame_number - state.last_checked_frame >= state.deadline_gap - QUALITY_WINDOW_FRAMES
            and frame_number - state.last_candidate_frame >= CANDIDATE_INTERVAL_FRAMES)

    # --- Step 3: landmarks + alignment + quality, in a single GPU call ---
    # Not on all the faces: a track without an identity needs them every frame (that is how
    # the prototype is chosen), an already recognised one only when it gives a sample.
    analyse_faces([f for f in faces if f.wants_sample or f.state.needs_landmarks]
                  if LANDMARK_ONLY_WHEN_NEEDED else faces)

    # --- Step 4: the running best shot for enrollment ---
    for face in faces:
        if face.landmarks is not None and face.aligned is not None:
            face.state.offer_unknown(face.aligned, face.quality, face.frontality, face.blur,
                                     face.size, frame_number)

    # --- Step 5: which faces enter the contest for "best frame" ---
    for face in faces:
        if not face.wants_sample:
            continue
        face.state.last_candidate_frame = frame_number   # keep the rate even if discarded
        if face.blur >= MIN_BLUR * BLUR_REJECT_FACTOR and face.aligned is not None:
            face.state.offer(face.aligned, face.quality, face.frontality, face.blur, face.size)

    # --- Step 6: for whom we run recognition in this frame ---
    to_recognize = []
    for track_id, state in active.items():
        due = frame_number - state.last_checked_frame >= state.deadline_gap
        usable = state.best_aligned is not None and state.best_quality >= QUALITY_MIN
        if usable and (state.best_quality >= QUALITY_GOOD_ENOUGH or due):
            state.last_checked_frame, state.last_check_failed = frame_number, False
            to_recognize.append((track_id, state))
        elif due:
            state.last_checked_frame, state.last_check_failed = frame_number, True
            state.clear_best()

    # --- Step 7: the check (who it is), without enrollment ---
    checked = set()
    for (track_id, state), embedding in zip(
            to_recognize, embed_aligned([s.best_aligned for _, s in to_recognize])):
        label, score = face_database.verify(embedding)
        quality = state.best_quality
        state.clear_best()
        checked.add(track_id)

        if AUTO_ENROLL and label == LABEL_UNKNOWN and not state.enrolled:
            state.unknown_streak += 1
            state.last_unknown_score = score
        else:
            state.unknown_streak = 0

        state.history.append(label)
        voted = majority_label(state.history)
        outcome = state.apply_decision(voted, score, frame_number)

        # Printed only when something changes, not at every re-confirmation.
        if outcome == "confirmed" and state.checks_confirmed == 1:
            print(f"[ALERT] track {track_id} -> {state.identity} (score={score:.3f}, "
                  f"quality={quality:.2f}, frame={frame_number})")
        elif outcome == "held":
            print(f"[CONTINUITY] track {track_id}: the check came out '{voted}' "
                  f"(score={score:.3f}), but I stay on {state.identity} "
                  f"({state.contra_checks}/{IDENTITY_FLIP_CHECKS} contradictions, "
                  f"frame={frame_number})")
        elif outcome == "switched":
            print(f"[ALERT] track {track_id} switched person -> {state.identity} "
                  f"(score={score:.3f}, frame={frame_number})")
        elif outcome == "lost":
            print(f"[ALERT] track {track_id} lost its identity after {IDENTITY_FLIP_CHECKS} "
                  f"'{voted}' checks (frame={frame_number})")

    # --- Step 8: enrollment, decoupled from the check cadence ---
    # A track is enrolled once it has both the streak and a good shot, in any order; tied to
    # the check itself, short tracks were lost.
    ready = [(tid, st) for tid, st in active.items() if ready_to_enroll(st)]
    for (track_id, state), embedding in zip(
            ready, embed_aligned([s.best_unknown["aligned"] for _, s in ready])):
        best = state.best_unknown
        name = enroll(embedding)
        state.confirm_enrolled(name, frame_number)
        print(f"[ENROLL] track {track_id} -> {name} from frame {best['frame']} "
              f"({best['size']}px, frontality {best['frontality']:.2f}, "
              f"quality {best['quality']:.2f}); {len(face_database)} identities")

    # --- Step 9: drawing ---
    if DRAW_OVERLAY:
        for face in faces:
            label_object(face.obj_meta, face.state, frame_number,
                         measured=face.landmarks is not None)
        for state, obj_meta in held_objects:
            label_object(obj_meta, state, frame_number, measured=False)
        for obj_meta in blank_objects:
            blank_object(obj_meta)
        draw_landmarks(surface, faces)
    return len(faces)

# ============================================================
# GSTREAMER PIPELINE
# ============================================================

def make_element(factory_names, name):
    """The first available element in the list (Jetson and desktop differ)."""
    if isinstance(factory_names, str):
        factory_names = [factory_names]
    for factory in factory_names:
        element = Gst.ElementFactory.make(factory, name)
        if element:
            if len(factory_names) > 1:
                print(f"  {name}: {factory}")
            return element
    raise RuntimeError(f"None of the elements {factory_names} is available (a missing "
                       f"GStreamer plugin?).")

def make_queue(name):
    """A queue limited only by buffer count, never leaky.

    Without queues the whole pipeline runs on one thread and the probe time adds to the rest
    instead of overlapping with it. Byte and time limits are removed so they are not hit
    first; the depth stays small because every waiting buffer is an NVMM surface.
    """
    queue = make_element("queue", name)
    queue.set_property("max-size-buffers", QUEUE_SIZE)
    queue.set_property("max-size-bytes", 0)
    queue.set_property("max-size-time", 0)
    return queue

def link_chain(elements):
    """Link a linear chain, saying which pair failed."""
    for upstream, downstream in zip(elements, elements[1:]):
        if not upstream.link(downstream):
            raise RuntimeError(f"Cannot link {upstream.get_name()} to "
                               f"{downstream.get_name()} (incompatible caps?).")

def on_pad_added(decodebin, pad, target):
    """Link only uridecodebin's video pad."""
    name = (pad.get_current_caps() or pad.query_caps()).to_string()
    if not name.startswith("video/"):
        return
    sinkpad = target.get_static_pad("sink")
    if not sinkpad.is_linked() and pad.link(sinkpad) != Gst.PadLinkReturn.OK:
        print(f"Error: cannot link the source to the converter ({name.split(',')[0]}).")

# A phone filming in portrait does not rotate the pixels: the stream stays 1920x1080 and the
# orientation is only a tag, which the decoder ignores; without the correction the faces
# reach the detector lying on their side. "Rotation" always means degrees CLOCKWISE.
# nvvideoconvert flip-method: 1 = 90 counter-clockwise, 2 = 180, 3 = 90 clockwise.
ROTATION_TO_FLIP_METHOD = {90: 3, 180: 2, 270: 1}

def probe_video_info(video_path):
    """The file's resolution and orientation, read before building the pipeline.

    OpenCV only, and never a decoder that touches the GPU: anything starting a decodebin
    ends up at nvv4l2decoder, which opens its surface pool at the SOURCE resolution and eats
    the NVMM memory before the real pipeline asks for any ("NvMapMemAllocInternalTagged ...
    error 12", then CUDNN_STATUS_INTERNAL_ERROR in the detector).
    """
    capture = cv2.VideoCapture(video_path)
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # CAP_PROP_ORIENTATION_META exists only from OpenCV 4.5 on.
        prop = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
        try:
            rotation = int(round(float(capture.get(prop)) / 90.0) * 90) % 360 if prop else 0
        except (TypeError, ValueError):
            rotation = 0
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Cannot determine the resolution of {video_path}; OpenCV could "
                           f"not open it.")
    return {"width": width, "height": height, "rotation": rotation}

def even(value):
    """Odd sides break NV12 and the encoder."""
    return max(2, int(round(value / 2.0)) * 2)

def working_resolution(source, args):
    """The pipeline resolution: the source's, swapped by the rotation actually applied and
    shrunk past --max-side (at 4K the buffers before streammux exhaust the NVMM memory).

    The source resolution is kept by default: a fixed 1920x1080 squashed portrait clips, and
    the detector finds nothing on squashed faces.
    """
    width, height = source["width"], source["height"]
    if args.rotation in (90, 270):
        width, height = height, width
    longest = max(width, height)
    if args.max_side and longest > args.max_side:
        factor = args.max_side / float(longest)
        return even(width * factor), even(height * factor)
    return even(width), even(height)

def build_pipeline(video_path, output_video, args):
    pipeline = Gst.Pipeline()
    print("Creating the pipeline elements...")

    source = make_element("uridecodebin", "source")
    source.set_property("uri", Gst.filename_to_uri(video_path))
    # Without these two, uridecodebin also tries to decode the audio track ("No decoder
    # available for type audio/mpeg"). We link only the video pad anyway.
    source.set_property("caps", Gst.Caps.from_string("video/x-raw(ANY)"))
    source.set_property("expose-all-streams", False)

    # The scaling to the working resolution is requested here and not left to nvstreammux:
    # otherwise everything between the decoder and streammux allocates at the source
    # resolution (12 MB per buffer on a 4K clip, which is how the NVMM memory runs out).
    # Rotation goes before the capsfilter, so the caps carry the post-rotation sides.
    vidconv_in = make_element("nvvideoconvert", "convert-in")
    if args.rotation:
        vidconv_in.set_property("flip-method", ROTATION_TO_FLIP_METHOD[args.rotation])
        print(f"  convert-in: rotating {args.rotation} degrees")

    caps_in = make_element("capsfilter", "caps-in")
    caps_in.set_property("caps", Gst.Caps.from_string(
        f"video/x-raw(memory:NVMM), format=NV12, width={args.width}, height={args.height}"))

    streammux = make_element("nvstreammux", "stream-muxer")
    streammux.set_property('width', args.width)
    streammux.set_property('height', args.height)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 40000)
    streammux.set_property('live-source', 0)

    pgie = make_element("nvinfer", "face-detector")
    pgie.set_property('config-file-path', args.pgie_config)

    tracker = make_element("nvtracker", "tracker")
    # The tracker scales the frame to these; on a portrait frame 640x384 would squash it, so
    # the sides are swapped. Multiples of 32, as DeepStream wants.
    tracker_w, tracker_h = (640, 384) if args.width >= args.height else (384, 640)
    tracker.set_property('tracker-width', tracker_w)
    tracker.set_property('tracker-height', tracker_h)
    tracker.set_property('gpu-id', 0)
    tracker.set_property('ll-lib-file', "/opt/nvidia/deepstream/deepstream/lib/"
                                        "libnvds_nvmultiobjecttracker.so")
    tracker.set_property('ll-config-file', args.tracker_config)
    # Scale on the GPU, not the VIC: on Jetson the VIC fails at some resolutions with
    # "NvVic handle" or CMA errors. Not present in every DeepStream version.
    if tracker.find_property("compute-hw"):
        tracker.set_property('compute-hw', 1)       # 0 default, 1 GPU, 2 VIC

    vidconv_osd = make_element("nvvideoconvert", "convert-osd")
    # A queue cannot fill beyond the upstream element's pool, so the threads would block
    # each other with the default 4 buffers.
    for converter in (vidconv_in, vidconv_osd):
        if converter.find_property("output-buffers"):
            converter.set_property("output-buffers", QUEUE_SIZE + 4)

    caps_rgba = make_element("capsfilter", "caps-rgba")
    caps_rgba.set_property('caps',
                           Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))

    # Without an output video everything after the drawing disappears (NVENC and its pools,
    # two conversions, the muxer) -- the biggest thing that can be cut on a board short of
    # memory without touching recognition.
    if args.no_video:
        sink = make_element("fakesink", "null-sink")
        sink.set_property('sync', False)
        sink.set_property('async', False)
        sink.set_property('enable-last-sample', False)
        tail = [sink]
    else:
        caps_out = make_element("capsfilter", "caps-out")
        encoder = make_element(["nvv4l2h264enc", "x264enc"], "encoder")
        bitrate = int(args.width * args.height * 4)   # ~4 bits/pixel/s: 8 Mbit/s at 1080p
        if encoder.get_factory().get_name() == "nvv4l2h264enc":
            encoder.set_property('bitrate', bitrate)
            # The default profile is Baseline (no CABAC, no B-frames), which is what looks
            # soft at the same bitrate; High costs the same on NVENC.
            for prop, value in (("profile", 4), ("control-rate", 0),
                                ("peak-bitrate", int(bitrate * 1.5))):
                if encoder.find_property(prop):
                    encoder.set_property(prop, value)
            caps_out.set_property('caps', Gst.Caps.from_string(
                "video/x-raw(memory:NVMM), format=NV12"))
        else:
            # desktop fallback: software encoder, system memory
            encoder.set_property('bitrate', max(1, bitrate // 1000))
            encoder.set_property('speed-preset', 'ultrafast')
            caps_out.set_property('caps', Gst.Caps.from_string("video/x-raw, format=I420"))
        sink = make_element("filesink", "filesink")
        sink.set_property('location', output_video)
        sink.set_property('sync', False)
        sink.set_property('async', False)
        tail = [make_element("nvdsosd", "osd"), make_element("nvvideoconvert", "convert-out"),
                caps_out, encoder, make_element("h264parse", "parser"),
                make_element("qtmux", "muxer"), sink]

    # Threads: [decode -> detector -> tracker] | [probe] | [osd -> encoder]
    queue_pre = make_queue("queue-detect")
    chain = [streammux, pgie, tracker, make_queue("queue-probe"), vidconv_osd, caps_rgba,
             make_queue("queue-out")] + tail
    for element in [source, vidconv_in, caps_in, queue_pre] + chain:
        pipeline.add(element)

    # Everything after streammux is linear; the special links are the source (pad created
    # late) and the streammux input (request pad).
    source.connect("pad-added", on_pad_added, vidconv_in)
    link_chain([vidconv_in, caps_in, queue_pre])
    queue_pre.get_static_pad("src").link(streammux.get_request_pad("sink_0"))
    link_chain(chain)

    # The box filter goes on the detector's output, i.e. before the tracker.
    if PRETRACK_FILTER:
        pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER,
                                             detection_filter_probe, 0)
    caps_rgba.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, media_probe, 0)
    return pipeline

# ============================================================
# MAIN
# ============================================================

def resolve_video(name):
    """Look for the file: as given, next to the script, then in sample/."""
    candidates = [name, os.path.join(SCRIPT_DIR, name),
                  os.path.join(SCRIPT_DIR, "sample", name)]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError("Cannot find the video. I looked in:\n  " + "\n  ".join(candidates))

def resolve_config(path, what):
    """Look for a config in the current folder, then next to the script.

    path can be a list of options in order of preference. Checked now, not when nvinfer
    starts, where the error is much harder to read.
    """
    candidates = []
    for option in ([path] if isinstance(path, str) else list(path)):
        candidates += [option, os.path.join(SCRIPT_DIR, os.path.basename(option))]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(f"Cannot find {what}. I looked in:\n  " + "\n  ".join(candidates))

def patch_tracker_config(source, destination):
    """Copy the tracker config with TRACKER_OVERRIDES written into it.

    Line-based on purpose: no PyYAML dependency, and a parser round trip would lose the
    comments. Keys that are not found are reported, not added -- a key the tracker does not
    know would be ignored silently.
    """
    with open(source, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    missing = []
    for key, value in TRACKER_OVERRIDES.items():
        # "  name: value   # comment", anywhere in the file; only the value is replaced.
        pattern = re.compile(r"^([ \t]*%s[ \t]*:[ \t]*)([^\s#]+)(.*)$" % re.escape(key),
                             re.MULTILINE)
        text, count = pattern.subn(lambda m: f"{m.group(1)}{value}{m.group(3)}", text)
        if not count:
            missing.append(key)
    if missing:
        warn_once("tracker_keys", f"the tracker config has no keys {', '.join(missing)}; its "
                                  f"own values stay. Check the names with: grep -n "
                                  f"'ShadowTrack\\|minDetectorConfidence' {source}")
    with open(destination, "w", encoding="utf-8") as handle:
        handle.write(text)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the face recognition pipeline on a video file. Thresholds are "
                    "constants at the top of this file; use _15 to measure them.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add = parser.add_argument
    add("video", help="the video file name (next to the script or in sample/)")
    add("--output", default=None, help="the output folder (default: the video's name); if it "
        "exists, it is reused together with its database")
    add("--database", default=FACE_DATABASE_PATH, help="the seed database, used only when "
        "the output folder does not have one yet; it is never modified")
    add("--pgie-config", default=YOLO_CONFIG_PATH,
        help="the nvinfer config for the face detector")
    add("--tracker-config", default=TRACKER_CONFIG_PATH,
        help="the low-level config for nvtracker")
    add("--landmark-engine", default=PFLD_MODEL_PATH)
    add("--recognition-engine", default=RECOGNITION_MODEL_PATH)
    add("--max-side", type=int, default=1920, help="the maximum side of the working "
        "resolution; above it the source is shrunk keeping the aspect ratio. 0 = no limit")
    add("--rotate", default="auto", choices=["auto", "0", "90", "180", "270"],
        help="rotation applied to the source, clockwise; 'auto' follows the tag in the file")
    add("--no-video", action="store_true", help="do not write the annotated video: takes the "
        "encoder and the drawing out of the pipeline. The first thing to try when the board "
        "runs out of memory; the database comes out unchanged")
    add("--no-enroll", action="store_true", help="do not add new identities to the database")
    add("--reset-db", action="store_true", help="ignore the database in the folder and start "
        "from the --database seed (with --database '' it starts empty)")
    return parser.parse_args(argv)

def main():
    global AUTO_ENROLL, DRAW_OVERLAY, DRAW_ALL_LANDMARKS

    args = parse_args()
    AUTO_ENROLL = not args.no_enroll
    DRAW_OVERLAY = DRAW_ALL_LANDMARKS = not args.no_video
    check_pyds_api()

    video_path = resolve_video(args.video)
    args.pgie_config = resolve_config(args.pgie_config, "the nvinfer config (the detector)")
    tracker_source = resolve_config(args.tracker_config, "the nvtracker config")

    # Gst.init before any element lookup. Probing the file does not need it, it goes
    # through OpenCV.
    Gst.init(None)
    source = probe_video_info(video_path)
    args.rotation = source["rotation"] if args.rotate == "auto" else int(args.rotate)
    args.width, args.height = working_resolution(source, args)
    print(f"Source:  {source['width']}x{source['height']}, rotation {source['rotation']} deg")
    print(f"Working: {args.width}x{args.height}"
          + (f", rotating by {args.rotation} degrees" if args.rotation else ""))

    # The folder is REUSED so its database persists: that is what makes "first run enrolls,
    # second recognises" work. Only the annotated clips are numbered.
    stem = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = args.output or os.path.join(SCRIPT_DIR, stem)
    reused = os.path.isdir(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    used = [int(m.group(1)) for m in
            (re.match(rf"^{re.escape(stem)}_annotated_(\d+)\.mp4$", n)
             for n in os.listdir(output_dir)) if m]
    output_video = os.path.join(output_dir, f"{stem}_annotated_{max(used, default=0) + 1:03d}.mp4")

    # The folder's database is both the source and the destination; --database is only the
    # seed for the first run and is never modified.
    database_out = os.path.join(output_dir, "face_database.json")
    if os.path.isfile(database_out) and not args.reset_db:
        database_in = database_out
    else:
        database_in = args.database if args.database and os.path.isfile(args.database) else None

    # The patched tracker config stays in the run folder, so it is visible later exactly
    # which parameters the tracking used.
    args.tracker_config = os.path.join(output_dir, "tracker.yml")
    patch_tracker_config(tracker_source, args.tracker_config)

    print(f"Video:   {video_path}")
    print(f"Output:  {output_dir} ({'reused' if reused else 'new'})")
    load_models(args.landmark_engine, args.recognition_engine, database_in, database_out)
    if database_in == database_out:
        print("Continuing the folder's database: whatever the previous runs enrolled is "
              "recognised now.")
    identities_start = len(face_database)

    pipeline = build_pipeline(video_path, output_video, args)
    loop = GLib.MainLoop()
    failure = []    # remembered so the failure also shows up in the exit code

    def bus_call(bus, message, loop):
        if message.type == Gst.MessageType.EOS:
            print("End of file.")
            loop.quit()
        elif message.type == Gst.MessageType.WARNING:
            print("GStreamer warning: %s: %s" % message.parse_warning())
        elif message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            print(f"GStreamer error: {error}: {debug}")
            failure.append(str(error))
            loop.quit()
        return True

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    def sigint_handler(sig, frame):
        # EOS, not a brutal stop, so qtmux can close the mp4.
        print("\nStop requested; sending EOS so the file closes properly...")
        pipeline.send_event(Gst.Event.new_eos())

    signal.signal(signal.SIGINT, sigint_handler)
    print("Starting processing. Ctrl+C for a controlled stop.")
    started = time.time()
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        face_database.save()
        elapsed = time.time() - started
        rate = f" ({STATS['frames'] / elapsed:.1f} FPS)" if elapsed > 0 else ""
        print("\n--- done ---")
        print(f"  frames:     {STATS['frames']}{rate}")
        print(f"  faces:      {STATS['faces']}")
        print(f"  identities: {identities_start} -> {len(face_database)}")
        if not args.no_video:
            print(f"  {output_video}")
        print(f"  {database_out}")

    return 1 if failure else 0

if __name__ == '__main__':
    sys.exit(main())
