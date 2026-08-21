"""Video-file variant of the face recognition pipeline.

The source is a video file given as an argument; the output is a folder named after it,
holding the annotated clip, the resulting database and a per-frame log. The working
resolution comes from the file, so portrait clips and clips with a rotation tag work too
(--width/--height/--max-side override it).

The output folder is REUSED across runs on the same video: the database inside it is
loaded at startup and written back there, so the first run enrolls and the second
recognises. The files of each run are numbered, so older runs stay.

Same pipeline as _13 -- same gates, same console messages, same keys in frames_*.jsonl
and summary_*.json -- in about half the lines: the overridable thresholds are declared
once in TUNING/CONTINUITY instead of being repeated in parse_args() and
apply_thresholds(), and the explanations were shortened without dropping any reason.

Usage:
    python3 complete_pipeline_tRT_media_15.py sample_vid.mp4
    python3 complete_pipeline_tRT_media_15.py sample_vid.mp4      # second time: recognises
    python3 complete_pipeline_tRT_media_15.py sample/sample_vid.mp4 --database db.json
    python3 complete_pipeline_tRT_media_15.py sample_vid.mp4 --reset-db --overwrite

The video name is looked up as given, next to the script, then in sample/.

Two behaviours worth knowing before reading the code:

  COOLDOWN. Once a track has a confirmed identity it is re-checked every
  IDENTIFIED_INTERVAL_FRAMES instead of VERIFY_INTERVAL_FRAMES, and in between it gets no
  landmarks at all (they were 63% of the probe time). The cost: if the tracker swaps the
  person under the same id the wrong label stays until the next check, and the points are
  not drawn on recognised people. Enrollment is not affected -- a track without an
  identity still gets landmarks every frame. Turn it off with --landmark-all.

  LABEL CONTINUITY. What we know about a person (TrackState.identity) is kept apart from
  what the last check produced (the vote in history): a conclusion versus a measurement.
  An identity falls only after IDENTITY_FLIP_CHECKS consecutive contradicting checks, and
  boxes that fail the gates are still labelled with it (dim green with "~"). Without both,
  one bad window or one turned head turned "person_13" back into nvdsosd's "face 1".
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
# CONFIGURATION
# ============================================================

# Some old bindings still ask for np.bool; on a recent numpy even the check warns.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    if not hasattr(np, "bool"):
        np.bool = bool

# The indexed-binding API is the TensorRT 8.5 one (JetPack 5.x) and is deprecated there;
# in TensorRT 10 it is gone and we get an AttributeError instead.
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

# Rewritten into a copy of the tracker config kept in the run folder, so the DeepStream
# original stays untouched. minDetectorConfidence 0.0 = the tracker gets every box (weak
# boxes are what keep a track linked); maxShadowTrackAge is how many frames a track
# survives with no detection, and the NvDCF default (~30) breaks most gaps, which reach
# 233 frames in our measurements.
TRACKER_OVERRIDES = {"minDetectorConfidence": 0.0, "maxShadowTrackAge": 90}

MIN_CONFIDENCE, MIN_FACE_SIZE, MIN_BLUR = 0.5, 40, 80.0   # detection gate: score, px, sharpness
# Confidence gate for a box on a track we already know. Lower on purpose: the tracker has
# already linked it to a history, and a turned head (0.3-0.4) must not fall out entirely.
TRACKED_MIN_CONFIDENCE = 0.30
# Boxes too small to ever be usable faces are dropped right after the detector, so the
# tracker never sees them (measured: 42 boxes per frame, 4 usable). The threshold is below
# MIN_FACE_SIZE on purpose, so a face oscillating around the gate keeps its track id, and
# it does NOT look at confidence: see detection_filter_probe.
PRETRACK_FILTER, PRETRACK_SIZE_FACTOR = True, 0.8
PRETRACK_MIN_SIZE = MIN_FACE_SIZE * PRETRACK_SIZE_FACTOR   # recomputed in apply_thresholds
# Samples below MIN_BLUR still compete: on a file the Laplacian variance depends on the
# codec and on scaling, so an absolute threshold can block a track forever. Only this
# safety floor is dropped outright.
BLUR_REJECT_FACTOR = 0.25
LOG_REJECTED = False          # per-box detail of every reject; by default only counted

VERIFY_INTERVAL_FRAMES = 15   # at most this many frames between checks of a track
RETRY_INTERVAL_ON_FAIL = 7    # sooner retry when the quality gate failed
IDENTIFIED_INTERVAL_FRAMES = 30   # the cooldown; see the header
LABEL_HISTORY_SIZE = 5        # recent decisions kept for the majority vote
TRACK_TIMEOUT_FRAMES = 300    # absent frames after which a track is forgotten
PRUNE_CHECK_INTERVAL = 90     # how often we look for dead tracks
IDENTITY_FLIP_CHECKS = 2      # consecutive contradicting checks needed to drop an identity
IDENTITY_STALE_FRAMES = 150   # after this long without confirmation: yellow, with "?"

DRAW_REJECTED_BOXES = True    # stateless detections: thin grey box, no text
HIDE_REJECTED_BOXES = False   # --hide-rejected: remove them from the metadata

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

# Points are written straight into the mapped surface (unified memory on Jetson); boxes
# and labels go through metadata, so they survive even if the surface is not writable.
# Landmarks are only requested for faces that need them: those giving a sample now, and
# those whose track has no identity yet.
LANDMARK_ONLY_WHEN_NEEDED = DRAW_ALL_LANDMARKS = True
DRAW_OVERLAY = True           # off with --no-video: nobody is left to see it
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

PROGRESS_EVERY_FRAMES = 100

# ============================================================
# pyds COMPATIBILITY
# ============================================================
# Optional symbols are resolved once, here, so a missing one does not raise every frame.

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

# Filled in by load_models(): the paths depend on the arguments.
landmark_model = None
recognition_model = None
face_database = None
LANDMARK_POINTS = 0

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
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
                host[:] = 0
                device = cuda.mem_alloc(host.nbytes)
                self.bindings[i] = int(device)
                target = self.inputs if self.engine.binding_is_input(i) else self.outputs
                target.append({"index": i, "shape": shape, "sample_shape": shape[1:],
                               "sample_elems": int(np.prod(shape[1:])), "dtype": dtype,
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
        if data.ndim < 2:
            raise ValueError("infer_batch expects a leading batch dimension.")
        count = int(data.shape[0])
        if not 1 <= count <= self.max_batch:
            raise ValueError(f"Batch of {count}, the engine supports between 1 and "
                             f"{self.max_batch}.")
        if data.size != count * source["sample_elems"]:
            raise ValueError(f"Input of {data.size} values for {count} samples, the engine "
                             f"expects {count * source['sample_elems']}.")

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

GPU_STATS = {"landmark_faces": 0, "landmark_calls": 0,
             "recognition_faces": 0, "recognition_calls": 0}

def run_batched(model, tensors, counter):
    """Run the model in chunks of at most max_batch, preserving the order."""
    results = []
    for start in range(0, len(tensors), model.max_batch):
        results.extend(model.infer_batch(np.stack(tensors[start:start + model.max_batch]))[0])
        GPU_STATS[counter + "_calls"] += 1
    GPU_STATS[counter + "_faces"] += len(tensors)
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
        self.path, self.save_path = path, save_path or path
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

    def verify(self, embedding, threshold, margin):
        """(label, score); between threshold-margin and threshold: LABEL_UNCERTAIN."""
        if not self.labels:
            return LABEL_UNKNOWN, -1.0
        scores = self.matrix @ np.asarray(embedding, dtype=np.float32)
        best = int(np.argmax(scores))
        best_score = float(scores[best])
        if best_score >= threshold:
            return self.labels[best], best_score
        return (LABEL_UNCERTAIN if best_score >= threshold - margin else LABEL_UNKNOWN,
                best_score)

    def save(self):
        """Atomic write into save_path (the run folder), never over the seed."""
        tmp = self.save_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({label: self.matrix[i].tolist()
                       for i, label in enumerate(self.labels)}, f)
        os.replace(tmp, self.save_path)

def load_models(landmark_path, recognition_path, database_path, database_out):
    global landmark_model, recognition_model, face_database, LANDMARK_POINTS
    print("Loading the TensorRT engines...")
    landmark_model = TrtModel(landmark_path)
    recognition_model = TrtModel(recognition_path)
    LANDMARK_POINTS = landmark_model.outputs[0]["sample_elems"] // 2
    print(f"Landmark model: {LANDMARK_POINTS} points")
    face_database = FaceDatabase(database_path, database_out)
    print(f"Seed database: {len(face_database)} identities "
          f"{face_database.labels if len(face_database) <= 12 else ''}")

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
        self.last_seen_frame, self.unknown_streak, self.checks = 0, 0, 0
        self.history = deque(maxlen=LABEL_HISTORY_SIZE)
        self.current_label, self.current_score = None, 0.0
        self.identity = None            # the name we believe in
        self.identity_score = 0.0       # score of the check that last confirmed it
        self.identity_frame = -999999   # when it was last confirmed
        self.contra_checks = 0          # consecutive checks contradicting it
        self.checks_confirmed = 0       # checks that supported it
        self.held_frames = 0            # frames drawn from memory
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
        large sharp profile would beat a smaller frontal shot, and profiles are useless
        for recognition.
        """
        if sample_blockers(size, blur, quality, front):
            return
        best = self.best_unknown
        if best is None or (front, quality) > (best["frontality"], best["quality"]):
            self.best_unknown = {"aligned": aligned, "quality": quality, "frontality": front,
                                 "blur": blur, "size": size, "frame": frame_number}

track_states = {}
# Full history of the tracks for summary.json; track_states is pruned, this is not.
track_reports = {}

class FaceSample:
    """A face visible in the current frame, with everything computed for it."""

    __slots__ = ("track_id", "state", "obj_meta", "crop", "blur", "size", "box",
                 "landmarks", "five_points", "aligned", "quality", "frontality",
                 "wants_sample", "action")

    def __init__(self, track_id, state, obj_meta, crop, blur, size, box):
        self.track_id, self.state, self.obj_meta = track_id, state, obj_meta
        self.crop, self.blur, self.size = crop, blur, size
        self.box = box                  # (x1, y1, x2, y2) in frame coordinates
        self.landmarks = self.five_points = self.aligned = None
        self.quality, self.frontality = 0.0, 0.0    # frontality = min(yaw, pitch)
        self.wants_sample = False
        self.action = "seen"

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

def postprocess_landmark(raw, crop_shape):
    """The model's normalised output, back in crop pixels."""
    h, w = crop_shape[:2]
    points = np.asarray(raw, dtype=np.float32).reshape(-1, 2).copy()
    points[:, 0] *= w
    points[:, 1] *= h
    return points

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

def preprocess_recognition(aligned_bgr):
    rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.transpose((rgb - 127.5) / 127.5, (2, 0, 1))

def face_geometry(five_points):
    """The frame the pose measurements live in, or None if the eyes coincide.

    (eye vector, its unit, its normal, eye centre, interocular distance, eye->mouth
    height); the last two are what everything else divides by.
    """
    left_eye, right_eye, _, left_mouth, right_mouth = np.asarray(five_points,
                                                                 dtype=np.float64)
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

def quality_parts(five_points, blur, min_side):
    """The quality components, each in [0, 1]. None if the points are degenerate."""
    geometry = face_geometry(five_points)
    if geometry is None:
        return None
    eye_vec, unit, normal, eye_center, interocular, height = geometry
    if height <= 1e-3:
        return None

    nose = np.asarray(five_points, dtype=np.float64)[2] - eye_center
    roll_deg = abs(np.degrees(np.arctan2(float(eye_vec[1]), float(eye_vec[0]))))
    return {
        "yaw": clamp01(1.0 - abs(float(np.dot(nose, unit))) / interocular / 0.25),
        "pitch": clamp01(1.0 - abs(float(np.dot(nose, normal)) / height
                                   - NOSE_RATIO_FRONTAL) / 0.25),
        "roll": clamp01(1.0 - min(roll_deg, 180.0 - roll_deg) / 30.0),
        "sharp": clamp01(blur / (2.0 * MIN_BLUR)),
        "size": clamp01(min_side / float(ENROLL_MIN_FACE)),
    }

def face_quality(parts):
    """How usable the shot is, in [0, 1]."""
    return sum(QUALITY_WEIGHTS[name] * parts[name] for name in QUALITY_WEIGHTS)

def frontality(parts):
    """min(yaw, pitch): how frontal the face is.

    Roll is excluded because the alignment corrects it; yaw and pitch cannot be corrected.
    The minimum, not the average: a head turned 90 degrees is useless however well it sits
    vertically.
    """
    return min(parts["yaw"], parts["pitch"])

def analyse_faces(faces):
    """Landmarks (a single GPU call) + alignment + quality, filled into the samples."""
    if not faces:
        return
    for face, raw in zip(faces, run_batched(landmark_model,
                                            [preprocess_landmark(f.crop) for f in faces],
                                            "landmark")):
        landmark = postprocess_landmark(raw, face.crop.shape)
        five = get_5_points(landmark)
        offset = np.array([face.box[0], face.box[1]], dtype=np.float32)
        face.landmarks, face.five_points = landmark + offset, five + offset

        face.aligned = align_face(face.crop, five)
        if face.aligned is not None:
            parts = quality_parts(five, face.blur, face.size)
            if parts is not None:
                face.quality, face.frontality = face_quality(parts), frontality(parts)

def embed_aligned(aligned_faces):
    if not aligned_faces:
        return []
    return [l2(emb) for emb in run_batched(
        recognition_model, [preprocess_recognition(a) for a in aligned_faces], "recognition")]

def verify_embedding(embedding):
    return face_database.verify(embedding, VERIFY_THRESHOLD, ENROLL_MARGIN)

def enroll(embedding):
    used = {int(m.group(1)) for m in
            (ENROLL_NAME_RE.match(label) for label in face_database.labels) if m}
    name = f"{ENROLL_NAME_PREFIX}_{max(used, default=0) + 1}"
    face_database.add(name, embedding)
    print(f"[ENROLL] new identity: {name} (total {len(face_database)})")
    return name

def majority_label(history):
    """The label with the most votes; on a tie, the most recent one."""
    counts = Counter(history)
    top = max(counts.values())
    for label in reversed(history):
        if counts[label] == top:
            return label

def prototype_record(best):
    if not best:
        return None
    return {"quality": r(best["quality"]), "frontality": r(best["frontality"]),
            "blur": r(best["blur"], 1), "size": best["size"], "frame": best["frame"]}

def ready_to_enroll(state):
    """Proof it is not in the database, plus a good shot to enroll it from.

    "identity is None" keeps a held identity from being enrolled a second time under a new
    name after one bad window.
    """
    return (AUTO_ENROLL and not state.enrolled and state.identity is None
            and state.best_unknown is not None
            and state.unknown_streak >= ENROLL_MIN_CHECKS
            and state.last_unknown_score < ENROLL_MAX_SCORE)

def sample_blockers(size, blur, quality, front):
    """What stops a shot from being a prototype. Empty list = it can be one."""
    return [name for name, failed in (("size", size < ENROLL_MIN_FACE),
                                      ("blur", blur < ENROLL_MIN_BLUR),
                                      ("quality", quality < ENROLL_MIN_QUALITY),
                                      ("profile", front < ENROLL_MIN_FRONTALITY)) if failed]

def enroll_blockers(state, score, size, blur, quality, front):
    """What stops this track from being enrolled now; counted into the summary."""
    blockers = [name for name, failed in
                (("held_identity", state.identity is not None),
                 ("streak", state.unknown_streak < ENROLL_MIN_CHECKS),
                 ("score", score >= ENROLL_MAX_SCORE)) if failed]
    if state.best_unknown is None:
        blockers.extend(sample_blockers(size, blur, quality, front) or ["no_good_sample"])
    return blockers

# ============================================================
# DRAWING
# ============================================================

def stamp_points(surface_rgba, points, disc, color):
    """Draw all the points at once, through a single indexed write.

    Pixels outside the frame are dropped, not clamped: the landmark model can output
    points outside the crop and clipping would pile them on the border.
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
        warn_once("surface", f"cannot draw on the surface ({error}); only the boxes and "
                             f"the labels remain.")

def display_for(state, frame_number, measured):
    """(text, colour, border width) for a track's box.

    measured = something was actually measured on the face in this frame. While the track
    has an identity it stays displayed; only how strongly we assert it changes: solid
    green when just confirmed, dim green with "~" from memory, yellow with "?" when stale.
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
# LOG
# ============================================================

def r(value, digits=3):
    return round(float(value), digits)

class FrameLogger:
    """One JSON line per frame, in frames_<run>.jsonl."""

    def __init__(self, path):
        self.path, self.handle, self.frames = path, open(path, "w", encoding="utf-8"), 0

    def write(self, record):
        self.handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.frames += 1
        if self.frames % 50 == 0:
            self.handle.flush()

    def close(self):
        self.handle.flush()
        self.handle.close()

class Run:
    """The state of the current run: where we write, what happened so far."""

    def __init__(self, args, output_dir, video_path, paths, run_index, database_in,
                 database_out):
        self.args, self.output_dir, self.video_path = args, output_dir, video_path
        self.paths, self.run_index = paths, run_index
        self.database_in, self.database_out = database_in, database_out
        self.logger = FrameLogger(paths["frames"])
        self.started, self.failure = time.time(), None

        self.frames = self.faces_seen = self.recognitions = 0
        self.detections = self.detections_kept = self.enroll_attempts = 0
        self.enrollments = []
        self.enroll_blockers, self.stage_s = Counter(), Counter()
        # Why a due check did not happen: without it, "no sample passed the quality gate"
        # looks exactly like "no face was seen".
        self.checks_skipped = Counter()
        # Label continuity: what each check did to the displayed identity, and how many
        # boxes were drawn from memory.
        self.identity_events = Counter()
        self.held_boxes = self.stale_boxes = self.blank_boxes = 0
        # Measured distributions, so a blocker count can be read against what the footage
        # actually contained.
        self.probe_ms, self.deadline_qualities, self.rejected_sizes = [], [], []
        self.face_sizes, self.face_blurs = [], []
        self.face_qualities, self.face_frontalities = [], []

RUN = None

# ============================================================
# MAIN PROBE
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
        doomed = []
        for obj_meta in iter_objects(frame_meta):
            rect = obj_meta.rect_params
            if max(rect.width, rect.height) < PRETRACK_MIN_SIZE:
                doomed.append(obj_meta)
                RUN.rejected_sizes.append(min(rect.width, rect.height))
        RUN.detections += frame_meta.num_obj_meta
        # After the walk: removing a node frees it, so l_obj.next would read freed memory.
        for obj_meta in doomed:
            _remove_obj_meta(frame_meta, obj_meta)
        RUN.detections_kept += frame_meta.num_obj_meta
    return Gst.PadProbeReturn.OK

def media_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    for frame_meta in iter_frames(gst_buffer):
        started = time.perf_counter()
        frame_number = frame_meta.frame_num
        # The surface is used in place: only the faces are cut out of it, and the drawing
        # goes back onto the same one.
        surface = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        try:
            frame_h, frame_w = surface.shape[:2]
            record = process_frame(frame_meta, frame_number, frame_w, frame_h, surface)
        finally:
            if _unmap_surface is not None:
                _unmap_surface(hash(gst_buffer), frame_meta.batch_id)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        record["ms"] = r(elapsed_ms, 2)
        logged = time.perf_counter()
        RUN.logger.write(record)
        log_s = time.perf_counter() - logged
        RUN.stage_s["log"] += log_s
        # probe_ms includes the logging, which runs on this thread too; "ms" in the log
        # stays the processing part alone.
        RUN.probe_ms.append(elapsed_ms + log_s * 1000.0)
        RUN.frames += 1

        if PROGRESS_EVERY_FRAMES and frame_number % PROGRESS_EVERY_FRAMES == 0:
            print(f"  frame {frame_number}: {len(record['faces'])} faces, "
                  f"{len(face_database)} identities, {elapsed_ms:.1f} ms")
    return Gst.PadProbeReturn.OK

def process_frame(frame_meta, frame_number, frame_w, frame_h, surface):
    """All the logic for one frame. Returns the record for the log."""
    pts_seconds = frame_meta.buf_pts / 1e9 if frame_meta.buf_pts else 0.0
    started = time.perf_counter()

    if frame_number % PRUNE_CHECK_INTERVAL == 0:
        for tid in [tid for tid, st in track_states.items()
                    if frame_number - st.last_seen_frame > TRACK_TIMEOUT_FRAMES]:
            del track_states[tid]

    # --- Step 1: collect the visible faces, without touching the GPU ---
    # Rejects are counted per gate; --log-rejected brings back the full list. held_objects
    # = failed the gates but belong to a track with state, so they are still labelled with
    # what we know; blank_objects = no state at all.
    faces, active, held_objects, blank_objects = [], {}, [], []
    rejected = Counter()
    rejected_detail = [] if LOG_REJECTED else None

    for obj_meta in iter_objects(frame_meta):
        track_id, confidence = int(obj_meta.object_id), float(obj_meta.confidence)
        rect = obj_meta.rect_params
        x1, y1 = max(0, int(rect.left)), max(0, int(rect.top))
        x2, y2 = (min(frame_w, int(rect.left + rect.width)),
                  min(frame_h, int(rect.top + rect.height)))
        w, h = x2 - x1, y2 - y1

        known = track_states.get(track_id)
        min_confidence = TRACKED_MIN_CONFIDENCE if known is not None else MIN_CONFIDENCE
        if confidence < min_confidence:
            gate = "low_confidence"
        elif w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            gate = "too_small"
        else:
            gate = None

        if gate is not None:
            rejected[gate] += 1
            if gate == "too_small":
                RUN.rejected_sizes.append(min(w, h))
            if rejected_detail is not None:
                rejected_detail.append({"track": track_id, "box": [x1, y1, x2, y2],
                                        "det": r(confidence), "gate": gate})
            # Mark a known track seen so its state (label, streak, prototype) is not
            # dropped. No new state is created here -- that would mean one TrackState per
            # background box.
            if known is not None:
                known.last_seen_frame = frame_number
                known.held_frames += 1
                held_objects.append((track_id, known, obj_meta, gate, (x1, y1, x2, y2)))
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

        report = track_reports.setdefault(track_id, {"first_frame": frame_number,
                                                     "checks": 0, "label": None,
                                                     "score": 0.0, "enrolled": False,
                                                     "max_size": 0})
        report["last_frame"] = frame_number
        # Biggest the face ever got: shows which tracks were cut by ENROLL_MIN_FACE alone.
        report["max_size"] = max(report["max_size"], min(w, h))

        crop = crop_bgr(surface, (x1, y1, x2, y2))
        faces.append(FaceSample(track_id, state, obj_meta, crop, blur_score(crop),
                                min(w, h), (x1, y1, x2, y2)))

    RUN.faces_seen += len(faces)
    RUN.stage_s["read"] += time.perf_counter() - started

    # --- Step 2: who asks for a sample in this frame (no GPU needed) ---
    for face in faces:
        state = face.state
        face.wants_sample = (
            frame_number - state.last_checked_frame >= state.deadline_gap - QUALITY_WINDOW_FRAMES
            and frame_number - state.last_candidate_frame >= CANDIDATE_INTERVAL_FRAMES)

    # --- Step 3: landmarks + alignment + quality, in a single GPU call ---
    # Not on all the faces: a track without an identity needs them every frame (that is how
    # the prototype is chosen), an already recognised one only when it gives a sample.
    started = time.perf_counter()
    analyse_faces([f for f in faces if f.wants_sample or f.state.needs_landmarks]
                  if LANDMARK_ONLY_WHEN_NEEDED else faces)
    RUN.stage_s["landmark"] += time.perf_counter() - started

    # --- Step 4: statistics + the prototype ---
    for face in faces:
        RUN.face_sizes.append(face.size)
        RUN.face_blurs.append(face.blur)
        if face.landmarks is None:
            # No landmarks means no quality and no frontality; they must not enter the
            # distributions as zeros.
            face.action = "tracked"
            continue
        RUN.face_qualities.append(face.quality)
        RUN.face_frontalities.append(face.frontality)
        if face.aligned is not None:
            face.state.offer_unknown(face.aligned, face.quality, face.frontality, face.blur,
                                     face.size, frame_number)

    # --- Step 5: which faces enter the contest for "best frame" ---
    for face in faces:
        if not face.wants_sample:
            continue
        face.state.last_candidate_frame = frame_number   # keep the rate even if discarded
        if face.blur < MIN_BLUR * BLUR_REJECT_FACTOR:
            face.action = "skipped_blur"
        elif face.aligned is None:
            face.action = "align_failed"
        else:
            face.action = "sample" if face.blur >= MIN_BLUR else "sample_blurry"
            face.state.offer(face.aligned, face.quality, face.frontality, face.blur,
                             face.size)

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
            if state.best_aligned is None:
                RUN.checks_skipped["no_sample"] += 1
            else:
                RUN.checks_skipped["quality_below_QUALITY_MIN"] += 1
                RUN.deadline_qualities.append(state.best_quality)
            state.clear_best()

    # --- Step 7: the check (who it is), without enrollment ---
    started = time.perf_counter()
    embeddings = embed_aligned([state.best_aligned for _, state in to_recognize])
    RUN.stage_s["recognition"] += time.perf_counter() - started

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
                # In the uncertainty band around somebody already known: not enrolled (it
                # would duplicate an identity) but counted -- if most blockers land here,
                # the problem is VERIFY_THRESHOLD/ENROLL_MARGIN, not a quality gate.
                RUN.enroll_blockers["uncertain"] += 1
                RUN.enroll_attempts += 1
            state.unknown_streak = 0

        state.history.append(label)
        voted = majority_label(state.history)
        outcome = state.apply_decision(voted, score, frame_number)
        RUN.identity_events[outcome] += 1

        report = track_reports[track_id]
        report["checks"], report["label"] = state.checks, state.current_label
        report["identity"], report["score"] = state.identity, r(state.current_score)

        decisions[track_id] = {"raw_label": label, "vote": voted, "effect": outcome,
                               "score": r(score), "quality": r(quality),
                               "frontality": r(front), "blur": r(blur, 1), "size": size,
                               "streak": state.unknown_streak, "enrolled": False}
        if outcome == "held":
            decisions[track_id]["held_identity"] = state.identity
            decisions[track_id]["contradictions"] = state.contra_checks
        if blockers:
            decisions[track_id]["blockers"] = blockers
            decisions[track_id]["best_sample"] = prototype_record(state.best_unknown)

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
    # A track is enrolled once it has both the streak and a good shot, in any order; tied
    # to the check itself, short tracks were lost.
    ready = [(tid, st) for tid, st in active.items() if ready_to_enroll(st)]
    for (track_id, state), embedding in zip(ready, embed_aligned(
            [state.best_unknown["aligned"] for _, state in ready])):
        best = state.best_unknown
        name = enroll(embedding)
        state.confirm_enrolled(name, frame_number)
        RUN.identity_events["enrolled"] += 1

        track_reports[track_id].update({"label": name, "identity": name, "enrolled": True})
        RUN.enrollments.append(dict({"name": name, "frame": frame_number,
                                     "track": track_id}, **prototype_record(best)))
        decisions.setdefault(track_id, {}).update(
            {"raw_label": name, "enrolled": True, "score": 1.0,
             "streak": state.unknown_streak, "best_sample": prototype_record(best)})
        print(f"[ENROLL] track {track_id} -> {name} from frame {best['frame']} "
              f"({best['size']}px, frontality {best['frontality']:.2f}, "
              f"quality {best['quality']:.2f})")

    # --- Step 9: drawing ---
    for face in faces:
        if face.track_id in decisions:
            face.action = "recognized"
    # Held boxes are counted even with no drawing, so --no-video gives the same summary.
    for _, state, _, _, _ in held_objects:
        RUN.held_boxes += 1
        if frame_number - state.identity_frame > IDENTITY_STALE_FRAMES:
            RUN.stale_boxes += 1
    RUN.blank_boxes += len(blank_objects)

    if DRAW_OVERLAY:
        started = time.perf_counter()
        for face in faces:
            label_object(face.obj_meta, face.state, frame_number,
                         measured=face.landmarks is not None)
        for _, state, obj_meta, _, _ in held_objects:
            label_object(obj_meta, state, frame_number, measured=False)
        for obj_meta in blank_objects:
            blank_object(obj_meta)
        draw_landmarks(surface, faces)
        RUN.stage_s["draw"] += time.perf_counter() - started

    # After the walk and after the drawing: removing a node frees it.
    if HIDE_REJECTED_BOXES and _remove_obj_meta is not None:
        for obj_meta in blank_objects:
            _remove_obj_meta(frame_meta, obj_meta)

    # --- Step 10: the record for the log ---
    face_records = []
    for face in faces:
        state = face.state
        entry = {"track": face.track_id, "box": [int(v) for v in face.box],
                 "det": r(float(face.obj_meta.confidence)), "blur": r(face.blur, 1),
                 "action": face.action, "label": state.current_label,
                 "score": r(state.current_score), "identity": state.identity}
        if state.doubted:
            entry["contradictions"] = state.contra_checks
        if face.landmarks is not None:
            entry["quality"], entry["frontality"] = r(face.quality), r(face.frontality)
            entry["points5"] = [[r(x, 1), r(y, 1)] for x, y in face.five_points]
        if face.track_id in decisions:
            entry["decision"] = decisions[face.track_id]
        face_records.append(entry)

    record = {"frame": frame_number, "time": r(pts_seconds), "faces": face_records,
              "rejected": dict(rejected),
              "gpu": {"landmark": len(faces), "recognition": len(to_recognize)},
              "identities": len(face_database)}
    # Boxes that failed the gates but carry a known identity stay out of "faces" (nothing
    # was measured on them) and must still show up in the log.
    if held_objects:
        record["held"] = [{"track": tid, "box": [int(v) for v in box], "gate": gate,
                           "identity": st.identity, "score": r(st.identity_score),
                           "age_frames": frame_number - st.identity_frame}
                          for tid, st, _, gate, box in held_objects]
    if blank_objects:
        record["anonymous"] = len(blank_objects)
    if rejected_detail:
        record["rejected_detail"] = rejected_detail
    return record

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

def make_queue(name, depth):
    """A queue limited only by buffer count, never leaky.

    Without queues the whole pipeline runs on one thread and the probe time adds to the
    rest instead of overlapping with it. Byte and time limits are removed so they are not
    hit first; the depth stays small because every waiting buffer is an NVMM surface.
    """
    queue = make_element("queue", name)
    queue.set_property("max-size-buffers", depth)
    queue.set_property("max-size-bytes", 0)
    queue.set_property("max-size-time", 0)
    return queue

def link_chain(elements):
    """Link a linear chain, saying which pair failed."""
    for upstream, downstream in zip(elements, elements[1:]):
        if not upstream.link(downstream):
            raise RuntimeError(f"Cannot link {upstream.get_name()} to "
                               f"{downstream.get_name()} (incompatible caps?).")

def configure_encoder(encoder, caps_out, bitrate, iframe_interval=None):
    """H.264 settings, on NVENC or on the desktop fallback."""
    if encoder.get_factory().get_name() == "nvv4l2h264enc":
        encoder.set_property('bitrate', bitrate)
        # The default profile is Baseline (no CABAC, no B-frames), which is what looks soft
        # at the same bitrate; High costs the same on NVENC.
        for prop, value in (("profile", 4),         # 0 Baseline, 2 Main, 4 High
                            ("control-rate", 0),    # 0 variable, 1 constant
                            ("peak-bitrate", int(bitrate * 1.5)),
                            ("iframeinterval", iframe_interval)):
            if value is not None and encoder.find_property(prop):
                encoder.set_property(prop, value)
        caps_out.set_property('caps',
                              Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"))
    else:
        # desktop fallback: software encoder, system memory
        encoder.set_property('bitrate', max(1, bitrate // 1000))
        encoder.set_property('speed-preset', 'ultrafast')
        caps_out.set_property('caps', Gst.Caps.from_string("video/x-raw, format=I420"))

def on_child_added(child_proxy, element, name, args):
    """Descend through decodebin to the hardware decoder and trim its pool.

    The elements appear as decodebin figures out the file, so the bins created along the
    way have to be followed too.
    """
    if "decodebin" in name:
        element.connect("child-added", on_child_added, args)
    elif "nvv4l2decoder" in name and element.find_property("num-extra-surfaces"):
        element.set_property("num-extra-surfaces", args.decoder_surfaces)
        print(f"  {name}: num-extra-surfaces={args.decoder_surfaces}")

def on_pad_added(decodebin, pad, target):
    """Link only uridecodebin's video pad."""
    name = (pad.get_current_caps() or pad.query_caps()).to_string()
    if not name.startswith("video/"):
        return
    sinkpad = target.get_static_pad("sink")
    if sinkpad.is_linked():
        return
    if pad.link(sinkpad) != Gst.PadLinkReturn.OK:
        print(f"Error: cannot link the source to the converter ({name.split(',')[0]}).")

# A phone filming in portrait does not rotate the pixels: the stream stays 1920x1080 and
# the orientation is only a tag, which the decoder ignores; without the correction the
# faces reach the detector lying on their side. "Rotation" always means degrees CLOCKWISE.
# nvvideoconvert flip-method: 1 = 90 counter-clockwise, 2 = 180, 3 = 90 clockwise.
ROTATION_TO_FLIP_METHOD = {90: 3, 180: 2, 270: 1}

def normalise_rotation(degrees):
    """Round to the nearest multiple of 90 and bring into [0, 360)."""
    try:
        return int(round(float(degrees) / 90.0) * 90) % 360
    except (TypeError, ValueError):
        return 0

def probe_video_info(video_path):
    """The file's resolution and orientation, read before building the pipeline.

    OpenCV only, and never a decoder that touches the GPU: anything starting a decodebin
    ends up at nvv4l2decoder, which opens its surface pool at the SOURCE resolution and
    eats the NVMM memory before the real pipeline asks for any ("NvMapMemAllocInternal-
    Tagged ... error 12", then CUDNN_STATUS_INTERNAL_ERROR in the detector).

    _11 tried ffprobe first; that path is gone because it never worked here (the binary is
    linked against libraries the container does not have). If OpenCV cannot open the file
    either, --width/--height and --rotate give the answers by hand.
    """
    capture = cv2.VideoCapture(video_path)
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # CAP_PROP_ORIENTATION_META exists only from OpenCV 4.5 on.
        prop = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
        rotation = normalise_rotation(capture.get(prop)) if prop is not None else 0
    finally:
        capture.release()

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Cannot determine the resolution of {video_path}. Give it "
                           f"directly, with --width and --height.")
    return {"width": width, "height": height, "rotation": rotation, "read_with": "OpenCV"}

def rotation_supported():
    """Can nvvideoconvert rotate? (flip-method is not on every platform)"""
    element = Gst.ElementFactory.make("nvvideoconvert", None)
    return element is not None and element.find_property("flip-method") is not None

def even(value):
    """Odd sides break NV12 and the encoder."""
    return max(2, int(round(value / 2.0)) * 2)

def auto_bitrate(width, height):
    """~4 bits per pixel per second: 8 Mbit/s at 1080p, 33 at 4K."""
    return int(width * height * 4)

def working_resolution(source, args):
    """The resolution the pipeline runs at, starting from the source's.

    The source resolution is kept by default and only shrunk past --max-side; a fixed
    1920x1080 squashed portrait clips, and the detector finds nothing on squashed faces.
    The sides are swapped by the rotation ACTUALLY applied, not by the tag, so an
    unrotated frame is not squashed either.
    """
    width, height = source["width"], source["height"]
    if args.rotation in (90, 270):
        width, height = height, width

    if args.width and args.height:
        given, actual = args.width / float(args.height), width / float(height)
        if abs(given - actual) / actual > 0.01:
            print(f"[WARNING] --width/--height ({args.width}x{args.height}) have a different "
                  f"aspect ratio than the source ({width}x{height}); the image will be "
                  f"distorted and the detection will suffer.")
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
        raise RuntimeError("The pipeline could not be created.")
    print("Creating the pipeline elements...")

    source = make_element("uridecodebin", "source")
    source.set_property("uri", Gst.filename_to_uri(video_path))
    # Without these two, uridecodebin also tries to decode the audio track ("No decoder
    # available for type audio/mpeg"). We link only the video pad anyway.
    source.set_property("caps", Gst.Caps.from_string("video/x-raw(ANY)"))
    source.set_property("expose-all-streams", False)
    source.connect("child-added", on_child_added, args)

    # The scaling to the working resolution is requested here and not left to nvstreammux:
    # otherwise everything between the decoder and streammux allocates at the source
    # resolution (12 MB per buffer on a 4K clip, which is how the NVMM memory runs out).
    vidconv_in = make_element("nvvideoconvert", "convert-in")
    # Rotation before the capsfilter, so the caps already carry the post-rotation sides.
    if args.rotation:
        vidconv_in.set_property("flip-method", ROTATION_TO_FLIP_METHOD[args.rotation])
        print(f"  convert-in: rotating {args.rotation} degrees "
              f"(flip-method={ROTATION_TO_FLIP_METHOD[args.rotation]})")

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
    # The tracker scales the frame to these; on a portrait frame 640x384 would squash it,
    # so the sides are swapped. Multiples of 32, as DeepStream wants.
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
        print("  tracker: compute-hw=1 (GPU)")

    vidconv_osd = make_element("nvvideoconvert", "convert-osd")
    # A queue cannot fill beyond the upstream element's pool, so the threads would block
    # each other with the default 4 buffers.
    for converter in (vidconv_in, vidconv_osd):
        if converter.find_property("output-buffers"):
            converter.set_property("output-buffers", args.queue_size + 4)

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
        bitrate = args.bitrate or auto_bitrate(args.width, args.height)
        print(f"  bitrate: {bitrate / 1e6:.1f} Mbit/s"
              + ("" if args.bitrate else " (computed from the resolution)"))
        configure_encoder(encoder, caps_out, bitrate)
        sink = make_element("filesink", "filesink")
        sink.set_property('location', output_video)
        sink.set_property('sync', False)
        sink.set_property('async', False)
        tail = [make_element("nvdsosd", "osd"), make_element("nvvideoconvert", "convert-out"),
                caps_out, encoder, make_element("h264parse", "parser"),
                make_element("qtmux", "muxer"), sink]

    # Threads: [decode -> detector -> tracker] | [probe] | [osd -> encoder]
    queue_pre = make_queue("queue-detect", args.queue_size)
    # Everything after streammux is linear; the special links are the source (pad created
    # late) and the streammux input (request pad).
    chain = [streammux, pgie, tracker, make_queue("queue-probe", args.queue_size),
             vidconv_osd, caps_rgba, make_queue("queue-out", args.queue_size)] + tail
    for element in [source, vidconv_in, caps_in, queue_pre] + chain:
        pipeline.add(element)

    print("Linking the pipeline elements"
          + (" (without an output video)..." if args.no_video else "..."))
    source.connect("pad-added", on_pad_added, vidconv_in)
    link_chain([vidconv_in, caps_in, queue_pre])
    queue_pre.get_static_pad("src").link(streammux.get_request_pad("sink_0"))
    link_chain(chain)

    # The box filter goes on the detector's output, i.e. before the tracker.
    if PRETRACK_FILTER:
        pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER,
                                             detection_filter_probe, 0)
        print(f"  pre-tracker filter: boxes with both sides below {PRETRACK_MIN_SIZE:.0f} px "
              f"(confidence is NOT filtered here)")
    caps_rgba.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, media_probe, 0)
    return pipeline

# ============================================================
# FINAL REPORT
# ============================================================

def percentile(ordered, fraction):
    """A percentile of an already sorted list."""
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]

def distribution(values, digits=1):
    if not values:
        return {}
    ordered = sorted(values)
    return {"min": r(ordered[0], digits), "p10": r(percentile(ordered, 0.10), digits),
            "median": r(percentile(ordered, 0.50), digits),
            "p90": r(percentile(ordered, 0.90), digits),
            "max": r(ordered[-1], digits), "n": len(ordered)}

def tracks_above_size():
    """How many tracks reach each size threshold, around ENROLL_MIN_FACE.

    Faces above the threshold can all belong to one person; what matters is how many
    different people had at least one big enough frame.
    """
    reached = [report.get("max_size", 0) for report in track_reports.values()]
    if not reached:
        return {}
    steps = sorted({max(MIN_FACE_SIZE, ENROLL_MIN_FACE + delta)
                    for delta in (-10, -8, -5, -2, 0, 5, 10)})
    return {str(step): sum(1 for size in reached if size >= step) for step in steps}

def write_summary(run):
    args = run.args
    elapsed = time.time() - run.started
    durations = sorted(run.probe_ms)
    identities = {}
    for track_id, report in track_reports.items():
        identities.setdefault(report["label"] or "no_decision", []).append(track_id)
    source = args.source_info

    summary = {
        "video": run.video_path, "folder": run.output_dir, "run": run.run_index,
        "error": run.failure,
        "source": {"resolution": f"{source['width']}x{source['height']}",
                   "rotation_tag": source["rotation"], "rotation_applied": args.rotation,
                   "working_resolution": f"{args.width}x{args.height}",
                   "read_with": source["read_with"],
                   "annotated_video": None if args.no_video else run.paths["video"],
                   "free_memory_mb": {"start": args.free_memory_mb,
                                      "end": available_memory_mb()[0]}},
        "frames": run.frames, "faces_processed": run.faces_seen,
        "recognitions": run.recognitions, "run_duration_s": r(elapsed, 1),
        # With queues the probe runs in parallel with the rest, so probe_of_total can
        # exceed 100%: it only means the probe is the long part.
        "speed": {"fps": r(run.frames / elapsed, 2) if elapsed > 0 else 0.0,
                  "ms_per_frame": r(1000.0 * elapsed / run.frames, 2) if run.frames else 0.0,
                  "probe_of_total_%": (r(100.0 * sum(run.probe_ms) / (1000.0 * elapsed), 1)
                                       if elapsed > 0 else 0.0),
                  "queue_size": args.queue_size},
        "detections": {"raw": run.detections, "to_tracker": run.detections_kept,
                       "dropped": run.detections - run.detections_kept,
                       "filter_active": PRETRACK_FILTER,
                       "threshold_px": r(PRETRACK_MIN_SIZE, 1),
                       "filters_confidence": False},  # deliberately: see the filter probe
        "tracker": {"config_source": os.path.basename(args.tracker_source),
                    "config_used": os.path.basename(args.tracker_config),
                    "overrides": args.tracker_overrides},
        "stage_time_ms": {name: r(1000.0 * total / run.frames, 2)
                          for name, total in run.stage_s.most_common()} if run.frames else {},
        "database": {"start": run.database_in, "written_to": run.database_out,
                     "continued": run.database_in == run.database_out,
                     "identities_start": face_database.source_count,
                     "identities_end": len(face_database),
                     "enrollments": run.enrollments},
        # If the run ends with zero identities, the largest cause here is the one blocking.
        "enroll_blockers": {"attempts": run.enroll_attempts,
                            "causes": dict(run.enroll_blockers.most_common()),
                            "tracks_above_size_threshold": tracks_above_size()},
        # "held" counts the checks that would have erased a known name; "lost" and
        # "switched" are the cases where the person really changed under the same id -- if
        # those are zero and "held" is large, IDENTITY_FLIP_CHECKS can go up, and the other
        # way round.
        "continuity": {"checks": dict(run.identity_events.most_common()),
                       "held_boxes": run.held_boxes, "stale_boxes": run.stale_boxes,
                       "anonymous_boxes": run.blank_boxes,
                       "thresholds": {"IDENTITY_FLIP_CHECKS": IDENTITY_FLIP_CHECKS,
                                      "IDENTITY_STALE_FRAMES": IDENTITY_STALE_FRAMES,
                                      "TRACKED_MIN_CONFIDENCE": TRACKED_MIN_CONFIDENCE}},
        # Large numbers here with empty enroll_blockers mean the faces never reached the
        # recognition model at all.
        "checks_skipped": {"causes": dict(run.checks_skipped.most_common()),
                           "quality_at_deadline": distribution(run.deadline_qualities, 3)},
        "gpu": dict(GPU_STATS),
        "calls_saved": {
            "landmark": GPU_STATS["landmark_faces"] - GPU_STATS["landmark_calls"],
            "recognition": GPU_STATS["recognition_faces"] - GPU_STATS["recognition_calls"],
            # Faces whose track already had an identity and gave no sample.
            "faces_only_tracked": run.faces_seen - GPU_STATS["landmark_faces"]},
        "probe_time_ms": {"mean": r(sum(durations) / len(durations), 2) if durations else 0.0,
                          "p50": r(percentile(durations, 0.50), 2),
                          "p95": r(percentile(durations, 0.95), 2),
                          "max": r(durations[-1], 2) if durations else 0.0},
        "tracks": track_reports,
        "identities": {label: sorted(tracks) for label, tracks in identities.items()},
        # The thresholds are in pixels at the working resolution, so they can only be
        # judged against these.
        "distributions": {"size_px": distribution(run.face_sizes, 0),
                          "blur": distribution(run.face_blurs, 1),
                          "quality": distribution(run.face_qualities, 3),
                          "frontality": distribution(run.face_frontalities, 3),
                          "rejected_size_px": distribution(run.rejected_sizes, 0)},
        "thresholds": dict(
            {"working_resolution": f"{args.width}x{args.height}",
             "MIN_CONFIDENCE": MIN_CONFIDENCE, "BLUR_REJECT_FACTOR": BLUR_REJECT_FACTOR,
             "ENROLL_MARGIN": ENROLL_MARGIN, "ENROLL_MAX_SCORE": ENROLL_MAX_SCORE,
             "QUALITY_GOOD_ENOUGH": QUALITY_GOOD_ENOUGH,
             "LANDMARK_ONLY_WHEN_NEEDED": LANDMARK_ONLY_WHEN_NEEDED},
            **{name: globals()[name] for _, name, _, _ in TUNING}),
    }
    with open(run.paths["summary"], "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary

def print_report(summary, run):
    """The end-of-run console report."""
    print("\n--- done ---")
    print(f"  frames processed:  {summary['frames']}")
    print(f"  faces processed:   {summary['faces_processed']}")
    print(f"  recognitions:      {summary['recognitions']}")
    print(f"  identities:        {summary['database']['identities_start']} "
          f"-> {summary['database']['identities_end']}")

    cont = summary["continuity"]
    if cont["checks"] or cont["held_boxes"]:
        effects = ", ".join(f"{name} x{count}" for name, count in cont["checks"].items())
        print(f"  continuity:        {effects or 'no checks'}")
        print(f"    boxes carried from memory: {cont['held_boxes']} ({cont['stale_boxes']} "
              f"over {IDENTITY_STALE_FRAMES} frames without a confirmation), "
              f"{cont['anonymous_boxes']} drawn anonymously")

    # If nobody entered the database, say which threshold stood in the way.
    blockers = summary["enroll_blockers"]
    if blockers["causes"]:
        causes = ", ".join(f"{n} x{c}" for n, c in blockers["causes"].items())
        print(f"  enrollments blocked: {blockers['attempts']} attempts -> {causes}")
        for name, key, threshold in (("size", "size_px", ENROLL_MIN_FACE),
                                     ("blur", "blur", ENROLL_MIN_BLUR),
                                     ("quality", "quality", ENROLL_MIN_QUALITY),
                                     ("profile", "frontality", ENROLL_MIN_FRONTALITY)):
            measured = summary["distributions"][key]
            if name in blockers["causes"] and measured:
                print(f"    {name}: measured {measured['min']}-{measured['max']} "
                      f"(median {measured['median']}), threshold {threshold}")
        scale = blockers["tracks_above_size_threshold"]
        if "size" in blockers["causes"] and scale:
            points = ", ".join(f"{step}px -> {count}" for step, count in scale.items())
            print(f"    tracks reaching the threshold: {points} (--enroll-min-face)")
    elif summary["faces_processed"] == 0:
        print("  (no face passed the gates: see 'rejected' in the log)")

    skipped = summary["checks_skipped"]["causes"]
    if skipped:
        print("  checks skipped:    " + ", ".join(f"{n} x{c}" for n, c in skipped.items()))

    speed = summary["speed"]
    print(f"  speed:             {speed['fps']} FPS ({speed['ms_per_frame']} ms/frame)")
    print(f"  probe time/frame:  {summary['probe_time_ms']['mean']} ms "
          f"(p95 {summary['probe_time_ms']['p95']} ms) = {speed['probe_of_total_%']}% "
          f"of the run")
    if summary["stage_time_ms"]:
        print("    of which:        "
              + ", ".join(f"{n} {v}" for n, v in summary["stage_time_ms"].items())
              + " (ms/frame)")
    only_tracked = summary["calls_saved"]["faces_only_tracked"]
    if summary["faces_processed"]:
        print(f"  landmarks:         {summary['gpu']['landmark_faces']} faces out of "
              f"{summary['faces_processed']} ({only_tracked} only tracked, "
              f"{100 * only_tracked / summary['faces_processed']:.0f}% saved)")

    # The detector may have run on frames that never reached the probe, so "frames" can be
    # 0 with non-zero detections.
    det = summary["detections"]
    if det["raw"] and summary["frames"]:
        print(f"  detections:        {det['raw'] / summary['frames']:.1f}/frame, to the "
              f"tracker {det['to_tracker'] / summary['frames']:.1f}/frame")
    elif det["raw"]:
        print(f"  detections:        {det['raw']} raw, {det['to_tracker']} to the tracker "
              f"(no frame reached the probe)")

    free_now, _ = available_memory_mb()
    if free_now is not None:
        print(f"  free memory:       {free_now} MB")
    print("")
    if not run.args.no_video:
        print(f"  {run.paths['video']}")
    print(f"  {run.database_out}")
    print(f"  {run.paths['frames']}")
    print(f"  {run.paths['summary']}")

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

# "  name: value   # comment", anywhere in the file; only the value is replaced.
TRACKER_KEY_RE = "^([ \t]*{key}[ \t]*:[ \t]*)([^\\s#]+)(.*)$"

def patch_tracker_config(source, overrides, destination):
    """Write a copy of the tracker config with the given keys rewritten.

    Line-based on purpose: no PyYAML dependency, and a parser round trip would lose the
    comments. Keys that are not found are reported, not added -- a key the tracker does not
    know would be ignored silently.
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
        warn_once("tracker_keys", f"the tracker config has no keys {', '.join(missing)}; its "
                                  f"own values stay. Check the names with: grep -n "
                                  f"'ShadowTrack\\|minDetectorConfidence' {source}")
    with open(destination, "w", encoding="utf-8") as handle:
        handle.write(text)
    return applied

def apply_tracker_overrides(args, output_dir, run_index):
    """Put our tracker values into a copy kept in the run folder.

    Keeping the copy means it is visible later exactly which parameters the tracking used,
    and the DeepStream original is never touched.
    """
    args.tracker_overrides, args.tracker_source = {}, args.tracker_config
    if args.no_tracker_patch:
        return
    overrides = dict(TRACKER_OVERRIDES)
    if args.shadow_track_age is not None:
        overrides["maxShadowTrackAge"] = args.shadow_track_age
    if args.min_detector_confidence is not None:
        overrides["minDetectorConfidence"] = args.min_detector_confidence

    patched = os.path.join(output_dir, f"tracker_{run_index:03d}.yml")
    args.tracker_overrides = patch_tracker_config(args.tracker_config, overrides, patched)
    if args.tracker_overrides:
        values = ", ".join(f"{k}={v}" for k, v in args.tracker_overrides.items())
        print(f"Tracker: {os.path.basename(args.tracker_config)} -> {values}")
    args.tracker_config = patched

RUN_INDEX_RE = re.compile(r"^summary_(\d+)\.json$")

def prepare_output_dir(stem, args):
    """The video's folder, REUSED across runs so its database persists.

    That is what makes "first run enrolls, second recognises" work; the run's files are
    numbered instead (see run_paths), so nothing is lost.
    """
    base = args.output or os.path.join(SCRIPT_DIR, stem)
    if args.new_dir and os.path.exists(base):
        index = 2
        while os.path.exists(f"{base}_{index}"):
            index += 1
        base = f"{base}_{index}"
        print(f"--new-dir: starting clean, in {base}.")
    existed = os.path.isdir(base)
    os.makedirs(base, exist_ok=True)
    return base, existed

def run_paths(output_dir, stem, args):
    """The current run's file paths, numbered after the existing runs."""
    used = sorted(int(m.group(1)) for m in
                  (RUN_INDEX_RE.match(name) for name in os.listdir(output_dir)) if m)
    index = (used[-1] if used else 1) if args.overwrite else (used[-1] if used else 0) + 1
    tag = f"{index:03d}"
    return index, {"video": os.path.join(output_dir, f"{stem}_annotated_{tag}.mp4"),
                   "frames": os.path.join(output_dir, f"frames_{tag}.jsonl"),
                   "summary": os.path.join(output_dir, f"summary_{tag}.json")}

def resolve_database(output_dir, args):
    """(where it is loaded from, where it is written).

    The folder's database is both; --database is only the seed for the first run and is
    never modified.
    """
    folder_db = os.path.join(output_dir, "face_database.json")
    if os.path.isfile(folder_db) and not args.reset_db:
        return folder_db, folder_db
    if os.path.isfile(folder_db) and args.reset_db:
        print(f"--reset-db: ignoring {folder_db} and starting over.")
    return (args.database if args.database and os.path.isfile(args.database) else None,
            folder_db)

# Thresholds that decide what ends up in the database. They are on the command line
# because they depend on the footage; the measured values are in summary.json under
# "distributions". Declared once here: parse_args() builds the options from these tables
# and apply_thresholds() copies them back over the module constants, so adding a threshold
# is one line instead of four places to keep in step. Not given = keep the script's value.
# Pixel thresholds are NOT scaled with the working resolution: what matters for
# recognition is how many pixels the face has, not what fraction of the frame it takes.
TUNING = [
    ("--min-face", "MIN_FACE_SIZE", int,
     "the minimum bbox side for a face to be processed"),
    ("--verify-interval", "VERIFY_INTERVAL_FRAMES", int,
     "at most how many frames between checks of a track"),
    ("--identified-interval", "IDENTIFIED_INTERVAL_FRAMES", int,
     "the cooldown: every how many frames a track that ALREADY has an identity is "
     "re-checked. Larger = faster, but an id swapped by the tracker is caught later"),
    ("--min-blur", "MIN_BLUR", float, "the Laplacian variance considered 'sharp'"),
    ("--quality-min", "QUALITY_MIN", float,
     "the minimum quality for the recognition model to be worth running"),
    ("--verify-threshold", "VERIFY_THRESHOLD", float,
     "the cosine score from which a face is recognised"),
    ("--enroll-min-checks", "ENROLL_MIN_CHECKS", int,
     "how many consecutive 'unknown' checks an enrollment requires"),
    ("--enroll-min-face", "ENROLL_MIN_FACE", int,
     "the minimum face side, in px, for a shot to become a prototype"),
    ("--enroll-min-blur", "ENROLL_MIN_BLUR", float,
     "the minimum sharpness for a shot to become a prototype"),
    ("--enroll-min-quality", "ENROLL_MIN_QUALITY", float,
     "the minimum quality for a shot to become a prototype"),
    ("--enroll-min-frontality", "ENROLL_MIN_FRONTALITY", float,
     "how frontal the person must be for their shot to become a prototype, min(yaw, pitch) "
     "in [0,1]"),
]

CONTINUITY = [
    ("--identity-flip-checks", "IDENTITY_FLIP_CHECKS", int,
     "how many CONSECUTIVE checks must contradict a confirmed identity for it to fall; "
     "1 = the first bad window erases the name"),
    ("--identity-stale", "IDENTITY_STALE_FRAMES", int,
     "after how many frames without a confirmation the displayed identity is drawn yellow "
     "with '?'; it is not erased"),
    ("--tracked-min-confidence", "TRACKED_MIN_CONFIDENCE", float,
     "the minimum confidence for the box of a track that already has state"),
]

# Values that cannot be taken literally from the command line.
CLAMPS = {"IDENTITY_FLIP_CHECKS": lambda v: max(1, v)}

def add_tuned(group, entries):
    """One option per table row, with the script's value shown as the default."""
    for flag, name, kind, help_text in entries:
        group.add_argument(flag, type=kind, default=None,
                           help=f"{help_text} (default {globals()[name]})")

def apply_thresholds(args):
    """Move the thresholds given on the command line over the module constants."""
    for flag, name, _, _ in TUNING + CONTINUITY:
        value = getattr(args, flag[2:].replace("-", "_"))
        if value is not None:
            globals()[name] = CLAMPS.get(name, lambda v: v)(value)
    globals()["HIDE_REJECTED_BOXES"] = args.hide_rejected
    # derived, so they have to be recomputed after the thresholds above
    globals()["ENROLL_MAX_SCORE"] = VERIFY_THRESHOLD - ENROLL_MARGIN
    globals()["PRETRACK_MIN_SIZE"] = MIN_FACE_SIZE * PRETRACK_SIZE_FACTOR

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the face recognition pipeline on a video file and write an "
                    "annotated video + the database + a per-frame log.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add = parser.add_argument
    add("video", help="the video file name (next to the script or in sample/)")
    add("--output", default=None, help="the output folder (default: the video's name); if "
        "it exists, it is reused together with its database")
    add("--database", default=FACE_DATABASE_PATH, help="the seed database, used only when "
        "the output folder does not have one yet; it is never modified")
    add("--pgie-config", default=YOLO_CONFIG_PATH,
        help="the nvinfer config for the face detector")
    add("--tracker-config", default=TRACKER_CONFIG_PATH, help="the low-level config for "
        "nvtracker; by default NvDCF_accuracy is tried first, then NvDCF_perf")
    add("--shadow-track-age", type=int, default=None, help=f"how many frames a track lives "
        f"without any detection (maxShadowTrackAge); this decides whether somebody who "
        f"turns their head keeps their id (default {TRACKER_OVERRIDES['maxShadowTrackAge']})")
    add("--min-detector-confidence", type=float, default=None, help=f"the confidence from "
        f"which nvtracker accepts a box; kept low on purpose, since weak boxes are what "
        f"link a track across occlusions "
        f"(default {TRACKER_OVERRIDES['minDetectorConfidence']})")
    add("--no-tracker-patch", action="store_true",
        help="use the tracker config exactly as it is on disk")
    add("--landmark-engine", default=PFLD_MODEL_PATH)
    add("--recognition-engine", default=RECOGNITION_MODEL_PATH)
    add("--width", type=int, default=None, help="the working width; by default taken from "
        "the file, so portrait footage works too. Given alone, the height is computed "
        "keeping the aspect ratio")
    add("--height", type=int, default=None, help="the working height; by default from the file")
    add("--max-side", type=int, default=1920, help="the maximum side of the working "
        "resolution; above it the source is shrunk keeping the aspect ratio (at 4K, the "
        "buffers between the decoder and streammux exhaust the NVMM memory). 0 = no limit")
    add("--decoder-surfaces", type=int, default=0, help="surfaces on top of the decoder's "
        "required minimum; on 4K sources each one costs ~12 MB of NVMM memory")
    add("--rotate", default="auto", choices=["auto", "0", "90", "180", "270"],
        help="the rotation applied to the source, in degrees, clockwise; 'auto' follows "
             "the tag in the file")
    add("--bitrate", type=int, default=0, help="the bitrate of the output video, in bits/s; "
        "0 = computed from the resolution (~8 Mbit/s at 1080p)")
    add("--no-video", action="store_true", help="do not write the annotated video: takes the "
        "encoder and the drawing out of the pipeline. The first thing to try when the board "
        "runs out of memory; the database and the log come out unchanged")
    add("--queue-size", type=int, default=4, help="how many frames can wait in each queue; "
        "this decides how much decoding/detection, the probe and the encoding overlap. "
        "Every waiting frame is an NVMM surface (~8 MB at 1080p RGBA). 1 = no overlap")
    add("--no-pretrack-filter", action="store_true", help="send the tracker all the "
        "detector's boxes; much slower on wide frames, only useful for comparison")
    add("--log-rejected", action="store_true", help="write every rejected box to the log, "
        "not only how many there were per gate (the log grows ~4x)")
    add("--no-enroll", action="store_true", help="do not add new identities to the database")
    add("--no-landmarks", action="store_true",
        help="draw only the 5 ArcFace points, not the whole set")
    add("--new-dir", action="store_true", help="start in a new folder (without the "
        "identities enrolled by the previous runs)")
    add("--overwrite", action="store_true",
        help="overwrite the files of the last run in the folder")
    add("--reset-db", action="store_true", help="ignore the database in the folder and start "
        "from the --database seed (with --database '' it starts empty)")

    tuning = parser.add_argument_group("thresholds (see 'distributions' in summary.json)")
    add_tuned(tuning, TUNING)
    tuning.add_argument("--landmark-all", action="store_true", help="request landmarks on "
                        "all the faces, every frame; only useful for comparison")

    cont = parser.add_argument_group("identity continuity")
    add_tuned(cont, CONTINUITY)
    cont.add_argument("--hide-rejected", action="store_true",
                      help="remove the stateless boxes from the drawing entirely")
    return parser.parse_args(argv)

def available_memory_mb():
    """(free, total) in MB from /proc/meminfo; None where it does not exist.

    On Jetson the memory is unified, so this is the number that decides whether the run
    passes or dies with ENOMEM.
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

# All the same thing seen from different places: the unified memory ran out.
MEMORY_ERROR_MARKERS = (
    "queue input batch",            # nvinfer could not push the batch
    "NVDSINFER_TENSORRT_ERROR",     # usually CUDNN_STATUS_INTERNAL_ERROR underneath
    "bufferpool",                   # "failed to activate bufferpool" at nvvideoconvert
    "NvMapMemAllocInternalTagged",  # the NVMM allocator, "error 12" = ENOMEM
    "Error in allocating buffer",
    "gst-resource-error-quark")

def explain_inference_error(text, args):
    """Translate an allocation failure into what can concretely be done."""
    if not any(marker in text for marker in MEMORY_ERROR_MARKERS):
        return
    free, total = available_memory_mb()
    source = args.source_info
    print("\n  The detector failed at inference. Almost certainly memory: "
          "CUDNN_STATUS_INTERNAL_ERROR")
    print("  usually means there is no room left for the workspace, not that the model is "
          "wrong.")
    if free is not None:
        print(f"  Free right now: {free} MB out of {total} MB.")
    print(f"  The source is decoded at its own resolution "
          f"({source['width']}x{source['height']}),\n  however small the working frame is, "
          f"and on top of that come the detector, the two engines and the encoder.")
    print("  Things to try, in order:")
    print("  1. --no-video: removes NVENC and the output conversions. The database and the "
          "log\n     come out the same, only the annotated video is not written.")
    print("  2. --max-side 1280: shrinks everything after the decoder.")
    print("  3. Free memory: stop the desktop session and other CUDA processes, and watch\n"
          "     with tegrastats how much stays free during the run.")

def main():
    global AUTO_ENROLL, DRAW_ALL_LANDMARKS, DRAW_OVERLAY, RUN
    global PRETRACK_FILTER, LOG_REJECTED, LANDMARK_ONLY_WHEN_NEEDED

    args = parse_args()
    AUTO_ENROLL = not args.no_enroll
    LANDMARK_ONLY_WHEN_NEEDED = not args.landmark_all
    DRAW_OVERLAY = not args.no_video
    DRAW_ALL_LANDMARKS = not args.no_landmarks and DRAW_OVERLAY
    PRETRACK_FILTER = not args.no_pretrack_filter
    LOG_REJECTED = args.log_rejected
    args.queue_size = max(1, args.queue_size)
    apply_thresholds(args)
    check_pyds_api()

    video_path = resolve_video(args.video)
    args.pgie_config = resolve_config(args.pgie_config,
                                      "the nvinfer config (the face detector)")
    args.tracker_config = resolve_config(args.tracker_config, "the nvtracker config")

    # Gst.init before rotation_supported(): the element factories need it. Probing the file
    # does not, it goes through OpenCV.
    Gst.init(None)
    source = probe_video_info(video_path)
    args.rotation = source["rotation"] if args.rotate == "auto" else int(args.rotate)
    if args.rotation and not rotation_supported():
        print(f"[WARNING] nvvideoconvert has no flip-method on this platform; I cannot "
              f"rotate the source by {args.rotation} degrees. The faces stay on their side, "
              f"so the detector will miss them.")
        args.rotation = 0
    args.width, args.height = working_resolution(source, args)
    args.source_info = source

    free, total = available_memory_mb()
    args.free_memory_mb = free
    print(f"Source:  {source['width']}x{source['height']}, rotation {source['rotation']} "
          f"degrees [{source['read_with']}]"
          + (f"; free memory {free} MB out of {total} MB" if free is not None else ""))
    print(f"Working: {args.width}x{args.height}"
          + (" (portrait)" if args.height > args.width else "")
          + (f", rotating by {args.rotation} degrees" if args.rotation else ""))
    # Shrinking the source halves the faces too, and the thresholds are in pixels.
    if args.width < source["width"]:
        print(f"         the source is shrunk {source['width'] / float(args.width):.1f}x, "
              f"and so are the faces: with --max-side {source['width']} they keep their size")

    stem = os.path.splitext(os.path.basename(video_path))[0]
    output_dir, reused = prepare_output_dir(stem, args)
    run_index, paths = run_paths(output_dir, stem, args)
    database_in, database_out = resolve_database(output_dir, args)
    apply_tracker_overrides(args, output_dir, run_index)

    print(f"Video:   {video_path}")
    print(f"Output:  {output_dir} ({'reused' if reused else 'new'}, run {run_index})")
    load_models(args.landmark_engine, args.recognition_engine, database_in, database_out)
    if database_in == database_out:
        print("Continuing the folder's database: whatever was enrolled by the previous runs "
              "is recognised now.")

    RUN = Run(args, output_dir, video_path, paths, run_index, database_in, database_out)
    pipeline = build_pipeline(video_path, paths["video"], args)
    loop = GLib.MainLoop()
    failure = []    # remembered so the failure also shows up in the exit code

    def bus_call(bus, message, loop):
        if message.type == Gst.MessageType.EOS:
            print("End of file.")
            loop.quit()
        elif message.type == Gst.MessageType.WARNING:
            warning, debug = message.parse_warning()
            print(f"GStreamer warning: {warning}: {debug}")
        elif message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            print(f"GStreamer error: {error}: {debug}")
            failure.append(str(error))
            explain_inference_error(f"{error} {debug}", args)
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
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        RUN.logger.close()
        face_database.save()
        RUN.failure = failure[0] if failure else None
        print_report(write_summary(RUN), RUN)

    return 1 if failure else 0

if __name__ == '__main__':
    sys.exit(main())
