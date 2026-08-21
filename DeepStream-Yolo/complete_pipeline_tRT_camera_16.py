"""Live camera variant of the face recognition pipeline.

Same processing as complete_pipeline_tRT_media_15.py, with two differences:

  1. the source is a camera -- v4l2 (USB webcam) or CSI (nvarguscamerasrc) -- and the
     default output is a window on screen. It can also record in parallel with the display
     (--record), or run with no image at all (--no-display).
  2. the presence of each identity is tracked, and when somebody leaves and comes back
     within RETURN_WINDOW_S seconds the console prints:

         [RETURNED] person_5 came back after 6.3 s

Same pipeline as _14 -- same gates, same console messages, same keys in frames_*.jsonl and
summary_*.json -- written shorter: the overridable thresholds are declared once in
TUNING/CONTINUITY/PRESENCE_TUNING instead of being repeated in parse_args() and
apply_thresholds(), and the explanations were shortened without dropping any reason.

Usage:
    python3 complete_pipeline_tRT_camera_16.py                       # /dev/video0
    python3 complete_pipeline_tRT_camera_16.py --device /dev/video1
    python3 complete_pipeline_tRT_camera_16.py --camera csi --sensor-id 0
    python3 complete_pipeline_tRT_camera_16.py --record             # also writes mp4
    python3 complete_pipeline_tRT_camera_16.py --no-display         # console only
    python3 complete_pipeline_tRT_camera_16.py --return-window 30

The output folder (camera_<device> by default) is REUSED: the database inside it is loaded
at startup, so people enrolled yesterday are recognised today. The database is also saved
periodically during the run (--save-every), not only on exit: a live session can last
hours, and a hard Ctrl+C or a power cut must not lose everything that was enrolled.

Presence counts only identities CONFIRMED by a pass through the recognition model
(checks_confirmed >= 1), and it also counts the tracks that did not pass the gates in the
current frame -- a turned head is not a departure. Two thresholds of its own:
ABSENCE_GRACE_S (below it, a gap is a flicker, not a departure) and RETURN_WINDOW_S (above
it, a reappearance is printed as [REAPPEARED], not [RETURNED]).
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
# Samples below MIN_BLUR still compete (an absolute threshold depends too much on the
# source); only this safety floor is dropped outright.
BLUR_REJECT_FACTOR = 0.25
LOG_REJECTED = False          # per-box detail of every reject; by default only counted

VERIFY_INTERVAL_FRAMES = 15   # at most this many frames between checks of a track
RETRY_INTERVAL_ON_FAIL = 7    # sooner retry when the quality gate failed
# Cadence for tracks that already have a confirmed identity. Between checks no landmarks
# are computed for them, which is the main speed lever; the cost is that an id swapped by
# the tracker is caught up to this many frames later.
IDENTIFIED_INTERVAL_FRAMES = 30
LABEL_HISTORY_SIZE = 5        # recent decisions kept for the majority vote
TRACK_TIMEOUT_FRAMES = 300    # absent frames after which a track is forgotten
PRUNE_CHECK_INTERVAL = 90     # how often we look for dead tracks
# Identity continuity: what we know about a person (TrackState.identity) is kept apart
# from what the last check produced (the vote in history): a conclusion vs a measurement.
IDENTITY_FLIP_CHECKS = 2      # consecutive contradicting checks needed to drop an identity
IDENTITY_STALE_FRAMES = 150   # after this long without confirmation: yellow, with "?"

RETURN_WINDOW_S = 15.0        # a return within this window prints [RETURNED]
ABSENCE_GRACE_S = 1.0         # below this, a gap is a flicker, not a departure
ANNOUNCE_DEPARTURE = True     # also print [PRESENT]/[LEFT], not only returns
DATABASE_SAVE_INTERVAL_S = 60.0   # periodic database save; 0 = only on exit

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
LANDMARK_ONLY_WHEN_NEEDED = DRAW_ALL_LANDMARKS = DRAW_OVERLAY = True
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
        """Atomic write into save_path (the run folder), never over the seed.

        Here save() is also called mid-run, from the GLib loop, while add() runs on the
        GStreamer thread; add() does the vstack before the append, so we snapshot both
        fields and write the consistent prefix.
        """
        labels, matrix = list(self.labels), self.matrix
        count = min(len(labels), matrix.shape[0])
        tmp = self.save_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({labels[i]: matrix[i].tolist() for i in range(count)}, f)
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

class PresenceLog:
    """Who is in frame, who left and who came back.

    Keyed by database NAME, not by tracker id: one person can go through several ids
    without ever leaving. Time is the wall clock, because the window is in seconds and the
    frame rate varies on a live run.
    """

    def __init__(self, window_s, grace_s):
        self.window_s, self.grace_s = window_s, grace_s
        self.present = {}       # visible name -> last time seen
        self.absent = {}        # departed name -> time last seen
        self.first_seen = {}
        self.events = []
        self.counts = Counter()

    def update(self, names, now):
        """Advance presence to 'now'; returns (kind, name, seconds_gone) events."""
        events = []
        for name in names:
            if name in self.present:
                self.present[name] = now
                continue
            self.present[name] = now
            left_at = self.absent.pop(name, None)
            if left_at is None:
                self.first_seen[name] = now
                events.append(("appeared", name, None))
                continue
            gone = now - left_at
            events.append(("returned" if gone <= self.window_s else "reappeared", name, gone))

        for name, last_seen in list(self.present.items()):
            if name in names or now - last_seen < self.grace_s:
                continue
            # The departure time is the last time seen, not the moment we noticed:
            # otherwise the grace period would count as absence.
            del self.present[name]
            self.absent[name] = last_seen
            events.append(("left", name, None))

        for kind, name, gone in events:
            self.counts[kind] += 1
            self.events.append({"kind": kind, "name": name, "at": r(now, 2),
                                "gone_s": None if gone is None else r(gone, 2)})
        return events

    def returns(self):
        return [event for event in self.events if event["kind"] == "returned"]

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
    """One JSON line per frame; --no-frame-log makes writing a no-op.

    A one-hour live session at 25 FPS is ~90 000 lines that nobody usually reads;
    summary.json is written either way.
    """

    def __init__(self, path, enabled=True):
        self.path = path if enabled else None
        self.handle = open(path, "w", encoding="utf-8") if enabled else None
        self.frames = 0

    def write(self, record):
        if self.handle is None:
            return
        self.handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.frames += 1
        if self.frames % 50 == 0:
            self.handle.flush()

    def close(self):
        if self.handle is not None:
            self.handle.flush()
            self.handle.close()

class Run:
    """The state of the current run: where we write, what happened so far."""

    def __init__(self, args, output_dir, video_path, paths, run_index, database_in,
                 database_out):
        self.args, self.output_dir, self.video_path = args, output_dir, video_path
        self.paths, self.run_index = paths, run_index
        self.database_in, self.database_out = database_in, database_out
        self.logger = FrameLogger(paths["frames"], enabled=not args.no_frame_log)
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
# Presence log, built in main() once the thresholds are known. Global like RUN, for the
# same reason: the GStreamer probe takes no extra parameter.
PRESENCE = None

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

    # --- Step 9b: who is in frame, who left, who came back ---
    # Gathered after the frame's decisions, so a return is announced in the very frame it
    # was confirmed. Held tracks count as present -- somebody turned away has not left.
    # Only identities confirmed by the recognition model (checks_confirmed >= 1) count.
    visible = list(active.values()) + [st for _, st, _, _, _ in held_objects]
    present_names = {st.identity for st in visible
                     if st.identity is not None and st.checks_confirmed >= 1}
    for kind, name, gone in PRESENCE.update(present_names, time.monotonic()):
        if kind == "returned":
            print(f"[RETURNED] {name} came back after {gone:.1f} s")
        elif kind == "reappeared":
            print(f"[REAPPEARED] {name} is in frame again, after {gone:.1f} s (past the "
                  f"{RETURN_WINDOW_S:.0f} s window)")
        elif kind == "appeared" and ANNOUNCE_DEPARTURE:
            print(f"[PRESENT] {name} entered the frame")
        elif kind == "left" and ANNOUNCE_DEPARTURE:
            print(f"[LEFT] {name} left the frame")

    # Held boxes are counted even with no drawing, so the summary is the same with and
    # without a display.
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
    if present_names:
        record["present"] = sorted(present_names)
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

def link_chain(elements, context=""):
    """Link a linear chain, saying which pair failed."""
    for upstream, downstream in zip(elements, elements[1:]):
        if not upstream.link(downstream):
            raise RuntimeError(f"Cannot link {upstream.get_name()} to "
                               f"{downstream.get_name()} (incompatible caps?){context}.")

def configure_encoder(encoder, caps_out, bitrate, iframe_interval=None):
    """H.264 settings, on NVENC or on the desktop fallback."""
    if encoder.get_factory().get_name() == "nvv4l2h264enc":
        encoder.set_property('bitrate', bitrate)
        # The default profile is Baseline (no CABAC, no B-frames), which is what looks soft
        # at the same bitrate; High costs the same on NVENC. A keyframe every couple of
        # seconds keeps the file readable from the middle, so a hard stop loses that much.
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

# A camera has no rotation tag, so rotation is requested explicitly with --rotate (useful
# for a sensor mounted sideways). nvvideoconvert flip-method: 1 = 90 counter-clockwise,
# 2 = 180, 3 = 90 clockwise.
ROTATION_TO_FLIP_METHOD = {90: 3, 180: 2, 270: 1}

def camera_kind(args):
    """v4l2 (USB webcam) or csi (on-board sensor), resolving 'auto'.

    Nothing is probed beforehand: opening the camera to interrogate it and then closing it
    to reopen it from GStreamer is the shortest path to "Device or resource busy". The
    resolution is REQUESTED through caps instead.
    """
    if args.camera != "auto":
        return args.camera
    if args.sensor_id is not None:
        return "csi"
    if os.path.exists(args.device):
        return "v4l2"
    return "csi" if Gst.ElementFactory.find("nvarguscamerasrc") is not None else "v4l2"

def build_camera_source(args, kind):
    """The elements from the camera up to NVMM/NV12, in link order.

    CSI delivers NVMM/NV12 directly. v4l2 delivers system memory (YUY2 or MJPEG), so it
    needs a CPU decode/convert before nvvideoconvert can upload it; --mjpeg is worth asking
    for when the camera supports it, since it puts far less traffic on the USB bus.
    """
    if kind == "csi":
        source = make_element("nvarguscamerasrc", "camera")
        source.set_property("sensor-id", args.sensor_id or 0)
        caps = make_element("capsfilter", "caps-camera")
        caps.set_property("caps", Gst.Caps.from_string(
            f"video/x-raw(memory:NVMM), width={args.capture_width}, "
            f"height={args.capture_height}, framerate={args.fps}/1, format=NV12"))
        return [source, caps], f"nvarguscamerasrc sensor-id={args.sensor_id or 0}"

    source = make_element("v4l2src", "camera")
    source.set_property("device", args.device)
    # io-mode=2 (mmap) is what almost every UVC webcam wants; without it some drivers fall
    # back to read() and give much lower rates.
    if source.find_property("io-mode"):
        source.set_property("io-mode", 2)

    caps = make_element("capsfilter", "caps-camera")
    if args.mjpeg:
        caps.set_property("caps", Gst.Caps.from_string(
            f"image/jpeg, width={args.capture_width}, height={args.capture_height}, "
            f"framerate={args.fps}/1"))
        chain = [source, caps, make_element(["nvjpegdec", "jpegdec"], "mjpeg-decoder")]
    else:
        caps.set_property("caps", Gst.Caps.from_string(
            f"video/x-raw, width={args.capture_width}, height={args.capture_height}, "
            f"framerate={args.fps}/1"))
        chain = [source, caps]

    # videoconvert covers whatever format the camera emits; nvvideoconvert alone does not
    # accept them all and negotiation then fails with "not-negotiated".
    chain.append(make_element("videoconvert", "camera-convert"))
    return chain, (f"v4l2src {args.device}, {'MJPEG' if args.mjpeg else 'raw'} "
                   f"{args.capture_width}x{args.capture_height}@{args.fps}")

def explain_camera_error(text, args):
    """Translate the usual camera errors into what to do about them."""
    lowered = (text or "").lower()
    if "not-negotiated" in lowered or "internal data stream error" in lowered:
        print(f"\n[CAMERA] The camera cannot deliver what was asked for "
              f"({args.capture_width}x{args.capture_height}@{args.fps}"
              f"{', MJPEG' if args.mjpeg else ''}).")
        print(f"         See what it can do:  v4l2-ctl -d {args.device} --list-formats-ext")
        print("         Then give exactly one combination from there, with "
              "--capture-width/--capture-height/--fps (and --mjpeg if needed).")
    elif "busy" in lowered or "permission denied" in lowered:
        print(f"\n[CAMERA] {args.device} is busy or you have no rights on it.")
        print(f"         Who holds it:  fuser -v {args.device}")
        print("         Rights:        add the user to the 'video' group.")
    elif "no such device" in lowered or "cannot identify device" in lowered:
        print(f"\n[CAMERA] {args.device} does not exist. Cameras the system sees:")
        print("         ls -l /dev/video*    or    v4l2-ctl --list-devices")

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

def working_resolution(args):
    """The resolution the pipeline runs at, starting from the capture one.

    Note that the size thresholds are in pixels at the WORKING resolution, so asking for a
    smaller one shrinks the faces, not just the frame.
    """
    width, height = args.capture_width, args.capture_height
    if args.rotation in (90, 270):
        width, height = height, width

    if args.width and args.height:
        given, actual = args.width / float(args.height), width / float(height)
        if abs(given - actual) / actual > 0.01:
            print(f"[WARNING] --width/--height ({args.width}x{args.height}) have a different "
                  f"aspect ratio than the camera ({width}x{height}); the image will be "
                  f"distorted and the detection will suffer.")
        return even(args.width), even(args.height)
    if args.width:
        return even(args.width), even(args.width * height / float(width))
    if args.height:
        return even(args.height * width / float(height)), even(args.height)
    return even(width), even(height)

def build_pipeline(args, output_video):
    pipeline = Gst.Pipeline()
    if not pipeline:
        raise RuntimeError("The pipeline could not be created.")
    print("Creating the pipeline elements...")

    camera_chain, camera_desc = build_camera_source(args, args.camera_kind)
    print(f"  source: {camera_desc}")

    # Uploads to NVMM (v4l2) and scales to the working resolution. Rotation happens before
    # the capsfilter, so the caps already carry the post-rotation sides.
    vidconv_in = make_element("nvvideoconvert", "convert-in")
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
    # live-source=1: the camera keeps producing whatever we do, so streammux must not try
    # to synchronise on timestamps. The timeout is tied to the camera rate so we do not add
    # latency for nothing.
    streammux.set_property('live-source', 1)
    streammux.set_property('batched-push-timeout', int(1e6 / max(1, args.fps)))

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

    # Output: window, recording, both (through a tee), or neither. All of them share one
    # nvdsosd -- the drawing is done once.
    def display_branch():
        convert = make_element("nvvideoconvert", "display-convert")
        # nv3dsink is the JetPack 5 sink and takes NVMM directly; nveglglessink needs
        # nvegltransform before it on Jetson.
        sink = make_element(["nv3dsink", "nveglglessink", "xvimagesink", "autovideosink"],
                            "display")
        # sync=False: on a live source, show the latest processed frame instead of waiting
        # for its nominal timestamp.
        for prop in ("sync", "async"):
            if sink.find_property(prop):
                sink.set_property(prop, False)
        if sink.get_factory().get_name() == "nveglglessink":
            return [convert, make_element("nvegltransform", "egl-transform"), sink]
        return [convert, sink]

    def record_branch():
        convert = make_element("nvvideoconvert", "record-convert")
        caps_out = make_element("capsfilter", "caps-record")
        encoder = make_element(["nvv4l2h264enc", "x264enc"], "encoder")
        bitrate = args.bitrate or auto_bitrate(args.width, args.height)
        print(f"  bitrate: {bitrate / 1e6:.1f} Mbit/s"
              + ("" if args.bitrate else " (computed from the resolution)"))
        configure_encoder(encoder, caps_out, bitrate, max(1, args.fps * 2))
        sink = make_element("filesink", "filesink")
        sink.set_property('location', output_video)
        sink.set_property('sync', False)
        sink.set_property('async', False)
        return [convert, caps_out, encoder, make_element("h264parse", "parser"),
                make_element("qtmux", "muxer"), sink]

    branches = []
    if not args.no_display:
        branches.append(display_branch())
    if args.record:
        branches.append(record_branch())

    # Threads: [camera -> detector -> tracker] | [probe] | [osd -> output]
    queue_pre = make_queue("queue-detect", args.queue_size)
    chain = [streammux, pgie, tracker, make_queue("queue-probe", args.queue_size),
             vidconv_osd, caps_rgba, make_queue("queue-out", args.queue_size)]
    if branches:
        chain.append(make_element("nvdsosd", "osd"))
    else:
        fake = make_element("fakesink", "null-sink")
        fake.set_property('sync', False)
        fake.set_property('async', False)
        fake.set_property('enable-last-sample', False)
        chain.append(fake)

    tee = None
    if len(branches) > 1:
        tee = make_element("tee", "output-tee")
        chain.append(tee)
    elif branches:
        chain.extend(branches[0])

    for element in camera_chain + [vidconv_in, caps_in, queue_pre] + chain:
        pipeline.add(element)
    if tee is not None:
        for branch in branches:
            for element in branch:
                pipeline.add(element)

    print("Linking the pipeline elements" + (" (no window)..." if args.no_display else "..."))
    try:
        link_chain(camera_chain)
    except RuntimeError:
        raise RuntimeError(f"Cannot link the camera elements. Most likely the camera cannot "
                           f"deliver {args.capture_width}x{args.capture_height}@{args.fps}"
                           f"{' MJPEG' if args.mjpeg else ''}; see what it can with "
                           f"v4l2-ctl -d {args.device} --list-formats-ext.")
    if not camera_chain[-1].link(vidconv_in):
        raise RuntimeError("Cannot link the camera to the input converter.")
    link_chain([vidconv_in, caps_in, queue_pre])
    queue_pre.get_static_pad("src").link(streammux.get_request_pad("sink_0"))
    link_chain(chain)

    # Each branch starts with its own queue, otherwise the slowest one (the encoder) would
    # hold up the window as well.
    if tee is not None:
        for index, branch in enumerate(branches):
            queue = make_queue(f"queue-branch-{index}", args.queue_size)
            pipeline.add(queue)
            pad = tee.get_request_pad("src_%u")
            if pad.link(queue.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
                raise RuntimeError(f"Cannot link the tee to branch {index}.")
            link_chain([queue] + branch, f" (branch {index})")

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
        "camera": run.video_path, "folder": run.output_dir, "run": run.run_index,
        "error": run.failure,
        "source": {"capture_resolution": f"{source['width']}x{source['height']}",
                   "fps_requested": args.fps, "mjpeg": args.mjpeg,
                   "rotation_applied": args.rotation,
                   "working_resolution": f"{args.width}x{args.height}",
                   "kind": source["kind"], "window": not args.no_display,
                   "recording": args.record_path,
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
        # "history" holds every movement in order, so who was present at any moment of the
        # session can be reconstructed from it.
        "presence": {"movements": dict(PRESENCE.counts.most_common()) if PRESENCE else {},
                     "returns": PRESENCE.returns() if PRESENCE else [],
                     "in_frame_at_end": sorted(PRESENCE.present) if PRESENCE else [],
                     "history": PRESENCE.events if PRESENCE else [],
                     "thresholds": {"RETURN_WINDOW_S": RETURN_WINDOW_S,
                                    "ABSENCE_GRACE_S": ABSENCE_GRACE_S}},
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
    args = run.args
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

    presence = summary["presence"]
    if presence["movements"]:
        print("  presence:          "
              + ", ".join(f"{n} x{c}" for n, c in presence["movements"].items()))
        for event in presence["returns"]:
            print(f"    {event['name']} came back after {event['gone_s']} s")
        if presence["in_frame_at_end"]:
            print(f"    in frame at stop: {', '.join(presence['in_frame_at_end'])}")

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
    if args.record_path:
        print(f"  {args.record_path}")
    print(f"  {run.database_out}")
    if not args.no_frame_log:
        print(f"  {run.paths['frames']}")
    print(f"  {run.paths['summary']}")

# ============================================================
# MAIN
# ============================================================

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
    """The camera's folder, REUSED across runs so its database persists.

    The run's files are numbered instead (see run_paths), so nothing is lost.
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

PRESENCE_TUNING = [
    ("--return-window", "RETURN_WINDOW_S", float,
     "within how many seconds of leaving a re-entry is announced as a return; past it, it "
     "is an ordinary appearance"),
    ("--absence-grace", "ABSENCE_GRACE_S", float,
     "how many seconds without any box carrying their name mean the person really left"),
]

SAVE_TUNING = [
    ("--save-every", "DATABASE_SAVE_INTERVAL_S", float,
     "how often the database is written to disk during the run, in seconds; 0 = only on exit"),
]

# Values that cannot be taken literally from the command line.
CLAMPS = {"IDENTITY_FLIP_CHECKS": lambda v: max(1, v),
          "DATABASE_SAVE_INTERVAL_S": lambda v: max(0.0, v)}

def add_tuned(group, entries):
    """One option per table row, with the script's value shown as the default."""
    for flag, name, kind, help_text in entries:
        group.add_argument(flag, type=kind, default=None,
                           help=f"{help_text} (default {globals()[name]})")

def apply_thresholds(args):
    """Move the thresholds given on the command line over the module constants."""
    for flag, name, _, _ in TUNING + CONTINUITY + PRESENCE_TUNING + SAVE_TUNING:
        value = getattr(args, flag[2:].replace("-", "_"))
        if value is not None:
            globals()[name] = CLAMPS.get(name, lambda v: v)(value)
    globals()["HIDE_REJECTED_BOXES"] = args.hide_rejected
    globals()["ANNOUNCE_DEPARTURE"] = not args.quiet_presence
    # derived, so they have to be recomputed after the thresholds above
    globals()["ENROLL_MAX_SCORE"] = VERIFY_THRESHOLD - ENROLL_MARGIN
    globals()["PRETRACK_MIN_SIZE"] = MIN_FACE_SIZE * PRETRACK_SIZE_FACTOR

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Live face recognition on a camera: a window on screen, automatic "
                    "enrollment, a persistent database and a console announcement when "
                    "somebody comes back into frame.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    camera = parser.add_argument_group("camera")
    camera.add_argument("--camera", default="auto", choices=["auto", "v4l2", "csi"],
                        help="v4l2 = USB webcam (--device), csi = on-board sensor "
                             "(nvarguscamerasrc, --sensor-id)")
    camera.add_argument("--device", default="/dev/video0",
                        help="the v4l2 camera device file")
    camera.add_argument("--sensor-id", type=int, default=None, help="the CSI sensor index")
    camera.add_argument("--capture-width", type=int, default=1280, help="the width REQUESTED "
                        "from the camera; it must be one the camera can deliver "
                        "(v4l2-ctl --list-formats-ext)")
    camera.add_argument("--capture-height", type=int, default=720,
                        help="the height requested from the camera")
    camera.add_argument("--fps", type=int, default=30,
                        help="the frame rate requested from the camera")
    camera.add_argument("--mjpeg", action="store_true", help="ask for MJPEG instead of raw. "
                        "Far less USB traffic at the same resolution, so higher rates; "
                        "costs a CPU decode")

    add = parser.add_argument
    add("--output", default=None, help="the output folder (default: camera_<device>); if it "
        "exists, it is reused together with its database")
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
    add("--width", type=int, default=None, help="the WORKING width, if it must differ from "
        "the capture one. Note that the size thresholds are in pixels at this resolution, "
        "so shrinking shrinks the faces too")
    add("--height", type=int, default=None, help="the working height; by default the capture one")
    add("--rotate", default="0", choices=["0", "90", "180", "270"], help="rotation applied "
        "to the camera image, in degrees, clockwise (for sideways-mounted cameras)")
    add("--bitrate", type=int, default=0, help="the recording bitrate, in bits/s; "
        "0 = computed from the resolution (~8 Mbit/s at 1080p)")
    add("--record", nargs="?", const=True, default=False, help="also record an mp4, in "
        "parallel with the window; without a value, into the run folder")
    add("--no-display", action="store_true", help="do not open the window. Without --record "
        "only the console is left: recognition, the announcements and the database are "
        "unchanged, but the encoder and the display sink leave the board")
    add("--no-frame-log", action="store_true", help="do not write frames_*.jsonl. A one-hour "
        "live session is ~100 000 lines that are usually never read; summary.json is "
        "written anyway")
    add_tuned(parser, SAVE_TUNING)
    add("--queue-size", type=int, default=4, help="how many frames can wait in each queue; "
        "this decides how much capture/detection, the probe and the output overlap. Every "
        "waiting frame is an NVMM surface (~8 MB at 1080p RGBA). 1 = no overlap")
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

    presence = parser.add_argument_group("presence and returns")
    add_tuned(presence, PRESENCE_TUNING)
    presence.add_argument("--quiet-presence", action="store_true",
                          help="do not print [PRESENT]/[LEFT], only the returns")
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
    print(f"  The camera delivers {source['width']}x{source['height']}, and on top of that "
          f"come the\n  detector, the two engines, the display sink and (if asked for) the "
          f"encoder.")
    print("  Things to try, in order:")
    print("  1. --no-display (without --record): removes both the display sink and NVENC.\n"
          "     Recognition, the announcements and the database are unchanged.")
    print("  2. --capture-width 1280 --capture-height 720: fewer pixels everywhere.\n"
          "     Careful: the faces shrink too, so the size thresholds are felt.")
    print("  3. Free memory: stop the desktop session and other CUDA processes, and watch\n"
          "     with tegrastats how much stays free during the run.")

def main():
    global AUTO_ENROLL, DRAW_ALL_LANDMARKS, DRAW_OVERLAY, RUN, PRESENCE
    global PRETRACK_FILTER, LOG_REJECTED, LANDMARK_ONLY_WHEN_NEEDED

    args = parse_args()
    AUTO_ENROLL = not args.no_enroll
    LANDMARK_ONLY_WHEN_NEEDED = not args.landmark_all
    # Drawing only makes sense if somebody sees it: a window or a recording.
    DRAW_OVERLAY = (not args.no_display) or bool(args.record)
    DRAW_ALL_LANDMARKS = not args.no_landmarks and DRAW_OVERLAY
    PRETRACK_FILTER = not args.no_pretrack_filter
    LOG_REJECTED = args.log_rejected
    args.queue_size = max(1, args.queue_size)
    args.fps = max(1, args.fps)
    apply_thresholds(args)
    check_pyds_api()

    args.pgie_config = resolve_config(args.pgie_config,
                                      "the nvinfer config (the face detector)")
    args.tracker_config = resolve_config(args.tracker_config, "the nvtracker config")

    # Gst.init before any factory lookup: camera_kind and rotation_supported both search
    # for elements.
    Gst.init(None)
    args.camera_kind = camera_kind(args)
    if args.camera_kind == "v4l2" and not os.path.exists(args.device):
        print(f"[WARNING] {args.device} does not exist right now. Cameras the system sees:  "
              f"ls -l /dev/video*")

    args.rotation = int(args.rotate)
    if args.rotation and not rotation_supported():
        print(f"[WARNING] nvvideoconvert has no flip-method on this platform; I cannot "
              f"rotate the image by {args.rotation} degrees. The faces stay on their side, "
              f"so the detector will miss them.")
        args.rotation = 0
    args.width, args.height = working_resolution(args)
    args.source_info = {"width": args.capture_width, "height": args.capture_height,
                        "rotation": args.rotation, "kind": args.camera_kind}

    free, total = available_memory_mb()
    args.free_memory_mb = free
    print(f"Camera:  {args.camera_kind}, capture "
          f"{args.capture_width}x{args.capture_height}@{args.fps}"
          + (", MJPEG" if args.mjpeg else "")
          + (f"; free memory {free} MB out of {total} MB" if free is not None else ""))
    print(f"Working: {args.width}x{args.height}"
          + (" (portrait)" if args.height > args.width else "")
          + (f", rotating by {args.rotation} degrees" if args.rotation else ""))
    if args.width < args.capture_width:
        print(f"         the image is shrunk "
              f"{args.capture_width / float(args.width):.1f}x, and so are the faces: the "
              f"size thresholds are in pixels at the working resolution")

    # The folder is named after the camera, so its database is found again on the next run
    # and people enrolled yesterday are recognised today.
    stem = "camera_" + (f"sensor{args.sensor_id or 0}" if args.camera_kind == "csi"
                        else os.path.basename(args.device.rstrip("/")) or "video0")
    output_dir, reused = prepare_output_dir(stem, args)
    run_index, paths = run_paths(output_dir, stem, args)
    database_in, database_out = resolve_database(output_dir, args)
    # --record without a value writes into the run folder; --record path writes there.
    args.record_path = (None if not args.record else
                        paths["video"] if args.record is True else args.record)
    apply_tracker_overrides(args, output_dir, run_index)

    print(f"Output:  {output_dir} ({'reused' if reused else 'new'}, run {run_index})")
    print(f"Screen:  {'no (--no-display)' if args.no_display else 'window'}"
          + (f", recording into {args.record_path}" if args.record else ""))
    print(f"Presence: returns within {RETURN_WINDOW_S:.0f} s are announced (a departure is "
          f"declared after {ABSENCE_GRACE_S:.1f} s without any box carrying the name)")

    load_models(args.landmark_engine, args.recognition_engine, database_in, database_out)
    if database_in == database_out:
        print("Continuing the folder's database: whatever was enrolled by the previous runs "
              "is recognised now.")

    RUN = Run(args, output_dir,
              args.device if args.camera_kind == "v4l2" else f"csi:{args.sensor_id or 0}",
              paths, run_index, database_in, database_out)
    PRESENCE = PresenceLog(RETURN_WINDOW_S, ABSENCE_GRACE_S)

    pipeline = build_pipeline(args, args.record_path)
    loop = GLib.MainLoop()
    failure = []    # remembered so the failure also shows up in the exit code

    def bus_call(bus, message, loop):
        if message.type == Gst.MessageType.EOS:
            print("The stream ended.")
            loop.quit()
        elif message.type == Gst.MessageType.WARNING:
            warning, debug = message.parse_warning()
            print(f"GStreamer warning: {warning}: {debug}")
        elif message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            print(f"GStreamer error: {error}: {debug}")
            failure.append(str(error))
            explain_camera_error(f"{error} {debug}", args)
            explain_inference_error(f"{error} {debug}", args)
            loop.quit()
        return True

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Periodic save: a live session can last hours, and a power cut or a second Ctrl+C
    # would lose everything enrolled in the meantime.
    def save_database_tick():
        if len(face_database) != face_database.source_count or RUN.enrollments:
            face_database.save()
        return True     # GLib repeats while this returns True

    if DATABASE_SAVE_INTERVAL_S > 0:
        GLib.timeout_add_seconds(int(DATABASE_SAVE_INTERVAL_S), save_database_tick)

    # The second Ctrl+C stops immediately: if EOS never arrives (camera stuck in an ioctl,
    # pipeline stopped before PLAYING), the first one would look ignored.
    stopping = []

    def sigint_handler(sig, frame):
        if stopping:
            print("\nAnother Ctrl+C: stopping right now.")
            loop.quit()
            return
        stopping.append(True)
        # EOS, not a brutal stop, so qtmux can close the mp4.
        print("\nStop requested; sending EOS so the file closes properly (another Ctrl+C "
              "stops immediately)...")
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
