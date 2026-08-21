"""Live camera variant of the face recognition pipeline -- runtime only.

Same recognition as complete_pipeline_tRT_camera_16.py, gate for gate, and the same
presence tracking: when somebody leaves and comes back within RETURN_WINDOW_S seconds the
console prints

    [RETURNED] person_5 came back after 6.3 s

Two redundancies that _18 still carried are gone here, both without changing a single
decision the pipeline makes:

  - a dead clause in ready_to_enroll(). It tested last_unknown_score < ENROLL_MAX_SCORE,
    but verify() only ever answers LABEL_UNKNOWN when the score is already below
    VERIFY_THRESHOLD - ENROLL_MARGIN, which is exactly ENROLL_MAX_SCORE, and that branch
    was the only writer of the field. The test could never fail, so it looked like a gate
    while gating nothing. The field and the constant went with it.
  - the two parallel "best shot" mechanisms. A track kept five loose best_* fields ranked
    on quality for the current check, and a separate best_unknown dict ranked on
    frontality for the enrollment prototype -- the same idea written twice. Both are now
    one Best holder differing only in its ranking key (and in the prototype gate).

What is gone compared with _16 is the tuning scaffolding, which had nothing to do with
recognising a face: summary_*.json and frames_*.jsonl with the counters behind them, the
enrollment-blocker diagnostics, the memory-error explanations, and 26 of the 41
command-line options. The thresholds are now plain constants below -- edit them here
instead of passing flags. Use _16 when you need to measure why a threshold blocks, then
copy the values you settled on into this file.

The source is a camera -- v4l2 (USB webcam) or CSI (nvarguscamerasrc) -- and the default
output is a window on screen. It can also record in parallel with the display (--record),
or run with no image at all (--no-display).

Usage:
    python3 complete_pipeline_tRT_camera_18.py                       # /dev/video0
    python3 complete_pipeline_tRT_camera_18.py --device /dev/video1
    python3 complete_pipeline_tRT_camera_18.py --camera csi --sensor-id 0
    python3 complete_pipeline_tRT_camera_18.py --record             # also writes mp4
    python3 complete_pipeline_tRT_camera_18.py --no-display         # console only

The output folder (camera_<device> by default) is REUSED: the database inside it is loaded
at startup, so people enrolled yesterday are recognised today. It is also saved every
DATABASE_SAVE_INTERVAL_S seconds during the run, not only on exit: a live session can last
hours, and a hard Ctrl+C or a power cut must not lose everything that was enrolled.

Presence counts only identities CONFIRMED by a pass through the recognition model
(checks_confirmed >= 1), and it also counts tracks that did not pass the gates in the
current frame -- a turned head is not a departure.

Two behaviours worth knowing before reading the code:

  COOLDOWN. Once a track has a confirmed identity it is re-checked every
  IDENTIFIED_INTERVAL_FRAMES instead of VERIFY_INTERVAL_FRAMES, and in between it gets no
  landmarks at all (they were 63% of the probe time). The cost: if the tracker swaps the
  person under the same id the wrong label stays until the next check. A track without an
  identity still gets landmarks every frame, so enrollment is not affected.

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
# Samples below MIN_BLUR still compete (an absolute threshold depends too much on the
# source); only this safety floor is dropped outright.
BLUR_REJECT_FACTOR = 0.25

VERIFY_INTERVAL_FRAMES = 15   # at most this many frames between checks of a track
RETRY_INTERVAL_ON_FAIL = 7    # sooner retry when the quality gate failed
IDENTIFIED_INTERVAL_FRAMES = 30   # the cooldown; see the header
LABEL_HISTORY_SIZE = 5        # recent decisions kept for the majority vote
TRACK_TIMEOUT_FRAMES = 300    # absent frames after which a track is forgotten
PRUNE_CHECK_INTERVAL = 90     # how often we look for dead tracks
IDENTITY_FLIP_CHECKS = 2      # consecutive contradicting checks needed to drop an identity
IDENTITY_STALE_FRAMES = 150   # after this long without confirmation: yellow, with "?"

RETURN_WINDOW_S = 15.0        # a return within this window prints [RETURNED]
ABSENCE_GRACE_S = 1.0         # below this, a gap is a flicker, not a departure
ANNOUNCE_DEPARTURE = True     # also print [PRESENT]/[LEFT], not only returns
DATABASE_SAVE_INTERVAL_S = 60.0   # periodic database save; 0 = only on exit

QUALITY_WINDOW_FRAMES = 6     # frames before the deadline in which samples are collected
CANDIDATE_INTERVAL_FRAMES = 2 # frames between samples inside that window
QUALITY_GOOD_ENOUGH = 0.65    # good enough to check immediately
QUALITY_MIN = 0.30            # below this the recognition model is not worth running
QUALITY_WEIGHTS = {"yaw": 0.30, "pitch": 0.20, "roll": 0.10, "sharp": 0.20, "size": 0.20}

VERIFY_THRESHOLD = 0.42       # cosine score from which a face is recognised
ENROLL_MARGIN = 0.10          # uncertainty band below the recognition threshold; below it
                              # the verdict is "unknown" and the track may be enrolled
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
LANDMARK_ONLY_WHEN_NEEDED = DRAW_ALL_LANDMARKS = DRAW_OVERLAY = True
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
presence = None             # built in main() -- the probe takes no extra parameter
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
        """Atomic write into save_path (the run folder), never over the seed.

        Also called mid-run, from the GLib loop, while add() runs on the GStreamer thread;
        add() does the vstack before the append, so we snapshot both fields and write the
        consistent prefix.
        """
        labels, matrix = list(self.labels), self.matrix
        count = min(len(labels), matrix.shape[0])
        tmp = self.save_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({labels[i]: matrix[i].tolist() for i in range(count)}, f)
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
# PER-TRACK STATE AND PRESENCE
# ============================================================

class Shot:
    """One usable capture of a face: the aligned crop plus what it scored."""

    __slots__ = ("aligned", "quality", "frontality", "blur", "size", "frame")

    def __init__(self, aligned, quality, frontality, blur, size, frame):
        self.aligned, self.quality, self.frontality = aligned, quality, frontality
        self.blur, self.size, self.frame = blur, size, frame

class Best:
    """The highest-ranking shot offered so far, by a given key.

    Every track keeps two of these and they differ only in the key: the check window ranks
    on quality alone, while the enrollment prototype ranks on frontality first and only
    accepts shots that could be a database entry at all. Filter first, choose after --
    otherwise the winner fails the size gate later; and rank on frontality, because quality
    mixes in sharpness and size, so a large sharp profile would beat a smaller frontal shot
    and profiles are useless for recognition.
    """

    __slots__ = ("key", "gate", "shot")

    def __init__(self, key, gate=None):
        self.key, self.gate, self.shot = key, gate, None

    def offer(self, shot):
        if self.gate is not None and not self.gate(shot):
            return
        if self.shot is None or self.key(shot) > self.key(self.shot):
            self.shot = shot

    def clear(self):
        self.shot = None

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
        # The shot that will go through the recognition model at the next check, reset
        # after every one; and the best prototype seen over the WHOLE track, so enrollment
        # uses the best shot the person ever gave, not whatever the last window caught.
        self.window = Best(lambda shot: shot.quality)
        self.prototype = Best(lambda shot: (shot.frontality, shot.quality),
                              gate=usable_as_prototype)

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

class PresenceLog:
    """Who is in frame, who left and who came back.

    Keyed by database NAME, not by tracker id: one person can go through several ids
    without ever leaving. Time is the wall clock, because the window is in seconds and the
    frame rate varies on a live run.
    """

    def __init__(self):
        self.present = {}       # visible name -> last time seen
        self.absent = {}        # departed name -> time last seen

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
                events.append(("appeared", name, None))
            else:
                gone = now - left_at
                events.append(("returned" if gone <= RETURN_WINDOW_S else "reappeared",
                               name, gone))

        for name, last_seen in list(self.present.items()):
            if name in names or now - last_seen < ABSENCE_GRACE_S:
                continue
            # The departure time is the last time seen, not the moment we noticed:
            # otherwise the grace period would count as absence.
            del self.present[name]
            self.absent[name] = last_seen
            events.append(("left", name, None))
        return events

class FaceSample:
    """A face visible in the current frame, with everything computed for it."""

    __slots__ = ("track_id", "state", "obj_meta", "crop", "blur", "size", "box",
                 "landmarks", "five_points", "aligned", "quality", "frontality",
                 "wants_sample", "shot")

    def __init__(self, track_id, state, obj_meta, crop, blur, size, box):
        self.track_id, self.state, self.obj_meta = track_id, state, obj_meta
        self.crop, self.blur, self.size = crop, blur, size
        self.box = box                  # (x1, y1, x2, y2) in frame coordinates
        self.landmarks = self.five_points = self.aligned = self.shot = None
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

def usable_as_prototype(shot):
    """Whether a shot is good enough to become somebody's database entry."""
    return (shot.size >= ENROLL_MIN_FACE and shot.blur >= ENROLL_MIN_BLUR
            and shot.quality >= ENROLL_MIN_QUALITY
            and shot.frontality >= ENROLL_MIN_FRONTALITY)

def ready_to_enroll(state):
    """Proof it is not in the database, plus a good shot to enroll it from.

    "identity is None" keeps a held identity from being enrolled a second time under a new
    name after one bad window. There is no test on the score: verify() answers
    LABEL_UNKNOWN only below VERIFY_THRESHOLD - ENROLL_MARGIN, so a track that built an
    unknown streak is already under that line by construction.
    """
    return (AUTO_ENROLL and not state.enrolled and state.identity is None
            and state.prototype.shot is not None
            and state.unknown_streak >= ENROLL_MIN_CHECKS)

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

    # --- Step 4: every aligned face offers its shot to the enrollment prototype ---
    for face in faces:
        if face.landmarks is not None and face.aligned is not None:
            face.shot = Shot(face.aligned, face.quality, face.frontality, face.blur,
                             face.size, frame_number)
            face.state.prototype.offer(face.shot)

    # --- Step 5: which faces enter the contest for "best frame" ---
    for face in faces:
        if not face.wants_sample:
            continue
        face.state.last_candidate_frame = frame_number   # keep the rate even if discarded
        if face.shot is not None and face.blur >= MIN_BLUR * BLUR_REJECT_FACTOR:
            face.state.window.offer(face.shot)

    # --- Step 6: for whom we run recognition in this frame ---
    to_recognize = []
    for track_id, state in active.items():
        due = frame_number - state.last_checked_frame >= state.deadline_gap
        best = state.window.shot
        if best is not None and best.quality >= QUALITY_MIN and (
                best.quality >= QUALITY_GOOD_ENOUGH or due):
            state.last_checked_frame, state.last_check_failed = frame_number, False
            to_recognize.append((track_id, state, best))
        elif due:
            state.last_checked_frame, state.last_check_failed = frame_number, True
            state.window.clear()

    # --- Step 7: the check (who it is), without enrollment ---
    for (track_id, state, best), embedding in zip(
            to_recognize, embed_aligned([shot.aligned for _, _, shot in to_recognize])):
        label, score = face_database.verify(embedding)
        state.window.clear()

        if AUTO_ENROLL and label == LABEL_UNKNOWN and not state.enrolled:
            state.unknown_streak += 1
        else:
            state.unknown_streak = 0

        state.history.append(label)
        voted = majority_label(state.history)
        outcome = state.apply_decision(voted, score, frame_number)

        # Printed only when something changes, not at every re-confirmation.
        if outcome == "confirmed" and state.checks_confirmed == 1:
            print(f"[ALERT] track {track_id} -> {state.identity} (score={score:.3f}, "
                  f"quality={best.quality:.2f}, frame={frame_number})")
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
    ready = [(tid, st, st.prototype.shot) for tid, st in active.items()
             if ready_to_enroll(st)]
    for (track_id, state, shot), embedding in zip(
            ready, embed_aligned([s.aligned for _, _, s in ready])):
        name = enroll(embedding)
        state.confirm_enrolled(name, frame_number)
        print(f"[ENROLL] track {track_id} -> {name} from frame {shot.frame} "
              f"({shot.size}px, frontality {shot.frontality:.2f}, "
              f"quality {shot.quality:.2f}); {len(face_database)} identities")

    # --- Step 9: who is in frame, who left, who came back ---
    # Gathered after this frame's decisions, so a return is announced in the very frame it
    # was confirmed. Held tracks count as present -- somebody turned away has not left.
    visible = list(active.values()) + [st for st, _ in held_objects]
    present_names = {st.identity for st in visible
                     if st.identity is not None and st.checks_confirmed >= 1}
    for kind, name, gone in presence.update(present_names, time.monotonic()):
        if kind == "returned":
            print(f"[RETURNED] {name} came back after {gone:.1f} s")
        elif kind == "reappeared":
            print(f"[REAPPEARED] {name} is in frame again, after {gone:.1f} s (past the "
                  f"{RETURN_WINDOW_S:.0f} s window)")
        elif kind == "appeared" and ANNOUNCE_DEPARTURE:
            print(f"[PRESENT] {name} entered the frame")
        elif kind == "left" and ANNOUNCE_DEPARTURE:
            print(f"[LEFT] {name} left the frame")

    # --- Step 10: drawing ---
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

def build_camera_source(args):
    """The elements from the camera up to NVMM/NV12, in link order.

    CSI delivers NVMM/NV12 directly. v4l2 delivers system memory (YUY2 or MJPEG), so it
    needs a CPU decode/convert before nvvideoconvert can upload it; --mjpeg is worth asking
    for when the camera supports it, since it puts far less traffic on the USB bus.
    """
    if args.camera_kind == "csi":
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
    elif "busy" in lowered or "permission denied" in lowered:
        print(f"\n[CAMERA] {args.device} is busy or you have no rights on it.")
        print(f"         Who holds it:  fuser -v {args.device}   Rights: the 'video' group.")
    elif "no such device" in lowered or "cannot identify device" in lowered:
        print(f"\n[CAMERA] {args.device} does not exist:  ls -l /dev/video*")

def even(value):
    """Odd sides break NV12 and the encoder."""
    return max(2, int(round(value / 2.0)) * 2)

def build_pipeline(args, output_video):
    pipeline = Gst.Pipeline()
    print("Creating the pipeline elements...")

    camera_chain, camera_desc = build_camera_source(args)
    print(f"  source: {camera_desc}")

    # Uploads to NVMM (v4l2) and scales to the working resolution. Rotation happens before
    # the capsfilter, so the caps already carry the post-rotation sides.
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
    # live-source=1: the camera keeps producing whatever we do, so streammux must not try to
    # synchronise on timestamps. The timeout follows the camera rate, so we add no latency.
    streammux.set_property('live-source', 1)
    streammux.set_property('batched-push-timeout', int(1e6 / args.fps))

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

    # Output: window, recording, both (through a tee), or neither. All of them share one
    # nvdsosd -- the drawing is done once.
    def display_branch():
        convert = make_element("nvvideoconvert", "display-convert")
        # nv3dsink is the JetPack 5 sink and takes NVMM directly; nveglglessink needs
        # nvegltransform before it on Jetson. sync=False: on a live source, show the latest
        # processed frame instead of waiting for its nominal timestamp.
        sink = make_element(["nv3dsink", "nveglglessink", "xvimagesink", "autovideosink"],
                            "display")
        for prop in ("sync", "async"):
            if sink.find_property(prop):
                sink.set_property(prop, False)
        if sink.get_factory().get_name() == "nveglglessink":
            return [convert, make_element("nvegltransform", "egl-transform"), sink]
        return [convert, sink]

    def record_branch():
        caps_out = make_element("capsfilter", "caps-record")
        encoder = make_element(["nvv4l2h264enc", "x264enc"], "encoder")
        bitrate = int(args.width * args.height * 4)   # ~4 bits/pixel/s: 8 Mbit/s at 1080p
        if encoder.get_factory().get_name() == "nvv4l2h264enc":
            encoder.set_property('bitrate', bitrate)
            # The default profile is Baseline (no CABAC, no B-frames), which is what looks
            # soft at the same bitrate; High costs the same on NVENC. A keyframe every
            # couple of seconds keeps the file readable from the middle.
            for prop, value in (("profile", 4), ("control-rate", 0),
                                ("peak-bitrate", int(bitrate * 1.5)),
                                ("iframeinterval", args.fps * 2)):
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
        return [make_element("nvvideoconvert", "record-convert"), caps_out, encoder,
                make_element("h264parse", "parser"), make_element("qtmux", "muxer"), sink]

    branches = []
    if not args.no_display:
        branches.append(display_branch())
    if output_video:
        branches.append(record_branch())

    # Threads: [camera -> detector -> tracker] | [probe] | [osd -> output]
    queue_pre = make_queue("queue-detect")
    chain = [streammux, pgie, tracker, make_queue("queue-probe"), vidconv_osd, caps_rgba,
             make_queue("queue-out")]
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

    try:
        link_chain(camera_chain)
        if not camera_chain[-1].link(vidconv_in):
            raise RuntimeError("cannot reach the input converter")
    except RuntimeError as error:
        raise RuntimeError(f"Cannot link the camera elements ({error}). Most likely the "
                           f"camera cannot deliver {args.capture_width}x"
                           f"{args.capture_height}@{args.fps}"
                           f"{' MJPEG' if args.mjpeg else ''}; see what it can with "
                           f"v4l2-ctl -d {args.device} --list-formats-ext.")
    link_chain([vidconv_in, caps_in, queue_pre])
    queue_pre.get_static_pad("src").link(streammux.get_request_pad("sink_0"))
    link_chain(chain)

    # Each branch starts with its own queue, otherwise the slowest one (the encoder) would
    # hold up the window as well.
    if tee is not None:
        for index, branch in enumerate(branches):
            queue = make_queue(f"queue-branch-{index}")
            pipeline.add(queue)
            if tee.get_request_pad("src_%u").link(
                    queue.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
                raise RuntimeError(f"Cannot link the tee to branch {index}.")
            link_chain([queue] + branch)

    # The box filter goes on the detector's output, i.e. before the tracker.
    if PRETRACK_FILTER:
        pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER,
                                             detection_filter_probe, 0)
    caps_rgba.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, media_probe, 0)
    return pipeline

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
        description="Live face recognition on a camera: a window on screen, automatic "
                    "enrollment, a persistent database and an announcement when somebody "
                    "comes back. Thresholds are constants at the top of this file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    camera = parser.add_argument_group("camera")
    camera.add_argument("--camera", default="auto", choices=["auto", "v4l2", "csi"],
                        help="v4l2 = USB webcam (--device), csi = on-board sensor")
    camera.add_argument("--device", default="/dev/video0", help="the v4l2 camera device file")
    camera.add_argument("--sensor-id", type=int, default=None, help="the CSI sensor index")
    camera.add_argument("--capture-width", type=int, default=1280, help="the width REQUESTED "
                        "from the camera; it must be one it can deliver "
                        "(v4l2-ctl --list-formats-ext)")
    camera.add_argument("--capture-height", type=int, default=720,
                        help="the height requested from the camera")
    camera.add_argument("--fps", type=int, default=30,
                        help="the frame rate requested from the camera")
    camera.add_argument("--mjpeg", action="store_true", help="ask for MJPEG instead of raw: "
                        "far less USB traffic, so higher rates; costs a CPU decode")

    add = parser.add_argument
    add("--output", default=None, help="the output folder (default: camera_<device>); if it "
        "exists, it is reused together with its database")
    add("--database", default=FACE_DATABASE_PATH, help="the seed database, used only when "
        "the output folder does not have one yet; it is never modified")
    add("--pgie-config", default=YOLO_CONFIG_PATH,
        help="the nvinfer config for the face detector")
    add("--tracker-config", default=TRACKER_CONFIG_PATH,
        help="the low-level config for nvtracker")
    add("--landmark-engine", default=PFLD_MODEL_PATH)
    add("--recognition-engine", default=RECOGNITION_MODEL_PATH)
    add("--rotate", default="0", choices=["0", "90", "180", "270"],
        help="rotation applied to the camera image, clockwise (sideways-mounted sensors)")
    add("--record", nargs="?", const=True, default=False, help="also record an mp4, in "
        "parallel with the window; without a value, into the run folder")
    add("--no-display", action="store_true", help="do not open the window. Without --record "
        "only the console is left: recognition, the announcements and the database are "
        "unchanged, but the encoder and the display sink leave the board")
    add("--no-enroll", action="store_true", help="do not add new identities to the database")
    add("--reset-db", action="store_true", help="ignore the database in the folder and start "
        "from the --database seed (with --database '' it starts empty)")
    return parser.parse_args(argv)

def main():
    global AUTO_ENROLL, DRAW_OVERLAY, DRAW_ALL_LANDMARKS, presence

    args = parse_args()
    AUTO_ENROLL = not args.no_enroll
    # Drawing only makes sense if somebody sees it: a window or a recording.
    DRAW_OVERLAY = DRAW_ALL_LANDMARKS = (not args.no_display) or bool(args.record)
    args.fps = max(1, args.fps)
    check_pyds_api()

    args.pgie_config = resolve_config(args.pgie_config, "the nvinfer config (the detector)")
    tracker_source = resolve_config(args.tracker_config, "the nvtracker config")

    # Gst.init before any factory lookup: camera_kind searches for elements.
    Gst.init(None)
    args.camera_kind = camera_kind(args)
    if args.camera_kind == "v4l2" and not os.path.exists(args.device):
        print(f"[WARNING] {args.device} does not exist right now:  ls -l /dev/video*")

    args.rotation = int(args.rotate)
    args.width, args.height = args.capture_width, args.capture_height
    if args.rotation in (90, 270):
        args.width, args.height = args.height, args.width
    args.width, args.height = even(args.width), even(args.height)

    print(f"Camera:  {args.camera_kind}, capture "
          f"{args.capture_width}x{args.capture_height}@{args.fps}"
          + (", MJPEG" if args.mjpeg else ""))
    print(f"Working: {args.width}x{args.height}"
          + (f", rotating by {args.rotation} degrees" if args.rotation else ""))

    # The folder is named after the camera, so its database is found again on the next run
    # and people enrolled yesterday are recognised today. Only the recordings are numbered.
    stem = "camera_" + (f"sensor{args.sensor_id or 0}" if args.camera_kind == "csi"
                        else os.path.basename(args.device.rstrip("/")) or "video0")
    output_dir = args.output or os.path.join(SCRIPT_DIR, stem)
    reused = os.path.isdir(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # --record without a value writes into the run folder; --record path writes there.
    if not args.record:
        output_video = None
    elif args.record is True:
        used = [int(m.group(1)) for m in
                (re.match(rf"^{re.escape(stem)}_(\d+)\.mp4$", n) for n in os.listdir(output_dir))
                if m]
        output_video = os.path.join(output_dir, f"{stem}_{max(used, default=0) + 1:03d}.mp4")
    else:
        output_video = args.record

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

    print(f"Output:  {output_dir} ({'reused' if reused else 'new'})")
    print(f"Screen:  {'no (--no-display)' if args.no_display else 'window'}"
          + (f", recording into {output_video}" if output_video else ""))
    print(f"Presence: returns within {RETURN_WINDOW_S:.0f} s are announced (a departure is "
          f"declared after {ABSENCE_GRACE_S:.1f} s without any box carrying the name)")

    load_models(args.landmark_engine, args.recognition_engine, database_in, database_out)
    if database_in == database_out:
        print("Continuing the folder's database: whatever the previous runs enrolled is "
              "recognised now.")
    identities_start = len(face_database)
    presence = PresenceLog()

    pipeline = build_pipeline(args, output_video)
    loop = GLib.MainLoop()
    failure = []    # remembered so the failure also shows up in the exit code

    def bus_call(bus, message, loop):
        if message.type == Gst.MessageType.EOS:
            print("The stream ended.")
            loop.quit()
        elif message.type == Gst.MessageType.WARNING:
            print("GStreamer warning: %s: %s" % message.parse_warning())
        elif message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            print(f"GStreamer error: {error}: {debug}")
            failure.append(str(error))
            explain_camera_error(f"{error} {debug}", args)
            loop.quit()
        return True

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Periodic save: a live session can last hours, and a power cut or a second Ctrl+C would
    # lose everything enrolled in the meantime.
    def save_database_tick():
        if len(face_database) != face_database.source_count:
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
        if presence.present:
            print(f"  in frame at stop: {', '.join(sorted(presence.present))}")
        if output_video:
            print(f"  {output_video}")
        print(f"  {database_out}")

    return 1 if failure else 0

if __name__ == '__main__':
    sys.exit(main())
