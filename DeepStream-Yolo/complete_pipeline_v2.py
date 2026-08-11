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
import onnxruntime as ort

# ============================================================
# CONFIGURARE
# ============================================================

YOLO_CONFIG_PATH = "config_infer_best_v2.txt"
TRACKER_CONFIG_PATH = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"

PFLD_MODEL_PATH = "/workspace/_landmark/pfld.onnx"
RECOGNITION_MODEL_PATH = "/workspace/_landmark/w600k_mbf.onnx"
FACE_DATABASE_PATH = "/workspace/DeepStream-Yolo/face_database.json"

MIN_CONFIDENCE = 0.5          # sub asta, nici nu incercam sa procesam fata
MIN_FACE_SIZE = 60            # px, latura minima a bbox-ului

# Varianta Laplacianului creste cu dimensiunea si contrastul crop-ului. 800 e o
# valoare de poza statica, sharp; pe camera live, cu miscare si compresie MJPEG,
# aproape niciun cadru nu trece de ea, deci poarta pica mereu si eticheta ramane
# "verificare..." la infinit. 100 lasa sa treaca fetele utilizabile.
MIN_BLUR = 100.0

VERIFY_INTERVAL_FRAMES = 15   # cat de des re-verificam un track activ
RETRY_INTERVAL_ON_FAIL = 5    # daca poarta de calitate a picat, reincercam mai repede
LABEL_HISTORY_SIZE = 5        # cate decizii recente pastram pentru vot majoritar
TRACK_TIMEOUT_FRAMES = 300    # dupa cate cadre de absenta stergem un track din memorie
PRUNE_CHECK_INTERVAL = 90     # la cate cadre verificam track-uri "moarte"

VERIFY_THRESHOLD = 0.42       # prag empiric, ajustat pe baza testelor offline anterioare

# --- inrolare live: fete noi intra singure in baza de date ---
AUTO_ENROLL = True
ENROLL_MAX_SCORE = 0.32       # peste asta seamana prea mult cu cineva existent ca sa fie "nou"
ENROLL_MIN_CHECKS = 4         # de cate ori la rand trebuie sa iasa "necunoscut"
ENROLL_MIN_FACE = 90          # px, mai strict decat MIN_FACE_SIZE: inrolam doar fete mari
ENROLL_MIN_BLUR = 250.0       # la fel, mai strict decat poarta de recunoastere
DB_SAVE_INTERVAL_FRAMES = 300 # cat de des scriem baza de date pe disc

ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)
ALIGN_SIZE = 112

# ============================================================
# MODELE ONNX (incarcate o singura data, globale)
# ============================================================

landmark_session = ort.InferenceSession(PFLD_MODEL_PATH)
landmark_input_name = landmark_session.get_inputs()[0].name

# Cate puncte scoate modelul decide maparea catre cele 5 puncte ArcFace.
# PFLD e antrenat pe WFLW si scoate de obicei 98; scripturile anterioare
# presupuneau 68 (iBUG). Verifica ce scrie aici la pornire.
_landmark_out = landmark_session.get_outputs()[0].shape
print(f"Model landmark-uri: output {_landmark_out}", end="")
if isinstance(_landmark_out[-1], int):
    print(f"  ->  {_landmark_out[-1] // 2} puncte")
else:
    print("  (dimensiune dinamica, se determina la prima inferenta)")

recognition_session = ort.InferenceSession(RECOGNITION_MODEL_PATH)
recognition_input_name = recognition_session.get_inputs()[0].name
recognition_output_name = recognition_session.get_outputs()[0].name

def l2(vec):
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-12 else vec


# Prototipurile TREBUIE sa aiba lungime 1: verify_embedding compara cu produs
# scalar, care e cosinus doar intre vectori normalizati. Daca scriptul de
# inrolare nu a normalizat, scorurile ies scalate aiurea si pragul nu mai are sens.
try:
    with open(FACE_DATABASE_PATH, "r") as f:
        face_database = {label: l2(vec) for label, vec in json.load(f).items()}
except FileNotFoundError:
    face_database = {}
    print(f"Nu exista {FACE_DATABASE_PATH}, pornesc cu baza de date goala.")

print(f"Baza de date incarcata: {list(face_database.keys())}")


def save_database():
    tmp = FACE_DATABASE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump({label: vec.tolist() for label, vec in face_database.items()}, f)
    os.replace(tmp, FACE_DATABASE_PATH)

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

track_states = {}

# ============================================================
# FUNCTII DE PROCESARE (identice cu formulele validate offline)
# ============================================================

def blur_score(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def run_landmark(crop_bgr):
    h, w = crop_bgr.shape[:2]
    img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (112, 112))
    tensor = img_resized.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]

    outputs = landmark_session.run(None, {landmark_input_name: tensor})
    landmark = outputs[0].reshape(-1, 2)

    landmark_pixels = landmark.copy()
    landmark_pixels[:, 0] *= w
    landmark_pixels[:, 1] *= h
    return landmark_pixels


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


def get_embedding(aligned_bgr):
    img_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    img_norm = (img_rgb.astype(np.float32) - 127.5) / 127.5
    img_chw = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]
    emb = recognition_session.run([recognition_output_name], {recognition_input_name: img_chw})[0][0]
    return emb / np.linalg.norm(emb)


def verify_embedding(embedding, db, threshold):
    best_label, best_score = None, -1.0
    for label, proto in db.items():
        score = float(np.dot(embedding, proto))
        if score > best_score:
            best_label, best_score = label, score
    if best_score >= threshold:
        return best_label, best_score
    return "necunoscut", best_score


def process_face(crop_bgr):
    """landmark -> align -> embedding -> verificare.

    Returneaza (label, score, embedding), sau None daca alinierea esueaza.
    Embedding-ul e intors si el, ca sa poata fi folosit la inrolarea live.
    """
    landmark = run_landmark(crop_bgr)
    five_points = get_5_points(landmark)
    aligned = align_face(crop_bgr, five_points)
    if aligned is None:
        return None
    embedding = get_embedding(aligned)
    label, score = verify_embedding(embedding, face_database, VERIFY_THRESHOLD)
    return label, score, embedding


def enroll(embedding):
    """Adauga o identitate noua si intoarce numele primit."""
    used = {int(m.group(1)) for m in
            (re.match(r"persoana_(\d+)$", label) for label in face_database) if m}
    name = f"persoana_{max(used, default=0) + 1}"
    face_database[name] = l2(embedding)
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

                retry_gap = RETRY_INTERVAL_ON_FAIL if state.last_check_failed else VERIFY_INTERVAL_FRAMES
                if frame_number - state.last_checked_frame >= retry_gap:
                    state.last_checked_frame = frame_number
                    crop = frame_image[y1:y2, x1:x2]
                    b_score = blur_score(crop)

                    if b_score < MIN_BLUR:
                        state.last_check_failed = True
                    else:
                        state.last_check_failed = False
                        result = process_face(crop)
                        if result is not None:
                            label, score, embedding = result

                            # Inrolare live: fata nu seamana cu nimeni din baza,
                            # a iesit "necunoscut" de mai multe ori la rand si e
                            # destul de mare si clara ca sa merite un prototip.
                            if (AUTO_ENROLL and label == "necunoscut"
                                    and not state.enrolled):
                                state.unknown_streak += 1
                                if (state.unknown_streak >= ENROLL_MIN_CHECKS
                                        and score < ENROLL_MAX_SCORE
                                        and min(w, h) >= ENROLL_MIN_FACE
                                        and b_score >= ENROLL_MIN_BLUR):
                                    label = enroll(embedding)
                                    score = 1.0
                                    state.enrolled = True
                                    state.history.clear()
                            elif label != "necunoscut":
                                state.unknown_streak = 0

                            state.history.append(label)
                            state.current_label = Counter(state.history).most_common(1)[0][0]
                            state.current_score = score

                            if state.current_label != "necunoscut":
                                print(f"[ALERTA] track {track_id} -> {state.current_label} "
                                      f"(scor={score:.3f}, frame={frame_number})")

                # scriem eticheta curenta peste text-ul implicit desenat de nvdsosd
                label_text = state.current_label if state.current_label else "..."
                display_text = f"{label_text} ({state.current_score:.2f})" if state.current_label else "verificare..."
                obj_meta.text_params.display_text = display_text
                obj_meta.text_params.set_bg_clr = 1
                obj_meta.text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)
                obj_meta.text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                obj_meta.text_params.font_params.font_size = 12

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

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
