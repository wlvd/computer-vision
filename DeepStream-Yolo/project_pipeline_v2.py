import sys
import gi
import cv2
import numpy as np
import pyds  # Biblioteca pentru metadatele DeepStream
import onnxruntime as ort

gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib

# Sablonul matematic tinta pentru 5 puncte (rezolutie 112x112, format ArcFace/MobileFaceNet)
ARC_FACE_TEMPLATE = np.array([
    [38.2946, 51.6963],  # Ochiul stang
    [73.5318, 51.5014],  # Ochiul drept
    [56.0252, 71.7366],  # Nas
    [41.5493, 92.3655],  # Colt gura stanga
    [70.7299, 92.2041]   # Colt gura dreapta
], dtype=np.float32)

# Cream niste optiuni de sesiune pentru a evita avertismentele de thread-uri
so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1

# ============================================================================
# FIX #1: cache pentru engine-ul TensorRT.
# Fara asta, onnxruntime compileaza engine-ul TRT de la zero de fiecare
# data cand porneste scriptul, ceea ce pe Jetson poate dura de la zeci de
# secunde pana la cateva minute -> exact simptomul de "freeze pe primul frame".
# Cu caching activat, compilarea se face o singura data; rularile ulterioare
# incarca engine-ul deja compilat de pe disc si pornesc aproape instant.
# ============================================================================
TRT_CACHE_DIR = "/workspace/trt_cache"

trt_provider_options = {
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path": TRT_CACHE_DIR,
    "trt_fp16_enable": True,   # mult mai rapid pe Jetson (Orin/Xavier/Nano)
}

providers = [
    ("TensorrtExecutionProvider", trt_provider_options),
    "CUDAExecutionProvider",
]

# Initializeaza modelele adaugand variabila sess_options=so
landmark_session = ort.InferenceSession(
    "/workspace/_landmark/2d106det.onnx",
    sess_options=so,
    providers=providers,
)
face_rec_session = ort.InferenceSession(
    "/workspace/_landmark/w600k_mbf.onnx",
    sess_options=so,
    providers=providers,
)

# Baza de date in-memory
baza_de_date_test = {}

# Pragul de decizie (Threshold).
# La similaritatea cosinus, 1.0 inseamna identic. De obicei, 0.50 - 0.60 este un prag bun pentru inceput.
PRAG_SIMILARITATE = 0.55


def calculeaza_similaritate(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def align_face(face_crop, points_106):
    # Din cele 106 puncte, extragem exact cele 5 puncte cheie.
    # Indicii [38, 88, 86, 52, 61] corespund maparii standard InsightFace
    # 106 -> 5 puncte. Daca modelul tau de landmarks NU e antrenat cu
    # schema InsightFace (ex: alt ordine/index de puncte), aceste valori
    # vor produce o aliniere gresita fara sa arunce vreo eroare vizibila
    # -> verifica vizual (deseneaza punctele pe imagine) daca ai dubii.
    points_5 = np.array([
        points_106[38],  # centrul ochiului stang
        points_106[88],  # centrul ochiului drept
        points_106[86],  # varful nasului
        points_106[52],  # gura colt stanga
        points_106[61],  # gura colt dreapta
    ], dtype=np.float32)

    # Calculam matricea de transformare
    M, _ = cv2.estimateAffinePartial2D(points_5, ARC_FACE_TEMPLATE, method=cv2.LMEDS)
    if M is None:
        return face_crop  # fail-safe

    # Aplicam deformarea spatiala pt a aduce fata la 112x112
    aligned_face = cv2.warpAffine(face_crop, M, (112, 112), flags=cv2.INTER_LANCZOS4)
    return aligned_face


def osd_sink_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    # Luam metadatele cadrului
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # Extragem pixelii imaginii din memoria GPU in Python folosind pyds
        n_frame_rgba = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)

        # ====================================================================
        # FIX #2: tot ce ține de procesarea obiectelor e in interiorul unui
        # try/finally. Inainte, daca aparea orice exceptie in bucla de obiecte
        # (index gresit, face_crop gol, eroare la un model etc.), codul sarea
        # peste pyds.unmap_nvds_buf_surface() de la finalul frame-ului.
        # Asta lasa suprafata GPU "agatata" -> la 30fps, in doar cateva
        # secunde se aduna zeci de buffere nemapate -> memoria GPU
        # (partajata cu CPU pe Jetson) se umple -> crash cu "out of memory".
        # Cu try/finally, unmap se executa GARANTAT, indiferent ce se
        # intampla in bucla de mai jos.
        # ====================================================================
        try:
            # TRANSFORMARE OBLIGATORIE DIN RGBA IN RGB (4 canale -> 3 canale)
            n_frame = cv2.cvtColor(n_frame_rgba, cv2.COLOR_RGBA2RGB)

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                # Fiecare obiect e procesat izolat: daca acesta pica, nu
                # trebuie sa duca la caderea intregului probe/pipeline si
                # nu trebuie sa mai sara peste unmap-ul de mai sus.
                try:
                    # Verifica daca obiectul e clasa corecta din YOLO (ex: 0 pentru fata)
                    if obj_meta.class_id == 0:
                        rect = obj_meta.rect_params
                        w, h = int(rect.width), int(rect.height)

                        # Adaugam un padding de 10% pentru modelul de landmarks
                        margin_x = int(w * 0.1)
                        margin_y = int(h * 0.1)

                        x1 = max(0, int(rect.left) - margin_x)
                        y1 = max(0, int(rect.top) - margin_y)
                        x2 = min(n_frame.shape[1], int(rect.left) + w + margin_x)
                        y2 = min(n_frame.shape[0], int(rect.top) + h + margin_y)

                        face_crop = n_frame[y1:y2, x1:x2]

                        if face_crop.size != 0:
                            # ================================================
                            # --- 1. Executia pentru Landmarks ---
                            # Redimensionam la 192x192 si pastram factorii de scalare pentru a corecta punctele la final
                            h_orig, w_orig = face_crop.shape[:2]
                            scale_x = w_orig / 192.0
                            scale_y = h_orig / 192.0

                            face_192 = cv2.resize(face_crop, (192, 192))
                            blob_192 = (face_192.astype(np.float32) - 127.5) / 128.0
                            blob_192 = np.transpose(blob_192, (2, 0, 1))  # Transformare din HWC in CHW
                            blob_192 = np.expand_dims(blob_192, axis=0)   # Adaugam dimensiunea de lot (batch)

                            # Rulam modelul de 106 puncte
                            input_name_lmk = landmark_session.get_inputs()[0].name
                            pts_106_raw = landmark_session.run(None, {input_name_lmk: blob_192})[0][0]
                            pts_106_raw = np.array(pts_106_raw).reshape(-1, 2)

                            # Punctele sunt generate pentru rezolutia de 192x192. Le scalam inapoi la dimesiunea reala a decupajului (face_crop)
                            pts_106 = np.zeros_like(pts_106_raw)
                            for i in range(106):
                                pts_106[i][0] = pts_106_raw[i][0] * scale_x
                                pts_106[i][1] = pts_106_raw[i][1] * scale_y

                            # --- 2. Alinierea Fetei ---
                            aligned_face = align_face(face_crop, pts_106)

                            # --- 3. Executia pentru Recunoastere (MobileFaceNet) ---
                            # aligned_face are deja 112x112 din functia align_face
                            blob_112 = (aligned_face.astype(np.float32) - 127.5) / 128.0
                            blob_112 = np.transpose(blob_112, (2, 0, 1))
                            blob_112 = np.expand_dims(blob_112, axis=0)

                            input_name_rec = face_rec_session.get_inputs()[0].name
                            embedding = face_rec_session.run(None, {input_name_rec: blob_112})[0][0]

                            # Vectorul `embedding` este acum pregatit pentru baza de date.
                            # Aplatizam vectorul de embedding (pentru a fi un simplu array 1D)
                            embedding = embedding.flatten()

                            nume_recunoscut = "Necunoscut"
                            culoare_text = pyds.NvOSD_ColorParams()

                            # Logica de auto-inregistrare: daca dictionarul e gol, salvam prima fata
                            if len(baza_de_date_test) == 0:
                                baza_de_date_test["Utilizator_Referinta"] = embedding
                                print("\n[INFO] Am inregistrat prima fata ca 'Utilizator referinta'!")
                                nume_recunoscut = "Utilizator_Referinta (Inregistrat)"
                            else:
                                # Logica de cautare
                                for nume_salvat, emb_salvat in baza_de_date_test.items():
                                    scor = calculeaza_similaritate(embedding, emb_salvat)
                                    if scor > PRAG_SIMILARITATE:
                                        nume_recunoscut = f"{nume_salvat} (Scor: {scor:.2f})"
                                        break

                            # Suprascriem textul afisat de OSD pe ecran
                            obj_meta.text_params.display_text = nume_recunoscut

                            # Optional: Schimbam fundalul textului pentru a fi vizibil
                            obj_meta.text_params.text_bg_clr.red = 0.0
                            obj_meta.text_params.text_bg_clr.green = 0.0
                            obj_meta.text_params.text_bg_clr.blue = 0.0
                            obj_meta.text_params.text_bg_clr.alpha = 0.5

                except Exception as e:
                    # Nu lasam o eroare la un singur obiect sa duca la
                    # scurgere de memorie sau la caderea probe-ului.
                    print(f"[EROARE la procesarea unui obiect] {e}")

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
        finally:
            # Se executa MEREU, indiferent daca a fost exceptie mai sus sau nu.
            pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("[INFO] End-of-stream")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"[EROARE GStreamer] {err}: {debug}")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print(f"[AVERTISMENT GStreamer] {err}: {debug}")
    return True


def main():
    Gst.init(None)
    pipeline = Gst.Pipeline()

    # ------------------------------------------------------------------
    # NOTA privind camera: multe camere USB nu ofera un mod raw (YUYV)
    # decat la fps foarte mic la 1920x1080. Solutia e sa cerem de la
    # camera fluxul ei hardware-comprimat MJPEG (fps mult mai mare) si
    # sa il decodificam cu decodorul hardware JPEG al Jetson-ului
    # (nvjpegdec), care scoate direct memorie NVMM, gata de trimis mai
    # departe in pipeline-ul DeepStream. De asta lantul e:
    # v4l2src -> caps(image/jpeg) -> jpegparse -> nvjpegdec -> nvvideoconvert
    # ------------------------------------------------------------------
    source = Gst.ElementFactory.make("v4l2src", "usb-cam")
    source.set_property('device', '/dev/video0')

    caps_v4l2 = Gst.ElementFactory.make("capsfilter", "v4l2-caps")
    caps_v4l2.set_property('caps', Gst.Caps.from_string(
        "image/jpeg, width=1920, height=1080, framerate=30/1"))

    jpegparse = Gst.ElementFactory.make("jpegparse", "jpeg-parser")

    # ------------------------------------------------------------------
    # IMPORTANT: nvjpegdec e un decodor STATELESS, gandit pentru o singura
    # imagine JPEG (nu pentru flux continuu). Folosit intr-un pipeline live
    # de la camera, recreeaza blocul hardware la FIECARE frame (vezi log:
    # "NvMMLiteBlockCreate ... Consume the extra signalling for EOS"
    # repetat continuu) -> overhead urias, resurse care nu se elibereaza
    # complet intre cicluri -> exact cauza OOM-ului dupa ~2 secunde si a
    # lipsei bounding box-urilor (pipeline-ul nu ajunge stabil pe PLAYING).
    #
    # Decodorul corect pentru flux MJPEG continuu pe Jetson e nvv4l2decoder
    # cu proprietatea mjpeg=1: foloseste motorul hardware de decodare
    # video ca pentru orice alt flux, fara sa recreeze blocul la fiecare
    # cadru.
    # ------------------------------------------------------------------
    jpegdec = Gst.ElementFactory.make("nvv4l2decoder", "jpeg-decoder")
    jpegdec.set_property('mjpeg', 1)

    vidconv1 = Gst.ElementFactory.make("nvvideoconvert", "converter1")
    caps_vidconv = Gst.ElementFactory.make("capsfilter", "nvmm-caps")
    caps_vidconv.set_property('caps', Gst.Caps.from_string(
        "video/x-raw(memory:NVMM), format=NV12"))

    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 40000)

    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property('config-file-path', "config_infer_best_v2.txt")

    # --- NOUL ELEMENT: TRACKER ---
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    # Calea catre biblioteca low-level inclusa nativ in DeepStream
    tracker.set_property('ll-lib-file', '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so')
    # Folosim fisierul de config din sistem (modifica calea daca il muti in folderul tau)
    tracker.set_property('ll-config-file', '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml')

    vidconv2 = Gst.ElementFactory.make("nvvideoconvert", "converter2")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    sink.set_property('sync', False)

    # Verificam ca toate elementele s-au creat cu succes
    elements = [source, caps_v4l2, jpegparse, jpegdec, vidconv1, caps_vidconv,
                streammux, pgie, tracker, vidconv2, nvosd, transform, sink]
    for i, el in enumerate(elements):
        if not el:
            print(f"[EROARE] Nu am putut crea elementul cu indexul {i} din lista.")
            sys.exit(1)

    # Adaugam probe-ul nostru inaintea OSD-ului pt a prelucra imaginea
    osd_sink_pad = nvosd.get_static_pad("sink")
    if not osd_sink_pad:
        print("Eroare: Nu am gasit pad-ul sink de la OSD")
    else:
        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    for el in elements:
        pipeline.add(el)

    source.link(caps_v4l2)
    caps_v4l2.link(jpegparse)
    jpegparse.link(jpegdec)
    jpegdec.link(vidconv1)
    vidconv1.link(caps_vidconv)

    sinkpad = streammux.get_request_pad("sink_0")
    srcpad = caps_vidconv.get_static_pad("src")
    srcpad.link(sinkpad)

    # Legam ordinea elementelor incluzand trackerul dupa PGIE
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(vidconv2)
    vidconv2.link(nvosd)
    nvosd.link(transform)
    transform.link(sink)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    print("Pornire procesare. Apasa Ctrl+C pentru a opri.")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nOprire ceruta de utilizator...")
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
