import sys
import gi
import cv2
import numpy as np
import pyds # Biblioteca pentru metadatele DeepStream
import onnxruntime as ort

gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib

# Sablonul matematic tinta pentru 5 puncte (rezolutie 112x112, format ArcFace/MobileFaceNet)
ARC_FACE_TEMPLATE = np.array([
    [38.2946, 51.6963], # Ochiul stang
    [73.5318, 51.5014], # Ochiul drept
    [56.0252, 71.7366], # Nas
    [41.5493, 92.3655], # Colt gura stanga
    [70.7299, 92.2041]  # Colt gura dreapta
], dtype=np.float32)

#Initializam modelele delegand calculele catre TensorRT
landmark_session = ort.InferenceSession("/workspace/_landmark/2d106det.onnx", providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
face_rec_session = ort.InferenceSession("/workspace/_landmark/w600k_mbf.onnx", providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])

# Baza de date in-memory
baza_de_date_test = {}

# Pragul de decizie (Threshold). 
# La similaritatea cosinus, 1.0 înseama identic. De obicei, 0.50 - 0.60 este un prag bun pentru început.
PRAG_SIMILARITATE = 0.55 

def calculeaza_similaritate(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def align_face(face_crop, points_106):
    # Din cele 106 puncte, extragem exact cele 5 puncte cheie.
    # Atentie: Inlocuieste indicii (ex: 38, 88, 86, 52, 61) cu indicii exacti din configuratia InsightFace 106
    points_5 = np.array([
        points_106[38], # estimativ: centrul ochiului stang
        points_106[88], # estimativ: centrul ochiului drept
        points_106[86], # estimativ: varful nasului
        points_106[52], # estimativ: gura colt stanga
        points_106[61]  # estimativ: gura colt dreapta
    ], dtype=np.float32)
    
    # Calculam matricea de transformare
    M, _ = cv2.estimateAffinePartial2D(points_5, ARC_FACE_TEMPLATE, method=cv2.LMEDS)
    if M is None:
        return face_crop # fail-safe
        
    # Aplicam deformarea spatiala pe GPU/CPU pt a aduce fata la 112x112
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
        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
                
            # Verifica daca obiectul e clasa corecta din YOLO (ex: 0 pentru fata)
            if obj_meta.class_id == 0:
                rect = obj_meta.rect_params
                x1, y1 = int(rect.left), int(rect.top)
                w, h = int(rect.width), int(rect.height)
                
                # Decupam cu grija marginile pt a nu iesi din limitele array-ului
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(n_frame.shape[1], x1 + w), min(n_frame.shape[0], y1 + h)
                face_crop = n_frame[y1:y2, x1:x2]
                
                if face_crop.size != 0:
                    # ========================================================
                    # --- 1. Executia pentru Landmarks ---
                    # Redimensionam la 192x192 si pastram factorii de scalare pentru a corecta punctele la final
                    h_orig, w_orig = face_crop.shape[:2]
                    scale_x = w_orig / 192.0
                    scale_y = h_orig / 192.0

                    face_192 = cv2.resize(face_crop, (192, 192))
                    blob_192 = (face_192.astype(np.float32) - 127.5) / 128.0
                    blob_192 = np.transpose(blob_192, (2, 0, 1)) # Transformare din HWC în CHW
                    blob_192 = np.expand_dims(blob_192, axis=0)  # Adaugam dimensiunea de lot (batch)

                    # Rulam modelul de 106 puncte
                    input_name_lmk = landmark_session.get_inputs()[0].name
                    pts_106_raw = landmark_session.run(None, {input_name_lmk: blob_192})[0][0]

                    # Punctele sunt generate pentru rezolutia de 192x192. Le scalam înapoi la dimesiunea reala a decupajului (face_crop)
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
                    
                    # Logica de auto-înregistrare: daa ditionarul e gol, salvam prima fata
                    if len(baza_de_date_test) == 0:
                        baza_de_date_test["Utilizator_Referinta"] = embedding
                        print("\n[INFO] Am înregistrat prima fta ca 'Utilizator referinta'!")
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
                    # Schimbam textul sa arate si ID-ul de tracking
                    obj_meta.text_params.display_text = f"Persoana_{obj_meta.object_id}"
                    
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
                
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
            
    return Gst.PadProbeReturn.OK

def main():
    Gst.init(None)
    pipeline = Gst.Pipeline()

    source = Gst.ElementFactory.make("v4l2src", "usb-cam")
    source.set_property('device', '/dev/video0')

    caps_v4l2 = Gst.ElementFactory.make("capsfilter", "v4l2-caps")
    caps_v4l2.set_property('caps', Gst.Caps.from_string("image/jpeg, width=1920, height=1080, framerate=30/1"))

    jpegdec = Gst.ElementFactory.make("jpegdec", "jpeg-decoder")
    
    vidconv1 = Gst.ElementFactory.make("nvvideoconvert", "converter1")
    caps_vidconv = Gst.ElementFactory.make("capsfilter", "nvmm-caps")
    caps_vidconv.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"))

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

    # Adaugam probe-ul nostru inaintea OSD-ului pt a prelucra imaginea
    osd_sink_pad = nvosd.get_static_pad("sink")
    if not osd_sink_pad:
        print("Eroare: Nu am gasit pad-ul sink de la OSD")
    else:
        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    elements = [source, caps_v4l2, jpegdec, vidconv1, caps_vidconv, streammux, pgie, tracker, vidconv2, nvosd, transform, sink]
    for el in elements:
        pipeline.add(el)

    source.link(caps_v4l2)
    caps_v4l2.link(jpegdec)
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

    print("Pornire procesare. Apasa Ctrl+C pentru a opri.")
    pipeline.set_state(Gst.State.PLAYING)
    
    try:
        loop = GLib.MainLoop()
        loop.run()
    except KeyboardInterrupt:
        print("\nOprire ceruta de utilizator...")
    finally:
        pipeline.set_state(Gst.State.NULL)
        print("Pipeline oprit cu succes!")

if __name__ == '__main__':
    main()
