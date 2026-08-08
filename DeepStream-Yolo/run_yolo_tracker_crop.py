import sys
import os
import numpy as np
import cv2
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib
import pyds
import signal

SAVE_DIR = "/workspace/output_crops"
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_EVERY_N_FRAMES = 15   # throttling: salvează un crop per ID la fiecare N frame-uri
MIN_CONFIDENCE = 0.5       # ignoră detecțiile slabe

last_saved_frame = {}      # track_id -> ultimul frame la care s-a salvat


def crop_probe(pad, info, u_data):
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

        # Extrage frame-ul complet ca numpy array (RGBA, pentru ca am convertit caps-urile mai jos)
        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        frame_image = np.array(n_frame, copy=True, order='C')
        frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGBA2BGR)

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
            x2 = min(frame_image.shape[1], int(rect.left + rect.width))
            y2 = min(frame_image.shape[0], int(rect.top + rect.height))

            should_save = (
                confidence >= MIN_CONFIDENCE
                and x2 > x1 and y2 > y1
                and (track_id not in last_saved_frame
                     or frame_number - last_saved_frame[track_id] >= SAVE_EVERY_N_FRAMES)
            )

            if should_save:
                crop = frame_image[y1:y2, x1:x2]
                filename = os.path.join(SAVE_DIR, f"id{track_id}_frame{frame_number}_conf{confidence:.2f}.jpg")
                cv2.imwrite(filename, crop)
                last_saved_frame[track_id] = frame_number

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
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 40000)

    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property('config-file-path', "config_infer_best_v2.txt")

    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    tracker.set_property('tracker-width', 640)
    tracker.set_property('tracker-height', 384)
    tracker.set_property('gpu-id', 0)
    tracker.set_property('ll-lib-file', "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property('ll-config-file', "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml")

    vidconv2 = Gst.ElementFactory.make("nvvideoconvert", "converter2")

    # NOU: convertim la RGBA ca sa putem citi buffer-ul din Python/OpenCV in probe
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

    # NOU: atasam probe-ul de crop pe pad-ul src al caps_rgba (deja are format RGBA negociat)
    rgba_src_pad = caps_rgba.get_static_pad("src")
    rgba_src_pad.add_probe(Gst.PadProbeType.BUFFER, crop_probe, 0)

    print("Pornire procesare. Apasa Ctrl+C pentru a opri.")
    pipeline.set_state(Gst.State.PLAYING)
    loop = GLib.MainLoop()

    # Bus watch: prinde EOS si erori, opreste bucla curat
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

    # Handler pentru Ctrl+C: trimite EOS in loc sa intrerupa brutal bucla
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
        print("Pipeline oprit cu succes!")


if __name__ == '__main__':
    main()
