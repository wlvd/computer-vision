import sys
import gi

gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib

def main():
    #Initializare GStreamer
    Gst.init(None)
    pipeline = Gst.Pipeline()

    if not pipeline:
        print("Eroare: Nu s-a putut crea pipeline-ul.")
        sys.exit(1)

    print("Creare elemente pipeline...")

    #Sursa (Camera USB)
    source = Gst.ElementFactory.make("v4l2src", "usb-cam")
    source.set_property('device', '/dev/video0')

    caps_v4l2 = Gst.ElementFactory.make("capsfilter", "v4l2-caps")
    caps_v4l2.set_property('caps', Gst.Caps.from_string("image/jpeg, width=1920, height=1080, framerate=30/1"))

    #Decodam MJPG-ul in format video brut
    jpegdec = Gst.ElementFactory.make("jpegdec", "jpeg-decoder")

    #Convertim in memoria GPU (NVMM) pe care o cere DeepStream
    vidconv1 = Gst.ElementFactory.make("nvvideoconvert", "converter1")
    caps_vidconv = Gst.ElementFactory.make("capsfilter", "nvmm-caps")
    caps_vidconv.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"))

    #Streammux (Obligatoriu in DeepStream inainte de inferenta)
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 40000)

    #Modelul YOLO (Modulul de inferenta)
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")

    pgie.set_property('config-file-path', "config_infer_best_v2.txt")

    #Tracker NvDCF varianta accuracy (correlation filter, rezolutie si setari pentru precizie maxima)
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    tracker.set_property('tracker-width', 960)
    tracker.set_property('tracker-height', 544)
    tracker.set_property('gpu-id', 0)
    tracker.set_property('ll-lib-file', "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property('ll-config-file', "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_accuracy.yml")

    #Conversie pentru desenarea bounding box-urilor
    vidconv2 = Gst.ElementFactory.make("nvvideoconvert", "converter2")

    #OSD (On Screen Display - deseneaza patratele pe fete)
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    #Afisarea pe ecran (Jetson necesita nvegltransform inainte de sink)
    transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    sink.set_property('sync', False) # Pentru a rula cat mai rapid fara delay

    #Adaugam toate elementele in pipeline
    elements = [source, caps_v4l2, jpegdec, vidconv1, caps_vidconv, streammux, pgie, tracker, vidconv2, nvosd, transform, sink]
    for el in elements:
        pipeline.add(el)

    print("Legare elemente pipeline...")
    #Legam Sursa -> Decodor -> Convertor
    source.link(caps_v4l2)
    caps_v4l2.link(jpegdec)
    jpegdec.link(vidconv1)
    vidconv1.link(caps_vidconv)

    #Legam la intrarea in multiplexor (Streammux)
    sinkpad = streammux.get_request_pad("sink_0")
    srcpad = caps_vidconv.get_static_pad("src")
    srcpad.link(sinkpad)

    #Legam restul procesului: Streammux -> YOLO -> Tracker -> OSD -> Ecran
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
