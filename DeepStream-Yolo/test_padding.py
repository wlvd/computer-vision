import glob
import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = "/workspace/_landmark/pfld.onnx"

# Comparam explicit pozele care au picat vs cele care au mers bine la verificare
TEST_IMAGES = [
    "/workspace/output_crops/id7_frame34_conf0.90.jpg",   # GRESIT
    "/workspace/output_crops/id7_frame52_conf0.70.jpg",   # GRESIT
    "/workspace/output_crops/id7_frame307_conf0.83.jpg",  # OK
    "/workspace/output_crops/id7_frame322_conf0.88.jpg",  # OK
    "/workpace/output_crops/id7_frame82_conf0.83.jpg",   # OK
]
# ajusteaza calea daca la tine crop-urile brute sunt in alt folder

session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name


def edge_proximity(img_bgr):
    h, w = img_bgr.shape[:2]

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (112, 112))
    tensor = img_resized.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]

    outputs = session.run(None, {input_name: tensor})
    landmark = outputs[0].reshape(-1, 2)

    landmark_pixels = landmark.copy()
    landmark_pixels[:, 0] *= w
    landmark_pixels[:, 1] *= h

    min_x, max_x = landmark_pixels[:, 0].min(), landmark_pixels[:, 0].max()
    min_y, max_y = landmark_pixels[:, 1].min(), landmark_pixels[:, 1].max()

    # cat de aproape (in procent din dimensiunea crop-ului) ajunge landmark-ul de fiecare margine
    dist_left = min_x / w
    dist_right = (w - max_x) / w
    dist_top = min_y / h
    dist_bottom = (h - max_y) / h

    return dist_left, dist_right, dist_top, dist_bottom


for path in TEST_IMAGES:
    img = cv2.imread(path)
    if img is None:
        print(f"{path.split('/')[-1]} -> NU S-A GASIT")
        continue
    dl, dr, dt, db = edge_proximity(img)
    print(f"{path.split('/')[-1]:35s} -> stanga={dl:.1%} dreapta={dr:.1%} sus={dt:.1%} jos={db:.1%}")
