import os
import glob
import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = "/workspace/_landmark/pfld.onnx"
INPUT_DIR = "/workspace/output_crops"
ALIGNED_OUTPUT_DIR = "/workspace/output_aligned"
MAX_IMAGES = 15

os.makedirs(ALIGNED_OUTPUT_DIR, exist_ok=True)

session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# Template canonic ArcFace/InsightFace, 5 puncte, pentru output aliniat 112x112
# Ordine: ochi stang, ochi drept, nas, colt gura stanga, colt gura dreapta
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

ALIGN_SIZE = 112


def get_5_points(landmark68):
    # Indexare standard iBUG/300W, 0-indexata
    left_eye = landmark68[36:42].mean(axis=0)
    right_eye = landmark68[42:48].mean(axis=0)
    nose_tip = landmark68[30]
    mouth_left = landmark68[48]
    mouth_right = landmark68[54]
    return np.array([left_eye, right_eye, nose_tip, mouth_left, mouth_right], dtype=np.float32)


image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")))[:MAX_IMAGES]
print(f"Aliniez {len(image_paths)} imagini...")

for path in image_paths:
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        continue
    h, w = img_bgr.shape[:2]

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (112, 112))
    tensor = img_resized.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)

    outputs = session.run(None, {input_name: tensor})
    landmark = outputs[0].reshape(-1, 2)

    landmark_pixels = landmark.copy()
    landmark_pixels[:, 0] *= w
    landmark_pixels[:, 1] *= h

    five_points = get_5_points(landmark_pixels)

    # Transformare de similaritate: mapeaza cele 5 puncte detectate peste template-ul canonic
    transform_matrix, _ = cv2.estimateAffinePartial2D(
        five_points, ARCFACE_TEMPLATE, method=cv2.LMEDS
    )

    if transform_matrix is None:
        print(f"{os.path.basename(path)} -> nu s-a putut calcula transformarea, sar peste")
        continue

    aligned = cv2.warpAffine(img_bgr, transform_matrix, (ALIGN_SIZE, ALIGN_SIZE), borderValue=0)

    out_path = os.path.join(ALIGNED_OUTPUT_DIR, os.path.basename(path))
    cv2.imwrite(out_path, aligned)

print(f"Gata. Fetele aliniate sunt in {ALIGNED_OUTPUT_DIR}")
