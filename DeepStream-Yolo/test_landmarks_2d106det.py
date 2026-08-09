import os
import glob
import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = "/workspace/_landmark/2d106det.onnx"
INPUT_DIR = "/workspace/output_crops"
OUTPUT_DIR = "/workspace/output_landmarks_106"
MAX_IMAGES = 15

INPUT_SIZE = 192
MARGIN_SCALE = 1.5  # replica exacta a formulei oficiale insightface

os.makedirs(OUTPUT_DIR, exist_ok=True)

session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")))[:MAX_IMAGES]
print(f"Testez pe {len(image_paths)} imagini...")

for path in image_paths:
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        continue

    h, w = img_bgr.shape[:2]
    max_dim = max(w, h)
    square_side = int(max_dim * MARGIN_SCALE)

    pad_w = square_side - w
    pad_h = square_side - h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    # Completam marja lipsa prin replicarea marginii (aproximare a pixelilor reali din jurul fetei)
    padded = cv2.copyMakeBorder(
        img_bgr, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_REPLICATE
    )

    img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE))
    tensor = (img_resized.astype(np.float32) - 127.5) / 128.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)

    outputs = session.run(None, {input_name: tensor})
    landmark = outputs[0].reshape(-1, 2)

    print(f"{os.path.basename(path)} -> landmark min={landmark.min():.3f} max={landmark.max():.3f}")

    # Reproiectare oficiala: (raw + 1) * (input_size // 2) -> spatiul 192x192
    landmark_192 = (landmark + 1.0) * (INPUT_SIZE // 2)

    # Din spatiul 192x192 in spatiul crop-ului PADDED (real, cu marja inclusa)
    landmark_padded = landmark_192.copy()
    landmark_padded[:, 0] = landmark_padded[:, 0] / INPUT_SIZE * square_side
    landmark_padded[:, 1] = landmark_padded[:, 1] / INPUT_SIZE * square_side

    # Desenam direct pe imaginea padded (asa vedem exact ce a "vazut" modelul, fara puncte taiate din cadru)
    annotated = padded.copy()
    for (x, y) in landmark_padded:
        cv2.circle(annotated, (int(x), int(y)), 1, (0, 255, 0), -1)

    out_path = os.path.join(OUTPUT_DIR, os.path.basename(path))
    cv2.imwrite(out_path, annotated)

print(f"Gata. Rezultatele adnotate sunt in {OUTPUT_DIR}")
