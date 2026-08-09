import os
import glob
import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = "/workspace/_landmark/pfld.onnx"
INPUT_DIR = "/workspace/output_crops"
OUTPUT_DIR = "/workspace/output_landmarks"
MAX_IMAGES = 15  # testam doar pe un esantion, nu pe tot folderul

os.makedirs(OUTPUT_DIR, exist_ok=True)

session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
print("Input model:", session.get_inputs()[0].shape)
print("Output model:", session.get_outputs()[0].shape)

image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")))[:MAX_IMAGES]
print(f"Testez pe {len(image_paths)} imagini...")

for path in image_paths:
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        continue

    h, w = img_bgr.shape[:2]

    # Preprocesare identica cu exemplul oficial: RGB, resize 112x112, /255, CHW, fara normalizare mean/std
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (112, 112))
    tensor = img_resized.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))  # HWC -> CHW
    tensor = np.expand_dims(tensor, axis=0)   # adaugam batch dim

    outputs = session.run(None, {input_name: tensor})
    landmark = outputs[0].reshape(-1, 2)

    # IMPORTANT: verificam intervalul brut inainte sa presupunem scalarea
    print(f"{os.path.basename(path)} -> landmark min={landmark.min():.3f} max={landmark.max():.3f} shape={landmark.shape}")

    # Presupunere: coordonate normalizate [0,1] relativ la crop-ul 112x112
    # Daca min/max printat mai sus arata valori de genul 0..112 (nu 0..1), comenteaza linia de mai jos si scaleaza direct fara inmultire cu w/h.
    landmark_pixels = landmark.copy()
    landmark_pixels[:, 0] *= w
    landmark_pixels[:, 1] *= h

    annotated = img_bgr.copy()
    for (x, y) in landmark_pixels:
        cv2.circle(annotated, (int(x), int(y)), 1, (0, 255, 0), -1)

    out_path = os.path.join(OUTPUT_DIR, os.path.basename(path))
    cv2.imwrite(out_path, annotated)

print(f"Gata. Rezultatele adnotate sunt in {OUTPUT_DIR}")
