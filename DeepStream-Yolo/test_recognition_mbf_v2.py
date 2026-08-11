import os
import re
import glob
import itertools
from collections import defaultdict

import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = "/workspace/_landmark/w600k_mbf.onnx"
INPUT_DIR = "/workspace/output_aligned"

MIN_SAMPLES_PER_ID = 5  # exclude ID-urile cu mai putin de atat din statistica

session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def blur_score(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def preprocess(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (112, 112))
    img_norm = (img_rgb.astype(np.float32) - 127.5) / 127.5
    img_chw = np.transpose(img_norm, (2, 0, 1))
    return img_chw[np.newaxis, ...]


def get_embedding(img_bgr):
    inp = preprocess(img_bgr)
    emb = session.run([output_name], {input_name: inp})[0][0]
    return emb / np.linalg.norm(emb)


# Colectam tot: embedding + scor de blur, grupat pe track_id
raw_data = defaultdict(list)
pattern = re.compile(r"id(\d+)_")

for path in sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg"))):
    match = pattern.search(os.path.basename(path))
    if not match:
        continue
    track_id = match.group(1)
    img = cv2.imread(path)
    if img is None:
        continue
    b_score = blur_score(img)
    emb = get_embedding(img)
    raw_data[track_id].append((os.path.basename(path), emb, b_score))

# Raport complet, ca sa vezi intervalul real de blur din datele tale
print("--- Toate ID-urile gasite, cu numar de poze si scor de blur ---")
for tid, items in raw_data.items():
    blur_values = [b for _, _, b in items]
    print(f"  id{tid}: {len(items)} poze | blur min={min(blur_values):.1f} max={max(blur_values):.1f}")
    for name, _, b in items:
        print(f"      {name} -> blur={b:.1f}")

# Filtram: doar ID-uri cu suficiente poze
embeddings_by_id = {
    tid: [(name, emb) for name, emb, _ in items]
    for tid, items in raw_data.items()
    if len(items) >= MIN_SAMPLES_PER_ID
}

excluded = [tid for tid in raw_data if tid not in embeddings_by_id]
if excluded:
    print(f"\nID-uri excluse din statistica (mai putin de {MIN_SAMPLES_PER_ID} poze): {excluded}")

print(f"\nID-uri folosite in statistica: {list(embeddings_by_id.keys())}")

print("\n--- Similaritate intra-ID (aceeasi persoana) ---")
intra_scores = []
for tid, items in embeddings_by_id.items():
    for (name_a, emb_a), (name_b, emb_b) in itertools.combinations(items, 2):
        sim = float(np.dot(emb_a, emb_b))
        intra_scores.append(sim)

print("\n--- Similaritate inter-ID (persoane diferite, presupus) ---")
inter_scores = []
ids = list(embeddings_by_id.keys())
for id_a, id_b in itertools.combinations(ids, 2):
    for name_a, emb_a in embeddings_by_id[id_a]:
        for name_b, emb_b in embeddings_by_id[id_b]:
            sim = float(np.dot(emb_a, emb_b))
            inter_scores.append(sim)
    # afisam si o singura pereche reprezentativa, ca sa nu inunde output-ul
    name_a, emb_a = embeddings_by_id[id_a][0]
    name_b, emb_b = embeddings_by_id[id_b][0]
    print(f"  id{id_a} vs id{id_b}: {name_a} vs {name_b} -> {float(np.dot(emb_a, emb_b)):.4f}")

print("\n--- Rezumat (dupa filtrare) ---")
if intra_scores:
    print(f"Similaritate medie intra-ID: {np.mean(intra_scores):.4f} ({len(intra_scores)} perechi)")
if inter_scores:
    print(f"Similaritate medie inter-ID: {np.mean(inter_scores):.4f} ({len(inter_scores)} perechi)")
