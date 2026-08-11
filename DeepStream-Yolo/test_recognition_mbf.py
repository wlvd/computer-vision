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

session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print("Input model:", session.get_inputs()[0].shape)
print("Output model:", session.get_outputs()[0].shape)


def preprocess(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (112, 112))
    img_norm = (img_rgb.astype(np.float32) - 127.5) / 127.5
    img_chw = np.transpose(img_norm, (2, 0, 1))  # HWC -> CHW
    return img_chw[np.newaxis, ...]  # NCHW: (1, 3, 112, 112)


def get_embedding(img_bgr):
    inp = preprocess(img_bgr)
    emb = session.run([output_name], {input_name: inp})[0][0]
    return emb / np.linalg.norm(emb)  # normalizare L2, pentru similaritate cosinus


# Grupam fisierele dupa track_id, extras din numele de forma id{N}_frame{M}_conf{C}.jpg
embeddings_by_id = defaultdict(list)
pattern = re.compile(r"id(\d+)_")

for path in sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg"))):
    match = pattern.search(os.path.basename(path))
    if not match:
        continue
    track_id = match.group(1)
    img = cv2.imread(path)
    if img is None:
        continue
    embeddings_by_id[track_id].append((os.path.basename(path), get_embedding(img)))

print(f"\nID-uri gasite: {list(embeddings_by_id.keys())}")
for tid, items in embeddings_by_id.items():
    print(f"  id{tid}: {len(items)} imagini")

# Similaritate INTRA-id (aceeasi persoana, cadre diferite) -> ar trebui sa fie MARE
print("\n--- Similaritate intra-ID (aceeasi persoana) ---")
intra_scores = []
for tid, items in embeddings_by_id.items():
    if len(items) < 2:
        continue
    for (name_a, emb_a), (name_b, emb_b) in itertools.combinations(items, 2):
        sim = float(np.dot(emb_a, emb_b))
        intra_scores.append(sim)
        print(f"  id{tid}: {name_a} vs {name_b} -> {sim:.4f}")

# Similaritate INTER-id (persoane diferite) -> ar trebui sa fie MICA
print("\n--- Similaritate inter-ID (persoane diferite) ---")
inter_scores = []
ids = list(embeddings_by_id.keys())
for id_a, id_b in itertools.combinations(ids, 2):
    name_a, emb_a = embeddings_by_id[id_a][0]
    name_b, emb_b = embeddings_by_id[id_b][0]
    sim = float(np.dot(emb_a, emb_b))
    inter_scores.append(sim)
    print(f"  id{id_a} vs id{id_b}: {name_a} vs {name_b} -> {sim:.4f}")

print("\n--- Rezumat ---")
if intra_scores:
    print(f"Similaritate medie intra-ID (acelasi om): {np.mean(intra_scores):.4f}")
if inter_scores:
    print(f"Similaritate medie inter-ID (oameni diferiti): {np.mean(inter_scores):.4f}")
