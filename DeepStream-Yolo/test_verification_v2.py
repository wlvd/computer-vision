import os
import re
import glob
import json
from collections import defaultdict

import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = "/workspace/_landmark/w600k_mbf.onnx"
INPUT_DIR = "/workspace/output_aligned"
DB_PATH = "/workspace/DeepStream-Yolo/face_database.json"

MIN_BLUR = 800.0
ENROLL_FRACTION = 0.6
VERIFY_THRESHOLD = 0.42

IDENTITY_MAP = {
    "persoana_A": ["5"],
    "persoana_B": ["6"],
    "persoana_C": ["7", "8"],
}

# NOU: identitatile din aceasta lista NU intra in baza de date,
# dar TOATE pozele lor merg la verificare, ca sa vezi daca sunt respinse corect ca "necunoscut"
EXCLUDE_FROM_ENROLLMENT = ["persoana_B"]

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


files_by_track = defaultdict(list)
pattern = re.compile(r"id(\d+)_")
for path in sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg"))):
    match = pattern.search(os.path.basename(path))
    if match:
        files_by_track[match.group(1)].append(path)

identity_samples = {}
for identity, track_ids in IDENTITY_MAP.items():
    samples = []
    for tid in track_ids:
        for path in files_by_track.get(tid, []):
            img = cv2.imread(path)
            if img is None:
                continue
            b = blur_score(img)
            if b < MIN_BLUR:
                continue
            emb = get_embedding(img)
            samples.append((path, emb, b))
    identity_samples[identity] = samples
    tag = " (EXCLUS din inrolare)" if identity in EXCLUDE_FROM_ENROLLMENT else ""
    print(f"{identity}: {len(samples)} poze utilizabile{tag}")

database = {}
test_set = []

for identity, samples in identity_samples.items():
    if len(samples) < 4:
        print(f"ATENTIE: {identity} are prea putine poze, sar peste.")
        continue

    if identity in EXCLUDE_FROM_ENROLLMENT:
        # nu construim prototip, dar trimitem TOATE pozele la verificare
        for path, emb, _ in samples:
            test_set.append((path, emb, identity))
        continue

    n_enroll = max(1, int(len(samples) * ENROLL_FRACTION))
    enroll_samples = samples[:n_enroll]
    test_samples = samples[n_enroll:]

    enroll_embeddings = np.stack([emb for _, emb, _ in enroll_samples])
    prototype = enroll_embeddings.mean(axis=0)
    prototype = prototype / np.linalg.norm(prototype)
    database[identity] = prototype.tolist()

    for path, emb, _ in test_samples:
        test_set.append((path, emb, identity))

print(f"\nBaza de date: {len(database)} identitati -> {list(database.keys())}")
with open(DB_PATH, "w") as f:
    json.dump(database, f)


def verify_face(embedding, db, threshold):
    best_label, best_score = None, -1.0
    for label, proto in db.items():
        score = float(np.dot(embedding, np.array(proto)))
        if score > best_score:
            best_label, best_score = label, score
    if best_score >= threshold:
        return best_label, best_score
    return "necunoscut", best_score


print(f"\n--- Verificare pe {len(test_set)} poze de test ---")
correct = 0
for path, emb, true_identity in test_set:
    predicted, score = verify_face(emb, database, VERIFY_THRESHOLD)

    if true_identity in EXCLUDE_FROM_ENROLLMENT:
        # persoana neinrolata: raspunsul corect e "necunoscut"
        is_correct = (predicted == "necunoscut")
    else:
        is_correct = (predicted == true_identity)

    correct += is_correct
    marker = "OK" if is_correct else "GRESIT"
    print(f"  [{marker}] {os.path.basename(path)} | adevarat={true_identity} | prezis={predicted} (scor={score:.4f})")

if test_set:
    print(f"\nAcuratete: {correct}/{len(test_set)} = {correct/len(test_set):.1%}")
