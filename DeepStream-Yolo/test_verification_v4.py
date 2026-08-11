"""Construieste baza de date de fete si o verifica, folosind TensorRT in loc de ONNX Runtime.

Identic cu v3 ca logica si ca preprocesare — deci embedding-urile sunt comparabile
cu cele produse inainte. Singura schimbare e motorul de inferenta: .engine prin
TensorRT, nu .onnx prin onnxruntime, pentru ca pe JetPack 5.x nu exista wheel de
onnxruntime-gpu si totul cadea pe CPU.

Clasa TrtModel e scrisa ca sa poata fi mutata ca atare in complete_pipeline.py.
"""

import os
import re
import glob
import json
from collections import defaultdict

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  — creeaza contextul CUDA la import

# ============================================================
# CONFIGURARE
# ============================================================

ENGINE_PATH = "/workspace/_landmark/w600k_mbf.engine"
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

# Identitatile din aceasta lista NU intra in baza de date, dar TOATE pozele lor
# merg la verificare, ca sa vezi daca sunt respinse corect ca "necunoscut".
EXCLUDE_FROM_ENROLLMENT = ["persoana_B"]

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ============================================================
# TENSORRT
# ============================================================


class TrtModel:
    """Un .engine TensorRT, cu buffere alocate o singura data.

    API-ul cu bindings indexate (execute_async_v2) e cel care merge pe TensorRT
    8.x, adica ce vine cu JetPack 5.x. In TensorRT 10 e scos si trebuie inlocuit
    cu set_tensor_address + execute_async_v3.
    """

    def __init__(self, engine_path):
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(
                f"Nu gasesc engine-ul: {engine_path}\n"
                f"Construieste-l din .onnx cu trtexec (vezi instructiunile)."
            )

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(
                f"Engine-ul {engine_path} nu a putut fi deserializat. Cel mai des "
                f"inseamna ca a fost construit cu alta versiune de TensorRT sau pe "
                f"alta placa: engine-urile NU sunt portabile, se rebuild-uiesc local."
            )

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Pasul 1: fixam formele intrarilor. Daca engine-ul a fost construit cu
        # batch dinamic, dimensiunea apare ca -1 si trebuie concretizata acum,
        # altfel formele iesirilor nu se pot calcula.
        for index in range(self.engine.num_bindings):
            if not self.engine.binding_is_input(index):
                continue
            shape = tuple(self.engine.get_binding_shape(index))
            if -1 in shape:
                self.context.set_binding_shape(
                    index, tuple(1 if dim == -1 else dim for dim in shape)
                )

        # Pasul 2: alocam cate un buffer host (pagelocked, pentru copiere async)
        # si unul device pentru fiecare binding.
        self.bindings = [0] * self.engine.num_bindings
        self.inputs, self.outputs = [], []

        for index in range(self.engine.num_bindings):
            shape = tuple(self.context.get_binding_shape(index))
            dtype = trt.nptype(self.engine.get_binding_dtype(index))
            host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            device = cuda.mem_alloc(host.nbytes)

            self.bindings[index] = int(device)
            entry = {
                "name": self.engine.get_binding_name(index),
                "shape": shape,
                "dtype": dtype,
                "host": host,
                "device": device,
            }
            if self.engine.binding_is_input(index):
                self.inputs.append(entry)
            else:
                self.outputs.append(entry)

    def describe(self):
        print(f"Engine: {os.path.basename(ENGINE_PATH)}")
        for entry in self.inputs:
            print(f"  intrare {entry['name']}: {entry['shape']} {np.dtype(entry['dtype']).name}")
        for entry in self.outputs:
            print(f"  iesire  {entry['name']}: {entry['shape']} {np.dtype(entry['dtype']).name}")

    def infer(self, array):
        """Ruleaza engine-ul pe un array si intoarce lista de iesiri, ca numpy."""
        source = self.inputs[0]
        data = np.ascontiguousarray(array, dtype=source["dtype"]).ravel()

        if data.size != source["host"].size:
            raise ValueError(
                f"Intrare de {data.size} valori, engine-ul asteapta "
                f"{source['host'].size} (forma {source['shape']})."
            )

        source["host"][:] = data
        cuda.memcpy_htod_async(source["device"], source["host"], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for entry in self.outputs:
            cuda.memcpy_dtoh_async(entry["host"], entry["device"], self.stream)
        self.stream.synchronize()

        return [entry["host"].reshape(entry["shape"]).copy() for entry in self.outputs]


model = TrtModel(ENGINE_PATH)
model.describe()

# ============================================================
# PROCESARE  (identica cu v3)
# ============================================================


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
    emb = model.infer(preprocess(img_bgr))[0][0]
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
