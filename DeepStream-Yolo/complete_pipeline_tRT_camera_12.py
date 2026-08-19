"""Varianta LIVE, pe camera, a pipeline-ului de recunoastere faciala.

Porneste de la complete_pipeline_tRT_media_11.py si schimba doua lucruri:

  1. sursa e o camera, nu un fisier: v4l2 (webcam USB) sau CSI (nvarguscamerasrc),
     iar iesirea implicita e o fereastra pe ecran. Se poate si inregistra, in
     paralel cu afisarea (--record), sau rula complet fara imagine (--no-display).
  2. se urmareste PREZENTA fiecarei identitati in cadru, si cand cineva iese si
     se intoarce in cel mult RETURN_WINDOW_S secunde se scrie in consola:

         [REVENIT] persoana_5 s-a intors dupa 6.3 s

Restul -- detectie, landmark-uri, aliniere, ArcFace, inrolare automata, baza de
date, logul pe cadru, summary-ul -- ramane exact ce era, inclusiv toate pragurile.

DE CE ORDINEA ASTA CONTEAZA. Detectia de reintoarcere se sprijina pe reparatia de
continuitate din _11 si NU ar fi fost corecta fara ea. In _10, un om care intorcea
capul isi pierdea eticheta pe ecran si redevenea o caseta anonima ("face 11"),
pentru ca o singura fereastra proasta rasturna identitatea si pentru ca o caseta
care nu trecea de porti nu mai era etichetata deloc. Pe o astfel de baza,
"prezenta" ar fi palpait de cateva ori pe secunda si consola s-ar fi umplut cu
"s-a intors" pentru un om care nu plecase niciodata.

Doua garantii, ambele din _11, tin acum evidenta curata:
  - cat timp track-ul traieste, identitatea lui se pastreaza si peste cadrele fara
    landmark-uri, deci un cap intors NU inseamna o plecare;
  - se numara ca prezent doar cine are identitate CONFIRMATA de o trecere prin
    modelul de recunoastere (checks_confirmed >= 1). Un id de tracker, o caseta
    "verificare..." sau o identitate doar preluata geometric de la un track rupt
    nu declanseaza nimic -- exact ca sa nu anuntam intoarcerea unui placeholder.

Peste ele, doua praguri proprii:
  - ABSENCE_GRACE_S: cineva e declarat plecat abia dupa atatea secunde in care
    nicio caseta nu-i mai poarta numele. Sub prag, e o clipire, nu o plecare;
  - RETURN_WINDOW_S: fereastra ceruta, 15 s. Sub ea se scrie "s-a intors"; peste
    ea, reaparitia se scrie ca o aparitie obisnuita ("[REAPARUT]"), ca sa se vada
    diferenta dintre "a iesit un minut din cadru" si "a venit dupa o ora".

Utilizare:
    python3 complete_pipeline_tRT_camera_12.py                       # /dev/video0
    python3 complete_pipeline_tRT_camera_12.py --device /dev/video1
    python3 complete_pipeline_tRT_camera_12.py --camera csi --sensor-id 0
    python3 complete_pipeline_tRT_camera_12.py --record             # si scrie mp4
    python3 complete_pipeline_tRT_camera_12.py --no-display         # doar consola
    python3 complete_pipeline_tRT_camera_12.py --return-window 30

Folderul de iesire (implicit camera_<device>) e REFOLOSIT: baza de date din el se
incarca la pornire, deci oamenii inrolati ieri sunt recunoscuti azi. Baza se
salveaza si periodic in timpul rularii (--save-every), nu doar la iesire: o
sesiune live poate tine ore, iar un Ctrl+C brutal sau o cadere de curent nu
trebuie sa piarda tot ce s-a inrolat.

Despre viteza. Masurat pe videotest3.mp4 (1080x1920, cadru larg): 711 cadre in
71.6 s, adica 9.9 FPS, cu 43.3 ms in probe (fix 24.8 ms/cadru + 4.3 ms/fata, la
4 fete pe cadru) si ~57 ms in restul pipeline-ului. Detectorul scotea 42 de
casete pe cadru, din care 38 sub pragul de marime -- si toate 38 erau urmarite
de tracker si scrise in log. Ce s-a facut, in ordinea castigului:

  1. queue-uri intre elemente. Fara ele tot pipeline-ul rula pe UN SINGUR fir,
     deci cele 43 ms de probe se adunau la cele 57 ms de decodare/detectie/
     encodare in loc sa se suprapuna cu ele.
  2. casetele care nu pot trece de porti se sterg INAINTE de tracker, nu dupa:
     NvDCF filtra 42 de tinte pe cadru ca sa foloseasca 4.
  3. din suprafata se copiaza doar crop-urile fetelor, nu tot cadrul: 4 bucati
     de ~5 KB in loc de 14 MB de copiat si convertit pe fiecare cadru.
  4. punctele se deseneaza vectorizat (o scriere numpy), nu cu 424 de apeluri
     cv2.circle pe cadru.
  5. respinsele se numara, nu se scriu una cate una: 69% din log erau ele.

Primele cinci nu ating deciziile de recunoastere: aceleasi cadre, aceleasi probe,
aceleasi etichete. A sasea, in schimb, e o alegere, nu o optimizare curata:

  6. RACIREA. Dupa ce un track are identitate confirmata, nu i se mai cer
     landmark-uri pe fiecare cadru, ci doar in fereastra dinaintea urmatoarei
     verificari, iar verificarile lui se rarese de la 10 la
     IDENTIFIED_INTERVAL_FRAMES cadre. Landmark-urile erau 63% din timpul de
     probe, si pe un clip cu oameni deja in baza aproape toate se calculau pentru
     oameni pe care ii stiam deja. Simulat pe videotest3.mp4, la o a doua rulare
     (baza plina): 4.0 -> 0.5 fete cu landmark pe cadru, 86% economisit.

     Ce se plateste: daca tracker-ul schimba persoana sub acelasi id in timpul
     racirii, eticheta gresita ramane pana la urmatoarea verificare. Si nu se mai
     deseneaza punctele pe oamenii recunoscuti -- caseta si eticheta raman, ele
     merg prin metadate. Inrolarea NU e atinsa: un track fara identitate primeste
     landmark-uri pe fiecare cadru, ca inainte, pentru ca din ele se alege
     prototipul. Se dezactiveaza cu --landmark-all.

Ce se masoara efectiv se vede in summary.json, la "viteza", "timp_etape_ms" si
"apeluri_economisite".

URMARIREA. Masurat pe videotest3.mp4: zero schimbari de id, dar 120 de goluri in
13 track-uri, unul de 233 de cadre -- oamenii isi pierdeau caseta cand intorceau
capul si o recapatau dupa. Cauzele si ce s-a facut:

  - filtrul dinaintea tracker-ului taia si pe incredere, nu doar pe marime. Cand
    cineva intoarce capul, detectorul nu rateaza fata, ii scade increderea; caseta
    aia slaba e tot ce are NvDCF ca sa lege track-ul. Acum se filtreaza DOAR pe
    marime, si dupa latura mare, nu dupa cea mica (o fata din profil se ingusteaza
    fara sa piarda din inaltime).
  - minDetectorConfidence si maxShadowTrackAge din configul nvtracker se ajusteaza
    la pornire, intr-o copie scrisa in folderul rularii (vezi patch_tracker_config).
  - implicit se cere NvDCF_accuracy, nu NvDCF_perf: costa mai mult per tinta, dar
    filtrul de mai sus a taiat tintele de la 42.8 la ~13 pe cadru.
  - starea unui track nu se mai sterge cat timp tracker-ul inca il vede, chiar daca
    fata nu trece de porti: altfel un om intors 300 de cadre revenea ca necunoscut
    si putea fi inrolat a doua oara, ca identitate dublata.

ReID la nivel de tracker NU se foloseste, desi NvDCF il are: modelele livrate sunt
pentru corpuri (Market-1501, crop-uri de ~128x256), iar aici obiectele sunt fete de
40-66 px. Re-identificarea adevarata o face oricum baza de date, prin embedding-ul
ArcFace: un track rupt isi recapata numele la prima verificare.

CONTINUITATEA ETICHETEI (nou fata de _10). Simptomul raportat: o caseta ajunge
"persoana_13", si cateva cadre mai tarziu redevine o caseta rosie cu "face 1",
desi e acelasi om si acelasi track. Nu era o problema de recunoastere, ci doua
scurgeri prin care eticheta stiuta disparea de pe ecran:

  A. CASETELE CARE NU TREC DE PORTI NU ERAU ETICHETATE DELOC. Cand cineva
     intoarce capul, detectorul nu rateaza fata: ii scade increderea sub
     MIN_CONFIDENCE (sau caseta scade sub MIN_FACE_SIZE). In _10, obiectul era
     sarit cu un "continue", deci label_object nu se mai apela pe el -- iar
     nvdsosd deseneaza atunci ce a pus nvinfer: numele clasei plus id-ul de
     track ("face 1"), cu culoarea implicita din config. Nimic nu "resetase"
     identitatea; doar nu mai scria nimeni peste desenul implicit.
     Acum: un obiect al carui track are deja stare primeste eticheta lui
     (vezi held_objects din process_frame), iar detectiile fara nicio stare se
     deseneaza ca o caseta subtire gri, fara text -- "fata detectata, inca
     neprocesata", nu o identitate gresita.

  B. O SINGURA VERIFICARE PROASTA RASTURNA IDENTITATEA. Dupa o inrolare,
     history = [nume]. Prima verificare care iesea "necunoscut" facea
     history = [nume, necunoscut], adica egalitate -- iar majority_label rupe
     egalitatile in favoarea celei mai RECENTE, deci eticheta afisata devenea
     rosu "necunoscut" dintr-un singur cadru prost (exact cadrele in care se
     pierd si landmark-urile). Acum identitatea confirmata se tine separat de
     votul ferestrei curente (TrackState.identity) si e nevoie de
     IDENTITY_FLIP_CHECKS verificari la rand care sa o contrazica pentru ca ea
     sa cada. Intre timp se afiseaza mai departe, cu marcajul "~" si un verde
     mai stins, ca sa se vada ca vine din memorie, nu dintr-o masuratoare.

  C. Ca efect secundar al lui B, un track cu identitate tinuta NU mai poate fi
     inrolat a doua oara (ready_to_enroll cere identity is None), deci nu se mai
     dubleaza identitati din cauza unui cadru prost.

  D. Cand tracker-ul rupe totusi track-ul si da alt id aceleiasi persoane, noul
     track porneste PROVIZORIU de la identitatea celui vechi, daca acela avea
     una, a disparut de curand (REID_CARRY_GAP_FRAMES) si caseta noua se
     suprapune cu ultima lui caseta (REID_CARRY_MIN_IOU). Provizoriu = prima
     verificare o confirma sau o rastoarna imediat. Se opreste cu
     --no-carry-identity.

Cat timp identitatea e pusa la indoiala (contra_checks > 0) se renunta si la
racire: track-ul primeste iar landmark-uri pe fiecare cadru si se verifica la
VERIFY_INTERVAL_FRAMES, nu la IDENTIFIED_INTERVAL_FRAMES. Indoiala se plateste in
calcul, nu in etichete gresite.

Ce se vede in summary.json, la "continuitate": de cate ori o verificare a
confirmat / a fost tinuta / a schimbat / a pierdut o identitate, cate casete au
fost desenate din memorie si ce identitati au fost preluate intre track-uri.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import warnings
from collections import deque, Counter

import numpy as np
import cv2
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds
import tensorrt as trt
import pycuda.driver as cuda

# ============================================================
# CONFIGURARE
# ============================================================

# numpy a scos np.bool, iar unele bindings vechi il cer inca. Pe numpy nou, chiar
# si verificarea emite FutureWarning, de aceea o facem tacut: la pornire ies deja
# zeci de linii de la GStreamer si TensorRT, iar peste ele o eroare adevarata
# trece neobservata.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    if not hasattr(np, "bool"):
        np.bool = bool

# API-ul cu bindings indexate (binding_is_input, get_binding_shape, ...) e cel
# din TensorRT 8.5, adica ce vine cu JetPack 5.x, si e deprecat de la 8.5 incolo.
# Merge, dar scoate zece randuri de avertismente la fiecare pornire. Le taiem:
# in TensorRT 10 API-ul chiar dispare, si atunci vom primi AttributeError, nu un
# avertisment ascuns (vezi si comentariul din TrtModel).
warnings.filterwarnings("ignore", category=DeprecationWarning, module="__main__")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_CONFIG_PATH = "config_infer_best_v2.txt"

# NvDCF_accuracy inaintea lui NvDCF_perf: are estimare de stare mai buna si tine
# track-urile in umbra mai mult, adica exact ce lipseste cand cineva intoarce
# capul. Costa mai mult per tinta, dar filtrul dinaintea tracker-ului a taiat
# tintele de la 42.8 la ~13 pe cadru, deci acum se poate plati. Daca fisierul nu
# exista pe versiunea instalata, se cade pe perf.
_DS_CONFIGS = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app"
TRACKER_CONFIG_PATH = [
   # f"{_DS_CONFIGS}/config_tracker_NvDCF_accuracy.yml",
    f"{_DS_CONFIGS}/config_tracker_NvDCF_perf.yml"
]

# Ce se schimba in configul low-level al tracker-ului inainte de rulare. Nu se
# editeaza fisierul din DeepStream: se scrie o copie in folderul rularii si
# nvtracker primeste copia, deci rularea ramane reproductibila si originalul
# neatins. None = se lasa cum e in fisier.
#
# minDetectorConfidence: 0.0, adica tracker-ul primeste TOATE casetele. Fetele
#   slabe (cap intors, ocluzie partiala) sunt tocmai cele care tin track-ul legat;
#   filtrarea lor pe calitate se face mai tarziu, in probe, unde nu strica decat
#   fetei respective. Vezi si detection_filter_probe.
# maxShadowTrackAge: cate cadre supravietuieste un track fara nicio detectie.
#   Masurat pe videotest3.mp4: 120 de goluri in 13 track-uri, cel mai lung de 233
#   de cadre. Implicitul NvDCF (~30) le rupe pe aproape toate.
TRACKER_OVERRIDES = {
    "minDetectorConfidence": 0.0,
    "maxShadowTrackAge": 90,
}

PFLD_MODEL_PATH = "/workspace/DeepStream-Yolo/pfld.engine"
RECOGNITION_MODEL_PATH = "/workspace/DeepStream-Yolo/w600k_mbf.engine"
FACE_DATABASE_PATH = "/workspace/DeepStream-Yolo/face_database.json"

MIN_CONFIDENCE = 0.5          # sub asta, nici nu incercam sa procesam fata
MIN_FACE_SIZE = 40            # px, latura minima a bbox-ului
MIN_BLUR = 80.0              # varianta Laplacianului la care consideram poza clara

# Casetele prea mici ca sa fie vreodata fete utilizabile se sterg din metadate
# imediat dupa detector, deci tracker-ul nici nu le vede. Masurat pe
# videotest3.mp4: 42 de casete pe cadru, din care 4 treceau -- NvDCF facea
# urmarire prin corelatie pentru 38 de casete degeaba, pe fiecare cadru.
#
# Pragul de aici e mai jos decat MIN_FACE_SIZE, cu factorul de mai jos, si asta
# e intentionat: o fata care oscileaza in jurul pragului trebuie sa-si pastreze
# track-ul si in cadrele in care e cu un pixel prea mica, altfel primeste alt id
# la fiecare oscilatie si nu aduna niciodata ENROLL_MIN_CHECKS verificari.
#
# Filtrul NU se uita la incredere: vezi detection_filter_probe pentru de ce.
PRETRACK_FILTER = True
PRETRACK_SIZE_FACTOR = 0.8
PRETRACK_MIN_SIZE = MIN_FACE_SIZE * PRETRACK_SIZE_FACTOR   # recalculat in apply_thresholds

# Detaliul fiecarei casete respinse in log. Implicit doar numarate pe poarta:
# vezi comentariul din process_frame.
LOG_REJECTED = False

# Varianta live arunca proba daca blur < MIN_BLUR. Aici nu: pe fisier, varianta
# Laplacianului depinde de codec si de scalarea la rezolutia de lucru (un clip
# 720p urcat la 1080p are varianta de ~2 ori mai mica pe aceeasi fata), asa ca un
# prag absolut poate bloca un track la nesfarsit -- fara proba nu se face nicio
# verificare, deci nici streak-ul de inrolare nu creste si rularea se termina cu
# zero identitati si zero explicatii. Claritatea intra oricum in scorul de
# calitate (termenul "sharp"), iar QUALITY_MIN filtreaza; aici pastram doar un
# prag de siguranta sub care poza chiar nu are informatie.
BLUR_REJECT_FACTOR = 0.25     # probe cu blur < MIN_BLUR * factor se arunca

# Masurat pe sample_vid.mp4: track-urile care trec de porti traiesc intre 2 si 25
# de cadre. La o verificare pe 15 cadre, majoritatea apuca una singura, deci nu
# ajung niciodata la ENROLL_MIN_CHECKS. Costul e mic (recunoasterea a insemnat 15
# apeluri in tot clipul), asa ca verificam mai des.
VERIFY_INTERVAL_FRAMES = 15   # la cat timp cel mult re-verificam un track activ
RETRY_INTERVAL_ON_FAIL = 7    # daca poarta de calitate a picat, reincercam mai repede

# Cadenta pentru track-urile care AU deja o identitate confirmata. Odata ce stim
# cine e cineva, tot ce mai facem cu el e sa ne asiguram ca tracker-ul nu ne-a
# schimbat omul sub acelasi id -- si asta nu cere o verificare la 10 cadre.
# Masurat pe videotest3.mp4: 4 fete pe cadru, landmark-urile 18.1 ms din 28.9 ms
# de probe, adica 63% din tot ce se calculeaza pe cadru mergea pe oameni deja
# recunoscuti. La 30 de cadre (~1.2 s la 25 FPS) ramane destul de des cat sa
# prindem un id schimbat, dar de trei ori mai rar.
#
# Riscul, explicit: daca tracker-ul schimba persoana sub acelasi id in timpul
# racirii, eticheta gresita ramane afisata pana la urmatoarea verificare.
IDENTIFIED_INTERVAL_FRAMES = 30
LABEL_HISTORY_SIZE = 5        # cate decizii recente pastram pentru vot majoritar
TRACK_TIMEOUT_FRAMES = 300    # dupa cate cadre de absenta stergem un track din memorie
PRUNE_CHECK_INTERVAL = 90     # la cate cadre verificam track-uri "moarte"

# ============================================================
# CONTINUITATEA IDENTITATII
# ============================================================
# Aici sta reparatia descrisa in antetul fisierului. Ideea in doua randuri: ce
# stim despre un om (TrackState.identity) nu mai e acelasi lucru cu ce a iesit la
# ultima verificare (votul din history). Prima e o proprietate a persoanei si se
# schimba greu; a doua e o masuratoare, si masuratorile au si cadre proaste.

# Cate verificari LA RAND trebuie sa contrazica identitatea afisata ca ea sa
# cada. 1 = comportamentul din _10 (prima verificare proasta rastoarna eticheta).
# 2 e minimul care rezolva simptomul raportat: o singura fereastra cu fata
# intoarsa, in care landmark-urile ies prost si embedding-ul iese "necunoscut",
# nu mai transforma pe persoana_5 inapoi intr-o caseta anonima.
IDENTITY_FLIP_CHECKS = 2

# Dupa cate cadre fara nicio confirmare identitatea afisata devine "veche". NU se
# sterge -- ar insemna sa ne intoarcem exact la problema pe care o reparam --
# dar se deseneaza galben si cu "?", ca sa nu para masurata acum. 150 de cadre
# sunt ~6 s la 25 FPS, adica de cinci ori intervalul de verificare al unui track
# recunoscut: daca in tot timpul asta n-a iesit nicio proba utilizabila, omul e
# de mult intors sau prea departe.
IDENTITY_STALE_FRAMES = 150

# Increderea minima pentru o caseta a unui track pe care il stim deja. E mai jos
# decat MIN_CONFIDENCE dinadins: tracker-ul a legat deja caseta de un track cu
# istoric, deci nu mai avem nevoie de detector ca sa ne convinga ca acolo e o
# fata. Fara asta, un cap intors (incredere 0.3-0.4) iesea complet din procesare
# si ramanea si fara eticheta, si fara sansa de a fi re-verificat.
TRACKED_MIN_CONFIDENCE = 0.30

# Preluarea identitatii cand tracker-ul da alt id aceleiasi persoane. Se cere si
# apropiere in timp, si suprapunere in spatiu -- altfel doi oameni care se
# incruciseaza si-ar putea schimba numele intre ei.
CARRY_IDENTITY = True
REID_CARRY_GAP_FRAMES = 45    # cat de recent trebuie sa fi disparut track-ul vechi
REID_CARRY_MIN_IOU = 0.30     # cat trebuie sa se suprapuna casetele

# Ce se face cu detectiile care n-au trecut niciodata de porti si deci n-au stare.
# In _10 ajungeau la nvdsosd neatinse, adica desenate cu textul implicit pus de
# nvinfer ("face 11") -- chiar sursa confuziei din raport. Implicit acum se
# deseneaza o caseta subtire gri, fara text: se vede ca acolo e o detectie, dar
# nu se pretinde ca ar fi cineva anume.
DRAW_REJECTED_BOXES = True
HIDE_REJECTED_BOXES = False   # --hide-respinse: le scoate de tot din metadate

# ============================================================
# PREZENTA SI REINTOARCEREA
# ============================================================
# Fereastra ceruta: cine iese din cadru si revine in cel mult atatea secunde e
# anuntat cu "s-a intors". Peste ea, reaparitia e doar o aparitie.
RETURN_WINDOW_S = 15.0

# Cat trebuie sa lipseasca cineva ca sa fie declarat plecat. E singurul filtru de
# palpaire de care mai e nevoie, tocmai pentru ca _11 tine identitatea peste
# cadrele fara masuratoare: aici nu mai acoperim capete intoarse (alea nici nu
# mai apar ca absente), ci doar cazul in care tracker-ul pierde track-ul pentru
# cateva cadre si il reface imediat. Sub prag, nu s-a intamplat nimic.
ABSENCE_GRACE_S = 1.0

# Se anunta si iesirea din cadru, nu doar intoarcerea. Pe o rulare live, fara
# linia de plecare nu se poate citi cea de intoarcere: nu se stie de cand se
# masoara cele 6.3 secunde.
ANNOUNCE_DEPARTURE = True

# Cat de des se scrie baza de date pe disc in timpul rularii, in secunde.
# 0 = doar la iesire, ca la varianta pe fisier.
DATABASE_SAVE_INTERVAL_S = 60.0

QUALITY_WINDOW_FRAMES = 6     # cu cate cadre inainte de termen strangem probe
CANDIDATE_INTERVAL_FRAMES = 2 # la cate cadre luam o proba in fereastra
QUALITY_GOOD_ENOUGH = 0.65    # poza asa buna incat nu mai are rost sa asteptam
QUALITY_MIN = 0.30            # sub asta nu merita consumat modelul de recunoastere

QUALITY_WEIGHTS = {"yaw": 0.30, "pitch": 0.20, "roll": 0.10, "sharp": 0.20, "size": 0.20}

VERIFY_THRESHOLD = 0.42       # prag empiric, ajustat pe baza testelor offline

AUTO_ENROLL = True
ENROLL_MARGIN = 0.10          # banda de incertitudine sub pragul de recunoastere
ENROLL_MAX_SCORE = VERIFY_THRESHOLD - ENROLL_MARGIN

# Pragurile de inrolare sunt mai relaxate decat in varianta live, din doua motive.
# Intai, aici inrolarea se face din cea mai buna proba a intregului track (vezi
# TrackState.best_unknown), nu din proba ferestrei curente: prototipul e ales, nu
# nimerit, deci nu mai e nevoie ca fiecare fereastra sa fie perfecta. Apoi,
# marimea si claritatea depind de rezolutia sursei si de scalare, iar pragurile
# calibrate pe camera 1080p taiau tot pe fisierele de test. Toate se pot schimba
# din linia de comanda (--enroll-min-face, --enroll-min-blur, ...).
ENROLL_MIN_CHECKS = 2         # 3 cerea ~45 de cadre de track neintrerupt; nu exista
ENROLL_MIN_FACE = 40

# Cea mai importanta poarta pentru ce ajunge in baza, si singura care nu se poate
# compensa. Masurat pe sample_vid.mp4: prototipurile cu frontalitate 0 au dat
# autosimilaritate 0.089 fata de alte poze ale aceleiasi persoane -- adica baza
# se umple cu intrari care nu vor recunoaste pe nimeni, niciodata. Cele cu 0.29
# si 0.52 au dat 0.80 si 0.85. Pragul taie exact profilurile, fara sa ceara poze
# de buletin: 40% din fetele clipului au frontalitate 0.
ENROLL_MIN_FRONTALITY = 0.20
ENROLL_MIN_BLUR = 80.0
ENROLL_MIN_QUALITY = 0.35

LABEL_UNKNOWN = "necunoscut"
LABEL_UNCERTAIN = "incert"

ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)
ALIGN_SIZE = 112

# --- desen ---
# Toate punctele se deseneaza cu OpenCV, direct pe suprafata mapata. Pe Jetson
# memoria e unificata, deci suprafata pe care o primim de la pyds e chiar cea pe
# care o citesc nvdsosd si encoder-ul: ce scriem aici ajunge in fisier. Cele 5
# puncte treceau inainte prin display meta (nvdsosd), adica doua cai de desen
# pentru acelasi lucru, cu limita de MAX_ELEMENTS_IN_DISPLAY_META si cu un simbol
# pyds care nu exista in toate versiunile. Daca suprafata nu e scriptibila
# (dGPU), se pierde tot desenul de puncte -- casetele si etichetele raman, ele
# merg prin metadate, nu prin pixeli.
# Landmark-urile se cer doar pentru fetele care chiar au nevoie de ele: cele care
# dau o proba in cadrul asta, si cele ale caror track-uri inca n-au identitate
# (acolo landmark-urile aleg prototipul, deci se cer pe fiecare cadru, ca pana
# acum). Restul -- oameni deja recunoscuti, intre doua verificari -- raman doar
# urmariti de tracker. Se pierde desenul punctelor pe ei; caseta si eticheta nu,
# ele merg prin metadate.
LANDMARK_ONLY_WHEN_NEEDED = True

DRAW_ALL_LANDMARKS = True
DRAW_OVERLAY = True        # se stinge cu --no-video: nu mai are cine sa vada desenul
LANDMARK_RADIUS = 2

# Culori in ordinea canalelor suprafetei (RGBA), nu normalizate: aici scriem
# pixeli, nu completam structuri nvdsosd.
LANDMARK_COLOR = (255, 255, 0, 255)     # setul complet, galben
POINT_COLORS = np.array([  # ochi stang, ochi drept, nas, gura stanga, gura dreapta
    (0, 204, 255, 255),
    (0, 204, 255, 255),
    (51, 255, 51, 255),
    (255, 102, 102, 255),
    (255, 102, 102, 255),
], dtype=np.uint8)


def disc_offsets(radius):
    """Deplasarile (dy, dx) ale pixelilor dintr-un disc de raza data.

    Se calculeaza o data, la import: un punct desenat inseamna apoi o singura
    scriere numpy indexata, nu un apel cv2.circle. La 106 puncte pe fata si 4
    fete pe cadru, diferenta e intre o operatie si 424.
    """
    span = np.arange(-radius, radius + 1)
    dy, dx = np.meshgrid(span, span, indexing="ij")
    inside = dy * dy + dx * dx <= radius * radius
    return dy[inside].ravel(), dx[inside].ravel()


LANDMARK_DISC = disc_offsets(LANDMARK_RADIUS)
FIVE_POINT_DISC = disc_offsets(LANDMARK_RADIUS + 1)
BOX_COLOR_KNOWN = (0.0, 1.0, 0.0, 1.0)
BOX_COLOR_UNCERTAIN = (1.0, 0.8, 0.0, 1.0)
BOX_COLOR_UNKNOWN = (1.0, 0.3, 0.3, 1.0)
BOX_COLOR_PENDING = (0.6, 0.6, 0.6, 1.0)
# Identitate stiuta, dar neconfirmata in cadrul asta: fie fata n-a trecut de
# porti (cap intors, caseta mica), fie ultima verificare a contrazis-o si inca nu
# s-a strans IDENTITY_FLIP_CHECKS. Verde mai stins, ca sa se citeasca dintr-o
# privire diferenta dintre "il vad si e el" si "stiu ca e el".
BOX_COLOR_HELD = (0.3, 0.85, 0.5, 1.0)
# Identitate mai veche de IDENTITY_STALE_FRAMES fara nicio confirmare.
BOX_COLOR_STALE = (0.9, 0.75, 0.2, 1.0)
BOX_WIDTH_CONFIRMED = 2
BOX_WIDTH_HELD = 1            # linie subtire = eticheta vine din memorie
BOX_WIDTH_REJECTED = 1

PROGRESS_EVERY_FRAMES = 100

# ============================================================
# COMPATIBILITATE pyds
# ============================================================
# Bindings-urile Python nu expun aceleasi simboluri in toate versiunile de
# DeepStream. Ce e optional il rezolvam o singura data, aici, si nu in probe --
# altfel un simbol lipsa arunca acelasi AttributeError pe fiecare cadru, iar
# pipeline-ul merge mai departe si umple consola cu acelasi traceback.

_unmap_surface = getattr(pyds, "unmap_nvds_buf_surface", None)
_remove_obj_meta = getattr(pyds, "nvds_remove_obj_meta_from_frame", None)

_warned = set()


def warn_once(key, message):
    """Un avertisment o singura data, nu pe fiecare cadru."""
    if key not in _warned:
        _warned.add(key)
        print(f"[AVERTISMENT] {message}")


def check_pyds_api():
    """Verifica la pornire ce ofera pyds-ul instalat.

    Ce lipseste si e obligatoriu opreste rularea acum, cu un mesaj clar, in loc
    sa strice fiecare cadru. Ce lipseste si e optional doar dezactiveaza o
    functie (desen, eliberarea suprafetei), cu avertisment.
    """
    required = ["gst_buffer_get_nvds_batch_meta", "get_nvds_buf_surface",
                "NvDsFrameMeta", "NvDsObjectMeta"]
    missing = [name for name in required if not hasattr(pyds, name)]
    if missing:
        raise RuntimeError(
            "Bindings-urile pyds nu au ce trebuie: " + ", ".join(missing) +
            ".\nVerifica versiunea de deepstream_python_apps fata de DeepStream."
        )

    if _unmap_surface is None:
        warn_once("unmap", "pyds nu are unmap_nvds_buf_surface; suprafetele nu se "
                           "elibereaza explicit (verifica memoria la rulari lungi).")

    global PRETRACK_FILTER
    if PRETRACK_FILTER and _remove_obj_meta is None:
        PRETRACK_FILTER = False
        warn_once("remove_obj", "pyds nu are nvds_remove_obj_meta_from_frame; "
                                "tracker-ul va primi si casetele prea mici, deci "
                                "rularea va fi mai lenta (rezultatele, aceleasi).")


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

cuda.init()
CUDA_CONTEXT = cuda.Device(0).retain_primary_context()

# Se completeaza in load_models(), nu la import: caile depind de argumente.
landmark_model = None
recognition_model = None
face_database = None
LANDMARK_POINTS = 0

# ============================================================
# MODELE TENSORRT
# ============================================================

class TrtModel:
    """Un .engine TensorRT, cu bufferele alocate o singura data, la batch maxim.

    Identic cu cel din complete_pipeline_tRT.py: engine-urile au batch 8, deci
    intr-un singur apel intra toate fetele unui cadru.
    """

    def __init__(self, engine_path):
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(
                f"Nu gasesc engine-ul: {engine_path}\n"
                f"Construieste-l din .onnx cu trtexec (vezi instructiunile)."
            )

        CUDA_CONTEXT.push()
        try:
            with open(engine_path, "rb") as f:
                runtime = trt.Runtime(TRT_LOGGER)
                self.engine = runtime.deserialize_cuda_engine(f.read())

            if self.engine is None:
                raise RuntimeError(
                    f"Engine-ul {engine_path} nu a putut fi deserializat: de obicei "
                    f"inseamna alta versiune de TensorRT sau alta placa."
                )

            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()

            self.dynamic = False
            self.max_batch = None
            for index in range(self.engine.num_bindings):
                if not self.engine.binding_is_input(index):
                    continue
                shape = tuple(self.engine.get_binding_shape(index))
                if shape[0] == -1:
                    self.dynamic = True
                    limit = int(self.engine.get_profile_shape(0, index)[2][0])
                else:
                    limit = int(shape[0])
                self.max_batch = limit if self.max_batch is None else min(self.max_batch, limit)
            self.max_batch = max(1, self.max_batch or 1)

            for index in range(self.engine.num_bindings):
                if not self.engine.binding_is_input(index):
                    continue
                shape = tuple(self.engine.get_binding_shape(index))
                if -1 in shape:
                    self.context.set_binding_shape(
                        index,
                        (self.max_batch,) + tuple(1 if d == -1 else d for d in shape[1:]),
                    )

            self.bindings = [0] * self.engine.num_bindings
            self.inputs, self.outputs = [], []

            for index in range(self.engine.num_bindings):
                shape = tuple(self.context.get_binding_shape(index))
                dtype = trt.nptype(self.engine.get_binding_dtype(index))
                host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
                host[:] = 0
                device = cuda.mem_alloc(host.nbytes)

                self.bindings[index] = int(device)
                entry = {
                    "index": index,
                    "name": self.engine.get_binding_name(index),
                    "shape": shape,
                    "sample_shape": shape[1:],
                    "sample_elems": int(np.prod(shape[1:])),
                    "dtype": dtype,
                    "host": host,
                    "device": device,
                }
                if self.engine.binding_is_input(index):
                    self.inputs.append(entry)
                else:
                    self.outputs.append(entry)

            self._current_batch = self.max_batch
        finally:
            CUDA_CONTEXT.pop()

        print(f"  {os.path.basename(engine_path)}: "
              f"intrare {self.inputs[0]['shape']} -> iesire {self.outputs[0]['shape']} "
              f"(batch max {self.max_batch}, {'dinamic' if self.dynamic else 'fix'})")

    def _set_batch(self, batch):
        if not self.dynamic or batch == self._current_batch:
            return
        for entry in self.inputs:
            self.context.set_binding_shape(entry["index"], (batch,) + entry["sample_shape"])
        self._current_batch = batch

    def infer_batch(self, array):
        """Ruleaza engine-ul pe n esantioane deodata (n <= max_batch)."""
        source = self.inputs[0]
        data = np.ascontiguousarray(array, dtype=source["dtype"])

        if data.ndim < 2:
            raise ValueError("infer_batch asteapta un tensor cu dimensiune de batch in fata.")
        count = int(data.shape[0])

        if not 1 <= count <= self.max_batch:
            raise ValueError(
                f"Batch de {count}, engine-ul suporta intre 1 si {self.max_batch}."
            )
        if data.size != count * source["sample_elems"]:
            raise ValueError(
                f"Intrare de {data.size} valori pentru {count} esantioane, "
                f"engine-ul asteapta {count * source['sample_elems']}."
            )

        CUDA_CONTEXT.push()
        try:
            self._set_batch(count)
            flat = data.ravel()
            source["host"][:flat.size] = flat

            # Se copiaza doar esantioanele cerute, nu tot bufferul alocat la batch
            # maxim. Bufferele sunt dimensionate o data, pentru cazul cel mai rau,
            # dar un cadru obisnuit are 4 fete si engine-urile au batch 8: fara
            # feliile de aici se mutau de doua ori mai multi octeti decat trebuia,
            # de doua ori pe apel (dus si intors), pe fiecare cadru.
            cuda.memcpy_htod_async(source["device"], source["host"][:flat.size],
                                   self.stream)
            self.context.execute_async_v2(
                bindings=self.bindings, stream_handle=self.stream.handle
            )
            results = []
            for entry in self.outputs:
                view = entry["host"][: count * entry["sample_elems"]]
                cuda.memcpy_dtoh_async(view, entry["device"], self.stream)
                results.append((entry, view))
            self.stream.synchronize()
        finally:
            CUDA_CONTEXT.pop()

        return [view.reshape((count,) + entry["sample_shape"]).copy()
                for entry, view in results]


# Cate fete au trecut prin fiecare model si in cate apeluri -- pentru raportul final.
GPU_STATS = {"landmark_faces": 0, "landmark_calls": 0,
             "recognition_faces": 0, "recognition_calls": 0}


def run_batched(model, tensors, counter):
    """Ruleaza modelul in transe de cel mult max_batch, pastrand ordinea."""
    results = []
    for start in range(0, len(tensors), model.max_batch):
        chunk = np.stack(tensors[start:start + model.max_batch])
        results.extend(model.infer_batch(chunk)[0])
        GPU_STATS[counter + "_calls"] += 1
    GPU_STATS[counter + "_faces"] += len(tensors)
    return results


def l2(vec):
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-12 else vec

# ============================================================
# BAZA DE DATE
# ============================================================

class FaceDatabase:
    """Prototipurile, tinute ca o matrice (N, 512): cautarea e un singur matmul.

    Citeste si scrie acelasi JSON {nume: [512 float]} ca celelalte scripturi din
    W7, deci baza produsa aici poate fi folosita direct de pipeline-ul live.
    """

    def __init__(self, path, save_path=None):
        self.path = path
        self.save_path = save_path or path
        self.labels = []
        self.matrix = np.zeros((0, 0), dtype=np.float32)
        self.source_count = 0

        if not path or not os.path.isfile(path):
            print(f"Baza de date de pornire nu exista ({path}); incep cu una goala.")
            return

        with open(path, "r") as f:
            raw = json.load(f)
        if raw:
            self.labels = list(raw)
            self.matrix = np.stack([l2(raw[label]) for label in self.labels])
        self.source_count = len(self.labels)

    def __len__(self):
        return len(self.labels)

    def add(self, label, embedding):
        vector = l2(embedding)[np.newaxis, :]
        self.matrix = vector if len(self.labels) == 0 else np.vstack([self.matrix, vector])
        self.labels.append(label)

    def verify(self, embedding, threshold, margin):
        """(eticheta, scor). Intre prag-margine si prag raspunsul e LABEL_UNCERTAIN."""
        if not self.labels:
            return LABEL_UNKNOWN, -1.0

        scores = self.matrix @ np.asarray(embedding, dtype=np.float32)
        best = int(np.argmax(scores))
        best_score = float(scores[best])

        if best_score >= threshold:
            return self.labels[best], best_score
        if best_score >= threshold - margin:
            return LABEL_UNCERTAIN, best_score
        return LABEL_UNKNOWN, best_score

    def save(self):
        """Scrie in save_path (folderul rularii), nu peste baza de pornire.

        Aici, spre deosebire de varianta pe fisier, save() poate fi chemat si in
        timpul rularii, de pe firul buclei GLib (vezi salvarea periodica din
        main), in timp ce add() ruleaza pe firul de streaming al lui GStreamer.
        De aceea se ia intai o fotografie a celor doua campuri si se scrie din
        ea: add() face vstack si abia apoi append, deci intre cele doua momente
        matricea are un rand in plus fata de etichete, iar o scriere care le-ar
        citi pe rand ar putea prinde exact acel moment.

        Scrierea ramane atomica (tmp + os.replace): o oprire in mijlocul ei lasa
        pe disc baza veche, intreaga, nu una injumatatita.
        """
        labels = list(self.labels)
        matrix = self.matrix
        count = min(len(labels), matrix.shape[0])

        tmp = self.save_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({labels[i]: matrix[i].tolist() for i in range(count)}, f)
        os.replace(tmp, self.save_path)


def load_models(landmark_path, recognition_path, database_path, database_out):
    """Incarca engine-urile si baza de date. Chemata din main, dupa argumente."""
    global landmark_model, recognition_model, face_database, LANDMARK_POINTS

    print("Incarc engine-urile TensorRT...")
    landmark_model = TrtModel(landmark_path)
    recognition_model = TrtModel(recognition_path)

    LANDMARK_POINTS = landmark_model.outputs[0]["sample_elems"] // 2
    print(f"Model landmark-uri: {LANDMARK_POINTS} puncte")

    face_database = FaceDatabase(database_path, database_out)
    print(f"Baza de date de pornire: {len(face_database)} identitati "
          f"{face_database.labels if len(face_database) <= 12 else ''}")

# ============================================================
# STARE PER TRACK
# ============================================================

class TrackState:
    """Ce stim despre un track.

    Doua lucruri care in _10 erau unul singur, si de aici venea simptomul din
    antet:

      current_label -- rezultatul ULTIMEI verificari (votul din history). E o
        masuratoare: se schimba la fiecare fereastra si are si cadre proaste.
      identity      -- cine credem ca e omul. E o concluzie: se pune cand o
        verificare da un nume adevarat, si nu cade decat dupa IDENTITY_FLIP_CHECKS
        verificari la rand care o contrazic.

    Pe ecran se afiseaza identity cat timp exista. Asa, un cadru in care fata e
    intoarsa -- landmark-uri proaste, embedding "necunoscut", sau nicio poarta
    trecuta si deci nicio masuratoare -- nu mai sterge numele deja aflat.
    """

    def __init__(self):
        self.last_checked_frame = -999999
        self.last_check_failed = False
        self.last_seen_frame = 0
        self.history = deque(maxlen=LABEL_HISTORY_SIZE)
        self.current_label = None
        self.current_score = 0.0
        self.unknown_streak = 0
        self.enrolled = False

        # --- identitatea, separata de votul ferestrei curente ---
        self.identity = None            # numele in care credem; None = inca nimeni
        self.identity_score = 0.0       # scorul verificarii care l-a confirmat ultima oara
        self.identity_frame = -999999   # cand a fost confirmat ultima oara
        self.contra_checks = 0          # verificari la rand care il contrazic
        self.carried_from = None        # de la ce track a fost preluat (daca a fost)
        self.carried_out_frame = -999999  # cand a dat el identitatea altui track
        self.checks_confirmed = 0       # verificari care au sustinut identitatea

        # Ultima caseta vazuta, pentru preluarea identitatii la un track rupt.
        self.last_box = None
        # Cate cadre a fost desenat din memorie (fara nicio masuratoare noua).
        self.held_frames = 0

        self.last_candidate_frame = -999999
        self.best_quality = -1.0
        self.best_aligned = None
        self.best_blur = 0.0
        self.best_size = 0
        self.best_frontality = 0.0

        # Cea mai buna proba iesita "necunoscut" de cand exista track-ul, cu
        # embedding-ul ei cu tot. Fara asta, inrolarea se judeca pe proba
        # ferestrei curente: un track poate avea 10 verificari, una dintre ele
        # dintr-un cadru foarte bun, si sa nu se inroleze niciodata pentru ca
        # exact la a patra verificare persoana era intoarsa. Prototipul care
        # ajunge in baza e ales, nu nimerit.
        self.best_unknown = None    # dict: aligned, quality, blur, size, frame
        self.last_unknown_score = -1.0

        self.checks = 0             # doar pentru raportul final

    @property
    def resolved(self):
        """Track-ul are deja o identitate, nu doar o banuiala.

        Se citeste acum din identity, nu din current_label. Diferenta conteaza
        exact in cadrele problema: dupa o verificare iesita "necunoscut", in _10
        current_label devenea LABEL_UNKNOWN si track-ul redevenea "nerezolvat",
        deci si eticheta, si culoarea, si cadenta se intorceau la zero. Identity
        supravietuieste unei singure verificari proaste, deci si ele.

        LABEL_UNKNOWN si LABEL_UNCERTAIN nu ajung niciodata in identity: primul
        inca vaneaza un prototip pentru inrolare, al doilea inca poate deveni o
        potrivire adevarata la o proba mai buna.
        """
        return self.identity is not None

    @property
    def doubted(self):
        """Identitatea afisata a fost contrazisa, dar inca nu de destule ori."""
        return self.identity is not None and self.contra_checks > 0

    @property
    def needs_landmarks(self):
        """Merita cerut modelul de landmark-uri pe fata asta, in afara probelor?

        Da pentru track-urile fara identitate (din landmark-urile lor se alege
        prototipul) si pentru cele a caror identitate e pusa la indoiala: acolo
        vrem sa ajungem la urmatoarea verificare cat mai repede si cu cea mai buna
        poza posibila. Racirea ramane in picioare pentru restul -- oamenii
        recunoscuti si neconstestati -- adica pentru majoritatea cadrelor.
        """
        return not self.resolved or self.doubted

    @property
    def deadline_gap(self):
        if self.last_check_failed:
            return RETRY_INTERVAL_ON_FAIL
        # O identitate contrazisa se re-verifica in ritmul normal, nu in cel racit:
        # cat timp nu stim daca tracker-ul ne-a schimbat omul, nu are rost sa
        # asteptam IDENTIFIED_INTERVAL_FRAMES ca sa aflam.
        if self.doubted:
            return VERIFY_INTERVAL_FRAMES
        return IDENTIFIED_INTERVAL_FRAMES if self.resolved else VERIFY_INTERVAL_FRAMES

    def apply_decision(self, voted, score, frame_number):
        """Muta rezultatul unei verificari in ce se afiseaza, cu histerezis.

        Intoarce ce s-a intamplat, pentru log si pentru summary:
          "confirmat"  -- verificarea sustine identitatea afisata (sau o pune, la
                          primul nume adevarat al track-ului);
          "necunoscut" -- track fara identitate, ramane fara;
          "tinut"      -- verificarea contrazice, dar nu de destule ori: se
                          pastreaza ce se stia. ASTA e cazul din raport;
          "schimbat"   -- destule verificari la rand spun alt nume: tracker-ul a
                          schimbat omul sub acelasi id, il credem;
          "pierdut"    -- destule verificari la rand spun "necunoscut": identitatea
                          cade si track-ul poate fi inrolat din nou.
        """
        real = voted not in (LABEL_UNKNOWN, LABEL_UNCERTAIN)

        if self.identity is None:
            self.current_label = voted
            self.current_score = score
            if real:
                self.identity = voted
                self.identity_score = score
                self.identity_frame = frame_number
                self.contra_checks = 0
                self.checks_confirmed = 1
                return "confirmat"
            return "necunoscut"

        if real and voted == self.identity:
            self.identity_score = score
            self.identity_frame = frame_number
            self.contra_checks = 0
            self.checks_confirmed += 1
            self.carried_from = None      # preluarea provizorie s-a dovedit buna
            self.current_label = voted
            self.current_score = score
            return "confirmat"

        self.contra_checks += 1
        if self.contra_checks < IDENTITY_FLIP_CHECKS:
            # Se pastreaza si numele, si scorul care il sustinea: altfel pe ecran
            # ar aparea "persoana_5 (0.11)", adica un nume corect cu un scor care
            # il contrazice, si nu se mai poate citi nimic din el.
            self.current_label = self.identity
            self.current_score = self.identity_score
            return "tinut"

        self.contra_checks = 0
        self.carried_from = None
        self.checks_confirmed = 0
        if real:
            self.identity = voted
            self.identity_score = score
            self.identity_frame = frame_number
            self.checks_confirmed = 1
            self.current_label = voted
            self.current_score = score
            return "schimbat"

        # Identitatea cade. history se goleste ca sa nu tarasca votul vechi peste
        # deciziile de acum, iar unknown_streak porneste de la verificarea asta:
        # de aici incolo track-ul e din nou candidat la inrolare.
        self.identity = None
        self.identity_score = 0.0
        self.identity_frame = -999999
        self.current_label = voted
        self.current_score = score
        self.history.clear()
        self.history.append(voted)
        return "pierdut"

    def confirm_enrolled(self, name, frame_number):
        """Dupa inrolare, identitatea e sigura: nu mai trece prin histerezis."""
        self.enrolled = True
        self.identity = name
        self.identity_score = 1.0
        self.identity_frame = frame_number
        self.contra_checks = 0
        # Inrolarea E o masuratoare: prototipul tocmai a trecut prin modelul de
        # recunoastere. Conteaza ca identitate confirmata, nu ca presupunere.
        self.checks_confirmed = 1
        self.carried_from = None
        self.current_label = name
        self.current_score = 1.0
        self.history.clear()
        self.history.append(name)

    def inherit(self, donor, donor_id, overlap, frame_number):
        """Preia PROVIZORIU identitatea unui track rupt care statea in acelasi loc.

        Se ia doar ce tine de persoana (numele, faptul ca a fost deja inrolata, ca
        sa nu intre a doua oara in baza), nu si ce tine de masuratori: history,
        prototipul si streak-ul raman goale, ele apartin track-ului vechi.

        contra_checks porneste de la 1, nu de la 0, si asta e intentia: preluarea
        e o presupunere geometrica, nu o recunoastere. Cu IDENTITY_FLIP_CHECKS = 2,
        prima verificare care spune altceva o rastoarna imediat, iar prima care o
        sustine o transforma in identitate adevarata. Pana atunci se deseneaza ca
        "tinuta" -- verde stins, cu "~".
        """
        self.identity = donor.identity
        self.identity_score = donor.identity_score
        self.identity_frame = donor.identity_frame
        self.current_label = donor.identity
        self.current_score = donor.identity_score
        self.enrolled = donor.enrolled
        self.contra_checks = 1
        self.carried_from = donor_id
        donor.carried_out_frame = frame_number
        return {"track_nou": None, "de_la": donor_id, "nume": donor.identity,
                "iou": r(overlap), "cadru": frame_number}

    def clear_best(self):
        self.best_quality = -1.0
        self.best_aligned = None
        self.best_blur = 0.0
        self.best_size = 0
        self.best_frontality = 0.0

    def offer(self, aligned, quality, front, blur, size):
        if quality > self.best_quality:
            self.best_quality = quality
            self.best_aligned = aligned
            self.best_blur = blur
            self.best_size = size
            self.best_frontality = front

    def offer_unknown(self, aligned, quality, front, blur, size, frame_number):
        """Cea mai buna poza a track-ului DINTRE CELE care pot fi prototip.

        Filtram intai, alegem dupa. Invers -- cea mai buna dupa calitate, si apoi
        vedem daca trece pragurile -- nu merge: calitatea pune pe marime doar
        0.20, asa ca o fata mica si frontala bate una mare si putin intoarsa.
        Proba castigatoare pica apoi la poarta de marime, iar track-ul ramane
        neinrolat desi avusese o proba perfect buna (masurat: track 50 avea la
        cadrul 197 o proba de 121 px, dar prototipul ramasese unul de 61 px).

        Alegerea se face dupa FRONTALITATE, nu dupa calitate. Calitatea amesteca
        unghiurile cu claritatea si marimea, asa ca o fata din profil, dar mare si
        clara, iese "mai buna" decat una ceva mai mica privita din fata -- si chiar
        asa s-a intamplat: track 23 a intrat in baza cu o poza din profil (yaw 0),
        desi avea la dispozitie una cu yaw 0.26. Aici nu cautam poza cea mai
        frumoasa, ci pe cea din care se poate recunoaste persoana mai tarziu.

        Se pastreaza poza aliniata, nu embedding-ul: asa poate fi oferita ORICE
        fata vazuta, nu doar cea pe care s-a nimerit sa cada o verificare, iar
        modelul de recunoastere se plateste o singura data, la inrolare.
        """
        if sample_blockers(size, blur, quality, front):
            return
        best = self.best_unknown
        if best is None or (front, quality) > (best["frontality"], best["quality"]):
            self.best_unknown = {"aligned": aligned, "quality": quality,
                                 "frontality": front, "blur": blur, "size": size,
                                 "frame": frame_number}


track_states = {}

# Istoricul complet al track-urilor, pentru summary.json: track_states se curata
# periodic, aici nu stergem nimic.
track_reports = {}


class PresenceLog:
    """Cine e in cadru acum, cine a plecat si cine s-a intors.

    Evidenta se tine pe NUME din baza de date, nu pe id-uri de track, si asta e
    toata ideea: un om poate trece prin trei id-uri de tracker intr-un minut
    (ocluzii, iesiri din cadru, reintrari) fara sa plece nicio secunda, iar un id
    de tracker poate ajunge la doi oameni diferiti. Numai numele identifica
    persoana, si numai el are voie sa declanseze un "s-a intors".

    Intrarile se dau din process_frame: multimea numelor CONFIRMATE vizibile in
    cadrul curent. Confirmate inseamna trecute macar o data prin modelul de
    recunoastere (TrackState.checks_confirmed >= 1) -- o identitate doar preluata
    geometric de la un track rupt nu se numara, ca sa nu anuntam intoarcerea unei
    presupuneri.

    Timpul e ceas de perete (time.monotonic), nu numar de cadre: fereastra ceruta
    e in secunde, iar la o rulare live rata de cadre variaza cu cate fete sunt in
    imagine. Numarate in cadre, 15 s ar insemna altceva la fiecare rulare.
    """

    def __init__(self, window_s, grace_s):
        self.window_s = window_s
        self.grace_s = grace_s
        self.present = {}       # nume vizibil -> ultimul moment in care a fost vazut
        self.absent = {}        # nume plecat  -> momentul ultimei vederi
        self.first_seen = {}    # nume -> prima aparitie din rulare
        self.events = []        # istoricul complet, pentru summary
        self.counts = Counter()

    def update(self, names, now):
        """Muta prezenta la momentul 'now'. Intoarce evenimentele de raportat.

        Fiecare eveniment e (fel, nume, secunde), unde 'secunde' e cat a lipsit
        (doar la reveniri). Felurile: "aparut", "plecat", "revenit", "reaparut".
        """
        events = []

        for name in names:
            if name in self.present:
                self.present[name] = now
                continue

            self.present[name] = now
            left_at = self.absent.pop(name, None)
            if left_at is None:
                self.first_seen[name] = now
                events.append(("aparut", name, None))
                continue
            gone = now - left_at
            events.append(("revenit" if gone <= self.window_s else "reaparut",
                           name, gone))

        for name, last_seen in list(self.present.items()):
            if name in names or now - last_seen < self.grace_s:
                continue
            # Momentul plecarii e ULTIMUL in care a fost vazut, nu momentul in
            # care ne-am dat seama: altfel gratia s-ar aduna la timpul de absenta
            # si cineva plecat 14.8 s ar cadea in afara ferestrei de 15 s.
            del self.present[name]
            self.absent[name] = last_seen
            events.append(("plecat", name, None))

        for kind, name, gone in events:
            self.counts[kind] += 1
            self.events.append({"fel": kind, "nume": name, "moment": r(now, 2),
                                "lipsa_s": None if gone is None else r(gone, 2)})
        return events

    def returns(self):
        """Doar revenirile in fereastra, pentru summary."""
        return [event for event in self.events if event["fel"] == "revenit"]


def box_iou(a, b):
    """Suprapunerea a doua casete (x1, y1, x2, y2), in [0, 1]."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = float(iw * ih)
    area_a = float((a[2] - a[0]) * (a[3] - a[1]))
    area_b = float((b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def find_identity_donor(track_id, box, frame_number):
    """Track-ul rupt de la care noul track ar putea prelua identitatea.

    Intoarce (track_id_donator, stare, suprapunere) sau (None, None, 0.0).

    Trei conditii, toate obligatorii:
      - donatorul chiar avea o identitate (altfel n-are ce preda);
      - a fost vazut ultima oara INAINTE de cadrul asta, nu in el. Asta e poarta
        importanta: un track vazut chiar acum e alt om, care se plimba pe langa
        cel nou, iar fara conditia asta doi oameni care se incruciseaza si-ar
        schimba numele intre ei;
      - a disparut de curand si statea in acelasi loc (REID_CARRY_GAP_FRAMES,
        REID_CARRY_MIN_IOU).

    Un donator care tocmai a dat identitatea altui track nu o mai da si la al
    treilea: altfel un track care se rupe in doua ar produce doi "persoana_5".
    """
    best_state, best_id, best_iou = None, None, REID_CARRY_MIN_IOU
    for other_id, other in track_states.items():
        if other_id == track_id or other.identity is None or other.last_box is None:
            continue
        gap = frame_number - other.last_seen_frame
        if gap <= 0 or gap > REID_CARRY_GAP_FRAMES:
            continue
        if frame_number - other.carried_out_frame <= REID_CARRY_GAP_FRAMES:
            continue
        overlap = box_iou(box, other.last_box)
        if overlap > best_iou:
            best_state, best_id, best_iou = other, other_id, overlap
    return best_id, best_state, (best_iou if best_state is not None else 0.0)


class FaceSample:
    """O fata vizibila in cadrul curent, cu tot ce s-a calculat pentru ea."""

    __slots__ = ("track_id", "state", "obj_meta", "crop", "blur", "size",
                 "box", "landmarks", "five_points", "aligned", "quality",
                 "frontality", "wants_sample", "action")

    def __init__(self, track_id, state, obj_meta, crop, blur, size, box):
        self.track_id = track_id
        self.state = state
        self.obj_meta = obj_meta
        self.crop = crop
        self.blur = blur
        self.size = size
        self.box = box              # (x1, y1, x2, y2) in cadru
        self.landmarks = None       # toate punctele, in coordonate de cadru
        self.five_points = None     # cele 5 puncte ArcFace, in coordonate de cadru
        self.aligned = None
        self.quality = 0.0
        self.frontality = 0.0       # min(yaw, pitch): cat de din fata e privita
        self.wants_sample = False   # e in fereastra de probe? (se afla fara GPU)
        self.action = "vazut"       # ce s-a intamplat cu fata, pentru log

# ============================================================
# PROCESARE
# ============================================================

def crop_bgr(surface_rgba, box):
    """Fata decupata din suprafata mapata, convertita in BGR.

    Se converteste DOAR dreptunghiul fetei, nu tot cadrul. Inainte se copia
    suprafata intreaga si se convertea RGBA->BGR pe ea (la 1080x1920: 8 MB de
    copiat plus 14 MB de citit-scris in conversie, pe fiecare cadru), desi din ea
    se foloseau patru bucati de cate ~43x43 px. Asta era grosul celor 24.8 ms
    fixe pe cadru din masuratoare.

    Rezultatul e un tablou nou, deci desenul de la finalul cadrului nu ajunge in
    crop-urile deja luate -- fetele intra curate in modele.
    """
    x1, y1, x2, y2 = box
    return cv2.cvtColor(surface_rgba[y1:y2, x1:x2], cv2.COLOR_RGBA2BGR)


def blur_score(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # CV_32F, nu CV_64F: varianta iese aceeasi pe crop-uri de zeci de pixeli, dar
    # se muta jumatate din octeti.
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def preprocess_landmark(crop_bgr):
    img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (112, 112))
    tensor = img_resized.astype(np.float32) / 255.0
    return np.transpose(tensor, (2, 0, 1))


def postprocess_landmark(raw, crop_shape):
    h, w = crop_shape[:2]
    landmark_pixels = np.asarray(raw, dtype=np.float32).reshape(-1, 2).copy()
    landmark_pixels[:, 0] *= w
    landmark_pixels[:, 1] *= h
    return landmark_pixels


LANDMARK_LAYOUTS = {
    68:  {"left_eye": range(36, 42), "right_eye": range(42, 48), "nose": 30, "mouth": (48, 54)},
    98:  {"left_eye": range(60, 68), "right_eye": range(68, 76), "nose": 54, "mouth": (76, 82)},
    106: {"left_eye": 38, "right_eye": 88, "nose": 86, "mouth": (52, 61)},
}


def get_5_points(landmark):
    """Cele 5 puncte ArcFace, indiferent de markup-ul modelului de landmark-uri."""
    count = landmark.shape[0]
    layout = LANDMARK_LAYOUTS.get(count)
    if layout is None:
        raise ValueError(
            f"Modelul de landmark-uri scoate {count} puncte, iar maparea catre "
            f"cele 5 puncte ArcFace nu e definita. Markup-uri cunoscute: "
            f"{sorted(LANDMARK_LAYOUTS)}."
        )

    def take(index):
        return landmark[index].mean(axis=0) if isinstance(index, range) else landmark[index]

    left_mouth, right_mouth = layout["mouth"]
    return np.array(
        [take(layout["left_eye"]), take(layout["right_eye"]), take(layout["nose"]),
         landmark[left_mouth], landmark[right_mouth]],
        dtype=np.float32,
    )


def umeyama_similarity(src, dst):
    """Transformarea (rotatie + scalare + translatie) care duce src peste dst."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    num, dim = src.shape

    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_demean, dst_demean = src - src_mean, dst - dst_mean

    covariance = dst_demean.T @ src_demean / num
    signs = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(covariance) < 0:
        signs[dim - 1] = -1.0

    u, singular, vt = np.linalg.svd(covariance)
    matrix = np.eye(dim + 1, dtype=np.float64)
    matrix[:dim, :dim] = u @ np.diag(signs) @ vt

    variance = src_demean.var(axis=0).sum()
    if variance < 1e-12:
        return None

    scale = float(singular @ signs) / variance
    matrix[:dim, dim] = dst_mean - scale * (matrix[:dim, :dim] @ src_mean)
    matrix[:dim, :dim] *= scale
    return matrix[:dim, :]


def align_face(crop_bgr, five_points):
    transform_matrix = umeyama_similarity(five_points, ARCFACE_TEMPLATE)
    if transform_matrix is None or not np.all(np.isfinite(transform_matrix)):
        return None
    return cv2.warpAffine(crop_bgr, transform_matrix, (ALIGN_SIZE, ALIGN_SIZE), borderValue=0)


def preprocess_recognition(aligned_bgr):
    img_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    img_norm = (img_rgb.astype(np.float32) - 127.5) / 127.5
    return np.transpose(img_norm, (2, 0, 1))


def _nose_ratio(points):
    left_eye, right_eye, nose, left_mouth, right_mouth = points
    eye_vec = right_eye - left_eye
    interocular = float(np.linalg.norm(eye_vec))
    unit = eye_vec / interocular
    normal = np.array([-unit[1], unit[0]], dtype=np.float64)
    eye_center = (left_eye + right_eye) / 2.0
    mouth_center = (left_mouth + right_mouth) / 2.0
    height = float(np.dot(mouth_center - eye_center, normal))
    return float(np.dot(nose - eye_center, normal)) / height


NOSE_RATIO_FRONTAL = _nose_ratio(ARCFACE_TEMPLATE.astype(np.float64))


def quality_parts(five_points, blur, min_side):
    """Componentele calitatii, fiecare in [0, 1]. None daca punctele sunt degenerate."""
    left_eye, right_eye, nose, left_mouth, right_mouth = np.asarray(
        five_points, dtype=np.float64
    )

    eye_vec = right_eye - left_eye
    interocular = float(np.linalg.norm(eye_vec))
    if interocular < 1e-3:
        return None

    unit = eye_vec / interocular
    normal = np.array([-unit[1], unit[0]], dtype=np.float64)
    eye_center = (left_eye + right_eye) / 2.0
    mouth_center = (left_mouth + right_mouth) / 2.0

    height = float(np.dot(mouth_center - eye_center, normal))
    if height <= 1e-3:
        return None

    def clamp01(value):
        return float(min(1.0, max(0.0, value)))

    lateral = abs(float(np.dot(nose - eye_center, unit))) / interocular
    yaw = clamp01(1.0 - lateral / 0.25)

    ratio = float(np.dot(nose - eye_center, normal)) / height
    pitch = clamp01(1.0 - abs(ratio - NOSE_RATIO_FRONTAL) / 0.25)

    roll_deg = abs(np.degrees(np.arctan2(float(eye_vec[1]), float(eye_vec[0]))))
    roll = clamp01(1.0 - min(roll_deg, 180.0 - roll_deg) / 30.0)

    sharp = clamp01(blur / (2.0 * MIN_BLUR))
    size = clamp01(min_side / float(ENROLL_MIN_FACE))

    return {"yaw": yaw, "pitch": pitch, "roll": roll, "sharp": sharp, "size": size}


def face_quality(parts):
    """Cat de utilizabila e poza, in [0, 1]: frontalitate + claritate + marime."""
    return sum(QUALITY_WEIGHTS[name] * parts[name] for name in QUALITY_WEIGHTS)


def frontality(parts):
    """Cat de din fata e privita persoana, in [0, 1]: min(yaw, pitch).

    Roll-ul nu intra: alinierea Umeyama roteste poza in plan, deci o fata inclinata
    ajunge oricum dreapta in crop-ul de 112x112. Yaw si pitch sunt rotatii in afara
    planului si nu se pot corecta -- o fata din profil, aliniata la un sablon
    frontal, iese intinsa, iar embedding-ul ei nu seamana nici macar cu alte poze
    ale aceleiasi persoane.

    Se ia minimul, nu media: un cap intors la 90 de grade ramane inutilizabil
    oricat de bine ar sta pe verticala.

    Masurat pe sample_vid.mp4: prototipurile cu yaw 0 au dat autosimilaritate 0.089
    (practic zgomot), iar cele cu 0.29 si 0.52 au dat 0.80 si 0.85.
    """
    return min(parts["yaw"], parts["pitch"])


def analyse_faces(faces):
    """Landmark-uri (un singur apel GPU) + aliniere + calitate, pentru tot cadrul.

    Completeaza direct campurile din FaceSample. Se ruleaza pe toate fetele
    vizibile, nu doar pe candidati: landmark-urile se deseneaza, si tot ele decid
    care poza ajunge prototip in baza de date.
    """
    if not faces:
        return

    raw_landmarks = run_batched(
        landmark_model, [preprocess_landmark(f.crop) for f in faces], "landmark"
    )

    for face, raw in zip(faces, raw_landmarks):
        landmark = postprocess_landmark(raw, face.crop.shape)
        five = get_5_points(landmark)

        offset = np.array([face.box[0], face.box[1]], dtype=np.float32)
        face.landmarks = landmark + offset
        face.five_points = five + offset

        face.aligned = align_face(face.crop, five)
        if face.aligned is not None:
            parts = quality_parts(five, face.blur, face.size)
            if parts is not None:
                face.quality = face_quality(parts)
                face.frontality = frontality(parts)


def embed_aligned(aligned_faces):
    if not aligned_faces:
        return []
    raw = run_batched(
        recognition_model, [preprocess_recognition(a) for a in aligned_faces], "recognition"
    )
    return [l2(emb) for emb in raw]


def verify_embedding(embedding):
    return face_database.verify(embedding, VERIFY_THRESHOLD, ENROLL_MARGIN)


def enroll(embedding):
    used = {int(m.group(1)) for m in
            (re.match(r"persoana_(\d+)$", label) for label in face_database.labels) if m}
    name = f"persoana_{max(used, default=0) + 1}"
    face_database.add(name, embedding)
    print(f"[INROLARE] identitate noua: {name} (total {len(face_database)})")
    return name


def majority_label(history):
    """Eticheta cu cele mai multe voturi; la egalitate, cea mai recenta.

    Counter.most_common ar da-o pe prima aparuta, ceea ce dupa o inrolare inseamna
    ca numele proaspat inrolat bate un "necunoscut" venit dupa el, si in video
    ramane afisata o identitate cu scor aproape zero.
    """
    counts = Counter(history)
    top = max(counts.values())
    for label in reversed(history):
        if counts[label] == top:
            return label


def prototype_record(best):
    """Prototipul, pentru log. None cand track-ul inca n-a prins nicio poza buna."""
    if not best:
        return None
    return {"calitate": r(best["quality"]), "frontalitate": r(best["frontality"]),
            "blur": r(best["blur"], 1), "marime": best["size"],
            "cadru": best["frame"]}


def ready_to_enroll(state):
    """Track-ul are si dovada ca nu e in baza, si o poza buna din care sa-l inrolam.

    "identity is None" e conditia noua si e tot din reparatia de continuitate: cat
    timp tinem o identitate (chiar contrazisa o data), track-ul NU se inroleaza.
    Fara ea, o singura fereastra proasta pe cineva deja recunoscut din baza
    (state.enrolled ramane False la recunoastere, se pune doar la inrolare) ar fi
    putut aduna streak si l-ar fi bagat in baza a doua oara, sub alt nume.
    Identitatea cade abia dupa IDENTITY_FLIP_CHECKS verificari la rand, si de-abia
    atunci se redeschide inrolarea.
    """
    return (AUTO_ENROLL and not state.enrolled
            and state.identity is None
            and state.best_unknown is not None
            and state.unknown_streak >= ENROLL_MIN_CHECKS
            and state.last_unknown_score < ENROLL_MAX_SCORE)


def sample_blockers(size, blur, quality, front):
    """Ce opreste o poza sa poata fi prototip. Lista goala = poate fi."""
    blockers = []
    if size < ENROLL_MIN_FACE:
        blockers.append("marime")
    if blur < ENROLL_MIN_BLUR:
        blockers.append("blur")
    if quality < ENROLL_MIN_QUALITY:
        blockers.append("calitate")
    if front < ENROLL_MIN_FRONTALITY:
        blockers.append("profil")
    return blockers


def enroll_blockers(state, score, size, blur, quality, front):
    """Ce anume opreste inrolarea track-ului acum. Lista goala = se inroleaza.

    Fara asta, o rulare care se termina cu zero identitati nu spune nimic: nu se
    stie daca fetele erau prea mici, prea neclare, prea din profil, sau daca pur
    si simplu niciun track n-a trait destul cat sa adune ENROLL_MIN_CHECKS
    verificari. Se numara pe toata rularea si ajunge in summary.json.

    Pragurile de proba (marime, blur, calitate) sunt deja aplicate cand se strange
    prototipul, in offer_unknown; aici se raporteaza doar cand track-ul inca n-are
    niciuna buna, ca sa se vada care poarta o taie.
    """
    blockers = []
    # Track-ul tine inca o identitate contrazisa o data. Nu se inroleaza -- ar
    # insemna sa punem in baza, sub nume nou, un om pe care tocmai il afisam cu
    # numele lui vechi -- si nici nu e o poarta de calitate, deci merita spus
    # separat: altfel in summary ar aparea o incercare fara nicio cauza.
    if state.identity is not None:
        blockers.append("identitate_tinuta")
    if state.unknown_streak < ENROLL_MIN_CHECKS:
        blockers.append("streak")
    if score >= ENROLL_MAX_SCORE:
        blockers.append("scor")
    if state.best_unknown is None:
        # Track-ul n-a prins inca nicio poza buna. Spunem de ce nu e buna cea de
        # acum -- e cel mai apropiat lucru de "ce prag il taie".
        blockers.extend(sample_blockers(size, blur, quality, front)
                        or ["fara_poza_buna"])
    return blockers

# ============================================================
# DESEN
# ============================================================

def stamp_points(surface_rgba, points, disc, color):
    """Deseneaza TOATE punctele deodata, printr-o singura scriere indexata.

    points e (n, 2) in coordonate de cadru, color e ori o culoare RGBA, ori un
    tablou (n, 4) cu cate una de fiecare punct. Discul e precalculat, deci aici
    nu ramane decat o adunare cu broadcast si o masca.

    Pixelii din afara cadrului se ARUNCA, nu se prind de margine: modelul de
    landmark-uri poate scoate puncte in afara crop-ului, iar o taiere cu clip
    le-ar aduna pe toate pe chenarul imaginii, unde n-are ce cauta niciun punct
    facial. Asa se comporta si cv2.circle, deci desenul iese la fel ca inainte.
    """
    if len(points) == 0:
        return
    height, width = surface_rgba.shape[:2]
    dy, dx = disc
    centres = np.rint(points).astype(np.int32)
    ys = centres[:, 1, None] + dy
    xs = centres[:, 0, None] + dx
    inside = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
    if isinstance(color, np.ndarray) and color.ndim == 2:
        # o culoare pe punct, repetata peste tot discul lui
        color = np.broadcast_to(color[:, None, :], ys.shape + (4,))[inside]
    surface_rgba[ys[inside], xs[inside]] = color


def draw_landmarks(surface_rgba, faces):
    """Punctele faciale, direct pe suprafata mapata.

    Cele 5 puncte ArcFace se deseneaza ultimele si cu raza mai mare, ca sa se
    vada peste setul complet. Punctele tuturor fetelor se aduna intr-un singur
    tablou: inainte fiecare punct insemna un apel cv2.circle, adica 106 puncte x
    4 fete = 424 de treceri prin bindings pe fiecare cadru, pentru cateva mii de
    pixeli scrisi in total.

    Desenul nu are voie sa strice procesarea: daca suprafata nu e scriptibila, se
    raporteaza o data si se merge mai departe -- recunoasterea si logul sunt
    oricum treaba importanta, adnotarea e doar ca sa se vada. Casetele si
    etichetele nu se pierd: ele merg prin metadate.
    """
    drawn = [face for face in faces if face.landmarks is not None]
    if not drawn:
        return

    try:
        if DRAW_ALL_LANDMARKS:
            stamp_points(surface_rgba,
                         np.concatenate([face.landmarks for face in drawn]),
                         LANDMARK_DISC, LANDMARK_COLOR)
        stamp_points(surface_rgba,
                     np.concatenate([face.five_points for face in drawn]),
                     FIVE_POINT_DISC, np.tile(POINT_COLORS, (len(drawn), 1)))
    except Exception as error:      # suprafata nescriptibila / alt layout
        warn_once("surface", f"nu pot desena pe suprafata ({error}); "
                             f"raman doar casetele si etichetele.")


def display_for(state, frame_number, measured):
    """Ce se scrie pe caseta unui track: (text, culoare, latime_bordura).

    measured = in cadrul asta chiar s-a masurat ceva pe fata (a trecut de porti
    si i s-au cerut landmark-uri). False inseamna ca tot ce afisam vine din
    memorie: fata n-a trecut de porti, sau e in racire intre doua verificari.

    Regula, si ea e toata reparatia: cat timp track-ul are o identitate, ea se
    afiseaza. Nu exista drum inapoi de la "persoana_5" la desenul implicit al lui
    nvdsosd; exista doar drumul spre un alt nume sau spre "necunoscut", si acela
    trece prin IDENTITY_FLIP_CHECKS verificari (vezi TrackState.apply_decision).
    Ce se schimba cand nu avem masuratoare e doar cat de tare o afirmam: verde
    plin cand tocmai a fost confirmata, verde stins cu "~" cand e din memorie,
    galben cu "?" cand n-a mai fost confirmata de IDENTITY_STALE_FRAMES.
    """
    if state.identity is not None:
        stale = frame_number - state.identity_frame > IDENTITY_STALE_FRAMES
        held = stale or state.doubted or not measured
        text = f"{state.identity} ({state.identity_score:.2f})"
        if stale:
            return text + " ?", BOX_COLOR_STALE, BOX_WIDTH_HELD
        if held:
            return text + " ~", BOX_COLOR_HELD, BOX_WIDTH_HELD
        return text, BOX_COLOR_KNOWN, BOX_WIDTH_CONFIRMED

    if state.current_label is None:
        return "verificare...", BOX_COLOR_PENDING, BOX_WIDTH_CONFIRMED

    text = f"{state.current_label} ({state.current_score:.2f})"
    if state.current_label == LABEL_UNKNOWN:
        return text, BOX_COLOR_UNKNOWN, BOX_WIDTH_CONFIRMED
    if state.current_label == LABEL_UNCERTAIN:
        return text, BOX_COLOR_UNCERTAIN, BOX_WIDTH_CONFIRMED
    # nu se ajunge aici cat timp identity se pune odata cu current_label, dar
    # daca se ajunge, un nume tot e un nume
    return text, BOX_COLOR_KNOWN, BOX_WIDTH_CONFIRMED


def label_object(obj_meta, state, frame_number, measured=True):
    """Scrie eticheta si culoarea casetei peste ce deseneaza nvdsosd implicit.

    Se apeleaza pentru TOATE obiectele cu stare, inclusiv pentru cele care n-au
    trecut de porti in cadrul asta. In _10 se apela doar pentru cele care
    trecusera, iar restul ramaneau cu textul pus de nvinfer -- numele clasei plus
    id-ul de track, adica exact "face 11" din raport. Nu se pierdea nicio
    identitate; doar nu mai scria nimeni peste desenul implicit.
    """
    text, color, width = display_for(state, frame_number, measured)

    obj_meta.text_params.display_text = text
    obj_meta.text_params.set_bg_clr = 1
    obj_meta.text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)
    obj_meta.text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
    obj_meta.text_params.font_params.font_size = 12

    obj_meta.rect_params.border_color.set(*color)
    obj_meta.rect_params.border_width = width


def blank_object(obj_meta):
    """O detectie fara nicio stare: caseta subtire gri, fara text.

    Astea sunt casetele care n-au trecut niciodata de porti si al caror track n-a
    ajuns sa aiba stare -- de obicei fete prea mici din fundal. Ele sunt cele care
    ajungeau la nvdsosd neatinse si se desenau cu textul implicit "face <id>".
    Textul se sterge pentru ca un id de tracker nu e o identitate si nu are ce
    cauta pe ecran langa niste nume adevarate; caseta ramane, subtire, ca sa se
    vada ca acolo chiar e o detectie.
    """
    obj_meta.text_params.display_text = ""
    obj_meta.text_params.set_bg_clr = 0
    obj_meta.rect_params.border_color.set(*BOX_COLOR_PENDING)
    obj_meta.rect_params.border_width = BOX_WIDTH_REJECTED if DRAW_REJECTED_BOXES else 0

# ============================================================
# LOG
# ============================================================

def r(value, digits=3):
    return round(float(value), digits)


class FrameLogger:
    """Un rand JSON pe cadru, in frames_<rulare>.jsonl.

    Se poate opri de tot (--no-frame-log). Pe fisier logul e ieftin -- o rulare
    tine cateva minute -- dar o sesiune live de o ora la 25 FPS inseamna 90 000 de
    randuri si zeci de MB pe care de obicei nu-i citeste nimeni. Cand e oprit,
    scrierea devine un no-op si nu se deschide niciun fisier; summary.json, cu
    tot ce s-a intamplat pe sesiune, se scrie oricum.
    """

    def __init__(self, path, enabled=True):
        self.path = path if enabled else None
        self.handle = open(path, "w", encoding="utf-8") if enabled else None
        self.frames = 0

    def write(self, record):
        if self.handle is None:
            return
        self.handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        self.handle.write("\n")
        self.frames += 1
        if self.frames % 50 == 0:
            self.handle.flush()

    def close(self):
        if self.handle is None:
            return
        self.handle.flush()
        self.handle.close()


class Run:
    """Starea rularii curente: unde scriem, ce s-a intamplat pana acum."""

    def __init__(self, args, output_dir, video_path, paths, run_index,
                 database_in, database_out):
        self.args = args
        self.output_dir = output_dir
        self.video_path = video_path
        self.paths = paths
        self.run_index = run_index
        self.database_in = database_in
        self.database_out = database_out
        self.logger = FrameLogger(paths["frames"], enabled=not args.no_frame_log)
        self.frames = 0
        self.faces_seen = 0
        self.recognitions = 0
        self.enrollments = []
        self.enroll_attempts = 0
        self.enroll_blockers = Counter()
        # De ce nu s-a facut verificarea cand era programata: fara asta, o rulare
        # in care nicio proba nu trece de calitate arata identic cu una in care
        # nu s-a vazut nicio fata (zero incercari de inrolare, zero blocaje).
        self.checks_skipped = Counter()
        self.deadline_qualities = []
        self.probe_ms = []
        # distributiile marimii, claritatii si calitatii fetelor din tot clipul:
        # fara ele, un "marime x35" in blocaje nu spune daca pragul e cu putin
        # prea sus sau cu totul nepotrivit pentru filmarea asta
        self.face_sizes = []
        self.face_blurs = []
        self.face_qualities = []
        self.face_frontalities = []
        self.started = time.time()
        # None = rularea a mers pana la capatul fisierului; altfel, eroarea de pe
        # magistrala GStreamer care a oprit-o.
        self.failure = None

        # Cate casete a scos detectorul si cate au ajuns la tracker. Diferenta e
        # munca economisita de filtrul dinaintea lui.
        self.detections = 0
        self.detections_kept = 0
        # Marimea casetelor taiate de poarta de marime, oriunde ar fi fost taiate
        # (inainte de tracker sau in probe). Fara ele nu se poate raspunde la
        # singura intrebare care conteaza cand baza iese goala: "cat as castiga
        # daca as cobori pragul?" -- detaliul fiecarei casete nu se mai scrie in
        # log, dar distributia costa o adaugare la lista si spune acelasi lucru.
        self.rejected_sizes = []
        # Unde se duc milisecundele din probe, insumate pe toata rularea. Fara
        # ele, "43 ms pe cadru" nu spune ce anume sa optimizezi in continuare;
        # perf_counter costa ~50 ns pe apel, deci masuram mereu, nu sub un flag.
        self.stage_s = Counter()

        # --- continuitatea etichetei ---
        # Ce a facut fiecare verificare cu identitatea afisata: confirmat, tinut,
        # schimbat, pierdut. "tinut" numara exact cazurile din raport, adica
        # cadrele in care _10 ar fi aruncat numele si ar fi desenat "face <id>".
        self.identity_events = Counter()
        # Casete desenate din memorie, pe categorii: fara masuratoare in cadrul
        # asta (fata n-a trecut de porti sau e in racire) si identitati vechi.
        self.held_boxes = 0
        self.stale_boxes = 0
        # Detectii fara nicio stare, desenate anonim in loc de "face <id>".
        self.blank_boxes = 0
        # Identitati preluate de la un track rupt la unul nou.
        self.carried = []


RUN = None

# Evidenta prezentei, completata in main() cand se stiu pragurile din linia de
# comanda. E globala, ca RUN, si din acelasi motiv: probe-ul GStreamer nu are
# unde sa primeasca un parametru in plus.
PRESENCE = None

# ============================================================
# PROBE PRINCIPAL
# ============================================================

def iter_frames(gst_buffer):
    """Cadrele din batch. Aceeasi plimbare prin lista inlantuita, o singura data."""
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            return
        yield frame_meta
        try:
            l_frame = l_frame.next
        except StopIteration:
            return


def iter_objects(frame_meta):
    """Obiectele unui cadru.

    Nodul urmator se citeste DUPA yield, deci consumatorul nu are voie sa stearga
    metadate in timp ce se plimba: intai se strange ce trebuie sters, abia apoi
    se sterge (vezi detection_filter_probe).
    """
    l_obj = frame_meta.obj_meta_list
    while l_obj is not None:
        try:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
        except StopIteration:
            return
        yield obj_meta
        try:
            l_obj = l_obj.next
        except StopIteration:
            return


def detection_filter_probe(pad, info, u_data):
    """Sterge, chiar dupa detector, casetele prea mici ca sa poata fi vreodata fete.

    Tracker-ul e cel mai scump element de dupa nvinfer si costa proportional cu
    numarul de tinte: NvDCF tine cate un filtru prin corelatie pentru fiecare.
    Masurat pe videotest3.mp4, detectorul scotea 42.2 casete pe cadru si doar 4.0
    treceau de porti -- restul erau urmarite fara sa poata ajunge la un model.

    SE FILTREAZA DOAR PE MARIME, niciodata pe incredere. O versiune anterioara
    arunca aici si casetele sub MIN_CONFIDENCE, si asta strica urmarirea exact
    cand conteaza mai mult: cand cineva intoarce capul, detectorul nu rateaza
    fata, ci ii scade increderea la 0.3-0.4. Caseta aia slaba e tot ce are NvDCF
    ca sa lege track-ul mai departe -- e chiar mecanismul pe care se bazeaza
    asocierea in doi pasi din ByteTrack. Aruncata aici, tracker-ul ramane fara
    nicio observatie si trebuie sa mearga pe predictie pana expira track-ul.
    Masurat: 120 de goluri in 13 track-uri, unul de 233 de cadre. Fetele slabe
    sunt oricum respinse mai tarziu, in probe, unde nu strica decat lor.

    Marimea se ia dupa latura MARE, nu dupa cea mica: o fata intoarsa din profil
    se ingusteaza pe orizontala fara sa piarda din inaltime, deci min(w, h) ar
    taia-o taman in cadrele in care tracker-ul are mai multa nevoie de ea. Poarta
    adevarata de marime, cu min(w, h), ramane in probe.

    Stergerea se face DUPA ce s-a terminat plimbarea prin lista: metadatele sunt
    o lista inlantuita, iar nvds_remove_obj_meta_from_frame elibereaza nodul, deci
    un l_obj.next luat dupa stergere ar citi memorie eliberata.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    for frame_meta in iter_frames(gst_buffer):
        doomed = []
        for obj_meta in iter_objects(frame_meta):
            rect = obj_meta.rect_params
            if max(rect.width, rect.height) < PRETRACK_MIN_SIZE:
                doomed.append(obj_meta)
                RUN.rejected_sizes.append(min(rect.width, rect.height))

        RUN.detections += frame_meta.num_obj_meta
        for obj_meta in doomed:
            _remove_obj_meta(frame_meta, obj_meta)
        RUN.detections_kept += frame_meta.num_obj_meta

    return Gst.PadProbeReturn.OK


def media_probe(pad, info, u_data):
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

        started = time.perf_counter()
        frame_number = frame_meta.frame_num

        # Suprafata se foloseste ca atare, fara copie: din ea se decupeaza doar
        # fetele (crop_bgr), iar tot pe ea se deseneaza la final. Copia intreaga
        # de dinainte era cel mai scump lucru din probe si servea doar ca sa fie
        # crop-urile curate -- dar crop_bgr scoate oricum tablouri noi, luate
        # inainte de pasul de desen.
        surface = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        try:
            frame_h, frame_w = surface.shape[:2]
            record = process_frame(frame_meta, frame_number,
                                   frame_w, frame_h, surface)
        finally:
            if _unmap_surface is not None:
                _unmap_surface(hash(gst_buffer), frame_meta.batch_id)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        record["ms"] = r(elapsed_ms, 2)

        logged = time.perf_counter()
        RUN.logger.write(record)
        log_s = time.perf_counter() - logged
        RUN.stage_s["log"] += log_s

        # In probe_ms intra si scrisul in log: ruleaza tot pe firul asta, deci tot
        # el tine cadrul pe loc. "ms" din log ramane doar partea de procesare, ca
        # sa se poata compara cu rularile de dinainte.
        RUN.probe_ms.append(elapsed_ms + log_s * 1000.0)
        RUN.frames += 1

        if PROGRESS_EVERY_FRAMES and frame_number % PROGRESS_EVERY_FRAMES == 0:
            print(f"  cadru {frame_number}: {len(record['faces'])} fete, "
                  f"{len(face_database)} identitati, {elapsed_ms:.1f} ms")

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def process_frame(frame_meta, frame_number, frame_w, frame_h, surface):
    """Toata logica pe un cadru. Intoarce inregistrarea pentru log."""
    pts_seconds = frame_meta.buf_pts / 1e9 if frame_meta.buf_pts else 0.0
    started = time.perf_counter()

    if frame_number % PRUNE_CHECK_INTERVAL == 0:
        stale = [tid for tid, st in track_states.items()
                 if frame_number - st.last_seen_frame > TRACK_TIMEOUT_FRAMES]
        for tid in stale:
            del track_states[tid]

    # --- Pasul 1: strangem fetele vizibile, fara sa atingem GPU-ul ---
    # Respinsele se NUMARA pe poarta, nu se scriu una cate una. Pe un cadru larg
    # detectorul scoate zeci de casete prea mici, iar detaliul lor era 69% din
    # log (1.9 MB din 2.7 MB pe o rulare de 711 cadre) fara sa fie citit vreodata:
    # ce se cauta acolo e "cate si de ce", si aia ramane. Cu --log-respinse se
    # intoarce lista intreaga, pentru cand chiar se vaneaza o caseta anume.
    faces = []
    rejected = Counter()
    rejected_detail = [] if LOG_REJECTED else None
    active = {}
    # Casete care n-au trecut de porti dar apartin unui track cu stare: nu se
    # proceseaza, dar se ETICHETEAZA cu ce stim deja despre om. Fara lista asta
    # ajungeau la nvdsosd neatinse si se desenau cu textul implicit "face <id>".
    held_objects = []
    # Casete fara nicio stare: nici ele nu au voie sa ajunga la nvdsosd neatinse.
    blank_objects = []

    for obj_meta in iter_objects(frame_meta):
        track_id = int(obj_meta.object_id)
        confidence = float(obj_meta.confidence)
        rect = obj_meta.rect_params

        x1 = max(0, int(rect.left))
        y1 = max(0, int(rect.top))
        x2 = min(frame_w, int(rect.left + rect.width))
        y2 = min(frame_h, int(rect.top + rect.height))
        w, h = x2 - x1, y2 - y1

        known = track_states.get(track_id)

        # Poarta de incredere e mai joasa pentru un track pe care il stim deja:
        # tracker-ul a legat caseta de un istoric, deci nu mai avem nevoie de
        # detector ca sa ne convinga ca acolo e o fata. Cand cineva intoarce
        # capul, increderea cade la 0.3-0.4 -- si tocmai atunci nu vrem sa-l
        # scoatem complet din procesare, pentru ca atunci ii pierdem si sansa de
        # a fi re-verificat, nu doar cadrul.
        min_confidence = TRACKED_MIN_CONFIDENCE if known is not None else MIN_CONFIDENCE

        if confidence < min_confidence:
            gate = "incredere_mica"
        elif w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            gate = "prea_mica"
        else:
            gate = None

        if gate is not None:
            rejected[gate] += 1
            if gate == "prea_mica":
                RUN.rejected_sizes.append(min(w, h))
            if rejected_detail is not None:
                rejected_detail.append({"track": track_id, "box": [x1, y1, x2, y2],
                                        "det": r(confidence), "poarta": gate})

            # Fata n-a trecut de porti, dar track-ul e viu si tracker-ul inca il
            # tine: il marcam ca vazut, ca sa nu-i stergem starea. Altfel un om
            # care sta intors 300 de cadre (masurat: goluri de pana la 233) isi
            # pierde eticheta, streak-ul si prototipul, revine ca necunoscut si
            # poate fi inrolat a doua oara, ca identitate dublata.
            #
            # NU se creeaza stare noua aici: ar insemna cate un TrackState pentru
            # fiecare dintre zecile de casete mici de pe cadru, care n-au trecut
            # niciodata de porti si nici n-o sa treaca.
            if known is not None:
                known.last_seen_frame = frame_number
                known.last_box = (x1, y1, x2, y2)
                known.held_frames += 1
                held_objects.append((track_id, known, obj_meta, gate, (x1, y1, x2, y2)))
            else:
                blank_objects.append(obj_meta)
            continue

        state = known
        if state is None:
            # Track nou. Daca tocmai s-a rupt unul cu identitate exact in locul
            # asta, e mai probabil sa fie acelasi om decat unul nou: ii preluam
            # numele provizoriu, ca sa nu apara o caseta anonima peste cineva pe
            # care tocmai il stiam. Prima verificare confirma sau rastoarna.
            state = TrackState()
            track_states[track_id] = state
            if CARRY_IDENTITY:
                donor_id, donor, overlap = find_identity_donor(
                    track_id, (x1, y1, x2, y2), frame_number)
                if donor is not None:
                    event = state.inherit(donor, donor_id, overlap, frame_number)
                    event["track_nou"] = track_id
                    RUN.carried.append(event)
                    print(f"[CONTINUITATE] track {track_id} preia {donor.identity} "
                          f"de la track {donor_id} (iou {overlap:.2f}, "
                          f"cadru {frame_number})")

        state.last_seen_frame = frame_number
        state.last_box = (x1, y1, x2, y2)
        active[track_id] = state
        report = track_reports.setdefault(
            track_id, {"primul_cadru": frame_number, "verificari": 0,
                       "eticheta": None, "scor": 0.0, "inrolat": False,
                       "marime_maxima": 0}
        )
        report["ultimul_cadru"] = frame_number
        # Cat de mare a ajuns fata track-ului, oricand in viata lui. Asta arata pe
        # loc care track-uri au fost taiate DOAR de ENROLL_MIN_FACE: unul cu
        # marime_maxima 48 si prag 50 n-a avut nicio sansa, oricat de bine ar fi
        # fost privit.
        report["marime_maxima"] = max(report["marime_maxima"], min(w, h))

        crop = crop_bgr(surface, (x1, y1, x2, y2))
        faces.append(FaceSample(track_id, state, obj_meta, crop,
                                blur_score(crop), min(w, h), (x1, y1, x2, y2)))

    RUN.faces_seen += len(faces)
    RUN.stage_s["citire"] += time.perf_counter() - started

    # --- Pasul 2: cine cere o proba in cadrul asta ---
    # Poarta asta -- fereastra dinaintea termenului si ritmul probelor -- nu are
    # nevoie de landmark-uri, deci se poate calcula inainte de ele. Aceleasi
    # reguli ca la varianta live, ca deciziile sa ramana comparabile.
    for face in faces:
        state = face.state
        frames_since_check = frame_number - state.last_checked_frame
        window_open = frames_since_check >= state.deadline_gap - QUALITY_WINDOW_FRAMES
        due_for_sample = frame_number - state.last_candidate_frame >= CANDIDATE_INTERVAL_FRAMES
        face.wants_sample = window_open and due_for_sample

    # --- Pasul 3: landmark-uri + aliniere + calitate, intr-un singur apel GPU ---
    # NU pe toate fetele. Un track fara identitate primeste landmark-uri pe fiecare
    # cadru, pentru ca din ele se alege prototipul si un track poate deveni bun
    # exact intre doua verificari (masurat pe sample_vid.mp4: track 47 avea un
    # singur cadru bun, la 161, intre verificarile de la 152 si 162). Dar un om
    # deja recunoscut nu mai are ce prototip sa caute: pentru el se calculeaza
    # ceva doar cand da o proba, adica in fereastra dinaintea urmatoarei
    # verificari. Intre timp ramane doar urmarit de tracker.
    #
    # Aici era grosul costului: 63% din timpul de probe se ducea pe landmark-uri,
    # iar pe un clip cu oameni deja in baza aproape toate erau pe ei.
    started = time.perf_counter()
    if LANDMARK_ONLY_WHEN_NEEDED:
        # needs_landmarks, nu "not resolved": un track a carui identitate tocmai
        # a fost contrazisa (sau preluata provizoriu de la un track rupt) iese din
        # racire si primeste iar landmark-uri pe fiecare cadru, ca sa ajunga la
        # urmatoarea verificare cu cea mai buna poza posibila.
        analysed = [face for face in faces
                    if face.wants_sample or face.state.needs_landmarks]
    else:
        analysed = faces
    analyse_faces(analysed)
    RUN.stage_s["landmark"] += time.perf_counter() - started

    # --- Pasul 4: statistici + prototipul, indiferent de cadenta verificarilor ---
    for face in faces:
        # Marimea si claritatea se stiu fara GPU, deci se numara pentru toti.
        RUN.face_sizes.append(face.size)
        RUN.face_blurs.append(face.blur)
        if face.landmarks is None:
            # Fara landmark-uri nu exista nici calitate, nici frontalitate: ele NU
            # se pun in distributii ca zerouri, altfel raportul ar arata o filmare
            # din profil acolo unde de fapt n-am masurat nimic.
            face.action = "urmarit"
            continue
        RUN.face_qualities.append(face.quality)
        RUN.face_frontalities.append(face.frontality)
        if face.aligned is not None:
            face.state.offer_unknown(face.aligned, face.quality, face.frontality,
                                     face.blur, face.size, frame_number)

    # --- Pasul 5: care fete intra in concursul pentru "cel mai bun cadru" ---
    for face in faces:
        state = face.state
        if not face.wants_sample:
            continue

        # Ritmul probelor se respecta si cand proba e aruncata: altfel o fata
        # neclara ar fi reincercata pe fiecare cadru din fereastra.
        state.last_candidate_frame = frame_number

        if face.blur < MIN_BLUR * BLUR_REJECT_FACTOR:
            face.action = "sarit_blur"
        elif face.aligned is None:
            face.action = "aliniere_esuata"
        else:
            # Probele sub MIN_BLUR intra in concurs, dar se vad in log: daca in
            # toata rularea nu apare decat "proba_neclara", pragul de claritate e
            # nepotrivit pentru filmarea asta, nu fetele sunt de vina.
            face.action = "proba" if face.blur >= MIN_BLUR else "proba_neclara"
            state.offer(face.aligned, face.quality, face.frontality,
                        face.blur, face.size)

    # --- Pasul 6: pentru cine rulam recunoasterea in acest cadru ---
    to_recognize = []
    for track_id, state in active.items():
        due = frame_number - state.last_checked_frame >= state.deadline_gap
        usable = state.best_aligned is not None and state.best_quality >= QUALITY_MIN
        good_enough = state.best_quality >= QUALITY_GOOD_ENOUGH

        if usable and (good_enough or due):
            state.last_checked_frame = frame_number
            state.last_check_failed = False
            to_recognize.append((track_id, state))
        elif due:
            state.last_checked_frame = frame_number
            state.last_check_failed = True
            if state.best_aligned is None:
                RUN.checks_skipped["fara_proba"] += 1
            else:
                RUN.checks_skipped["calitate_sub_QUALITY_MIN"] += 1
                RUN.deadline_qualities.append(state.best_quality)
            state.clear_best()

    # --- Pasul 7: verificare (cine e), fara inrolare ---
    # Un singur apel GPU pentru toate fetele alese la pasul anterior.
    started = time.perf_counter()
    embeddings = embed_aligned([state.best_aligned for _, state in to_recognize])
    RUN.stage_s["recunoastere"] += time.perf_counter() - started
    decisions = {}
    for (track_id, state), embedding in zip(to_recognize, embeddings):
        label, score = verify_embedding(embedding)
        quality, blur, size = state.best_quality, state.best_blur, state.best_size
        front = state.best_frontality
        state.clear_best()
        state.checks += 1
        RUN.recognitions += 1

        blockers = None
        if AUTO_ENROLL and label == LABEL_UNKNOWN and not state.enrolled:
            state.unknown_streak += 1
            state.last_unknown_score = score
            blockers = enroll_blockers(state, score, size, blur, quality, front)
            if blockers:
                for blocker in blockers:
                    RUN.enroll_blockers[blocker] += 1
                RUN.enroll_attempts += 1
        else:
            if AUTO_ENROLL and label == LABEL_UNCERTAIN and not state.enrolled:
                # A semanat destul cu cineva din baza cat sa cada in banda de
                # incertitudine: nu inrolam (am dubla o identitate existenta), dar
                # se numara. Daca grosul blocajelor e aici, in cale nu sta o
                # poarta de calitate, ci VERIFY_THRESHOLD/ENROLL_MARGIN fata de
                # baza cu care s-a pornit.
                RUN.enroll_blockers["incert"] += 1
                RUN.enroll_attempts += 1
            state.unknown_streak = 0

        state.history.append(label)
        voted = majority_label(state.history)
        # Votul NU se mai muta direct in eticheta afisata. Trece intai prin
        # histerezisul din apply_decision: o identitate deja confirmata nu cade
        # dintr-o singura fereastra proasta. Vezi punctul B din antet.
        outcome = state.apply_decision(voted, score, frame_number)
        RUN.identity_events[outcome] += 1

        report = track_reports[track_id]
        report["verificari"] = state.checks
        report["eticheta"] = state.current_label
        report["identitate"] = state.identity
        report["scor"] = r(state.current_score)

        decisions[track_id] = {
            "eticheta_bruta": label, "vot": voted, "efect": outcome,
            "scor": r(score), "calitate": r(quality),
            "frontalitate": r(front), "blur": r(blur, 1), "marime": size,
            "streak": state.unknown_streak, "inrolat": False,
        }
        if outcome == "tinut":
            decisions[track_id]["identitate_tinuta"] = state.identity
            decisions[track_id]["contraziceri"] = state.contra_checks
        best = state.best_unknown
        if blockers:
            decisions[track_id]["blocaje"] = blockers
            # prototipul strans pana acum, ca sa se vada daca track-ul are deja o
            # poza buna si asteapta doar streak-ul, sau inca n-a prins niciuna
            decisions[track_id]["cea_mai_buna_proba"] = prototype_record(best)

        # Se scrie doar cand se SCHIMBA ceva, nu la fiecare confirmare a unui om
        # deja cunoscut: altfel consola se umple cu acelasi nume la fiecare
        # IDENTIFIED_INTERVAL_FRAMES si ce conteaza cu adevarat -- o identitate
        # tinuta sau pierduta -- trece neobservat printre ele.
        if outcome == "confirmat" and state.checks_confirmed == 1:
            print(f"[ALERTA] track {track_id} -> {state.identity} "
                  f"(scor={score:.3f}, calitate={quality:.2f}, cadru={frame_number})")
        elif outcome == "tinut":
            print(f"[CONTINUITATE] track {track_id}: verificarea a iesit '{voted}' "
                  f"(scor={score:.3f}), dar raman pe {state.identity} "
                  f"({state.contra_checks}/{IDENTITY_FLIP_CHECKS} contraziceri, "
                  f"cadru={frame_number})")
        elif outcome == "schimbat":
            print(f"[ALERTA] track {track_id} si-a schimbat omul -> {state.identity} "
                  f"(scor={score:.3f}, cadru={frame_number})")
        elif outcome == "pierdut":
            print(f"[ALERTA] track {track_id} si-a pierdut identitatea dupa "
                  f"{IDENTITY_FLIP_CHECKS} verificari '{voted}' (cadru={frame_number})")

    # --- Pasul 8: inrolarea, decuplata de cadenta verificarilor ---
    # Un track se inroleaza cand are si dovada ca nu e in baza (streak-ul de
    # verificari "necunoscut"), si o poza buna -- in ORICE ordine ar veni cele
    # doua. Legate de momentul verificarii, se pierdeau track-urile care devin
    # bune imediat dupa o verificare si dispar inainte de urmatoarea (masurat:
    # track 65 avea cadre bune la 223-226, cu verificari la 212 si 222).
    ready = [(track_id, state) for track_id, state in active.items()
             if ready_to_enroll(state)]
    for (track_id, state), embedding in zip(ready, embed_aligned(
            [state.best_unknown["aligned"] for _, state in ready])):
        best = state.best_unknown
        name = enroll(embedding)
        # Inrolarea e cea mai tare confirmare posibila: prototipul TOCMAI a fost
        # pus in baza, deci identitatea nu mai trece prin histerezis.
        state.confirm_enrolled(name, frame_number)
        RUN.identity_events["inrolat"] += 1

        track_reports[track_id]["eticheta"] = name
        track_reports[track_id]["identitate"] = name
        track_reports[track_id]["inrolat"] = True
        RUN.enrollments.append(dict({"nume": name, "cadru": frame_number,
                                     "track": track_id}, **prototype_record(best)))
        decisions.setdefault(track_id, {}).update(
            {"eticheta_bruta": name, "inrolat": True, "scor": 1.0,
             "streak": state.unknown_streak,
             "cea_mai_buna_proba": prototype_record(best)}
        )
        print(f"[INROLARE] track {track_id} -> {name} din cadrul {best['frame']} "
              f"({best['size']}px, frontalitate {best['frontality']:.2f}, "
              f"calitate {best['quality']:.2f})")

    # --- Pasul 9: desen ---
    for face in faces:
        if face.track_id in decisions:
            face.action = "recunoscut"

    # --- Pasul 9b: cine e in cadru, cine a plecat, cine s-a intors ---
    # Se aduna dupa deciziile cadrului, ca o intoarcere sa fie anuntata chiar in
    # cadrul in care a fost confirmata, nu unul mai tarziu.
    #
    # Intra si track-urile care n-au trecut de porti in cadrul asta (held_objects):
    # un om intors cu spatele e in continuare in cadru. Fara ei, prezenta ar
    # palpai la fiecare cap intors si "s-a intors" ar aparea pentru cineva care nu
    # plecase -- exact ce se semnala ca risc.
    #
    # Se cere checks_confirmed >= 1: identitatea trebuie sa fi trecut macar o data
    # prin modelul de recunoastere. Un id de tracker sau o identitate preluata
    # geometric de la un track rupt nu declanseaza nimic.
    visible = [state for state in active.values()]
    visible.extend(state for _, state, _, _, _ in held_objects)
    present_names = {state.identity for state in visible
                     if state.identity is not None and state.checks_confirmed >= 1}
    for kind, name, gone in PRESENCE.update(present_names, time.monotonic()):
        if kind == "revenit":
            print(f"[REVENIT] {name} s-a intors dupa {gone:.1f} s")
        elif kind == "reaparut":
            print(f"[REAPARUT] {name} e din nou in cadru, dupa {gone:.1f} s "
                  f"(peste fereastra de {RETURN_WINDOW_S:.0f} s)")
        elif kind == "aparut" and ANNOUNCE_DEPARTURE:
            print(f"[PREZENT] {name} a intrat in cadru")
        elif kind == "plecat" and ANNOUNCE_DEPARTURE:
            print(f"[PLECAT] {name} a iesit din cadru")

    # Casetele tinute se numara si fara desen: ele sunt masura reparatiei, iar cu
    # --no-video trebuie sa se vada in summary la fel ca la o rulare cu video.
    for _, state, _, _, _ in held_objects:
        RUN.held_boxes += 1
        if frame_number - state.identity_frame > IDENTITY_STALE_FRAMES:
            RUN.stale_boxes += 1
    RUN.blank_boxes += len(blank_objects)

    # Fara video de iesire nu are cine sa vada desenul, deci nu il mai facem:
    # scutim si conversiile, si scrisul in suprafata.
    if DRAW_OVERLAY:
        started = time.perf_counter()
        for face in faces:
            # measured = i s-au cerut landmark-uri in cadrul asta. Un om
            # recunoscut aflat in racire are caseta desenata din memorie, si e
            # cinstit sa se vada asta (verde stins, "~"), nu sa para masurat acum.
            label_object(face.obj_meta, face.state, frame_number,
                         measured=face.landmarks is not None)
        # Aici e reparatia propriu-zisa a simptomului: casetele care n-au trecut
        # de porti primesc ELE ce stim despre om, in loc sa fie lasate cu textul
        # implicit al lui nvinfer ("face 11").
        for _, state, obj_meta, _, _ in held_objects:
            label_object(obj_meta, state, frame_number, measured=False)
        for obj_meta in blank_objects:
            blank_object(obj_meta)
        draw_landmarks(surface, faces)
        RUN.stage_s["desen"] += time.perf_counter() - started

    # Detectiile fara stare pot fi scoase de tot din metadate, daca deranjeaza si
    # caseta gri. Se face DUPA plimbarea prin lista si dupa desen, pentru ca
    # nvds_remove_obj_meta_from_frame elibereaza nodul (vezi iter_objects).
    if HIDE_REJECTED_BOXES and _remove_obj_meta is not None:
        for obj_meta in blank_objects:
            _remove_obj_meta(frame_meta, obj_meta)

    # --- Pasul 10: inregistrarea pentru log ---
    face_records = []
    for face in faces:
        state = face.state
        entry = {
            "track": face.track_id,
            "box": [int(v) for v in face.box],
            "det": r(float(face.obj_meta.confidence)),
            "blur": r(face.blur, 1),
            "actiune": face.action,
            "eticheta": state.current_label,
            "scor": r(state.current_score),
            # Ce se AFISEAZA, care de acum poate sa difere de ultima masuratoare.
            "identitate": state.identity,
        }
        if state.doubted:
            entry["contraziceri"] = state.contra_checks
        if face.landmarks is not None:
            entry["calitate"] = r(face.quality)
            entry["frontalitate"] = r(face.frontality)
            entry["puncte5"] = [[r(x, 1), r(y, 1)] for x, y in face.five_points]
        if face.track_id in decisions:
            entry["decizie"] = decisions[face.track_id]
        face_records.append(entry)

    # Casetele care n-au trecut de porti dar poarta o identitate stiuta. Nu intra
    # in "faces" (pe ele nu s-a masurat nimic si n-au voie sa intre in
    # distributii sau in fete_procesate), dar trebuie sa se vada in log: fara
    # ele, un om intors 200 de cadre lipseste cu totul din log tocmai in cadrele
    # in care se vede pe ecran cu numele lui.
    held_records = [
        {"track": track_id, "box": [int(v) for v in box], "poarta": gate,
         "identitate": state.identity, "scor": r(state.identity_score),
         "vechime_cadre": frame_number - state.identity_frame}
        for track_id, state, _, gate, box in held_objects
    ]

    record = {
        "cadru": frame_number,
        "timp": r(pts_seconds),
        "faces": face_records,
        "respinse": dict(rejected),
        "gpu": {"landmark": len(faces), "recunoastere": len(to_recognize)},
        "identitati": len(face_database),
    }
    if present_names:
        record["prezenti"] = sorted(present_names)
    if held_records:
        record["tinute"] = held_records
    if blank_objects:
        record["anonime"] = len(blank_objects)
    if rejected_detail:
        record["respinse_detaliu"] = rejected_detail
    return record

# ============================================================
# PIPELINE GSTREAMER
# ============================================================

def make_element(factory_names, name):
    """Primul element disponibil din lista (Jetson vs. desktop au alte encodere)."""
    if isinstance(factory_names, str):
        factory_names = [factory_names]
    for factory in factory_names:
        element = Gst.ElementFactory.make(factory, name)
        if element:
            if len(factory_names) > 1:
                print(f"  {name}: {factory}")
            return element
    raise RuntimeError(
        f"Niciunul dintre elementele {factory_names} nu e disponibil "
        f"(lipseste un plugin GStreamer?)."
    )


def make_queue(name, depth):
    """Un queue cu limite doar pe numarul de buffere, niciodata cu pierderi.

    Fara queue-uri, TOT pipeline-ul ruleaza pe un singur fir: fiecare element il
    apeleaza direct pe urmatorul, deci cele ~43 ms petrecute in probe se adunau
    la cele ~57 ms de decodare, detectie, urmarire si encodare in loc sa se
    suprapuna cu ele. Un queue rupe lantul in doua fire, iar cele doua bucati
    merg in paralel pe cadre diferite.

    Limitele pe octeti si pe timp se scot, altfel se ating primele si queue-ul
    blocheaza mai devreme decat vrem; adancimea se tine mica pentru ca fiecare
    buffer in asteptare e o suprafata NVMM (la 1080x1920 RGBA, ~8 MB bucata).
    Fara pierderi (leaky ramane 0): se logheaza fiecare cadru, deci un cadru
    aruncat ar fi o gaura in raport, nu doar o imagine lipsa.
    """
    queue = make_element("queue", name)
    queue.set_property("max-size-buffers", depth)
    queue.set_property("max-size-bytes", 0)
    queue.set_property("max-size-time", 0)
    return queue


# ------------------------------------------------------------
# Orientarea
# ------------------------------------------------------------
# Pe camera nu exista eticheta de rotatie din container, deci rotatia se cere
# explicit, cu --rotate. E de folos cand camera e montata pe o parte (montaj pe
# perete, senzor CSI intors): fara ea, fetele ajung culcate la detector si nu le
# mai gaseste. "Rotatie" inseamna peste tot cate grade IN SENSUL ACELOR DE CEAS
# trebuie rotita imaginea ca sa se vada corect.

# nvvideoconvert: 1 = 90 grade trigonometric, 2 = 180, 3 = 90 in sensul acelor
# de ceas.
ROTATION_TO_FLIP_METHOD = {90: 3, 180: 2, 270: 1}


# ------------------------------------------------------------
# Camera
# ------------------------------------------------------------
# Spre deosebire de varianta pe fisier, aici nu se sondeaza nimic inainte: nu
# exista antet de citit, iar deschiderea camerei ca s-o interoghezi si apoi
# inchiderea ei ca s-o redeschizi din GStreamer e cea mai buna cale spre "Device
# or resource busy". Rezolutia se CERE camerei, prin caps; daca driverul nu poate
# da exact ce s-a cerut, negocierea esueaza si mesajul de pe magistrala spune ce
# formate exista (vezi si explain_camera_error).


def camera_kind(args):
    """v4l2 (webcam USB) sau csi (senzor pe placa), cu 'auto' rezolvat aici.

    Auto inseamna: daca s-a dat --sensor-id sau nu exista fisierul de device,
    incearca CSI; altfel v4l2. Pe un Jetson cu camera pe panglica nu exista
    /dev/video0 decat dupa ce porneste argus, deci ordinea asta ghiceste corect
    in cazul obisnuit, si oricum se poate spune explicit cu --camera.
    """
    if args.camera != "auto":
        return args.camera
    if args.sensor_id is not None:
        return "csi"
    if os.path.exists(args.device):
        return "v4l2"
    if Gst.ElementFactory.find("nvarguscamerasrc") is not None:
        return "csi"
    return "v4l2"


def build_camera_source(args, kind):
    """Elementele de la camera pana la NVMM/NV12, in ordinea legarii.

    Intoarce (lista de elemente, descriere pentru consola). Ultimul element din
    lista e cel care se leaga mai departe la queue-ul de dinaintea lui streammux.

    Doua drumuri, pentru ca cele doua camere livreaza lucruri diferite:

      CSI (nvarguscamerasrc) scoate direct NVMM/NV12, deci nu are nevoie de nimic
        pe CPU: se cere rezolutia si rata in caps si gata.
      v4l2 scoate in memorie de sistem, de obicei YUY2 sau MJPEG. MJPEG se
        decodeaza cu jpegdec (CPU), YUY2 se converteste cu videoconvert (CPU).
        Abia dupa aceea nvvideoconvert urca datele in NVMM. Conversia pe CPU e
        pretul webcam-ului USB si se vede in FPS la 1080p; la 720p e neglijabila.
        --mjpeg merita cerut cand camera o suporta: la aceeasi rezolutie, iese
        mult mai putin trafic pe USB, deci rate mai mari.
    """
    if kind == "csi":
        source = make_element("nvarguscamerasrc", "camera")
        source.set_property("sensor-id", args.sensor_id or 0)
        caps = make_element("capsfilter", "caps-camera")
        caps.set_property("caps", Gst.Caps.from_string(
            f"video/x-raw(memory:NVMM), width={args.capture_width}, "
            f"height={args.capture_height}, framerate={args.fps}/1, format=NV12"))
        return [source, caps], (f"nvarguscamerasrc sensor-id="
                                f"{args.sensor_id or 0}")

    source = make_element("v4l2src", "camera")
    source.set_property("device", args.device)
    # io-mode=2 (mmap) e ce vor aproape toate webcam-urile UVC; fara el, unele
    # driver-e cad pe read() si dau rate mult mai mici.
    if source.find_property("io-mode"):
        source.set_property("io-mode", 2)

    caps = make_element("capsfilter", "caps-camera")
    if args.mjpeg:
        caps.set_property("caps", Gst.Caps.from_string(
            f"image/jpeg, width={args.capture_width}, "
            f"height={args.capture_height}, framerate={args.fps}/1"))
        decoder = make_element(["nvjpegdec", "jpegdec"], "decodor-mjpeg")
        chain = [source, caps, decoder]
    else:
        caps.set_property("caps", Gst.Caps.from_string(
            f"video/x-raw, width={args.capture_width}, "
            f"height={args.capture_height}, framerate={args.fps}/1"))
        chain = [source, caps]

    # videoconvert acopera orice format ar scoate camera (YUY2, RGB, I420...);
    # nvvideoconvert singur nu accepta toate formatele de intrare din memorie de
    # sistem, si atunci negocierea pica cu "not-negotiated" fara alta explicatie.
    convert = make_element("videoconvert", "conversie-camera")
    chain.append(convert)
    return chain, (f"v4l2src {args.device}, "
                   f"{'MJPEG' if args.mjpeg else 'raw'} "
                   f"{args.capture_width}x{args.capture_height}@{args.fps}")


def explain_camera_error(text, args):
    """Traduce erorile obisnuite de camera in ce e de facut."""
    lowered = (text or "").lower()
    if "not-negotiated" in lowered or "internal data stream error" in lowered:
        print("\n[CAMERA] Camera nu poate da exact ce i s-a cerut "
              f"({args.capture_width}x{args.capture_height}@{args.fps}"
              f"{', MJPEG' if args.mjpeg else ''}).")
        print(f"         Vezi ce stie:  v4l2-ctl -d {args.device} "
              f"--list-formats-ext")
        print("         Apoi da exact o combinatie de acolo, cu "
              "--capture-width/--capture-height/--fps (si --mjpeg daca e cazul).")
    elif "busy" in lowered or "permission denied" in lowered:
        print(f"\n[CAMERA] {args.device} e ocupata sau nu ai drepturi pe ea.")
        print(f"         Cine o tine:   fuser -v {args.device}")
        print("         Drepturi:      adauga utilizatorul in grupul 'video'.")
    elif "no such device" in lowered or "cannot identify device" in lowered:
        print(f"\n[CAMERA] {args.device} nu exista. Camerele vazute de sistem:")
        print("         ls -l /dev/video*    sau    v4l2-ctl --list-devices")


def rotation_supported():
    """nvvideoconvert stie sa roteasca? (flip-method nu exista pe toate platformele)"""
    element = Gst.ElementFactory.make("nvvideoconvert", None)
    return element is not None and element.find_property("flip-method") is not None


def even(value):
    """Latimi/inaltimi impare strica NV12 si encoder-ul; rotunjim la par."""
    return max(2, int(round(value / 2.0)) * 2)


def auto_bitrate(width, height):
    """Bitrate potrivit rezolutiei, in biti/s.

    4 Mbit/s fix insemna ~0.06 biti pe pixel pe cadru la 1080p si de patru ori
    mai putin la 4K -- de aici imaginea moale, cu blocuri in scenele cu miscare.
    Aproximativ 4 biti pe pixel pe secunda inseamna 8 Mbit/s la 1080p si 33 la 4K.
    """
    return int(width * height * 4)


def working_resolution(args):
    """Rezolutia la care ruleaza pipeline-ul, pornind de la cea a camerei.

    Implicit e chiar rezolutia de captura: la o camera nu exista motivul de la
    fisiere pentru care s-ar micsora (nu se decodeaza 4K in bufferele dinaintea
    lui streammux). Cine vrea sa lucreze mai jos decat filmeaza camera -- de
    exemplu capteaza 1920x1080 dar proceseaza 1280x720 -- da --width/--height.

    ATENTIE la praguri cand se face asta: MIN_FACE_SIZE, ENROLL_MIN_FACE si
    celelalte sunt in pixeli la rezolutia de LUCRU, deci o injumatatire a laturii
    injumatateste si fetele, nu doar cadrul.
    """
    width, height = args.capture_width, args.capture_height
    if args.rotation in (90, 270):
        width, height = height, width

    if args.width and args.height:
        given = args.width / float(args.height)
        actual = width / float(height)
        if abs(given - actual) / actual > 0.01:
            print(f"[AVERTISMENT] --width/--height ({args.width}x{args.height}) au alt "
                  f"raport decat camera ({width}x{height}); imaginea va fi deformata, "
                  f"iar detectia are de suferit.")
        return even(args.width), even(args.height)
    if args.width:
        return even(args.width), even(args.width * height / float(width))
    if args.height:
        return even(args.height * width / float(height)), even(args.height)
    return even(width), even(height)


def build_pipeline(args, output_video):
    pipeline = Gst.Pipeline()
    if not pipeline:
        raise RuntimeError("Nu s-a putut crea pipeline-ul.")

    print("Creare elemente pipeline...")

    camera_chain, camera_desc = build_camera_source(args, args.camera_kind)
    print(f"  sursa: {camera_desc}")

    # Convertorul de dupa camera urca datele in NVMM si le duce la rezolutia de
    # lucru. La CSI intrarea e deja NVMM si atunci nu face decat scalarea; la
    # v4l2 face si urcarea in memoria GPU.
    vidconv_in = make_element("nvvideoconvert", "conversie-intrare")

    # Rotatia se face tot aici, inaintea capsfilter-ului: dupa ea, latimea si
    # inaltimea sunt deja cele din caps (vezi working_resolution, care le-a
    # inversat pentru 90/270).
    if args.rotation:
        vidconv_in.set_property("flip-method", ROTATION_TO_FLIP_METHOD[args.rotation])
        print(f"  conversie-intrare: rotesc {args.rotation} grade "
              f"(flip-method={ROTATION_TO_FLIP_METHOD[args.rotation]})")

    caps_in = make_element("capsfilter", "caps-intrare")
    caps_in.set_property("caps", Gst.Caps.from_string(
        f"video/x-raw(memory:NVMM), format=NV12, "
        f"width={args.width}, height={args.height}"))

    streammux = make_element("nvstreammux", "stream-muxer")
    streammux.set_property('width', args.width)
    streammux.set_property('height', args.height)
    streammux.set_property('batch-size', 1)
    # live-source=1: sursa nu poate fi incetinita. Pe fisier, daca pipeline-ul nu
    # tine pasul, decodorul asteapta si nu se pierde niciun cadru; camera, in
    # schimb, produce mai departe indiferent ce facem noi, iar streammux trebuie
    # sa stie asta ca sa nu incerce sa sincronizeze dupa timestamp-uri.
    streammux.set_property('live-source', 1)
    # Timeout-ul de batch, in microsecunde: cat asteapta streammux dupa un cadru
    # inainte sa trimita batch-ul asa cum e. Il legam de rata camerei -- la 30
    # FPS, un cadru vine la 33 ms -- ca sa nu adaugam latenta degeaba.
    streammux.set_property('batched-push-timeout', int(1e6 / max(1, args.fps)))

    pgie = make_element("nvinfer", "detector-fete")
    pgie.set_property('config-file-path', args.pgie_config)

    tracker = make_element("nvtracker", "tracker")
    # nvtracker scaleaza cadrul la dimensiunile astea; pe o filmare verticala,
    # 640x384 (16:9) inseamna ca tracker-ul lucreaza pe o imagine turtita, cu
    # fete deformate altfel decat cele pe care le-a dat detectorul. Le intoarcem
    # dupa cadru. Raman multipli de 32, cum cere DeepStream.
    tracker_width, tracker_height = (640, 384) if args.width >= args.height else (384, 640)
    tracker.set_property('tracker-width', tracker_width)
    tracker.set_property('tracker-height', tracker_height)
    tracker.set_property('gpu-id', 0)
    tracker.set_property('ll-lib-file',
                         "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property('ll-config-file', args.tracker_config)

    # Scalarea cadrelor pentru tracker se face pe GPU, nu pe VIC. Pe Jetson,
    # implicit ajunge la VIC, iar la rezolutiile care nu-i convin scoate "NvVic
    # handle" sau erori 12 (memorie CMA) si pipeline-ul cade. Proprietatea nu
    # exista pe toate versiunile de DeepStream, deci o punem doar daca e acolo.
    if tracker.find_property("compute-hw"):
        tracker.set_property('compute-hw', 1)       # 0 implicit, 1 GPU, 2 VIC
        print("  tracker: compute-hw=1 (GPU)")

    vidconv_osd = make_element("nvvideoconvert", "conversie-osd")
    # Un queue nu se poate umple peste cate buffere are pool-ul elementului de
    # dinaintea lui: cu pool-ul implicit (4), cele trei fire s-ar bloca unul pe
    # altul si suprapunerea s-ar pierde. Proprietatea nu exista pe toate
    # versiunile de DeepStream, deci se pune doar daca e acolo.
    for converter in (vidconv_in, vidconv_osd):
        if converter.find_property("output-buffers"):
            converter.set_property("output-buffers", args.queue_size + 4)

    caps_rgba = make_element("capsfilter", "caps-rgba")
    caps_rgba.set_property('caps', Gst.Caps.from_string(
        "video/x-raw(memory:NVMM), format=RGBA"))

    # Iesirea. Trei variante, si toate trec prin acelasi nvdsosd -- desenul se
    # face o singura data, indiferent cine se uita la el:
    #
    #   implicit         osd -> fereastra pe ecran
    #   --record         osd -> tee -> fereastra
    #                              \-> encoder -> mp4
    #   --no-display     osd -> encoder -> mp4        (sau fakesink, daca nici nu
    #                                                  se inregistreaza)
    #
    # Fara fereastra si fara inregistrare ramane doar consola: recunoasterea,
    # anunturile de intoarcere, baza de date si logul ies neschimbate, iar de pe
    # placa dispar si encoderul, si sink-ul de afisare.
    branches = []       # liste de elemente, fiecare legata la cate un pad de tee

    def display_branch():
        """nvvideoconvert -> (nvegltransform) -> sink de ecran."""
        convert = make_element("nvvideoconvert", "conversie-afisare")
        # nv3dsink e sink-ul de pe JetPack 5 si primeste NVMM direct.
        # nveglglessink are nevoie inaintea lui de nvegltransform pe Jetson;
        # fara el, negocierea pica cu "not-negotiated" fara alt indiciu.
        sink = make_element(["nv3dsink", "nveglglessink", "xvimagesink",
                             "autovideosink"], "afisare")
        factory = sink.get_factory().get_name()
        # sync=False: nu incetinim pipeline-ul ca sa respectam timestamp-uri.
        # Pe o sursa live, ce apare pe ecran trebuie sa fie ultimul cadru
        # procesat, nu unul asteptat pana la momentul lui teoretic.
        if sink.find_property("sync"):
            sink.set_property("sync", False)
        if sink.find_property("async"):
            sink.set_property("async", False)
        if factory == "nveglglessink":
            transform = make_element("nvegltransform", "transform-egl")
            return [convert, transform, sink]
        return [convert, sink]

    def record_branch():
        """nvvideoconvert -> encoder -> h264parse -> qtmux -> filesink."""
        convert = make_element("nvvideoconvert", "conversie-inregistrare")
        caps_out = make_element("capsfilter", "caps-inregistrare")
        encoder = make_element(["nvv4l2h264enc", "x264enc"], "encoder")
        bitrate = args.bitrate or auto_bitrate(args.width, args.height)
        print(f"  bitrate: {bitrate / 1e6:.1f} Mbit/s"
              + ("" if args.bitrate else " (calculat din rezolutie)"))

        if encoder.get_factory().get_name() == "nvv4l2h264enc":
            encoder.set_property('bitrate', bitrate)
            # Implicit, nvv4l2h264enc encodeaza in Baseline (profilul 66 din log):
            # fara CABAC si fara cadre B, adica exact ce se vede ca "imagine slaba"
            # la acelasi bitrate. High costa la fel de mult de encodat pe NVENC.
            if encoder.find_property("profile"):
                encoder.set_property('profile', 4)      # 0 Baseline, 2 Main, 4 High
            # Bitrate variabil: scenele simple consuma putin, iar cele cu miscare --
            # exact acolo unde se pierd fetele -- primesc cat le trebuie.
            if encoder.find_property("control-rate"):
                encoder.set_property('control-rate', 0)  # 0 variabil, 1 constant
            if encoder.find_property("peak-bitrate"):
                encoder.set_property('peak-bitrate', int(bitrate * 1.5))
            # Pe o sesiune live, un keyframe la cateva secunde inseamna ca
            # fisierul e citibil de la mijloc si ca o oprire brutala pierde cel
            # mult atatea cadre.
            if encoder.find_property("iframeinterval"):
                encoder.set_property('iframeinterval', max(1, args.fps * 2))
            caps_out.set_property('caps', Gst.Caps.from_string(
                "video/x-raw(memory:NVMM), format=NV12"))
        else:
            # cale de rezerva pe desktop: encoder software, deci memorie de sistem
            encoder.set_property('bitrate', max(1, bitrate // 1000))
            encoder.set_property('speed-preset', 'ultrafast')
            caps_out.set_property('caps', Gst.Caps.from_string(
                "video/x-raw, format=I420"))

        parser = make_element("h264parse", "parser")
        muxer = make_element("qtmux", "muxer")
        sink = make_element("filesink", "filesink")
        sink.set_property('location', output_video)
        sink.set_property('sync', False)
        sink.set_property('async', False)
        return [convert, caps_out, encoder, parser, muxer, sink]

    if not args.no_display:
        branches.append(display_branch())
    if args.record:
        branches.append(record_branch())

    nvosd = None
    if branches:
        nvosd = make_element("nvdsosd", "osd")

    # Firele de executie. Un queue inainte de conversia pentru probe si unul dupa
    # ea taie lantul in trei bucati care merg in paralel, pe cadre diferite:
    #
    #   [camera -> streammux -> detector -> tracker] | [probe] | [osd -> iesire]
    #
    # Probe-ul ruleaza pe firul care impinge in queue-ul de dupa caps_rgba, deci
    # nici detectorul dinaintea lui, nici iesirea de dupa nu il mai asteapta.
    queue_pre = make_queue("coada-detectie", args.queue_size)
    queue_probe = make_queue("coada-probe", args.queue_size)
    queue_post = make_queue("coada-iesire", args.queue_size)

    chain = [streammux, pgie, tracker, queue_probe, vidconv_osd, caps_rgba,
             queue_post]
    if nvosd is not None:
        chain.append(nvosd)
    else:
        fake = make_element("fakesink", "iesire-nula")
        fake.set_property('sync', False)
        fake.set_property('async', False)
        fake.set_property('enable-last-sample', False)
        chain.append(fake)

    tee = None
    if len(branches) > 1:
        tee = make_element("tee", "iesire-tee")
        chain.append(tee)
    elif branches:
        chain.extend(branches[0])

    for element in camera_chain + [vidconv_in, caps_in, queue_pre] + chain:
        pipeline.add(element)
    if tee is not None:
        for branch in branches:
            for element in branch:
                pipeline.add(element)

    print("Legare elemente pipeline"
          + (" (fara fereastra)..." if args.no_display else "..."))

    # Capul de lant: camera -> ... -> conversie -> caps -> queue -> streammux.
    for upstream, downstream in zip(camera_chain, camera_chain[1:]):
        if not upstream.link(downstream):
            raise RuntimeError(
                f"Nu pot lega {upstream.get_name()} de {downstream.get_name()}. "
                f"Cel mai probabil camera nu poate da "
                f"{args.capture_width}x{args.capture_height}@{args.fps}"
                f"{' MJPEG' if args.mjpeg else ''}; vezi ce stie cu "
                f"v4l2-ctl -d {args.device} --list-formats-ext."
            )
    if not camera_chain[-1].link(vidconv_in):
        raise RuntimeError("Nu pot lega camera de conversia de intrare.")
    vidconv_in.link(caps_in)
    caps_in.link(queue_pre)
    queue_pre.get_static_pad("src").link(streammux.get_request_pad("sink_0"))

    for upstream, downstream in zip(chain, chain[1:]):
        if not upstream.link(downstream):
            raise RuntimeError(
                f"Nu pot lega {upstream.get_name()} de {downstream.get_name()} "
                f"(caps incompatibile?)."
            )

    # Ramurile de dupa tee. Fiecare incepe cu queue-ul ei: fara el, ramura cea mai
    # lenta (encoderul) ar tine pe loc si fereastra, pentru ca tee-ul impinge in
    # toate ramurile pe acelasi fir.
    if tee is not None:
        for index, branch in enumerate(branches):
            queue = make_queue(f"coada-ramura-{index}", args.queue_size)
            pipeline.add(queue)
            pad = tee.get_request_pad("src_%u")
            if pad.link(queue.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
                raise RuntimeError(f"Nu pot lega tee-ul la ramura {index}.")
            for upstream, downstream in zip([queue] + branch, branch):
                if not upstream.link(downstream):
                    raise RuntimeError(
                        f"Nu pot lega {upstream.get_name()} de "
                        f"{downstream.get_name()} (ramura {index})."
                    )

    # Filtrul de casete se pune pe IESIREA detectorului, deci inainte de tracker.
    if PRETRACK_FILTER:
        pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER,
                                             detection_filter_probe, 0)
        print(f"  filtru inainte de tracker: casete cu ambele laturi sub "
              f"{PRETRACK_MIN_SIZE:.0f} px (increderea NU se filtreaza aici)")

    caps_rgba.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, media_probe, 0)
    return pipeline

# ============================================================
# RAPORT FINAL
# ============================================================

def percentile(ordered, fraction):
    """Percentila dintr-o lista DEJA sortata (apelantul sorteaza o singura data)."""
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def distribution(values, digits=1):
    """min / p10 / median / p90 / max, ca sa se vada daca un prag e realist."""
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "min": r(ordered[0], digits),
        "p10": r(percentile(ordered, 0.10), digits),
        "median": r(percentile(ordered, 0.50), digits),
        "p90": r(percentile(ordered, 0.90), digits),
        "max": r(ordered[-1], digits),
        "n": len(ordered),
    }


def tracks_above_size():
    """Cate track-uri ating fiecare prag de marime, in jurul lui ENROLL_MIN_FACE.

    Cand "marime" e blocajul principal, distributia pe fete nu ajunge: 25% din
    fete peste prag pot fi toate ale aceluiasi om. Ce conteaza e cati OAMENI
    diferiti au avut macar un cadru destul de mare, si asta se vede doar numarand
    track-uri. Masurat pe videotest3.mp4: la prag 50 -> 3 track-uri, la 42 -> 9,
    la 40 -> 13. Aceeasi filmare, acelasi model, alt prag.
    """
    reached = [report.get("marime_maxima", 0) for report in track_reports.values()]
    if not reached:
        return {}
    steps = sorted({max(MIN_FACE_SIZE, ENROLL_MIN_FACE + delta)
                    for delta in (-10, -8, -5, -2, 0, 5, 10)})
    return {str(step): sum(1 for size in reached if size >= step) for step in steps}


def write_summary(run):
    args, output_dir = run.args, run.output_dir
    elapsed = time.time() - run.started
    durations = sorted(run.probe_ms)
    identities = {}
    for track_id, report in track_reports.items():
        label = report["eticheta"] or "fara_decizie"
        identities.setdefault(label, []).append(track_id)

    source = args.source_info
    summary = {
        "camera": run.video_path,
        "folder": output_dir,
        "rulare": run.run_index,
        # None = rularea a mers pana la capatul fisierului.
        "eroare": run.failure,
        # Rezolutia de lucru nu mai e fixa, deci fara ea nu se pot citi nici
        # distributiile de mai jos: pragurile sunt in pixeli la rezolutia asta.
        "sursa": {
            "rezolutie_captura": f"{source['width']}x{source['height']}",
            "fps_cerut": args.fps,
            "mjpeg": args.mjpeg,
            "rotatie_aplicata": args.rotation,
            "rezolutie_lucru": f"{args.width}x{args.height}",
            "tip": source["citit_cu"],
            "fereastra": not args.no_display,
            "inregistrare": args.record_path,
            "memorie_libera_mb": {"pornire": args.free_memory_mb,
                                  "final": available_memory_mb()[0]},
        },
        "cadre": run.frames,
        "fete_procesate": run.faces_seen,
        "recunoasteri": run.recognitions,
        "durata_rulare_s": r(elapsed, 1),
        # Cat s-a mers, si cat din asta a stat pipeline-ul dupa probe. Cu
        # queue-uri, probe-ul ruleaza in paralel cu restul, deci "probe_din_total"
        # poate trece bine peste 100%: inseamna doar ca probe-ul e acum partea
        # lunga si ca de el trebuie sa te legi mai departe.
        "viteza": {
            "fps": r(run.frames / elapsed, 2) if elapsed > 0 else 0.0,
            "ms_pe_cadru": r(1000.0 * elapsed / run.frames, 2) if run.frames else 0.0,
            "probe_din_total_%": (r(100.0 * sum(run.probe_ms) / (1000.0 * elapsed), 1)
                                  if elapsed > 0 else 0.0),
            "queue_size": args.queue_size,
        },
        # Cate casete a scos detectorul si cate au ajuns la tracker. Raportul
        # dintre ele arata cat de mult ajuta filtrul de dinaintea tracker-ului --
        # pe un cadru larg cu multa lume e diferenta intre 42 si 4 tinte urmarite.
        "detectii": {
            "brute": run.detections,
            "la_tracker": run.detections_kept,
            "aruncate": run.detections - run.detections_kept,
            "filtru_activ": PRETRACK_FILTER,
            "prag_px": r(PRETRACK_MIN_SIZE, 1),
            "filtreaza_increderea": False,   # intentionat: vezi detection_filter_probe
        },
        # Cu ce a fost urmarit. Fara asta nu se poate compara o rulare cu alta
        # cand se umbla la parametrii tracker-ului -- si ei sunt cei care decid
        # daca un om care intoarce capul isi pastreaza id-ul.
        "tracker": {
            "config_sursa": os.path.basename(args.tracker_source),
            "config_folosit": os.path.basename(args.tracker_config),
            "ajustari": args.tracker_overrides,
        },
        # Unde se duc milisecundele din probe, mediu pe cadru. Aici se citeste ce
        # merita optimizat la runda urmatoare, fara sa mai fie nevoie de profiler.
        "timp_etape_ms": {name: r(1000.0 * total / run.frames, 2)
                          for name, total in run.stage_s.most_common()} if run.frames else {},
        "baza_de_date": {
            "pornire": run.database_in,
            "scrisa_in": run.database_out,
            "continuata": run.database_in == run.database_out,
            "identitati_initiale": face_database.source_count,
            "identitati_finale": len(face_database),
            "inrolari": run.enrollments,
        },
        # De cate ori a picat fiecare conditie de inrolare. Daca rularea se
        # termina cu zero identitati, aici scrie de ce: pragul cu numarul cel
        # mai mare e cel care blocheaza.
        "blocaje_inrolare": {
            "incercari": run.enroll_attempts,
            "cauze": dict(run.enroll_blockers.most_common()),
            # Cate track-uri (adica oameni, nu cadre) ar ajunge la ENROLL_MIN_FACE
            # daca pragul ar fi altul. Se citeste cand "marime" e cauza de sus.
            "track_uri_peste_prag_marime": tracks_above_size(),
        },
        # Ce a facut fiecare verificare cu identitatea AFISATA, si cate casete au
        # fost desenate din memorie. Aici se citeste efectul reparatiei din antet:
        #   "tinut"  -- de cate ori o verificare proasta ar fi sters, in _10, un
        #               nume deja aflat (si ar fi lasat pe ecran "face <id>");
        #   "casete_tinute" -- cadre in care caseta a purtat identitatea desi fata
        #               n-a trecut de porti; in _10 fiecare din ele era o caseta
        #               cu textul implicit al detectorului;
        #   "pierdut"/"schimbat" -- cazurile in care chiar s-a schimbat omul sub
        #               acelasi id, adica exact ce NU trebuie inabusit de
        #               histerezis. Daca astea sunt zero si "tinut" e mare, se
        #               poate creste IDENTITY_FLIP_CHECKS; invers, se scade.
        "continuitate": {
            "verificari": dict(run.identity_events.most_common()),
            "casete_tinute": run.held_boxes,
            "casete_vechi": run.stale_boxes,
            "casete_anonime": run.blank_boxes,
            "identitati_preluate": run.carried,
            "praguri": {
                "IDENTITY_FLIP_CHECKS": IDENTITY_FLIP_CHECKS,
                "IDENTITY_STALE_FRAMES": IDENTITY_STALE_FRAMES,
                "TRACKED_MIN_CONFIDENCE": TRACKED_MIN_CONFIDENCE,
                "CARRY_IDENTITY": CARRY_IDENTITY,
                "REID_CARRY_GAP_FRAMES": REID_CARRY_GAP_FRAMES,
                "REID_CARRY_MIN_IOU": REID_CARRY_MIN_IOU,
            },
        },
        # Cine a fost in cadru si cand. "reveniri" sunt exact evenimentele
        # anuntate cu "[REVENIT] ... s-a intors dupa N s"; "istoric" le are pe
        # toate, in ordine, deci din el se poate reconstrui cine a fost prezent in
        # orice moment al sesiunii.
        "prezenta": {
            "miscari": dict(PRESENCE.counts.most_common()) if PRESENCE else {},
            "reveniri": PRESENCE.returns() if PRESENCE else [],
            "in_cadru_la_final": sorted(PRESENCE.present) if PRESENCE else [],
            "istoric": PRESENCE.events if PRESENCE else [],
            "praguri": {
                "RETURN_WINDOW_S": RETURN_WINDOW_S,
                "ABSENCE_GRACE_S": ABSENCE_GRACE_S,
            },
        },
        # Verificari programate care nu s-au facut deloc. Daca aici sunt numere
        # mari si "blocaje_inrolare" e gol, problema nu e la pragurile de
        # inrolare: fetele nici n-au ajuns la modelul de recunoastere.
        "verificari_sarite": {
            "cauze": dict(run.checks_skipped.most_common()),
            "calitate_la_termen": distribution(run.deadline_qualities, 3),
        },
        "gpu": dict(GPU_STATS),
        "apeluri_economisite": {
            "landmark": GPU_STATS["landmark_faces"] - GPU_STATS["landmark_calls"],
            "recunoastere": GPU_STATS["recognition_faces"] - GPU_STATS["recognition_calls"],
            # Fete vazute pe care NU s-a mai cerut modelul de landmark-uri:
            # track-ul lor avea deja identitate si nu dadea proba in cadrul acela.
            # Raportat la "fete_procesate", asta arata cat a lucrat racirea.
            "fete_doar_urmarite": run.faces_seen - GPU_STATS["landmark_faces"],
        },
        "timp_probe_ms": {
            "mediu": r(sum(durations) / len(durations), 2) if durations else 0.0,
            "p50": r(percentile(durations, 0.50), 2),
            "p95": r(percentile(durations, 0.95), 2),
            "maxim": r(durations[-1], 2) if durations else 0.0,
        },
        "track_uri": track_reports,
        "identitati": {label: sorted(tracks) for label, tracks in identities.items()},
        # Ce s-a masurat efectiv pe fetele din clip. Pragurile sunt in pixeli la
        # rezolutia de lucru, deci fara distributiile astea nu se poate spune
        # daca un prag e cu putin prea sus sau complet nepotrivit filmarii.
        "distributii": {
            "marime_px": distribution(run.face_sizes, 0),
            "blur": distribution(run.face_blurs, 1),
            "calitate": distribution(run.face_qualities, 3),
            # Cea mai utila dintre toate cand baza iese proasta: daca mediana e
            # langa zero, filmarea e din profil si degeaba se relaxeaza restul.
            "frontalitate": distribution(run.face_frontalities, 3),
            # Casetele taiate de poarta de marime. Citita impreuna cu "marime_px"
            # de mai sus, spune cat s-ar castiga coborand MIN_FACE_SIZE: daca p90
            # de aici e mult sub prag, nu se castiga nimic, oamenii sunt pur si
            # simplu prea departe de camera.
            "marime_respinse_px": distribution(run.rejected_sizes, 0),
        },
        "praguri": {
            "rezolutie_lucru": f"{args.width}x{args.height}",
            "MIN_CONFIDENCE": MIN_CONFIDENCE, "MIN_FACE_SIZE": MIN_FACE_SIZE,
            "MIN_BLUR": MIN_BLUR, "BLUR_REJECT_FACTOR": BLUR_REJECT_FACTOR,
            "VERIFY_THRESHOLD": VERIFY_THRESHOLD,
            "ENROLL_MARGIN": ENROLL_MARGIN, "ENROLL_MAX_SCORE": ENROLL_MAX_SCORE,
            "ENROLL_MIN_FACE": ENROLL_MIN_FACE, "ENROLL_MIN_BLUR": ENROLL_MIN_BLUR,
            "ENROLL_MIN_CHECKS": ENROLL_MIN_CHECKS,
            "ENROLL_MIN_FRONTALITY": ENROLL_MIN_FRONTALITY,
            "QUALITY_GOOD_ENOUGH": QUALITY_GOOD_ENOUGH, "QUALITY_MIN": QUALITY_MIN,
            "ENROLL_MIN_QUALITY": ENROLL_MIN_QUALITY,
            "VERIFY_INTERVAL_FRAMES": VERIFY_INTERVAL_FRAMES,
            "IDENTIFIED_INTERVAL_FRAMES": IDENTIFIED_INTERVAL_FRAMES,
            "LANDMARK_ONLY_WHEN_NEEDED": LANDMARK_ONLY_WHEN_NEEDED,
        },
    }

    with open(run.paths["summary"], "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary

# ============================================================
# MAIN
# ============================================================

def resolve_config(path, what):
    """Configurile se cauta in folderul curent, apoi langa script.

    Pipeline-ul live foloseste cai relative (se ruleaza din /workspace/DeepStream-Yolo),
    dar scriptul asta poate fi pornit de oriunde. Verificam existenta acum, nu cand
    porneste nvinfer: acolo eroarea e mult mai greu de citit.

    path poate fi si o lista de variante, in ordinea preferintei: asa configul de
    tracker cere intai NvDCF_accuracy si cade pe NvDCF_perf daca versiunea
    instalata nu-l are.
    """
    wanted = [path] if isinstance(path, str) else list(path)
    candidates = []
    for option in wanted:
        candidates += [option, os.path.join(SCRIPT_DIR, os.path.basename(option))]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        f"Nu gasesc {what}. Am cautat:\n  " + "\n  ".join(candidates)
    )


# Chei de forma "  nume: valoare   # comentariu", oriunde in fisier. Se inlocuieste
# doar valoarea, deci indentarea, sectiunile si comentariile raman neatinse.
TRACKER_KEY_RE = "^([ \t]*{key}[ \t]*:[ \t]*)([^\\s#]+)(.*)$"


def patch_tracker_config(source, overrides, destination):
    """Scrie o copie a configului de tracker, cu cheile date rescrise.

    Se lucreaza pe linii, cu regex, si NU cu un parser YAML. Doua motive: nu
    depindem de PyYAML, care nu e garantat in containerul DeepStream, si un
    round-trip prin parser ar rescrie tot fisierul -- ar pierde comentariile si ar
    putea reordona chei, iar configurile astea sunt lungi si pline de explicatii
    utile. Aici se schimba exact ce cerem si nimic altceva.

    Intoarce dict cu ce s-a schimbat efectiv. Cheile negasite se raporteaza, nu se
    adauga: daca versiunea instalata numeste altfel un parametru, vrem sa aflam,
    nu sa scriem o cheie pe care tracker-ul o va ignora tacut.
    """
    with open(source, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()

    applied, missing = {}, []
    for key, value in overrides.items():
        if value is None:
            continue
        pattern = re.compile(TRACKER_KEY_RE.format(key=re.escape(key)), re.MULTILINE)
        text, count = pattern.subn(lambda m: f"{m.group(1)}{value}{m.group(3)}", text)
        if count:
            applied[key] = value
        else:
            missing.append(key)

    if missing:
        warn_once("tracker_keys",
                  f"configul de tracker nu are cheile {', '.join(missing)}; "
                  f"raman valorile lui. Verifica numele cu: "
                  f"grep -n 'ShadowTrack\\|minDetectorConfidence' {source}")

    with open(destination, "w", encoding="utf-8") as handle:
        handle.write(text)
    return applied


RUN_INDEX_RE = re.compile(r"^summary_(\d+)\.json$")


def prepare_output_dir(stem, args):
    """Folderul cu numele videoclipului, REFOLOSIT la rulari succesive.

    Asta e ce face posibila secventa "prima rulare inroleaza, a doua recunoaste":
    folderul e memoria rularilor pe videoclipul asta, iar baza de date din el se
    incarca la pornire (vezi resolve_database). Un folder nou pe rulare ar
    insemna ca fiecare rulare porneste iar de la zero identitati.

    Ce se pierde prin refolosire -- suprascrierea rularii anterioare -- se evita
    numerotand fisierele rularii (vezi run_paths), nu folderul.
    """
    base = args.output or os.path.join(SCRIPT_DIR, stem)

    if args.new_dir and os.path.exists(base):
        index = 2
        while os.path.exists(f"{base}_{index}"):
            index += 1
        base = f"{base}_{index}"
        print(f"--new-dir: pornesc curat, in {base}.")

    existed = os.path.isdir(base)
    os.makedirs(base, exist_ok=True)
    return base, existed


def run_paths(output_dir, stem, args):
    """Caile fisierelor rularii curente, numerotate dupa rularile deja existente."""
    used = sorted(int(match.group(1)) for match in
                  (RUN_INDEX_RE.match(name) for name in os.listdir(output_dir)) if match)
    if args.overwrite:
        index = used[-1] if used else 1
    else:
        index = (used[-1] if used else 0) + 1

    tag = f"{index:03d}"
    return index, {
        "video": os.path.join(output_dir, f"{stem}_adnotat_{tag}.mp4"),
        "frames": os.path.join(output_dir, f"frames_{tag}.jsonl"),
        "summary": os.path.join(output_dir, f"summary_{tag}.json"),
    }


def resolve_database(output_dir, args):
    """(de unde se incarca, unde se scrie) baza de date.

    Baza din folderul de iesire e si intrare, si iesire: prima rulare o creeaza
    (pornind, daca exista, de la cea data cu --database), rularile urmatoare o
    gasesc si recunosc ce a inrolat prima. --database ramane doar samanta pentru
    prima rulare si nu e niciodata modificata; --reset-db o ia de la capat.
    """
    folder_db = os.path.join(output_dir, "face_database.json")

    if os.path.isfile(folder_db) and not args.reset_db:
        return folder_db, folder_db

    if os.path.isfile(folder_db) and args.reset_db:
        print(f"--reset-db: ignor {folder_db} si o iau de la capat.")

    seed = args.database if args.database and os.path.isfile(args.database) else None
    return seed, folder_db


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Recunoastere faciala live, pe camera: fereastra pe ecran, "
                    "inrolare automata, baza de date persistenta si anunt in "
                    "consola cand cineva se intoarce in cadru.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    camera = parser.add_argument_group("camera")
    camera.add_argument("--camera", default="auto", choices=["auto", "v4l2", "csi"],
                        help="v4l2 = webcam USB (--device), csi = senzor pe placa "
                             "(nvarguscamerasrc, --sensor-id)")
    camera.add_argument("--device", default="/dev/video0",
                        help="fisierul camerei v4l2")
    camera.add_argument("--sensor-id", type=int, default=None,
                        help="indicele senzorului CSI")
    camera.add_argument("--capture-width", type=int, default=1280,
                        help="latimea CERUTA camerei; trebuie sa fie una pe care "
                             "camera chiar o poate da (v4l2-ctl --list-formats-ext)")
    camera.add_argument("--capture-height", type=int, default=720,
                        help="inaltimea ceruta camerei")
    camera.add_argument("--fps", type=int, default=30,
                        help="rata ceruta camerei, in cadre pe secunda")
    camera.add_argument("--mjpeg", action="store_true",
                        help="cere MJPEG in loc de raw. La aceeasi rezolutie iese "
                             "mult mai putin trafic pe USB, deci rate mai mari; "
                             "in schimb se plateste o decodare pe CPU")
    parser.add_argument("--output", default=None,
                        help="folderul de iesire (implicit: camera_<device>); "
                             "daca exista, e refolosit impreuna cu baza lui de date")
    parser.add_argument("--database", default=FACE_DATABASE_PATH,
                        help="baza de date samanta, folosita doar cand folderul de "
                             "iesire nu are inca una; nu e modificata")
    parser.add_argument("--pgie-config", default=YOLO_CONFIG_PATH,
                        help="configul nvinfer pentru detectorul de fete")
    parser.add_argument("--tracker-config", default=TRACKER_CONFIG_PATH,
                        help="configul low-level pentru nvtracker; implicit se "
                             "incearca NvDCF_accuracy, apoi NvDCF_perf")
    parser.add_argument("--shadow-track-age", type=int, default=None,
                        help=f"cate cadre traieste un track fara nicio detectie "
                             f"(maxShadowTrackAge). Asta decide daca cineva care "
                             f"intoarce capul isi pastreaza id-ul; masurat pe "
                             f"videotest3.mp4, golurile ajung la 233 de cadre "
                             f"(implicit {TRACKER_OVERRIDES['maxShadowTrackAge']})")
    parser.add_argument("--min-detector-confidence", type=float, default=None,
                        help=f"increderea de la care nvtracker accepta o caseta. "
                             f"Se tine jos intentionat: casetele slabe sunt cele "
                             f"care leaga track-ul peste ocluzii, iar calitatea se "
                             f"filtreaza oricum mai tarziu "
                             f"(implicit {TRACKER_OVERRIDES['minDetectorConfidence']})")
    parser.add_argument("--no-tracker-patch", action="store_true",
                        help="foloseste configul de tracker exact cum e pe disc, "
                             "fara ajustarile de mai sus")
    parser.add_argument("--landmark-engine", default=PFLD_MODEL_PATH)
    parser.add_argument("--recognition-engine", default=RECOGNITION_MODEL_PATH)
    parser.add_argument("--width", type=int, default=None,
                        help="latimea de LUCRU, daca trebuie sa difere de cea de "
                             "captura. Atentie: pragurile de marime sunt in pixeli "
                             "la rezolutia asta, deci micsorarea micsoreaza si fetele")
    parser.add_argument("--height", type=int, default=None,
                        help="inaltimea de lucru; implicit cea de captura")
    parser.add_argument("--rotate", default="0", choices=["0", "90", "180", "270"],
                        help="rotatia aplicata imaginii camerei, in grade, in sensul "
                             "acelor de ceas (pentru camerele montate pe o parte)")
    parser.add_argument("--bitrate", type=int, default=0,
                        help="bitrate-ul inregistrarii, in biti/s; "
                             "0 = calculat din rezolutie (~8 Mbit/s la 1080p)")
    parser.add_argument("--record", nargs="?", const=True, default=False,
                        help="inregistreaza si un mp4, in paralel cu fereastra; "
                             "fara valoare, in folderul rularii")
    parser.add_argument("--no-display", action="store_true",
                        help="nu deschide fereastra. Fara --record ramane doar "
                             "consola: recunoasterea, anunturile si baza de date "
                             "ies neschimbate, dar de pe placa dispar encoderul si "
                             "sink-ul de afisare")
    parser.add_argument("--no-frame-log", action="store_true",
                        help="nu scrie frames_*.jsonl. O sesiune live de o ora "
                             "inseamna ~100 000 de randuri, si de obicei nu sunt "
                             "citite; summary.json ramane oricum")
    parser.add_argument("--save-every", type=float, default=None,
                        help=f"la cate secunde se scrie baza de date pe disc in "
                             f"timpul rularii; 0 = doar la iesire "
                             f"(implicit {DATABASE_SAVE_INTERVAL_S})")
    parser.add_argument("--queue-size", type=int, default=4,
                        help="cate cadre pot astepta in fiecare queue; asta decide "
                             "cat se suprapun decodarea/detectia, probe-ul si "
                             "encodarea. Fiecare cadru in asteptare e o suprafata "
                             "NVMM (~8 MB la 1080p RGBA), deci mai mult nu e "
                             "gratis. 1 = practic fara suprapunere")
    parser.add_argument("--no-pretrack-filter", action="store_true",
                        help="trimite la tracker toate casetele detectorului, nu "
                             "doar pe cele care pot trece de porti. Mult mai lent "
                             "pe cadre largi; de folosit doar ca sa se compare")
    parser.add_argument("--log-respinse", action="store_true",
                        help="scrie in log fiecare caseta respinsa, nu doar cate "
                             "au fost pe fiecare poarta (logul creste de ~4 ori)")
    parser.add_argument("--no-enroll", action="store_true",
                        help="nu adauga identitati noi in baza de date")
    parser.add_argument("--no-landmarks", action="store_true",
                        help="deseneaza doar cele 5 puncte ArcFace, nu tot setul")
    parser.add_argument("--new-dir", action="store_true",
                        help="porneste intr-un folder nou (fara identitatile "
                             "inrolate la rularile anterioare)")
    parser.add_argument("--overwrite", action="store_true",
                        help="scrie peste fisierele ultimei rulari din folder, in "
                             "loc sa le numeroteze mai departe")
    parser.add_argument("--reset-db", action="store_true",
                        help="ignora baza de date din folder si o ia de la samanta "
                             "--database (cu --database '' porneste complet goala)")

    # Pragurile care decid ce ajunge in baza. Sunt in linia de comanda pentru ca
    # depind de filmare (rezolutie, codec, distanta fata de camera), iar valorile
    # masurate efectiv sunt in summary.json, la "distributii": se citesc de acolo
    # si se dau aici, fara sa fie nevoie de modificat scriptul.
    # Implicit None peste tot, ca sa ramana in vigoare valorile din script pentru
    # ce nu e dat explicit. Toate marimile sunt in pixeli la rezolutia de lucru.
    tuning = parser.add_argument_group("praguri (vezi 'distributii' din summary.json)")
    tuning.add_argument("--min-face", type=int, default=None,
                        help=f"latura minima a bbox-ului ca fata sa fie procesata "
                             f"(implicit {MIN_FACE_SIZE})")
    tuning.add_argument("--verify-interval", type=int, default=None,
                        help=f"la cate cadre cel mult se re-verifica un track; mai "
                             f"des inseamna mai multe sanse ca un track scurt sa "
                             f"apuce ENROLL_MIN_CHECKS (implicit {VERIFY_INTERVAL_FRAMES})")
    tuning.add_argument("--identified-interval", type=int, default=None,
                        help=f"racirea: la cate cadre se re-verifica un track care "
                             f"ARE deja identitate. Intre verificari nu i se mai cer "
                             f"landmark-uri, deci asta e pargia principala de viteza "
                             f"pe un clip cu oameni deja in baza. Mai mare = mai "
                             f"rapid, dar un id schimbat de tracker se prinde mai "
                             f"tarziu (implicit {IDENTIFIED_INTERVAL_FRAMES})")
    tuning.add_argument("--landmark-all", action="store_true",
                        help="cere landmark-uri pe toate fetele, in fiecare cadru, "
                             "ca inainte de racire; de folosit doar ca sa se compare")
    tuning.add_argument("--min-blur", type=float, default=None,
                        help=f"varianta Laplacianului considerata 'clar' "
                             f"(implicit {MIN_BLUR})")
    tuning.add_argument("--quality-min", type=float, default=None,
                        help=f"calitatea minima ca sa merite rulat modelul de "
                             f"recunoastere (implicit {QUALITY_MIN})")
    tuning.add_argument("--verify-threshold", type=float, default=None,
                        help=f"scorul cosinus de la care o fata e recunoscuta "
                             f"(implicit {VERIFY_THRESHOLD})")
    tuning.add_argument("--enroll-min-checks", type=int, default=None,
                        help=f"cate verificari 'necunoscut' la rand cere o inrolare "
                             f"(implicit {ENROLL_MIN_CHECKS})")
    tuning.add_argument("--enroll-min-face", type=int, default=None,
                        help=f"implicit {ENROLL_MIN_FACE}")
    tuning.add_argument("--enroll-min-blur", type=float, default=None)
    tuning.add_argument("--enroll-min-quality", type=float, default=None)
    tuning.add_argument("--enroll-min-frontality", type=float, default=None,
                        help=f"cat de din fata trebuie privita persoana ca poza ei "
                             f"sa ajunga prototip, min(yaw, pitch) in [0,1] "
                             f"(implicit {ENROLL_MIN_FRONTALITY})")

    # Cat de tare se agata eticheta de om. Grupul asta e reparatia din antet,
    # scoasa in linia de comanda ca sa se poata compara direct cu _10:
    # --identity-flip-checks 1 --tracked-min-confidence 0.5 --no-carry-identity
    # da inapoi comportamentul vechi, cu aceleasi praguri de recunoastere.
    cont = parser.add_argument_group("continuitatea identitatii")
    cont.add_argument("--identity-flip-checks", type=int, default=None,
                      help=f"cate verificari LA RAND trebuie sa contrazica o "
                           f"identitate deja confirmata ca ea sa cada. 1 = ca in "
                           f"_10, adica prima fereastra proasta sterge numele "
                           f"(implicit {IDENTITY_FLIP_CHECKS})")
    cont.add_argument("--identity-stale", type=int, default=None,
                      help=f"dupa cate cadre fara confirmare identitatea afisata "
                           f"se deseneaza galben si cu '?'; nu se sterge "
                           f"(implicit {IDENTITY_STALE_FRAMES})")
    cont.add_argument("--tracked-min-confidence", type=float, default=None,
                      help=f"increderea minima pentru caseta unui track care are "
                           f"deja stare; mai joasa decat --min-confidence pentru "
                           f"ca tracker-ul a legat-o deja de un istoric "
                           f"(implicit {TRACKED_MIN_CONFIDENCE})")
    cont.add_argument("--no-carry-identity", action="store_true",
                      help="un track nou nu mai preia identitatea unuia rupt in "
                           "acelasi loc; fiecare id nou porneste de la zero")
    cont.add_argument("--carry-iou", type=float, default=None,
                      help=f"suprapunerea minima ceruta pentru preluare "
                           f"(implicit {REID_CARRY_MIN_IOU})")
    cont.add_argument("--carry-gap", type=int, default=None,
                      help=f"de cate cadre poate lipsi track-ul vechi ca sa mai "
                           f"poata da identitatea (implicit {REID_CARRY_GAP_FRAMES})")
    cont.add_argument("--hide-respinse", action="store_true",
                      help="scoate de tot din desen casetele fara stare, in loc "
                           "sa le deseneze subtire si fara text")

    presence = parser.add_argument_group("prezenta si reintoarcerea")
    presence.add_argument("--return-window", type=float, default=None,
                          help=f"in cate secunde de la plecare o reintrare se "
                               f"anunta cu 's-a intors'; peste, e o aparitie "
                               f"obisnuita (implicit {RETURN_WINDOW_S})")
    presence.add_argument("--absence-grace", type=float, default=None,
                          help=f"cate secunde fara nicio caseta care sa-i poarte "
                               f"numele inseamna ca omul chiar a plecat; sub prag "
                               f"e o palpaire (implicit {ABSENCE_GRACE_S})")
    presence.add_argument("--quiet-presence", action="store_true",
                          help="nu scrie [PLECAT]/[PREZENT], doar reintoarcerile")
    return parser.parse_args(argv)


def apply_thresholds(args):
    """Muta pragurile date in linia de comanda peste constantele modulului."""
    global MIN_FACE_SIZE, MIN_BLUR, QUALITY_MIN, VERIFY_THRESHOLD
    global ENROLL_MIN_CHECKS, ENROLL_MIN_FACE, ENROLL_MIN_BLUR
    global ENROLL_MIN_QUALITY, ENROLL_MAX_SCORE, VERIFY_INTERVAL_FRAMES
    global ENROLL_MIN_FRONTALITY, PRETRACK_MIN_SIZE, IDENTIFIED_INTERVAL_FRAMES
    global IDENTITY_FLIP_CHECKS, IDENTITY_STALE_FRAMES, TRACKED_MIN_CONFIDENCE
    global CARRY_IDENTITY, REID_CARRY_MIN_IOU, REID_CARRY_GAP_FRAMES
    global HIDE_REJECTED_BOXES
    global RETURN_WINDOW_S, ABSENCE_GRACE_S, ANNOUNCE_DEPARTURE
    global DATABASE_SAVE_INTERVAL_S

    if args.return_window is not None:
        RETURN_WINDOW_S = args.return_window
    if args.absence_grace is not None:
        ABSENCE_GRACE_S = args.absence_grace
    if args.save_every is not None:
        DATABASE_SAVE_INTERVAL_S = max(0.0, args.save_every)
    ANNOUNCE_DEPARTURE = not args.quiet_presence

    if args.identity_flip_checks is not None:
        IDENTITY_FLIP_CHECKS = max(1, args.identity_flip_checks)
    if args.identity_stale is not None:
        IDENTITY_STALE_FRAMES = args.identity_stale
    if args.tracked_min_confidence is not None:
        TRACKED_MIN_CONFIDENCE = args.tracked_min_confidence
    if args.carry_iou is not None:
        REID_CARRY_MIN_IOU = args.carry_iou
    if args.carry_gap is not None:
        REID_CARRY_GAP_FRAMES = args.carry_gap
    CARRY_IDENTITY = not args.no_carry_identity
    HIDE_REJECTED_BOXES = args.hide_respinse

    if args.identified_interval is not None:
        IDENTIFIED_INTERVAL_FRAMES = args.identified_interval
    if args.min_face is not None:
        MIN_FACE_SIZE = args.min_face
    if args.verify_interval is not None:
        VERIFY_INTERVAL_FRAMES = args.verify_interval
    if args.min_blur is not None:
        MIN_BLUR = args.min_blur
    if args.quality_min is not None:
        QUALITY_MIN = args.quality_min
    if args.verify_threshold is not None:
        VERIFY_THRESHOLD = args.verify_threshold
    if args.enroll_min_checks is not None:
        ENROLL_MIN_CHECKS = args.enroll_min_checks
    if args.enroll_min_face is not None:
        ENROLL_MIN_FACE = args.enroll_min_face
    if args.enroll_min_blur is not None:
        ENROLL_MIN_BLUR = args.enroll_min_blur
    if args.enroll_min_quality is not None:
        ENROLL_MIN_QUALITY = args.enroll_min_quality
    if args.enroll_min_frontality is not None:
        ENROLL_MIN_FRONTALITY = args.enroll_min_frontality
    # derivate, deci trebuie recalculate dupa pragurile de mai sus
    ENROLL_MAX_SCORE = VERIFY_THRESHOLD - ENROLL_MARGIN
    PRETRACK_MIN_SIZE = MIN_FACE_SIZE * PRETRACK_SIZE_FACTOR


# Pragurile in pixeli NU se scaleaza cu rezolutia de lucru, desi la un moment dat
# faceam asta. Motivul: ce conteaza pentru recunoastere e cati pixeli are efectiv
# fata, nu ce fractiune din cadru ocupa. Aceeasi persoana filmata 4K si redusa la
# 1080p chiar are jumatate din detaliu, iar un prag care se scaleaza odata cu
# cadrul ar declara cele doua cazuri echivalente -- si ar anula tocmai castigul
# pentru care ai creste --max-side.


def available_memory_mb():
    """(liber, total) in MB, din /proc/meminfo. None acolo unde nu exista.

    Pe Jetson memoria e unificata: pool-urile NVMM, workspace-ul cuDNN si
    procesele obisnuite trag din acelasi loc, deci numarul asta e chiar cel care
    decide daca rularea trece sau pica cu ENOMEM.
    """
    try:
        values = {}
        with open("/proc/meminfo", "r") as handle:
            for line in handle:
                name, _, rest = line.partition(":")
                if name in ("MemTotal", "MemAvailable"):
                    values[name] = int(rest.split()[0]) // 1024
        return values.get("MemAvailable"), values.get("MemTotal")
    except Exception:
        return None, None


MEMORY_ERROR_MARKERS = (
    "queue input batch",            # nvinfer n-a putut trimite batch-ul
    "NVDSINFER_TENSORRT_ERROR",     # de obicei CUDNN_STATUS_INTERNAL_ERROR sub el
    "bufferpool",                   # "failed to activate bufferpool" la nvvideoconvert
    "NvMapMemAllocInternalTagged",  # alocatorul NVMM, "error 12" = ENOMEM
    "Error in allocating buffer",
    "gst-resource-error-quark",
)


def explain_inference_error(text, args):
    """Traduce un esec de alocare in ce se poate face concret.

    Niciunul dintre mesajele de mai sus nu spune nimic singur, iar cauza adevarata
    e aceeasi pentru toate: memoria unificata s-a terminat. "failed to activate
    bufferpool" la conversie-osd si "CUDNN_STATUS_INTERNAL_ERROR" in detector sunt
    acelasi ENOMEM, prins in doua locuri diferite.
    """
    if not any(marker in text for marker in MEMORY_ERROR_MARKERS):
        return

    free, total = available_memory_mb()
    source = args.source_info
    print("\n  Detectorul a picat la inferenta. Aproape sigur memoria: "
          "CUDNN_STATUS_INTERNAL_ERROR")
    print("  inseamna de obicei ca nu mai e loc de workspace, nu ca modelul e gresit.")
    if free is not None:
        print(f"  Liber acum: {free} MB din {total} MB.")
    print(f"  Camera livreaza {source['width']}x{source['height']}, iar peste asta "
          f"vin detectorul,\n  cele doua engine-uri, sink-ul de afisare si "
          f"(daca s-a cerut) encoderul.")
    print("  De incercat, in ordine:")
    print("  1. --no-display (fara --record): scoate si sink-ul de ecran, si NVENC. "
          "Recunoasterea,\n     anunturile si baza de date ies neschimbate.")
    print("  2. --capture-width 1280 --capture-height 720: mai putini pixeli peste "
          "tot.\n     Atentie: si fetele se micsoreaza, deci si pragurile de marime "
          "se simt.")
    print("  3. Elibereaza memorie: opreste interfata grafica, alte procese CUDA, "
          "si urmareste\n     cu tegrastats cat ramane liber in timpul rularii.")


def main():
    global AUTO_ENROLL, DRAW_ALL_LANDMARKS, DRAW_OVERLAY, RUN, PRESENCE
    global PRETRACK_FILTER, LOG_REJECTED, LANDMARK_ONLY_WHEN_NEEDED

    args = parse_args()
    AUTO_ENROLL = not args.no_enroll
    LANDMARK_ONLY_WHEN_NEEDED = not args.landmark_all
    # Desenul are sens daca il vede cineva: fereastra sau inregistrare. Fara
    # niciuna, nvdsosd nici nu mai intra in pipeline.
    DRAW_OVERLAY = (not args.no_display) or bool(args.record)
    DRAW_ALL_LANDMARKS = not args.no_landmarks and DRAW_OVERLAY
    PRETRACK_FILTER = not args.no_pretrack_filter
    LOG_REJECTED = args.log_respinse
    args.queue_size = max(1, args.queue_size)
    args.fps = max(1, args.fps)
    apply_thresholds(args)

    check_pyds_api()

    args.pgie_config = resolve_config(args.pgie_config, "configul nvinfer (detectorul de fete)")
    args.tracker_config = resolve_config(args.tracker_config, "configul nvtracker")

    # Gst.init inainte de orice interogare de fabrici: si camera_kind, si
    # rotation_supported cauta elemente.
    Gst.init(None)
    args.camera_kind = camera_kind(args)
    if args.camera_kind == "v4l2" and not os.path.exists(args.device):
        print(f"[AVERTISMENT] {args.device} nu exista acum. Camerele vazute de "
              f"sistem:  ls -l /dev/video*")

    args.rotation = int(args.rotate)
    if args.rotation and not rotation_supported():
        print(f"[AVERTISMENT] nvvideoconvert nu are flip-method pe platforma asta; "
              f"nu pot roti imaginea cu {args.rotation} grade. Fetele raman culcate, "
              f"deci detectorul le va rata.")
        args.rotation = 0
    args.width, args.height = working_resolution(args)
    args.source_info = {"width": args.capture_width, "height": args.capture_height,
                        "rotation": args.rotation, "citit_cu": args.camera_kind}

    free, total = available_memory_mb()
    args.free_memory_mb = free
    orientation = " (verticala)" if args.height > args.width else ""
    print(f"Camera: {args.camera_kind}, captura "
          f"{args.capture_width}x{args.capture_height}@{args.fps}"
          + (", MJPEG" if args.mjpeg else "")
          + (f"; memorie libera {free} MB din {total} MB" if free is not None else ""))
    print(f"Lucru:  {args.width}x{args.height}{orientation}"
          + (f", rotesc cu {args.rotation} grade" if args.rotation else ""))
    if args.width < args.capture_width:
        factor = args.capture_width / float(args.width)
        print(f"        imaginea e micsorata de {factor:.1f} ori, deci si fetele: "
              f"pragurile de marime sunt in pixeli la rezolutia de lucru")

    # Folderul se numeste dupa camera, nu dupa un fisier: asa baza de date a
    # camerei asteia se regaseste la rularea urmatoare, si oamenii inrolati ieri
    # sunt recunoscuti azi.
    stem = ("camera_" + (f"sensor{args.sensor_id or 0}" if args.camera_kind == "csi"
                         else os.path.basename(args.device.rstrip("/")) or "video0"))
    output_dir, reused = prepare_output_dir(stem, args)
    run_index, paths = run_paths(output_dir, stem, args)
    database_in, database_out = resolve_database(output_dir, args)
    # --record fara valoare scrie in folderul rularii; --record cale scrie acolo.
    if not args.record:
        output_video = None
    elif args.record is True:
        output_video = paths["video"]
    else:
        output_video = args.record
    args.record_path = output_video

    # Configul de tracker se ajusteaza acum, cand stim unde scriem: copia patchuita
    # ramane in folderul rularii, langa log si summary, deci se vede mai tarziu
    # exact cu ce parametri s-a urmarit.
    args.tracker_overrides = {}
    args.tracker_source = args.tracker_config
    if not args.no_tracker_patch:
        overrides = dict(TRACKER_OVERRIDES)
        if args.shadow_track_age is not None:
            overrides["maxShadowTrackAge"] = args.shadow_track_age
        if args.min_detector_confidence is not None:
            overrides["minDetectorConfidence"] = args.min_detector_confidence
        patched = os.path.join(output_dir, f"tracker_{run_index:03d}.yml")
        args.tracker_overrides = patch_tracker_config(
            args.tracker_config, overrides, patched
        )
        if args.tracker_overrides:
            valori = ", ".join(f"{k}={v}" for k, v in args.tracker_overrides.items())
            print(f"Tracker: {os.path.basename(args.tracker_config)} -> {valori}")
        args.tracker_config = patched

    print(f"Iesire: {output_dir} ({'refolosit' if reused else 'nou'}, rularea {run_index})")
    print(f"Ecran:  {'nu (--no-display)' if args.no_display else 'fereastra'}"
          + (f", inregistrez in {output_video}" if args.record else ""))
    print(f"Prezenta: anunt 's-a intors' pentru reintrari in cel mult "
          f"{RETURN_WINDOW_S:.0f} s (plecarea se declara dupa "
          f"{ABSENCE_GRACE_S:.1f} s fara nicio caseta cu numele lui)")

    load_models(args.landmark_engine, args.recognition_engine,
                database_in, database_out)
    if database_in == database_out:
        print("Continui baza de date a folderului: ce s-a inrolat la rularile "
              "anterioare se recunoaste acum.")

    RUN = Run(args, output_dir, args.device if args.camera_kind == "v4l2"
              else f"csi:{args.sensor_id or 0}", paths, run_index,
              database_in, database_out)
    PRESENCE = PresenceLog(RETURN_WINDOW_S, ABSENCE_GRACE_S)

    pipeline = build_pipeline(args, output_video)
    loop = GLib.MainLoop()

    # Retinem esecul ca sa iasa si in codul de retur: pana acum o rulare cazuta
    # in prima secunda si una terminata cu bine ieseau amandoua cu 0, deci un
    # script care le porneste in serie n-avea de unde sa stie.
    failure = []

    def bus_call(bus, message, loop):
        message_type = message.type
        if message_type == Gst.MessageType.EOS:
            print("Fluxul s-a incheiat.")
            loop.quit()
        elif message_type == Gst.MessageType.WARNING:
            warning, debug = message.parse_warning()
            print(f"Avertisment GStreamer: {warning}: {debug}")
        elif message_type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            print(f"Eroare GStreamer: {error}: {debug}")
            failure.append(str(error))
            explain_camera_error(f"{error} {debug}", args)
            explain_inference_error(f"{error} {debug}", args)
            loop.quit()
        return True

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Salvarea periodica a bazei. Pe fisier nu era nevoie: rularea se termina in
    # cateva minute si scrierea de la final acopera tot. O sesiune live poate tine
    # ore, iar o cadere de curent, un OOM sau un Ctrl+C repetat (al doilea nu mai
    # trece prin EOS) ar pierde tot ce s-a inrolat intre timp.
    def save_database_tick():
        if len(face_database) != face_database.source_count or RUN.enrollments:
            face_database.save()
        return True     # GLib repeta cat timp se intoarce True

    if DATABASE_SAVE_INTERVAL_S > 0:
        GLib.timeout_add_seconds(int(DATABASE_SAVE_INTERVAL_S), save_database_tick)

    # Al doilea Ctrl+C opreste pe loc: daca EOS nu ajunge (camera blocata intr-un
    # ioctl, pipeline oprit inainte de PLAYING), primul Ctrl+C ar parea ignorat.
    stopping = []

    def sigint_handler(sig, frame):
        if stopping:
            print("\nInca un Ctrl+C: opresc pe loc.")
            loop.quit()
            return
        stopping.append(True)
        # EOS, nu oprire brutala: altfel qtmux nu apuca sa inchida fisierul mp4
        # si inregistrarea ramane necitibila.
        print("\nOprire ceruta; trimit EOS ca sa se inchida corect fisierul "
              "(inca un Ctrl+C opreste imediat)...")
        pipeline.send_event(Gst.Event.new_eos())

    signal.signal(signal.SIGINT, sigint_handler)

    print("Pornire procesare. Ctrl+C pentru oprire controlata.")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        RUN.logger.close()
        face_database.save()
        RUN.failure = failure[0] if failure else None
        summary = write_summary(RUN)

        print("\n--- gata ---")
        print(f"  cadre procesate:   {summary['cadre']}")
        print(f"  fete procesate:    {summary['fete_procesate']}")
        print(f"  recunoasteri:      {summary['recunoasteri']}")
        print(f"  identitati:        {summary['baza_de_date']['identitati_initiale']} "
              f"-> {summary['baza_de_date']['identitati_finale']}")

        # Cat a lucrat continuitatea etichetei. Cifrele astea sunt exact cadrele
        # in care _10 ar fi desenat "face <id>" peste cineva pe care il stia deja.
        cont = summary["continuitate"]
        if cont["verificari"] or cont["casete_tinute"]:
            efecte = ", ".join(f"{name} x{count}"
                               for name, count in cont["verificari"].items())
            print(f"  continuitate:      {efecte or 'nicio verificare'}")
            print(f"    casete purtate din memorie: {cont['casete_tinute']} "
                  f"({cont['casete_vechi']} peste {IDENTITY_STALE_FRAMES} cadre "
                  f"fara confirmare), {cont['casete_anonime']} desenate anonim")
            for event in cont["identitati_preluate"]:
                print(f"    track {event['de_la']} -> {event['track_nou']}: "
                      f"{event['nume']} preluat la cadrul {event['cadru']} "
                      f"(iou {event['iou']})")

        # Cine a intrat, cine a plecat, cine s-a intors -- adunat pe toata sesiunea.
        prezenta = summary["prezenta"]
        if prezenta["miscari"]:
            miscari = ", ".join(f"{name} x{count}"
                                for name, count in prezenta["miscari"].items())
            print(f"  prezenta:          {miscari}")
            for event in prezenta["reveniri"]:
                print(f"    {event['nume']} s-a intors dupa {event['lipsa_s']} s")
            if prezenta["in_cadru_la_final"]:
                print(f"    in cadru la oprire: "
                      f"{', '.join(prezenta['in_cadru_la_final'])}")

        # Daca n-a intrat nimeni in baza, spunem pe loc care prag a stat in cale.
        blocaje = summary["blocaje_inrolare"]
        sarite = summary["verificari_sarite"]["cauze"]
        if blocaje["cauze"]:
            cauze = ", ".join(f"{name} x{count}" for name, count in blocaje["cauze"].items())
            print(f"  inrolari blocate:  {blocaje['incercari']} incercari -> {cauze}")
            for name, key, prag in (("marime", "marime_px", ENROLL_MIN_FACE),
                                    ("blur", "blur", ENROLL_MIN_BLUR),
                                    ("calitate", "calitate", ENROLL_MIN_QUALITY),
                                    ("profil", "frontalitate", ENROLL_MIN_FRONTALITY)):
                if name not in blocaje["cauze"]:
                    continue
                masurat = summary["distributii"][key]
                if masurat:
                    print(f"    {name}: masurat {masurat['min']}-{masurat['max']} "
                          f"(median {masurat['median']}), prag {prag}")
            # Cate persoane s-ar castiga coborand pragul de marime. Fara asta,
            # "marime x453" spune ca pragul taie, dar nu si daca sub el mai e
            # cineva de prins.
            scara = summary["blocaje_inrolare"]["track_uri_peste_prag_marime"]
            if "marime" in blocaje["cauze"] and scara:
                puncte = ", ".join(f"{prag}px -> {count}" for prag, count in scara.items())
                print(f"    track-uri care ating pragul: {puncte} "
                      f"(--enroll-min-face)")
        elif summary["fete_procesate"] == 0:
            print("  (nicio fata nu a trecut de porti: vezi 'respinse' in log)")
        if sarite:
            cauze = ", ".join(f"{name} x{count}" for name, count in sarite.items())
            print(f"  verificari sarite: {cauze}")
        viteza = summary["viteza"]
        print(f"  viteza:            {viteza['fps']} FPS "
              f"({viteza['ms_pe_cadru']} ms/cadru)")
        print(f"  timp probe/cadru:  {summary['timp_probe_ms']['mediu']} ms "
              f"(p95 {summary['timp_probe_ms']['p95']} ms) = "
              f"{viteza['probe_din_total_%']}% din rulare")
        if summary["timp_etape_ms"]:
            etape = ", ".join(f"{name} {value}"
                              for name, value in summary["timp_etape_ms"].items())
            print(f"    din care:        {etape} (ms/cadru)")
        urmarite = summary["apeluri_economisite"]["fete_doar_urmarite"]
        if summary["fete_procesate"]:
            print(f"  landmark-uri:      {summary['gpu']['landmark_faces']} fete din "
                  f"{summary['fete_procesate']} ({urmarite} doar urmarite, "
                  f"{100 * urmarite / summary['fete_procesate']:.0f}% economisit)")
        # Detectorul poate sa fi rulat pe cadre care n-au ajuns niciodata la probe
        # (pipeline cazut intre tracker si conversie), deci "cadre" poate fi 0 cu
        # detectii nenule -- de aici trebuie impartit cu grija.
        detectii = summary["detectii"]
        if detectii["brute"] and summary["cadre"]:
            print(f"  detectii:          {detectii['brute'] / summary['cadre']:.1f}/cadru, "
                  f"la tracker {detectii['la_tracker'] / summary['cadre']:.1f}/cadru")
        elif detectii["brute"]:
            print(f"  detectii:          {detectii['brute']} brute, "
                  f"{detectii['la_tracker']} la tracker (niciun cadru n-a ajuns la probe)")
        free_now, _ = available_memory_mb()
        if free_now is not None:
            print(f"  memorie libera:    {free_now} MB")
        print("")
        if args.record:
            print(f"  {output_video}")
        print(f"  {database_out}")
        if not args.no_frame_log:
            print(f"  {paths['frames']}")
        print(f"  {paths['summary']}")

    return 1 if failure else 0


if __name__ == '__main__':
    sys.exit(main())
