# =============================================================================
# config.py
# Source de verite unique pour tout le projet OrangIdentifier.
# Importe par tous les scripts : from config import *
# Ne jamais hardcoder un chemin ailleurs que dans ce fichier.
# =============================================================================

from pathlib import Path
import json
from datetime import datetime

# =============================================================================
# CHEMINS PRINCIPAUX
# =============================================================================

BASE_DIR = Path(r"D:\OrangIdentifier")

# Donnees brutes
PHOTOS_DIR  = BASE_DIR / "PHOTOS"           # Photos originales par individu
VIDEOS_DIR  = BASE_DIR / "VIDEOS"           # Videos originales

# Pipeline YOLO (detection de visages)
DATASET_YOLO_DIR   = BASE_DIR / "DATASET_YOLO"        # images/ + labels/ bruts
DATASET_SPLIT_DIR  = BASE_DIR / "DATASET_YOLO_SPLIT"  # train/ + val/ pour YOLO
DONE_FILE          = DATASET_YOLO_DIR / "done.txt"     # Stems annotes par l'utilisateur

# Pipeline ResNet50 (classification individuelle)
DATASET_CLASSIF_DIR = BASE_DIR / "DATASET_CLASSIFICATION"
CLASSIF_RAW_DIR     = DATASET_CLASSIF_DIR / "raw"      # Visages extraits par YOLO
CLASSIF_TRAIN_DIR   = DATASET_CLASSIF_DIR / "train"
CLASSIF_VAL_DIR     = DATASET_CLASSIF_DIR / "val"
CLASSIF_TEST_DIR    = DATASET_CLASSIF_DIR / "test"

# Modeles sauvegardes
MODELS_DIR         = BASE_DIR / "MODELS"
YOLO_MODEL_DIR     = MODELS_DIR / "yolo_orangs"
RESNET_MODEL_DIR   = MODELS_DIR / "resnet_orangs"
YOLO_BEST          = YOLO_MODEL_DIR / "best.pt"         # Meilleur YOLO entraine
RESNET_BEST        = RESNET_MODEL_DIR / "best.pth"      # Meilleur ResNet entraine

# Resultats et logs
RESULTS_DIR        = BASE_DIR / "RESULTS"
SESSION_LOG        = BASE_DIR / "session_log.json"
RUNS_DIR           = BASE_DIR / "runs"                  # Sorties Ultralytics

# Scripts
SCRIPTS_DIR        = BASE_DIR / "scripts"
SCRIPTS_ANNEXES    = SCRIPTS_DIR / "scripts_annexes"

# =============================================================================
# INDIVIDUS
# Liste officielle. Ajouter un individu ici suffit pour l'impacter partout.
# =============================================================================

INDIVIDUS = [
    "Auti",
    "Jula",
    "Mathai",
    "Molly",
    "NOAH",
    "PULCO",
    "PUTRI",
    "Sari",
    "Sinta",
    "Ujian",
    # "Kawan",  # Disponible uniquement en video pour l'instant
]

NUM_CLASSES = len(INDIVIDUS)

# =============================================================================
# HYPERPARAMETRES YOLO
# =============================================================================

YOLO_MODEL_BASE  = "yolov8n.pt"   # Modele de base pre-entraine COCO
YOLO_EPOCHS      = 100
YOLO_IMGSZ       = 640
YOLO_BATCH       = 8              # RTX 3050 4Go VRAM
YOLO_PATIENCE    = 20
YOLO_CONF_SEUIL  = 0.25           # Seuil de confiance pour la detection
YOLO_IOU_NMS     = 0.45           # Seuil NMS

# Augmentation YOLO - adaptee orangs-outangs
# Rotation augmentee vs gorilles (poses tres variees vues a l'annotation)
# HSV reduit pour preserver le pelage roux comme feature discriminante
YOLO_DEGREES     = 15.0
YOLO_TRANSLATE   = 0.1
YOLO_SCALE       = 0.5
YOLO_FLIPUD      = 0.5
YOLO_FLIPLR      = 0.5
YOLO_HSV_H       = 0.015
YOLO_HSV_S       = 0.4            # Gorilles : 0.7 — reduit pour le roux
YOLO_HSV_V       = 0.3

# =============================================================================
# HYPERPARAMETRES RESNET50
# =============================================================================

RESNET_EPOCHS        = 100
RESNET_BATCH         = 32
RESNET_LR            = 1e-3
RESNET_PATIENCE      = 10         # Early stopping
RESNET_LR_FACTOR     = 0.5        # ReduceLROnPlateau : divise le LR par 2
RESNET_LR_PATIENCE   = 5          # Epochs avant reduction du LR
RESNET_DROPOUT       = 0.5
RESNET_IMG_SIZE      = 224        # Taille d'entree ResNet50

# Augmentation ResNet - train uniquement
# Saturation reduite pour preserver le pelage roux
RESNET_BRIGHTNESS    = 0.2
RESNET_CONTRAST      = 0.2
RESNET_SATURATION    = 0.2        # Gorilles : 0.2 — on garde conservateur
RESNET_HUE           = 0.1
RESNET_ROTATION      = 15
RESNET_FLIP          = 0.5

# Normalisation ImageNet (obligatoire pour ResNet50 pre-entraine)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Split dataset classification
RESNET_TRAIN_RATIO = 0.617        # Identique aux gorilles (61.7%)
RESNET_VAL_RATIO   = 0.183        # 18.3%
RESNET_TEST_RATIO  = 0.200        # 20%

# =============================================================================
# PIPELINE VIDEO
# =============================================================================

VIDEO_TRACKER        = "botsort"  # BoT-SORT comme les gorilles
VIDEO_CONF_SEUIL     = 0.25
VIDEO_SEUIL_FIABLE   = 0.60       # En dessous : verification manuelle recommandee
VIDEO_FPS_EXTRACTION = 1          # Frames/sec extraites des videos pour le dataset
VIDEO_TRACK_BUFFER   = 300        # Frames de tolerance aux occlusions (BoT-SORT)

# =============================================================================
# ANNOTATION
# =============================================================================

ANNOT_OBJECTIF_PAR_INDIVIDU = 30  # Minimum avant de lancer YOLO

# =============================================================================
# JOURNAL DE SESSION
# =============================================================================

def log_action(script, action, details=None):
    """
    Enregistre une action dans session_log.json.
    Appele automatiquement par chaque script a chaque etape importante.

    Exemple :
        from config import log_action
        log_action("2_train_yolo", "entrainement_termine", {"mAP50": 0.92, "epochs": 68})
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "script":    script,
        "action":    action,
        "details":   details or {}
    }

    log = []
    if SESSION_LOG.exists():
        try:
            log = json.loads(SESSION_LOG.read_text(encoding="utf-8"))
        except Exception:
            log = []

    log.append(entry)
    SESSION_LOG.write_text(
        json.dumps(log, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

# =============================================================================
# CREATION DES DOSSIERS
# Appele une fois au debut de chaque script pour s'assurer que tout existe.
# =============================================================================

def creer_dossiers():
    dossiers = [
        PHOTOS_DIR, VIDEOS_DIR,
        DATASET_YOLO_DIR, DATASET_SPLIT_DIR,
        CLASSIF_RAW_DIR, CLASSIF_TRAIN_DIR, CLASSIF_VAL_DIR, CLASSIF_TEST_DIR,
        MODELS_DIR, YOLO_MODEL_DIR, RESNET_MODEL_DIR,
        RESULTS_DIR, RUNS_DIR, SCRIPTS_ANNEXES,
    ]
    for d in dossiers:
        d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# VERIFICATION DE L'ETAT DU PROJET (utilisee par dataset_status.py)
# =============================================================================

def lire_done():
    if not DONE_FILE.exists():
        return set()
    return set(s.strip() for s in DONE_FILE.read_text().splitlines() if s.strip())

def compter_valides_par_individu(done=None):
    from collections import defaultdict
    if done is None:
        done = lire_done()
    labels_dir = DATASET_YOLO_DIR / "labels"
    compteur = defaultdict(int)
    for stem in done:
        lbl = labels_dir / (stem + ".txt")
        if lbl.exists() and lbl.stat().st_size > 0:
            compteur[stem.split("_")[0]] += 1
    return compteur