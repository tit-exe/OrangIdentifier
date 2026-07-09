r"""
build_gallery.py  --  OrangIdentifier maintenance
=================================================
OPTION A: add new individuals WITHOUT retraining the brain.

What this script does
---------------------
It takes the existing V6 brain (the backbone, already trained) and, for every
individual it finds in the crop folders, it computes a small set of reference
vectors ("exemplars"). It writes all of them into a single file, gallery.json.
That file is the list of known animals the phone app compares against.

Use this script when you add new animals that look CLEARLY DIFFERENT from each
other (the normal zoo case). You do not need a graphics card and it takes only
a few minutes. The brain file (the .tflite) does not change at all, so you only
copy the new gallery.json into the app.

Do NOT use this if the new animals look almost identical to each other. In that
case the brain itself has to be retrained: see option_B_retrain_brain.

Before running this
-------------------
1. You must already have the crops (small 224x224 head pictures). If you only
   have raw photos, run crop_photos.py first.
2. Every individual must be one folder named exactly with the animal name, for
   example:  2_crops_appear_here/Rosa/img001.jpg

How to run (in the "orangs" Anaconda environment, on Windows)
-------------------------------------------------------------
    conda activate orangs
    python maintenance/scripts/build_gallery.py

The script writes gallery.json and, if DEPLOY_TO_APP is True, also copies it
straight into the Android app. Then rebuild the app in Android Studio.
"""

import os
# HF/Torch caches use the OS default location

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys, json, shutil, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
from PIL import Image, ImageFile
REPO = Path(__file__).resolve().parents[2]  # repository root (portable)

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

# ==============================================================================
# SETTINGS  --  this is the only part you may need to change
# ==============================================================================
# The trained brain (backbone) produced by V6. Do not change unless you moved it.
BACKBONE_PT = (REPO / "models" / "megadesc_T_arcface_v6_15ind_acc98.pt")

# Folders that contain the crops, one sub-folder per individual.
# You can list as many folders as you want. All individuals found in all of
# them are merged and sorted by name.
CROP_DIRS = [
    (REPO / "data" / "crops" / "known"),   # the 10 original animals
    (REPO / "data" / "crops" / "new"),        # the 5 animals added in V6
    (REPO / "maintenance" / "new_animals" / "2_crops_appear_here"),  # your new animals
]

# Sub-folders whose name starts with "_" are ignored, plus any name listed here.
EXCLUDE_FOLDERS = ["_a_verifier"]

# Where to write the new gallery. This is a NEW file, so the current V6 gallery
# (the downloaded V6 gallery) is kept untouched as a safe backup.
OUTPUT_GALLERY = REPO / "maintenance" / "new_animals" / "updated_gallery.json"

# The Android app is a SEPARATE repository (OrangIdentifier-Android).
# If True, copy the gallery to APP_GALLERY (set it to your local app repo assets).
DEPLOY_TO_APP  = False
APP_GALLERY    = REPO / "maintenance" / "new_animals" / "gallery.json"

# ==============================================================================
# FIXED VALUES  --  must match the way the brain was trained. Do not change.
# ==============================================================================
IMG_SIZE = 224
MEAN = STD = [0.5, 0.5, 0.5]
EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

K_EXEMPLARS = 25     # how many reference vectors to keep per individual
QUALITY_MIN = 0.60   # a crop is "good" if it is at least this close to the average

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg=""):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)

def section(title):
    bar = "-" * 68
    log(); log(bar); log(f"  {title}"); log(bar)


# ==============================================================================
# IMAGE LOADING  --  same preprocessing as training and as the app
# ==============================================================================
_val_tf = T.Compose([
    T.Resize(IMG_SIZE),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

class CropDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        return _val_tf(img), int(self.labels[idx])


def load_individuals(crop_dirs, exclude=None):
    """Find every individual folder, merge, sort by name. Returns paths, labels, names."""
    excl = set(exclude or [])
    folders = []
    for base in crop_dirs:
        if not base.exists():
            log(f"  [WARNING] folder not found, skipped: {base}")
            continue
        for d in base.iterdir():
            if d.is_dir() and not d.name.startswith("_") and d.name not in excl:
                folders.append(d)

    folders.sort(key=lambda d: d.name.lower())

    paths, labels, names = [], [], []
    for i, d in enumerate(folders):
        imgs = sorted([f for f in d.iterdir() if f.suffix in EXTS])
        if not imgs:
            log(f"  [WARNING] {d.name}: no images, skipped")
            continue
        for f in imgs:
            paths.append(f)
            labels.append(i)
        names.append(d.name)
    return paths, labels, names


# ==============================================================================
# BRAIN LOADING
# ==============================================================================
def load_backbone():
    log("  Building the MegaDescriptor-T-224 architecture...")
    bb = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224", pretrained=False, num_classes=0)

    if not BACKBONE_PT.exists():
        log(f"  [ERROR] brain file not found: {BACKBONE_PT}")
        sys.exit(1)

    log(f"  Loading the trained weights from {BACKBONE_PT.name} "
        f"({BACKBONE_PT.stat().st_size/1e6:.0f} MB)...")
    ck = torch.load(str(BACKBONE_PT), map_location="cpu", weights_only=False)
    state = ck.get("backbone_state") or ck.get("model_state_dict") or ck
    missing, unexpected = bb.load_state_dict(state, strict=False)
    log(f"  Weights loaded ({len(missing)} missing, {len(unexpected)} unexpected)")

    bb = bb.eval().to(DEVICE)
    with torch.no_grad():
        emb_dim = bb(torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)).shape[1]
    log(f"  Vector size: {emb_dim} (expected 768)")
    return bb, emb_dim


# ==============================================================================
# GALLERY BUILDING  --  identical logic to the V6 training script
# ==============================================================================
@torch.no_grad()
def build_gallery(bb, paths, labels, names, emb_dim):
    section("Computing vectors for every crop")
    dl = DataLoader(CropDataset(paths, labels), batch_size=64, num_workers=0)
    embs, labs = [], []
    for imgs, lbs in dl:
        embs.append(F.normalize(bb(imgs.to(DEVICE).float()), dim=1).cpu().numpy())
        labs.extend(lbs.tolist())
    embs = np.concatenate(embs).astype(np.float32)
    labs = np.array(labs)

    section("Selecting reference vectors per individual")
    individuals = {}
    all_exemplars = {}   # name -> array of reference vectors

    for i, name in enumerate(names):
        ei = embs[labs == i]
        if len(ei) == 0:
            log(f"  [WARNING] {name}: 0 crops")
            continue

        # The centroid is the "average look" of this individual.
        centroid = ei.mean(0)
        centroid /= (np.linalg.norm(centroid) + 1e-8)

        # Keep the crops closest to the average (the cleanest ones).
        sims = ei @ centroid
        good = ei[sims >= QUALITY_MIN]
        if len(good) < 3:
            good = ei[np.argsort(-sims)[:max(3, K_EXEMPLARS // 2)]]

        gs = good @ centroid
        top_k = min(K_EXEMPLARS, len(good))
        best = good[np.argsort(-gs)[:top_k]]
        norms = np.linalg.norm(best, axis=1, keepdims=True)
        best = best / np.where(norms > 1e-8, norms, 1)
        all_exemplars[name] = best

        individuals[name] = {
            "class_index":   i,
            "is_zoo":        True,
            "num_crops":     int(len(ei)),
            "num_exemplars": len(best),
            "mean_intra":    round(float(np.mean(sims)), 4),
            "embedding":     centroid.tolist(),   # single average vector
            "exemplars":     best.tolist(),       # the reference vectors
        }
        log(f"  {name:<16}: {len(ei):3d} crops -> {len(best):2d} reference vectors")

    # ---- Threshold calibration (same method as the app scoring) --------------
    section("Calibrating the unknown threshold")
    pos_sims, neg_sims = [], []
    for i, name in enumerate(names):
        if name not in all_exemplars:
            continue
        ei = embs[labs == i]
        if len(ei) == 0:
            continue
        own = all_exemplars[name]
        pos_sims.extend((ei @ own.T).max(1).tolist())
        others = [v for k, v in all_exemplars.items() if k != name]
        if others:
            other = np.vstack(others)
            neg_sims.extend((ei @ other.T).max(1).tolist())

    pos = np.array(pos_sims)
    neg = np.array(neg_sims) if neg_sims else np.zeros(1)
    gap = float(pos.mean() - neg.mean())
    log(f"  Same animal   (should be high): {pos.mean():.4f}")
    log(f"  Other animals (should be low) : {neg.mean():.4f}")
    log(f"  Separation gap                : {gap:.4f}")

    thresholds = np.linspace(0, 1, 500)
    f1s = []
    for t in thresholds:
        tp = int((pos >= t).sum())
        fp = int((neg >= t).sum())
        fn = int((pos < t).sum())
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1s.append(2 * p * r / (p + r + 1e-9))
    best_threshold = float(thresholds[int(np.argmax(f1s))])
    log(f"  Chosen threshold: {best_threshold:.4f}  (F1 = {max(f1s):.4f})")

    gallery = {
        "version": "v6",
        "created": datetime.now().isoformat(),
        "model": "MegaDescriptor-T-224 + SubCenterArcFace V6",
        "embedding_dim": emb_dim,
        "similarity_metric": "cosine",
        "normalization": "megadescriptor",
        "unknown_threshold": round(best_threshold, 4),
        "separability_gap": round(gap, 4),
        "num_individuals": len(individuals),
        "n_zoo": len(individuals),
        "inference_note": "score = max(dot(query, exemplar)) -- app: max(anchor, field)",
        "individuals": individuals,
    }
    OUTPUT_GALLERY.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_GALLERY.write_text(
        json.dumps(gallery, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8")
    log(f"  Gallery written: {OUTPUT_GALLERY} ({OUTPUT_GALLERY.stat().st_size/1024:.0f} KB)")
    return best_threshold, gap


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    section(f"BUILD GALLERY  --  {datetime.now():%Y-%m-%d %H:%M}")
    log(f"  Device: {DEVICE}")

    section("Finding the individuals")
    paths, labels, names = load_individuals(CROP_DIRS, EXCLUDE_FOLDERS)
    if not names:
        log("  [ERROR] No individual found. Check CROP_DIRS at the top of this file.")
        sys.exit(1)
    log(f"  Found {len(names)} individuals, {len(paths)} crops total.")
    log(f"  Names: {names}")

    section("Loading the trained brain")
    bb, emb_dim = load_backbone()

    threshold, gap = build_gallery(bb, paths, labels, names, emb_dim)

    if DEPLOY_TO_APP:
        section("Copying the gallery into the Android app")
        APP_GALLERY.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(OUTPUT_GALLERY), str(APP_GALLERY))
        log(f"  Copied to: {APP_GALLERY}")
        log("  The brain file (.tflite) was NOT changed, only the gallery.")

    section("DONE")
    log(f"""
  Individuals in the gallery : {len(names)}
  Unknown threshold          : {threshold:.4f}
  Separation gap             : {gap:.4f}

  Gallery file : {OUTPUT_GALLERY}
  {'App updated  : ' + str(APP_GALLERY) if DEPLOY_TO_APP else 'App not touched (DEPLOY_TO_APP is False)'}

  NEXT STEP:
    Open the app in Android Studio, then Build > Clean Project > Rebuild Project.
""")


if __name__ == "__main__":
    main()
