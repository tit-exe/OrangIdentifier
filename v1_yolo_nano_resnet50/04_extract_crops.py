# =============================================================================
# 04_extract_crops.py
# Automatic face extraction from all the original photos.
# Uses the trained YOLO model (best.pt).
#
# IMPORTANT: deletes and recreates data/crops/known/ on every run.
#
# Output structure:
#   data/crops/known/
#       Auti/              <- 1 crop 224x224 per photo, certain identity
#       Molly/
#       ...
#       _a_verifier/
#           Auti/          <- original photo + JSON boxes if 2+ faces
#   data/crops.json        <- YOLO coordinates of each crop, for the review app
#
# Usage: python v1_yolo_nano_resnet50/04_extract_crops.py
# =============================================================================

import sys
import time
import shutil
import cv2
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.config_loader import (
    PHOTOS_DIR, CROPS_KNOWN_DIR, CROPS_JSON,
    YOLO_V2_PT as YOLO_BEST, OUTPUT_DIR, YOLO_CONFIDENCE, ensure_dirs,
    to_relative,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONF_EXTRACTION  = YOLO_CONFIDENCE  # from config.yaml (default 0.30)
MARGE_CROP       = 0.05             # margin around box (5%)
TAILLE_FACE      = 224              # ResNet50 / MegaDescriptor input size

# Cache file: YOLO coordinates of each extracted crop
CLASSIF_RAW_DIR   = CROPS_KNOWN_DIR        # data/crops/known/
DATASET_CLASSIF_DIR = CROPS_KNOWN_DIR.parent  # data/crops/
BOXES_CACHE       = CROPS_JSON             # data/crops.json (unified)

# =============================================================================
# VERIFICATION
# =============================================================================

def verifier_prerequisites():
    if not YOLO_BEST.exists():
        print(f"ERROR: YOLO model not found: {YOLO_BEST}")
        sys.exit(1)
    if not PHOTOS_DIR.exists():
        print(f"ERROR: PHOTOS folder not found: {PHOTOS_DIR}")
        sys.exit(1)

    individus = sorted([d.name for d in PHOTOS_DIR.iterdir() if d.is_dir()])
    if not individus:
        print(f"ERROR: No subfolder in {PHOTOS_DIR}")
        sys.exit(1)

    print("=" * 60)
    print("STEP 3 - FACE EXTRACTION")
    print("=" * 60)
    print(f"  Model    : {YOLO_BEST.name}")
    print(f"  Photos   : {PHOTOS_DIR}")
    print(f"  Output   : {CLASSIF_RAW_DIR}")
    print(f"  Cache    : {BOXES_CACHE.name}")
    print(f"  Conf     : {CONF_EXTRACTION}")
    print(f"  Margin   : {MARGE_CROP*100:.0f}%")
    print(f"  Size     : {TAILLE_FACE}x{TAILLE_FACE}px")
    print(f"  Individuals: {individus}")
    print("=" * 60)
    return individus

# =============================================================================
# CLEANUP AND PREPARATION
# =============================================================================

def nettoyer_et_preparer(individus):
    # Delete the whole raw/ folder if it exists
    if CLASSIF_RAW_DIR.exists():
        print(f"\n  Deleting existing raw/ ...")
        shutil.rmtree(CLASSIF_RAW_DIR)
        print(f"  Deleted.")

    # Delete existing cache
    if BOXES_CACHE.exists():
        BOXES_CACHE.unlink()

    # Recreate the structure
    for ind in individus:
        (CLASSIF_RAW_DIR / ind).mkdir(parents=True, exist_ok=True)

    verif_dir = CLASSIF_RAW_DIR / "_a_verifier"
    for ind in individus:
        (verif_dir / ind).mkdir(parents=True, exist_ok=True)

    DATASET_CLASSIF_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Folders created.")
    return verif_dir

# =============================================================================
# CROP
# =============================================================================

def extraire_crop(img, x1, y1, x2, y2):
    """
    Extract a crop from an OpenCV image with a margin,
    resized to TAILLE_FACE x TAILLE_FACE.
    Returns (resized_face, x1_final, y1_final, x2_final, y2_final)
    so the actual coordinates used can be stored in the cache.
    """
    h, w = img.shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    mx = int(bw * MARGE_CROP)
    my = int(bh * MARGE_CROP)

    x1f = max(0, x1 - mx)
    y1f = max(0, y1 - my)
    x2f = min(w, x2 + mx)
    y2f = min(h, y2 + my)

    crop = img[y1f:y2f, x1f:x2f]
    if crop.size == 0:
        return None, x1f, y1f, x2f, y2f

    face = cv2.resize(crop, (TAILLE_FACE, TAILLE_FACE), interpolation=cv2.INTER_LANCZOS4)
    return face, x1f, y1f, x2f, y2f

# =============================================================================
# PROGRESS BAR
# =============================================================================

def progress_bar(current, total, start, width=40):
    pct     = current / total if total > 0 else 0
    fill    = int(width * pct)
    b       = "=" * fill + "-" * (width - fill)
    elapsed = time.time() - start
    eta     = str(timedelta(seconds=int(elapsed / current * (total - current)))) if current > 0 else "?"
    print(f"\r  [{b}] {current}/{total} ({pct*100:.1f}%)  ETA: {eta}   ", end="", flush=True)

# =============================================================================
# MAIN EXTRACTION
# =============================================================================

def extraire(individus, verif_dir):
    from ultralytics import YOLO

    print("\n  Loading YOLO model...")
    model = YOLO(str(YOLO_BEST))
    print("  Model loaded.\n")

    # Collect all the photos
    toutes = []
    for ind in individus:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in (PHOTOS_DIR / ind).glob(ext):
                toutes.append((ind, p))
    total = len(toutes)
    print(f"  {total} photos | {len(individus)} individuals\n")

    debut = time.time()

    # Boxes cache: key = "Individual/stem"
    # Holds all the info needed to recover the original coordinates
    boxes_cache = {}

    # Statistics
    stats = defaultdict(lambda: {
        "photos": 0, "faces": 0, "multi": 0, "rates": 0, "erreurs": 0
    })
    gs = {"total": total, "faces": 0, "multi": 0, "rates": 0, "erreurs": 0}
    journal_multi = []

    for i, (individu, photo_path) in enumerate(toutes):
        progress_bar(i + 1, total, debut)

        img = cv2.imread(str(photo_path))
        if img is None:
            stats[individu]["erreurs"] += 1
            gs["erreurs"] += 1
            continue

        h, w = img.shape[:2]

        try:
            results = model.predict(
                source=str(photo_path),
                conf=CONF_EXTRACTION,
                verbose=False,
                device=0,
            )
        except Exception:
            stats[individu]["erreurs"] += 1
            gs["erreurs"] += 1
            continue

        boxes = results[0].boxes
        stats[individu]["photos"] += 1
        nom   = photo_path.stem
        cle   = f"{individu}/{nom}"

        if boxes is None or len(boxes) == 0:
            stats[individu]["rates"] += 1
            gs["rates"] += 1
            continue

        nb = len(boxes)

        if nb == 1:
            # --- Simple case: 1 face ---
            x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[0].tolist()]
            conf = float(boxes.conf[0])
            face, x1f, y1f, x2f, y2f = extraire_crop(img, x1, y1, x2, y2)

            if face is not None:
                dest = CLASSIF_RAW_DIR / individu / f"{nom}.jpg"
                cv2.imwrite(str(dest), face)

                # Store in the cache (RELATIVE paths for portability)
                boxes_cache[cle] = {
                    "individu":      individu,
                    "stem":          nom,
                    "source_type":   "known",
                    "photo_source":  to_relative(photo_path),
                    "crop_file":     to_relative(dest),
                    "img_w":         w,
                    "img_h":         h,
                    # Raw YOLO box (before margin)
                    "yolo_x1": x1, "yolo_y1": y1,
                    "yolo_x2": x2, "yolo_y2": y2,
                    "yolo_conf": round(conf, 3),
                    # Final crop box (with margin) — used by reviewer
                    "crop_x1": x1f, "crop_y1": y1f,
                    "crop_x2": x2f, "crop_y2": y2f,
                    "n_faces": 1,
                    "statut": "auto",
                    "ts": datetime.now().isoformat(),
                }

                stats[individu]["faces"] += 1
                gs["faces"] += 1

        else:
            # --- Multi case: 2+ faces ---
            stats[individu]["multi"] += 1
            gs["multi"] += 1

            dest_verif = verif_dir / individu / f"{nom}.jpg"
            shutil.copy(str(photo_path), str(dest_verif))

            boites_info = []
            for j, box in enumerate(boxes):
                bx1, by1, bx2, by2 = [int(v) for v in box.xyxy[0].tolist()]
                bconf = float(box.conf[0])
                boites_info.append({
                    "index": j,
                    "x1": bx1, "y1": by1, "x2": bx2, "y2": by2,
                    "conf": round(bconf, 3),
                })

            # JSON next to the photo in _a_verifier
            json_path = verif_dir / individu / f"{nom}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "individu":      individu,
                    "stem":          nom,
                    "photo_source":  str(photo_path),
                    "img_w":         w,
                    "img_h":         h,
                    "nb_detections": nb,
                    "boites":        boites_info,
                }, f, indent=2, ensure_ascii=False)

            # Also in the global cache with status "multi"
            boxes_cache[cle] = {
                "individu":      individu,
                "stem":          nom,
                "photo_source":  str(photo_path),
                "crop_dest":     None,  # not extracted yet
                "img_w":         w,
                "img_h":         h,
                "nb_detections": nb,
                "boites":        boites_info,
                "statut":        "multi_a_verifier",
                "ts_extraction": datetime.now().isoformat(),
            }

            journal_multi.append({
                "individu": individu,
                "stem":     nom,
                "nb":       nb,
            })

    print()

    # Save the cache
    with open(BOXES_CACHE, 'w', encoding='utf-8') as f:
        json.dump(boxes_cache, f, indent=2, ensure_ascii=False)
    print(f"\n  Cache saved: {BOXES_CACHE}")
    print(f"  Entries in cache: {len(boxes_cache)}")

    return stats, gs, journal_multi, boxes_cache

# =============================================================================
# RAPPORT
# =============================================================================

def rapport(stats, gs, journal_multi, debut):
    duree = timedelta(seconds=int(time.time() - debut))

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(f"  Duration           : {duree}")
    print(f"  Photos processed   : {gs['total']}")
    print(f"  Faces extracted    : {gs['faces']}")
    print(f"  Multi-detections   : {gs['multi']}  -> _a_verifier/")
    print(f"  Photos without face: {gs['rates']}")
    print(f"  Errors             : {gs['erreurs']}")

    print(f"\n  {'Individual':<28} {'Photos':>7} {'Faces':>6} {'Multi':>6} {'Miss':>6}")
    print(f"  {'-'*28} {'-'*7} {'-'*6} {'-'*6} {'-'*6}")
    for ind in sorted(stats.keys()):
        s = stats[ind]
        print(f"  {ind:<28} {s['photos']:>7} {s['faces']:>6} {s['multi']:>6} {s['rates']:>6}")

    taux = gs['faces'] / max(gs['total'], 1) * 100
    print(f"\n  Auto rate : {taux:.1f}%")
    print("=" * 60)
    print("STEP 3 COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run common/review_crops.py to correct/validate the crops")
    print("  2. Run 05_train_resnet50.py once the dataset is clean")

    metriques = {
        "faces": gs["faces"], "multi": gs["multi"],
        "rates": gs["rates"], "taux": round(taux, 2),
        "duree_s": int(time.time() - debut)
    }

    # Save JSON report
    (OUTPUT_DIR / "results").mkdir(parents=True, exist_ok=True)
    rp = OUTPUT_DIR / "results" / "extraction_rapport.json"
    with open(rp, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp":    datetime.now().isoformat(),
            "global":       gs,
            "metriques":    metriques,
            "multi_detail": journal_multi,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Report : {rp}")

    return metriques

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    debut = time.time()

    individus = verifier_prerequisites()

    crops_exist = CLASSIF_RAW_DIR.exists() and any(CLASSIF_RAW_DIR.rglob("*.jpg"))
    if crops_exist:
        print("\n  NOTE: Existing crops in data/crops/known/ will be cleared and regenerated.")
        print("        Your original photos in data/photos/ are NOT touched.")
        confirmation = input("\n  Ready to start? (y/n): ").strip().lower()
    else:
        print("\n  Photos found. YOLO will detect faces and save 224x224 crops to data/crops/known/")
        print("  Your original photos in data/photos/ are NOT modified.")
        confirmation = input("\n  Start extraction? (y/n): ").strip().lower()

    if confirmation not in ('y', 'o'):
        print("Cancelled.")
        sys.exit(0)

    verif_dir = nettoyer_et_preparer(individus)
    stats, gs, journal_multi, cache = extraire(individus, verif_dir)
    metriques = rapport(stats, gs, journal_multi, debut)
    print(f"  Metrics: {metriques}")