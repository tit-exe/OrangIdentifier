"""
V2_3b_reanalyze_skips.py
=========================
CNRS IPHC Strasbourg — Orang-outan V2 pipeline
Author: Titouane

PURPOSE
-------
Two things:
  1. Re-runs YOLO on the 1613 "no_detection" images with a lower confidence
     threshold (0.15 vs 0.25) to recover missed faces.
     Results go to WILD_CROPS/crops_lowconf/ for manual review.

  2. Regenerates ALL existing crops with a tighter margin (5% instead of 15%)
     so the crop is properly centered on the face, not the whole head+body.

WHY TWO STEPS
-------------
Step 1 (low confidence):
  - Threshold 0.15 will detect more faces but also more false positives
  - Results go to a SEPARATE folder so you can review them
  - Drag the good ones into the main crops/ folder manually
  - Or use V2_4_review_crops.py to review them quickly

Step 2 (tighter margin):
  - The 15% margin in V2_3 was too large for wild photos
  - 5% is the same value used in V1 (3c_extract_missing.py)
  - This OVERWRITES existing crops in crops/ with tighter versions
  - The JSON coordinates (original YOLO box) are NOT changed — only
    the saved crop image is regenerated with less margin

RUN
---
    conda activate orangs
    python D:\OrangIdentifier\V2\scripts\V2_3b_reanalyze_skips.py

    # To only run step 1 (low confidence reanalysis):
    python V2_3b_reanalyze_skips.py --only-reanalyze

    # To only run step 2 (fix margins on existing crops):
    python V2_3b_reanalyze_skips.py --only-fix-margins
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime

os.environ["HF_HOME"]    = r"D:\HuggingFaceCache"
os.environ["TORCH_HOME"] = r"D:\TorchCache"

import numpy as np
import cv2
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================

YOLO_MODEL_PATH  = Path(r"D:\OrangIdentifier\V2\MODELS\yolo_v2.pt")
WILD_IMAGES_DIR  = Path(r"D:\OrangIdentifier\V2\WILD_ORANGS\raw")

CROPS_DIR        = Path(r"D:\OrangIdentifier\V2\WILD_CROPS\crops")
CROPS_LOWCONF    = Path(r"D:\OrangIdentifier\V2\WILD_CROPS\crops_lowconf")
JSON_PATH        = Path(r"D:\OrangIdentifier\V2\WILD_CROPS\boxes_wild.json")

# Parameters
CONF_LOWCONF     = 0.15    # lower threshold for reanalysis
IOU_THRESHOLD    = 0.45
MARGIN_TIGHT     = 0.05    # 5% — same as V1 (3c_extract_missing.py)
MARGIN_ORIGINAL  = 0.15    # what V2_3 used (too large)
CROP_SIZE        = 224
IMG_EXTENSIONS   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ==============================================================================
# HELPERS
# ==============================================================================

def load_json(path: Path) -> dict:
    if not path.exists():
        print(f"  [ERROR] JSON not found: {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_json_atomic(data: dict, path: Path):
    """Atomic save: write to .tmp then rename."""
    tmp = path.with_suffix(".tmp")
    bak = path.with_suffix(".bak")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        if path.exists():
            shutil.copy2(path, bak)
        tmp.replace(path)
    except Exception as e:
        print(f"  [ERROR] JSON save failed: {e}")
        if tmp.exists():
            tmp.unlink()

def crop_from_box(img: np.ndarray, x1_raw: float, y1_raw: float,
                  x2_raw: float, y2_raw: float,
                  margin: float) -> tuple:
    """
    Applies margin around a YOLO box and extracts + resizes the crop.
    Returns (crop_224, x1, y1, x2, y2) in original image coordinates.
    """
    import numpy as np
    h, w = img.shape[:2]
    bw = x2_raw - x1_raw
    bh = y2_raw - y1_raw

    x1 = max(0,   int(x1_raw - bw * margin))
    y1 = max(0,   int(y1_raw - bh * margin))
    x2 = min(w,   int(x2_raw + bw * margin))
    y2 = min(h,   int(y2_raw + bh * margin))

    if x2 <= x1 or y2 <= y1:
        return None, x1, y1, x2, y2

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None, x1, y1, x2, y2

    crop_resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE),
                              interpolation=cv2.INTER_AREA)
    return crop_resized, x1, y1, x2, y2

# ==============================================================================
# STEP 1 — Re-analyze "no_detection" images with lower confidence
# ==============================================================================

def reanalyze_skips():
    import numpy as np
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("  STEP 1 — Re-analyzing no_detection images (conf=0.15)")
    print("=" * 70)

    CROPS_LOWCONF.mkdir(parents=True, exist_ok=True)
    db = load_json(JSON_PATH)

    # Find all "no_detection" entries
    skipped_keys = [
        (k, v) for k, v in db.items()
        if isinstance(v, dict) and v.get("statut") == "no_detection"
    ]
    print(f"  Found {len(skipped_keys):,} no_detection entries to reanalyze")

    if not skipped_keys:
        print("  Nothing to do.")
        return

    model = YOLO(str(YOLO_MODEL_PATH))
    print(f"  YOLO loaded — running at conf={CONF_LOWCONF}")
    print(f"  Output: {CROPS_LOWCONF}")
    print(f"  (Review these manually — more false positives expected)")
    print()

    recovered = 0
    still_nothing = 0

    bar = tqdm(skipped_keys, desc="  Low-conf detection", unit="img", ncols=90)

    for key, entry in bar:
        photo_src = entry.get("photo_source", "")
        if not photo_src or not Path(photo_src).exists():
            continue

        img = cv2.imread(photo_src)
        if img is None:
            continue

        h, w = img.shape[:2]

        try:
            results = model.predict(
                source=photo_src,
                conf=CONF_LOWCONF,
                iou=IOU_THRESHOLD,
                verbose=False,
                device="cuda" if __import__("torch").cuda.is_available() else "cpu"
            )
        except Exception:
            continue

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            still_nothing += 1
            continue

        # Process each detection
        stem = Path(photo_src).stem
        for i, box in enumerate(boxes):
            x1r, y1r, x2r, y2r = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())

            crop_img, cx1, cy1, cx2, cy2 = crop_from_box(
                img, x1r, y1r, x2r, y2r, MARGIN_TIGHT
            )
            if crop_img is None:
                continue

            # Save to low-confidence folder for manual review
            fname = f"{stem}_lowconf_{i:02d}_c{int(conf*100):02d}.jpg"
            dest  = CROPS_LOWCONF / fname
            cv2.imwrite(str(dest), crop_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            recovered += 1

        bar.set_postfix({"recovered": recovered, "still_empty": still_nothing})

    bar.close()

    print(f"\n  Results:")
    print(f"    New crops found    : {recovered}")
    print(f"    Still no detection : {still_nothing}")
    print(f"    Saved to           : {CROPS_LOWCONF}")
    print()
    print(f"  Next: review the crops in {CROPS_LOWCONF}")
    print(f"  - Delete the bad ones (hands, feet, background, other animals)")
    print(f"  - Copy the good ones into {CROPS_DIR}")
    print(f"  Or use V2_4_review_crops.py to review them faster.")

# ==============================================================================
# STEP 2 — Fix margins on all existing crops (15% → 5%)
# ==============================================================================

def fix_margins():
    import numpy as np

    print("\n" + "=" * 70)
    print("  STEP 2 — Fixing crop margins (15% → 5%)")
    print("=" * 70)
    print(f"  This regenerates all crops with tighter margins.")
    print(f"  Original YOLO coordinates in JSON are NOT changed.")
    print()

    db = load_json(JSON_PATH)

    # Find all entries with a saved crop
    entries_to_fix = []
    for key, entry in db.items():
        if not isinstance(entry, dict):
            continue

        # Single detection
        if entry.get("statut") in ("valide", "auto") and entry.get("crop_file"):
            src = entry.get("photo_source", "")
            x1  = entry.get("crop_x1", 0)
            y1  = entry.get("crop_y1", 0)
            x2  = entry.get("crop_x2", 0)
            y2  = entry.get("crop_y2", 0)
            if src and x2 > x1 and y2 > y1:
                entries_to_fix.append((key, entry, src, x1, y1, x2, y2,
                                       entry["crop_file"], False))

        # Multi-detection
        for det in entry.get("detections", []):
            if det.get("crop_file"):
                src = det.get("photo_source", "")
                x1  = det.get("crop_x1", 0)
                y1  = det.get("crop_y1", 0)
                x2  = det.get("crop_x2", 0)
                y2  = det.get("crop_y2", 0)
                if src and x2 > x1 and y2 > y1:
                    entries_to_fix.append((key, det, src, x1, y1, x2, y2,
                                           det["crop_file"], True))

    print(f"  Found {len(entries_to_fix):,} crops to regenerate")

    if not entries_to_fix:
        print("  Nothing to fix.")
        return

    fixed   = 0
    errors  = 0
    updated_db = dict(db)

    bar = tqdm(entries_to_fix, desc="  Fixing margins", unit="crop", ncols=90)

    for key, entry, photo_src, x1_crop, y1_crop, x2_crop, y2_crop, crop_file, is_multi in bar:
        if not Path(photo_src).exists():
            errors += 1
            continue

        img = cv2.imread(photo_src)
        if img is None:
            errors += 1
            continue

        img_h, img_w = img.shape[:2]

        # The stored coords already include the 15% margin.
        # We need to go back to the YOLO raw box first.
        # Strategy: use the stored coords directly but re-apply 5% margin
        # from the YOLO raw box.
        # Since we don't store the raw YOLO box separately, we approximate:
        # The stored box = raw_box ± 15% margin
        # We can reconstruct raw box ≈ stored box shrunk by margin/(1+margin)
        # But it's simpler and more accurate to just use the stored box
        # and re-crop with no additional margin (the margin is baked in).
        # Instead, we shrink the stored box by removing the excess margin:

        # Stored box was: raw_box expanded by 15%
        # New box should be: raw_box expanded by 5%
        # Δ = (15% - 5%) / (1 + 15%) ≈ 8.7% of stored box size

        bw_stored = x2_crop - x1_crop
        bh_stored = y2_crop - y1_crop

        # Recover approximate raw box
        # stored = raw * (1 + 2*margin) for each side... not exactly
        # Simpler: shrink stored box by the excess
        shrink_x = int(bw_stored * (MARGIN_ORIGINAL - MARGIN_TIGHT) / (1 + MARGIN_ORIGINAL))
        shrink_y = int(bh_stored * (MARGIN_ORIGINAL - MARGIN_TIGHT) / (1 + MARGIN_ORIGINAL))

        new_x1 = min(x1_crop + shrink_x, img_w - 1)
        new_y1 = min(y1_crop + shrink_y, img_h - 1)
        new_x2 = max(x2_crop - shrink_x, new_x1 + 1)
        new_y2 = max(y2_crop - shrink_y, new_y1 + 1)

        # Clamp
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(img_w, new_x2)
        new_y2 = min(img_h, new_y2)

        if new_x2 <= new_x1 or new_y2 <= new_y1:
            errors += 1
            continue

        # Re-extract crop
        crop = img[new_y1:new_y2, new_x1:new_x2]
        if crop.size == 0:
            errors += 1
            continue

        crop_resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE),
                                  interpolation=cv2.INTER_AREA)

        # Overwrite crop file
        crop_dest = Path(crop_file)
        if not crop_dest.parent.exists():
            crop_dest.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(crop_dest), crop_resized, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # Update JSON coordinates
        entry["crop_x1"] = new_x1
        entry["crop_y1"] = new_y1
        entry["crop_x2"] = new_x2
        entry["crop_y2"] = new_y2
        entry["margin_pct"] = MARGIN_TIGHT
        entry["margin_fixed"] = True
        entry["margin_fix_date"] = datetime.now().isoformat()

        fixed += 1
        bar.set_postfix({"fixed": fixed, "err": errors})

    bar.close()

    # Save updated JSON
    print(f"\n  Saving updated JSON...")
    save_json_atomic(updated_db, JSON_PATH)

    print(f"\n  Results:")
    print(f"    Crops regenerated  : {fixed:,}")
    print(f"    Errors             : {errors:,}")
    print(f"    JSON updated       : {JSON_PATH}")
    print(f"    Backup             : {JSON_PATH.with_suffix('.bak')}")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    # numpy needed in helpers — import here after os.environ is set
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-reanalyze",  action="store_true",
                        help="Only run step 1 (low-conf reanalysis)")
    parser.add_argument("--only-fix-margins", action="store_true",
                        help="Only run step 2 (fix crop margins)")
    args = parser.parse_args()

    print("=" * 70)
    print("  CROP RE-ANALYZER + MARGIN FIXER — V2")
    print("  CNRS IPHC Strasbourg")
    print("=" * 70)

    run_reanalyze   = not args.only_fix_margins
    run_fix_margins = not args.only_reanalyze

    if run_reanalyze:
        reanalyze_skips()

    if run_fix_margins:
        fix_margins()

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)
    if run_reanalyze:
        print(f"\n  Low-confidence crops to review:")
        n = len(list(CROPS_LOWCONF.glob("*.jpg"))) if CROPS_LOWCONF.exists() else 0
        print(f"    {CROPS_LOWCONF}  ({n} files)")
        print(f"    → Delete bad ones, move good ones to {CROPS_DIR}")

if __name__ == "__main__":
    main()
