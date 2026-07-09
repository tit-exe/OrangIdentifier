"""
V2_3_extract_faces_wild.py
===========================
Orang-outan V2 pipeline


PURPOSE
-------
Runs YOLO v2 on all ~7500 wild images in WILD_ORANGS/raw/
to detect orangutan faces and extract face crops.

Saves two things per detected face:
  1. The crop image (224x224, same format as DATASET_CLASSIFICATION/raw/)
  2. A JSON entry with all bounding box coordinates

The JSON format is EXACTLY the same as boxes_cache.json from V1
so you can reuse the existing review tools (3b_reviser_faces.py, etc.)
and, most importantly, directly feed the annotations to retrain YOLO v3
if needed — with zero extra work.

OUTPUT STRUCTURE
----------------
data/
  WILD_CROPS\
    crops\         <- face crops (224x224 JPEGs)
    boxes_wild.json  <- all detections in boxes_cache format
    stats.json       <- per-image stats
    yolo_dataset\    <- ready-to-use YOLO annotation dataset
      train\
        images\
        labels\
      val\
        images\
        labels\
      data.yaml      <- plug directly into YOLO training

WHY SAVE COORDINATES
--------------------
The JSON lets you:
  - Review and correct bad detections later (same workflow as V1)
  - Retrain YOLO on wild orangutan data (= more robust detector)
  - Audit which images produced crops
  - Reproduce the exact crop from any photo

The YOLO dataset output means you can immediately do:
    yolo train data=data/yolo_dataset_split/data.yaml
    model=yolov8s.pt epochs=100
to get a YOLO v3 trained on wild + zoo data — pushing towards 100% detection.

WHAT IT DOES
------------
1. Loads YOLO v2 model (best.pt)
2. For each image in WILD_ORANGS/raw/ (and subdirs):
   - Runs YOLO detection
   - For each face found (conf > 0.25, NMS IoU 0.45):
     * Adds 15% margin around box (same as V1 — see 3_extract_faces.py)
     * Crops + resizes to 224x224
     * Saves crop as JPEG
     * Logs entry in boxes_wild.json
3. Builds YOLO annotation dataset from all valid detections
4. Generates stats and visualizations

RUN
---
    conda activate orangs
    python data/scripts\V2_3_extract_faces_wild.py
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent))
from common.config_loader import (
    apply_cache_env,
    PHOTOS_DIR, WILD_IMAGES_DIR, CROPS_KNOWN_DIR, CROPS_WILD_DIR, CROPS_JSON,
    MODELS_DIR, OUTPUT_DIR, YOLO_V2_PT,
    V3_PT, V4_PT, UNKNOWN_THRESHOLD,
    ARC_SCALE, ARC_MARGIN, MAX_EPOCHS, PATIENCE, PATIENCE_START,
    LR_BACKBONE, LR_HEAD, BATCH_SIZE, DEVICE, ensure_dirs, to_relative,
)
apply_cache_env()  # sets HF_HOME/TORCH_HOME before any heavy imports


import os
import sys
import json
import shutil
import random
import time
from pathlib import Path
from datetime import datetime




import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Input
YOLO_MODEL_PATH = YOLO_V2_PT


# Output
V2_BASE = OUTPUT_DIR / "v2"
CROPS_DIR = CROPS_WILD_DIR
JSON_PATH = CROPS_JSON
STATS_PATH      = V2_BASE / "WILD_CROPS" / "stats.json"
YOLO_DATASET    = V2_BASE / "WILD_CROPS" / "yolo_dataset"

# YOLO parameters — identical to V1 (see 3_extract_faces.py in the doc)
CONF_THRESHOLD  = 0.25      # minimum confidence to keep a detection
IOU_THRESHOLD   = 0.45      # NMS IoU threshold
MARGIN          = 0.15      # 15% margin around face box — same as V1
CROP_SIZE       = 224       # ResNet50 input size
MAX_FACES_PER_IMAGE = 5     # ignore images with too many detections (noise)

# YOLO dataset split
TRAIN_RATIO     = 0.80      # 80% train, 20% val
RANDOM_SEED     = 42

# Valid image extensions
IMG_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ==============================================================================
# SETUP
# ==============================================================================

for d in [CROPS_DIR, YOLO_DATASET / "train" / "images",
          YOLO_DATASET / "train" / "labels",
          YOLO_DATASET / "val"   / "images",
          YOLO_DATASET / "val"   / "labels"]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  WILD ORANGUTAN FACE EXTRACTOR — V2")
print("  ")
print("=" * 70)
print(f"  YOLO model : {YOLO_MODEL_PATH}")
print(f"  Input dir  : {WILD_IMAGES_DIR}")
print(f"  Crops out  : {CROPS_DIR}")
print(f"  JSON out   : {JSON_PATH}")

# ==============================================================================
# LOAD YOLO MODEL
# ==============================================================================

print("\n" + "─" * 70)
print("  Loading YOLO v2 model...")
print("─" * 70)

if not YOLO_MODEL_PATH.exists():
    print(f"  [ERROR] Model not found: {YOLO_MODEL_PATH}")
    print(f"  Make sure V2_0_setup_and_baseline.py has been run first.")
    sys.exit(1)

model = YOLO(str(YOLO_MODEL_PATH))
print(f"  Model loaded OK")

# ==============================================================================
# COLLECT ALL IMAGES
# ==============================================================================

print("\n" + "─" * 70)
print("  Scanning for images...")
print("─" * 70)

# Find all images — including in subdirectories (iNaturalist, GBIF, etc.)
all_images = []
for ext in IMG_EXTENSIONS:
    all_images.extend(WILD_IMAGES_DIR.rglob(f"*{ext}"))
    all_images.extend(WILD_IMAGES_DIR.rglob(f"*{ext.upper()}"))

# Remove duplicates and sort for reproducibility
all_images = sorted(set(all_images))
print(f"  Found {len(all_images):,} images in {WILD_IMAGES_DIR}")

# ==============================================================================
# LOAD EXISTING JSON (for resumability)
# ==============================================================================

if JSON_PATH.exists():
    with open(JSON_PATH, encoding="utf-8") as f:
        boxes_cache = json.load(f)
    already_done = set(boxes_cache.keys())
    print(f"  Resuming — {len(already_done):,} images already processed")
else:
    boxes_cache   = {}
    already_done  = set()

# ==============================================================================
# MAIN EXTRACTION LOOP
# ==============================================================================

print("\n" + "─" * 70)
print("  Running YOLO detection on all images...")
print("─" * 70)

stats = {
    "total_images":       len(all_images),
    "processed":          0,
    "skipped_already":    len(already_done),
    "no_detection":       0,
    "too_many_faces":     0,
    "crops_saved":        0,
    "errors":             0,
    "per_image":          {}
}

# YOLO annotation entries (for dataset construction later)
yolo_annotations = []   # list of (image_path, boxes_in_yolo_format)

save_interval = 200     # save JSON every N images

pending = [img for img in all_images
           if str(img.relative_to(WILD_IMAGES_DIR)) not in already_done]

print(f"  {len(pending):,} images to process ({len(already_done):,} already done)")

bar = tqdm(pending, desc="  Detecting", unit="img", ncols=90,
           bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")

crops_total = stats["crops_saved"]

for img_idx, img_path in enumerate(bar):
    rel_key = str(img_path.relative_to(WILD_IMAGES_DIR))

    # ── Load image ────────────────────────────────────────────────────────────
    try:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            stats["errors"] += 1
            continue
        img_h, img_w = img_bgr.shape[:2]
    except Exception:
        stats["errors"] += 1
        continue

    # ── Run YOLO ──────────────────────────────────────────────────────────────
    try:
        results = model.predict(
            source=str(img_path),
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
            device="cuda" if __import__("torch").cuda.is_available() else "cpu"
        )
    except Exception as e:
        stats["errors"] += 1
        boxes_cache[rel_key] = {"statut": "erreur", "message": str(e)}
        continue

    detections = results[0].boxes
    n_faces    = len(detections)

    # ── No face detected ──────────────────────────────────────────────────────
    if n_faces == 0:
        stats["no_detection"] += 1
        boxes_cache[rel_key] = {
            "statut":       "no_detection",
            "img_w":        img_w,
            "img_h":        img_h,
            "photo_source": str(img_path),
            "n_faces":      0,
        }
        already_done.add(rel_key)
        stats["processed"] += 1
        continue

    # ── Too many faces (likely noise / group photo) ───────────────────────────
    if n_faces > MAX_FACES_PER_IMAGE:
        stats["too_many_faces"] += 1
        boxes_cache[rel_key] = {
            "statut":       "too_many_faces",
            "img_w":        img_w,
            "img_h":        img_h,
            "photo_source": str(img_path),
            "n_faces":      n_faces,
        }
        already_done.add(rel_key)
        stats["processed"] += 1
        continue

    # ── Process each detected face ────────────────────────────────────────────
    yolo_boxes_for_this_image = []  # for YOLO dataset
    image_entries = []              # all detections from this image

    for face_idx, box in enumerate(detections):
        xyxy       = box.xyxy[0].cpu().numpy()
        confidence = float(box.conf[0].cpu().numpy())

        x1_raw, y1_raw, x2_raw, y2_raw = xyxy

        # Add margin around the face (same 15% as V1 — doc phase 3)
        w_box = x2_raw - x1_raw
        h_box = y2_raw - y1_raw
        margin_x = w_box * MARGIN
        margin_y = h_box * MARGIN

        x1 = max(0,       int(x1_raw - margin_x))
        y1 = max(0,       int(y1_raw - margin_y))
        x2 = min(img_w,   int(x2_raw + margin_x))
        y2 = min(img_h,   int(y2_raw + margin_y))

        if x2 <= x1 or y2 <= y1:
            continue

        # ── Extract and resize crop ───────────────────────────────────────────
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE),
                                  interpolation=cv2.INTER_AREA)

        # ── Save crop ─────────────────────────────────────────────────────────
        # Filename: original_stem + face index + .jpg
        stem      = img_path.stem
        crop_name = f"{stem}_face{face_idx:02d}.jpg"
        crop_dest = CROPS_DIR / crop_name

        # Handle filename collision (different source images, same stem)
        if crop_dest.exists():
            crop_name = f"{stem}_{abs(hash(str(img_path))):08x}_face{face_idx:02d}.jpg"
            crop_dest = CROPS_DIR / crop_name

        cv2.imwrite(str(crop_dest), crop_resized,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

        # ── JSON entry (boxes_cache format, compatible with V1 tools) ─────────
        entry = {
            "statut":          "valide",          # default — can be changed in review
            "crop_x1":         x1,
            "crop_y1":         y1,
            "crop_x2":         x2,
            "crop_y2":         y2,
            "img_w":           img_w,
            "img_h":           img_h,
            "photo_source":    str(img_path),
            "crop_file":       str(crop_dest),
            "confiance_yolo":  round(confidence, 4),
            "face_index":      face_idx,
            "n_faces_image":   n_faces,
            "source_dir":      img_path.parent.name,
        }
        image_entries.append(entry)

        # ── YOLO annotation (for dataset construction) ────────────────────────
        # Convert absolute pixel coords to YOLO normalized format
        # xc, yc, w, h — all normalized by image dimensions
        xc_norm = (x1_raw + x2_raw) / 2 / img_w
        yc_norm = (y1_raw + y2_raw) / 2 / img_h
        w_norm  = (x2_raw - x1_raw) / img_w
        h_norm  = (y2_raw - y1_raw) / img_h

        # Clamp to [0, 1]
        xc_norm = max(0.0, min(1.0, xc_norm))
        yc_norm = max(0.0, min(1.0, yc_norm))
        w_norm  = max(0.0, min(1.0, w_norm))
        h_norm  = max(0.0, min(1.0, h_norm))

        yolo_boxes_for_this_image.append(
            f"0 {xc_norm:.6f} {yc_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
        )

        crops_total += 1
        stats["crops_saved"] += 1

    # ── Save to cache ─────────────────────────────────────────────────────────
    if len(image_entries) == 1:
        # Single face — store directly under the image key
        boxes_cache[rel_key] = image_entries[0]
    elif len(image_entries) > 1:
        # Multiple faces — store as list (same as _a_verifier logic in V1)
        boxes_cache[rel_key] = {
            "statut":       "multi_visage",
            "img_w":        img_w,
            "img_h":        img_h,
            "photo_source": str(img_path),
            "n_faces":      len(image_entries),
            "detections":   image_entries,
        }

    if yolo_boxes_for_this_image:
        yolo_annotations.append((img_path, yolo_boxes_for_this_image))

    already_done.add(rel_key)
    stats["processed"] += 1

    # Update progress bar
    bar.set_postfix({
        "crops": crops_total,
        "skip":  stats["no_detection"],
        "err":   stats["errors"]
    })

    # ── Checkpoint: save JSON periodically ───────────────────────────────────
    if (img_idx + 1) % save_interval == 0:
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(boxes_cache, f, indent=2, ensure_ascii=False)

bar.close()

# Final JSON save
with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(boxes_cache, f, indent=2, ensure_ascii=False)

print(f"\n  Extraction complete.")
print(f"  Total crops saved : {stats['crops_saved']:,}")
print(f"  No detection      : {stats['no_detection']:,} images")
print(f"  Too many faces    : {stats['too_many_faces']:,} images")
print(f"  Errors            : {stats['errors']:,} images")

# ==============================================================================
# BUILD YOLO DATASET (for potential YOLO v3 training)
# ==============================================================================

print("\n" + "─" * 70)
print("  Building YOLO annotation dataset...")
print("─" * 70)

"""
This creates a complete Ultralytics-compatible YOLO dataset from all
valid wild detections. You can use it to retrain YOLO on wild data,
combined with the existing V1 dataset, to push detection towards 100%.

The format is identical to DATASET_YOLO_V2/ from V1.
"""

random.seed(RANDOM_SEED)
random.shuffle(yolo_annotations)

n_train = int(len(yolo_annotations) * TRAIN_RATIO)
splits  = {
    "train": yolo_annotations[:n_train],
    "val":   yolo_annotations[n_train:],
}

for split_name, split_data in splits.items():
    img_dir   = YOLO_DATASET / split_name / "images"
    label_dir = YOLO_DATASET / split_name / "labels"

    bar = tqdm(split_data, desc=f"  Building {split_name:5s}", unit="img", ncols=90)
    for img_path, yolo_lines in bar:
        # Copy image
        dst_img = img_dir / img_path.name
        if dst_img.exists() and img_path.name != dst_img.name:
            # Handle name collision
            stem    = f"{img_path.stem}_{abs(hash(str(img_path))):08x}"
            dst_img = img_dir / f"{stem}{img_path.suffix}"
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)

        # Write label file
        label_path = label_dir / (dst_img.stem + ".txt")
        label_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

    bar.close()

print(f"  Train : {len(splits['train']):,} images")
print(f"  Val   : {len(splits['val']):,} images")

# Write data.yaml
yaml_content = f"""# YOLO dataset — wild orangutan faces
# Generated by V2_3_extract_faces_wild.py
# {datetime.now().strftime('%Y-%m-%d')}
#
# To train YOLO v3 (wild + zoo combined):
#   yolo train model=yolov8s.pt data={YOLO_DATASET}/data.yaml epochs=100 batch=8 imgsz=640

path: {YOLO_DATASET}
train: train/images
val:   val/images

nc: 1
names: ['face']

# Dataset stats:
# Total images with faces : {len(yolo_annotations):,}
# Total crops extracted   : {stats['crops_saved']:,}
# Source                  : {WILD_IMAGES_DIR}
"""

yaml_path = YOLO_DATASET / "data.yaml"
yaml_path.write_text(yaml_content, encoding="utf-8")
print(f"  data.yaml written: {yaml_path}")

# ==============================================================================
# STATS AND REPORT
# ==============================================================================

print("\n" + "─" * 70)
print("  Generating stats...")
print("─" * 70)

n_valid     = sum(1 for v in boxes_cache.values()
                  if isinstance(v, dict) and v.get("statut") == "valide")
n_multi     = sum(1 for v in boxes_cache.values()
                  if isinstance(v, dict) and v.get("statut") == "multi_visage")
n_no_det    = sum(1 for v in boxes_cache.values()
                  if isinstance(v, dict) and v.get("statut") == "no_detection")
n_too_many  = sum(1 for v in boxes_cache.values()
                  if isinstance(v, dict) and v.get("statut") == "too_many_faces")

# Detection rate
n_processed = n_valid + n_multi + n_no_det + n_too_many
det_rate    = (n_valid + n_multi) / max(n_processed, 1)

stats_out = {
    "date":                 datetime.now().isoformat(),
    "total_images_scanned": len(all_images),
    "total_processed":      n_processed,
    "crops_saved":          stats["crops_saved"],
    "detection_rate":       round(det_rate, 4),
    "single_face":          n_valid,
    "multi_face":           n_multi,
    "no_detection":         n_no_det,
    "too_many_faces":       n_too_many,
    "errors":               stats["errors"],
    "yolo_dataset_train":   len(splits["train"]),
    "yolo_dataset_val":     len(splits["val"]),
    "json_path":            str(JSON_PATH),
    "crops_dir":            str(CROPS_DIR),
    "yolo_dataset_path":    str(YOLO_DATASET),
}

with open(STATS_PATH, "w", encoding="utf-8") as f:
    json.dump(stats_out, f, indent=2)

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

report = f"""
{'=' * 70}
  EXTRACTION COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'=' * 70}

  Input:
    {len(all_images):,} images scanned from {WILD_IMAGES_DIR}

  Detection results:
    Single face detected  : {n_valid:,} images
    Multi face detected   : {n_multi:,} images  (all crops saved)
    No face detected      : {n_no_det:,} images  (normal — many tourist photos)
    Too many faces (>{MAX_FACES_PER_IMAGE})  : {n_too_many:,} images  (ignored, likely noise)
    Errors                : {stats['errors']:,} images

    Detection rate        : {det_rate*100:.1f}%
    Total crops extracted : {stats['crops_saved']:,}

  Outputs:
    Crops (224x224 JPEGs) : {CROPS_DIR}
    Bounding box JSON     : {JSON_PATH}
    YOLO dataset          : {YOLO_DATASET}

  YOLO dataset (for potential YOLO v3 training):
    Train : {len(splits['train']):,} images
    Val   : {len(splits['val']):,} images
    Config: {yaml_path}

  NEXT STEPS:
    Option A — Use crops directly for MegaDescriptor domain adaptation:
      Run V2_4_domain_adaptation.py
      Input: {CROPS_DIR} (all {stats['crops_saved']:,} crops)

    Option B — Review + retrain YOLO v3 for better detection:
      Review detections in {JSON_PATH}
      Then: yolo train model=yolov8s.pt data={yaml_path} epochs=100 batch=8
      This will push YOLO detection towards 100% on wild images.
{'=' * 70}
"""

print(report)

# Save report
report_path = V2_BASE / "WILD_CROPS" / "extraction_report.txt"
report_path.write_text(report, encoding="utf-8")
print(f"  Report saved: {report_path}")
