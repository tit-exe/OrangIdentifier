import os
import sys
import cv2
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.config_loader import PHOTOS_DIR, YOLO_DATASET_DIR, ensure_dirs
from ultralytics import YOLO

# ================================================================
# CONFIGURATION
# ================================================================

OUTPUT_DIR      = YOLO_DATASET_DIR
YOLO_MODEL      = "yolov8n.pt"       # downloaded automatically from ultralytics
CONFIDENCE      = 0.15               # low threshold — catch everything for manual review
ANIMAL_CLASSES  = list(range(15, 30))  # ImageNet animal-like class IDs

# ================================================================
# FOLDER SETUP
# ================================================================

images_dir = OUTPUT_DIR / "images"
labels_dir = OUTPUT_DIR / "labels"
ensure_dirs(images_dir, labels_dir)

print("=" * 60)
print("STEP 1 — AUTO-ANNOTATION")
print("Model : YOLOv8n generic (ImageNet)")
print(f"Source: {PHOTOS_DIR}")
print(f"Output: {OUTPUT_DIR}")
print("=" * 60)

# ================================================================
# LOAD MODEL
# ================================================================

print("\nLoading generic YOLO model...")
model = YOLO(YOLO_MODEL)
print("Model loaded.")

# ================================================================
# PROCESS ALL PHOTOS
# ================================================================

individuals = [d for d in PHOTOS_DIR.iterdir() if d.is_dir()]

total_images    = 0
total_detected  = 0
total_missed    = 0
stats_per_ind   = {}

print(f"\n{len(individuals)} individuals found: {[d.name for d in individuals]}")
print("\nStarting detection...\n")

for ind_dir in sorted(individuals):
    name   = ind_dir.name
    photos = (list(ind_dir.glob("*.jpg")) +
              list(ind_dir.glob("*.jpeg")) +
              list(ind_dir.glob("*.png")))

    detected = 0
    missed   = 0

    print(f"[{name}] — {len(photos)} photos")

    for photo_path in photos:
        total_images += 1

        img = cv2.imread(str(photo_path))
        if img is None:
            print(f"  ERROR reading: {photo_path.name}")
            missed += 1
            continue

        h, w = img.shape[:2]

        results = model.predict(
            source=str(photo_path),
            conf=CONFIDENCE,
            classes=ANIMAL_CLASSES,
            verbose=False
        )

        boxes = results[0].boxes

        # Sanitize filename (no spaces for YOLO)
        safe_name  = name.replace(" ", "_").replace("(", "").replace(")", "")
        stem       = f"{safe_name}_{photo_path.stem}"

        # Copy image to YOLO dataset
        dest_img = images_dir / f"{stem}.jpg"
        shutil.copy(str(photo_path), str(dest_img))

        # Write YOLO label (class 0 = face)
        dest_label = labels_dir / f"{stem}.txt"

        if boxes is not None and len(boxes) > 0:
            with open(dest_label, 'w') as f:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    xc = ((x1 + x2) / 2) / w
                    yc = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            detected += 1
        else:
            # Empty label — keep image for manual annotation in LabelImg
            open(dest_label, 'w').close()
            missed += 1

    stats_per_ind[name] = {'detected': detected, 'missed': missed, 'total': len(photos)}
    total_detected += detected
    total_missed   += missed

    print(f"  -> {detected}/{len(photos)} auto-detections ({missed} need manual annotation)")

# ================================================================
# FINAL REPORT
# ================================================================

print("\n" + "=" * 60)
print("FINAL REPORT")
print("=" * 60)
print(f"Images processed  : {total_images}")
print(f"Auto-detected     : {total_detected} ({100*total_detected/max(total_images,1):.1f}%)")
print(f"Needs correction  : {total_missed}  ({100*total_missed/max(total_images,1):.1f}%)")

print("\nBreakdown by individual:")
print(f"  {'Individual':<35} {'Auto':>6} {'Manual':>8} {'Total':>7}")
print(f"  {'-'*35} {'-'*6} {'-'*8} {'-'*7}")
for name, s in sorted(stats_per_ind.items()):
    print(f"  {name:<35} {s['detected']:>6} {s['missed']:>8} {s['total']:>7}")

print("\n" + "=" * 60)
print("STEP 1 COMPLETE")
print("=" * 60)
print(f"\nYOLO dataset ready:")
print(f"  Images -> {images_dir}")
print(f"  Labels -> {labels_dir}")
print(f"\nNEXT STEPS:")
print(f"  1. Open LabelImg")
print(f"  2. Load image folder : {images_dir}")
print(f"  3. Load label folder : {labels_dir}")
print(f"  4. Review and fix incorrect boxes")
print(f"  5. Run 02_train_yolo_nano.py when ~300 images are corrected")
