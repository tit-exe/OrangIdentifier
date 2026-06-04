# =============================================================================
# 02_train_yolo_nano.py
# Trains a YOLOv8-nano face detector on annotated photos.
# Run this after 01_auto_annotate.py + manual correction in LabelImg.
# =============================================================================

import os
import sys
import shutil
import random
import yaml
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.config_loader import (
    YOLO_DATASET_DIR, YOLO_SPLIT_DIR, MODELS_DIR, OUTPUT_DIR, ensure_dirs
)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_YOLO = YOLO_DATASET_DIR
SPLIT_DIR    = YOLO_SPLIT_DIR
MODELS_DIR   = MODELS_DIR / "yolo_v1_nano"
RESULTS_DIR  = OUTPUT_DIR / "results"

TRAIN_RATIO  = 0.80
RANDOM_SEED  = 42

# YOLO hyperparameters
EPOCHS       = 100
IMGSZ        = 640
BATCH        = 8       # reduce to 4 if < 4 GB VRAM
PATIENCE     = 20

# Augmentation — tuned for variable animal poses (upside-down, side angles)
DEGREES      = 15.0
TRANSLATE    = 0.1
SCALE        = 0.5
FLIPUD       = 0.5    # vertical flip — useful for animals in unusual positions
FLIPLR       = 0.5
HSV_H        = 0.015
HSV_S        = 0.4    # keep conservative to preserve coat/fur color as a feature
HSV_V        = 0.3

# =============================================================================
# DATASET VERIFICATION
# =============================================================================

def get_annotated_images():
    """
    Reads done.txt to get only user-reviewed images.
    done.txt contains one stem per line, produced by the annotator and clean_annotations.py.
    """
    images_dir = DATASET_YOLO / "images"
    done_file  = DATASET_YOLO / "done.txt"

    if not done_file.exists():
        print("  ERROR: done.txt not found.")
        print("  Run clean_annotations.py in tools/ first.")
        sys.exit(1)

    stems = set(s.strip() for s in done_file.read_text().splitlines() if s.strip())
    images = [images_dir / (stem + ".jpg") for stem in stems
              if (images_dir / (stem + ".jpg")).exists()]

    print(f"  Images read from done.txt: {len(images)}")
    return images


def verify_dataset():
    images_dir = DATASET_YOLO / "images"
    labels_dir = DATASET_YOLO / "labels"

    if not images_dir.exists():
        print(f"ERROR: images folder not found: {images_dir}")
        print("Run 01_auto_annotate.py first.")
        sys.exit(1)

    if not labels_dir.exists():
        print(f"ERROR: labels folder not found: {labels_dir}")
        sys.exit(1)

    all_images = get_annotated_images()
    valid_pairs = []
    empty_labels = []
    missing_labels = []
    corrupt_labels = []

    for img in all_images:
        lbl = labels_dir / (img.stem + ".txt")
        if not lbl.exists():
            missing_labels.append(img.name)
            continue
        content = lbl.read_text().strip()
        if not content:
            empty_labels.append(img.name)
            continue
        ok = True
        for line in content.splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                ok = False; break
            try:
                vals = list(map(float, parts))
                if any(v < 0 or v > 1 for v in vals[1:]):
                    ok = False; break
            except ValueError:
                ok = False; break
        if ok:
            valid_pairs.append(img)
        else:
            corrupt_labels.append(img.name)

    print("=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)
    print(f"  Total images          : {len(all_images)}")
    print(f"  Valid pairs           : {len(valid_pairs)}")
    print(f"  Skipped (empty label) : {len(empty_labels)}")
    print(f"  Missing labels        : {len(missing_labels)}")
    print(f"  Corrupt labels        : {len(corrupt_labels)}")

    if len(valid_pairs) < 50:
        print(f"\nERROR: only {len(valid_pairs)} valid images.")
        print("Need at least 50 annotated images to train YOLO.")
        sys.exit(1)

    print("\n  Distribution by individual:")
    print(f"  {'Individual':<30} {'Images':>7} {'Boxes':>7}")
    print(f"  {'-'*30} {'-'*7} {'-'*7}")

    per_individual = defaultdict(lambda: {'images': 0, 'boxes': 0})
    for img in valid_pairs:
        ind = img.stem.split("_")[0]
        lbl = labels_dir / (img.stem + ".txt")
        n_boxes = len([l for l in lbl.read_text().splitlines() if l.strip()])
        per_individual[ind]['images'] += 1
        per_individual[ind]['boxes'] += n_boxes

    total_boxes = 0
    for ind, stats in sorted(per_individual.items()):
        print(f"  {ind:<30} {stats['images']:>7} {stats['boxes']:>7}")
        total_boxes += stats['boxes']
    print(f"  {'TOTAL':<30} {len(valid_pairs):>7} {total_boxes:>7}")
    print(f"\n  Average boxes/image: {total_boxes/len(valid_pairs):.2f}")
    print("=" * 60)

    return valid_pairs

# =============================================================================
# TRAIN / VAL SPLIT — stratified by individual
# =============================================================================

def make_split(valid_pairs):
    print("\nTRAIN / VAL SPLIT")
    print(f"  Ratio : {int(TRAIN_RATIO*100)}% train / {int((1-TRAIN_RATIO)*100)}% val")
    print(f"  Seed  : {RANDOM_SEED}")

    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)
        print(f"  Previous split removed: {SPLIT_DIR}")

    for split in ['train', 'val']:
        (SPLIT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (SPLIT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

    labels_dir = DATASET_YOLO / "labels"

    per_individual = defaultdict(list)
    for img in valid_pairs:
        ind = img.stem.split("_")[0]
        per_individual[ind].append(img)

    random.seed(RANDOM_SEED)
    train_imgs, val_imgs = [], []

    for ind, imgs in per_individual.items():
        random.shuffle(imgs)
        n_train = max(1, int(len(imgs) * TRAIN_RATIO))
        train_imgs.extend(imgs[:n_train])
        val_imgs.extend(imgs[n_train:])

    def copy_split(imgs, split):
        for img in imgs:
            lbl = labels_dir / (img.stem + ".txt")
            shutil.copy(img, SPLIT_DIR / split / 'images' / img.name)
            shutil.copy(lbl, SPLIT_DIR / split / 'labels' / (img.stem + ".txt"))

    copy_split(train_imgs, 'train')
    copy_split(val_imgs,   'val')

    print(f"  Train: {len(train_imgs)} images")
    print(f"  Val  : {len(val_imgs)} images")

    return len(train_imgs), len(val_imgs)

# =============================================================================
# data.yaml
# =============================================================================

def generate_yaml():
    yaml_path = SPLIT_DIR / "data.yaml"

    config = {
        'path':  str(SPLIT_DIR),
        'train': 'train/images',
        'val':   'val/images',
        'nc':    1,
        'names': ['face']   # generic — adapt to your species if needed
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"\n  data.yaml written: {yaml_path}")
    return yaml_path

# =============================================================================
# TRAINING
# =============================================================================

def train(yaml_path, n_train, n_val):
    from ultralytics import YOLO

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("YOLOV8 TRAINING — V1 NANO")
    print("=" * 60)
    print(f"  Base model : yolov8n.pt (nano, COCO pretrained)")
    print(f"  Train      : {n_train} images")
    print(f"  Val        : {n_val} images")
    print(f"  Epochs     : {EPOCHS}  (early stopping patience={PATIENCE})")
    print(f"  Batch      : {BATCH}")
    print(f"  Img size   : {IMGSZ}")
    print(f"  Rotation   : ±{DEGREES}°  FlipUD={FLIPUD}  FlipLR={FLIPLR}")
    print(f"  HSV_S      : {HSV_S}  (conservative — preserves coat color)")
    print("=" * 60)

    confirmation = input("\nStart training? (y/n): ").strip().lower()
    if confirmation != 'y':
        print("Cancelled.")
        sys.exit(0)

    print("\nLoading model...")
    model = YOLO('yolov8n.pt')

    print("Training started...\n")

    model.train(
        data        = str(yaml_path),
        model       = 'yolov8n.pt',
        epochs      = EPOCHS,
        imgsz       = IMGSZ,
        batch       = BATCH,
        patience    = PATIENCE,
        degrees     = DEGREES,
        translate   = TRANSLATE,
        scale       = SCALE,
        flipud      = FLIPUD,
        fliplr      = FLIPLR,
        hsv_h       = HSV_H,
        hsv_s       = HSV_S,
        hsv_v       = HSV_V,
        save        = True,
        save_period = 10,
        project     = str(OUTPUT_DIR / 'runs'),
        name        = 'face_detector_v1_nano',
        exist_ok    = True,
        verbose     = True,
        plots       = True,
        device      = 0,
    )

# =============================================================================
# FINAL REPORT
# =============================================================================

def final_report():
    run_dir = OUTPUT_DIR / 'runs' / 'face_detector_v1_nano'

    if not run_dir.exists():
        print("\nRun folder not found.")
        return

    best_pt = run_dir / 'weights' / 'best.pt'
    if best_pt.exists():
        dest = MODELS_DIR / 'best.pt'
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_pt, dest)
        print(f"\n  Best model saved: {dest}")
    else:
        print("\n  best.pt not found in run folder.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for png in run_dir.glob("*.png"):
        shutil.copy(png, RESULTS_DIR / f"yolo_v1_{png.name}")
        print(f"  Curve saved: {RESULTS_DIR}/yolo_v1_{png.name}")

    results_csv = run_dir / 'results.csv'
    if results_csv.exists():
        import csv
        rows = list(csv.DictReader(open(results_csv)))
        if rows:
            best = max(rows, key=lambda r: float(r.get('metrics/mAP50(B)', 0) or 0))
            print("\n" + "=" * 60)
            print("FINAL RESULTS")
            print("=" * 60)
            try:
                map50  = float(best.get('metrics/mAP50(B)', 0))
                prec   = float(best.get('metrics/precision(B)', 0))
                recall = float(best.get('metrics/recall(B)', 0))
                print(f"  Best mAP@0.5 : {map50:.4f}")
                print(f"  Precision    : {prec:.4f}")
                print(f"  Recall       : {recall:.4f}")
                print(f"  Total epochs : {len(rows)}")

                print("\n  Interpretation:")
                if map50 >= 0.95:
                    print("  Excellent — ready for face extraction (step 3)")
                elif map50 >= 0.85:
                    print("  Good — annotate 100 more images to improve further")
                elif map50 >= 0.70:
                    print("  Moderate — annotate 200 more images before proceeding")
                else:
                    print("  Insufficient — check annotation quality (boxes too large?)")

            except (ValueError, TypeError):
                print("  Metrics unreadable — check results.csv manually.")

    print("\n" + "=" * 60)
    print("STEP 2 COMPLETE")
    print("=" * 60)
    print("\nNext step: run 04_extract_crops.py")
    print("It will use the trained YOLO to extract face crops from all photos.")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 2 — YOLOV8 FACE DETECTOR TRAINING (V1 NANO)")
    print("=" * 60)

    valid_pairs = verify_dataset()
    n_train, n_val = make_split(valid_pairs)
    yaml_path = generate_yaml()
    train(yaml_path, n_train, n_val)
    final_report()
