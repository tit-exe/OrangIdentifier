# =============================================================================
# 03_train_yolo_medium.py
# Trains YOLO V2 (yolov8s small) on manually-corrected annotations.
#
# Source: crops.json — crop_x1/y1/x2/y2 boxes corrected by the reviewer
# Model:  yolov8s.pt (small) — better precision than nano with ~2000 images
# Output: models/yolo_v2_medium/best.pt
#
# YOLO format conversion:
#   xc = (crop_x1 + crop_x2) / 2 / img_w
#   yc = (crop_y1 + crop_y2) / 2 / img_h
#   w  = (crop_x2 - crop_x1) / img_w
#   h  = (crop_y2 - crop_y1) / img_h
#   label line: 0 xc yc w h
# =============================================================================

import sys
import shutil
import random
import json
import yaml
import time
from pathlib import Path
from datetime import timedelta
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.config_loader import (
    CROPS_JSON, YOLO_SPLIT_DIR, MODELS_DIR, OUTPUT_DIR, ensure_dirs,
    YOLO_FACE_MARGIN as YOLO_SCALE,
)

RESULTS_DIR = OUTPUT_DIR / "results"

# Hyperparameters — kept inline (tuned for field-condition annotations)
YOLO_EPOCHS    = 200
YOLO_BATCH     = 8
YOLO_PATIENCE  = 30
YOLO_DEGREES   = 15.0
YOLO_TRANSLATE = 0.1
YOLO_FLIPUD    = 0.5
YOLO_FLIPLR    = 0.5
YOLO_HSV_H     = 0.015
YOLO_HSV_S     = 0.4
YOLO_HSV_V     = 0.3

def log_action(script, action, details=None):
    pass  # logging handled by training output

# =============================================================================
# CONFIGURATION
# =============================================================================

CROPS_CACHE  = CROPS_JSON
DATASET_V2   = YOLO_SPLIT_DIR
MODELS_V2    = MODELS_DIR / "yolo_v2_medium"
RUN_NAME     = "face_detector_v2_medium"

TRAIN_RATIO  = 0.80
RANDOM_SEED  = 42

YOLO_MODEL   = "yolov8s.pt"   # small — ~3x more params than nano
EPOCHS       = 200
BATCH        = 8
PATIENCE     = 30
IMGSZ        = 640
WORKERS      = 4

DEGREES      = YOLO_DEGREES
TRANSLATE    = YOLO_TRANSLATE
SCALE        = YOLO_SCALE      # face margin (5%)
FLIPUD       = YOLO_FLIPUD
FLIPLR       = YOLO_FLIPLR
HSV_H        = YOLO_HSV_H
HSV_S        = YOLO_HSV_S
HSV_V        = YOLO_HSV_V

# =============================================================================
# LOAD AND VALIDATE ANNOTATIONS
# =============================================================================

def load_annotations():
    """
    Loads crops.json and returns only valid entries with all required fields.
    Verifies coordinate consistency.
    """
    if not CROPS_CACHE.exists():
        print(f"ERROR: {CROPS_CACHE} not found.")
        sys.exit(1)

    with open(CROPS_CACHE, encoding='utf-8') as f:
        cache = json.load(f)

    annotations = []
    errors      = []

    for key, info in cache.items():
        # Only use known labeled individuals — skip wild/background crops (individu=null)
        if info.get('source_type') == 'wild':
            continue
        if info.get('individu') is None:
            continue
        if info.get('statut') != 'valide':
            continue

        fields = ['crop_x1','crop_y1','crop_x2','crop_y2','img_w','img_h','photo_source']
        if not all(c in info for c in fields):
            errors.append(f"{key}: missing fields")
            continue

        x1 = info['crop_x1']; y1 = info['crop_y1']
        x2 = info['crop_x2']; y2 = info['crop_y2']
        W  = info['img_w'];   H  = info['img_h']

        if x2 <= x1 or y2 <= y1:
            errors.append(f"{key}: invalid box x1={x1} x2={x2}")
            continue

        xc = ((x1 + x2) / 2) / W
        yc = ((y1 + y2) / 2) / H
        w  = (x2 - x1) / W
        h  = (y2 - y1) / H

        if not all(0 < v <= 1 for v in [xc, yc, w, h]):
            xc = max(0.001, min(0.999, xc))
            yc = max(0.001, min(0.999, yc))
            w  = max(0.001, min(0.999, w))
            h  = max(0.001, min(0.999, h))

        photo = Path(info['photo_source'])
        if not photo.exists():
            errors.append(f"{key}: photo source not found")
            continue

        annotations.append({
            'key':      key,
            'individu': info.get('individu', key.split('/')[0]),
            'stem':     info['stem'],
            'photo':    photo,
            'label':    f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}",
        })

    return annotations, errors

# =============================================================================
# DATASET REPORT
# =============================================================================

def report_dataset(annotations, errors):
    per_ind = defaultdict(int)
    for a in annotations:
        per_ind[a['individu']] += 1

    print("=" * 60)
    print("DATASET V2 — MANUALLY CORRECTED ANNOTATIONS")
    print("=" * 60)
    print(f"  Valid annotations : {len(annotations)}")
    print(f"  Ignored entries   : {len(errors)}")
    if errors:
        for e in errors[:5]:
            print(f"    - {e}")

    print(f"\n  {'Individual':<25} {'Images':>7}  {'% of total':>10}")
    print(f"  {'-'*25} {'-'*7}  {'-'*10}")
    total = len(annotations)
    for ind in sorted(per_ind):
        pct  = per_ind[ind] / total * 100
        bar  = "#" * int(pct / 2)
        print(f"  {ind:<25} {per_ind[ind]:>7}  {pct:>8.1f}%  {bar}")
    print(f"  {'TOTAL':<25} {total:>7}")

    counts = list(per_ind.values())
    min_c  = min(counts)
    max_c  = max(counts)
    ratio  = max_c / min_c if min_c > 0 else float('inf')
    print(f"\n  Least represented : {min_c} images")
    print(f"  Most represented  : {max_c} images")
    print(f"  Max/min ratio     : {ratio:.1f}x")

    if ratio > 5:
        print("  WARNING: significant class imbalance.")
        print("  YOLO is robust to imbalance for single-class detection.")
    else:
        print("  Distribution balanced.")

    if len(annotations) < 100:
        print(f"\nERROR: only {len(annotations)} annotations. Minimum: 100.")
        sys.exit(1)

    print("=" * 60)
    return per_ind

# =============================================================================
# BUILD YOLO DATASET
# =============================================================================

def build_dataset(annotations):
    """
    Creates DATASET_V2/train/ and val/ with images and labels.
    Stratified split by individual.
    """
    print("\nBUILDING DATASET")

    if DATASET_V2.exists():
        shutil.rmtree(DATASET_V2)
        print(f"  Previous dataset removed.")

    for split in ['train', 'val']:
        (DATASET_V2 / split / 'images').mkdir(parents=True)
        (DATASET_V2 / split / 'labels').mkdir(parents=True)

    per_ind = defaultdict(list)
    for a in annotations:
        per_ind[a['individu']].append(a)

    random.seed(RANDOM_SEED)
    train_list = []
    val_list   = []

    for ind, items in sorted(per_ind.items()):
        random.shuffle(items)
        n_train = max(1, int(len(items) * TRAIN_RATIO))
        train_list.extend(items[:n_train])
        val_list.extend(items[n_train:])
        print(f"  {ind:<25} train={n_train:>4}  val={len(items)-n_train:>4}")

    print(f"\n  Total train: {len(train_list)}")
    print(f"  Total val  : {len(val_list)}")

    debut = time.time()
    total = len(train_list) + len(val_list)

    def copy_batch(items, split):
        img_dir = DATASET_V2 / split / 'images'
        lbl_dir = DATASET_V2 / split / 'labels'
        for a in items:
            dest_img = img_dir / a['photo'].name
            shutil.copy(a['photo'], dest_img)
            dest_lbl = lbl_dir / (a['photo'].stem + ".txt")
            dest_lbl.write_text(a['label'] + "\n", encoding='utf-8')

    for i, a in enumerate(train_list + val_list, 1):
        pct    = i / total
        fill   = int(40 * pct)
        elapsed = time.time() - debut
        eta    = str(timedelta(seconds=int(elapsed/i*(total-i)))) if i > 0 else "?"
        print(f"\r  Copying [{'='*fill}{'-'*(40-fill)}] {i}/{total}  ETA:{eta}  ",
              end="", flush=True)

    print()

    copy_batch(train_list, 'train')
    copy_batch(val_list,   'val')

    return len(train_list), len(val_list)

# =============================================================================
# data.yaml
# =============================================================================

def generate_yaml():
    yaml_path = DATASET_V2 / "data.yaml"
    config = {
        'path':  str(DATASET_V2),
        'train': 'train/images',
        'val':   'val/images',
        'nc':    1,
        'names': ['face']
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"  data.yaml written: {yaml_path}")
    return yaml_path

# =============================================================================
# TRAINING
# =============================================================================

def train(yaml_path, n_train, n_val):
    from ultralytics import YOLO

    MODELS_V2.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("YOLOV8 V2 TRAINING — SMALL MODEL")
    print("=" * 60)
    print(f"  Base model : {YOLO_MODEL}  (small — better than nano)")
    print(f"  Train      : {n_train} manually-corrected images")
    print(f"  Val        : {n_val} images")
    print(f"  Epochs     : {EPOCHS}  (patience={PATIENCE})")
    print(f"  Batch/Imgsz: {BATCH} / {IMGSZ}")
    print(f"  Augmentation: rotation={DEGREES}° flipud={FLIPUD} fliplr={FLIPLR}")
    print(f"               hsv_s={HSV_S}  scale={SCALE}  translate={TRANSLATE}")
    print(f"  Output     : {MODELS_V2}/best.pt")
    print("=" * 60)

    confirmation = input("\nStart training? (y/n): ").strip().lower()
    if confirmation != 'y':
        print("Cancelled.")
        sys.exit(0)

    print("\nLoading model...")
    model = YOLO(YOLO_MODEL)

    print("Training started...\n")

    model.train(
        data        = str(yaml_path),
        model       = YOLO_MODEL,
        epochs      = EPOCHS,
        imgsz       = IMGSZ,
        batch       = BATCH,
        patience    = PATIENCE,
        workers     = WORKERS,
        degrees     = DEGREES,
        translate   = TRANSLATE,
        scale       = SCALE,
        flipud      = FLIPUD,
        fliplr      = FLIPLR,
        hsv_h       = HSV_H,
        hsv_s       = HSV_S,
        hsv_v       = HSV_V,
        save        = True,
        save_period = 20,
        project     = str(OUTPUT_DIR / 'runs'),
        name        = RUN_NAME,
        exist_ok    = True,
        verbose     = True,
        plots       = True,
        device      = 0,
        resume      = False,
    )

# =============================================================================
# FINAL REPORT
# =============================================================================

def final_report():
    run_dir = OUTPUT_DIR / 'runs' / RUN_NAME

    if not run_dir.exists():
        print("\nRun folder not found.")
        return

    best_pt = run_dir / 'weights' / 'best.pt'
    if best_pt.exists():
        dest = MODELS_V2 / 'best.pt'
        shutil.copy(best_pt, dest)
        print(f"\n  Best model saved: {dest}")
    else:
        print("\n  best.pt not found in run folder.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for png in run_dir.glob("*.png"):
        shutil.copy(png, RESULTS_DIR / f"yolo_v2_{png.name}")
        print(f"  Curve: {RESULTS_DIR}/yolo_v2_{png.name}")

    results_csv = run_dir / 'results.csv'
    if results_csv.exists():
        import csv
        rows = list(csv.DictReader(open(results_csv)))
        if rows:
            best   = max(rows, key=lambda r: float(r.get('metrics/mAP50(B)', 0) or 0))
            map50  = float(best.get('metrics/mAP50(B)', 0))
            prec   = float(best.get('metrics/precision(B)', 0))
            recall = float(best.get('metrics/recall(B)', 0))
            epoch  = best.get('                  epoch', '?')

            print("\n" + "=" * 60)
            print("YOLO V2 RESULTS")
            print("=" * 60)
            print(f"  mAP@0.5       : {map50:.4f}")
            print(f"  Precision     : {prec:.4f}")
            print(f"  Recall        : {recall:.4f}")
            print(f"  Best epoch    : {epoch}")
            print(f"  Total epochs  : {len(rows)}")

            print("\n  Interpretation:")
            if map50 >= 0.95:
                print("  Excellent — ready for field use.")
            elif map50 >= 0.90:
                print("  Very good — usable in the field.")
            elif map50 >= 0.85:
                print("  Good — some difficult cases will be missed.")
            else:
                print("  Insufficient — add more annotations of difficult cases.")

            metrics = {
                "map50": round(map50, 4),
                "precision": round(prec, 4),
                "recall": round(recall, 4),
            }
            log_action("03_train_yolo_medium", "training_complete", metrics)

    print("\n" + "=" * 60)
    print("STEP 3 COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  Run 04_extract_crops.py to extract face crops with the new model.")
    print("  Then run v3_megadesc_arcface_10ind/04_train_arcface.py")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("YOLO V2 TRAINING — MANUALLY CORRECTED ANNOTATIONS")
    print("=" * 60)

    print("\nLoading annotations...")
    annotations, errors = load_annotations()

    report_dataset(annotations, errors)

    n_train, n_val = build_dataset(annotations)

    yaml_path = generate_yaml()

    train(yaml_path, n_train, n_val)

    final_report()
