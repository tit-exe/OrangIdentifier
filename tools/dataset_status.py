# =============================================================================
# dataset_status.py
# Prints the complete project status in a few seconds.
# Run at the start of a session to see where things stand.
# Usage: python tools/dataset_status.py
# =============================================================================

import sys
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.config_loader import (
    REPO_ROOT, YOLO_DATASET_DIR, CROPS_KNOWN_DIR, CROPS_JSON,
    YOLO_V2_PT as YOLO_BEST, MODELS_DIR, OUTPUT_DIR, PHOTOS_DIR,
)

# Derived at runtime so they stay in sync with actual data
INDIVIDUS    = sorted([d.name for d in PHOTOS_DIR.iterdir() if d.is_dir()]) \
               if PHOTOS_DIR.exists() else []
RESULTS_DIR  = OUTPUT_DIR / "results"
VIDEOS_DIR   = REPO_ROOT / "data" / "videos"   # optional — may not exist

DONE_FILE       = YOLO_DATASET_DIR / "done.txt"
CLASSIF_RAW_DIR = CROPS_KNOWN_DIR
RESNET_BEST     = MODELS_DIR / "resnet50_classifier.pt"
SESSION_LOG     = OUTPUT_DIR / "session_log.json"
TARGET_PER_IND  = 30

def sep(title=""):
    if title:
        print(f"\n{'=' * 20} {title} {'=' * 20}")
    else:
        print("=" * 60)

def status(condition):
    return "OK" if condition else "MISSING"

# =============================================================================
# HEADER
# =============================================================================

sep()
print("PROJECT STATUS — Wildlife Individual ID Pipeline")
print(f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}")
print(f"Root : {REPO_ROOT}")
sep()

# =============================================================================
# ANNOTATIONS (done.txt)
# =============================================================================

sep("ANNOTATIONS")

done = set()
if DONE_FILE.exists():
    done = set(s.strip() for s in DONE_FILE.read_text().splitlines() if s.strip())

labels_dir = YOLO_DATASET_DIR / "labels"
valid   = defaultdict(int)
skipped = defaultdict(int)

for stem in done:
    lbl = labels_dir / (stem + ".txt")
    ind = stem.split("_")[0]
    if lbl.exists() and lbl.stat().st_size > 0:
        valid[ind] += 1
    else:
        skipped[ind] += 1

total_valid   = sum(valid.values())
total_skipped = sum(skipped.values())
all_ok = all(valid.get(ind, 0) >= TARGET_PER_IND for ind in INDIVIDUS)

print(f"  Processed total    : {len(done)}")
print(f"  With box (usable)  : {total_valid}")
print(f"  Skips              : {total_skipped}")
print(f"\n  {'Individual':<28} {'Valid':>8} {'Skips':>6}  Status")
print(f"  {'-'*28} {'-'*8} {'-'*6}  {'------'}")

for ind in INDIVIDUS:
    v = valid.get(ind, 0)
    s = skipped.get(ind, 0)
    if v >= TARGET_PER_IND:
        status_str = "OK"
    else:
        status_str = f"need {TARGET_PER_IND - v} more"
    print(f"  {ind:<28} {v:>8} {s:>6}  {status_str}")

print(f"\n  Overall target: {'REACHED' if all_ok else 'IN PROGRESS'}")

# =============================================================================
# EXTRACTED CROPS (face crops for classification)
# =============================================================================

sep("EXTRACTED CROPS")

if CLASSIF_RAW_DIR.exists():
    total_faces = 0
    print(f"  {'Individual':<28} {'Crops':>8}")
    print(f"  {'-'*28} {'-'*8}")
    for ind in INDIVIDUS:
        d  = CLASSIF_RAW_DIR / ind
        nb = len(list(d.glob("*.jpg"))) if d.exists() else 0
        total_faces += nb
        status_str  = "OK" if nb >= 30 else f"few ({nb})"
        print(f"  {ind:<28} {nb:>8}  {status_str}")
    print(f"\n  Total crops: {total_faces}")
else:
    print("  04_extract_crops.py has not been run yet.")

# =============================================================================
# MODELS
# =============================================================================

sep("MODELS")

yolo_ok   = YOLO_BEST.exists()
resnet_ok = RESNET_BEST.exists()

print(f"  YOLO best.pt     : {status(yolo_ok)}")
if yolo_ok:
    mtime = datetime.fromtimestamp(YOLO_BEST.stat().st_mtime)
    print(f"    Trained: {mtime.strftime('%d/%m/%Y %H:%M')}")

print(f"  ResNet best.pth  : {status(resnet_ok)}")
if resnet_ok:
    mtime = datetime.fromtimestamp(RESNET_BEST.stat().st_mtime)
    print(f"    Trained: {mtime.strftime('%d/%m/%Y %H:%M')}")
    meta_path = RESNET_BEST.parent / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"    Test accuracy  : {meta.get('test_accuracy', 'N/A')}")
        print(f"    Num individuals: {meta.get('num_classes', 'N/A')}")
        print(f"    Best epoch     : {meta.get('best_epoch', 'N/A')}")

# =============================================================================
# RESULTS
# =============================================================================

sep("RESULTS")

if RESULTS_DIR.exists():
    pngs = list(RESULTS_DIR.glob("*.png"))
    csvs = list(RESULTS_DIR.glob("*.csv"))
    if pngs or csvs:
        for f in sorted(pngs + csvs):
            print(f"  {f.name}")
    else:
        print("  No results generated yet.")
else:
    print("  Results folder does not exist yet.")

# =============================================================================
# VIDEOS (optional)
# =============================================================================

sep("VIDEOS")

if VIDEOS_DIR.exists():
    videos = list(VIDEOS_DIR.glob("*.mp4"))
    total_size = sum(v.stat().st_size for v in videos) / 1e9
    print(f"  {len(videos)} videos ({total_size:.1f} GB)")
    for v in sorted(videos):
        size = v.stat().st_size / 1e6
        print(f"  {v.name:<60} {size:.0f} MB")
else:
    print(f"  Videos folder not found ({VIDEOS_DIR})")

# =============================================================================
# SESSION LOG (last 5 actions)
# =============================================================================

sep("SESSION LOG (last 5 actions)")

if SESSION_LOG.exists():
    try:
        log = json.loads(SESSION_LOG.read_text(encoding="utf-8"))
        for entry in log[-5:]:
            ts      = entry.get("timestamp", "")[:16].replace("T", " ")
            script  = entry.get("script", "")
            action  = entry.get("action", "")
            details = entry.get("details", {})
            detail_str = "  ".join(f"{k}={v}" for k,v in details.items())
            print(f"  [{ts}] {script} -> {action}  {detail_str}")
    except Exception as e:
        print(f"  Error reading log: {e}")
else:
    print("  No session log yet (session_log.json).")

# =============================================================================
# NEXT SUGGESTED STEP
# =============================================================================

sep("NEXT STEP")

if not yolo_ok:
    print("  -> Run 02_train_yolo_nano.py")
elif not CLASSIF_RAW_DIR.exists() or sum(
        len(list((CLASSIF_RAW_DIR/ind).glob("*.jpg")))
        for ind in INDIVIDUS if (CLASSIF_RAW_DIR/ind).exists()
    ) < 100:
    print("  -> Run 04_extract_crops.py")
elif not resnet_ok:
    print("  -> Run 05_train_resnet50.py")
else:
    print("  -> Pipeline complete! Run export scripts or test open-set.")

sep()
