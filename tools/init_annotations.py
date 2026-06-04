# =============================================================================
# init_annotations.py
# Initialises done.txt with the first N images (from the original list)
# that have at least one bounding box (non-empty label).
# =============================================================================

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.config_loader import YOLO_DATASET_DIR

IMAGES_DIR = YOLO_DATASET_DIR / "images"
LABELS_DIR = YOLO_DATASET_DIR / "labels"
DONE_FILE  = YOLO_DATASET_DIR / "done.txt"

TARGET_PER_IND = 30   # minimum valid images per individual
BATCH_SIZE     = 500  # total images to initialise

# Build ordered list (prioritise individuals below target)
per_ind = defaultdict(list)
for p in sorted(Path(IMAGES_DIR).glob("*.jpg")):
    per_ind[p.stem.split("_")[0]].append(p)

priority, rest = [], []
for k, v in sorted(per_ind.items()):
    priority.extend(v[:TARGET_PER_IND])
    rest.extend(v[TARGET_PER_IND:])

ordered   = priority + rest
to_process = ordered[:BATCH_SIZE]

print("=" * 55)
print("INITIALISE done.txt")
print("=" * 55)
print(f"First image : {to_process[0].stem}")
print(f"Last image  : {to_process[-1].stem}")

labels_dir = Path(LABELS_DIR)
with_box   = []
no_box     = []

for img in to_process:
    lbl = labels_dir / (img.stem + ".txt")
    if lbl.exists() and lbl.stat().st_size > 0:
        with_box.append(img.stem)
    else:
        no_box.append(img.stem)

print(f"\nAmong the {BATCH_SIZE} images to process:")
print(f"  With box (valid) : {len(with_box)}")
print(f"  Without box (skip): {len(no_box)}")

print("\nValid distribution per individual:")
per_ind_v = defaultdict(int)
for stem in with_box:
    per_ind_v[stem.split("_")[0]] += 1

for ind, nb in sorted(per_ind_v.items()):
    status = "OK" if nb >= TARGET_PER_IND else f"need {TARGET_PER_IND - nb} more"
    print(f"  {ind:<30} {nb:>4}  {status}")

done_path = Path(DONE_FILE)
if done_path.exists():
    print(f"\ndone.txt already exists.")
    confirm = input("Overwrite? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        exit(0)

with open(done_path, 'w') as f:
    for img in to_process:
        f.write(img.stem + "\n")

print(f"\ndone.txt written: {BATCH_SIZE} stems.")
print("Now run the annotator (annotate_boxes.py or annotate_keyboard.py).")
print("=" * 55)
