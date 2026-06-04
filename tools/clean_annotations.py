# =============================================================================
# clean_annotations.py
# Reads done.txt, removes entries with no bounding box,
# and prints complete stats. No manual action required.
# =============================================================================

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.config_loader import YOLO_DATASET_DIR

LABELS_DIR = YOLO_DATASET_DIR / "labels"
DONE_FILE  = YOLO_DATASET_DIR / "done.txt"

done_path = Path(DONE_FILE)
if not done_path.exists():
    print("ERROR: done.txt not found.")
    exit(1)

original_stems = [s.strip() for s in done_path.read_text().splitlines() if s.strip()]
labels_dir = Path(LABELS_DIR)

with_box    = []
no_box      = []
no_label    = []

for stem in original_stems:
    lbl = labels_dir / (stem + ".txt")
    if not lbl.exists():
        no_label.append(stem)
    elif lbl.stat().st_size == 0:
        no_box.append(stem)
    else:
        valid_lines = []
        for line in lbl.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    vals = list(map(float, parts))
                    if all(0 <= v <= 1 for v in vals[1:]):
                        valid_lines.append(line)
                except ValueError:
                    pass
        if valid_lines:
            with_box.append(stem)
        else:
            no_box.append(stem)

valid_per_ind  = defaultdict(int)
removed_per_ind = defaultdict(int)

for stem in with_box:
    valid_per_ind[stem.split("_")[0]] += 1
for stem in no_box + no_label:
    removed_per_ind[stem.split("_")[0]] += 1

print("=" * 60)
print("ANNOTATION CLEANUP — done.txt")
print("=" * 60)
print(f"  Original entries        : {len(original_stems)}")
print(f"  With box (kept)         : {len(with_box)}")
print(f"  Without box (removed)   : {len(no_box)}")
print(f"  Missing label (removed) : {len(no_label)}")
print(f"  Total removed           : {len(no_box) + len(no_label)}")

print("\nValid distribution per individual after cleanup:")
print(f"  {'Individual':<30} {'Valid':>8} {'Removed':>8}")
print(f"  {'-'*30} {'-'*8} {'-'*8}")
all_inds = sorted(set(list(valid_per_ind.keys()) + list(removed_per_ind.keys())))
for ind in all_inds:
    v = valid_per_ind.get(ind, 0)
    r = removed_per_ind.get(ind, 0)
    status = "OK" if v >= 30 else f"need {30 - v} more"
    print(f"  {ind:<30} {v:>8} {r:>8}  {status}")

print(f"\n  Total valid (final): {len(with_box)}")

# Write cleaned done.txt
done_path.write_text("\n".join(with_box) + "\n")
print(f"\ndone.txt updated: {len(with_box)} entries kept.")
print("=" * 60)
