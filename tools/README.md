# tools/

Helper scripts for the annotation phase (step 1 of the pipeline).

All scripts read paths from `common/config_loader.py`. No hardcoded paths.

---

## When to use each tool

### Annotation workflow (in order)

```
01  init_annotations.py       ← run once before starting
02  annotate_keyboard.py      ← run every annotation session
03  clean_annotations.py      ← run after each session to clean up
```

Then feed the result into `v1_yolo_nano_resnet50/03_train_yolo_medium.py`.

---

### `init_annotations.py`

**When:** Once, before you start annotating for the first time.

**What it does:** Scans `data/yolo_dataset/images/`, builds a priority-ordered list (individuals with fewer annotations first), and writes the first 500 image stems to `done.txt`.

**Run:** `python tools/init_annotations.py`

---

### `annotate_keyboard.py`

**When:** Every annotation session. This is the main annotation tool.

**What it does:** Opens a GUI showing one image at a time. YOLO pre-filled the boxes, you verify and correct them.

| Key | Action |
|-----|--------|
| `D` | Validate (box is correct) |
| `S` | Skip (no face or unusable) |
| `A` / `←` | Go back to previous |
| `Z` | Reset box to original |
| Right-click | Delete that box |
| `Esc` | Quit |

Moves automatically to the next individual once the target (30 valid images) is reached.

**Run:** `python tools/annotate_keyboard.py`

---

### `clean_annotations.py`

**When:** After an annotation session, before training.

**What it does:** Reads `done.txt`, removes entries with empty or invalid labels (skipped images), rewrites `done.txt` with only valid entries, and prints per-individual stats.

**Run:** `python tools/clean_annotations.py`

---

### `dataset_status.py`

**When:** Anytime. Good to run at the start of a work session.

**What it does:** Prints a full dashboard:
- Annotation progress per individual (valid count vs target)
- Extracted crops per individual
- Which models are trained and when
- Session log (last 5 actions)
- Suggested next step

**Run:** `python tools/dataset_status.py`
