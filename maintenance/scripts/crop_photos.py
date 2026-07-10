r"""
crop_photos.py  --  OrangIdentifier maintenance
===============================================
Step 1 for BOTH options: turn raw photos into clean 224x224 head pictures.

The app and the training both work on small square pictures that contain only
the animal's head, not the whole photo. This script finds the head in every
photo automatically (using the YOLO detector), cuts it out, resizes it to
224x224, and saves it. It also measures how bright and how sharp each crop is,
so you can spot bad pictures later.

You run this first, whether you go with Option A (add individuals) or Option B
(retrain the brain).

How to prepare your photos
--------------------------
Put your photos in the input folder, ONE sub-folder per animal, named exactly
with the animal name. Example:

    new_animals/1_put_raw_photos_here/
        Rosa/
            photo1.jpg
            photo2.jpg
        Yori/
            photo1.jpg
            ...

The crops will appear in new_animals/2_crops_appear_here/ with the same layout.
Nothing is deleted from your original photos.

How to run (in the "orangs" Anaconda environment, on Windows)
-------------------------------------------------------------
    conda activate orangs
    python maintenance/scripts/crop_photos.py

You can run it again any time: photos already processed are skipped, so it
picks up where it stopped.
"""

import os
# HF/Torch caches use the OS default location


import cv2
import json
import shutil
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
REPO = Path(__file__).resolve().parents[2]  # repository root (portable)

# ==============================================================================
# SETTINGS  --  change these only if your folders are somewhere else
# ==============================================================================
INPUT_DIR  = (REPO / "maintenance" / "new_animals" / "1_put_raw_photos_here")
OUTPUT_DIR = (REPO / "maintenance" / "new_animals" / "2_crops_appear_here")
RESULTS    = (REPO / "maintenance" / "new_animals")
JSON_PATH  = (REPO / "maintenance" / "new_animals" / "quality_report.json")

# The head detector (already trained). Do not change unless you moved it.
YOLO_MODEL = (REPO / "models" / "yolo_v2_medium_mAP99.pt")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# FIXED VALUES  --  must match the way the brain was trained. Do not change.
# ==============================================================================
CROP_SIZE   = 224
MARGIN      = 0.05       # extra border around the detected head, as a fraction
CONF_THRESH = 0.25       # minimum YOLO confidence to accept a detection
IOU_THRESH  = 0.45
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
SAVE_EVERY  = 50         # save progress every N photos

# Quality flags (only informative, nothing is rejected automatically)
FLAG_DARK   = 0.20       # brightness below this is flagged as too dark
FLAG_BLUR   = 80.0       # sharpness below this is flagged as blurry

# ==============================================================================
# SMALL JSON HELPERS
# ==============================================================================
def load_json(path):
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        bak = path.with_suffix(".bak")
        if bak.exists():
            try:
                with open(bak, encoding="utf-8") as f:
                    data = json.load(f)
                print("  [WARNING] JSON was corrupt, loaded from backup instead")
                return data
            except Exception:
                pass
    return {}

def save_json(data, path):
    tmp = path.with_suffix(".tmp")
    bak = path.with_suffix(".bak")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        if path.exists():
            shutil.copy2(path, bak)
        tmp.replace(path)
    except Exception as e:
        print(f"  [ERROR] Could not save JSON: {e}")
        if tmp.exists():
            try: tmp.unlink()
            except Exception: pass

# ==============================================================================
# CROPPING AND QUALITY
# ==============================================================================
def extract_crop(img_bgr, x1r, y1r, x2r, y2r):
    """Cut out the head with a small border and resize to CROP_SIZE x CROP_SIZE."""
    h, w = img_bgr.shape[:2]
    bw = x2r - x1r; bh = y2r - y1r
    x1 = max(0, int(x1r - bw * MARGIN))
    y1 = max(0, int(y1r - bh * MARGIN))
    x2 = min(w, int(x2r + bw * MARGIN))
    y2 = min(h, int(y2r + bh * MARGIN))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LANCZOS4)

def quality_metrics(crop_bgr):
    """Measure brightness and sharpness of the 224x224 crop."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean() / 255.0)
    sharpness  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return brightness, sharpness

# ==============================================================================
# QUALITY HISTOGRAMS
# ==============================================================================
def plot_quality_histograms(db, out_path):
    """One brightness and one sharpness histogram per individual."""
    individuals = sorted(set(v["individual"] for v in db.values()
                             if v.get("status") in ("ok", "multi")))
    if not individuals:
        return

    n = len(individuals)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    fig.suptitle("Crop quality per individual", fontsize=13, y=1.01)
    if n == 1:
        axes = [axes]

    COLORS = ["#3b82f6", "#16a34a", "#f97316", "#dc2626", "#9333ea"]

    for row, (ind, ax_row) in enumerate(zip(individuals, axes)):
        color = COLORS[row % len(COLORS)]
        entries = [v for v in db.values()
                   if v.get("individual") == ind and v.get("status") in ("ok", "multi")]

        brightnesses = [v["brightness"] for v in entries]
        sharpnesses  = [v["sharpness"]  for v in entries]

        ax_row[0].hist(brightnesses, bins=30, color=color, alpha=0.80, edgecolor="white")
        ax_row[0].axvline(FLAG_DARK, color="red", lw=2, linestyle="--",
                          label=f"dark limit ({FLAG_DARK:.2f})")
        pct_dark = sum(b < FLAG_DARK for b in brightnesses) / max(len(brightnesses), 1) * 100
        ax_row[0].set_title(f"{ind} - brightness  ({pct_dark:.0f}% too dark)", fontsize=10)
        ax_row[0].set_xlabel("Average crop brightness [0-1]")
        ax_row[0].set_ylabel("Number of crops")
        ax_row[0].set_xlim(0, 1)
        ax_row[0].legend(fontsize=8)
        ax_row[0].grid(alpha=0.3)

        ax_row[1].hist(sharpnesses, bins=30, color=color, alpha=0.80, edgecolor="white")
        ax_row[1].axvline(FLAG_BLUR, color="orange", lw=2, linestyle="--",
                          label=f"blur limit ({FLAG_BLUR:.0f})")
        pct_blur = sum(s < FLAG_BLUR for s in sharpnesses) / max(len(sharpnesses), 1) * 100
        ax_row[1].set_title(f"{ind} - sharpness  ({pct_blur:.0f}% blurry)", fontsize=10)
        ax_row[1].set_xlabel("Sharpness (Laplacian variance)")
        ax_row[1].set_ylabel("Number of crops")
        ax_row[1].set_xlim(left=0)
        ax_row[1].legend(fontsize=8)
        ax_row[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Histograms saved: {out_path.name}")

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 70)
    print("  crop_photos.py  --  cut heads and check quality")
    print(f"  {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 70)

    if not INPUT_DIR.exists():
        print(f"  [ERROR] Input folder not found: {INPUT_DIR}"); return
    if not YOLO_MODEL.exists():
        print(f"  [ERROR] Head detector not found: {YOLO_MODEL}"); return

    individuals = sorted([d.name for d in INPUT_DIR.iterdir() if d.is_dir()])
    if not individuals:
        print(f"  [ERROR] No animal folder inside {INPUT_DIR}")
        print("          Create one folder per animal and put the photos inside.")
        return

    print(f"\n  {len(individuals)} animals found:")
    for name in individuals:
        imgs = [f for f in (INPUT_DIR / name).iterdir() if f.suffix in IMG_EXTS]
        print(f"    {name:<16}: {len(imgs):4d} photos")
        (OUTPUT_DIR / name).mkdir(parents=True, exist_ok=True)

    db = load_json(JSON_PATH)
    already_done = set(db.keys())
    print(f"\n  Already processed: {len(already_done)} (they will be skipped)")

    tasks = []
    for individual in individuals:
        for img_path in sorted((INPUT_DIR / individual).iterdir()):
            if img_path.suffix not in IMG_EXTS:
                continue
            key = f"{individual}/{img_path.stem}"
            if key not in already_done:
                tasks.append((individual, img_path, key))

    print(f"  To process: {len(tasks)} photos\n")
    if not tasks:
        print("  Everything is already cropped. Drawing the quality histograms...")
        plot_quality_histograms(db, RESULTS / "crop_quality_histograms.png")
        _print_summary(db, individuals)
        return

    print("  Loading the head detector (YOLO)...")
    # -- Dependency guard: on a missing package, show the exact install command -
    import importlib.util as _ilu
    _missing = [m for m in ("torch", "ultralytics") if _ilu.find_spec(m) is None]
    if _missing:
        print("\n[STOP] Missing Python package(s): " + ", ".join(_missing))
        print("Install everything in the 'orangs' environment (Anaconda Prompt):\n")
        print("  conda activate orangs")
        print("  pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 --index-url https://download.pytorch.org/whl/cu124")
        print("  pip install timm==0.9.16 ultralytics==8.2.0 opencv-python==4.9.0.80 Pillow==10.3.0 numpy==1.26.4 scikit-learn==1.4.2 huggingface_hub==0.23.2 tqdm==4.66.4 rich pywin32")
        print("\nFull guide: maintenance/00_first_time_setup/1_install_training_tools.md")
        raise SystemExit(1)
    import torch
    from ultralytics import YOLO
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = YOLO(str(YOLO_MODEL))
    print(f"  Detector ready on {device}\n")

    n_ok = n_multi = n_no_det = n_dark = n_blur = n_error = 0

    bar = tqdm(tasks, desc="  Cropping", unit="img", ncols=90,
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for i, (individual, img_path, key) in enumerate(bar):
        stem     = img_path.stem
        crop_dst = OUTPUT_DIR / individual / f"{stem}.jpg"

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            db[key] = {"individual": individual, "stem": stem,
                       "status": "read_error", "crop_file": None}
            n_error += 1
            continue

        h, w = img_bgr.shape[:2]

        try:
            results    = model.predict(source=str(img_path),
                                        conf=CONF_THRESH, iou=IOU_THRESH,
                                        verbose=False, device=device)
            detections = results[0].boxes
        except Exception as e:
            db[key] = {"individual": individual, "stem": stem,
                       "status": "yolo_error", "crop_file": None, "error": str(e)}
            n_error += 1
            continue

        n_faces = len(detections) if detections is not None else 0

        if n_faces == 0:
            db[key] = {"individual": individual, "stem": stem,
                       "img_w": w, "img_h": h, "n_faces": 0,
                       "status": "no_detection", "crop_file": None}
            n_no_det += 1
            bar.set_postfix({"ok": n_ok, "no_det": n_no_det, "err": n_error})
            continue

        # Keep the most confident detection.
        best_idx = int(detections.conf.argmax())
        x1r, y1r, x2r, y2r = detections[best_idx].xyxy[0].cpu().numpy()
        yolo_conf = float(detections[best_idx].conf[0].cpu().numpy())

        crop = extract_crop(img_bgr, x1r, y1r, x2r, y2r)
        if crop is None:
            db[key] = {"individual": individual, "stem": stem,
                       "status": "crop_error", "crop_file": None}
            n_error += 1
            continue

        brightness, sharpness = quality_metrics(crop)

        flags = []
        if brightness < FLAG_DARK:
            flags.append("dark"); n_dark += 1
        if sharpness < FLAG_BLUR:
            flags.append("blurry"); n_blur += 1

        cv2.imwrite(str(crop_dst), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

        status = "multi" if n_faces > 1 else "ok"
        if n_faces > 1: n_multi += 1
        else:           n_ok    += 1

        db[key] = {
            "individual":   individual,
            "stem":         stem,
            "photo_source": str(img_path),
            "crop_file":    str(crop_dst),
            "img_w": w, "img_h": h,
            "yolo_conf":    round(yolo_conf, 4),
            "n_faces":      n_faces,
            "brightness":   round(brightness, 4),
            "sharpness":    round(sharpness,  2),
            "flags":        flags,
            "status":       status,
            "ts":           datetime.now().isoformat(),
        }

        bar.set_postfix({"ok": n_ok, "multi": n_multi,
                          "no_det": n_no_det, "dark": n_dark, "err": n_error})

        if (i + 1) % SAVE_EVERY == 0:
            save_json(db, JSON_PATH)

    bar.close()
    save_json(db, JSON_PATH)

    plot_quality_histograms(db, RESULTS / "crop_quality_histograms.png")

    _print_summary(db, individuals)
    print(f"""
{'=' * 70}
  CROPPING DONE  --  {datetime.now():%Y-%m-%d %H:%M}
{'=' * 70}
  Total processed : {len(tasks)}
  Good crops      : {n_ok}
  Several heads   : {n_multi}   (kept the best one)
  No head found   : {n_no_det}
  Dark flag       : {n_dark}   (brightness < {FLAG_DARK})
  Blurry flag     : {n_blur}    (sharpness  < {FLAG_BLUR})
  Errors          : {n_error}

  Crops    -> {OUTPUT_DIR}
  Report   -> {JSON_PATH.name}
  Graphs   -> crop_quality_histograms.png

  NEXT STEP:
    Option A (add individuals) -> run build_gallery.py
    Option B (retrain brain)   -> check the crops, then run train_brain.py
{'=' * 70}
""")

def _print_summary(db, individuals):
    print(f"\n  {'Animal':<16} {'Good':>7} {'No head':>8} {'Dark':>6} {'Blurry':>7}")
    print(f"  {'-'*50}")
    for ind in individuals:
        entries = [v for v in db.values() if v.get("individual") == ind]
        n_ok_i    = sum(1 for v in entries if v.get("status") in ("ok", "multi"))
        n_nodet_i = sum(1 for v in entries if v.get("status") == "no_detection")
        n_dark_i  = sum(1 for v in entries if "dark"   in v.get("flags", []))
        n_blur_i  = sum(1 for v in entries if "blurry" in v.get("flags", []))
        warn = "  <- check" if n_dark_i / max(n_ok_i, 1) > 0.5 else ""
        print(f"  {ind:<16} {n_ok_i:>7} {n_nodet_i:>8} {n_dark_i:>6} {n_blur_i:>7}{warn}")
    print()

if __name__ == "__main__":
    main()
