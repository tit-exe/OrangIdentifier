# NEW_1_extract_crops.py
# CNRS IPHC Strasbourg — Orang-outan V2 pipeline
# Author: Titouane
#
# Runs YOLO v2 on all images in NEW_ORANGS/<IndividualName>/
# Saves 224x224 crops to NEW_ORANGS_CROPS/<IndividualName>/
# Saves all bounding boxes to boxes_new_orangs.json
#
# RUN:
#   conda activate orangs
#   python D:\OrangIdentifier\V2\scripts\NEW_1_extract_crops.py

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

os.environ["HF_HOME"]    = r"D:\HuggingFaceCache"
os.environ["TORCH_HOME"] = r"D:\TorchCache"

import cv2
import numpy as np
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================

INPUT_DIR   = Path(r"D:\OrangIdentifier\V2\NEW_ORANGS")
OUTPUT_DIR  = Path(r"D:\OrangIdentifier\V2\NEW_ORANGS_CROPS")
JSON_PATH   = OUTPUT_DIR / "boxes_new_orangs.json"
YOLO_MODEL  = Path(r"D:\OrangIdentifier\V2\MODELS\yolo_v2.pt")

CROP_SIZE   = 224
MARGIN      = 0.05
CONF_THRESH = 0.25
IOU_THRESH  = 0.45
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}

# ==============================================================================
# JSON — atomic save + self-healing from backup
# ==============================================================================

def load_json(path):
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        pass
    bak = path.with_suffix(".bak")
    if bak.exists():
        try:
            with open(bak, encoding="utf-8") as f:
                data = json.load(f)
            print(f"  [WARN] Loaded from backup")
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
        print(f"  [ERROR] JSON save failed: {e}")
        if tmp.exists():
            try: tmp.unlink()
            except: pass

# ==============================================================================
# CROP EXTRACTION
# ==============================================================================

def extract_crop(img, x1r, y1r, x2r, y2r):
    h, w  = img.shape[:2]
    bw    = x2r - x1r
    bh    = y2r - y1r
    x1    = max(0, int(x1r - bw * MARGIN))
    y1    = max(0, int(y1r - bh * MARGIN))
    x2    = min(w, int(x2r + bw * MARGIN))
    y2    = min(h, int(y2r + bh * MARGIN))
    if x2 <= x1 or y2 <= y1:
        return None, x1, y1, x2, y2
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None, x1, y1, x2, y2
    resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE),
                         interpolation=cv2.INTER_LANCZOS4)
    return resized, x1, y1, x2, y2

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("  NEW ORANGS — Face Extraction")
    print("  CNRS IPHC Strasbourg")
    print("=" * 70)

    if not INPUT_DIR.exists():
        print(f"  [ERROR] Not found: {INPUT_DIR}")
        sys.exit(1)
    if not YOLO_MODEL.exists():
        print(f"  [ERROR] YOLO model not found: {YOLO_MODEL}")
        sys.exit(1)

    individuals = sorted([d.name for d in INPUT_DIR.iterdir() if d.is_dir()])
    if not individuals:
        print(f"  [ERROR] No subfolders found in {INPUT_DIR}")
        sys.exit(1)

    print(f"\n  Found {len(individuals)} individuals:")
    for name in individuals:
        imgs = [f for f in (INPUT_DIR/name).iterdir() if f.suffix in IMG_EXTS]
        print(f"    {name:<22}: {len(imgs)} photos")
        (OUTPUT_DIR / name).mkdir(parents=True, exist_ok=True)

    db           = load_json(JSON_PATH)
    already_done = set(db.keys())
    print(f"\n  Already processed: {len(already_done)} (will skip)")

    tasks = []
    for individu in individuals:
        for img_path in sorted((INPUT_DIR / individu).iterdir()):
            if img_path.suffix not in IMG_EXTS:
                continue
            key = f"{individu}/{img_path.stem}"
            if key not in already_done:
                tasks.append((individu, img_path, key))

    print(f"  To process: {len(tasks)} images\n")

    if not tasks:
        print("  All done. Run NEW_2_review_crops.py to review.")
        return

    print("  Loading YOLO v2...")
    from ultralytics import YOLO
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = YOLO(str(YOLO_MODEL))
    print(f"  YOLO loaded on {device}\n")

    n_auto = n_multi = n_no_det = n_error = 0
    SAVE_EVERY = 50

    bar = tqdm(tasks, desc="  Extracting", unit="img", ncols=85,
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}] {postfix}")

    for i, (individu, img_path, key) in enumerate(bar):
        stem     = img_path.stem
        crop_dst = OUTPUT_DIR / individu / f"{stem}.jpg"

        img = cv2.imread(str(img_path))
        if img is None:
            db[key] = {"individu": individu, "stem": stem,
                       "photo_source": str(img_path),
                       "statut": "erreur", "crop_file": None}
            n_error += 1
            continue

        h, w = img.shape[:2]

        try:
            results    = model.predict(source=str(img_path),
                                        conf=CONF_THRESH, iou=IOU_THRESH,
                                        verbose=False, device=device)
            detections = results[0].boxes
        except Exception as e:
            db[key] = {"individu": individu, "stem": stem,
                       "photo_source": str(img_path),
                       "statut": "erreur", "crop_file": None,
                       "error": str(e)}
            n_error += 1
            continue

        n_faces = len(detections) if detections is not None else 0

        if n_faces == 0:
            db[key] = {"individu": individu, "stem": stem,
                       "photo_source": str(img_path), "crop_file": None,
                       "img_w": w, "img_h": h, "n_faces": 0,
                       "statut": "no_detection"}
            n_no_det += 1
            bar.set_postfix({"ok": n_auto, "no_det": n_no_det, "err": n_error})
            continue

        best    = detections[0]
        x1r, y1r, x2r, y2r = best.xyxy[0].cpu().numpy()
        conf    = float(best.conf[0].cpu().numpy())

        crop_img, cx1, cy1, cx2, cy2 = extract_crop(img, x1r, y1r, x2r, y2r)

        if crop_img is None:
            db[key] = {"individu": individu, "stem": stem,
                       "photo_source": str(img_path), "crop_file": None,
                       "statut": "erreur_crop"}
            n_error += 1
            continue

        cv2.imwrite(str(crop_dst), crop_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        statut = "auto" if n_faces == 1 else "multi"
        if n_faces > 1: n_multi += 1
        else:           n_auto  += 1

        db[key] = {
            "individu":    individu,
            "stem":        stem,
            "photo_source": str(img_path),
            "crop_file":   str(crop_dst),
            "img_w": w, "img_h": h,
            "crop_x1": cx1, "crop_y1": cy1,
            "crop_x2": cx2, "crop_y2": cy2,
            "yolo_x1": int(x1r), "yolo_y1": int(y1r),
            "yolo_x2": int(x2r), "yolo_y2": int(y2r),
            "yolo_conf": round(conf, 4),
            "n_faces": n_faces,
            "statut": statut,
            "ts": datetime.now().isoformat(),
        }

        bar.set_postfix({"ok": n_auto, "multi": n_multi,
                          "no_det": n_no_det, "err": n_error})

        if (i + 1) % SAVE_EVERY == 0:
            save_json(db, JSON_PATH)

    bar.close()
    save_json(db, JSON_PATH)

    print(f"""
{'=' * 70}
  EXTRACTION COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'=' * 70}
  Processed : {len(tasks)}
  Auto ok   : {n_auto}
  Multi     : {n_multi}  (best face kept, check with reviewer)
  No detect : {n_no_det}
  Errors    : {n_error}

  Crops : {OUTPUT_DIR}
  JSON  : {JSON_PATH}

  NEXT: python NEW_2_review_crops.py
{'=' * 70}
""")

if __name__ == "__main__":
    main()