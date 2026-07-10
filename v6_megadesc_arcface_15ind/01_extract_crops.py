"""
V6_1_extract_crops.py — OrangIdentifier V6
===========================================
Étape 1 : Extraction des crops des 5 nouveaux individus zoo
           via YOLOv2 + per-crop quality analysis (brightness + sharpness).

Source   : data/photos\\<Individu>\\*.jpg
Sorties  : data/crops/new\\<Individu>\\*.jpg
           output/v6/quality_report.json
           output/v6/results\\01_quality_histograms.png

Quality metric (computed on the 224x224 crop, not the raw image):
  - brightness : moyenne pixel niveaux de gris ∈ [0, 1]
  - sharpness  : Laplacian variance (blurry -> low, sharp -> high)
  - yolo_conf  : YOLO confidence of the detection

No image is rejected automatically.
Les seuils sont indicatifs — lancer V6_2_review_quality.py pour filtrer.

RUN :
    conda activate orangs
    python v6_megadesc_arcface_15ind/01_extract_crops.py
"""

import os
os.environ["HF_HOME"]    = r"D:\HuggingFaceCache"
os.environ["TORCH_HOME"] = r"D:\TorchCache"

import cv2
import json
import shutil
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]  # repository root (portable)

# ══════════════════════════════════════════════════════════════════════════════
# CHEMINS
# ══════════════════════════════════════════════════════════════════════════════
INPUT_DIR  = (REPO / "data" / "photos")
OUTPUT_DIR = (REPO / "data" / "crops" / "new")
RESULTS    = (REPO / "output" / "v6" / "results")
JSON_PATH  = (REPO / "output" / "v6" / "quality_report.json")
YOLO_MODEL = (REPO / "models" / "yolo_v2_medium_mAP99.pt")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CROP_SIZE   = 224
MARGIN      = 0.05       # marge bbox en fraction de la largeur/hauteur
CONF_THRESH = 0.25       # YOLO confidence threshold
IOU_THRESH  = 0.45
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
SAVE_EVERY  = 50         # save JSON every N images

# Indicative thresholds (for flags, no automatic rejection)
FLAG_DARK   = 0.20       # brightness < 0.20 -> dark flag
FLAG_BLUR   = 80.0       # sharpness < 80   -> blurry flag

# ══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES JSON
# ══════════════════════════════════════════════════════════════════════════════
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
                print("  [WARN] JSON corrupted, loading from backup")
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
        print(f"  [ERROR] JSON save failed : {e}")
        if tmp.exists():
            try: tmp.unlink()
            except: pass

# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTION + QUALITÉ
# ══════════════════════════════════════════════════════════════════════════════
def extract_crop(img_bgr, x1r, y1r, x2r, y2r):
    """Extrait un crop avec marge, redimensionne en CROP_SIZE×CROP_SIZE."""
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
    """Compute brightness and sharpness on the 224x224 crop."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)  # uint8 — Laplacian CV_64F OK
    brightness = float(gray.mean() / 255.0)
    sharpness  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return brightness, sharpness

# ══════════════════════════════════════════════════════════════════════════════
# HISTOGRAMMES
# ══════════════════════════════════════════════════════════════════════════════
def plot_quality_histograms(db, out_path):
    """Generate brightness + sharpness histograms per individual."""
    individuals = sorted(set(v["individu"] for v in db.values() if v.get("statut") in ("ok", "multi")))
    if not individuals:
        return

    n = len(individuals)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    fig.suptitle("V6 - Crop quality per individual", fontsize=13, y=1.01)
    if n == 1:
        axes = [axes]

    COLORS = ["#3b82f6", "#16a34a", "#f97316", "#dc2626", "#9333ea"]

    for row, (ind, ax_row) in enumerate(zip(individuals, axes)):
        color = COLORS[row % len(COLORS)]
        entries = [v for v in db.values() if v.get("individu") == ind and v.get("statut") in ("ok", "multi")]

        brightnesses = [v["brightness"] for v in entries]
        sharpnesses  = [v["sharpness"]  for v in entries]

        # - Brightness -
        ax_row[0].hist(brightnesses, bins=30, color=color, alpha=0.80, edgecolor="white")
        ax_row[0].axvline(FLAG_DARK, color="red", lw=2, linestyle="--",
                          label=f"Dark threshold ({FLAG_DARK:.2f})")
        pct_dark = sum(b < FLAG_DARK for b in brightnesses) / max(len(brightnesses), 1) * 100
        ax_row[0].set_title(f"{ind} - Brightness  ({pct_dark:.0f}% too dark)", fontsize=10)
        ax_row[0].set_xlabel("Mean crop brightness [0-1]")
        ax_row[0].set_ylabel("Nb crops")
        ax_row[0].set_xlim(0, 1)
        ax_row[0].legend(fontsize=8)
        ax_row[0].grid(alpha=0.3)

        # - Sharpness -
        ax_row[1].hist(sharpnesses, bins=30, color=color, alpha=0.80, edgecolor="white")
        ax_row[1].axvline(FLAG_BLUR, color="orange", lw=2, linestyle="--",
                          label=f"Blur threshold ({FLAG_BLUR:.0f})")
        pct_blur = sum(s < FLAG_BLUR for s in sharpnesses) / max(len(sharpnesses), 1) * 100
        ax_row[1].set_title(f"{ind} - Sharpness (Laplacian variance)  ({pct_blur:.0f}% blurry)", fontsize=10)
        ax_row[1].set_xlabel("Variance Laplacien")
        ax_row[1].set_ylabel("Nb crops")
        ax_row[1].set_xlim(left=0)
        ax_row[1].legend(fontsize=8)
        ax_row[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Histogrammes → {out_path.name}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  V6_1_extract_crops.py - Extraction + quality analysis")
    print(f"  {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 70)

    if not INPUT_DIR.exists():
        print(f"  [ERROR] Folder not found : {INPUT_DIR}"); return
    if not YOLO_MODEL.exists():
        print(f"  [ERROR] YOLO not found : {YOLO_MODEL}"); return

    # ── Discovery of individuals ───────────────────────────────────────────────
    individuals = sorted([d.name for d in INPUT_DIR.iterdir() if d.is_dir()])
    if not individuals:
        print(f"  [ERROR] No subfolder in {INPUT_DIR}"); return

    print(f"\n  {len(individuals)} individuals found :")
    for name in individuals:
        imgs = [f for f in (INPUT_DIR / name).iterdir() if f.suffix in IMG_EXTS]
        print(f"    {name:<14}: {len(imgs):4d} photos")
        (OUTPUT_DIR / name).mkdir(parents=True, exist_ok=True)

    # ── Resume (skip already processed) ───────────────────────────────────────────
    db = load_json(JSON_PATH)
    already_done = set(db.keys())
    print(f"\n  Already processed : {len(already_done)} (will be skipped)")

    tasks = []
    for individu in individuals:
        for img_path in sorted((INPUT_DIR / individu).iterdir()):
            if img_path.suffix not in IMG_EXTS:
                continue
            key = f"{individu}/{img_path.stem}"
            if key not in already_done:
                tasks.append((individu, img_path, key))

    print(f"  À traiter : {len(tasks)} images\n")
    if not tasks:
        print("  Everything is already extracted. Generating histograms...")
        plot_quality_histograms(db, RESULTS / "01_quality_histograms.png")
        _print_summary(db, individuals)
        return

    # -- Loading YOLO ───────────────────────────────────────────────────────
    print("  Loading YOLO v2...")
    import torch
    from ultralytics import YOLO
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = YOLO(str(YOLO_MODEL))
    print(f"  YOLO loaded on {device}\n")

    n_ok = n_multi = n_no_det = n_dark = n_blur = n_error = 0

    bar = tqdm(tasks, desc="  Extraction", unit="img", ncols=90,
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for i, (individu, img_path, key) in enumerate(bar):
        stem     = img_path.stem
        crop_dst = OUTPUT_DIR / individu / f"{stem}.jpg"

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            db[key] = {"individu": individu, "stem": stem,
                       "statut": "erreur_lecture", "crop_file": None}
            n_error += 1
            continue

        h, w = img_bgr.shape[:2]

        try:
            results    = model.predict(source=str(img_path),
                                        conf=CONF_THRESH, iou=IOU_THRESH,
                                        verbose=False, device=device)
            detections = results[0].boxes
        except Exception as e:
            db[key] = {"individu": individu, "stem": stem,
                       "statut": "erreur_yolo", "crop_file": None, "error": str(e)}
            n_error += 1
            continue

        n_faces = len(detections) if detections is not None else 0

        if n_faces == 0:
            db[key] = {"individu": individu, "stem": stem,
                       "img_w": w, "img_h": h, "n_faces": 0,
                       "statut": "no_detection", "crop_file": None}
            n_no_det += 1
            bar.set_postfix({"ok": n_ok, "no_det": n_no_det, "err": n_error})
            continue

        # Best detection (highest confidence)
        best_idx = int(detections.conf.argmax())
        x1r, y1r, x2r, y2r = detections[best_idx].xyxy[0].cpu().numpy()
        yolo_conf = float(detections[best_idx].conf[0].cpu().numpy())

        crop = extract_crop(img_bgr, x1r, y1r, x2r, y2r)
        if crop is None:
            db[key] = {"individu": individu, "stem": stem,
                       "statut": "erreur_crop", "crop_file": None}
            n_error += 1
            continue

        brightness, sharpness = quality_metrics(crop)

        # Quality flags (informative, no rejection)
        flags = []
        if brightness < FLAG_DARK:
            flags.append("dark"); n_dark += 1
        if sharpness < FLAG_BLUR:
            flags.append("blurry");   n_blur += 1

        cv2.imwrite(str(crop_dst), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

        statut = "multi" if n_faces > 1 else "ok"
        if n_faces > 1: n_multi += 1
        else:           n_ok    += 1

        db[key] = {
            "individu":    individu,
            "stem":        stem,
            "photo_source": str(img_path),
            "crop_file":   str(crop_dst),
            "img_w": w, "img_h": h,
            "yolo_conf":   round(yolo_conf, 4),
            "n_faces":     n_faces,
            "brightness":  round(brightness, 4),
            "sharpness":   round(sharpness,  2),
            "flags":       flags,
            "statut":      statut,
            "ts":          datetime.now().isoformat(),
        }

        bar.set_postfix({"ok": n_ok, "multi": n_multi,
                          "no_det": n_no_det, "dark": n_dark, "err": n_error})

        if (i + 1) % SAVE_EVERY == 0:
            save_json(db, JSON_PATH)

    bar.close()
    save_json(db, JSON_PATH)

    # ── Quality histograms ──────────────────────────────────────────────────
    plot_quality_histograms(db, RESULTS / "01_quality_histograms.png")

    # -- Summary ────────────────────────────────────────────────────────────────
    _print_summary(db, individuals)
    print(f"""
{'=' * 70}
  EXTRACTION TERMINÉE — {datetime.now():%Y-%m-%d %H:%M}
{'=' * 70}
  Total processed  : {len(tasks)}
  Crops OK      : {n_ok}
  Multi-face    : {n_multi}   (best face kept)
  Not detected  : {n_no_det}
  Flag sombre   : {n_dark}   (brightness < {FLAG_DARK})
  Flag flou     : {n_blur}    (sharpness  < {FLAG_BLUR})
  Erreurs       : {n_error}

  Crops     → {OUTPUT_DIR}
  Rapport   → {JSON_PATH.name}
  Histos    → results/01_quality_histograms.png

  ÉTAPE SUIVANTE :
    python v6_megadesc_arcface_15ind/01c_review_quality.py
{'=' * 70}
""")

def _print_summary(db, individuals):
    print(f"\n  {'Individual':<14} {'Crops OK':>9} {'No det':>7} {'Dark':>8} {'Blurry':>6}")
    print(f"  {'─'*50}")
    for ind in individuals:
        entries = [v for v in db.values() if v.get("individu") == ind]
        n_ok_i    = sum(1 for v in entries if v.get("statut") in ("ok", "multi"))
        n_nodet_i = sum(1 for v in entries if v.get("statut") == "no_detection")
        n_dark_i  = sum(1 for v in entries if "dark" in v.get("flags", []))
        n_blur_i  = sum(1 for v in entries if "blurry" in v.get("flags", []))
        warn = " !" if n_dark_i / max(n_ok_i, 1) > 0.5 else ""
        print(f"  {ind:<14} {n_ok_i:>9} {n_nodet_i:>7} {n_dark_i:>8} {n_blur_i:>6}{warn}")
    print()

if __name__ == "__main__":
    main()