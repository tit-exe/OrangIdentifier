"""
V6_1b_brighten.py — OrangIdentifier V6
=======================================
Étape 1b : Correction luminosité des crops sombres via CLAHE (espace LAB).

Problème : Bukit, Rosa, Yori ont ~50-84% de crops avec brightness < 0.20.
           Rejeter ces crops ferait perdre trop de données d'entraînement.

Solution : CLAHE (Contrast Limited Adaptive Histogram Equalization) sur le
           canal L de l'espace LAB — améliore la luminosité et le contraste
           local sans toucher les canaux couleur A/B ni saturer les zones déjà
           claires. Standard en reconnaissance faciale pour la normalisation
           d'éclairage.

Comportement :
  - Traite uniquement les crops avec brightness < BRIGHT_THRESH (défaut 0.25)
  - Applique CLAHE (clipLimit=3, tileGrid=8×8) sur le canal L
  - Sauvegarde en place (écrase le crop original)
  - Met à jour brightness + sharpness dans quality_report.json
  - Génère un histogramme avant/après

RUN :
    conda activate orangs
    python v6_megadesc_arcface_15ind/01b_brighten.py

    # Tester sans modifier les fichiers :
    python v6_megadesc_arcface_15ind/01b_brighten.py --dry-run

    # Ajuster le seuil (défaut 0.25) :
    python v6_megadesc_arcface_15ind/01b_brighten.py --thresh 0.30
"""

import argparse
import cv2
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

REPO = Path(__file__).resolve().parents[1]  # repository root (portable)

# ══════════════════════════════════════════════════════════════════════════════
# ARGS
# ══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument("--thresh",   type=float, default=0.25,
                    help="Seuil brightness en dessous duquel corriger (défaut 0.25)")
parser.add_argument("--clip",     type=float, default=3.0,
                    help="CLAHE clipLimit — plus élevé = plus d'amplification (défaut 3.0)")
parser.add_argument("--dry-run",  action="store_true",
                    help="Afficher les stats sans modifier les fichiers")
ARGS = parser.parse_args()

BRIGHT_THRESH = ARGS.thresh
CLIP_LIMIT    = ARGS.clip

# ══════════════════════════════════════════════════════════════════════════════
# CHEMINS
# ══════════════════════════════════════════════════════════════════════════════
JSON_PATH = (REPO / "output" / "v6" / "quality_report.json")
RESULTS   = (REPO / "output" / "v6" / "results")
RESULTS.mkdir(parents=True, exist_ok=True)

if not JSON_PATH.exists():
    print(f"[ERROR] {JSON_PATH} introuvable — lancez d'abord V6_1_extract_crops.py")
    raise SystemExit(1)

with open(JSON_PATH, encoding="utf-8") as f:
    db = json.load(f)

# ══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════
def clahe_brighten(crop_bgr):
    """CLAHE sur canal L uniquement (LAB). Préserve les couleurs."""
    lab              = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    l, a, b          = cv2.split(lab)
    clahe            = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=(8, 8))
    l_eq             = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)

def quality_metrics(crop_bgr):
    gray       = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean() / 255.0)
    sharpness  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return brightness, sharpness

# ══════════════════════════════════════════════════════════════════════════════
# SÉLECTION DES CROPS À CORRIGER
# ══════════════════════════════════════════════════════════════════════════════
candidates = [
    (key, v) for key, v in db.items()
    if v.get("statut") in ("ok", "multi")
    and v.get("crop_file")
    and v.get("brightness", 1.0) < BRIGHT_THRESH
]

print("=" * 65)
print("  V6_1b_brighten.py — Correction luminosité CLAHE (LAB)")
print(f"  {datetime.now():%Y-%m-%d %H:%M}")
print("=" * 65)
print(f"\n  Seuil brightness : < {BRIGHT_THRESH}  |  clipLimit : {CLIP_LIMIT}")
print(f"  Crops à corriger : {len(candidates)} / {len(db)} total")
if ARGS.dry_run:
    print("  MODE --dry-run : aucun fichier modifié\n")

# Stats par individu avant correction
individuals = sorted(set(v["individu"] for _, v in candidates))
print(f"\n  {'Individu':<14} {'À corriger':>11} {'Bright µ avant':>15}")
print(f"  {'─'*45}")
for ind in individuals:
    entries_i = [(k, v) for k, v in candidates if v["individu"] == ind]
    b_mean    = np.mean([v["brightness"] for _, v in entries_i])
    print(f"  {ind:<14} {len(entries_i):>11} {b_mean:>14.3f}")

if not candidates:
    print("\n  Aucun crop sous le seuil — rien à faire.")
    raise SystemExit(0)

# ══════════════════════════════════════════════════════════════════════════════
# CORRECTION
# ══════════════════════════════════════════════════════════════════════════════
bright_before = []
bright_after  = []
sharp_before  = []
sharp_after   = []

n_ok = n_err = 0

for key, v in candidates:
    src = Path(v["crop_file"])
    if not src.exists():
        n_err += 1
        continue

    crop_bgr = cv2.imread(str(src))
    if crop_bgr is None:
        n_err += 1
        continue

    b_before, s_before = quality_metrics(crop_bgr)
    bright_before.append(b_before)
    sharp_before.append(s_before)

    corrected = clahe_brighten(crop_bgr)
    b_after, s_after = quality_metrics(corrected)
    bright_after.append(b_after)
    sharp_after.append(s_after)

    if not ARGS.dry_run:
        cv2.imwrite(str(src), corrected, [cv2.IMWRITE_JPEG_QUALITY, 95])
        db[key]["brightness"] = round(b_after, 4)
        db[key]["sharpness"]  = round(s_after, 2)
        # Retirer le flag "sombre" si brightness maintenant >= BRIGHT_THRESH
        flags = db[key].get("flags", [])
        if b_after >= BRIGHT_THRESH and "sombre" in flags:
            flags.remove("sombre")
        db[key]["flags"] = flags
        db[key]["clahe_applied"] = True

    n_ok += 1

print(f"\n  Traités : {n_ok}  |  Erreurs : {n_err}")
print(f"\n  Brightness — avant : {np.mean(bright_before):.3f} → après : {np.mean(bright_after):.3f}")
print(f"  Netteté    — avant : {np.mean(sharp_before):.1f}  → après : {np.mean(sharp_after):.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# SAUVEGARDE JSON + HISTOGRAMME AVANT/APRÈS
# ══════════════════════════════════════════════════════════════════════════════
if not ARGS.dry_run and n_ok > 0:
    tmp = JSON_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
    tmp.replace(JSON_PATH)
    print(f"  quality_report.json mis à jour")

# Histogramme avant/après par individu
fig, axes = plt.subplots(len(individuals), 2, figsize=(14, 4 * len(individuals)), squeeze=False)
fig.suptitle(f"V6 — Correction CLAHE (seuil brightness < {BRIGHT_THRESH})\n"
             f"Avant (rouge) vs Après (bleu) — clipLimit={CLIP_LIMIT}",
             fontsize=13, y=1.01)

for row, ind in enumerate(individuals):
    entries_i = [(k, v) for k, v in candidates if v["individu"] == ind]
    n_i = len(entries_i)

    # Recalculer depuis les données collectées pour cet individu
    idx_i    = [i for i, (_, v) in enumerate(candidates) if v["individu"] == ind]
    bb_i     = [bright_before[i] for i in idx_i]
    ba_i     = [bright_after[i]  for i in idx_i]
    sb_i     = [sharp_before[i]  for i in idx_i]
    sa_i     = [sharp_after[i]   for i in idx_i]

    ax = axes[row][0]
    ax.hist(bb_i, bins=20, color="#dc2626", alpha=0.60, label=f"Avant (µ={np.mean(bb_i):.3f})")
    ax.hist(ba_i, bins=20, color="#3b82f6", alpha=0.60, label=f"Après (µ={np.mean(ba_i):.3f})")
    ax.axvline(BRIGHT_THRESH, color="gray", lw=1.5, linestyle="--",
               label=f"Seuil {BRIGHT_THRESH}")
    ax.set_title(f"{ind} — Luminosité ({n_i} crops)", fontsize=10)
    ax.set_xlabel("Brightness [0–1]"); ax.set_ylabel("N crops")
    ax.set_xlim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[row][1]
    ax.hist(sb_i, bins=20, color="#dc2626", alpha=0.60, label=f"Avant (µ={np.mean(sb_i):.0f})")
    ax.hist(sa_i, bins=20, color="#3b82f6", alpha=0.60, label=f"Après (µ={np.mean(sa_i):.0f})")
    ax.set_title(f"{ind} — Netteté Laplacien", fontsize=10)
    ax.set_xlabel("Variance Laplacien"); ax.set_ylabel("N crops")
    ax.set_xlim(left=0); ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
out = RESULTS / "01b_clahe_before_after.png"
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
print(f"  Histogramme avant/après → {out.name}")

# ══════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ FINAL
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  {'Individu':<14} {'Crops corrigés':>15} {'Bright µ avant':>15} {'Bright µ après':>15}")
print(f"  {'─'*62}")
for ind in individuals:
    idx_i = [i for i, (_, v) in enumerate(candidates) if v["individu"] == ind]
    print(f"  {ind:<14} {len(idx_i):>15} {np.mean([bright_before[i] for i in idx_i]):>14.3f}"
          f" {np.mean([bright_after[i] for i in idx_i]):>14.3f}")

if ARGS.dry_run:
    print("\n  --dry-run : aucun fichier modifié. Relancer sans --dry-run pour appliquer.")
else:
    print(f"""
  ÉTAPE SUIVANTE :
    python v6_megadesc_arcface_15ind/01c_review_quality.py
""")