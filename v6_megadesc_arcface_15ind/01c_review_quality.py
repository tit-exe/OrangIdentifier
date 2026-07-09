"""
V6_2_review_quality.py — OrangIdentifier V6
============================================
Étape 2 : Revue qualité des crops extraits par V6_1.

Lit quality_report.json, affiche les statistiques détaillées par individu,
et déplace les crops trop sombres / trop flous vers <Individu>/_rejet/
si l'utilisateur confirme les seuils.

Les crops dans _rejet/ ne seront PAS chargés par V6_3_train.py
(load_dir exclut les dossiers commençant par "_").

RUN :
    conda activate orangs
    python v6_megadesc_arcface_15ind/01c_review_quality.py

    # Appliquer des seuils personnalisés (sans interaction) :
    python v6_megadesc_arcface_15ind/01c_review_quality.py --auto --bright 0.25 --sharp 60
"""

import argparse
import shutil
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

REPO = Path(__file__).resolve().parents[1]  # repository root (portable)

# ══════════════════════════════════════════════════════════════════════════════
# CHEMINS
# ══════════════════════════════════════════════════════════════════════════════
CROPS_DIR  = (REPO / "data" / "crops" / "new")
JSON_PATH  = (REPO / "output" / "v6" / "quality_report.json")
RESULTS    = (REPO / "output" / "v6" / "results")
RESULTS.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# ARGS
# ══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument("--auto",   action="store_true",
                    help="Appliquer les seuils sans confirmation interactive")
parser.add_argument("--bright", type=float, default=None,
                    help="Seuil luminosité (défaut : interactif)")
parser.add_argument("--sharp",  type=float, default=None,
                    help="Seuil netteté (défaut : interactif)")
parser.add_argument("--dry-run", action="store_true",
                    help="Afficher sans déplacer aucun fichier")
ARGS = parser.parse_args()

# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT
# ══════════════════════════════════════════════════════════════════════════════
if not JSON_PATH.exists():
    print(f"[ERROR] Rapport introuvable : {JSON_PATH}")
    print("        Lancez d'abord V6_1_extract_crops.py")
    raise SystemExit(1)

with open(JSON_PATH, encoding="utf-8") as f:
    db = json.load(f)

ok_entries = [v for v in db.values() if v.get("statut") in ("ok", "multi") and v.get("crop_file")]
individuals = sorted(set(v["individu"] for v in ok_entries))

print("=" * 65)
print("  V6_2_review_quality.py — Revue des crops")
print(f"  {datetime.now():%Y-%m-%d %H:%M}")
print("=" * 65)
print(f"\n  {len(ok_entries)} crops valides sur {len(db)} photos analysées\n")

# ══════════════════════════════════════════════════════════════════════════════
# STATISTIQUES PAR INDIVIDU
# ══════════════════════════════════════════════════════════════════════════════
stats = {}
for ind in individuals:
    entries_i = [v for v in ok_entries if v["individu"] == ind]
    brights   = np.array([v["brightness"] for v in entries_i])
    sharps    = np.array([v["sharpness"]  for v in entries_i])
    stats[ind] = {"entries": entries_i, "brights": brights, "sharps": sharps}

print(f"  {'Individu':<14} {'N':>5} {'Bright µ':>10} {'Bright <.20':>12} {'Sharp µ':>9} {'Sharp <80':>10}")
print(f"  {'─'*60}")
for ind in individuals:
    s   = stats[ind]
    n   = len(s["brights"])
    b_m = s["brights"].mean()
    b_p = (s["brights"] < 0.20).mean() * 100
    sh_m = s["sharps"].mean()
    sh_p = (s["sharps"] < 80).mean() * 100
    warn = "  ! SOMBRE" if b_p > 50 else ""
    print(f"  {ind:<14} {n:>5} {b_m:>10.3f} {b_p:>11.0f}% {sh_m:>9.1f} {sh_p:>9.0f}%{warn}")

# ══════════════════════════════════════════════════════════════════════════════
# GRAPHE SCATTER : luminosité vs netteté par individu
# ══════════════════════════════════════════════════════════════════════════════
COLORS = ["#3b82f6", "#16a34a", "#f97316", "#dc2626", "#9333ea"]
fig, ax = plt.subplots(figsize=(10, 7))
for idx, ind in enumerate(individuals):
    s = stats[ind]
    ax.scatter(s["brights"], s["sharps"],
               alpha=0.30, s=10, color=COLORS[idx % len(COLORS)], label=ind)

ax.axvline(0.20, color="red",    lw=1.5, linestyle="--", label="Seuil sombre (0.20)")
ax.axhline(80,   color="orange", lw=1.5, linestyle="--", label="Seuil flou (80)")
ax.set_xlabel("Luminosité crop [0–1]",     fontsize=11)
ax.set_ylabel("Netteté (variance Laplacien)", fontsize=11)
ax.set_title("V6 — Qualité des crops : luminosité × netteté", fontsize=12)
ax.legend(fontsize=8, markerscale=3)
ax.grid(alpha=0.3)
plt.tight_layout()
scatter_path = RESULTS / "02_quality_scatter.png"
plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Scatter → {scatter_path.name}")

# ══════════════════════════════════════════════════════════════════════════════
# SEUILS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("  Définition des seuils de rejet")
print("─" * 65)

if ARGS.auto and ARGS.bright is not None and ARGS.sharp is not None:
    thresh_bright = ARGS.bright
    thresh_sharp  = ARGS.sharp
    print(f"  Mode --auto : luminosité < {thresh_bright}  |  netteté < {thresh_sharp}")
else:
    print("\n  Entrez les seuils (laisser vide = pas de filtre sur ce critère) :")
    b_in = input("  Seuil luminosité  [défaut 0.20, ex: 0.25] : ").strip()
    s_in = input("  Seuil netteté     [défaut 80,   ex: 60   ] : ").strip()
    thresh_bright = float(b_in) if b_in else 0.20
    thresh_sharp  = float(s_in) if s_in else 80.0

# ── Identifier les crops à rejeter ───────────────────────────────────────────
to_reject = []
for v in ok_entries:
    reject = False
    reasons = []
    if v["brightness"] < thresh_bright:
        reject = True; reasons.append("sombre")
    if v["sharpness"] < thresh_sharp:
        reject = True; reasons.append("flou")
    if reject:
        to_reject.append((v, reasons))

print(f"\n  → {len(to_reject)} crops à rejeter / {len(ok_entries)} total")
print(f"\n  Résumé par individu :")
for ind in individuals:
    rej_i = [(v, r) for v, r in to_reject if v["individu"] == ind]
    keep  = len(stats[ind]["entries"]) - len(rej_i)
    print(f"    {ind:<14}: {keep:3d} gardés  /  {len(rej_i):3d} rejetés")

# ══════════════════════════════════════════════════════════════════════════════
# AVERTISSEMENT si un individu perd trop de crops
# ══════════════════════════════════════════════════════════════════════════════
MIN_CROPS = 20
print()
for ind in individuals:
    rej_i = sum(1 for v, r in to_reject if v["individu"] == ind)
    keep  = len(stats[ind]["entries"]) - rej_i
    if keep < MIN_CROPS:
        print(f"  !  {ind} : seulement {keep} crops après filtre "
              f"(minimum recommandé : {MIN_CROPS})")
        print(f"     Envisager d'assouplir les seuils ou d'exclure cet individu du training.")

# ══════════════════════════════════════════════════════════════════════════════
# DÉPLACEMENT VERS _rejet/
# ══════════════════════════════════════════════════════════════════════════════
if ARGS.dry_run:
    print("\n  --dry-run : aucun fichier déplacé.")
elif not to_reject:
    print("\n  Aucun crop à rejeter avec ces seuils.")
else:
    if not ARGS.auto:
        confirm = input(f"\n  Déplacer {len(to_reject)} crops vers _rejet/ ? [o/N] : ").strip().lower()
        do_move = confirm in ("o", "oui", "y", "yes")
    else:
        do_move = True

    if do_move:
        moved = 0
        for v, reasons in to_reject:
            src = Path(v["crop_file"])
            if not src.exists():
                continue
            dst_dir = src.parent / "_rejet"
            dst_dir.mkdir(exist_ok=True)
            dst = dst_dir / src.name
            shutil.move(str(src), str(dst))
            moved += 1
        print(f"\n  {moved} crops déplacés vers leurs dossiers _rejet/")
        print("  (load_dir dans V6_3_train.py ignore automatiquement _rejet/)")
    else:
        print("\n  Annulé — aucun fichier déplacé.")

# ══════════════════════════════════════════════════════════════════════════════
# DÉCOMPTE FINAL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("  Crops disponibles pour l'entraînement V6 :")
print("─" * 65)
for ind in individuals:
    ind_dir = CROPS_DIR / ind
    if ind_dir.exists():
        n_final = len([f for f in ind_dir.iterdir()
                       if f.suffix in {".jpg",".jpeg",".png"} and not f.name.startswith("_")])
    else:
        n_final = 0
    status = "✓" if n_final >= MIN_CROPS else "! INSUFFISANT"
    print(f"  {ind:<14}: {n_final:3d} crops  {status}")

print(f"""
  ÉTAPE SUIVANTE :
    python v6_megadesc_arcface_15ind/02_train.py
""")