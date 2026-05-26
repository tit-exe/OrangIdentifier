# =============================================================================
# dataset_status.py
# Affiche l'etat complet du projet en quelques secondes.
# Lance le matin pour savoir ou tu en es avant de commencer.
# Usage : python scripts/scripts_annexes/dataset_status.py
# =============================================================================

import sys
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Ajouter le dossier scripts au path pour importer config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    BASE_DIR, DONE_FILE, DATASET_YOLO_DIR, CLASSIF_RAW_DIR,
    YOLO_BEST, RESNET_BEST, SESSION_LOG, RESULTS_DIR, VIDEOS_DIR,
    INDIVIDUS, ANNOT_OBJECTIF_PAR_INDIVIDU, NUM_CLASSES
)

def sep(titre=""):
    if titre:
        print(f"\n{'=' * 20} {titre} {'=' * 20}")
    else:
        print("=" * 60)

def check(condition):
    return "OK" if condition else "MANQUANT"

# =============================================================================
# EN-TETE
# =============================================================================

sep()
print("ETAT DU PROJET - OrangIdentifier")
print(f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}")
print(f"Base : {BASE_DIR}")
sep()

# =============================================================================
# ANNOTATIONS (done.txt)
# =============================================================================

sep("ANNOTATIONS")

done = set()
if DONE_FILE.exists():
    done = set(s.strip() for s in DONE_FILE.read_text().splitlines() if s.strip())

labels_dir = DATASET_YOLO_DIR / "labels"
valides = defaultdict(int)
skips   = defaultdict(int)

for stem in done:
    lbl = labels_dir / (stem + ".txt")
    ind = stem.split("_")[0]
    if lbl.exists() and lbl.stat().st_size > 0:
        valides[ind] += 1
    else:
        skips[ind] += 1

total_valides = sum(valides.values())
total_skips   = sum(skips.values())
tous_ok = all(valides.get(ind, 0) >= ANNOT_OBJECTIF_PAR_INDIVIDU for ind in INDIVIDUS)

print(f"  Traites total      : {len(done)}")
print(f"  Avec boite (utiles): {total_valides}")
print(f"  Skips              : {total_skips}")
print(f"\n  {'Individu':<28} {'Valides':>8} {'Skips':>6}  Statut")
print(f"  {'-'*28} {'-'*8} {'-'*6}  {'------'}")

for ind in INDIVIDUS:
    v = valides.get(ind, 0)
    s = skips.get(ind, 0)
    if v >= ANNOT_OBJECTIF_PAR_INDIVIDU:
        statut = "OK"
    else:
        statut = f"manque {ANNOT_OBJECTIF_PAR_INDIVIDU - v}"
    print(f"  {ind:<28} {v:>8} {s:>6}  {statut}")

print(f"\n  Objectif global : {'ATTEINT' if tous_ok else 'EN COURS'}")

# =============================================================================
# DATASET CLASSIFICATION (visages extraits)
# =============================================================================

sep("VISAGES EXTRAITS (ResNet50)")

if CLASSIF_RAW_DIR.exists():
    total_faces = 0
    print(f"  {'Individu':<28} {'Visages':>8}")
    print(f"  {'-'*28} {'-'*8}")
    for ind in INDIVIDUS:
        d = CLASSIF_RAW_DIR / ind
        nb = len(list(d.glob("*.jpg"))) if d.exists() else 0
        total_faces += nb
        statut = "OK" if nb >= 30 else f"peu ({nb})"
        print(f"  {ind:<28} {nb:>8}  {statut}")
    print(f"\n  Total visages : {total_faces}")
else:
    print("  Script 3 (extract_faces) pas encore lance.")

# =============================================================================
# MODELES
# =============================================================================

sep("MODELES")

yolo_ok   = YOLO_BEST.exists()
resnet_ok = RESNET_BEST.exists()

print(f"  YOLO best.pt     : {check(yolo_ok)}")
if yolo_ok:
    mtime = datetime.fromtimestamp(YOLO_BEST.stat().st_mtime)
    print(f"    -> Entraine le {mtime.strftime('%d/%m/%Y %H:%M')}")

print(f"  ResNet best.pth  : {check(resnet_ok)}")
if resnet_ok:
    mtime = datetime.fromtimestamp(RESNET_BEST.stat().st_mtime)
    print(f"    -> Entraine le {mtime.strftime('%d/%m/%Y %H:%M')}")
    # Lire les metadonnees si disponibles
    meta_path = RESNET_BEST.parent / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"    -> Accuracy test : {meta.get('test_accuracy', 'N/A')}")
        print(f"    -> Individus     : {meta.get('num_classes', 'N/A')}")
        print(f"    -> Epoch         : {meta.get('best_epoch', 'N/A')}")

# =============================================================================
# RESULTATS
# =============================================================================

sep("RESULTATS DISPONIBLES")

if RESULTS_DIR.exists():
    pngs = list(RESULTS_DIR.glob("*.png"))
    csvs = list(RESULTS_DIR.glob("*.csv"))
    if pngs or csvs:
        for f in sorted(pngs + csvs):
            print(f"  {f.name}")
    else:
        print("  Aucun resultat encore genere.")
else:
    print("  Dossier RESULTS inexistant.")

# =============================================================================
# VIDEOS
# =============================================================================

sep("VIDEOS")

if VIDEOS_DIR.exists():
    videos = list(VIDEOS_DIR.glob("*.mp4"))
    taille_totale = sum(v.stat().st_size for v in videos) / 1e9
    print(f"  {len(videos)} videos ({taille_totale:.1f} Go)")
    for v in sorted(videos):
        taille = v.stat().st_size / 1e6
        print(f"  {v.name:<60} {taille:.0f} Mo")
else:
    print("  Dossier VIDEOS introuvable.")

# =============================================================================
# JOURNAL DE SESSION (5 dernieres actions)
# =============================================================================

sep("JOURNAL (5 dernieres actions)")

if SESSION_LOG.exists():
    try:
        log = json.loads(SESSION_LOG.read_text(encoding="utf-8"))
        for entry in log[-5:]:
            ts = entry.get("timestamp", "")[:16].replace("T", " ")
            script = entry.get("script", "")
            action = entry.get("action", "")
            details = entry.get("details", {})
            detail_str = "  ".join(f"{k}={v}" for k,v in details.items())
            print(f"  [{ts}] {script} -> {action}  {detail_str}")
    except Exception as e:
        print(f"  Erreur lecture log : {e}")
else:
    print("  Pas encore de journal (session_log.json).")

# =============================================================================
# PROCHAINE ETAPE SUGGEREE
# =============================================================================

sep("PROCHAINE ETAPE")

if not yolo_ok:
    print("  -> Lance 2_train_yolo.py")
elif not CLASSIF_RAW_DIR.exists() or sum(
        len(list((CLASSIF_RAW_DIR/ind).glob("*.jpg")))
        for ind in INDIVIDUS if (CLASSIF_RAW_DIR/ind).exists()
    ) < 100:
    print("  -> Lance 3_extract_faces.py")
elif not resnet_ok:
    print("  -> Lance 4_train_resnet.py")
else:
    print("  -> Pipeline complet ! Lance 5_video_pipeline.py")

sep()