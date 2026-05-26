# =============================================================================
# 2b_train_yolo_v2.py
# Entrainement YOLO v2 sur les annotations corrigees manuellement.
#
# Source : boxes_cache.json — coordonnees crop_x1/y1/x2/y2 corrigees
#          par l'utilisateur image par image.
# Modele : yolov8s.pt (small) — meilleure precision qu'nano avec ~2000 images
# Sortie : MODELS/yolo_orangs_v2/best.pt  — n'ecrase pas v1
#
# Conversion YOLO (format .txt par image) :
#   xc = (crop_x1 + crop_x2) / 2 / img_w
#   yc = (crop_y1 + crop_y2) / 2 / img_h
#   w  = (crop_x2 - crop_x1) / img_w
#   h  = (crop_y2 - crop_y1) / img_h
#   ligne : 0 xc yc w h
# =============================================================================

import sys
import shutil
import random
import json
import yaml
import time
from pathlib import Path
from datetime import timedelta
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    BASE_DIR, DATASET_CLASSIF_DIR, PHOTOS_DIR,
    RESULTS_DIR, MODELS_DIR,
    YOLO_EPOCHS, YOLO_BATCH, YOLO_PATIENCE,
    YOLO_DEGREES, YOLO_TRANSLATE, YOLO_SCALE,
    YOLO_FLIPUD, YOLO_FLIPLR,
    YOLO_HSV_H, YOLO_HSV_S, YOLO_HSV_V,
    log_action
)

# =============================================================================
# CONFIGURATION V2
# =============================================================================

BOXES_CACHE   = DATASET_CLASSIF_DIR / "boxes_cache.json"
DATASET_V2    = BASE_DIR / "DATASET_YOLO_V2"          # nouveau dataset propre
MODELS_V2     = MODELS_DIR.parent / "yolo_orangs_v2"   # n'ecrase pas v1
RUN_NAME      = "orang_face_detector_v2"

TRAIN_RATIO   = 0.80
RANDOM_SEED   = 42

# yolov8s (small) au lieu de nano : ~3x plus de params, meilleure precision
# Sur RTX 3050 4Go avec batch=8 et imgsz=640, c'est confortable
YOLO_MODEL    = "yolov8s.pt"
EPOCHS        = 200     # Plus d'epochs avec plus de donnees
BATCH         = 8
PATIENCE      = 30      # Plus de patience pour laisser converger
IMGSZ         = 640
WORKERS       = 4

# Augmentation identique a v1 (validee) + un peu plus de rotation
DEGREES       = YOLO_DEGREES     # 15 deg
TRANSLATE     = YOLO_TRANSLATE   # 0.1
SCALE         = YOLO_SCALE       # 0.5
FLIPUD        = YOLO_FLIPUD      # 0.5 — important pour orangs a l'envers
FLIPLR        = YOLO_FLIPLR      # 0.5
HSV_H         = YOLO_HSV_H       # 0.015
HSV_S         = YOLO_HSV_S       # 0.4
HSV_V         = YOLO_HSV_V       # 0.3

# =============================================================================
# CHARGEMENT ET VALIDATION DU CACHE
# =============================================================================

def charger_annotations():
    """
    Charge le cache et retourne uniquement les entrees valides avec
    tous les champs necessaires. Verifie la coherence des coordonnees.
    """
    if not BOXES_CACHE.exists():
        print(f"ERREUR : {BOXES_CACHE} introuvable.")
        sys.exit(1)

    with open(BOXES_CACHE, encoding='utf-8') as f:
        cache = json.load(f)

    annotations = []
    erreurs = []

    for cle, info in cache.items():
        if info.get('statut') != 'valide':
            continue

        # Verifier que tous les champs sont presents
        champs = ['crop_x1','crop_y1','crop_x2','crop_y2','img_w','img_h','photo_source']
        if not all(c in info for c in champs):
            erreurs.append(f"{cle} : champs manquants")
            continue

        x1 = info['crop_x1']; y1 = info['crop_y1']
        x2 = info['crop_x2']; y2 = info['crop_y2']
        W  = info['img_w'];    H  = info['img_h']

        # Verifier que la boite est valide
        if x2 <= x1 or y2 <= y1:
            erreurs.append(f"{cle} : boite invalide x1={x1} x2={x2}")
            continue

        # Convertir en YOLO normalise
        xc = ((x1 + x2) / 2) / W
        yc = ((y1 + y2) / 2) / H
        w  = (x2 - x1) / W
        h  = (y2 - y1) / H

        # Verifier que tout est dans [0, 1]
        if not all(0 < v <= 1 for v in [xc, yc, w, h]):
            # Clamper si tres legerement hors bornes (erreurs d'arrondi)
            xc = max(0.001, min(0.999, xc))
            yc = max(0.001, min(0.999, yc))
            w  = max(0.001, min(0.999, w))
            h  = max(0.001, min(0.999, h))

        # Verifier que la photo source existe
        photo = Path(info['photo_source'])
        if not photo.exists():
            erreurs.append(f"{cle} : photo source introuvable")
            continue

        annotations.append({
            'cle':      cle,
            'individu': info.get('individu', cle.split('/')[0]),
            'stem':     info['stem'],
            'photo':    photo,
            'label':    f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}",
        })

    return annotations, erreurs

# =============================================================================
# VERIFICATION ET RAPPORT DU DATASET
# =============================================================================

def verifier_dataset(annotations, erreurs):
    par_ind = defaultdict(int)
    for a in annotations:
        par_ind[a['individu']] += 1

    print("=" * 60)
    print("DATASET V2 — ANNOTATIONS CORRIGEES MANUELLEMENT")
    print("=" * 60)
    print(f"  Annotations valides  : {len(annotations)}")
    print(f"  Entrees ignorees     : {len(erreurs)}")
    if erreurs:
        for e in erreurs[:5]:
            print(f"    - {e}")

    print(f"\n  {'Individu':<25} {'Images':>7}  {'% du total':>10}")
    print(f"  {'-'*25} {'-'*7}  {'-'*10}")
    total = len(annotations)
    for ind in sorted(par_ind):
        pct = par_ind[ind] / total * 100
        barre = "#" * int(pct / 2)
        print(f"  {ind:<25} {par_ind[ind]:>7}  {pct:>8.1f}%  {barre}")
    print(f"  {'TOTAL':<25} {total:>7}")

    # Verifier l'equilibre entre individus
    counts = list(par_ind.values())
    min_c  = min(counts)
    max_c  = max(counts)
    ratio  = max_c / min_c if min_c > 0 else float('inf')
    print(f"\n  Individu le moins represente : {min_c} images")
    print(f"  Individu le plus represente  : {max_c} images")
    print(f"  Ratio max/min                : {ratio:.1f}x")

    if ratio > 5:
        print("  AVERTISSEMENT : desequilibre important.")
        print("  YOLO est robuste a ce desequilibre pour la detection (1 seule classe).")
        print("  Ce n'est pas un probleme — YOLO apprend a detecter des visages")
        print("  independamment de l'individu.")
    else:
        print("  Distribution equilibree.")

    if len(annotations) < 100:
        print(f"\nERREUR : seulement {len(annotations)} annotations. Minimum : 100.")
        sys.exit(1)

    print("=" * 60)
    return par_ind

# =============================================================================
# CONSTRUCTION DU DATASET YOLO
# =============================================================================

def construire_dataset(annotations):
    """
    Cree DATASET_YOLO_V2/train/ et val/ avec images et labels.
    Split stratifie par individu pour avoir chaque individu dans les deux sets.
    """
    print("\nCONSTRUCTION DU DATASET")

    # Nettoyer si existant
    if DATASET_V2.exists():
        shutil.rmtree(DATASET_V2)
        print(f"  Ancien dataset v2 supprime.")

    for split in ['train', 'val']:
        (DATASET_V2 / split / 'images').mkdir(parents=True)
        (DATASET_V2 / split / 'labels').mkdir(parents=True)

    # Grouper par individu pour le split stratifie
    par_ind = defaultdict(list)
    for a in annotations:
        par_ind[a['individu']].append(a)

    random.seed(RANDOM_SEED)
    train_list = []
    val_list   = []

    for ind, items in sorted(par_ind.items()):
        random.shuffle(items)
        n_train = max(1, int(len(items) * TRAIN_RATIO))
        train_list.extend(items[:n_train])
        val_list.extend(items[n_train:])
        print(f"  {ind:<25} train={n_train:>4}  val={len(items)-n_train:>4}")

    print(f"\n  Total train : {len(train_list)}")
    print(f"  Total val   : {len(val_list)}")

    # Copier images et ecrire labels
    debut = time.time()
    total = len(train_list) + len(val_list)

    def copier_batch(items, split):
        img_dir = DATASET_V2 / split / 'images'
        lbl_dir = DATASET_V2 / split / 'labels'
        for a in items:
            # Copier la photo originale
            dest_img = img_dir / a['photo'].name
            shutil.copy(a['photo'], dest_img)
            # Ecrire le label YOLO
            dest_lbl = lbl_dir / (a['photo'].stem + ".txt")
            dest_lbl.write_text(a['label'] + "\n", encoding='utf-8')

    i = 0
    for a in train_list + val_list:
        i += 1
        pct = i / total
        fill = int(40 * pct)
        elapsed = time.time() - debut
        eta = str(timedelta(seconds=int(elapsed/i*(total-i)))) if i > 0 else "?"
        print(f"\r  Copie [{('='*fill)+('-'*(40-fill))}] {i}/{total}  ETA:{eta}  ",
              end="", flush=True)

    print()

    copier_batch(train_list, 'train')
    copier_batch(val_list,   'val')

    return len(train_list), len(val_list)

# =============================================================================
# GENERATION data.yaml
# =============================================================================

def generer_yaml():
    yaml_path = DATASET_V2 / "data.yaml"
    config = {
        'path':  str(DATASET_V2),
        'train': 'train/images',
        'val':   'val/images',
        'nc':    1,
        'names': ['visage_orang']
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"  data.yaml genere : {yaml_path}")
    return yaml_path

# =============================================================================
# ENTRAINEMENT
# =============================================================================

def entrainer(yaml_path, n_train, n_val):
    from ultralytics import YOLO

    MODELS_V2.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("ENTRAINEMENT YOLOV8 V2")
    print("=" * 60)
    print(f"  Modele de base : {YOLO_MODEL}  (small — meilleur que nano)")
    print(f"  Train          : {n_train} images corrigees manuellement")
    print(f"  Val            : {n_val} images")
    print(f"  Epochs         : {EPOCHS}  (patience early stopping : {PATIENCE})")
    print(f"  Batch / IMGSZ  : {BATCH} / {IMGSZ}")
    print(f"  Workers        : {WORKERS}")
    print(f"  Augmentation   : rotation={DEGREES}° flipud={FLIPUD} fliplr={FLIPLR}")
    print(f"                   hsv_s={HSV_S} (pelage roux preserve)")
    print(f"                   scale={SCALE} translate={TRANSLATE}")
    print(f"  Sortie         : {MODELS_V2}/best.pt")
    print(f"  Run            : {RUN_NAME}")
    print("=" * 60)
    print()
    print("  Note : yolov8s a 11M parametres vs 3M pour nano.")
    print("  Avec 1986 images corrigees, il convergera mieux.")
    print("  L'entrainement durera ~2-3h sur RTX 3050.")
    print()

    confirmation = input("Lancer l'entrainement ? (o/n) : ").strip().lower()
    if confirmation != 'o':
        print("Annule.")
        sys.exit(0)

    print("\nChargement du modele...")
    model = YOLO(YOLO_MODEL)

    print("Debut de l'entrainement...\n")

    model.train(
        data        = str(yaml_path),
        model       = YOLO_MODEL,
        epochs      = EPOCHS,
        imgsz       = IMGSZ,
        batch       = BATCH,
        patience    = PATIENCE,
        workers     = WORKERS,

        # Augmentation
        degrees     = DEGREES,
        translate   = TRANSLATE,
        scale       = SCALE,
        flipud      = FLIPUD,
        fliplr      = FLIPLR,
        hsv_h       = HSV_H,
        hsv_s       = HSV_S,
        hsv_v       = HSV_V,

        # Sauvegarde et logs
        save        = True,
        save_period = 20,
        project     = str(BASE_DIR / 'runs'),
        name        = RUN_NAME,
        exist_ok    = True,
        verbose     = True,
        plots       = True,
        device      = 0,

        # Pas de resume — nouveau modele propre
        resume      = False,
    )

# =============================================================================
# RAPPORT FINAL
# =============================================================================

def rapport_final():
    run_dir = BASE_DIR / 'runs' / RUN_NAME

    if not run_dir.exists():
        print("\nRun introuvable.")
        return

    best_pt = run_dir / 'weights' / 'best.pt'
    if best_pt.exists():
        dest = MODELS_V2 / 'best.pt'
        shutil.copy(best_pt, dest)
        print(f"\n  Meilleur modele : {dest}")
    else:
        print("\n  best.pt introuvable dans le run.")

    # Copier les courbes
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for png in run_dir.glob("*.png"):
        shutil.copy(png, RESULTS_DIR / f"yolo_v2_{png.name}")
        print(f"  Courbe : RESULTS/yolo_v2_{png.name}")

    # Lire les metriques
    results_csv = run_dir / 'results.csv'
    if results_csv.exists():
        import csv
        rows = list(csv.DictReader(open(results_csv)))
        if rows:
            best = max(rows, key=lambda r: float(r.get('metrics/mAP50(B)', 0) or 0))
            map50  = float(best.get('metrics/mAP50(B)', 0))
            prec   = float(best.get('metrics/precision(B)', 0))
            recall = float(best.get('metrics/recall(B)', 0))
            epoch  = best.get('                  epoch', '?')

            print("\n" + "=" * 60)
            print("RESULTATS YOLO V2")
            print("=" * 60)
            print(f"  mAP@0.5    : {map50:.4f}")
            print(f"  Precision  : {prec:.4f}")
            print(f"  Rappel     : {recall:.4f}")
            print(f"  Meilleure epoch : {epoch}")
            print(f"  Epochs total    : {len(rows)}")

            print("\n  Interpretation :")
            if map50 >= 0.95:
                print("  Excellent — pret pour le terrain.")
            elif map50 >= 0.90:
                print("  Tres bon — utilisable sur le terrain.")
            elif map50 >= 0.85:
                print("  Bon — quelques cas difficiles rateront.")
            else:
                print("  Insuffisant — ajouter des annotations difficiles.")

            metriques = {
                "map50": round(map50, 4),
                "precision": round(prec, 4),
                "recall": round(recall, 4),
            }
            log_action("2b_train_yolo_v2", "entrainement_termine", metriques)

    print("\n" + "=" * 60)
    print("ETAPE 2b TERMINEE")
    print("=" * 60)
    print("\nProchaines etapes :")
    print("  Relance 3_extract_faces.py en pointant sur yolo_orangs_v2/best.pt")
    print("  puis lance 4_train_resnet.py")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ENTRAINEMENT YOLO V2 — ANNOTATIONS CORRIGEES MANUELLEMENT")
    print("Orangs-outangs | CNRS IPHC Strasbourg")
    print("=" * 60)

    # 1. Charger et valider les annotations
    print("\nChargement du cache...")
    annotations, erreurs = charger_annotations()

    # 2. Verifier et afficher le rapport
    verifier_dataset(annotations, erreurs)

    # 3. Construire le dataset YOLO
    n_train, n_val = construire_dataset(annotations)

    # 4. Generer data.yaml
    yaml_path = generer_yaml()

    # 5. Entrainer
    entrainer(yaml_path, n_train, n_val)

    # 6. Rapport et sauvegarde
    rapport_final()
