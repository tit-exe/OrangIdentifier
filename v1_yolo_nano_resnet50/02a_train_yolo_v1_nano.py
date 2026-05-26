# =============================================================================
# 2_train_yolo.py
# Entrainement du detecteur de visages YOLOv8 sur les orangs-outangs
# Adapte du pipeline gorilles (Gorilla_IA.md) - sans Roboflow, tout en local
# =============================================================================

import os
import sys
import shutil
import random
import yaml
from pathlib import Path
from collections import defaultdict

# =============================================================================
# CONFIGURATION
# Modifie ces valeurs si necessaire, tout le reste s'adapte automatiquement
# =============================================================================

BASE_DIR     = Path(r"D:\OrangIdentifier")
DATASET_YOLO = BASE_DIR / "DATASET_YOLO"   # images/ et labels/ generes par script 1
SPLIT_DIR    = BASE_DIR / "DATASET_YOLO_SPLIT"  # structure train/val pour YOLO
MODELS_DIR   = BASE_DIR / "MODELS" / "yolo_orangs"
RESULTS_DIR  = BASE_DIR / "RESULTS"

# Split
TRAIN_RATIO  = 0.80   # 80% train, 20% val (meme que gorilles)
RANDOM_SEED  = 42

# Hyperparametres YOLO
# Gorilles : epochs=100, imgsz=640, batch=16, patience=20
# Orangs   : batch=8 (RTX 3050 4Go VRAM), patience=20, rotation plus forte
EPOCHS       = 100
IMGSZ        = 640
BATCH        = 8      # Reduit par rapport aux gorilles (4Go VRAM vs 10Go)
PATIENCE     = 20     # Early stopping : arret si pas d'amelioration pendant 20 epochs

# Augmentation - adaptee aux orangs-outangs en zoo
# Plus de rotation car poses tres variees (tetes a l'envers vues lors de l'annotation)
# HSV reduit par rapport aux gorilles car le pelage roux est une feature discriminante
DEGREES      = 15.0   # Gorilles : 10.0 - augmente car poses variees
TRANSLATE    = 0.1    # Identique aux gorilles
SCALE        = 0.5    # Identique aux gorilles
FLIPUD       = 0.5    # Flip vertical - utile car orangs souvent en position atypique
FLIPLR       = 0.5    # Flip horizontal - identique aux gorilles
HSV_H        = 0.015  # Teinte - reduit (gorilles : 0.015, on garde conservateur)
HSV_S        = 0.4    # Saturation - reduit vs gorilles (0.7) pour preserver le roux
HSV_V        = 0.3    # Luminosite - reduit vs gorilles (0.4)

# =============================================================================
# VERIFICATION INITIALE
# =============================================================================

def get_images_utilisateur():
    """
    Lit done.txt pour obtenir uniquement les images traitees par l utilisateur.
    done.txt contient un stem par ligne, genere par l annotateur et clean_done.py.
    """
    images_dir = DATASET_YOLO / "images"
    done_file  = DATASET_YOLO / "done.txt"

    if not done_file.exists():
        print("  ERREUR : done.txt introuvable.")
        print("  Lance d abord clean_done.py dans scripts_annexes/")
        sys.exit(1)

    stems_valides = set(s.strip() for s in done_file.read_text().splitlines() if s.strip())
    images = [images_dir / (stem + ".jpg") for stem in stems_valides
              if (images_dir / (stem + ".jpg")).exists()]

    print(f"  Images lues depuis done.txt : {len(images)}")
    return images


def verifier_dataset():
    images_dir = DATASET_YOLO / "images"
    labels_dir = DATASET_YOLO / "labels"

    if not images_dir.exists():
        print(f"ERREUR : dossier images introuvable : {images_dir}")
        print("Lance d'abord le script 1_annotation_auto.py")
        sys.exit(1)

    if not labels_dir.exists():
        print(f"ERREUR : dossier labels introuvable : {labels_dir}")
        sys.exit(1)

    # Uniquement les images annotees par l'utilisateur (label non vide)
    # Les skips ont un label vide -> ignores automatiquement
    toutes_images = [
        img for img in get_images_utilisateur()
        if (labels_dir / (img.stem + ".txt")).exists()
        and (labels_dir / (img.stem + ".txt")).stat().st_size > 0
    ]
    print(f"  Images skippes ignores : {len(get_images_utilisateur()) - len(toutes_images)}")
    paires_valides = []
    paires_vides   = []
    paires_sans_label = []
    erreurs_label  = []

    for img in toutes_images:
        lbl = labels_dir / (img.stem + ".txt")
        if not lbl.exists():
            paires_sans_label.append(img.name)
            continue
        contenu = lbl.read_text().strip()
        if not contenu:
            paires_vides.append(img.name)
            continue
        # Verifier format des labels
        ok = True
        for ligne in contenu.splitlines():
            parts = ligne.strip().split()
            if len(parts) != 5:
                ok = False; break
            try:
                vals = list(map(float, parts))
                if any(v < 0 or v > 1 for v in vals[1:]):
                    ok = False; break
            except ValueError:
                ok = False; break
        if ok:
            paires_valides.append(img)
        else:
            erreurs_label.append(img.name)

    print("=" * 60)
    print("VERIFICATION DU DATASET")
    print("=" * 60)
    print(f"  Total images              : {len(toutes_images)}")
    print(f"  Paires valides (annotees) : {len(paires_valides)}")
    print(f"  Images skippees (vides)   : {len(paires_vides)}")
    print(f"  Sans label                : {len(paires_sans_label)}")
    print(f"  Labels corrompus          : {len(erreurs_label)}")

    if len(paires_valides) < 50:
        print(f"\nERREUR : seulement {len(paires_valides)} images valides.")
        print("Il faut au minimum 50 images annotees pour entrainner YOLO.")
        print("Continue l'annotation avec l'annotateur.py")
        sys.exit(1)

    # Compter les boites par individu
    print("\n  Distribution par individu :")
    print(f"  {'Individu':<30} {'Images':>7} {'Boites':>7}")
    print(f"  {'-'*30} {'-'*7} {'-'*7}")

    individus = defaultdict(lambda: {'images': 0, 'boites': 0})
    for img in paires_valides:
        ind = img.stem.split("_")[0]
        lbl = labels_dir / (img.stem + ".txt")
        nb_boites = len([l for l in lbl.read_text().splitlines() if l.strip()])
        individus[ind]['images'] += 1
        individus[ind]['boites'] += nb_boites

    total_boites = 0
    for ind, stats in sorted(individus.items()):
        print(f"  {ind:<30} {stats['images']:>7} {stats['boites']:>7}")
        total_boites += stats['boites']
    print(f"  {'TOTAL':<30} {len(paires_valides):>7} {total_boites:>7}")

    print(f"\n  Ratio moyen boites/image : {total_boites/len(paires_valides):.2f}")
    print("=" * 60)

    return paires_valides

# =============================================================================
# SPLIT TRAIN / VAL
# Stratifie par individu : meme proportion dans train et val
# =============================================================================

def faire_split(paires_valides):
    print("\nPREPARATION DU SPLIT TRAIN/VAL")
    print(f"  Ratio : {int(TRAIN_RATIO*100)}% train / {int((1-TRAIN_RATIO)*100)}% val")
    print(f"  Seed  : {RANDOM_SEED}")

    # Supprimer l'ancien split si existant
    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)
        print(f"  Ancien split supprime : {SPLIT_DIR}")

    # Creer la structure attendue par YOLO
    for split in ['train', 'val']:
        (SPLIT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (SPLIT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

    labels_dir = DATASET_YOLO / "labels"

    # Grouper par individu pour le split stratifie
    par_individu = defaultdict(list)
    for img in paires_valides:
        ind = img.stem.split("_")[0]
        par_individu[ind].append(img)

    random.seed(RANDOM_SEED)
    train_imgs, val_imgs = [], []

    for ind, imgs in par_individu.items():
        random.shuffle(imgs)
        n_train = max(1, int(len(imgs) * TRAIN_RATIO))
        train_imgs.extend(imgs[:n_train])
        val_imgs.extend(imgs[n_train:])

    # Copier les fichiers
    def copier(imgs, split):
        for img in imgs:
            lbl = labels_dir / (img.stem + ".txt")
            shutil.copy(img,  SPLIT_DIR / split / 'images' / img.name)
            shutil.copy(lbl,  SPLIT_DIR / split / 'labels' / (img.stem + ".txt"))

    copier(train_imgs, 'train')
    copier(val_imgs,   'val')

    print(f"  Train : {len(train_imgs)} images")
    print(f"  Val   : {len(val_imgs)} images")

    return len(train_imgs), len(val_imgs)

# =============================================================================
# GENERATION DU data.yaml
# YOLO a besoin de ce fichier pour savoir ou sont les donnees et combien de classes
# =============================================================================

def generer_yaml():
    yaml_path = SPLIT_DIR / "data.yaml"

    config = {
        'path':  str(SPLIT_DIR),
        'train': 'train/images',
        'val':   'val/images',
        'nc':    1,              # 1 seule classe : visage_orang
        'names': ['visage_orang']
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"\n  data.yaml genere : {yaml_path}")
    return yaml_path

# =============================================================================
# ENTRAINEMENT YOLO
# =============================================================================

def entrainer(yaml_path, n_train, n_val):
    from ultralytics import YOLO

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("ENTRAINEMENT YOLOV8")
    print("=" * 60)
    print(f"  Modele    : yolov8n.pt (nano, pre-entraine COCO)")
    print(f"  Train     : {n_train} images")
    print(f"  Val       : {n_val} images")
    print(f"  Epochs    : {EPOCHS} (early stopping patience={PATIENCE})")
    print(f"  Batch     : {BATCH} (reduit pour RTX 3050 4Go)")
    print(f"  Imgsz     : {IMGSZ}")
    print(f"  Rotation  : +/-{DEGREES} deg (augmente vs gorilles car poses variees)")
    print(f"  HSV_S     : {HSV_S} (reduit vs gorilles pour preserver pelage roux)")
    print("=" * 60)

    confirmation = input("\nLancer l'entrainement ? (o/n) : ").strip().lower()
    if confirmation != 'o':
        print("Entrainement annule.")
        sys.exit(0)

    print("\nChargement du modele...")
    model = YOLO('yolov8n.pt')

    print("Debut de l'entrainement...\n")

    results = model.train(
        # Dataset
        data        = str(yaml_path),

        # Architecture
        model       = 'yolov8n.pt',

        # Hyperparametres
        epochs      = EPOCHS,
        imgsz       = IMGSZ,
        batch       = BATCH,
        patience    = PATIENCE,

        # Augmentation adaptee orangs-outangs
        degrees     = DEGREES,
        translate   = TRANSLATE,
        scale       = SCALE,
        flipud      = FLIPUD,
        fliplr      = FLIPLR,
        hsv_h       = HSV_H,
        hsv_s       = HSV_S,
        hsv_v       = HSV_V,

        # Sauvegarde
        save        = True,
        save_period = 10,
        project     = str(BASE_DIR / 'runs'),
        name        = 'orang_face_detector',
        exist_ok    = True,

        # Comportement
        verbose     = True,
        plots       = True,   # Genere automatiquement les courbes
        device      = 0,      # GPU 0 (RTX 3050)
    )

    return results

# =============================================================================
# RAPPORT FINAL
# =============================================================================

def rapport_final():
    run_dir = BASE_DIR / 'runs' / 'orang_face_detector'

    if not run_dir.exists():
        print("\nRun introuvable, verifie le dossier runs/")
        return

    # Chercher le meilleur modele
    best_pt = run_dir / 'weights' / 'best.pt'
    if best_pt.exists():
        dest = MODELS_DIR / 'best.pt'
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_pt, dest)
        print(f"\n  Meilleur modele copie : {dest}")
    else:
        print("\n  best.pt introuvable dans le run")

    # Copier les courbes dans RESULTS/
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for png in run_dir.glob("*.png"):
        shutil.copy(png, RESULTS_DIR / f"yolo_{png.name}")
        print(f"  Courbe copiee : RESULTS/yolo_{png.name}")

    # Lire les metriques finales
    results_csv = run_dir / 'results.csv'
    if results_csv.exists():
        import csv
        rows = list(csv.DictReader(open(results_csv)))
        if rows:
            last = rows[-1]
            best = max(rows, key=lambda r: float(r.get('metrics/mAP50(B)', 0) or 0))
            print("\n" + "=" * 60)
            print("RESULTATS FINAUX")
            print("=" * 60)
            try:
                print(f"  Meilleur mAP@0.5    : {float(best.get('metrics/mAP50(B)', 0)):.4f}")
                print(f"  Meilleure precision  : {float(best.get('metrics/precision(B)', 0)):.4f}")
                print(f"  Meilleur rappel      : {float(best.get('metrics/recall(B)', 0)):.4f}")
                print(f"  Epochs total         : {len(rows)}")

                map50 = float(best.get('metrics/mAP50(B)', 0))
                print("\n  Interpretation :")
                if map50 >= 0.95:
                    print("  Excellent - pret pour l'extraction des visages (script 3)")
                elif map50 >= 0.85:
                    print("  Bon - acceptable, mais annote 100 images de plus pour ameliorer")
                elif map50 >= 0.70:
                    print("  Moyen - annote 200 images de plus avant de continuer")
                else:
                    print("  Insuffisant - verifie la qualite des annotations (boites trop grandes ?)")

            except (ValueError, TypeError):
                print("  Metriques non lisibles, consulte le fichier results.csv")

    print("\n" + "=" * 60)
    print("ETAPE 2 TERMINEE")
    print("=" * 60)
    print("\nProchaine etape : lance 3_extract_faces.py")
    print("Il va utiliser YOLO entraine pour extraire automatiquement")
    print("tous les visages des 2589 photos.")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ETAPE 2 - ENTRAINEMENT DETECTEUR YOLOV8")
    print("Orangs-outangs | Adapte depuis pipeline gorilles")
    print("=" * 60)

    # 1. Verifier et analyser le dataset
    paires_valides = verifier_dataset()

    # 2. Faire le split train/val
    n_train, n_val = faire_split(paires_valides)

    # 3. Generer data.yaml
    yaml_path = generer_yaml()

    # 4. Entrainer
    entrainer(yaml_path, n_train, n_val)

    # 5. Rapport et sauvegarde
    rapport_final()