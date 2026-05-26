import os
import sys
import cv2
import shutil
from pathlib import Path
from ultralytics import YOLO

# ================================================================
# CONFIGURATION
# ================================================================

PHOTOS_DIR  = r"D:\OrangIdentifier\PHOTOS"
OUTPUT_DIR  = r"D:\OrangIdentifier\DATASET_YOLO"
MODELE_YOLO = "yolov8n.pt"       # telecharge automatiquement
CONFIANCE   = 0.15               # seuil bas expres pour ne rien rater
# Classes ImageNet qui ressemblent a des animaux
CLASSES_ANIMAUX = list(range(15, 30))

# ================================================================
# PREPARATION DES DOSSIERS
# ================================================================

images_dir = Path(OUTPUT_DIR) / "images"
labels_dir = Path(OUTPUT_DIR) / "labels"
images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("ETAPE 1 - ANNOTATION AUTOMATIQUE")
print("Modele : YOLOv8n generique (ImageNet)")
print(f"Source : {PHOTOS_DIR}")
print(f"Output : {OUTPUT_DIR}")
print("=" * 60)

# ================================================================
# CHARGEMENT DU MODELE
# ================================================================

print("\nChargement du modele YOLO generique...")
model = YOLO(MODELE_YOLO)
print("Modele charge.")

# ================================================================
# TRAITEMENT DE TOUTES LES PHOTOS
# ================================================================

photos_dir = Path(PHOTOS_DIR)
individus = [d for d in photos_dir.iterdir() if d.is_dir()]

total_images     = 0
total_annotees   = 0
total_ratees     = 0
stats_individus  = {}

print(f"\n{len(individus)} individus trouves : {[d.name for d in individus]}")
print("\nDemarrage de la detection...\n")

for individu_dir in sorted(individus):
    nom = individu_dir.name
    photos = list(individu_dir.glob("*.jpg")) + \
             list(individu_dir.glob("*.jpeg")) + \
             list(individu_dir.glob("*.png"))

    annotees = 0
    ratees   = 0

    print(f"[{nom}] - {len(photos)} photos")

    for photo_path in photos:
        total_images += 1

        # Lecture image
        img = cv2.imread(str(photo_path))
        if img is None:
            print(f"  ERREUR lecture : {photo_path.name}")
            ratees += 1
            continue

        h, w = img.shape[:2]

        # Detection YOLO generique
        results = model.predict(
            source=str(photo_path),
            conf=CONFIANCE,
            classes=CLASSES_ANIMAUX,
            verbose=False
        )

        boxes = results[0].boxes

        # Nom de fichier de sortie (sans espaces pour YOLO)
        nom_propre = nom.replace(" ", "_").replace("(", "").replace(")", "")
        nom_fichier = f"{nom_propre}_{photo_path.stem}"

        # Copie de l'image dans DATASET_YOLO/images/
        dest_img = images_dir / f"{nom_fichier}.jpg"
        shutil.copy(str(photo_path), str(dest_img))

        # Creation du fichier label YOLO
        dest_label = labels_dir / f"{nom_fichier}.txt"

        if boxes is not None and len(boxes) > 0:
            with open(dest_label, 'w') as f:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x_centre = ((x1 + x2) / 2) / w
                    y_centre = ((y1 + y2) / 2) / h
                    largeur  = (x2 - x1) / w
                    hauteur  = (y2 - y1) / h
                    # Classe 0 = visage_orang
                    f.write(f"0 {x_centre:.6f} {y_centre:.6f} {largeur:.6f} {hauteur:.6f}\n")
            annotees += 1
        else:
            # Fichier label vide = image sans detection
            # On la garde quand meme pour la corriger dans LabelImg
            open(dest_label, 'w').close()
            ratees += 1

    stats_individus[nom] = {'annotees': annotees, 'ratees': ratees, 'total': len(photos)}
    total_annotees += annotees
    total_ratees   += ratees

    print(f"  -> {annotees}/{len(photos)} detections ({ratees} a corriger manuellement)")

# ================================================================
# RAPPORT FINAL
# ================================================================

print("\n" + "=" * 60)
print("RAPPORT FINAL")
print("=" * 60)
print(f"Images traitees    : {total_images}")
print(f"Detections auto    : {total_annotees} ({100*total_annotees/max(total_images,1):.1f}%)")
print(f"A corriger manuell : {total_ratees}  ({100*total_ratees/max(total_images,1):.1f}%)")

print("\nDetail par individu :")
print(f"  {'Individu':<35} {'Auto':>6} {'Manuel':>8} {'Total':>7}")
print(f"  {'-'*35} {'-'*6} {'-'*8} {'-'*7}")
for nom, s in sorted(stats_individus.items()):
    print(f"  {nom:<35} {s['annotees']:>6} {s['ratees']:>8} {s['total']:>7}")

print("\n" + "=" * 60)
print("ETAPE 1 TERMINEE")
print("=" * 60)
print(f"\nTon dossier DATASET_YOLO est pret :")
print(f"  Images  -> {images_dir}")
print(f"  Labels  -> {labels_dir}")
print(f"\nPROCHAINES ETAPES :")
print(f"  1. Ouvre LabelImg")
print(f"  2. Charge le dossier : {images_dir}")
print(f"  3. Charge les labels : {labels_dir}")
print(f"  4. Verifie et corrige les annotations incorrectes")
print(f"  5. Lance 2_train_yolo.py quand tu as corrige ~300 images")