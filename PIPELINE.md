# PIPELINE — Guide étape par étape

## Prérequis

```bash
conda create -n orangs python=3.10
conda activate orangs
pip install -r requirements.txt
python setup.py       # vérifie l'environnement
python models/download_models.py
```

## V1 — YOLO + ResNet50 classifieur fermé

```bash
cd v1_yolo_nano_resnet50
python 01_annotation_auto.py   # annote les photos brutes
python 02_train_yolo.py        # entraîne YOLO nano (1h)
python 02b_train_yolo_v2.py    # entraîne YOLO medium (2h)
python 03_extract_faces.py     # extrait les crops
python common/review_crops.py <dossier_crops>  # révision manuelle
python 04_train_resnet.py      # entraîne ResNet50 (30min)
python 05_export_tflite.py     # export Android
```

## V2 — ResNet50 + galerie embeddings open-set

```bash
cd v2_resnet50_embeddings_openset
# Nécessite d'avoir fait V1 d'abord
python 06_build_embeddings.py  # construit la galerie + calibre le seuil
```

## V3 — MegaDescriptor + Sub-center ArcFace (10 individus)

```bash
cd v3_megadesc_arcface_10ind
python 01_download_wild.py     # télécharge images internet (6000+)
python 03_extract_faces_wild.py  # YOLO sur les images wild
python common/review_crops.py <wild_crops>  # révision manuelle
python 04_train_arcface.py     # entraînement ~30min
python 07_test_open_set.py     # test sur individus jamais vus
python 08_visualize.py         # graphiques diagnostics
python 06_export_gallery.py    # génère embeddings.json
```

## V4 — MegaDescriptor + ArcFace amélioré (40 individus)

```bash
cd v4_megadesc_arcface_40ind
# Nécessite d'avoir des crops BOS Foundation labelisés
python 02_extract_new_individuals.py  # YOLO sur photos BOS
python common/review_crops.py <bos_crops>  # révision
python 01_train_improved.py    # entraînement ~8h
python 02_stress_test.py       # test robustesse
python 03_export_gallery.py    # galerie 40 individus
```

## Benchmark comparatif

```bash
python common/benchmark.py     # compare V1 à V4
```

## Adapter à une autre espèce

1. Modifier `config.yaml` (species, chemins)
2. Photographier les individus (>50 photos par individu recommandé)
3. Suivre la pipeline V3 ou V4
4. Le modèle de détection YOLO est entraîné par espèce
