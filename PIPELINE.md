# PIPELINE

## Prerequisites

```bash
conda create -n orangs python=3.10
conda activate orangs
pip install -r requirements.txt
python check_env.py              # verify environment
python models/download_models.py # download pretrained weights
```

## V1: YOLO + ResNet50 closed-set classifier

```bash
python v1_yolo_nano_resnet50/01_auto_annotate.py    # auto-annotate with generic YOLO
python tools/annotate_keyboard.py                   # manual correction in LabelImg-style tool
python v1_yolo_nano_resnet50/02_train_yolo_nano.py  # train YOLO nano (1h), quick test
python v1_yolo_nano_resnet50/04_extract_crops.py    # extract crops → data/crops/known/
python common/review_crops.py                       # manual crop review (drag & drop)
python v1_yolo_nano_resnet50/03_train_yolo_medium.py # train YOLO medium from corrected boxes (2h)
python v1_yolo_nano_resnet50/05_train_resnet50.py   # train ResNet50 closed-set (30min)
python v1_yolo_nano_resnet50/06_export_tflite.py    # export to TFLite format (see OrangIdentifier-Android repo)
```

## V2: ResNet50 + open-set embedding gallery

```bash
# Requires V1 first
python v2_resnet50_embeddings_openset/01_build_gallery.py  # build gallery + calibrate threshold
```

## V3: MegaDescriptor + Sub-center ArcFace (10 individuals)

```bash
# Put images in data/wild_images/raw/ (see README.md in that folder, aim for 3000+)
python v3_megadesc_arcface_10ind/01_extract_wild_crops.py    # YOLO on wild images
python common/review_crops.py                                # review wild crops (drag & drop)
python v3_megadesc_arcface_10ind/02_train_arcface.py         # train (~30min on RTX 3050)
python v3_megadesc_arcface_10ind/02_train_arcface.py --resume  # resume interrupted training
python v3_megadesc_arcface_10ind/03_export_gallery.py        # export gallery.json → load into Android app (see OrangIdentifier-Android repo)
python v3_megadesc_arcface_10ind/04_test_open_set.py         # test on unseen individuals
python v3_megadesc_arcface_10ind/05_visualize.py             # diagnostic plots
```

Optional: recover missed detections
```bash
```

## V4: MegaDescriptor + improved ArcFace (40 individuals)

```bash
# Put new individual photos in data/photos/<IndividualName>/
python v4_megadesc_arcface_40ind/01_extract_new_crops.py  # YOLO on new individual photos
python common/review_crops.py                             # review new crops (drag & drop)
python v4_megadesc_arcface_40ind/02_train_improved.py     # train improved V4 (~8h)
python v4_megadesc_arcface_40ind/02_train_improved.py --resume  # resume if interrupted
python v4_megadesc_arcface_40ind/03_export_gallery.py     # export V4 gallery
```

## V5: MegaDescriptor + invariance training (40 individuals)

```bash
# Uses data/crops/known (zoo) + data/crops/bos (rescue center) + data/crops/wild
python v5_megadesc_arcface_invariance/01_train.py            # train with invariance + curriculum (GPU, several hours)
python v5_megadesc_arcface_invariance/01_train.py --dry-run  # quick check
python v5_megadesc_arcface_invariance/02_export_tflite.py    # export backbone to TFLite (WSL)
```

## V6: production model, 15 zoo individuals (deployed)

```bash
# Crop the 5 new zoo individuals and score their quality
python v6_megadesc_arcface_15ind/01_extract_crops.py
python v6_megadesc_arcface_15ind/01b_brighten.py             # brighten dark crops (CLAHE)
python common/review_crops.py                                # review / correct boxes
python v6_megadesc_arcface_15ind/01c_review_quality.py       # keep only good-quality crops
python v6_megadesc_arcface_15ind/02_train.py                 # train the production model (GPU, several hours)
python v6_megadesc_arcface_15ind/03_tune_threshold.py        # calibrate the acceptance threshold
python v6_megadesc_arcface_15ind/04_test_open_set.py         # test rejection of unknown animals
python v6_megadesc_arcface_15ind/05_export_tflite.py         # export to .tflite for the app (WSL)
```

> **To update the deployed app** (add animals or retrain) without reading the research
> scripts, follow the plain-language guide in **`maintenance/`**. It wraps the same steps
> for a non-specialist.

## Adapting to another species

1. Edit `config.yaml`: change `species` and `project_name`
2. Put the photos in `data/photos/<IndividualName>/`
3. Photograph each individual (>50 photos per individual recommended)
4. Follow the V3 to V6 pipeline. MegaDescriptor-T works well on new species with limited data
5. The YOLO detection model is trained per species (V1 step)
