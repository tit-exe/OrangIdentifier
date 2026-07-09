# V6: production model — 15 zoo individuals

**This is the deployed model.** Its backbone and gallery are the files that run in
the Android app.

## What changed vs V5

- **Zoo only.** 15 individuals (the 10 original + 5 new: Bukit, Indah, Kembali,
  Rosa, Yori). The rescue-center (BOS) animals are removed from the shipped
  gallery; they are treated as unknown again.
- **Crop quality pipeline.** Every crop is scored for brightness and sharpness,
  and dark crops are brightened (CLAHE), so only usable faces enter the gallery.
- **25 exemplars per individual** in the gallery.
- **Threshold tuned by a field simulation** instead of by hand.
- Keeps the V5 methods: invariance loss, degradation curriculum.

## Architecture

- Backbone: MegaDescriptor-T-224 (768-dim), started from the V3 model.
- Loss: Sub-center ArcFace, margin 0.35, K = 2 zoo / 5 wild.
- Normalization: mean = std = 0.5.

## Results (validation)

| Metric | Value |
|--------|-------|
| Identification accuracy (15 individuals) | 98.44% |
| Accuracy on degraded images | about 96% (vs about 19% for V3) |
| Average confidence | 0.91 |
| Wild internet false acceptance | 1% |
| Acceptance threshold | 0.5371 |
| Reference vectors per individual | 25 |

## Scripts (run in order)

```
python v6_megadesc_arcface_15ind/01_extract_crops.py      # crop new photos + quality metrics
python v6_megadesc_arcface_15ind/01b_brighten.py          # brighten dark crops (CLAHE)
python common/review_crops.py                             # review/correct boxes (universal reviewer)
python v6_megadesc_arcface_15ind/01c_review_quality.py    # keep only good-quality crops
python v6_megadesc_arcface_15ind/02_train.py              # train (GPU, several hours)
python v6_megadesc_arcface_15ind/03_tune_threshold.py     # calibrate the acceptance threshold
python v6_megadesc_arcface_15ind/04_test_open_set.py      # test rejection of unknown animals
python v6_megadesc_arcface_15ind/05_export_tflite.py      # export to .tflite (WSL, see maintenance/)
python v6_megadesc_arcface_15ind/06_field_simulation.py   # recognition vs number of reference photos
```

## Data used

`data/crops/known` (10 zoo) + `data/crops/new` (5 new) + `data/crops/wild` (background).
The rescue-center crops in `data/crops/bos` are used only to test rejection.

## Models

```
python models/download_models.py --version v6
```
