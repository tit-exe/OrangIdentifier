# QUICKSTART

---

## Before you start (one-time setup)

**Install the environment** (Anaconda Prompt):
```
conda create -n orangs python=3.10 -y
conda activate orangs
pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install timm==0.9.16 ultralytics==8.2.0 Pillow==10.3.0 opencv-python==4.9.0.80 numpy==1.26.4 scikit-learn==1.4.2 umap-learn==0.5.6 matplotlib==3.9.0 PyQt5==5.15.10 tqdm==4.66.4 huggingface_hub==0.23.2
```

**Verify everything works:**
```
python check_env.py
```
All lines should show `[OK]`. If you see `[!!] GPU present but PyTorch has NO CUDA`, run:
```
pip uninstall torch torchvision -y
pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 --index-url https://download.pytorch.org/whl/cu124
```

**Download the pretrained models** (~480 MB, face detector + identity models):
```
python models/download_models.py
```

> Every time you open a new terminal, run `conda activate orangs` and `cd` to the repo first.

---

## Recipe A: Train an identity model from your photos

Use this when you have labeled photos of known individuals and want to train a model that recognizes them.

---

### Step 1: Organize your photos

Create one subfolder per individual inside `data/photos/`:
```
data/photos/
    Molly/
        IMG_001.jpg
        IMG_002.jpg
        ...
    Auti/
        IMG_001.jpg
        ...
    Jula/
        ...
```

Rules:
- Folder name = individual's name (used as the label everywhere)
- Any `.jpg`, `.jpeg`, or `.png` works
- Aim for **at least 30 photos per individual** with varied angles, lighting, distances
- Photos don't need to be cropped. The script detects faces automatically

---

### Step 2: Extract face crops

YOLO scans every photo and saves a 224×224 face crop for each detection:
```
python v1_yolo_nano_resnet50/04_extract_crops.py
```
Type `y` to confirm. The script:
- Reads photos from `data/photos/<Individual>/`
- Saves face crops to `data/crops/known/<Individual>/`
- Creates `data/crops.json` (tracks every crop and its source photo)
- If 2+ faces are detected in one photo → saved separately in `data/crops/known/_a_verifier/` for manual check

At the end it prints a report: how many faces were extracted, how many were missed.

---

### Step 3: Review and validate crops

Open the crop reviewer:
```
python common/review_crops.py
```

**How to use it:**
- Open File Explorer alongside the reviewer window
- Navigate to `data/crops/known/<Individual>/`
- Drag and drop crop images onto the reviewer window (select multiple with Ctrl+A, then drag)
- For each crop:
  - `Enter` : face is good, validate
  - `D` + draw a box : redraw the box if it's off (click and drag), then `Enter`
  - `R` : face is too blurry, partial, wrong angle → reject (keeps the file, marks it rejected)
  - `Del` : delete this crop entirely
  - `S` : skip for now
  - `A` / `←` : go back to previous
  - `Q` : quit

Review all individuals. The goal: keep only clean, full-face crops. Reject blurry, occluded, or wrongly-detected crops.

---

### Step 4: Add background images (recommended)

The model needs "unknown" examples to learn to say "I don't recognize this individual."

**Manually download** unlabeled images of your species from iNaturalist, GBIF, Google Images, etc. and put them in `data/wild_images/raw/`. Read the instructions in `data/wild_images/raw/README.md`.

- Minimum: **1,000 images**
- Recommended: **3,000 – 5,000 images**

Then extract face crops from them:
```
python v3_megadesc_arcface_10ind/01_extract_wild_crops.py
```
Crops go to `data/crops/wild/`.

Optionally review wild crops to remove garbage:
```
python common/review_crops.py
```
For wild crops, `Del` removes both the crop and the source image.

---

### Step 5: Train the identity model

**V3: up to ~15 individuals, ~30 min on RTX 3050:**
```
python v3_megadesc_arcface_10ind/02_train_arcface.py
```

**V4: up to 40+ individuals, ~8h on RTX 3050:**
```
python v4_megadesc_arcface_40ind/02_train_improved.py
```

If the training is interrupted (power cut, Ctrl+C), resume exactly where it stopped:
```
python v3_megadesc_arcface_10ind/02_train_arcface.py --resume
python v4_megadesc_arcface_40ind/02_train_improved.py --resume
```

The trained model is saved to `models/`.

---

### Step 6: Export the gallery

The gallery is a JSON file with one embedding vector per individual.
```
python v3_megadesc_arcface_10ind/03_export_gallery.py   # if you trained V3
python v4_megadesc_arcface_40ind/03_export_gallery.py   # if you trained V4
```

The gallery JSON is saved to `output/`. Load it into the Android app or any inference app.

> **Android app** → [tit0000/OrangIdentifier-Android](https://github.com/tit0000/OrangIdentifier-Android) *(separate repository, coming soon)*

---

## Recipe B: Add new individuals to an existing model

Use this when you already have a trained V4 model and want to add more individuals without retraining from scratch.

**1. Put the new individual's photos** in `data/photos/<NewName>/` (same structure as before)

**2. Extract crops** for the new individuals only:
```
python v4_megadesc_arcface_40ind/01_extract_new_crops.py
```

**3. Review the new crops:**
```
python common/review_crops.py
```

**4. Retrain V4** with all individuals (old + new):
```
python v4_megadesc_arcface_40ind/02_train_improved.py
```

**5. Export the updated gallery:**
```
python v4_megadesc_arcface_40ind/03_export_gallery.py
```

---

## Recipe C: Evaluate the model

After training, check how well the model handles unknown individuals and degraded photos:

```
python v3_megadesc_arcface_10ind/04_test_open_set.py    # rejection rate on unseen individuals
python common/stress_test.py                             # robustness under blur, low-res, JPEG compression, etc.
python common/benchmark.py                               # compare all versions side by side
```

---

## Troubleshooting

**`data/crops.json not found` in the reviewer** → you need to run Step 2 first (extract crops).

**Reviewer opens but drag & drop does nothing** → the crop file you dropped has no entry in `crops.json`. This happens if you drop a file from outside `data/crops/` or if extraction didn't complete. Re-run Step 2.

**`Individuals: ['photos']` in extract_crops** → your folder structure has an extra level. It should be `data/photos/Molly/IMG_001.jpg`, not `data/photos/photos/Molly/IMG_001.jpg`.

**CUDA not detected** → run `check_env.py`, follow the `[!!]` instructions to reinstall PyTorch with CUDA support.

**Training crashes with `individu=null`** → your `crops.json` contains wild crops mixed with known ones. The filter handles this automatically in V3/V4 — update to the latest version of `03_train_yolo_medium.py`.
