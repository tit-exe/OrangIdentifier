# Scripts

These are the programs the steps use. You do not run them by guessing. The
Option A and Option B guides tell you exactly when and how to run each one, with
the full command to copy. This page is just a summary.

Run order for Option A (add individuals):

1. `crop_photos.py`     cut heads from your photos
2. `build_gallery.py`   rebuild the gallery and copy it into the app

Run order for Option B (retrain the brain):

1. `crop_photos.py`       cut heads from your photos
2. `train_brain.py`       retrain the brain (needs a graphics card, several hours)
3. `export_to_tflite.py`  make the phone file and copy it into the app (needs WSL)

What each script does:

| Script | Does | Runs in |
|---|---|---|
| `crop_photos.py` | finds the head in each photo and saves a 224x224 crop | Windows, `orangs` env |
| `build_gallery.py` | builds the gallery from the crops, using the existing brain | Windows, `orangs` env |
| `train_brain.py` | trains a new brain from scratch with all animals | Windows, `orangs` env, GPU |
| `export_to_tflite.py` | turns the new brain into the `.tflite` phone file | WSL, `orangs_export` env |

Each script has a settings block near the top with the folder paths. You only
need to touch it if your folders are somewhere other than the default
`the repository root`.

Nothing here overwrites the current V6 model. New results go to
`maintenance\new_brain\` (Option B) or `maintenance\new_animals\updated_gallery.json`
(Option A). The app is only changed at the final copy step.
