# Option B: retrain the brain

Use this only when the new animals look almost identical to each other and
Option A gives too many confusions. Here the brain itself is retrained so it can
tell the animals apart. This is what you would try for lookalike animals such as
the BOS animals.

This is the long path. Be honest with yourself first:

- It needs a graphics card (GPU). On the CPU it would take days.
- It takes several hours, sometimes a full day.
- The last step needs WSL (a small Linux inside Windows).
- It is not guaranteed to work for animals that truly look the same. Retraining
  helps, but if the pictures are too few or too similar, the result can still be
  weak. That is a limit of the data, not a bug.

Before you start, install the tools once:
- Windows training tools: `../00_first_time_setup/1_install_training_tools.md`
- WSL for the export step: `../00_first_time_setup/2_install_wsl_for_export.md`

Nothing here overwrites the current V6 model. The new model is written to
`../new_brain/`, and it only reaches the app in the very last step, when you
choose to deploy it.

## Step 1: put your photos in place

Same as Option A. Go to `../new_animals/1_put_raw_photos_here/` and make one
folder per animal, named with the animal name, with the photos inside. For
lookalike animals, more photos is better. Aim for as many good, varied photos
per animal as you can get.

## Step 2: cut the heads out of the photos

```
conda activate orangs
python maintenance\scripts\crop_photos.py
```

The crops appear in `../new_animals/2_crops_appear_here/`.

## Step 3: check the crops

This step matters more for Option B than for Option A, because bad crops confuse
the training. Open `../new_animals/2_crops_appear_here/` and look through the
folders. Delete any crop that is not a clean, front-facing head of the right
animal. Also open `../new_animals/crop_quality_histograms.png` to see which
animals have many dark or blurry crops.

## Step 4: train the brain

```
conda activate orangs
python maintenance\scripts\train_brain.py
```

This runs for several hours. A live panel shows the progress. Two numbers matter:
- "Clean accuracy": how well it recognizes good pictures.
- "Degraded acc.": how well it recognizes hard pictures (dark, blurry, far).

You can close the window at any time. Run the same command again and it
continues exactly where it stopped. Nothing is lost.

For a quick test that it all runs (a few minutes, not a real model), use:

```
python maintenance\scripts\train_brain.py --dry-run
```

Everything is written to `../new_brain/`:
- `models/new_backbone_only.pt`  the new brain
- `models/new_gallery.json`      the new gallery
- `results/new_report.json`      the final numbers
- `results/new_curves.png`       the training curves

## Step 5: turn the new brain into a phone file (needs WSL)

The phone needs the brain in `.tflite` format. That conversion only works on
Linux, so it runs inside WSL. Full details and the exact command are in
`step_5_export_tflite_wsl.md`. In short, from PowerShell:

```
wsl -d Ubuntu -- bash -c "/root/miniconda3/bin/conda run -n orangs_export --no-capture-output python /mnt/d/<path-to-your-repo>/maintenance/scripts/export_to_tflite.py"
```

This makes the `.tflite` file and copies both the new brain and the new gallery
into the app.

## Step 6: rebuild the app

1. Open the project in Android Studio.
2. Build, then Clean Project.
3. Build, then Rebuild Project.
4. Run on a phone and test.

## If you want to go back

The old model is untouched in the `models/` folder (the downloaded V6 model). If the new model is
worse, copy the old `v6_backbone.tflite` and `v6_gallery.json` back into the app
assets and rebuild. See `../reference/file_map.md` for the exact file names.

## If something goes wrong

See `../reference/troubleshooting.md`.
