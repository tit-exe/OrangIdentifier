# File map

What every important file is, and where it lives. Paths start from
the repository root.

## The files inside the app

Folder: the assets folder of the separate Android app repository (OrangIdentifier-Android)

| File | What it is | Changed by |
|---|---|---|
| `megadesc_v6_backbone.tflite` | the brain, the phone version | Option B (export) |
| `gallery.json` | the list of known animals | Option A and Option B |
| `yolo_v2_detector.tflite` | the head detector | never |

The app always loads these exact names. Even a new brain keeps the name
`megadesc_v6_backbone.tflite`.

## The current V6 model (the one in use)

Folder: `v6\models\`

| File | What it is |
|---|---|
| `v6_backbone.tflite` | the V6 phone brain (safe backup of the brain) |
| `v6_backbone_only.pt` | the V6 brain, PC version (used by build_gallery) |
| `v6_best.pt` | the full V6 model (brain plus recognition head) |
| `v6_gallery.json` | the original V6 gallery (safe backup) |
| `v6_resume.pt` | the training checkpoint of V6 |

Nothing in the maintenance folder overwrites these. They are your safe copy to
go back to.

## The new model you produce with Option B

Folder: `maintenance\new_brain\models\`

| File | What it is |
|---|---|
| `new_backbone_only.pt` | the new brain, PC version |
| `new_backbone.tflite` | the new brain, phone version |
| `new_best.pt` | the new full model |
| `new_gallery.json` | the new gallery |
| `new_resume.pt` | the training checkpoint (lets you continue if stopped) |

## The photos and crops

| Folder | What it is |
|---|---|
| `maintenance\new_animals\1_put_raw_photos_here\` | you put raw photos here |
| `maintenance\new_animals\2_crops_appear_here\` | cropping fills this |
| `v1\data\crops_dataset\raw\` | the 10 original animals (crops) |
| `v6\data\new_zoo_crops\` | the 5 animals added in V6 (crops) |
| `v3\data\wild_crops\` | wild pictures, used to teach "unknown" during training |

## The scripts

Folder: `maintenance\scripts\`

| Script | What it does | Where it runs |
|---|---|---|
| `crop_photos.py` | cut heads from photos | Windows (orangs env) |
| `build_gallery.py` | rebuild the gallery, Option A | Windows (orangs env) |
| `train_brain.py` | retrain the brain, Option B | Windows (orangs env), GPU |
| `export_to_tflite.py` | make the .tflite and copy to app | WSL (orangs_export env) |

## Other models used

| File | What it is |
|---|---|
| `v2\models\yolo_v2_medium_mAP99.pt` | the head detector, used by crop_photos.py |
| `v3\models\megadesc_T_arcface_final_epoch21_acc99.pt` | the V3 brain, the starting point for training |

## How to go back to the old model

If a new model is worse, copy the old files back into the app assets:

1. Copy `v6\models\v6_backbone.tflite` into the app assets, renamed to
   `megadesc_v6_backbone.tflite`.
2. Copy `v6\models\v6_gallery.json` into the app assets, renamed to
   `gallery.json`.
3. Rebuild the app in Android Studio.
