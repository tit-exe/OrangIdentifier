# V5: MegaDescriptor-T-224 + ArcFace with invariance training (40 individuals)

## What changed vs V4

V3 and V4 are strong on clean images but collapse on degraded ones (blur, low
resolution). V5 attacks that weakness in the training objective itself.

- **Invariance loss.** Every crop is passed through the network twice, once clean
  and once degraded, and the two embeddings are forced to be equal. This directly
  teaches the model that a blurred face and its clean version are the same animal.
- **Degradation curriculum.** The amount of degradation is raised stage by stage
  during training (4 phases), gentle first, harsh later.
- **BOS spreading loss.** Pushes the look-alike rescue-center animals apart.
- **Exemplar gallery.** Each individual is stored as 15 to 20 reference vectors
  instead of a single average, and scoring takes the best match.

## Architecture

- Backbone: MegaDescriptor-T-224 (Swin-Tiny, 768-dim), started from the V3 model.
- Loss: Sub-center ArcFace, margin 0.35, K = 2 zoo / 1 BOS / 5 wild.
- Individuals: 40 supervised (10 zoo + 30 rescue-center), wild crops as background.
- Normalization: mean = std = 0.5 (MegaDescriptor, NOT ImageNet).

## Role

V5 is the turning point of the project: the invariance loss, the curriculum and
the exemplar gallery are exactly what make the production model (V6) robust.

## Scripts

```
python v5_megadesc_arcface_invariance/01_train.py            # train (GPU, several hours)
python v5_megadesc_arcface_invariance/01_train.py --dry-run  # quick check
python v5_megadesc_arcface_invariance/02_export_tflite.py    # export backbone (WSL, see maintenance/)
```

## Data used

`data/crops/known` (10 zoo) + `data/crops/bos` (30) + `data/crops/wild` (background).

## Models

```
python models/download_models.py --version v5
```

