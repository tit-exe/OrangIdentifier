# V2: ResNet50 + open-set embedding gallery

## Architecture
- **Detection**: YOLO v2 medium (unchanged from V1)
- **Backbone**: ResNet50 from V1 without the fc head
- **Embedding**: 2048-dim vectors, L2-normalized
- **Gallery**: per-individual averaged prototypes
- **Threshold**: calibrated by leave-one-individual-out (max F1)

## Results (fair cross-version benchmark)
| Metric | Value |
|--------|-------|
| Clean identification (10 zoo) | 96.5% |
| Separability gap | 0.23 |
| Open-set rejection (ROC AUC) | 0.83 |
| Calibrated threshold | 0.4885 |

## Innovation
First open-set system: it can answer "unknown".
Adding an individual means computing its prototype (a few minutes, zero retraining).

## Limitation
Weak open-set rejection compared with the later ArcFace models: the ResNet50 features
were not optimised for embeddings. Solved in V3 with Sub-center ArcFace and wild
background crops.

## Models
Download with `python models/download_models.py --version v2`
- `resnet50_backbone_2048dim.pt`
