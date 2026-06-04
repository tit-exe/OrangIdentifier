# V2: ResNet50 + open-set embedding gallery

## Architecture
- **Detection**: YOLO v2 medium (unchanged from V1)
- **Backbone**: ResNet50 from V1 WITHOUT the fc head
- **Embedding**: 2048-dim vectors, L2-normalized
- **Gallery**: per-individual averaged prototypes
- **Threshold**: calibrated by leave-one-individual-out (max F1)

## Results
| Metric | Value |
|--------|-------|
| Zoo accuracy (nearest prototype) | ~98% |
| ROC AUC | 0.9821 |
| Separability | 1.7203 |
| Calibrated threshold | 0.4885 |
| BOS rejection | 27.5% ⚠️ |

## Innovation
First open-set system: can say "unknown".
Adding an individual = computing its prototype (10 min, zero retraining).

## Limitation
Insufficient robustness to BOS unknowns: only 27.5%.
→ Solved in V3 with ArcFace and wild crops.

## Models
Download with `python models/download_models.py --version v2`
- `resnet50_backbone_2048dim.pt`
