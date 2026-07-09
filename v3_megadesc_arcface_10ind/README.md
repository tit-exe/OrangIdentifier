# V3: MegaDescriptor-T-224 + Sub-center ArcFace (10 individuals)

## Architecture
- **Detection**: YOLO v2 medium (unchanged)
- **Backbone**: MegaDescriptor-T-224 (Swin-Tiny, 27.5M params, 768-dim)
  - Pretrained on animal re-identification (Čermák et al. WACV 2024)
- **Loss**: Sub-center ArcFace (K=1 known, K=5 wild), scale=64, margin=0.5
- **Dataset**: 2127 zoo crops + 5429 wild internet crops (background class)

## Results (fair cross-version benchmark)
| Metric | Value |
|--------|-------|
| Clean identification (10 zoo) | 99.2% |
| Separability gap | 0.85 |
| Open-set rejection (ROC AUC, never-seen individuals) | 0.998 |

This is the reference embedding model: the jump from ResNet50 (separability gap 0.23)
to MegaDescriptor with Sub-center ArcFace (0.85) is the single largest improvement of
the project, and open-set rejection becomes near-perfect.

## Robustness limitation
V3 is fragile to degraded photos. Identification drops to chance level (about 11% for a
10-way problem) as soon as blur becomes moderate, and to about 55% under moderate low
resolution. This weakness is what motivated the invariance training introduced in V5.

## Models
Download with `python models/download_models.py --version v3`
- `megadesc_T_arcface_final_epoch21_acc99.pt`
