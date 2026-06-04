# V3: MegaDescriptor-T-224 + Sub-center ArcFace (10 individuals)

## Architecture
- **Detection**: YOLO v2 medium (unchanged)
- **Backbone**: MegaDescriptor-T-224 (Swin-Tiny, 27.5M params, 768-dim)
  - Pretrained on animal re-identification (Čermák et al. WACV 2024)
- **Loss**: Sub-center ArcFace (K=1 known, K=5 wild) scale=64 margin=0.5
- **Dataset**: 2127 zoo crops + 5429 wild internet crops (background class)

## Results
| Metric | Value |
|--------|-------|
| Val accuracy (nearest prototype) | 99.06% |
| Separability gap | 0.9586 |
| BOS unknown rejection | 96.3% (1622 crops, 30 never-seen individuals) |
| Wild internet rejection | 93.2% |
| Inference time (RTX 3050) | ~17ms/image |

## Robustness (stress test)
| Degradation | Light | Moderate |
|-------------|-------|----------|
| Blur | 98.3% | 25.0% |
| Low resolution | 98.3% | 64.2% |
| Rotation | 99.2% | 99.2% |
| Exposure | 99.2% | 98.3% |

## Models
Download with `python models/download_models.py --version v3`
- `megadesc_T_arcface_final_epoch21_acc99.pt`
