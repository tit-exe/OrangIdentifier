# V4: MegaDescriptor-T-224 + improved ArcFace (40 individuals) ★ CURRENT

## Differences from V3
1. **40 supervised individuals** instead of 10 (10 zoo + 30 BOS Foundation)
2. **Improved augmentations**: low-resolution simulation + strong blur
   - Low-res simulation: resize to 14–45% then back to 224×224
   - GaussianBlur sigma up to 6.0 (vs 2.5 in V3)
3. **Lower LR**: backbone 1e-5 (vs 2e-5) because it is fine-tuning a fine-tuning

## Results
| Metric | V3 | V4 |
|--------|-----|-----|
| Zoo val accuracy | 99.2% | 99.2% |
| BOS unknown rejection | 97.5% | 97.5% |
| Wild internet rejection | 93.2% | 93.0% |
| Separability gap | 0.883 | 0.885 |
| Moderate low resolution | 64.2% | 73.3% |
| Light combined | 65.8% | 80.0% |
| Recognized individuals | 10 | 40 |

## Models
Download with `python models/download_models.py --version v4`
- `megadesc_T_arcface_v4_40individuals_acc99.pt`
