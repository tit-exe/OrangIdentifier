# V4: MegaDescriptor-T-224 + ArcFace (40 individuals)

## Differences from V3
1. **40 supervised individuals** instead of 10 (10 zoo + 30 rescue-center BOS)
2. **Stronger augmentations**: low-resolution simulation and heavier blur
   - Low-res simulation: resize to 14 to 45% then back to 224x224
   - GaussianBlur sigma up to 6.0 (vs 2.5 in V3)
3. **Lower learning rate**: backbone 1e-5 (vs 2e-5), because it continues from V3

## Results (fair cross-version benchmark)
| Metric | V3 | V4 |
|--------|-----|-----|
| Clean identification (10 zoo) | 99.2% | 99.2% |
| Separability gap (10 zoo) | 0.85 | 0.86 |
| Moderate low resolution | 55% | 62% |
| Light combined degradation | 59% | 63% |
| Supervised individuals | 10 | 40 |

The extra augmentation gave only a small robustness gain over V3, and both versions still
collapse under moderate blur (about 11%). Real robustness only arrived with the invariance
training of V5.

## Important: no BOS rejection figure for V4
V4 made the 30 rescue-center (BOS) animals **supervised classes**. It can therefore no
longer be tested for "rejecting" them, since it was trained to recognise them. Any BOS
rejection number reported for V4 would be data leakage.

On its full native task (40 individuals) V4 reaches only **73.7% micro / 55.6% macro**
accuracy: the rescue-center animals are extremely hard to tell apart (their separability
gap is essentially zero). This is what motivated V5, and ultimately the zoo-only V6.

## Models
Download with `python models/download_models.py --version v4`
- `megadesc_T_arcface_v4_40individuals_acc99.pt`
