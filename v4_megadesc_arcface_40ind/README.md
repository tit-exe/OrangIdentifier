# V4 — MegaDescriptor-T-224 + ArcFace amélioré (40 individus) ★ ACTUEL

## Différences avec V3
1. **40 individus supervisés** au lieu de 10 (10 zoo + 30 BOS Foundation)
2. **Augmentations améliorées** : simulation basse résolution + blur fort
   - Low-res simulation : resize 14-45% puis retour à 224×224
   - GaussianBlur sigma jusqu'à 6.0 (vs 2.5 en V3)
3. **LR plus bas** : backbone 1e-5 (vs 2e-5) car fine-tuning d'un fine-tuning

## Résultats
| Métrique | V3 | V4 |
|----------|-----|-----|
| Val accuracy zoo | 99.2% | 99.2% |
| Rejet BOS inconnus | 97.5% | 97.5% |
| Rejet wild internet | 93.2% | 93.0% |
| Séparabilité gap | 0.883 | 0.885 |
| Basse résolution modérée | 64.2% | 73.3% |
| Combiné léger | 65.8% | 80.0% |
| Individus reconnus | 10 | 40 |

## Modèles
Télécharger via `python models/download_models.py --version v4`
- `megadesc_T_arcface_v4_40individus_acc99.pt`
