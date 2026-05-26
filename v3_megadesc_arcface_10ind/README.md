# V3 — MegaDescriptor-T-224 + Sub-center ArcFace (10 individus)

## Architecture
- **Détection** : YOLO v2 medium (inchangé)
- **Backbone** : MegaDescriptor-T-224 (Swin-Tiny, 27.5M params, 768-dim)
  - Pré-entraîné sur réidentification animale (Čermák et al. WACV 2024)
- **Loss** : Sub-center ArcFace (K=1 connus, K=5 wild) scale=64 margin=0.5
- **Dataset** : 2127 crops zoo + 5429 wild crops internet (classe background)

## Résultats
| Métrique | Valeur |
|----------|--------|
| Val accuracy (nearest prototype) | 99.06% |
| Séparabilité gap | 0.9586 |
| Rejet BOS inconnus | 96.3% (1622 crops, 30 individus jamais vus) |
| Rejet wild internet | 93.2% |
| Temps inférence (RTX 3050) | ~17ms/image |

## Robustesse (stress test)
| Dégradation | Légère | Modérée |
|-------------|--------|---------|
| Flou | 98.3% | 25.0% |
| Basse résolution | 98.3% | 64.2% |
| Rotation | 99.2% | 99.2% |
| Exposition | 99.2% | 98.3% |

## Modèles
Télécharger via `python models/download_models.py --version v3`
- `megadesc_T_arcface_final_epoch21_acc99.pt`
