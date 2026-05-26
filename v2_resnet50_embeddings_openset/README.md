# V2 — ResNet50 + galerie embeddings open-set

## Architecture
- **Détection** : YOLO v2 medium (inchangé depuis V1)
- **Backbone** : ResNet50 de V1 SANS la tête fc
- **Embedding** : vecteurs 2048-dim, L2-normalisés
- **Galerie** : prototypes moyennés par individu
- **Seuil** : calibré par leave-one-individual-out (F1 max)

## Résultats
| Métrique | Valeur |
|----------|--------|
| Accuracy zoo (nearest prototype) | ~98% |
| ROC AUC | 0.9821 |
| Séparabilité | 1.7203 |
| Seuil calibré | 0.4885 |
| BOS rejection | 27.5% ⚠️ |

## Innovation
Premier système open-set : peut dire "inconnu".
Ajouter un individu = calculer son prototype (10 min, zero réentraînement).

## Limitation
Robustesse insuffisante aux inconnus BOS : 27.5% seulement.
→ Résolu en V3 avec ArcFace et wild crops.

## Modèles
Télécharger via `python models/download_models.py --version v2`
- `resnet50_backbone_2048dim.pt`
