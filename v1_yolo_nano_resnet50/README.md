# V1 — YOLO nano/medium + ResNet50 classifieur fermé

## Architecture
- **Détection** : YOLOv8 nano (mAP@50=91.98%) puis medium (mAP@50=99.39%)
- **Identification** : ResNet50 ImageNet fine-tuné, tête fc Linear(2048→10)
- **Loss** : Cross-entropy + label smoothing 0.05 + Mixup
- **Dataset** : 2127 crops zoo, 10 individus

## Résultats
| Métrique | Valeur |
|----------|--------|
| Accuracy test set | 96.3% |
| YOLO v1 mAP@50 | 91.98% |
| YOLO v2 mAP@50 | 99.39% |
| Temps inférence (RTX 3050) | ~12ms/image |

## Limitation principale
**Classifieur FERMÉ** : dit toujours un nom même pour un individu inconnu.
Inutilisable sur le terrain. → Résolu en V2 et V3.

## Modèles
Télécharger via `python models/download_models.py --version v1`
- `yolo_v1_nano_mAP92.pt`
- `yolo_v2_medium_mAP99.pt`
- `resnet50_classifier_10classes_acc96.pt`
