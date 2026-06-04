# V1: YOLO nano/medium + ResNet50 closed-set classifier

## Architecture
- **Detection**: YOLOv8 nano (mAP@50=91.98%) then medium (mAP@50=99.39%)
- **Identification**: ResNet50 ImageNet fine-tuned, fc head Linear(2048→10)
- **Loss**: Cross-entropy + label smoothing 0.05 + Mixup
- **Dataset**: 2127 zoo crops, 10 individuals

## Results
| Metric | Value |
|--------|-------|
| Test set accuracy | 96.3% |
| YOLO v1 mAP@50 | 91.98% |
| YOLO v2 mAP@50 | 99.39% |
| Inference time (RTX 3050) | ~12ms/image |

## Main limitation
**CLOSED classifier**: always returns a name, even for an unknown individual.
Unusable in the field. → Solved in V2 and V3.

## Models
Download with `python models/download_models.py --version v1`
- `yolo_v1_nano_mAP92.pt`
- `yolo_v2_medium_mAP99.pt`
- `resnet50_classifier_10classes_acc96.pt`
