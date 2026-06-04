# models/

The `.pt` files are not included in this repository (too large for Git).

## Download

```bash
python models/download_models.py --version all
```

## Or manually on HuggingFace

https://huggingface.co/tit0000/OrangIdentifier

| File | Version | Size | Description |
|------|---------|------|-------------|
| `yolo_v1_nano_mAP92.pt` | V1 | 6 MB | YOLO nano — mAP@50 = 91.98% |
| `yolo_v2_medium_mAP99.pt` | V1–V4 | 85 MB | YOLO medium — mAP@50 = 99.39% |
| `resnet50_classifier_10classes_acc96.pt` | V1 | 90 MB | Closed-set classifier, acc = 96.3% |
| `resnet50_backbone_2048dim.pt` | V2 | 90 MB | Embedding backbone, 2048-dim |
| `megadesc_T_arcface_final_epoch21_acc99.pt` | V3 | 105 MB | ArcFace, 10 individuals |
| `megadesc_T_arcface_v4_40individuals_acc99.pt` | **V4 ★** | 105 MB | ArcFace, 40 individuals |