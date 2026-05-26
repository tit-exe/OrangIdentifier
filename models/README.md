# models/

Les fichiers `.pt` ne sont pas dans ce repo (trop lourds).

## Téléchargement

```bash
python models/download_models.py --version all
```

## Ou manuellement sur HuggingFace

https://huggingface.co/titouane-iphc/orangutan-identifier

| Modèle | Version | Taille | Description |
|--------|---------|--------|-------------|
| `yolo_v1_nano_mAP92.pt` | V1 | 6 MB | YOLO nano, mAP@50=91.98% |
| `yolo_v2_medium_mAP99.pt` | V1/V2/V3/V4 | 85 MB | YOLO medium, mAP@50=99.39% |
| `resnet50_classifier_10classes_acc96.pt` | V1 | 90 MB | Classifieur fermé, acc=96.3% |
| `resnet50_backbone_2048dim.pt` | V2 | 90 MB | Backbone embeddings 2048-dim |
| `megadesc_T_arcface_final_epoch21_acc99.pt` | V3 | 105 MB | ArcFace 10 individus |
| `megadesc_T_arcface_v4_40individus_acc99.pt` | V4 ★ | 105 MB | ArcFace 40 individus |
