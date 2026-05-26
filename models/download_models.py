"""
download_models.py — Télécharge les modèles depuis HuggingFace Hub
Les fichiers .pt ne sont pas dans le repo Git (trop lourds).

Usage:
    python models/download_models.py              # tous les modèles
    python models/download_models.py --version v3 # V3 seulement
"""

import os
import sys
import argparse
from pathlib import Path

# Forcer le cache sur un disque avec de la place
# Modifier ces chemins si nécessaire
HF_CACHE = os.environ.get("HF_HOME", "D:/HuggingFaceCache")
os.environ["HF_HOME"]    = HF_CACHE
os.environ["TORCH_HOME"] = os.environ.get("TORCH_HOME", "D:/TorchCache")

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("[ERR] huggingface_hub non installé")
    print("      pip install huggingface_hub")
    sys.exit(1)

# ==============================================================================
# CATALOGUE DES MODÈLES
# ==============================================================================
# TODO: remplacer "titouane-iphc" par votre nom d'utilisateur HuggingFace
# après avoir uploadé les modèles sur https://huggingface.co

REPO_ID = "titouane-iphc/orangutan-identifier"

MODELS = {
    "yolo_v1": {
        "file":    "yolo_v1_nano_mAP92.pt",
        "dest":    "models/yolo_v1_nano_mAP92.pt",
        "desc":    "YOLO nano V1 — mAP@50=91.98%",
        "size_mb": 6,
    },
    "yolo_v2": {
        "file":    "yolo_v2_medium_mAP99.pt",
        "dest":    "models/yolo_v2_medium_mAP99.pt",
        "desc":    "YOLO medium V2 — mAP@50=99.39% (production)",
        "size_mb": 85,
    },
    "resnet50": {
        "file":    "resnet50_classifier_10classes_acc96.pt",
        "dest":    "models/resnet50_classifier_10classes_acc96.pt",
        "desc":    "ResNet50 classifieur fermé V1 — acc=96.3%",
        "size_mb": 90,
    },
    "resnet50_backbone": {
        "file":    "resnet50_backbone_2048dim.pt",
        "dest":    "models/resnet50_backbone_2048dim.pt",
        "desc":    "ResNet50 backbone embeddings V2 — 2048-dim",
        "size_mb": 90,
    },
    "v3": {
        "file":    "megadesc_T_arcface_final_epoch21_acc99.pt",
        "dest":    "models/megadesc_T_arcface_final_epoch21_acc99.pt",
        "desc":    "MegaDescriptor+ArcFace V3 — acc=99%, rejet BOS=96.3%",
        "size_mb": 105,
    },
    "v4": {
        "file":    "megadesc_T_arcface_v4_40individus_acc99.pt",
        "dest":    "models/megadesc_T_arcface_v4_40individus_acc99.pt",
        "desc":    "MegaDescriptor+ArcFace V4 — 40 individus, acc=99%",
        "size_mb": 105,
    },
}

VERSION_MAP = {
    "v1": ["yolo_v1", "yolo_v2", "resnet50"],
    "v2": ["yolo_v2", "resnet50_backbone"],
    "v3": ["yolo_v2", "v3"],
    "v4": ["yolo_v2", "v4"],
    "all": list(MODELS.keys()),
}

def download(key: str, dry: bool = False):
    m    = MODELS[key]
    dest = Path(m["dest"])
    if dest.exists():
        print(f"  [OK] {m['dest']} existe déjà")
        return
    print(f"  Téléchargement {m['desc']} (~{m['size_mb']} MB)...")
    if not dry:
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            path = hf_hub_download(
                repo_id   = REPO_ID,
                filename  = m["file"],
                local_dir = str(dest.parent),
            )
            print(f"  [OK] {path}")
        except Exception as e:
            print(f"  [ERR] {e}")
            print(f"  Téléchargez manuellement depuis :")
            print(f"  https://huggingface.co/{REPO_ID}/resolve/main/{m['file']}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="all",
                        choices=list(VERSION_MAP.keys()),
                        help="Quelle version télécharger (défaut: all)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    keys = VERSION_MAP[args.version]
    total_mb = sum(MODELS[k]["size_mb"] for k in keys)
    print(f"  Modèles à télécharger : {len(keys)} ({total_mb} MB estimés)")
    print(f"  Repo : {REPO_ID}")
    print()

    for k in keys:
        download(k, dry=args.dry_run)

    print()
    print("  Liens directs si le téléchargement échoue :")
    print(f"  https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    main()
