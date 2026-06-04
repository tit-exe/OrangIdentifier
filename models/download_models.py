"""
download_models.py — Download the models from HuggingFace Hub
The .pt files are not in the Git repo (too large).

Usage:
    python models/download_models.py              # all models
    python models/download_models.py --version v3 # V3 only
"""

import os
import sys
import argparse
from pathlib import Path

# Bootstrap config_loader so HF/Torch cache is read from config.yaml
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.config_loader import apply_cache_env, MODELS_DIR, HF_CACHE
apply_cache_env()  # sets HF_HOME / TORCH_HOME before huggingface_hub import

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("[ERR] huggingface_hub not installed")
    print("      pip install huggingface_hub")
    sys.exit(1)

# ==============================================================================
# MODEL CATALOGUE
# ==============================================================================
# TODO: replace "titouane-iphc" with your HuggingFace username
# after uploading the models to https://huggingface.co

REPO_ID = "tit0000/OrangIdentifier"

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
        "desc":    "ResNet50 closed-set classifier V1 — acc=96.3%",
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
        "desc":    "MegaDescriptor+ArcFace V3 — acc=99%, BOS rejection=96.3%",
        "size_mb": 105,
    },
    "v4": {
        "file":    "megadesc_T_arcface_v4_40individuals_acc99.pt",
        "dest":    "models/megadesc_T_arcface_v4_40individuals_acc99.pt",
        "desc":    "MegaDescriptor+ArcFace V4 — 40 individuals, acc=99%",
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
        print(f"  [OK] {m['dest']} already exists")
        return
    print(f"  Downloading {m['desc']} (~{m['size_mb']} MB)...")
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
            print(f"  Download manually from:")
            print(f"  https://huggingface.co/{REPO_ID}/resolve/main/{m['file']}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="all",
                        choices=list(VERSION_MAP.keys()),
                        help="Which version to download (default: all)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    keys = VERSION_MAP[args.version]
    total_mb = sum(MODELS[k]["size_mb"] for k in keys)
    print(f"  Models to download: {len(keys)} ({total_mb} MB estimated)")
    print(f"  Repo : {REPO_ID}")
    print()

    for k in keys:
        download(k, dry=args.dry_run)

    print()
    print("  Direct links if the download fails:")
    print(f"  https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    main()
