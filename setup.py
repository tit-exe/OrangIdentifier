"""
setup.py — Vérification et configuration de l'environnement
Lancer en premier sur toute nouvelle machine.

Usage:
    python setup.py
"""

import sys
import subprocess
import platform
from pathlib import Path

def check(condition, msg_ok, msg_fail):
    if condition:
        print(f"  [OK] {msg_ok}")
    else:
        print(f"  [KO] {msg_fail}")
    return condition

def main():
    print("=" * 60)
    print("  OrangIdentifier — Vérification environnement")
    print("=" * 60)

    ok = True

    # Python
    major, minor = sys.version_info.major, sys.version_info.minor
    ok &= check(major == 3 and minor >= 9,
                f"Python {major}.{minor}",
                f"Python {major}.{minor} — besoin de 3.9+")

    # PyTorch + CUDA
    try:
        import torch
        cuda = torch.cuda.is_available()
        check(True, f"PyTorch {torch.__version__}", "")
        if cuda:
            props = torch.cuda.get_device_properties(0)
            check(True,
                  f"CUDA disponible — {props.name} "
                  f"({props.total_memory/1e9:.1f} GB VRAM)", "")
        else:
            print("  [--] Pas de CUDA — entraînement possible mais lent")
    except ImportError:
        check(False, "", "PyTorch non installé")
        ok = False

    # timm
    try:
        import timm
        check(True, f"timm {timm.__version__}", "")
    except ImportError:
        check(False, "", "timm non installé — pip install timm")
        ok = False

    # ultralytics
    try:
        import ultralytics
        check(True, f"ultralytics {ultralytics.__version__}", "")
    except ImportError:
        check(False, "", "ultralytics non installé — pip install ultralytics")
        ok = False

    # PyQt5
    try:
        import PyQt5
        check(True, "PyQt5 disponible (reviewer)", "")
    except ImportError:
        print("  [--] PyQt5 absent — reviewer non disponible "
              "(pip install PyQt5)")

    # config.yaml
    cfg = Path("config.yaml")
    check(cfg.exists(), "config.yaml trouvé", "config.yaml manquant")

    # Vérification cache
    print()
    print("  Chemins cache suggérés :")
    if platform.system() == "Windows":
        print("    HF_HOME    : D:\\HuggingFaceCache")
        print("    TORCH_HOME : D:\\TorchCache")
        print("  Pour les forcer : set HF_HOME=D:\\HuggingFaceCache")
    else:
        print("    HF_HOME    : ~/hf_cache")
        print("    TORCH_HOME : ~/torch_cache")

    print()
    if ok:
        print("  Environnement OK — vous pouvez lancer la pipeline.")
        print("  Ordre recommandé :")
        print("    1. python models/download_models.py")
        print("    2. python v1_yolo_nano_resnet50/02_extract_faces.py")
        print("    3. python common/review_crops.py <dossier_crops>")
        print("    4. python v3_megadesc_arcface_10ind/04_train_arcface.py")
        print("    5. python v4_megadesc_arcface_40ind/01_train_improved.py")
    else:
        print("  Des problèmes ont été détectés.")
        print("  Installez les dépendances : pip install -r requirements.txt")

if __name__ == "__main__":
    main()
