"""
V2_0_setup_and_baseline.py
==========================
CNRS IPHC Strasbourg — Orang-outan V2 pipeline
Author: Titouane

PURPOSE
-------
1. Creates the V2 folder structure on D:
2. Copies the existing models from V1
3. Forces HuggingFace + Torch cache to D: (not C:)
4. Downloads MegaDescriptor-T-224
5. Runs a baseline test on the existing 2127 crops
6. Compares directly with the ResNet50 results

RUN
---
    conda activate orangs
    python D:\OrangIdentifier\V2\scripts\V2_0_setup_and_baseline.py

The script is safe to re-run — it skips steps already done.
"""

import os
import sys
import shutil
import json
import time
from pathlib import Path
from datetime import datetime

# ==============================================================================
# STEP 0 — Force ALL downloads to D: before importing anything heavy
# This must happen before importing timm, torch, huggingface_hub, etc.
# ==============================================================================

os.environ["HF_HOME"]       = r"D:\HuggingFaceCache"
os.environ["TORCH_HOME"]    = r"D:\TorchCache"
os.environ["HF_DATASETS_CACHE"] = r"D:\HuggingFaceCache\datasets"
os.environ["TRANSFORMERS_CACHE"] = r"D:\HuggingFaceCache\transformers"

Path(r"D:\HuggingFaceCache").mkdir(parents=True, exist_ok=True)
Path(r"D:\TorchCache").mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  ORANG-OUTAN V2 — Setup + MegaDescriptor-T-224 Baseline")
print("  CNRS IPHC Strasbourg")
print("=" * 70)
print(f"  HuggingFace cache : D:\\HuggingFaceCache")
print(f"  Torch cache       : D:\\TorchCache")

# Now safe to import the rest
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# ==============================================================================
# PATHS
# ==============================================================================

# V1 paths (source)
V1_BASE    = Path(r"D:\OrangIdentifier")
V1_MODELS  = V1_BASE / "MODELS"
V1_DATASET = V1_BASE / "DATASET_CLASSIFICATION" / "raw"

# V2 paths (destination)
V2_BASE    = Path(r"D:\OrangIdentifier\V2")
V2_MODELS  = V2_BASE / "MODELS"
V2_SCRIPTS = V2_BASE / "scripts"
V2_RESULTS = V2_BASE / "RESULTS" / "baseline"
V2_ANDROID = V2_BASE / "ANDROID_EXPORT"
V2_WILD    = V2_BASE / "WILD_ORANGS"

INDIVIDUALS = sorted([
    "Auti", "Jula", "Mathai", "Molly", "NOAH",
    "PULCO", "PUTRI", "Sari", "Sinta", "Ujian"
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# STEP 1 — Create V2 folder structure
# ==============================================================================

print("\n" + "─" * 70)
print("  STEP 1 — Creating V2 folder structure")
print("─" * 70)

folders = [
    V2_MODELS, V2_SCRIPTS, V2_RESULTS, V2_ANDROID,
    V2_WILD / "raw", V2_WILD / "crops",
    V2_BASE / "RESULTS" / "arcface_training",
    V2_BASE / "RESULTS" / "gallery",
    V2_BASE / "EMBEDDINGS",
]
for folder in folders:
    folder.mkdir(parents=True, exist_ok=True)
    print(f"  OK  {folder}")

# ==============================================================================
# STEP 2 — Copy V1 models to V2 (never overwrite if already there)
# ==============================================================================

print("\n" + "─" * 70)
print("  STEP 2 — Copying V1 models to V2")
print("─" * 70)

copies = [
    (V1_MODELS / "yolo_orangs_v2" / "best.pt", V2_MODELS / "yolo_v2.pt"),
    (V1_MODELS / "resnet_orangs.pt",            V2_MODELS / "resnet_orangs.pt"),
    (V1_MODELS / "backbone_orangs.pt",          V2_MODELS / "backbone_orangs.pt"),
    (V1_MODELS / "embeddings.json",             V2_MODELS / "resnet_embeddings.json"),
]

for src, dst in copies:
    if dst.exists():
        print(f"  SKIP (already exists) {dst.name}")
        continue
    if not src.exists():
        print(f"  WARN (source not found): {src}")
        continue
    shutil.copy2(src, dst)
    size_mb = dst.stat().st_size / 1e6
    print(f"  COPY {src.name} → {dst.name} ({size_mb:.1f} MB)")

# ==============================================================================
# STEP 3 — Download MegaDescriptor-T-224
# ==============================================================================

print("\n" + "─" * 70)
print("  STEP 3 — Downloading MegaDescriptor-T-224")
print("─" * 70)
print("  Model: BVRA/MegaDescriptor-T-224 (Swin-Tiny, 204 MB)")
print("  Source: HuggingFace — saving to D:\\HuggingFaceCache")

try:
    import timm
    print("  Loading model (first run = download, subsequent runs = cache)...")
    t0 = time.time()
    mega_T = timm.create_model(
        "hf-hub:BVRA/MegaDescriptor-T-224",
        pretrained=True,
        num_classes=0       # no classification head — outputs embeddings directly
    )
    mega_T = mega_T.eval().to(DEVICE)
    elapsed = time.time() - t0

    # Quick shape check
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
        emb   = mega_T(dummy)

    emb_dim = emb.shape[1]
    print(f"  Model loaded in {elapsed:.1f}s")
    print(f"  Output embedding dimension: {emb_dim}")
    print(f"  Parameters: {sum(p.numel() for p in mega_T.parameters()) / 1e6:.1f} M")

except ImportError:
    print("  timm not installed. Run:")
    print("    pip install timm")
    sys.exit(1)
except Exception as e:
    print(f"  ERROR: {e}")
    sys.exit(1)

# Save local copy of model weights to V2/MODELS for reproducibility
mega_T_local_path = V2_MODELS / "megadesc_T_224_baseline.pt"
if not mega_T_local_path.exists():
    torch.save(mega_T.state_dict(), mega_T_local_path)
    size_mb = mega_T_local_path.stat().st_size / 1e6
    print(f"  Local copy saved: {mega_T_local_path} ({size_mb:.1f} MB)")
else:
    print(f"  Local copy already exists: {mega_T_local_path.name}")

# ==============================================================================
# STEP 4 — Load existing 2127 crops and extract embeddings
# ==============================================================================

print("\n" + "─" * 70)
print("  STEP 4 — Extracting MegaDescriptor-T-224 embeddings")
print("─" * 70)

# MegaDescriptor uses [0.5, 0.5, 0.5] normalization (not ImageNet)
mega_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

class FaceDataset(Dataset):
    def __init__(self, root: Path, individuals: list, transform):
        self.transform = transform
        self.samples   = []
        for idx, name in enumerate(individuals):
            indiv_dir = root / name
            if not indiv_dir.exists():
                print(f"  WARN: directory not found: {indiv_dir}")
                continue
            images = sorted([
                f for f in indiv_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
            ])
            for img_path in images:
                self.samples.append((img_path, idx, name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, idx, name = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), idx, name

dataset = FaceDataset(V1_DATASET, INDIVIDUALS, mega_transform)
print(f"  Total crops: {len(dataset)}")
for idx, name in enumerate(INDIVIDUALS):
    n = sum(1 for s in dataset.samples if s[2] == name)
    print(f"    [{idx:2d}] {name:<10}: {n:4d} crops")

loader = DataLoader(dataset, batch_size=64, shuffle=False,
                    num_workers=0, pin_memory=True)

all_embeddings = []
all_labels     = []

print(f"\n  Extracting embeddings on {DEVICE}...")
t0 = time.time()

mega_T.eval()
with torch.no_grad():
    for imgs, labels, _ in loader:
        imgs = imgs.to(DEVICE)
        embs = mega_T(imgs)                          # [B, emb_dim]
        # L2 normalize
        embs = nn.functional.normalize(embs, dim=1)
        all_embeddings.append(embs.cpu())
        all_labels.extend(labels.tolist())

elapsed = time.time() - t0
all_embeddings = torch.cat(all_embeddings, dim=0)
all_labels     = torch.tensor(all_labels)

print(f"  Done in {elapsed:.1f}s — shape: {list(all_embeddings.shape)}")
print(f"  Mean L2 norm: {all_embeddings.norm(dim=1).mean():.6f}")

# ==============================================================================
# STEP 5 — Build prototypes and measure separability
# ==============================================================================

print("\n" + "─" * 70)
print("  STEP 5 — Computing separability metrics")
print("─" * 70)

num_classes = len(INDIVIDUALS)
prototype_matrix = torch.zeros(num_classes, emb_dim)
per_indiv_counts = {}

for idx, name in enumerate(INDIVIDUALS):
    mask  = (all_labels == idx)
    count = mask.sum().item()
    per_indiv_counts[name] = count
    if count == 0:
        continue
    proto = all_embeddings[mask].mean(dim=0)
    proto = proto / proto.norm()
    prototype_matrix[idx] = proto

# Positive similarities (same individual → own prototype)
sims_positive = []
sims_negative = []

for idx in range(num_classes):
    mask = (all_labels == idx)
    if not mask.any():
        continue
    embs_i = all_embeddings[mask]

    # Similarity to own prototype
    own_proto  = prototype_matrix[idx]
    sims_own   = (embs_i @ own_proto).tolist()
    sims_positive.extend(sims_own)

    # Best similarity to OTHER prototypes (hardest negative)
    other_protos = torch.cat([
        prototype_matrix[:idx], prototype_matrix[idx+1:]
    ], dim=0)
    sims_other = (embs_i @ other_protos.T).max(dim=1).values.tolist()
    sims_negative.extend(sims_other)

sims_positive = np.array(sims_positive)
sims_negative = np.array(sims_negative)
separability  = sims_positive.mean() / sims_negative.mean()

# Top-1 accuracy
sim_matrix = all_embeddings @ prototype_matrix.T   # [N, 10]
top1_preds  = sim_matrix.argmax(dim=1)
top1_acc    = (top1_preds == all_labels).float().mean().item()

print(f"\n  MegaDescriptor-T-224 BASELINE results:")
print(f"  {'Metric':<35} {'MegaDesc-T':>12}  {'ResNet50 V1':>12}")
print(f"  {'─'*60}")
print(f"  {'Positive similarity (mean)':<35} {sims_positive.mean():>12.4f}  {'0.7206':>12}")
print(f"  {'Negative similarity (mean)':<35} {sims_negative.mean():>12.4f}  {'0.4189':>12}")
print(f"  {'Separability ratio':<35} {separability:>12.4f}  {'1.7203':>12}")
print(f"  {'Top-1 accuracy':<35} {top1_acc*100:>11.2f}%  {'98.54%':>12}")
print(f"  {'─'*60}")

if separability > 1.7203:
    delta = separability - 1.7203
    print(f"\n  MegaDescriptor-T-224 is BETTER than ResNet50")
    print(f"  Separability gain: +{delta:.4f} ({delta/1.7203*100:.1f}%)")
else:
    delta = 1.7203 - separability
    print(f"\n  MegaDescriptor-T-224 is slightly WORSE than ResNet50 (baseline, no fine-tuning)")
    print(f"  Separability gap: -{delta:.4f} — fine-tuning with ArcFace will close this")

# Per-individual breakdown
print(f"\n  Per-individual accuracy (MegaDescriptor-T-224 baseline):")
for idx, name in enumerate(INDIVIDUALS):
    mask  = (all_labels == idx)
    if not mask.any():
        continue
    preds_i  = top1_preds[mask]
    correct  = (preds_i == idx).sum().item()
    total    = mask.sum().item()
    mean_sim = (all_embeddings[mask] @ prototype_matrix[idx]).mean().item()
    print(f"    {name:<10}: {correct:4d}/{total:4d} = {correct/total*100:6.2f}%  "
          f"mean_sim={mean_sim:.4f}")

# ==============================================================================
# STEP 6 — Save results and comparison report
# ==============================================================================

print("\n" + "─" * 70)
print("  STEP 6 — Saving results")
print("─" * 70)

results = {
    "date": datetime.now().isoformat(),
    "model": "MegaDescriptor-T-224 (BVRA/MegaDescriptor-T-224)",
    "status": "baseline — no fine-tuning",
    "embedding_dim": int(emb_dim),
    "num_crops": len(dataset),
    "metrics": {
        "top1_accuracy": round(top1_acc, 6),
        "positive_similarity_mean": round(float(sims_positive.mean()), 4),
        "positive_similarity_std": round(float(sims_positive.std()), 4),
        "negative_similarity_mean": round(float(sims_negative.mean()), 4),
        "negative_similarity_std": round(float(sims_negative.std()), 4),
        "separability_ratio": round(float(separability), 4),
    },
    "comparison_resnet50_v1": {
        "top1_accuracy": 0.9854,
        "positive_similarity_mean": 0.7206,
        "negative_similarity_mean": 0.4189,
        "separability_ratio": 1.7203,
    },
    "verdict": "better" if separability > 1.7203 else "worse_before_finetuning"
}

results_path = V2_RESULTS / "megadesc_T_baseline.json"
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"  Results saved: {results_path}")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

print(f"""
{'=' * 70}
  DONE
{'=' * 70}

  V2 structure created : {V2_BASE}
  Models copied        : {V2_MODELS}
  MegaDescriptor-T-224 : downloaded + tested
  Results              : {V2_RESULTS}

  KEY NUMBERS:
    Separability MegaDesc-T (baseline) : {separability:.4f}
    Separability ResNet50 V1           : 1.7203
    {'→ Already better — fine-tuning will push it further' if separability > 1.7203
     else '→ Fine-tuning with ArcFace on your data will make it better'}

  NEXT STEP:
    V2_1_download_wild_orangs.py   ← download 5000+ images from iNaturalist
{'=' * 70}
""")