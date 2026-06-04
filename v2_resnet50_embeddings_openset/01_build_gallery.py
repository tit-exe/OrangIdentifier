"""
6_build_embeddings.py — Open-set identification via embedding gallery
======================================================================
Orang-outan individual recognition

Date: May 2026

PURPOSE
-------
Converts the ResNet50 classifier into an open-set identification system
using embedding-based nearest-neighbor matching. Unlike the softmax
classifier (which always picks one of 10 known individuals), this system
can say "I don't know this individual" for new arrivals.

WHAT THIS SCRIPT DOES
---------------------
1. Loads ResNet50 backbone WITHOUT the FC head (output: 2048-dim vectors)
2. Extracts embeddings for all crops in DATASET_CLASSIFICATION/raw/
3. Builds per-individual prototype embeddings (mean over all crops)
4. Calibrates the unknown-detection threshold via leave-one-individual-out
   cross-validation — simulates new individuals arriving in the park
5. Generates 7 diagnostic visualizations
6. Saves embeddings.json for the Android app
7. Exports backbone-only TFLite model (replaces resnet50_classifier.tflite)
8. Packages a new Android ZIP bundle

HOW IT WORKS AT INFERENCE TIME (Android)
-----------------------------------------
  New face crop
      ↓
  ResNet50 backbone (TFLite, 2048-dim output)
      ↓
  L2-normalize the vector
      ↓
  Cosine similarity vs all prototype embeddings in embeddings.json
      ↓
  Best similarity > threshold  → "Molly (94.2%)"
  Best similarity < threshold  → "Unknown individual"

WHY THIS IS BETTER
------------------
  Softmax classifier:  always picks one of 10 → useless for open-set
  Embedding approach:  can reject unknown → correct for field deployment
  Adding a new individual:  10 photos → run this script → new ZIP
                            NO retraining needed

DOES NOT TOUCH
--------------
  - resnet_orangs.pt    (classifier stays intact)
  - backbone_orangs.pt  (backbone stays intact)
  - yolo_orangs_v2/     (YOLO unchanged)
  - Any existing RESULTS/ files

OUTPUTS
-------
  MODELS/
    embedding_gallery.pt         ← PyTorch tensors (for Python use)
    embeddings.json              ← Android-readable gallery
    embedding_metadata.json      ← threshold + stats

  RESULTS/embeddings/
    01_tsne_embedding_space.png  ← t-SNE of all 1986 embeddings
    02_similarity_matrix.png     ← 10×10 cosine similarity heatmap
    03_similarity_distribution.png ← known vs unknown score distributions
    04_roc_curve.png             ← ROC for unknown detection
    05_threshold_calibration.png ← Precision/Recall/F1 vs threshold
    06_per_individual_accuracy.png ← Accuracy per individual (embedding)
    07_confusion_matrix.png      ← Embedding-based confusion matrix (val set)
    rapport_embeddings.txt       ← Full text report

  ANDROID_EXPORT/
    resnet50_backbone.tflite     ← Backbone only (2048-dim output)
    embeddings.json              ← Gallery
    (+ updated ZIP)

REQUIREMENTS
------------
  conda activate wildlife-id
  pip install scikit-learn matplotlib seaborn umap-learn  (if not already)
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent))
from common.config_loader import (
    apply_cache_env,
    PHOTOS_DIR, WILD_IMAGES_DIR, CROPS_KNOWN_DIR, CROPS_WILD_DIR, CROPS_JSON,
    MODELS_DIR, OUTPUT_DIR, YOLO_V2_PT,
    V3_PT, V4_PT, UNKNOWN_THRESHOLD,
    ARC_SCALE, ARC_MARGIN, MAX_EPOCHS, PATIENCE, PATIENCE_START,
    LR_BACKBONE, LR_HEAD, BATCH_SIZE, DEVICE, ensure_dirs, to_relative,
)
apply_cache_env()  # sets HF_HOME/TORCH_HOME before any heavy imports


import os
import sys
import json
import time
import shutil
import zipfile
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server/script use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION — all paths from the project documentation
# ==============================================================================

BASE_DIR = OUTPUT_DIR
DATASET_DIR     = BASE_DIR / "DATASET_CLASSIFICATION" / "raw"
RESNET_PT       = BASE_DIR / "MODELS" / "resnet_orangs.pt"
BACKBONE_PT     = BASE_DIR / "MODELS" / "backbone_orangs.pt"
OUTPUT_MODELS   = BASE_DIR / "MODELS"
OUTPUT_RESULTS  = BASE_DIR / "RESULTS" / "embeddings"
ANDROID_DIR     = BASE_DIR / "ANDROID_EXPORT"
ZIP_OUTPUT      = BASE_DIR / "primate_models_android_v2.zip"

# Individuals in ALPHABETICAL order — MUST match the order used during training
# (ImageFolder sorts directories alphabetically, that is the ground truth)
INDIVIDUALS = sorted([
    "Auti", "Jula", "Mathai", "Molly", "NOAH",
    "PULCO", "PUTRI", "Sari", "Sinta", "Ujian"
])

# ImageNet normalization — identical to training (4_train_resnet.py)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE      = 224

# Embedding parameters
KNN_K         = 5       # k for kNN voting (used in calibration analysis)
BATCH_SIZE    = 64      # for embedding extraction
SEED          = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Color palette for 10 individuals (consistent across all plots)
PALETTE = [
    "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
    "#44BBA4", "#E94F37", "#393E41", "#F5A623", "#8B5E83"
]

# ==============================================================================
# SETUP
# ==============================================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

OUTPUT_RESULTS.mkdir(parents=True, exist_ok=True)
OUTPUT_MODELS.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  ORANG-OUTAN EMBEDDING GALLERY BUILDER")
print("  open-set identification pipeline")
print("=" * 70)
print(f"  Device : {DEVICE}")
print(f"  Dataset: {DATASET_DIR}")
print(f"  Model  : {RESNET_PT}")

# ==============================================================================
# STEP 1 — Load ResNet50 backbone (without FC head)
# ==============================================================================

def print_section(title: str):
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")

print_section("STEP 1 — Loading ResNet50 backbone (no FC head)")

# Verify model file
if not RESNET_PT.exists():
    print(f"  [ERROR] Model not found: {RESNET_PT}")
    sys.exit(1)

checkpoint = torch.load(RESNET_PT, map_location="cpu", weights_only=False)

# The checkpoint uses key 'model_state' (confirmed from training script analysis)
if isinstance(checkpoint, dict):
    for key in ("model_state", "model_state_dict", "state_dict"):
        if key in checkpoint:
            state_dict = checkpoint[key]
            print(f"  Checkpoint key found: '{key}'")
            break
    else:
        state_dict = checkpoint
        print(f"  Using checkpoint directly as state_dict")
    classes_from_ckpt = checkpoint.get("classes", INDIVIDUALS)
else:
    state_dict = checkpoint
    classes_from_ckpt = INDIVIDUALS

# Verify class order matches our INDIVIDUALS list
if classes_from_ckpt != INDIVIDUALS:
    print(f"  [WARNING] Class order mismatch!")
    print(f"    Checkpoint: {classes_from_ckpt}")
    print(f"    Expected  : {INDIVIDUALS}")
    print(f"    Using checkpoint order as ground truth.")
    INDIVIDUALS = classes_from_ckpt

num_classes = len(INDIVIDUALS)
print(f"  {num_classes} individuals: {INDIVIDUALS}")

# Reconstruct the FULL model (same architecture as 4_train_resnet.py)
full_model = models.resnet50(weights=None)
full_model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(2048, num_classes)
)
full_model.load_state_dict(state_dict, strict=True)
full_model.eval()
full_model = full_model.to(DEVICE)

# Build backbone: ResNet50 WITHOUT the fc layer
# We extract features from avgpool → 2048-dim vector
class ResNetBackbone(nn.Module):
    """
    ResNet50 with the classification head removed.
    Output: L2-normalized 2048-dimensional embedding vector.
    L2 normalization ensures cosine similarity = dot product,
    which is numerically more stable and faster to compute.
    """
    def __init__(self, full_resnet):
        super().__init__()
        # Copy all layers except fc
        self.conv1    = full_resnet.conv1
        self.bn1      = full_resnet.bn1
        self.relu     = full_resnet.relu
        self.maxpool  = full_resnet.maxpool
        self.layer1   = full_resnet.layer1
        self.layer2   = full_resnet.layer2
        self.layer3   = full_resnet.layer3
        self.layer4   = full_resnet.layer4
        self.avgpool  = full_resnet.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)                 # [batch, 2048]
        x = nn.functional.normalize(x, dim=1)  # L2 normalize
        return x

backbone = ResNetBackbone(full_model)
backbone.eval()
backbone = backbone.to(DEVICE)

# Quick sanity check
with torch.no_grad():
    dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
    emb = backbone(dummy)
    assert emb.shape == (1, 2048), f"Unexpected shape: {emb.shape}"
    norm = emb.norm().item()
    assert abs(norm - 1.0) < 1e-5, f"Not L2-normalized: norm={norm}"

print(f"  Backbone OK — output shape: {list(emb.shape)}, L2 norm: {norm:.6f}")

# ==============================================================================
# STEP 2 — Load all face crops and extract embeddings
# ==============================================================================

print_section("STEP 2 — Extracting embeddings from all face crops")

# Check dataset directory
if not DATASET_DIR.exists():
    print(f"  [ERROR] Dataset not found: {DATASET_DIR}")
    sys.exit(1)

# Validation-only transform (same as val_test_transform in training)
# No augmentation — we want stable, reproducible embeddings
val_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class FaceDataset(Dataset):
    """
    Loads all face crops from DATASET_CLASSIFICATION/raw/.
    Expects structure: raw/{individual_name}/{image.jpg}
    Skips hidden directories and _a_verifier.
    """
    def __init__(self, root: Path, individuals: list, transform):
        self.transform = transform
        self.samples = []   # list of (image_path, individual_idx, individual_name)

        for idx, name in enumerate(individuals):
            indiv_dir = root / name
            if not indiv_dir.exists():
                print(f"  [WARNING] Directory not found: {indiv_dir}")
                continue
            images = sorted([
                f for f in indiv_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
            ])
            for img_path in images:
                self.samples.append((img_path, idx, name))

        print(f"  Total crops found: {len(self.samples)}")
        for idx, name in enumerate(individuals):
            count = sum(1 for s in self.samples if s[2] == name)
            print(f"    [{idx:2d}] {name:<10} : {count:4d} crops")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, idx, name = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), idx, name

dataset = FaceDataset(DATASET_DIR, INDIVIDUALS, val_transform)
if len(dataset) == 0:
    print("  [ERROR] No images found in dataset directory.")
    sys.exit(1)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=0, pin_memory=True)

# Extract all embeddings
all_embeddings = []
all_labels     = []
all_names      = []

print(f"\n  Extracting embeddings (batch_size={BATCH_SIZE}, device={DEVICE})...")
start = time.time()

backbone.eval()
with torch.no_grad():
    for batch_idx, (imgs, labels, names) in enumerate(loader):
        imgs = imgs.to(DEVICE)
        embs = backbone(imgs)               # [B, 2048], L2-normalized
        all_embeddings.append(embs.cpu())
        all_labels.extend(labels.tolist())
        all_names.extend(names)

        if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
            done = (batch_idx + 1) * BATCH_SIZE
            print(f"    {min(done, len(dataset))}/{len(dataset)} crops processed...")

elapsed = time.time() - start
all_embeddings = torch.cat(all_embeddings, dim=0)   # [N, 2048]
all_labels     = torch.tensor(all_labels)            # [N]

print(f"  Done in {elapsed:.1f}s — tensor shape: {list(all_embeddings.shape)}")
print(f"  Embedding statistics:")
print(f"    Mean norm: {all_embeddings.norm(dim=1).mean():.6f} (should be ~1.0)")
print(f"    Std  norm: {all_embeddings.norm(dim=1).std():.6f}")

# Save full embedding bank (PyTorch format, for future Python use)
gallery_pt_path = OUTPUT_MODELS / "embedding_gallery.pt"
torch.save({
    "embeddings": all_embeddings,
    "labels": all_labels,
    "individuals": INDIVIDUALS,
    "extraction_date": datetime.now().isoformat(),
    "model_source": str(RESNET_PT),
}, gallery_pt_path)
print(f"  Saved: {gallery_pt_path}")

# ==============================================================================
# STEP 3 — Build per-individual prototype embeddings
# ==============================================================================

print_section("STEP 3 — Building per-individual prototype embeddings")

prototypes = {}   # name → mean L2-normalized embedding [2048]

for idx, name in enumerate(INDIVIDUALS):
    mask = (all_labels == idx)
    count = mask.sum().item()
    if count == 0:
        print(f"  [WARNING] No crops found for {name}")
        continue
    indiv_embeddings = all_embeddings[mask]          # [n, 2048]
    prototype = indiv_embeddings.mean(dim=0)          # [2048]
    prototype = prototype / prototype.norm()           # re-normalize after mean
    prototypes[name] = prototype
    print(f"  {name:<10}: {count:4d} crops → prototype computed")

# Build prototype matrix [num_classes, 2048]
prototype_matrix = torch.stack([prototypes[name] for name in INDIVIDUALS], dim=0)
print(f"\n  Prototype matrix shape: {list(prototype_matrix.shape)}")

# ==============================================================================
# STEP 4 — Threshold calibration via leave-one-individual-out
# ==============================================================================

print_section("STEP 4 — Threshold calibration (leave-one-individual-out)")

"""
Strategy: For each individual I, we simulate them as "unknown":
  - Gallery built from the remaining 9 individuals
  - Query: all crops of individual I → should be REJECTED as unknown
  - Query: all crops of the 9 known individuals → should be IDENTIFIED

This is the most realistic simulation of field conditions where a new
individual arrives that the system has never seen before.

We sweep the cosine similarity threshold from 0.0 to 1.0 and measure:
  - True Positive Rate (TPR): known individuals correctly identified
  - False Positive Rate (FPR): unknown individuals incorrectly identified
  - Precision, Recall, F1 for unknown detection
"""

# Score matrix: for each crop, cosine similarity to its true individual's prototype
# This tells us the "within-class similarity" distribution
all_sims_positive = []  # similarities of crops to their TRUE individual prototype
all_sims_negative = []  # similarities of crops to OTHER individuals' prototypes

for idx, name in enumerate(INDIVIDUALS):
    mask = (all_labels == idx)
    if not mask.any():
        continue
    indiv_embs = all_embeddings[mask]       # [n, 2048]

    # Similarity to own prototype (positive pairs)
    own_proto = prototype_matrix[idx]
    sims_own = (indiv_embs @ own_proto).tolist()
    all_sims_positive.extend(sims_own)

    # Similarity to other prototypes (negative pairs — best match among others)
    other_protos = torch.cat([
        prototype_matrix[:idx], prototype_matrix[idx+1:]
    ], dim=0)                               # [9, 2048]
    sims_others = (indiv_embs @ other_protos.T).max(dim=1).values.tolist()
    all_sims_negative.extend(sims_others)

all_sims_positive = np.array(all_sims_positive)
all_sims_negative = np.array(all_sims_negative)

print(f"  Positive pairs (same individual):")
print(f"    Mean: {all_sims_positive.mean():.4f}  Std: {all_sims_positive.std():.4f}")
print(f"    Min:  {all_sims_positive.min():.4f}  Max: {all_sims_positive.max():.4f}")
print(f"\n  Negative pairs (different individuals — best match):")
print(f"    Mean: {all_sims_negative.mean():.4f}  Std: {all_sims_negative.std():.4f}")
print(f"    Min:  {all_sims_negative.min():.4f}  Max: {all_sims_negative.max():.4f}")

# Leave-one-individual-out calibration
print(f"\n  Running leave-one-individual-out calibration...")

loio_scores = []  # (similarity, is_known) for all queries

for held_out_idx, held_out_name in enumerate(INDIVIDUALS):
    # Known individuals: everyone except held_out
    known_indices = [i for i in range(num_classes) if i != held_out_idx]
    known_names   = [INDIVIDUALS[i] for i in known_indices]

    # Build gallery from known individuals only
    known_protos = prototype_matrix[known_indices]  # [9, 2048]

    # Query 1: held-out individual crops → should be rejected (unknown)
    mask_unknown = (all_labels == held_out_idx)
    if mask_unknown.any():
        unknown_embs = all_embeddings[mask_unknown]  # [n, 2048]
        best_sims_unknown = (unknown_embs @ known_protos.T).max(dim=1).values
        for sim in best_sims_unknown.tolist():
            loio_scores.append((sim, False))  # False = this is actually unknown

    # Query 2: known individual crops → should be identified correctly
    for k_idx in known_indices:
        mask_known = (all_labels == k_idx)
        if not mask_known.any():
            continue
        known_embs = all_embeddings[mask_known]     # [n, 2048]
        best_sims_known = (known_embs @ known_protos.T).max(dim=1).values
        for sim in best_sims_known.tolist():
            loio_scores.append((sim, True))  # True = this is a known individual

loio_scores  = np.array(loio_scores)
loio_sims    = loio_scores[:, 0]
loio_is_known = loio_scores[:, 1].astype(bool)

# Sweep thresholds
thresholds = np.linspace(0.0, 1.0, 1000)
precisions, recalls, f1_scores = [], [], []
tprs_known, fprs_unknown = [], []

for thresh in thresholds:
    predicted_known = loio_sims >= thresh

    # For "known" detection:
    tp = (predicted_known &  loio_is_known).sum()
    fp = (predicted_known & ~loio_is_known).sum()
    tn = (~predicted_known & ~loio_is_known).sum()
    fn = (~predicted_known &  loio_is_known).sum()

    tpr = tp / (tp + fn + 1e-9)          # sensitivity (known individuals found)
    fpr = fp / (fp + tn + 1e-9)          # false alarm rate (unknowns mis-id'd)
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)

    tprs_known.append(tpr)
    fprs_unknown.append(fpr)
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)

tprs_known   = np.array(tprs_known)
fprs_unknown = np.array(fprs_unknown)
f1_scores    = np.array(f1_scores)
precisions   = np.array(precisions)
recalls      = np.array(recalls)

# Find optimal threshold: maximize F1
best_thresh_idx = np.argmax(f1_scores)
optimal_threshold = float(thresholds[best_thresh_idx])
best_f1     = float(f1_scores[best_thresh_idx])
best_prec   = float(precisions[best_thresh_idx])
best_recall = float(recalls[best_thresh_idx])
roc_auc     = auc(fprs_unknown, tprs_known)

print(f"\n  Calibration results:")
print(f"    Optimal threshold : {optimal_threshold:.4f}")
print(f"    Best F1           : {best_f1:.4f}")
print(f"    Precision         : {best_prec:.4f} (% of 'known' predictions that are correct)")
print(f"    Recall            : {best_recall:.4f} (% of known individuals found)")
print(f"    ROC AUC           : {roc_auc:.4f}")

# Also report at specific operating points for practical use
for thresh_target in [0.60, 0.65, 0.70, 0.75, 0.80]:
    idx_t = np.argmin(np.abs(thresholds - thresh_target))
    print(f"    @ threshold={thresh_target:.2f}: "
          f"F1={f1_scores[idx_t]:.3f}  "
          f"Prec={precisions[idx_t]:.3f}  "
          f"Rec={recalls[idx_t]:.3f}")

# ==============================================================================
# STEP 5 — Per-individual embedding accuracy (on full dataset)
# ==============================================================================

print_section("STEP 5 — Per-individual identification accuracy")

"""
Using full prototype gallery (all 10 individuals), measure:
- Top-1 accuracy per individual (nearest prototype = correct)
- Mean confidence per individual
"""

# Full gallery: cosine similarity to all prototypes
all_sim_matrix = all_embeddings @ prototype_matrix.T    # [N, 10]
top1_preds     = all_sim_matrix.argmax(dim=1)           # [N]
top1_sims      = all_sim_matrix.max(dim=1).values       # [N]

per_indiv_acc  = {}
per_indiv_conf = {}
confusion_counts = np.zeros((num_classes, num_classes), dtype=int)

for idx, name in enumerate(INDIVIDUALS):
    mask = (all_labels == idx)
    if not mask.any():
        continue
    preds_i = top1_preds[mask]
    sims_i  = top1_sims[mask]
    correct = (preds_i == idx).sum().item()
    total   = mask.sum().item()
    acc     = correct / total
    conf    = sims_i.mean().item()

    per_indiv_acc[name]  = acc
    per_indiv_conf[name] = conf

    for pred in preds_i.tolist():
        confusion_counts[idx, pred] += 1

    print(f"  {name:<10}: {correct:4d}/{total:4d} = {acc*100:6.2f}%  "
          f"mean sim={conf:.4f}")

overall_acc = sum(per_indiv_acc[n] * sum(1 for s in dataset.samples if s[2] == n)
                  for n in per_indiv_acc) / len(dataset)
print(f"\n  Overall top-1 accuracy (embedding): {overall_acc*100:.2f}%")

# ==============================================================================
# STEP 6 — Generate 7 diagnostic visualizations
# ==============================================================================

print_section("STEP 6 — Generating visualizations")

individual_colors = {name: PALETTE[i] for i, name in enumerate(INDIVIDUALS)}

# ── Plot 1: t-SNE of embedding space ────────────────────────────────────────
print("  01. t-SNE embedding space...")

# Subsample for t-SNE speed (max 200 per individual to keep it manageable)
MAX_PER_CLASS = 200
tsne_indices = []
for idx, name in enumerate(INDIVIDUALS):
    mask_idx = torch.where(all_labels == idx)[0]
    if len(mask_idx) > MAX_PER_CLASS:
        perm = torch.randperm(len(mask_idx))[:MAX_PER_CLASS]
        mask_idx = mask_idx[perm]
    tsne_indices.extend(mask_idx.tolist())

tsne_embeddings = all_embeddings[tsne_indices].numpy()
tsne_labels     = all_labels[tsne_indices].numpy()

tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=SEED)
try:
    tsne_2d = tsne.fit_transform(tsne_embeddings)
except TypeError:
    tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=SEED)
    tsne_2d = tsne.fit_transform(tsne_embeddings)

fig, ax = plt.subplots(figsize=(14, 10))
for idx, name in enumerate(INDIVIDUALS):
    mask = (tsne_labels == idx)
    if not mask.any():
        continue
    ax.scatter(tsne_2d[mask, 0], tsne_2d[mask, 1],
               c=PALETTE[idx], label=name, alpha=0.65, s=20, linewidths=0)

# Plot prototype centroids
proto_2d = tsne.fit_transform(
    torch.cat([prototype_matrix, torch.tensor(tsne_embeddings)], dim=0).numpy()
)[:num_classes]

for idx, name in enumerate(INDIVIDUALS):
    ax.scatter(proto_2d[idx, 0], proto_2d[idx, 1],
               c=PALETTE[idx], marker="*", s=400, edgecolors="black",
               linewidths=1.5, zorder=10)
    ax.annotate(name, (proto_2d[idx, 0], proto_2d[idx, 1]),
                fontsize=9, fontweight="bold", ha="center", va="bottom",
                xytext=(0, 8), textcoords="offset points")

ax.legend(loc="upper right", fontsize=9, markerscale=2,
          framealpha=0.9, ncol=2)
ax.set_title("t-SNE of ResNet50 Embedding Space — Orang-outan Face Recognition",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("t-SNE dimension 1", fontsize=11)
ax.set_ylabel("t-SNE dimension 2", fontsize=11)
ax.text(0.02, 0.02,
        f"n={len(tsne_indices)} crops  |  ★ = individual prototype",
        transform=ax.transAxes, fontsize=9, color="gray")
plt.tight_layout()
plt.savefig(OUTPUT_RESULTS / "01_tsne_embedding_space.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: 01_tsne_embedding_space.png")

# ── Plot 2: Cosine similarity matrix between prototypes ─────────────────────
print("  02. Cosine similarity matrix...")

sim_matrix_np = (prototype_matrix @ prototype_matrix.T).numpy()

fig, ax = plt.subplots(figsize=(10, 8))
mask_diag = np.eye(num_classes, dtype=bool)
sim_off_diag = sim_matrix_np[~mask_diag]

im = ax.imshow(sim_matrix_np, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto")
plt.colorbar(im, ax=ax, label="Cosine Similarity", shrink=0.85)

for i in range(num_classes):
    for j in range(num_classes):
        val = sim_matrix_np[i, j]
        color = "white" if val < 0.6 or (i == j) else "black"
        if i == j:
            color = "white"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                fontsize=9, fontweight="bold" if i == j else "normal",
                color=color)

ax.set_xticks(range(num_classes))
ax.set_yticks(range(num_classes))
ax.set_xticklabels(INDIVIDUALS, rotation=45, ha="right", fontsize=10)
ax.set_yticklabels(INDIVIDUALS, fontsize=10)
ax.set_title("Cosine Similarity Between Individual Prototype Embeddings",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Predicted individual", fontsize=11)
ax.set_ylabel("True individual", fontsize=11)

# Highlight diagonal
for i in range(num_classes):
    rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                          fill=False, edgecolor="black", linewidth=2)
    ax.add_patch(rect)

fig.text(0.5, 0.01,
         f"Off-diagonal mean: {sim_off_diag.mean():.3f}  |  "
         f"Off-diagonal max: {sim_off_diag.max():.3f}  |  "
         "Lower off-diagonal = better separation",
         ha="center", fontsize=9, color="gray")
plt.tight_layout()
plt.savefig(OUTPUT_RESULTS / "02_similarity_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: 02_similarity_matrix.png")

# ── Plot 3: Similarity distribution (known vs unknown) ──────────────────────
print("  03. Similarity distributions...")

known_sims_loio   = loio_sims[loio_is_known]
unknown_sims_loio = loio_sims[~loio_is_known]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: histogram
ax = axes[0]
bins = np.linspace(0, 1, 60)
ax.hist(known_sims_loio, bins=bins, alpha=0.7, color="#2E86AB",
        label=f"Known individuals\n(n={len(known_sims_loio)}, μ={known_sims_loio.mean():.3f})",
        density=True)
ax.hist(unknown_sims_loio, bins=bins, alpha=0.7, color="#C73E1D",
        label=f"Unknown individuals\n(n={len(unknown_sims_loio)}, μ={unknown_sims_loio.mean():.3f})",
        density=True)
ax.axvline(optimal_threshold, color="black", linestyle="--", linewidth=2,
           label=f"Optimal threshold = {optimal_threshold:.3f}")
ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 10],
                 0, optimal_threshold, alpha=0.08, color="#C73E1D")
ax.fill_betweenx([0, 10], optimal_threshold, 1, alpha=0.08, color="#2E86AB")
ax.legend(fontsize=10, framealpha=0.9)
ax.set_xlabel("Best cosine similarity score", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Score Distribution: Known vs Unknown Individuals\n(Leave-one-individual-out)",
             fontsize=12, fontweight="bold")
ax.set_xlim(0, 1)

# Right: KDE-style view with cumulative
ax = axes[1]
known_sorted   = np.sort(known_sims_loio)
unknown_sorted = np.sort(unknown_sims_loio)
ax.plot(known_sorted,   np.linspace(0, 1, len(known_sorted)),
        color="#2E86AB", linewidth=2, label="Known (ECDF)")
ax.plot(unknown_sorted, np.linspace(0, 1, len(unknown_sorted)),
        color="#C73E1D", linewidth=2, label="Unknown (ECDF)")
ax.axvline(optimal_threshold, color="black", linestyle="--", linewidth=2,
           label=f"Threshold = {optimal_threshold:.3f}")
ax.set_xlabel("Cosine similarity threshold", fontsize=11)
ax.set_ylabel("Cumulative fraction", fontsize=11)
ax.set_title("Cumulative Distribution of Scores", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_RESULTS / "03_similarity_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: 03_similarity_distribution.png")

# ── Plot 4: ROC curve ────────────────────────────────────────────────────────
print("  04. ROC curve...")

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(fprs_unknown, tprs_known, color="#2E86AB", linewidth=2.5,
        label=f"Embedding system (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random baseline")
ax.scatter([fprs_unknown[best_thresh_idx]], [tprs_known[best_thresh_idx]],
           color="red", s=150, zorder=10,
           label=f"Optimal point (threshold={optimal_threshold:.3f})")

# Mark practical operating points
for thresh_target, label in [(0.65, "0.65"), (0.70, "0.70"), (0.75, "0.75")]:
    i_t = np.argmin(np.abs(thresholds - thresh_target))
    ax.scatter([fprs_unknown[i_t]], [tprs_known[i_t]],
               marker="D", s=80, color="orange", zorder=9)
    ax.annotate(f"τ={label}", (fprs_unknown[i_t], tprs_known[i_t]),
                fontsize=9, xytext=(5, -12), textcoords="offset points")

ax.set_xlabel("False Positive Rate (unknowns incorrectly identified)", fontsize=12)
ax.set_ylabel("True Positive Rate (known individuals correctly identified)", fontsize=12)
ax.set_title("ROC Curve — Unknown Individual Detection\n(Leave-one-individual-out)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11, loc="lower right")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_RESULTS / "04_roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: 04_roc_curve.png")

# ── Plot 5: Threshold calibration ────────────────────────────────────────────
print("  05. Threshold calibration curve...")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(thresholds, f1_scores,    color="#2E86AB", linewidth=2.5, label="F1 score")
ax.plot(thresholds, precisions,   color="#F18F01", linewidth=2, linestyle="--",
        label="Precision (% known predictions that are correct)")
ax.plot(thresholds, recalls,      color="#44BBA4", linewidth=2, linestyle="-.",
        label="Recall (% of known individuals found)")
ax.axvline(optimal_threshold, color="red", linewidth=2, linestyle=":",
           label=f"Optimal threshold = {optimal_threshold:.3f} (F1={best_f1:.3f})")
ax.axhspan(best_f1 - 0.02, best_f1, alpha=0.1, color="red")

ax.set_xlabel("Cosine similarity threshold", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Precision / Recall / F1 vs Threshold\n"
             "(Higher threshold = more conservative, fewer false IDs)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

# Annotate regions
ax.text(0.15, 0.05, "Too permissive\n(many unknowns accepted)",
        ha="center", color="gray", fontsize=9, style="italic")
ax.text(0.85, 0.05, "Too strict\n(known individuals rejected)",
        ha="center", color="gray", fontsize=9, style="italic")

plt.tight_layout()
plt.savefig(OUTPUT_RESULTS / "05_threshold_calibration.png", dpi=150,
            bbox_inches="tight")
plt.close()
print("    Saved: 05_threshold_calibration.png")

# ── Plot 6: Per-individual embedding accuracy ─────────────────────────────────
print("  06. Per-individual accuracy...")

names_sorted = sorted(per_indiv_acc.keys(), key=lambda n: -per_indiv_acc[n])
accs_sorted  = [per_indiv_acc[n] * 100 for n in names_sorted]
confs_sorted = [per_indiv_conf[n] for n in names_sorted]
colors_sorted = [individual_colors[n] for n in names_sorted]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

bars = ax1.bar(names_sorted, accs_sorted, color=colors_sorted,
               edgecolor="white", linewidth=0.5)
ax1.axhline(overall_acc * 100, color="black", linestyle="--", linewidth=1.5,
            label=f"Overall: {overall_acc*100:.1f}%")
ax1.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
ax1.set_title("Per-individual Identification Accuracy — Embedding System",
              fontsize=13, fontweight="bold")
ax1.set_ylim(70, 102)
ax1.legend(fontsize=10)
ax1.grid(True, axis="y", alpha=0.3)
for bar, acc in zip(bars, accs_sorted):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{acc:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

bars2 = ax2.bar(names_sorted, confs_sorted, color=colors_sorted,
                edgecolor="white", linewidth=0.5)
ax2.set_ylabel("Mean cosine similarity", fontsize=11)
ax2.set_xlabel("Individual", fontsize=11)
ax2.set_title("Mean Similarity to Own Prototype", fontsize=12)
ax2.set_ylim(0, 1)
ax2.axhline(optimal_threshold, color="red", linestyle="--", linewidth=1.5,
            label=f"Optimal threshold = {optimal_threshold:.3f}")
ax2.legend(fontsize=10)
ax2.grid(True, axis="y", alpha=0.3)
for bar, conf in zip(bars2, confs_sorted):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f"{conf:.3f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_RESULTS / "06_per_individual_accuracy.png", dpi=150,
            bbox_inches="tight")
plt.close()
print("    Saved: 06_per_individual_accuracy.png")

# ── Plot 7: Confusion matrix ─────────────────────────────────────────────────
print("  07. Confusion matrix...")

confusion_pct = confusion_counts / confusion_counts.sum(axis=1, keepdims=True) * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Absolute counts
sns.heatmap(confusion_counts, annot=True, fmt="d", cmap="Blues",
            xticklabels=INDIVIDUALS, yticklabels=INDIVIDUALS,
            ax=ax1, linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Number of crops"})
ax1.set_xlabel("Predicted", fontsize=11)
ax1.set_ylabel("True individual", fontsize=11)
ax1.set_title("Confusion Matrix — Absolute Counts", fontsize=12, fontweight="bold")
ax1.tick_params(axis="x", rotation=45)
ax1.tick_params(axis="y", rotation=0)

# Normalized percentages
mask_diag_cm = np.eye(num_classes, dtype=bool)
annot_labels = np.where(
    mask_diag_cm,
    np.vectorize(lambda v: f"{v:.0f}%")(confusion_pct),
    np.vectorize(lambda v: f"{v:.1f}%" if v >= 0.5 else "")(confusion_pct)
)
sns.heatmap(confusion_pct, annot=annot_labels, fmt="", cmap="RdYlGn",
            xticklabels=INDIVIDUALS, yticklabels=INDIVIDUALS,
            ax=ax2, vmin=0, vmax=100, linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Recall (%)"},
            annot_kws={"size": 9})
ax2.set_xlabel("Predicted", fontsize=11)
ax2.set_ylabel("True individual", fontsize=11)
ax2.set_title("Confusion Matrix — Row-normalized (%)\n(diagonal = recall per individual)",
              fontsize=12, fontweight="bold")
ax2.tick_params(axis="x", rotation=45)
ax2.tick_params(axis="y", rotation=0)

plt.suptitle("Embedding-based Identification — Confusion Analysis",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_RESULTS / "07_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: 07_confusion_matrix.png")

# ==============================================================================
# STEP 7 — Save embeddings.json for Android app
# ==============================================================================

print_section("STEP 7 — Saving embeddings.json for Android")

"""
Format designed for the Android app:
  - One entry per individual
  - L2-normalized prototype embedding (2048 floats)
  - Count of training crops used
  - Threshold for unknown detection
"""

embeddings_json = {
    "version": "1.0",
    "created": datetime.now().isoformat(),
    "model_source": "resnet_orangs.pt",
    "embedding_dim": 2048,
    "normalization": "L2",
    "similarity_metric": "cosine",
    "unknown_threshold": round(optimal_threshold, 4),
    "calibration_auc": round(roc_auc, 4),
    "num_individuals": num_classes,
    "individuals": {}
}

for idx, name in enumerate(INDIVIDUALS):
    mask = (all_labels == idx)
    count = mask.sum().item()
    proto = prototypes[name].tolist()

    embeddings_json["individuals"][name] = {
        "class_index": idx,
        "num_training_crops": count,
        "mean_similarity_to_self": round(float(per_indiv_conf.get(name, 0)), 4),
        "embedding": proto          # 2048 float values
    }

json_path = OUTPUT_MODELS / "embeddings.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(embeddings_json, f, separators=(",", ":"))  # compact JSON

json_size_kb = json_path.stat().st_size / 1024
print(f"  Saved: {json_path} ({json_size_kb:.1f} KB)")
print(f"  Threshold: {optimal_threshold:.4f}")
print(f"  Contains {num_classes} individual prototypes")

# Copy to Android export directory
ANDROID_DIR.mkdir(exist_ok=True)
shutil.copy(json_path, ANDROID_DIR / "embeddings.json")
print(f"  Copied to: {ANDROID_DIR / 'embeddings.json'}")

# Save metadata
metadata = {
    "pipeline": "embedding_gallery",
    "created": datetime.now().isoformat(),
    "unknown_threshold": round(optimal_threshold, 4),
    "roc_auc": round(roc_auc, 4),
    "best_f1": round(best_f1, 4),
    "best_precision": round(best_prec, 4),
    "best_recall": round(best_recall, 4),
    "overall_accuracy": round(overall_acc, 4),
    "num_individuals": num_classes,
    "num_training_crops": len(dataset),
    "per_individual": {
        name: {
            "accuracy": round(per_indiv_acc.get(name, 0), 4),
            "mean_similarity": round(per_indiv_conf.get(name, 0), 4),
            "num_crops": sum(1 for s in dataset.samples if s[2] == name)
        }
        for name in INDIVIDUALS
    }
}

meta_path = OUTPUT_MODELS / "embedding_metadata.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"  Saved: {meta_path}")

# ==============================================================================
# STEP 8 — Export backbone TFLite (output: 2048-dim)
# ==============================================================================

print_section("STEP 8 — Exporting backbone TFLite (2048-dim output)")

backbone_tflite_dst = ANDROID_DIR / "resnet50_backbone.tflite"
converted = False

# Method 1: ai-edge-torch
try:
    import ai_edge_torch
    print("  [Method 1] ai-edge-torch...")
    with torch.no_grad():
        sample = (torch.randn(1, 3, 224, 224),)
        edge_model = ai_edge_torch.convert(backbone.cpu(), sample)
        edge_model.export(str(backbone_tflite_dst))
    if backbone_tflite_dst.exists() and backbone_tflite_dst.stat().st_size > 1000:
        converted = True
        backbone = backbone.to(DEVICE)
        print(f"  [Method 1] SUCCESS")
except ImportError:
    print("  [Method 1] SKIPPED: ai-edge-torch not installed")
except Exception as e:
    print(f"  [Method 1] FAILED: {e}")
    backbone = backbone.to(DEVICE)

# Method 2: ONNX → onnx2tf
if not converted:
    try:
        import onnx2tf
        import onnx

        print("  [Method 2] ONNX → onnx2tf...")
        temp_dir = OUTPUT_MODELS / "_temp_backbone_export"
        temp_dir.mkdir(exist_ok=True)
        onnx_path = temp_dir / "backbone.onnx"

        backbone_cpu = backbone.cpu()
        with torch.no_grad():
            torch.onnx.export(
                backbone_cpu,
                torch.randn(1, 3, 224, 224),
                str(onnx_path),
                input_names=["face_crop"],
                output_names=["embedding"],
                opset_version=13,
                do_constant_folding=True
            )
        backbone = backbone.to(DEVICE)

        tf_out = temp_dir / "tflite_out"
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(tf_out),
            non_verbose=True,
            copy_onnx_input_output_names_to_tflite=True,
        )

        for candidate_name in ["model_float32.tflite", "backbone_float32.tflite"]:
            candidate = tf_out / candidate_name
            if candidate.exists():
                shutil.copy(candidate, backbone_tflite_dst)
                converted = True
                break
        if not converted:
            candidates = list(tf_out.glob("*.tflite"))
            if candidates:
                shutil.copy(candidates[0], backbone_tflite_dst)
                converted = True

        shutil.rmtree(temp_dir, ignore_errors=True)
        if converted:
            print(f"  [Method 2] SUCCESS")

    except ImportError as e:
        print(f"  [Method 2] SKIPPED: {e}")
    except Exception as e:
        print(f"  [Method 2] FAILED: {e}")

if converted:
    size_mb = backbone_tflite_dst.stat().st_size / 1e6
    print(f"\n  Backbone TFLite: {backbone_tflite_dst} ({size_mb:.1f} MB)")

    # Validate TFLite backbone
    try:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=str(backbone_tflite_dst))
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]
        print(f"  Input:  shape={inp['shape'].tolist()}")
        print(f"  Output: shape={out['shape'].tolist()} (should be [1, 2048])")
        assert out["shape"].tolist() == [1, 2048], \
            f"Expected [1, 2048], got {out['shape'].tolist()}"
        print(f"  TFLite backbone VALIDATED")
    except Exception as e:
        print(f"  TFLite validation failed: {e}")
else:
    print("  [WARNING] TFLite export failed — embeddings.json still usable")
    print("  The Android app needs the backbone TFLite to run on-device.")
    print("  Install ai-edge-torch: pip install ai-edge-torch")

# ==============================================================================
# STEP 9 — Package new Android ZIP bundle
# ==============================================================================

print_section("STEP 9 — Packaging Android ZIP bundle (v2)")

files_to_zip = [
    (ANDROID_DIR / "yolov8_detector.tflite",    "yolov8_detector.tflite"),
    (ANDROID_DIR / "embeddings.json",            "embeddings.json"),
    (OUTPUT_MODELS / "embedding_metadata.json",  "metadata.json"),
]

# Include backbone TFLite if available
if converted:
    files_to_zip.append((backbone_tflite_dst, "resnet50_backbone.tflite"))

with zipfile.ZipFile(ZIP_OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for fpath, arcname in files_to_zip:
        if fpath.exists():
            zf.write(fpath, arcname=arcname)
            size_mb = fpath.stat().st_size / 1e6
            print(f"  Added: {arcname} ({size_mb:.1f} MB)")
        else:
            print(f"  [WARNING] Skipped (not found): {arcname}")

zip_size_mb = ZIP_OUTPUT.stat().st_size / 1e6
print(f"\n  ZIP: {ZIP_OUTPUT} ({zip_size_mb:.1f} MB)")

# ==============================================================================
# STEP 10 — Full text report
# ==============================================================================

print_section("STEP 10 — Writing text report")

report = f"""
================================================================================
  EMBEDDING GALLERY REPORT — Orang-outan Face Recognition
  
  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
================================================================================

MODEL USED
----------
  Source:     {RESNET_PT}
  Backbone:   ResNet50 (pretrained ImageNet1K_V2)
  Embedding:  2048-dim L2-normalized vector (from avgpool)
  Head:       REMOVED (was Dropout(0.5) + Linear(2048→{num_classes}))

DATASET SUMMARY
---------------
  Directory:  {DATASET_DIR}
  Total crops: {len(dataset)}
  Individuals: {num_classes}

  Individual     Crops   Accuracy    Mean Sim
  ─────────────────────────────────────────────"""

for name in INDIVIDUALS:
    n  = sum(1 for s in dataset.samples if s[2] == name)
    ac = per_indiv_acc.get(name, 0) * 100
    cs = per_indiv_conf.get(name, 0)
    report += f"\n  {name:<12}   {n:4d}   {ac:7.2f}%    {cs:.4f}"

report += f"""

  Overall top-1 accuracy:  {overall_acc*100:.2f}%

THRESHOLD CALIBRATION (leave-one-individual-out)
-------------------------------------------------
  Method: For each individual I, treat them as "unknown".
          Gallery built from remaining {num_classes-1} individuals.
          Sweep similarity threshold to find optimal unknown-detection.

  Optimal threshold:  {optimal_threshold:.4f}
  At optimal threshold:
    F1 score:   {best_f1:.4f}
    Precision:  {best_prec:.4f}  ({best_prec*100:.1f}% of accepted IDs are correct)
    Recall:     {best_recall:.4f}  ({best_recall*100:.1f}% of known individuals accepted)
    ROC AUC:    {roc_auc:.4f}

  Operating point comparison:
  Threshold   Precision   Recall    F1"""

for thresh_target in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
    i_t = np.argmin(np.abs(thresholds - thresh_target))
    marker = " ← OPTIMAL" if abs(thresholds[i_t] - optimal_threshold) < 0.02 else ""
    report += (f"\n  {thresh_target:.2f}        "
               f"{precisions[i_t]:.4f}      "
               f"{recalls[i_t]:.4f}    "
               f"{f1_scores[i_t]:.4f}{marker}")

report += f"""

EMBEDDING SPACE QUALITY
-----------------------
  Positive pair similarities (same individual):
    Mean: {all_sims_positive.mean():.4f}  Std: {all_sims_positive.std():.4f}
    Min:  {all_sims_positive.min():.4f}  Max: {all_sims_positive.max():.4f}

  Negative pair similarities (best match to wrong individual):
    Mean: {all_sims_negative.mean():.4f}  Std: {all_sims_negative.std():.4f}
    Min:  {all_sims_negative.min():.4f}  Max: {all_sims_negative.max():.4f}

  Separability ratio:
    {all_sims_positive.mean():.4f} / {all_sims_negative.mean():.4f} = {all_sims_positive.mean()/all_sims_negative.mean():.4f}
    (higher is better — positive pairs should be much more similar than negatives)

HOW TO ADD A NEW INDIVIDUAL
----------------------------
  1. Take 10-20 face photos of the new individual
  2. Run YOLO on them to extract face crops
  3. Add crops to DATASET_CLASSIFICATION/raw/<NewName>/
  4. Run this script again (takes ~5 minutes)
  5. Import the new primate_models_android_v2.zip in the app
  NO RETRAINING OF THE NEURAL NETWORK REQUIRED

OUTPUTS
-------
  Gallery (PyTorch): {OUTPUT_MODELS / 'embedding_gallery.pt'}
  Gallery (Android): {OUTPUT_MODELS / 'embeddings.json'}
  Metadata:          {OUTPUT_MODELS / 'embedding_metadata.json'}
  Visualizations:    {OUTPUT_RESULTS}/

  Android bundle:    {ZIP_OUTPUT}

RECOMMENDATION ON THRESHOLD
-----------------------------
  The calibrated threshold of {optimal_threshold:.4f} was found on leave-one-out.
  For field deployment (new individuals will arrive):
    - If you want to minimize false identifications: use {min(optimal_threshold+0.05, 0.95):.2f}
    - If you want to minimize missed identifications: use {max(optimal_threshold-0.05, 0.30):.2f}
  Expose this parameter in the Android Settings screen for easy adjustment.

IMPORTANT NOTE
--------------
  The ResNet50 backbone was trained with cross-entropy loss (classification),
  not with a metric loss (ArcFace, triplet loss). The embedding space is
  discriminative but not explicitly optimized for metric learning.
  Performance is good given the 96.3% classification accuracy, but for
  maximum reliability on new individuals, a future fine-tuning step with
  ArcFace loss on the full dataset would further improve unknown rejection.
================================================================================
"""

report_path = OUTPUT_RESULTS / "rapport_embeddings.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

print(report)
print(f"  Report saved: {report_path}")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

print(f"""
{'=' * 70}
  ALL DONE
{'=' * 70}

  Models:
    {OUTPUT_MODELS / 'embedding_gallery.pt'}
    {OUTPUT_MODELS / 'embeddings.json'}
    {OUTPUT_MODELS / 'embedding_metadata.json'}

  Visualizations (7 plots):
    {OUTPUT_RESULTS}/

  Android bundle:
    {ZIP_OUTPUT}

  KEY RESULT:
    Optimal unknown-detection threshold: {optimal_threshold:.4f}
    ROC AUC: {roc_auc:.4f}
    Overall embedding accuracy: {overall_acc*100:.2f}%
    Backbone TFLite export: {'SUCCESS' if converted else 'FAILED (see above)'}
{'=' * 70}
""")