"""
V2_5_train_arcface.py
======================
Orang-outan V2 pipeline


WHAT THIS DOES
--------------
Fine-tunes MegaDescriptor-T-224 with Sub-center ArcFace loss on:
  - 2127 zoo crops (10 labeled individuals — clean, ground truth)
  - 5429 wild crops (11th "background" class — noisy, diverse)

WHY SUB-CENTER ARCFACE
-----------------------
Standard ArcFace assumes each class forms ONE tight cluster.
With noisy web images (blurry, IA-generated, bad angles),
the "unknown" class has high intra-class variance.

Sub-center ArcFace (Deng et al. ECCV 2020) assigns K sub-centers per class:
  - Known individuals: K=1 (they are clean and consistent)
  - Wild/unknown class: K=5 (handles noise, poses, quality variation)
  
This is exactly the method used by MiewID (ConservationXLabs 2024) which
outperforms MegaDescriptor by 19.2% on unseen species.

WHY WILD CROPS AS CLASS 11
---------------------------
By making wild crops a labeled "background" class during training,
the model learns explicitly:
  - What orangutan faces look like IN GENERAL (from 5000 diverse images)
  - What the 10 SPECIFIC individuals look like (from 2127 clean crops)
  - To PUSH the 10 individual clusters AWAY from the background cloud

At inference time, a new unknown individual will land close to the
"background" cluster rather than being forced into one of the 10 known
clusters. The cosine similarity to the 10 known prototypes will be low
→ detected as "Unknown individual".

AUGMENTATION STRATEGY
---------------------
Heavy augmentation simulates field conditions:
  - Rotation ±30° (orangs hang in all positions)
  - Blur (motion blur, out of focus)
  - JPEG compression artifacts (phone cameras)
  - Color jitter heavy (lighting varies in jungle)
  - Cutout (partial occlusion by branches)
  - Random erasing (simulate partial face visibility)

On wild crops: apply EXTRA blur and noise to handle already-degraded images.

GRACEFUL QUIT
-------------
Press Ctrl+C at any time. The script catches the interrupt and:
  1. Saves the current best model immediately
  2. Generates the gallery + calibration report
  3. Exports TFLite
You will NOT lose any work.

HARDWARE
--------
RTX 3050 (4GB VRAM):
  - Batch 32, image 224×224 → ~2.8GB VRAM
  - ~25-35 seconds per epoch (2127+1000 wild = ~3127 crops/epoch)
  - 50 epochs → ~25 minutes
  - Target: done in 45 minutes total

RUN
---
  conda activate orangs
  python v3_megadesc_arcface_10ind/04_train_arcface.py
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
import math
import time
import shutil
import signal
import random
import warnings
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, deque




import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import timm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

# ==============================================================================
# PATHS
# ==============================================================================

V2_BASE = OUTPUT_DIR / "v2"
ZOO_DIR = CROPS_KNOWN_DIR
WILD_DIR = CROPS_WILD_DIR

RESULTS_DIR  = V2_BASE / "RESULTS" / "arcface_training"
EMBED_DIR    = V2_BASE / "EMBEDDINGS"

MODEL_SAVE     = MODELS_DIR / "megadesc_T_arcface.pt"
BACKBONE_SAVE  = MODELS_DIR / "megadesc_T_arcface_backbone.pt"
GALLERY_JSON   = EMBED_DIR  / "embeddings_arcface.json"
METADATA_SAVE  = MODELS_DIR / "arcface_metadata.json"
# Resume checkpoint — contains optimizer + scheduler state + current epoch
# Written every epoch so training can be interrupted and resumed cleanly
RESUME_CKPT    = MODELS_DIR / "megadesc_T_arcface_resume.pt"

for d in [MODELS_DIR, RESULTS_DIR, EMBED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================

IMG_SIZE     = 224
BATCH_SIZE   = 32        # fits in 4GB VRAM
SEED         = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ArcFace parameters (from literature — optimal for wildlife re-id)
ARC_SCALE    = 64        # feature norm scale — higher = more discriminative
ARC_MARGIN   = 0.50     # angular margin in radians — from MiewID paper
K_KNOWN      = 1        # sub-centers for clean known individuals
K_UNKNOWN    = 5        # sub-centers for noisy wild "background" class

# Training
LR_BACKBONE  = 2e-5     # very low — backbone is already good
LR_HEAD      = 1e-3     # higher for the new ArcFace head
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 60       # with early stopping
PATIENCE     = 15       # stop if no improvement for N epochs
WARMUP_EPOCHS = 3       # linear warmup

# Wild crops: how many to sample per epoch (avoid overwhelming zoo crops)
WILD_SAMPLES_PER_EPOCH = 1000   # out of ~5429

# Val split: 15% of zoo crops only (wild crops not validated — too noisy)
VAL_RATIO = 0.15

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==============================================================================
# GRACEFUL QUIT — save on Ctrl+C
# ==============================================================================

_interrupt_flag = False
_best_model_state = None
_best_epoch = 0

def _handle_sigint(sig, frame):
    global _interrupt_flag
    if not _interrupt_flag:
        print("\n\n  [Ctrl+C] Interrupt received. Saving best model...")
        _interrupt_flag = True
    else:
        print("\n  [Ctrl+C twice] Force quitting now.")
        sys.exit(1)

signal.signal(signal.SIGINT, _handle_sigint)

# ==============================================================================
# SUB-CENTER ARCFACE LOSS
# ==============================================================================

class SubCenterArcFaceLoss(nn.Module):
    """
    Sub-center ArcFace (Deng et al. ECCV 2020).
    
    Each class has K sub-centers instead of 1.
    The sample is assigned to its CLOSEST sub-center.
    
    Benefits:
      - Handles intra-class variation (pose, lighting)
      - Robust to label noise (noisy samples cluster in non-dominant sub-centers)
      - Maintains ArcFace discriminativity for the dominant sub-center
    
    Args:
      embedding_dim: feature vector size (768 for MegaDescriptor-T)
      num_classes: total number of classes (10 zoo + 1 wild = 11)
      k_per_class: list of K values per class (e.g. [1]*10 + [5])
      scale: feature norm scale (s=64 from MiewID paper)
      margin: angular margin in radians (m=0.5 from MiewID paper)
    """

    def __init__(self, embedding_dim: int, num_classes: int,
                 k_per_class: list, scale: float = 64.0, margin: float = 0.5):
        super().__init__()
        self.scale       = scale
        self.margin      = margin
        self.num_classes = num_classes
        self.k_per_class = k_per_class

        # Sub-centers for each class: concatenated [K0 + K1 + ... + K_{N-1}, emb_dim]
        total_subcenters = sum(k_per_class)
        self.weight = nn.Parameter(
            torch.FloatTensor(total_subcenters, embedding_dim)
        )
        nn.init.xavier_uniform_(self.weight)

        # Precompute which sub-centers belong to which class
        # class_start[i] = index of first sub-center for class i
        self.register_buffer("class_start",
                             torch.tensor([sum(k_per_class[:i]) for i in range(num_classes)],
                                          dtype=torch.long))
        self.register_buffer("class_k",
                             torch.tensor(k_per_class, dtype=torch.long))

        # ArcFace cos/sin for margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th    = math.cos(math.pi - margin)
        self.mm    = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [B, emb_dim], L2-normalized
        labels:     [B], class indices
        Returns: scalar loss
        """
        # L2 normalize sub-center weights
        w_norm = F.normalize(self.weight, dim=1)   # [total_K, emb_dim]

        # Cosine similarity of each embedding to ALL sub-centers
        cos_all = embeddings @ w_norm.T            # [B, total_K]

        # For each class, keep only the max similarity among its sub-centers
        # (assign each sample to the closest sub-center of each class)
        logits = torch.zeros(embeddings.size(0), self.num_classes,
                             device=embeddings.device)
        for c in range(self.num_classes):
            start = self.class_start[c].item()
            k     = self.class_k[c].item()
            logits[:, c] = cos_all[:, start:start+k].max(dim=1).values

        # Apply ArcFace margin to the correct class only
        # cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        cos_theta = logits.clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - cos_theta ** 2).clamp(0, 1)
        phi       = cos_theta * self.cos_m - sin_theta * self.sin_m

        # Safe: use phi where cos_theta > th, else use linear approximation
        phi = torch.where(cos_theta > self.th, phi,
                          cos_theta - self.mm)

        # Replace target class logit with phi
        one_hot     = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        output      = one_hot * phi + (1.0 - one_hot) * logits
        output     *= self.scale

        return F.cross_entropy(output, labels, label_smoothing=0.05)

# ==============================================================================
# DATASETS
# ==============================================================================

# MegaDescriptor normalization
MEGA_MEAN = [0.5, 0.5, 0.5]
MEGA_STD  = [0.5, 0.5, 0.5]

def get_zoo_transforms():
    """Augmentation for clean zoo crops — simulate field conditions."""
    norm = T.Normalize(MEGA_MEAN, MEGA_STD)

    train_tf = T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.65, 1.0), ratio=(0.85, 1.15)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.1),           # orangs hang upside down
        T.RandomRotation(degrees=30),
        T.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.3, hue=0.1),
        T.RandomGrayscale(p=0.05),
        T.RandomPerspective(distortion_scale=0.3, p=0.3),
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.5)),
        T.ToTensor(),
        # Simulate JPEG artifacts from phone cameras
        T.RandomApply([T.GaussianBlur(3, sigma=(1.0, 3.0))], p=0.2),
        T.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.0), value="random"),
        norm,
    ])

    val_tf = T.Compose([
        T.Resize(IMG_SIZE),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        norm,
    ])

    return train_tf, val_tf

def get_wild_transforms():
    """
    Extra degradation for already-noisy wild crops.
    These images may be blurry, compressed, AI-generated.
    We add more noise to make the model robust to them.
    """
    norm = T.Normalize(MEGA_MEAN, MEGA_STD)
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0), ratio=(0.7, 1.4)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=45),
        T.ColorJitter(brightness=0.7, contrast=0.6, saturation=0.5, hue=0.15),
        T.RandomGrayscale(p=0.15),
        T.GaussianBlur(kernel_size=7, sigma=(0.5, 4.0)),
        T.ToTensor(),
        T.RandomErasing(p=0.3, scale=(0.05, 0.25), ratio=(0.3, 3.0), value="random"),
        norm,
    ])

class ZooDataset(Dataset):
    def __init__(self, paths: list, labels: list, transform):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        return self.transform(img), self.labels[idx]

class WildDataset(Dataset):
    """
    Wild orangutan crops — used as the "background/unknown" class.
    Subsampled each epoch for balance.
    """

    def __init__(self, wild_dir: Path, transform, subsample: int, unknown_class: int):
        self.transform     = transform
        self.subsample     = subsample
        self.unknown_class = unknown_class   # passed explicitly — safe in workers
        exts = {".jpg", ".jpeg", ".png"}
        all_files = sorted([f for f in wild_dir.iterdir()
                            if f.suffix.lower() in exts])
        self.all_files = all_files
        print(f"  Wild crops found: {len(all_files):,}")
        self._resample()

    def _resample(self):
        """Pick a random subset each epoch."""
        n = min(self.subsample, len(self.all_files))
        self.files = random.sample(self.all_files, n)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.files[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (100, 80, 60))
        return self.transform(img), self.unknown_class

# ==============================================================================
# LOAD MEGADESCRIPTOR
# ==============================================================================

def load_backbone() -> nn.Module:
    print("  Loading MegaDescriptor-T-224...")
    model = timm.create_model(
        "hf-hub:BVRA/MegaDescriptor-T-224",
        pretrained=True,
        num_classes=0          # no classification head
    )
    model.eval()
    # Get embedding dim
    with torch.no_grad():
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        emb   = model(dummy)
    emb_dim = emb.shape[1]
    print(f"  Backbone OK — embedding dim: {emb_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model, emb_dim

# ==============================================================================
# ZOO DATASET LOADING
# ==============================================================================

def load_zoo_dataset():
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    dirs = sorted([d for d in ZOO_DIR.iterdir()
                   if d.is_dir() and not d.name.startswith("_")])
    classes = [d.name for d in dirs]
    c2i = {c: i for i, c in enumerate(classes)}

    paths, labels = [], []
    for d in dirs:
        imgs = sorted([f for f in d.iterdir() if f.suffix in exts])
        for img in imgs:
            paths.append(img)
            labels.append(c2i[d.name])

    print(f"  Zoo individuals: {classes}")
    print(f"  Zoo crops total: {len(paths):,}")
    counts = Counter(labels)
    for i, name in enumerate(classes):
        print(f"    [{i:2d}] {name:<12}: {counts[i]:4d} crops")

    # Stratified train/val split
    from sklearn.model_selection import train_test_split
    idx = list(range(len(paths)))
    tr_idx, va_idx, _, _ = train_test_split(
        idx, labels, test_size=VAL_RATIO, stratify=labels, random_state=SEED
    )
    tr_paths  = [paths[i] for i in tr_idx]
    tr_labels = [labels[i] for i in tr_idx]
    va_paths  = [paths[i] for i in va_idx]
    va_labels = [labels[i] for i in va_idx]

    print(f"  Train: {len(tr_paths):,}  Val: {len(va_paths):,}")
    return tr_paths, tr_labels, va_paths, va_labels, classes

# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train_one_epoch(backbone, arc_loss, train_loader, optimizer, scheduler_warmup,
                    epoch, max_epochs, emb_dim):
    backbone.train()
    arc_loss.train()

    total_loss = 0.0
    total_n    = 0
    t_start    = time.time()

    bar_width = 40
    for i, (imgs, labs) in enumerate(train_loader):
        if _interrupt_flag:
            break

        imgs = imgs.to(DEVICE, non_blocking=True)
        labs = labs.to(DEVICE, non_blocking=True)

        embeddings = backbone(imgs)
        embeddings = F.normalize(embeddings, dim=1)

        loss = arc_loss(embeddings, labs)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(backbone.parameters()) + list(arc_loss.parameters()),
            max_norm=1.0
        )
        optimizer.step()
        if scheduler_warmup is not None:
            scheduler_warmup.step()

        total_loss += loss.item() * imgs.size(0)
        total_n    += imgs.size(0)

        # Progress bar
        done    = i + 1
        total_b = len(train_loader)
        pct     = done / total_b
        fill    = int(bar_width * pct)
        bar     = "█" * fill + "░" * (bar_width - fill)
        elapsed = time.time() - t_start
        eta     = elapsed / done * (total_b - done)
        print(f"\r  Train [{bar}] {done}/{total_b}"
              f"  loss={total_loss/total_n:.4f}"
              f"  ETA={str(timedelta(seconds=int(eta)))}",
              end="", flush=True)

    print()
    return total_loss / max(total_n, 1)

@torch.no_grad()
def validate(backbone, val_loader, num_zoo_classes):
    """
    Validation: compute top-1 accuracy on zoo crops ONLY.
    Uses the prototype (mean embedding) of each class on training data.
    We do nearest-prototype classification.
    """
    backbone.eval()
    all_embs  = []
    all_labels = []

    for imgs, labs in val_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        embs = F.normalize(backbone(imgs), dim=1)
        all_embs.append(embs.cpu())
        all_labels.extend(labs.tolist())

    all_embs   = torch.cat(all_embs, dim=0)
    all_labels = torch.tensor(all_labels)

    # Note: val set is zoo only, labels 0..9
    # Build prototype from val set itself (leave-one-out approximation)
    proto = torch.zeros(num_zoo_classes, all_embs.shape[1])
    for c in range(num_zoo_classes):
        mask = (all_labels == c)
        if mask.any():
            proto[c] = F.normalize(all_embs[mask].mean(dim=0), dim=0)

    # Nearest prototype
    sims  = all_embs @ proto.T              # [N, num_zoo_classes]
    preds = sims.argmax(dim=1)
    acc   = (preds == all_labels).float().mean().item()
    return acc

# ==============================================================================
# SAVE BEST MODEL
# ==============================================================================

def save_model(backbone, arc_loss, classes, emb_dim, epoch, val_acc, note=""):
    state = {
        "backbone_state": backbone.state_dict(),
        "arc_loss_state": arc_loss.state_dict(),
        "classes":        classes,
        "num_classes":    len(classes),
        "emb_dim":        emb_dim,
        "epoch":          epoch,
        "val_acc":        val_acc,
        "img_size":       IMG_SIZE,
        "arc_scale":      ARC_SCALE,
        "arc_margin":     ARC_MARGIN,
        "normalization":  {"mean": MEGA_MEAN, "std": MEGA_STD},
        "save_time":      datetime.now().isoformat(),
        "note":           note,
    }
    # Atomic save
    tmp = MODEL_SAVE.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(MODEL_SAVE)

    # Backbone only
    bb_tmp = BACKBONE_SAVE.with_suffix(".tmp")
    torch.save({
        "backbone_state": backbone.state_dict(),
        "emb_dim":        emb_dim,
        "classes":        classes,
        "img_size":       IMG_SIZE,
        "normalization":  {"mean": MEGA_MEAN, "std": MEGA_STD},
    }, bb_tmp)
    bb_tmp.replace(BACKBONE_SAVE)

    size_mb = MODEL_SAVE.stat().st_size / 1e6
    print(f"  Saved: {MODEL_SAVE.name} ({size_mb:.1f} MB){note}")

# ==============================================================================
# POST-TRAINING: GALLERY + CALIBRATION
# ==============================================================================

@torch.no_grad()
def build_gallery(backbone, zoo_tr_paths, zoo_tr_labels, classes, emb_dim):
    """
    Builds the embedding gallery for Android app:
      - One L2-normalized prototype per known individual
      - Calibrated cosine similarity threshold
    """
    print("\n  Building embedding gallery...")
    backbone.eval()

    val_tf = T.Compose([
        T.Resize(IMG_SIZE),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(MEGA_MEAN, MEGA_STD),
    ])

    ds = ZooDataset(zoo_tr_paths, zoo_tr_labels, val_tf)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    all_embs   = []
    all_labels = []

    for imgs, labs in dl:
        imgs = imgs.to(DEVICE, non_blocking=True)
        embs = F.normalize(backbone(imgs), dim=1)
        all_embs.append(embs.cpu())
        all_labels.extend(labs.tolist())

    all_embs   = torch.cat(all_embs, dim=0)
    all_labels = torch.tensor(all_labels)
    num_known  = len(classes)

    # Prototypes
    prototypes = {}
    proto_matrix = torch.zeros(num_known, emb_dim)
    for i, name in enumerate(classes):
        mask  = (all_labels == i)
        count = mask.sum().item()
        proto = all_embs[mask].mean(dim=0)
        proto = F.normalize(proto, dim=0)
        prototypes[name] = proto
        proto_matrix[i]  = proto
        print(f"    {name:<12}: {count:4d} crops → prototype OK")

    # Separability metrics
    sims_pos, sims_neg = [], []
    for i in range(num_known):
        mask    = (all_labels == i)
        embs_i  = all_embs[mask]
        own_sim = (embs_i @ proto_matrix[i]).tolist()
        sims_pos.extend(own_sim)
        other   = torch.cat([proto_matrix[:i], proto_matrix[i+1:]], dim=0)
        best_other = (embs_i @ other.T).max(dim=1).values.tolist()
        sims_neg.extend(best_other)

    sims_pos = np.array(sims_pos)
    sims_neg = np.array(sims_neg)
    sep      = sims_pos.mean() / sims_neg.mean()

    print(f"\n  Positive similarity : {sims_pos.mean():.4f} ± {sims_pos.std():.4f}")
    print(f"  Negative similarity : {sims_neg.mean():.4f} ± {sims_neg.std():.4f}")
    print(f"  Separability ratio  : {sep:.4f}")
    print(f"  (V1 ResNet50 was 1.7203 — higher is better)")

    # Threshold calibration (leave-one-individual-out)
    thresholds = np.linspace(0.0, 1.0, 500)
    f1_scores  = []
    for t in thresholds:
        all_sims_loio_known  = sims_pos
        all_sims_loio_unknown = sims_neg
        tp = (all_sims_loio_known   >= t).sum()
        fp = (all_sims_loio_unknown >= t).sum()
        fn = (all_sims_loio_known   <  t).sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        f1_scores.append(f1)

    best_idx   = int(np.argmax(f1_scores))
    opt_thresh = float(thresholds[best_idx])
    best_f1    = float(f1_scores[best_idx])
    print(f"\n  Calibrated threshold: {opt_thresh:.4f}  (F1={best_f1:.4f})")

    # Save embeddings.json for Android
    gallery = {
        "version":            "2.0-arcface",
        "created":            datetime.now().isoformat(),
        "model":              "MegaDescriptor-T-224 + SubCenterArcFace",
        "embedding_dim":      emb_dim,
        "normalization":      "L2",
        "similarity_metric":  "cosine",
        "unknown_threshold":  round(opt_thresh, 4),
        "separability_ratio": round(float(sep), 4),
        "num_individuals":    num_known,
        "individuals":        {
            name: {
                "class_index":      i,
                "num_training_crops": int((all_labels == i).sum().item()),
                "mean_similarity":  round(float((all_embs[all_labels==i] @ proto_matrix[i]).mean().item()), 4),
                "embedding":        proto_matrix[i].tolist()
            }
            for i, name in enumerate(classes)
        }
    }

    json_tmp = GALLERY_JSON.with_suffix(".tmp")
    with open(json_tmp, "w", encoding="utf-8") as f:
        json.dump(gallery, f, separators=(",", ":"))
    json_tmp.replace(GALLERY_JSON)

    # Also copy to ANDROID_EXPORT
    android_dir = V2_BASE / "ANDROID_EXPORT"
    android_dir.mkdir(exist_ok=True)
    shutil.copy2(GALLERY_JSON, android_dir / "embeddings.json")

    print(f"\n  Gallery saved: {GALLERY_JSON} ({GALLERY_JSON.stat().st_size/1024:.1f} KB)")
    print(f"  Copied to   : {android_dir / 'embeddings.json'}")

    return opt_thresh, sep

# ==============================================================================
# MAIN
# ==============================================================================

def save_resume_ckpt(backbone, arc_loss, optimizer, scheduler, epoch,
                     best_val_acc, patience_count, history):
    """
    Save a full resume checkpoint so training can be restarted exactly
    from this point with --resume. Written atomically every epoch.
    Contains everything needed to reconstruct training state:
    backbone weights, ArcFace head weights, optimizer state, scheduler state,
    current epoch, best val accuracy, and patience counter.
    """
    tmp = RESUME_CKPT.with_suffix(".tmp")
    torch.save({
        "backbone_state":   backbone.state_dict(),
        "arc_loss_state":   arc_loss.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "scheduler_state":  scheduler.state_dict(),
        "epoch":            epoch,
        "best_val_acc":     best_val_acc,
        "patience_count":   patience_count,
        "history":          history,
        "saved_at":         datetime.now().isoformat(),
    }, tmp)
    tmp.replace(RESUME_CKPT)


def load_resume_ckpt(backbone, arc_loss, optimizer, scheduler):
    """
    Load a resume checkpoint produced by save_resume_ckpt().
    Returns (start_epoch, best_val_acc, patience_count, history).
    start_epoch is the NEXT epoch to run (checkpoint epoch + 1).
    Raises RuntimeError if checkpoint is incompatible.
    """
    if not RESUME_CKPT.exists():
        raise FileNotFoundError(
            f"Resume checkpoint not found: {RESUME_CKPT}\n"
            "Run without --resume to start a fresh training."
        )
    ckpt = torch.load(str(RESUME_CKPT), map_location=DEVICE, weights_only=False)
    backbone.load_state_dict(ckpt["backbone_state"])
    arc_loss.load_state_dict(ckpt["arc_loss_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch   = int(ckpt["epoch"]) + 1
    best_val_acc  = float(ckpt.get("best_val_acc", 0.0))
    patience_count = int(ckpt.get("patience_count", 0))
    history       = ckpt.get("history", {"train_loss": [], "val_acc": []})
    saved_at      = ckpt.get("saved_at", "?")
    print(f"  Resumed from checkpoint: epoch {ckpt['epoch']}")
    print(f"  Checkpoint saved at    : {saved_at}")
    print(f"  Best val acc so far    : {best_val_acc*100:.2f}%")
    print(f"  Patience count         : {patience_count}/{PATIENCE}")
    print(f"  Resuming at epoch      : {start_epoch}")
    return start_epoch, best_val_acc, patience_count, history


def main(resume: bool = False):
    global _best_model_state, _best_epoch

    t0 = time.time()

    print("=" * 70)
    print("  ARCFACE FINE-TUNING — MegaDescriptor-T-224")
    print("  Sub-center ArcFace | zoo + wild crops")
    print("  ")
    print("=" * 70)
    if resume:
        print("  Mode: RESUME from checkpoint")
    else:
        print("  Mode: fresh training (use --resume to continue an interrupted run)")
    print(f"  Device : {DEVICE}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU    : {props.name} ({props.total_memory/1e9:.1f} GB VRAM)")
    print(f"  Batch  : {BATCH_SIZE}  |  ArcFace scale={ARC_SCALE} margin={ARC_MARGIN}")
    print(f"  Sub-centers : known={K_KNOWN}  wild={K_UNKNOWN}")

    # ── 1. Load zoo dataset ───────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Loading zoo dataset...")
    print("─" * 70)
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("  pip install scikit-learn")
        sys.exit(1)

    tr_paths, tr_labels, va_paths, va_labels, classes = load_zoo_dataset()
    num_zoo = len(classes)
    unknown_class = num_zoo  # class index for wild crops
    num_total_classes = num_zoo + 1   # 10 zoo + 1 wild

    # ── 2. Load backbone ──────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Loading MegaDescriptor-T-224 backbone...")
    print("─" * 70)
    backbone, emb_dim = load_backbone()
    backbone = backbone.to(DEVICE)

    # ── 3. Sub-center ArcFace head ───────────────────────────────────────────
    k_per_class = [K_KNOWN] * num_zoo + [K_UNKNOWN]
    arc_loss    = SubCenterArcFaceLoss(
        embedding_dim=emb_dim,
        num_classes=num_total_classes,
        k_per_class=k_per_class,
        scale=ARC_SCALE,
        margin=ARC_MARGIN,
    ).to(DEVICE)

    total_subcent = sum(k_per_class)
    print(f"\n  Sub-center ArcFace:")
    print(f"    Known classes:  {num_zoo} × K={K_KNOWN} = {num_zoo*K_KNOWN} sub-centers")
    print(f"    Wild class:     1 × K={K_UNKNOWN} = {K_UNKNOWN} sub-centers")
    print(f"    Total:          {total_subcent} sub-centers × {emb_dim}-dim")

    # ── 4. DataLoaders ────────────────────────────────────────────────────────
    zoo_train_tf, zoo_val_tf = get_zoo_transforms()
    wild_tf                  = get_wild_transforms()

    zoo_train_ds = ZooDataset(tr_paths, tr_labels, zoo_train_tf)
    zoo_val_ds   = ZooDataset(va_paths, va_labels, zoo_val_tf)

    # Weighted sampler for zoo (compensate class imbalance)
    counts  = Counter(tr_labels)
    weights = [1.0 / counts[l] for l in tr_labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    zoo_train_loader = DataLoader(zoo_train_ds, batch_size=BATCH_SIZE,
                                  sampler=sampler, num_workers=4,
                                  pin_memory=True, persistent_workers=True)
    zoo_val_loader   = DataLoader(zoo_val_ds, batch_size=64, shuffle=False,
                                  num_workers=2, pin_memory=True,
                                  persistent_workers=True)

    # Wild dataset
    if WILD_DIR.exists():
        wild_ds = WildDataset(WILD_DIR, wild_tf, WILD_SAMPLES_PER_EPOCH, unknown_class)
        wild_loader = DataLoader(wild_ds, batch_size=BATCH_SIZE, shuffle=True,
                                 num_workers=4, pin_memory=True,
                                 persistent_workers=True)
        use_wild = True
        print(f"\n  Wild crops: {len(wild_ds.all_files):,} available, "
              f"{WILD_SAMPLES_PER_EPOCH} sampled/epoch")
    else:
        use_wild = False
        print(f"\n  [WARN] Wild crops dir not found: {WILD_DIR}")
        print(f"  Training on zoo crops only.")

    # ── 5. Optimizer + scheduler ──────────────────────────────────────────────
    optimizer = optim.AdamW([
        {"params": backbone.parameters(),  "lr": LR_BACKBONE},
        {"params": arc_loss.parameters(),  "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    # Cosine annealing with warmup
    total_steps   = MAX_EPOCHS * (len(zoo_train_loader) +
                                  (len(wild_loader) if use_wild else 0))
    warmup_steps  = WARMUP_EPOCHS * (len(zoo_train_loader) +
                                     (len(wild_loader) if use_wild else 0))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── 6. Resume from checkpoint (if --resume) ──────────────────────────────
    start_epoch    = 1
    best_val_acc   = 0.0
    patience_count = 0
    history        = {"train_loss": [], "val_acc": []}

    if resume:
        print("\n" + "─" * 70)
        print("  Loading resume checkpoint...")
        print("─" * 70)
        try:
            start_epoch, best_val_acc, patience_count, history = \
                load_resume_ckpt(backbone, arc_loss, optimizer, scheduler)
            _best_epoch = start_epoch - 1  # best epoch was before the resume point
        except FileNotFoundError as e:
            print(f"\n  ERROR: {e}")
            sys.exit(1)

    # ── 7. Training loop ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TRAINING — Ctrl+C to stop and save best model")
    if resume:
        print(f"  Resuming from epoch {start_epoch}/{MAX_EPOCHS}")
    print("=" * 70)

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        if _interrupt_flag:
            print("\n  Interrupt detected — stopping training loop.")
            break

        print(f"\n  Epoch {epoch}/{MAX_EPOCHS}  "
              f"patience={patience_count}/{PATIENCE}  "
              f"best_val={best_val_acc*100:.2f}%  "
              f"elapsed={str(timedelta(seconds=int(time.time()-t0)))}")

        # Resample wild crops each epoch
        if use_wild:
            wild_ds._resample()

        # Train on zoo crops
        zoo_loss = train_one_epoch(backbone, arc_loss, zoo_train_loader,
                                   optimizer, scheduler, epoch,
                                   MAX_EPOCHS, emb_dim)

        # Train on wild crops
        if use_wild and not _interrupt_flag:
            print(f"  Wild crops:")
            wild_loss = train_one_epoch(backbone, arc_loss, wild_loader,
                                        optimizer, scheduler, epoch,
                                        MAX_EPOCHS, emb_dim)
            avg_loss = (zoo_loss + wild_loss) / 2
        else:
            avg_loss = zoo_loss

        history["train_loss"].append(avg_loss)

        # Validate (zoo only)
        if not _interrupt_flag:
            print(f"  Validation:")
            val_acc = validate(backbone, zoo_val_loader, num_zoo)
            history["val_acc"].append(val_acc)
            print(f"  Val accuracy (nearest prototype): {val_acc*100:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc   = val_acc
                patience_count = 0
                save_model(backbone, arc_loss, classes, emb_dim, epoch,
                           val_acc, f"  ← BEST")
                _best_epoch = epoch
            else:
                patience_count += 1

            if patience_count >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}")
                break

            # ETA
            elapsed = time.time() - t0
            eta     = elapsed / epoch * (MAX_EPOCHS - epoch)
            print(f"  ETA: {str(timedelta(seconds=int(eta)))} "
                  f"| Total elapsed: {str(timedelta(seconds=int(elapsed)))}")

            # Save resume checkpoint every epoch (overwrite previous)
            # Allows clean restart with --resume after any interruption
            save_resume_ckpt(backbone, arc_loss, optimizer, scheduler,
                             epoch, best_val_acc, patience_count, history)

    # ── 7. Save on interrupt ──────────────────────────────────────────────────
    if _interrupt_flag and not MODEL_SAVE.exists():
        print("\n  Saving current state (no best checkpoint yet)...")
        save_model(backbone, arc_loss, classes, emb_dim, epoch, 0.0,
                   "  (saved on interrupt)")

    # ── 8. Load best model and build gallery ──────────────────────────────────
    print("\n" + "=" * 70)
    print("  POST-TRAINING: Loading best model + building gallery")
    print("=" * 70)

    if MODEL_SAVE.exists():
        ckpt = torch.load(MODEL_SAVE, map_location=DEVICE, weights_only=False)
        backbone.load_state_dict(ckpt["backbone_state"])
        print(f"  Best model loaded: epoch {ckpt.get('epoch','?')} "
              f"val_acc={ckpt.get('val_acc',0)*100:.2f}%")
    else:
        print("  No checkpoint found — using current model state.")

    threshold, sep = build_gallery(backbone, tr_paths, tr_labels, classes, emb_dim)

    # ── 9. Metadata ───────────────────────────────────────────────────────────
    elapsed_total = time.time() - t0
    metadata = {
        "created":           datetime.now().isoformat(),
        "method":            "Sub-center ArcFace + MegaDescriptor-T-224",
        "backbone":          "BVRA/MegaDescriptor-T-224",
        "embedding_dim":     emb_dim,
        "arc_scale":         ARC_SCALE,
        "arc_margin":        ARC_MARGIN,
        "k_known":           K_KNOWN,
        "k_unknown":         K_UNKNOWN,
        "num_zoo_classes":   num_zoo,
        "classes":           classes,
        "best_val_acc":      round(best_val_acc * 100, 2),
        "best_epoch":        _best_epoch,
        "separability":      round(sep, 4),
        "threshold":         round(threshold, 4),
        "zoo_crops":         len(tr_paths) + len(va_paths),
        "wild_crops_used":   WILD_SAMPLES_PER_EPOCH if use_wild else 0,
        "training_time_min": round(elapsed_total / 60, 1),
    }

    with open(METADATA_SAVE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # ── 10. Final report ──────────────────────────────────────────────────────
    print(f"""
{'=' * 70}
  TRAINING COMPLETE
{'=' * 70}

  Method:       Sub-center ArcFace (k_known={K_KNOWN}, k_wild={K_UNKNOWN})
  Backbone:     MegaDescriptor-T-224 ({emb_dim}-dim embeddings)
  Best val acc: {best_val_acc*100:.2f}%  (epoch {_best_epoch})
  Separability: {sep:.4f}  (V1 ResNet50 was 1.7203 — higher is better)
  Threshold:    {threshold:.4f}  (cosine similarity cutoff for "Unknown")
  Training time: {elapsed_total/60:.1f} min

  Outputs:
    Model:    {MODEL_SAVE}
    Backbone: {BACKBONE_SAVE}
    Gallery:  {GALLERY_JSON}
    Metadata: {METADATA_SAVE}

  Android:
    {V2_BASE / 'ANDROID_EXPORT' / 'embeddings.json'}
    (Replace backbone TFLite via Settings → Update backbone)

  NEXT: Run 5_export_tflite.py adapted for V2 backbone
{'=' * 70}
""")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Train MegaDescriptor-T + Sub-center ArcFace"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            f"Resume training from {RESUME_CKPT.name}. "
            "The script saves a resume checkpoint after every epoch, so you can "
            "safely interrupt at any time (Ctrl+C) and restart with --resume."
        )
    )
    args = parser.parse_args()
    main(resume=args.resume)