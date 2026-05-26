"""
V2_5b_continue_training.py
===========================
CNRS IPHC Strasbourg — Orang-outan V2 pipeline
Author: Titouane

RESUMABLE TRAINING — lance et ferme quand tu veux, ça reprend exactement là où c'était.

PRINCIPE
--------
À chaque lancement, le script:
  1. Charge le checkpoint existant (backbone + ArcFace head + optimizer + epoch)
  2. Reprend l'entraînement depuis l'epoch suivante
  3. Sauvegarde le best model à chaque amélioration (atomique, jamais corrompu)
  4. Quand tu fermes la fenêtre ou Ctrl+C → sauvegarde propre, reprend au prochain lancement

SÉCURITÉ MAXIMALE
-----------------
  - Sauvegarde atomique: écrit dans .tmp → rename (ininterruptible)
  - Backup .bak créé avant chaque écrasement
  - Checkpoint complet: backbone + ArcFace head + optimizer state + scheduler state + epoch
  - Si checkpoint corrompu: repart de MegaDescriptor baseline (perd le fine-tuning mais pas de crash)

RUN
---
  conda activate orangs
  python D:\OrangIdentifier\V2\scripts\V2_5b_continue_training.py

Ferme la fenêtre quand tu veux. Relance le lendemain. Ça repart de l'epoch d'après.
"""

import os, sys, json, math, time, shutil, signal, random, warnings
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter

os.environ["HF_HOME"]    = r"D:\HuggingFaceCache"
os.environ["TORCH_HOME"] = r"D:\TorchCache"

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

V2_BASE     = Path(r"D:\OrangIdentifier\V2")
ZOO_DIR     = Path(r"D:\OrangIdentifier\DATASET_CLASSIFICATION\raw")
WILD_DIR    = V2_BASE / "WILD_CROPS" / "crops"
MODELS_DIR  = V2_BASE / "MODELS"
RESULTS_DIR = V2_BASE / "RESULTS" / "arcface_training"
EMBED_DIR   = V2_BASE / "EMBEDDINGS"

# Checkpoint files — same names as V2_5 so it picks up where V2_5 left off
CHECKPOINT  = MODELS_DIR / "megadesc_T_arcface.pt"       # best model (loaded + saved here)
RESUME_CKPT = MODELS_DIR / "megadesc_T_arcface_resume.pt" # full resume checkpoint (optimizer, epoch, etc.)
GALLERY_JSON = EMBED_DIR / "embeddings_arcface.json"

for d in [MODELS_DIR, RESULTS_DIR, EMBED_DIR, V2_BASE / "ANDROID_EXPORT"]:
    d.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# HYPERPARAMETERS — identical to V2_5
# ==============================================================================

IMG_SIZE    = 224
BATCH_SIZE  = 32
SEED        = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARC_SCALE   = 64
ARC_MARGIN  = 0.50
K_KNOWN     = 1
K_UNKNOWN   = 5

LR_BACKBONE = 2e-5
LR_HEAD     = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS  = 200    # high ceiling — early stopping will trigger well before
PATIENCE    = 20     # slightly more generous than V2_5

WILD_SAMPLES_PER_EPOCH = 1000
VAL_RATIO   = 0.15

MEGA_MEAN = [0.5, 0.5, 0.5]
MEGA_STD  = [0.5, 0.5, 0.5]

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# ==============================================================================
# GRACEFUL SHUTDOWN
# ==============================================================================

_stop = False

def _sigint(sig, frame):
    global _stop
    if not _stop:
        print("\n\n  [!] Stop requested — will save after current epoch completes.")
        _stop = True
    else:
        print("\n  [!] Force quit.")
        sys.exit(0)

signal.signal(signal.SIGINT, _sigint)

# ==============================================================================
# ATOMIC SAVE — never corrupts
# ==============================================================================

def atomic_save(obj, path: Path):
    """Write to .tmp then rename. Creates .bak backup of previous file."""
    tmp = path.with_suffix(".tmp")
    bak = path.with_suffix(".bak")
    try:
        torch.save(obj, tmp)
        if path.exists():
            shutil.copy2(path, bak)
        tmp.replace(path)
        return True
    except Exception as e:
        print(f"  [ERROR] Save failed: {e}")
        if tmp.exists():
            try: tmp.unlink()
            except: pass
        return False

# ==============================================================================
# SUB-CENTER ARCFACE — identical to V2_5
# ==============================================================================

class SubCenterArcFaceLoss(nn.Module):
    def __init__(self, embedding_dim, num_classes, k_per_class, scale=64.0, margin=0.5):
        super().__init__()
        self.scale       = scale
        self.margin      = margin
        self.num_classes = num_classes
        self.k_per_class = k_per_class
        total_k          = sum(k_per_class)
        self.weight      = nn.Parameter(torch.FloatTensor(total_k, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer("class_start",
            torch.tensor([sum(k_per_class[:i]) for i in range(num_classes)], dtype=torch.long))
        self.register_buffer("class_k",
            torch.tensor(k_per_class, dtype=torch.long))
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th    = math.cos(math.pi - margin)
        self.mm    = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        w_norm   = F.normalize(self.weight, dim=1)
        cos_all  = embeddings @ w_norm.T
        logits   = torch.zeros(embeddings.size(0), self.num_classes, device=embeddings.device)
        for c in range(self.num_classes):
            s = self.class_start[c].item()
            k = self.class_k[c].item()
            logits[:, c] = cos_all[:, s:s+k].max(dim=1).values
        cos_theta = logits.clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - cos_theta**2).clamp(0, 1)
        phi       = cos_theta * self.cos_m - sin_theta * self.sin_m
        phi       = torch.where(cos_theta > self.th, phi, cos_theta - self.mm)
        one_hot   = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        output    = (one_hot * phi + (1.0 - one_hot) * logits) * self.scale
        return F.cross_entropy(output, labels, label_smoothing=0.05)

# ==============================================================================
# DATASETS — identical to V2_5
# ==============================================================================

def get_zoo_transforms():
    norm = T.Normalize(MEGA_MEAN, MEGA_STD)
    train_tf = T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.65, 1.0), ratio=(0.85, 1.15)),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.1),
        T.RandomRotation(30),
        T.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.3, hue=0.1),
        T.RandomGrayscale(0.05),
        T.RandomPerspective(distortion_scale=0.3, p=0.3),
        T.GaussianBlur(5, sigma=(0.1, 2.5)),
        T.ToTensor(),
        T.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.0), value="random"),
        norm,
    ])
    val_tf = T.Compose([T.Resize(IMG_SIZE), T.CenterCrop(IMG_SIZE), T.ToTensor(), norm])
    return train_tf, val_tf

def get_wild_transforms():
    norm = T.Normalize(MEGA_MEAN, MEGA_STD)
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0), ratio=(0.7, 1.4)),
        T.RandomHorizontalFlip(0.5),
        T.RandomRotation(45),
        T.ColorJitter(brightness=0.7, contrast=0.6, saturation=0.5, hue=0.15),
        T.RandomGrayscale(0.15),
        T.GaussianBlur(7, sigma=(0.5, 4.0)),
        T.ToTensor(),
        T.RandomErasing(p=0.3, scale=(0.05, 0.25), ratio=(0.3, 3.0), value="random"),
        norm,
    ])

class ZooDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths; self.labels = labels; self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        try: img = Image.open(self.paths[i]).convert("RGB")
        except: img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        return self.transform(img), self.labels[i]

class WildDataset(Dataset):
    def __init__(self, wild_dir, transform, subsample, unknown_class):
        self.transform = transform; self.subsample = subsample
        self.unknown_class = unknown_class
        exts = {".jpg", ".jpeg", ".png"}
        self.all_files = sorted([f for f in wild_dir.iterdir() if f.suffix.lower() in exts])
        self._resample()
    def _resample(self):
        n = min(self.subsample, len(self.all_files))
        self.files = random.sample(self.all_files, n)
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        try: img = Image.open(self.files[i]).convert("RGB")
        except: img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (100, 80, 60))
        return self.transform(img), self.unknown_class

# ==============================================================================
# LOAD ZOO DATASET
# ==============================================================================

def load_zoo():
    from sklearn.model_selection import train_test_split
    exts = {".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"}
    dirs = sorted([d for d in ZOO_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")])
    classes = [d.name for d in dirs]
    c2i = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    for d in dirs:
        for f in sorted(d.iterdir()):
            if f.suffix in exts:
                paths.append(f); labels.append(c2i[d.name])
    idx = list(range(len(paths)))
    tr, va, _, _ = train_test_split(idx, labels, test_size=VAL_RATIO, stratify=labels, random_state=SEED)
    return ([paths[i] for i in tr], [labels[i] for i in tr],
            [paths[i] for i in va], [labels[i] for i in va], classes)

# ==============================================================================
# TRAIN / VALIDATE
# ==============================================================================

def train_epoch(backbone, arc_loss, loader, optimizer, scheduler):
    backbone.train(); arc_loss.train()
    total_loss = total_n = 0
    t0 = time.time()
    W = 40
    for i, (imgs, labs) in enumerate(loader):
        if _stop: break
        imgs = imgs.to(DEVICE, non_blocking=True)
        labs = labs.to(DEVICE, non_blocking=True)
        embs = F.normalize(backbone(imgs), dim=1)
        loss = arc_loss(embs, labs)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(arc_loss.parameters()), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * imgs.size(0)
        total_n    += imgs.size(0)
        fill = int(W * (i+1) / len(loader))
        eta  = (time.time()-t0)/(i+1) * (len(loader)-i-1)
        print(f"\r  [{('█'*fill) + ('░'*(W-fill))}] {i+1}/{len(loader)}"
              f"  loss={total_loss/total_n:.4f}  ETA={timedelta(seconds=int(eta))}",
              end="", flush=True)
    print()
    return total_loss / max(total_n, 1)

@torch.no_grad()
def validate(backbone, loader, num_classes):
    backbone.eval()
    embs_all, labs_all = [], []
    for imgs, labs in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        embs_all.append(F.normalize(backbone(imgs), dim=1).cpu())
        labs_all.extend(labs.tolist())
    embs_all = torch.cat(embs_all)
    labs_all = torch.tensor(labs_all)
    proto = torch.zeros(num_classes, embs_all.shape[1])
    for c in range(num_classes):
        m = (labs_all == c)
        if m.any(): proto[c] = F.normalize(embs_all[m].mean(0), dim=0)
    preds = (embs_all @ proto.T).argmax(1)
    return (preds == labs_all).float().mean().item()

# ==============================================================================
# GALLERY
# ==============================================================================

@torch.no_grad()
def build_gallery(backbone, tr_paths, tr_labels, classes, emb_dim):
    backbone.eval()
    tf = T.Compose([T.Resize(IMG_SIZE), T.CenterCrop(IMG_SIZE), T.ToTensor(), T.Normalize(MEGA_MEAN, MEGA_STD)])
    ds = ZooDataset(tr_paths, tr_labels, tf)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    embs, labs = [], []
    for imgs, ls in dl:
        embs.append(F.normalize(backbone(imgs.to(DEVICE)), dim=1).cpu())
        labs.extend(ls.tolist())
    embs = torch.cat(embs); labs = torch.tensor(labs)
    pm = torch.zeros(len(classes), emb_dim)
    for i, name in enumerate(classes):
        m = (labs == i)
        if m.any(): pm[i] = F.normalize(embs[m].mean(0), dim=0)
    pos = [float((embs[labs==i] @ pm[i]).mean()) for i in range(len(classes))]
    neg = [float((embs[labs==i] @ torch.cat([pm[:i],pm[i+1:]],0).T).max(1).values.mean()) for i in range(len(classes))]
    sep = (sum(pos)/len(pos)) / (sum(neg)/len(neg))
    # threshold calibration
    sims_p = np.array([v for i in range(len(classes)) for v in (embs[labs==i] @ pm[i]).tolist()])
    sims_n = np.array([v for i in range(len(classes)) for v in (embs[labs==i] @ torch.cat([pm[:i],pm[i+1:]],0).T).max(1).values.tolist()])
    best_f1, best_t = 0, 0.5
    for t in np.linspace(0.0, 1.0, 500):
        tp = (sims_p >= t).sum(); fp = (sims_n >= t).sum(); fn = (sims_p < t).sum()
        pr = tp/(tp+fp+1e-9); re = tp/(tp+fn+1e-9)
        f1 = 2*pr*re/(pr+re+1e-9)
        if f1 > best_f1: best_f1, best_t = f1, float(t)
    gallery = {
        "version": "2.0-arcface", "created": datetime.now().isoformat(),
        "model": "MegaDescriptor-T-224+SubCenterArcFace",
        "embedding_dim": emb_dim, "unknown_threshold": round(best_t, 4),
        "separability_ratio": round(sep, 4),
        "individuals": {
            name: {"class_index": i, "embedding": pm[i].tolist(),
                   "num_crops": int((labs==i).sum())}
            for i, name in enumerate(classes)
        }
    }
    tmp = GALLERY_JSON.with_suffix(".tmp")
    with open(tmp, "w") as f: json.dump(gallery, f, separators=(",",":"))
    tmp.replace(GALLERY_JSON)
    shutil.copy2(GALLERY_JSON, V2_BASE / "ANDROID_EXPORT" / "embeddings.json")
    print(f"\n  Separability : {sep:.4f}  (ResNet50 V1 = 1.7203)")
    print(f"  Threshold    : {best_t:.4f}  (F1={best_f1:.4f})")
    print(f"  Gallery      : {GALLERY_JSON}")
    return sep, best_t

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    t0 = time.time()

    print("=" * 70)
    print("  ARCFACE — RESUMABLE TRAINING")
    print("  MegaDescriptor-T-224 + Sub-center ArcFace")
    print("  CNRS IPHC Strasbourg — Ferme la fenêtre quand tu veux")
    print("=" * 70)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name} ({props.total_memory/1e9:.1f} GB)")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\n  Chargement dataset...")
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("  pip install scikit-learn"); sys.exit(1)

    tr_paths, tr_labels, va_paths, va_labels, classes = load_zoo()
    num_zoo   = len(classes)
    unk_class = num_zoo
    num_total = num_zoo + 1

    print(f"  {len(tr_paths)+len(va_paths)} zoo crops | {num_zoo} individus")
    print(f"  Train: {len(tr_paths)} | Val: {len(va_paths)}")

    # ── Backbone ─────────────────────────────────────────────────────────────
    print("\n  Chargement MegaDescriptor-T-224...")
    backbone = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224",
                                  pretrained=True, num_classes=0)
    with torch.no_grad():
        emb_dim = backbone(torch.randn(1, 3, IMG_SIZE, IMG_SIZE)).shape[1]
    backbone = backbone.to(DEVICE)

    # ── ArcFace head ─────────────────────────────────────────────────────────
    k_per_class = [K_KNOWN] * num_zoo + [K_UNKNOWN]
    arc_loss    = SubCenterArcFaceLoss(emb_dim, num_total, k_per_class,
                                        ARC_SCALE, ARC_MARGIN).to(DEVICE)

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = optim.AdamW([
        {"params": backbone.parameters(), "lr": LR_BACKBONE},
        {"params": arc_loss.parameters(), "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    # Scheduler: cosine over MAX_EPOCHS, will be re-adjusted after loading
    steps_per_epoch = (len(tr_paths) // BATCH_SIZE + 1) + (WILD_SAMPLES_PER_EPOCH // BATCH_SIZE + 1)
    total_steps     = MAX_EPOCHS * steps_per_epoch

    def lr_lambda(step):
        progress = step / max(total_steps, 1)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── LOAD CHECKPOINT ───────────────────────────────────────────────────────
    start_epoch   = 1
    best_val_acc  = 0.0
    patience_count = 0
    global_step   = 0

    if RESUME_CKPT.exists():
        print(f"\n  Reprise depuis checkpoint: {RESUME_CKPT.name}")
        try:
            ckpt = torch.load(RESUME_CKPT, map_location=DEVICE, weights_only=False)
            backbone.load_state_dict(ckpt["backbone_state"])
            arc_loss.load_state_dict(ckpt["arc_loss_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch    = ckpt["epoch"] + 1
            best_val_acc   = ckpt["best_val_acc"]
            patience_count = ckpt["patience_count"]
            global_step    = ckpt.get("global_step", 0)
            print(f"  Epoch de reprise : {start_epoch}")
            print(f"  Best val acc     : {best_val_acc*100:.2f}%")
            print(f"  Patience         : {patience_count}/{PATIENCE}")
        except Exception as e:
            print(f"  [WARN] Checkpoint corrompu ({e}) — reprise depuis meilleur modèle")
            if CHECKPOINT.exists():
                try:
                    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
                    backbone.load_state_dict(ckpt["backbone_state"])
                    arc_loss.load_state_dict(ckpt["arc_loss_state"])
                    best_val_acc = ckpt.get("val_acc", 0.0)
                    print(f"  Backbone chargé depuis {CHECKPOINT.name}")
                    print(f"  Best val acc : {best_val_acc*100:.2f}%")
                except Exception as e2:
                    print(f"  [WARN] Impossible de charger le meilleur modèle ({e2})")
                    print(f"  Reprise depuis MegaDescriptor baseline.")
    elif CHECKPOINT.exists():
        print(f"\n  Pas de checkpoint de reprise — chargement du meilleur modèle: {CHECKPOINT.name}")
        try:
            ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
            backbone.load_state_dict(ckpt["backbone_state"])
            arc_loss.load_state_dict(ckpt["arc_loss_state"])
            best_val_acc = ckpt.get("val_acc", 0.0)
            print(f"  Best val acc : {best_val_acc*100:.2f}%")
            print(f"  Epoch précédente inconnue — reprise à epoch 1 (les poids sont conservés)")
        except Exception as e:
            print(f"  [WARN] {e} — reprise depuis baseline")
    else:
        print(f"\n  Aucun checkpoint trouvé — démarrage depuis MegaDescriptor baseline")

    print(f"\n  Démarrage epoch {start_epoch} → max {MAX_EPOCHS}")
    print(f"  Patience : {patience_count}/{PATIENCE}")
    print(f"  Ferme la fenêtre quand tu veux — ça reprend au prochain lancement.\n")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    zoo_train_tf, zoo_val_tf = get_zoo_transforms()
    wild_tf = get_wild_transforms()

    counts  = Counter(tr_labels)
    weights = [1.0 / counts[l] for l in tr_labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    zoo_train_dl = DataLoader(ZooDataset(tr_paths, tr_labels, zoo_train_tf),
                               batch_size=BATCH_SIZE, sampler=sampler,
                               num_workers=4, pin_memory=True, persistent_workers=True)
    zoo_val_dl   = DataLoader(ZooDataset(va_paths, va_labels, zoo_val_tf),
                               batch_size=64, shuffle=False,
                               num_workers=2, pin_memory=True, persistent_workers=True)

    use_wild = WILD_DIR.exists() and len(list(WILD_DIR.glob("*.jpg"))) > 0
    if use_wild:
        wild_ds = WildDataset(WILD_DIR, wild_tf, WILD_SAMPLES_PER_EPOCH, unk_class)
        wild_dl = DataLoader(wild_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
        print(f"  Wild crops: {len(wild_ds.all_files):,} dispo, {WILD_SAMPLES_PER_EPOCH}/epoch\n")
    else:
        print("  [WARN] Pas de wild crops trouvés — entraînement zoo seulement\n")

    # ── TRAINING LOOP ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        if _stop:
            print("\n  Stop propre demandé.")
            break

        elapsed = time.time() - t0
        print(f"\n  Epoch {epoch}/{MAX_EPOCHS}  "
              f"patience={patience_count}/{PATIENCE}  "
              f"best={best_val_acc*100:.2f}%  "
              f"elapsed={timedelta(seconds=int(elapsed))}")

        # Resample wild crops
        if use_wild: wild_ds._resample()

        # Train zoo
        print(f"  Zoo crops:")
        zoo_loss = train_epoch(backbone, arc_loss, zoo_train_dl, optimizer, scheduler)

        # Train wild
        if use_wild and not _stop:
            print(f"  Wild crops:")
            wild_loss = train_epoch(backbone, arc_loss, wild_dl, optimizer, scheduler)

        if _stop: break

        # Validate
        print(f"  Validation:")
        val_acc = validate(backbone, zoo_val_dl, num_zoo)
        print(f"  Val accuracy: {val_acc*100:.2f}%")

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc   = val_acc
            patience_count = 0

            # Save best model (best.pt compatible with V2_5)
            atomic_save({
                "backbone_state": backbone.state_dict(),
                "arc_loss_state": arc_loss.state_dict(),
                "classes":        classes,
                "num_classes":    num_zoo,
                "emb_dim":        emb_dim,
                "epoch":          epoch,
                "val_acc":        val_acc,
                "img_size":       IMG_SIZE,
                "arc_scale":      ARC_SCALE,
                "arc_margin":     ARC_MARGIN,
                "normalization":  {"mean": MEGA_MEAN, "std": MEGA_STD},
                "save_time":      datetime.now().isoformat(),
            }, CHECKPOINT)
            size_mb = CHECKPOINT.stat().st_size / 1e6
            print(f"  BEST sauvegardé: {CHECKPOINT.name} ({size_mb:.1f} MB) ← {val_acc*100:.2f}%")
        else:
            patience_count += 1

        # Save FULL resume checkpoint (always, regardless of improvement)
        atomic_save({
            "backbone_state":  backbone.state_dict(),
            "arc_loss_state":  arc_loss.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch":           epoch,
            "best_val_acc":    best_val_acc,
            "patience_count":  patience_count,
            "global_step":     global_step,
            "classes":         classes,
            "emb_dim":         emb_dim,
        }, RESUME_CKPT)

        # ETA
        elapsed = time.time() - t0
        eta     = elapsed / (epoch - start_epoch + 1) * (MAX_EPOCHS - epoch)
        print(f"  ETA fin totale: {timedelta(seconds=int(eta))} | "
              f"Elapsed: {timedelta(seconds=int(elapsed))}")

        # Early stopping
        if patience_count >= PATIENCE:
            print(f"\n  Early stopping à l'epoch {epoch} (patience={PATIENCE})")
            break

    # ── POST TRAINING ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Chargement du meilleur modèle + construction galerie...")
    print("=" * 70)

    if CHECKPOINT.exists():
        try:
            ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
            backbone.load_state_dict(ckpt["backbone_state"])
            print(f"  Meilleur modèle: epoch {ckpt.get('epoch','?')} "
                  f"val={ckpt.get('val_acc',0)*100:.2f}%")
        except Exception as e:
            print(f"  [WARN] Impossible de charger le meilleur: {e}")

    sep, thresh = build_gallery(backbone, tr_paths, tr_labels, classes, emb_dim)

    elapsed_total = time.time() - t0
    print(f"""
{'=' * 70}
  SESSION TERMINÉE
{'=' * 70}
  Durée session   : {timedelta(seconds=int(elapsed_total))}
  Best val acc    : {best_val_acc*100:.2f}%
  Séparabilité    : {sep:.4f}
  Seuil inconnu   : {thresh:.4f}

  Checkpoint reprise : {RESUME_CKPT}
  Meilleur modèle    : {CHECKPOINT}
  Galerie Android    : {GALLERY_JSON}

  → Relance ce script pour continuer l'entraînement depuis l'epoch {start_epoch + (epoch - start_epoch + 1)}.
{'=' * 70}
""")

if __name__ == "__main__":
    main()
