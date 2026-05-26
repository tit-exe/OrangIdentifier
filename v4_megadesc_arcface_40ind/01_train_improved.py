# V4_train_improved.py
# CNRS IPHC Strasbourg - Orang-outan V4
# Author: Titouane
#
# WHAT THIS FIXES (from stress test results):
#   - Blur motion/focus : rupture at moderate -> add strong blur augmentation
#   - Low resolution    : rupture at moderate -> add low-res simulation
#   - Combined          : rupture at moderate -> consequence of above two
#
# WHAT'S NEW VS V3:
#   - 40 supervised classes (10 zoo + 30 BOS) instead of 10
#   - Low-res simulation in augmentation (resize 14-45% then back to 224)
#   - Stronger blur (sigma up to 6.0 vs 2.5 in V3)
#   - Lower LR (fine-tuning of fine-tuned model)
#
# NEVER OVERWRITES V3:
#   All outputs -> D:\OrangIdentifier\V2\V4_improved\
#
# RESUMABLE:
#   Close terminal anytime. Relaunch -> continues from last epoch.
#
# RUN:
#   conda activate orangs
#   python D:\OrangIdentifier\V2\scripts\V4_train_improved.py

import os, sys, json, math, time, shutil, signal, random, warnings, io
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict

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
from PIL import Image, ImageFile, ImageFilter, ImageEnhance
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ==============================================================================
# PATHS
# ==============================================================================

V3_MODEL    = Path(r"D:\OrangIdentifier\V2\MODELS\megadesc_T_arcface.pt")
ZOO_DIR     = Path(r"D:\OrangIdentifier\DATASET_CLASSIFICATION\raw")
BOS_DIR     = Path(r"D:\OrangIdentifier\V2\NEW_ORANGS_CROPS")
WILD_DIR    = Path(r"D:\OrangIdentifier\V2\WILD_CROPS\crops")
OUT_DIR     = Path(r"D:\OrangIdentifier\V2\V4_improved")

MODELS_DIR  = OUT_DIR / "models"
EMBED_DIR   = OUT_DIR / "embeddings"
RESULTS_DIR = OUT_DIR / "results"
LOGS_DIR    = OUT_DIR / "logs"

BEST_MODEL  = MODELS_DIR / "v4_best.pt"
RESUME_CKP  = MODELS_DIR / "v4_resume.pt"
STATE_FILE  = OUT_DIR / "state.json"
LOG_FILE    = LOGS_DIR / "training.log"
GALLERY_OUT = EMBED_DIR / "embeddings_v4.json"

for d in [MODELS_DIR, EMBED_DIR, RESULTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================

IMG_SIZE             = 224
BATCH_SIZE           = 32
SEED                 = 42
DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEGA_MEAN            = [0.5, 0.5, 0.5]
MEGA_STD             = [0.5, 0.5, 0.5]
ARC_SCALE            = 64
ARC_MARGIN           = 0.50
K_KNOWN              = 1
K_UNKNOWN            = 5
LR_BACKBONE          = 1e-5
LR_HEAD              = 5e-4
WEIGHT_DECAY         = 1e-4
MAX_EPOCHS           = 100
PATIENCE             = 25
PATIENCE_START       = 35
WARMUP_EPOCHS        = 5
WILD_PER_EPOCH       = 1200
VAL_RATIO_ZOO        = 0.15
THRESHOLD            = 0.22

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==============================================================================
# LOGGER
# ==============================================================================

_log_fh = open(LOG_FILE, "a", encoding="utf-8", buffering=1)

def log(msg, level="INFO"):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}][{level}] {msg}"
    print(line)
    _log_fh.write(line + "\n")

def title(t):
    sep = "=" * 70
    log(""); log(sep); log(f"  {t}"); log(sep)

def section(t):
    log(f"\n  --- {t} ---")

# ==============================================================================
# INTERRUPT HANDLER
# ==============================================================================

_interrupt = False

def _sigint(sig, frame):
    global _interrupt
    if not _interrupt:
        log("\n  [Ctrl+C] Caught. Will stop after this epoch and save.", "WARN")
        _interrupt = True
    else:
        log("\n  [Ctrl+C] Force quit.", "WARN")
        _log_fh.close()
        sys.exit(1)

signal.signal(signal.SIGINT, _sigint)

# ==============================================================================
# AUGMENTATIONS  <-- KEY IMPROVEMENT OVER V3
# ==============================================================================

class LowResSimulation:
    """Simulates a photo taken from far away: resize down then back up."""
    def __init__(self, min_frac=0.14, max_frac=0.45, p=0.35):
        self.min_frac = min_frac
        self.max_frac = max_frac
        self.p        = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h  = img.size
        frac  = random.uniform(self.min_frac, self.max_frac)
        sw    = max(int(w * frac), 12)
        sh    = max(int(h * frac), 12)
        return img.resize((sw, sh), Image.BILINEAR).resize((w, h), Image.BILINEAR)

class JPEGDegradation:
    """Simulates WhatsApp/phone compression."""
    def __init__(self, min_q=15, max_q=60, p=0.25):
        self.min_q = min_q
        self.max_q = max_q
        self.p     = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        q   = random.randint(self.min_q, self.max_q)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).copy()

norm_tf = T.Normalize(MEGA_MEAN, MEGA_STD)

def get_train_tf_zoo():
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.60, 1.0), ratio=(0.80, 1.20)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.08),
        T.RandomRotation(degrees=30),
        T.ColorJitter(brightness=0.5, contrast=0.45, saturation=0.35, hue=0.12),
        T.RandomGrayscale(p=0.05),
        # KEY FIX 1 : low-res simulation
        T.Lambda(LowResSimulation(min_frac=0.14, max_frac=0.45, p=0.35)),
        # KEY FIX 2 : strong blur
        T.RandomApply([T.GaussianBlur(kernel_size=13, sigma=(1.0, 6.0))], p=0.40),
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        T.Lambda(JPEGDegradation(min_q=15, max_q=60, p=0.25)),
        T.ToTensor(),
        T.RandomErasing(p=0.20, scale=(0.02, 0.15), ratio=(0.3, 3.0), value="random"),
        norm_tf,
    ])

def get_train_tf_bos():
    # BOS photos already have field conditions - augment similarly but a bit less
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.60, 1.0), ratio=(0.80, 1.20)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.08),
        T.RandomRotation(degrees=35),
        T.ColorJitter(brightness=0.55, contrast=0.50, saturation=0.40, hue=0.12),
        T.RandomGrayscale(p=0.08),
        T.Lambda(LowResSimulation(min_frac=0.14, max_frac=0.45, p=0.35)),
        T.RandomApply([T.GaussianBlur(kernel_size=13, sigma=(1.0, 6.0))], p=0.40),
        T.Lambda(JPEGDegradation(min_q=10, max_q=55, p=0.30)),
        T.ToTensor(),
        T.RandomErasing(p=0.25, scale=(0.02, 0.20), ratio=(0.3, 3.0), value="random"),
        norm_tf,
    ])

def get_train_tf_wild():
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.50, 1.0), ratio=(0.70, 1.40)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=45),
        T.ColorJitter(brightness=0.70, contrast=0.60, saturation=0.50, hue=0.15),
        T.RandomGrayscale(p=0.15),
        T.Lambda(LowResSimulation(min_frac=0.14, max_frac=0.50, p=0.40)),
        T.RandomApply([T.GaussianBlur(kernel_size=15, sigma=(1.0, 7.0))], p=0.45),
        T.Lambda(JPEGDegradation(min_q=8, max_q=50, p=0.35)),
        T.ToTensor(),
        T.RandomErasing(p=0.30, scale=(0.05, 0.25), ratio=(0.3, 3.0), value="random"),
        norm_tf,
    ])

val_tf = T.Compose([
    T.Resize(IMG_SIZE),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    norm_tf,
])

# ==============================================================================
# DATASETS
# ==============================================================================

exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

class CropDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 100, 80))
        return self.transform(img), self.labels[idx]

class MultiTfDataset(Dataset):
    """Dataset where each sample can have its own transform (zoo vs BOS)."""
    def __init__(self, paths, labels, transforms_list):
        self.paths  = paths
        self.labels = labels
        self.tfs    = transforms_list

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 100, 80))
        return self.tfs[idx](img), self.labels[idx]

class WildDataset(Dataset):
    def __init__(self, wild_dir, transform, subsample, label):
        self.transform = transform
        self.subsample = subsample
        self.label     = label
        all_files      = sorted([f for f in wild_dir.iterdir()
                                  if f.suffix.lower() in {".jpg",".jpeg",".png"}])
        self.all_files = all_files
        log(f"    Wild crops available: {len(all_files)}")
        self._resample()

    def _resample(self):
        n = min(self.subsample, len(self.all_files))
        self.files = random.sample(self.all_files, n)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.files[idx]).convert("RGB")
        except:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (100, 80, 60))
        return self.transform(img), self.label

# ==============================================================================
# SUB-CENTER ARCFACE  (same as V3)
# ==============================================================================

class SubCenterArcFace(nn.Module):
    def __init__(self, emb_dim, num_classes, k_per_class, scale=64.0, margin=0.5):
        super().__init__()
        self.scale   = scale
        self.margin  = margin
        self.n_cls   = num_classes
        total_k      = sum(k_per_class)
        self.weight  = nn.Parameter(torch.FloatTensor(total_k, emb_dim))
        nn.init.xavier_uniform_(self.weight)
        starts = [sum(k_per_class[:i]) for i in range(num_classes)]
        self.register_buffer("c_start", torch.tensor(starts, dtype=torch.long))
        self.register_buffer("c_k",     torch.tensor(k_per_class, dtype=torch.long))
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th    = math.cos(math.pi - margin)
        self.mm    = math.sin(math.pi - margin) * margin

    def forward(self, emb, labels):
        wn  = F.normalize(self.weight, dim=1)
        ca  = emb @ wn.T
        lg  = torch.zeros(emb.size(0), self.n_cls, device=emb.device)
        for c in range(self.n_cls):
            s = self.c_start[c].item()
            k = self.c_k[c].item()
            lg[:, c] = ca[:, s:s+k].max(dim=1).values
        ct  = lg.clamp(-1, 1)
        st  = (1.0 - ct**2).clamp(0, 1).sqrt()
        phi = ct * self.cos_m - st * self.sin_m
        phi = torch.where(ct > self.th, phi, ct - self.mm)
        oh  = torch.zeros_like(lg).scatter_(1, labels.unsqueeze(1), 1.0)
        out = oh * phi + (1.0 - oh) * lg
        return F.cross_entropy(out * self.scale, labels, label_smoothing=0.05)

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_zoo(zoo_dir, val_ratio):
    dirs    = sorted([d for d in zoo_dir.iterdir() if d.is_dir()
                      and not d.name.startswith("_")])
    classes = [d.name for d in dirs]
    c2i     = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    for d in dirs:
        imgs = sorted([f for f in d.iterdir() if f.suffix in exts])
        for p in imgs:
            paths.append(p)
            labels.append(c2i[d.name])
    log(f"    Zoo: {len(classes)} classes, {len(paths)} crops")

    from sklearn.model_selection import train_test_split
    idx = list(range(len(paths)))
    tr_idx, va_idx, _, _ = train_test_split(idx, labels,
                                             test_size=val_ratio,
                                             stratify=labels, random_state=SEED)
    tr_p  = [paths[i] for i in tr_idx]
    tr_l  = [labels[i] for i in tr_idx]
    va_p  = [paths[i] for i in va_idx]
    va_l  = [labels[i] for i in va_idx]
    return tr_p, tr_l, va_p, va_l, classes

def load_bos(bos_dir, offset):
    """Load BOS individuals. offset = first class index (= len(zoo_classes))."""
    indivs = sorted([d.name for d in bos_dir.iterdir() if d.is_dir()])
    tr_p, tr_l, va_p, va_l, names = [], [], [], [], []
    for j, name in enumerate(indivs):
        d    = bos_dir / name
        imgs = sorted([f for f in d.iterdir() if f.suffix.lower() in exts])
        if not imgs:
            continue
        n_val = max(3, min(8, len(imgs) // 5))
        random.shuffle(imgs)
        val_imgs   = imgs[:n_val]
        train_imgs = imgs[n_val:]
        label      = offset + j
        tr_p.extend(train_imgs);  tr_l.extend([label] * len(train_imgs))
        va_p.extend(val_imgs);    va_l.extend([label] * len(val_imgs))
        names.append(name)
        log(f"      BOS {name:<14}: train={len(train_imgs)}, val={len(val_imgs)}")
    log(f"    BOS: {len(names)} individuals, {len(tr_p)} train, {len(va_p)} val crops")
    return tr_p, tr_l, va_p, va_l, names

# ==============================================================================
# VALIDATION (nearest prototype)
# ==============================================================================

@torch.no_grad()
def validate(backbone, val_paths, val_labels, all_classes, proto_classes_range):
    """
    Compute nearest-prototype accuracy.
    proto_classes_range : range of class indices to include in prototypes.
    """
    backbone.eval()
    embs, labs = [], []
    for i in range(0, len(val_paths), 64):
        batch = []
        for p in val_paths[i:i+64]:
            try:   batch.append(val_tf(Image.open(p).convert("RGB")))
            except: batch.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))
        t  = torch.stack(batch).to(DEVICE)
        e  = F.normalize(backbone(t), dim=1)
        embs.append(e.cpu())
        labs.extend(val_labels[i:i+64])
    embs = torch.cat(embs)
    labs = torch.tensor(labs)
    # Build prototypes for each class in range
    idx_list = sorted(set(proto_classes_range) & set(labs.tolist()))
    if not idx_list:
        return 0.0
    proto = torch.zeros(max(idx_list)+1, embs.shape[1])
    for c in idx_list:
        mask = (labs == c)
        if mask.any():
            proto[c] = F.normalize(embs[mask].mean(0), dim=0)
    # Nearest prototype for each sample
    sims  = embs @ proto[idx_list].T
    preds = torch.tensor(idx_list)[sims.argmax(dim=1)]
    return (preds == labs).float().mean().item()

# ==============================================================================
# GALLERY BUILDING
# ==============================================================================

@torch.no_grad()
def build_gallery(backbone, all_tr_paths, all_tr_labels, all_classes, emb_dim):
    backbone.eval()
    embs_by_class = defaultdict(list)
    for i in range(0, len(all_tr_paths), 64):
        batch = []
        for p in all_tr_paths[i:i+64]:
            try:   batch.append(val_tf(Image.open(p).convert("RGB")))
            except: batch.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))
        t   = torch.stack(batch).to(DEVICE)
        e   = F.normalize(backbone(t), dim=1).cpu()
        for k, (emb, lbl) in enumerate(zip(e, all_tr_labels[i:i+64])):
            embs_by_class[lbl].append(emb)
        if (i // 64) % 10 == 0:
            print(f"\r    Gallery: {min(i+64, len(all_tr_paths))}/{len(all_tr_paths)}", end="", flush=True)
    print()

    proto_matrix = torch.zeros(len(all_classes), emb_dim)
    pos_sims, neg_sims = [], []
    gallery = {}
    for i, name in enumerate(all_classes):
        if i not in embs_by_class: continue
        stacked = torch.stack(embs_by_class[i])
        p       = F.normalize(stacked.mean(0), dim=0)
        proto_matrix[i] = p
        own_sim = (stacked @ p).tolist()
        pos_sims.extend(own_sim)
        gallery[name] = {
            "class_index":       i,
            "num_training_crops": len(embs_by_class[i]),
            "mean_within_sim":   round(float(np.mean(own_sim)), 4),
            "embedding":         p.tolist(),
        }
        log(f"    {name:<16}: {len(embs_by_class[i]):4d} crops, within_sim={np.mean(own_sim):.4f}")

    # Neg sims
    for i, name in enumerate(all_classes):
        if i not in embs_by_class: continue
        stacked = torch.stack(embs_by_class[i])
        others  = [j for j in range(len(all_classes)) if j != i and j in embs_by_class]
        if others:
            other_protos = proto_matrix[others]
            best_neg = (stacked @ other_protos.T).max(dim=1).values.tolist()
            neg_sims.extend(best_neg)

    sep_gap = float(np.mean(pos_sims)) - float(np.mean(neg_sims)) if neg_sims else 0

    # Threshold calibration (leave-one-out on gallery)
    thresholds = np.linspace(0.0, 1.0, 500)
    f1_scores  = []
    ps = np.array(pos_sims)
    ns = np.array(neg_sims) if neg_sims else np.array([0.0])
    for t in thresholds:
        tp = (ps >= t).sum(); fp = (ns >= t).sum(); fn = (ps < t).sum()
        pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
        f1_scores.append(2 * pr * rc / (pr + rc + 1e-9))
    opt_t  = float(thresholds[int(np.argmax(f1_scores))])
    best_f1 = float(max(f1_scores))

    out = {
        "version":           "4.0-arcface-improved",
        "created":           datetime.now().isoformat(),
        "model":             "MegaDescriptor-T-224 + SubCenterArcFace V4",
        "embedding_dim":     emb_dim,
        "normalization":     {"mean": MEGA_MEAN, "std": MEGA_STD},
        "similarity_metric": "cosine",
        "unknown_threshold": round(opt_t, 4),
        "calibration_f1":    round(best_f1, 4),
        "separability_gap":  round(sep_gap, 4),
        "pos_sim_mean":      round(float(np.mean(pos_sims)), 4),
        "neg_sim_mean":      round(float(np.mean(neg_sims)), 4) if neg_sims else None,
        "num_individuals":   len(gallery),
        "individuals":       gallery,
    }
    tmp = GALLERY_OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    tmp.replace(GALLERY_OUT)
    log(f"    Gallery saved: {GALLERY_OUT} ({GALLERY_OUT.stat().st_size/1024:.1f} KB)")
    log(f"    Separability gap  : {sep_gap:.4f}")
    log(f"    Calibrated threshold: {opt_t:.4f} (F1={best_f1:.4f})")
    return opt_t, sep_gap, float(np.mean(pos_sims)), float(np.mean(neg_sims)) if neg_sims else 0

# ==============================================================================
# STRESS TEST COMPARISON (V3 vs V4)
# ==============================================================================

@torch.no_grad()
def stress_compare(v3_backbone, v4_backbone, zoo_dir, v3_classes, v4_classes):
    """Quick stress test on zoo crops comparing V3 vs V4. Returns results dict."""
    import io as _io
    from PIL import ImageFilter as IF

    N_CROPS = 6  # per individual

    def degrade(img, kind, sev):
        if kind == "blur":
            r = [0, 1.5, 3.0, 5.0, 8.0][sev]
            return img.filter(IF.GaussianBlur(radius=r))
        elif kind == "lowres":
            f = [1.0, 0.50, 0.25, 0.15, 0.08][sev]
            w, h = img.size
            sw = max(int(w*f), 8); sh = max(int(h*f), 8)
            return img.resize((sw,sh), Image.BILINEAR).resize((w,h), Image.BILINEAR)
        elif kind == "jpeg":
            q = [95, 50, 25, 10, 3][sev]
            buf = _io.BytesIO()
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            return Image.open(buf).copy()
        elif kind == "combined":
            if sev == 0: return img
            img = degrade(img, "blur", min(sev, 3))
            img = degrade(img, "jpeg", min(sev, 3))
            return img
        return img

    deg_types = ["blur", "lowres", "jpeg", "combined"]
    results   = {"V3": {}, "V4": {}}

    for model_name, backbone, classes in [("V3", v3_backbone, v3_classes),
                                           ("V4", v4_backbone, v4_classes)]:
        backbone.eval()
        # Build zoo prototypes
        zoo_classes_only = [c for c in classes
                            if not any(c == b for b in ["background"])][:10]
        proto = {}
        for cls in zoo_classes_only:
            d    = zoo_dir / cls
            if not d.exists(): continue
            imgs = sorted([f for f in d.iterdir() if f.suffix in exts])[:50]
            embs = []
            for p in imgs:
                try:
                    t   = val_tf(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE)
                    embs.append(F.normalize(backbone(t), dim=1).cpu())
                except: pass
            if embs:
                proto[cls] = F.normalize(torch.cat(embs).mean(0), dim=0)

        if not proto: continue
        pnames = list(proto.keys())
        pm     = torch.stack([proto[c] for c in pnames])

        for dtype in deg_types:
            results[model_name][dtype] = {}
            for sev in range(5):
                accs = []
                for cls in pnames:
                    d    = zoo_dir / cls
                    imgs = sorted([f for f in d.iterdir() if f.suffix in exts])
                    samp = random.sample(imgs, min(N_CROPS, len(imgs)))
                    for p in samp:
                        try:
                            img = Image.open(p).convert("RGB")
                            dg  = degrade(img, dtype, sev)
                            t   = val_tf(dg).unsqueeze(0).to(DEVICE)
                            e   = F.normalize(backbone(t), dim=1).cpu()
                            pred = pnames[(e @ pm.T).argmax().item()]
                            accs.append(pred == cls)
                        except: pass
                results[model_name][dtype][sev] = float(np.mean(accs)) if accs else 0.0
    return results, deg_types

# ==============================================================================
# PLOTTING
# ==============================================================================

def make_plots(history, stress_results, deg_types, all_classes, gallery_data, pos_sims_v4, neg_sims_v4):
    fig = plt.figure(figsize=(20, 24))
    fig.patch.set_facecolor("#0f0f0f")
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    colors = {"V3": "#4488ff", "V4": "#00ff88"}
    sev_labels = ["Aucune", "Legere", "Moderee", "Forte", "Extreme"]

    # 1. Training curves
    ax = fig.add_subplot(gs[0, :])
    ax.set_facecolor("#111")
    epochs = list(range(1, len(history["zoo_val_acc"]) + 1))
    ax.plot(epochs, [v*100 for v in history["zoo_val_acc"]],
            color="#00ff88", linewidth=2, label="Zoo val accuracy (%)")
    if history.get("bos_val_acc"):
        ax.plot(epochs, [v*100 for v in history["bos_val_acc"]],
                color="#4488ff", linewidth=2, label="BOS val accuracy (%)")
    ax2 = ax.twinx()
    ax2.plot(epochs, history["train_loss"], color="#ffaa00",
             linewidth=1.5, alpha=0.7, label="Train loss")
    ax2.set_ylabel("Loss", color="#ffaa00", fontsize=9)
    ax2.tick_params(colors="#ffaa00")
    ax.set_title("Courbes d'entrainement V4", color="white", fontsize=12)
    ax.set_xlabel("Epoch", color="gray"); ax.set_ylabel("Accuracy (%)", color="gray")
    ax.tick_params(colors="gray"); ax.spines[:].set_color("#333")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.3, labelcolor="white", facecolor="#111")
    best_ep = int(np.argmax(history["zoo_val_acc"])) + 1
    ax.axvline(best_ep, color="white", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(best_ep, ax.get_ylim()[0], f" best={best_ep}", color="white", fontsize=8)

    # 2-5. Stress comparison by degradation type
    for di, dtype in enumerate(deg_types[:4]):
        row = 1 + di // 2
        col = di % 2
        ax  = fig.add_subplot(gs[row, col])
        ax.set_facecolor("#111")
        x = np.arange(5)
        for mname, color in colors.items():
            if mname in stress_results and dtype in stress_results[mname]:
                ys = [stress_results[mname][dtype][s]*100 for s in range(5)]
                ax.plot(x, ys, color=color, marker="o", linewidth=2,
                        markersize=6, label=mname)
        ax.axhline(80, color="white", linestyle="--", alpha=0.4, linewidth=1)
        ax.set_xticks(x); ax.set_xticklabels(sev_labels, rotation=20, fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_title(f"Stress: {dtype}", color="white", fontsize=10)
        ax.set_ylabel("Accuracy (%)", color="gray")
        ax.tick_params(colors="gray"); ax.spines[:].set_color("#333")
        ax.legend(fontsize=8, framealpha=0.3, labelcolor="white", facecolor="#111")

    # 6. Similarity distributions
    ax = fig.add_subplot(gs[3, 0])
    ax.set_facecolor("#111")
    bins = np.linspace(-0.2, 1.05, 60)
    ax.hist(pos_sims_v4, bins=bins, alpha=0.7, color="#00ff88",
            label=f"Positif (intra-indiv) n={len(pos_sims_v4)}", density=True)
    ax.hist(neg_sims_v4, bins=bins, alpha=0.7, color="#ff4444",
            label=f"Negatif (inter-indiv) n={len(neg_sims_v4)}", density=True)
    ax.axvline(gallery_data["unknown_threshold"], color="white",
               linestyle="--", linewidth=1.5, label=f"Seuil={gallery_data['unknown_threshold']}")
    ax.set_title("Distributions similarite V4", color="white", fontsize=10)
    ax.tick_params(colors="gray"); ax.spines[:].set_color("#333")
    ax.legend(fontsize=7, framealpha=0.3, labelcolor="white", facecolor="#111")

    # 7. BOS within-individual coherence
    ax = fig.add_subplot(gs[3, 1])
    ax.set_facecolor("#111")
    if "individuals" in gallery_data:
        indiv_names  = list(gallery_data["individuals"].keys())
        within_sims  = [gallery_data["individuals"][n]["mean_within_sim"]
                        for n in indiv_names]
        sorted_pairs = sorted(zip(within_sims, indiv_names))
        xs = [p[0] for p in sorted_pairs]
        ys = [p[1] for p in sorted_pairs]
        bar_colors = ["#ff4444" if v < 0.85 else "#00ff88" for v in xs]
        bars = ax.barh(range(len(ys)), xs, color=bar_colors, alpha=0.8, edgecolor="none")
        ax.set_yticks(range(len(ys)))
        ax.set_yticklabels(ys, fontsize=6, color="gray")
        ax.set_xlim(0.5, 1.05)
        ax.axvline(0.85, color="white", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_title("Coherence intra-individu (galerie V4)", color="white", fontsize=10)
        ax.tick_params(colors="gray"); ax.spines[:].set_color("#333")

    plt.suptitle("V4 Diagnostic complet", color="white",
                 fontsize=16, fontweight="bold", y=1.01)

    out = RESULTS_DIR / "v4_diagnostic.png"
    plt.savefig(str(out), dpi=130, bbox_inches="tight", facecolor="#0f0f0f")
    plt.close()
    log(f"  Plot saved: {out}")

# ==============================================================================
# CHECKPOINT SAVE / LOAD
# ==============================================================================

def save_resume(backbone, arc_loss, optimizer, scheduler,
                epoch, best_val, history, all_classes):
    state = {
        "backbone": backbone.state_dict(),
        "arc_loss": arc_loss.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
        "history": history,
        "all_classes": all_classes,
        "rng_np": np.random.get_state(),
        "rng_torch": torch.get_rng_state().tolist(),
    }
    tmp = RESUME_CKP.with_suffix(".tmp")
    torch.save(state, str(tmp))
    tmp.replace(RESUME_CKP)

def save_best(backbone, arc_loss, all_classes, emb_dim, epoch, val_acc):
    state = {
        "backbone_state": backbone.state_dict(),
        "arc_loss_state": arc_loss.state_dict(),
        "classes":  all_classes,
        "emb_dim":  emb_dim,
        "epoch":    epoch,
        "val_acc":  val_acc,
        "img_size": IMG_SIZE,
        "arc_scale": ARC_SCALE,
        "arc_margin": ARC_MARGIN,
        "normalization": {"mean": MEGA_MEAN, "std": MEGA_STD},
        "save_time": datetime.now().isoformat(),
    }
    tmp = BEST_MODEL.with_suffix(".tmp")
    torch.save(state, str(tmp))
    tmp.replace(BEST_MODEL)
    log(f"  Best model saved: epoch={epoch} zoo_val_acc={val_acc*100:.2f}%")

def save_state(epoch, best_val, patience_count):
    s = {"epoch": epoch, "best_val": best_val,
         "patience_count": patience_count,
         "updated": datetime.now().isoformat()}
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=2))
    tmp.replace(STATE_FILE)

# ==============================================================================
# TRAINING EPOCH
# ==============================================================================

def train_epoch(backbone, arc_loss, loader, optimizer, scheduler, epoch):
    backbone.train(); arc_loss.train()
    total_loss = 0.0; total_n = 0; t0 = time.time()
    for i, (imgs, labs) in enumerate(loader):
        if _interrupt: break
        imgs = imgs.to(DEVICE, non_blocking=True)
        labs = labs.to(DEVICE, non_blocking=True)
        emb  = F.normalize(backbone(imgs), dim=1)
        loss = arc_loss(emb, labs)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(backbone.parameters()) + list(arc_loss.parameters()), 1.0)
        optimizer.step()
        if scheduler is not None: scheduler.step()
        total_loss += loss.item() * imgs.size(0)
        total_n    += imgs.size(0)
        done = i + 1; total_b = len(loader)
        pct  = done / total_b
        fill = int(40 * pct)
        bar  = "#" * fill + "." * (40 - fill)
        eta  = (time.time() - t0) / done * (total_b - done)
        print(f"\r  [{bar}] {done}/{total_b} loss={total_loss/total_n:.4f} ETA={timedelta(seconds=int(eta))}",
              end="", flush=True)
    print()
    return total_loss / max(total_n, 1)

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    title("V4 TRAINING — Improved augmentations + 40 supervised individuals")
    log(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Device   : {DEVICE}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        log(f"  GPU      : {p.name} ({p.total_memory/1e9:.1f} GB)")
    log(f"  Output   : {OUT_DIR}")

    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        log("pip install scikit-learn", "ERR"); sys.exit(1)

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    section("Loading zoo dataset")
    zoo_tr_p, zoo_tr_l, zoo_va_p, zoo_va_l, zoo_classes = load_zoo(ZOO_DIR, VAL_RATIO_ZOO)
    n_zoo = len(zoo_classes)

    section("Loading BOS dataset")
    bos_tr_p, bos_tr_l, bos_va_p, bos_va_l, bos_classes = load_bos(BOS_DIR, offset=n_zoo)
    n_bos = len(bos_classes)

    all_classes  = zoo_classes + bos_classes
    wild_class   = len(all_classes)
    n_total_cls  = wild_class + 1

    log(f"\n  Total supervised classes: {len(all_classes)} (zoo={n_zoo}, BOS={n_bos})")
    log(f"  Wild background class index: {wild_class}")

    # Combined train paths/labels
    all_tr_p = zoo_tr_p + bos_tr_p
    all_tr_l = zoo_tr_l + bos_tr_l
    all_va_p = zoo_va_p + bos_va_p
    all_va_l = zoo_va_l + bos_va_l

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    section("Loading MegaDescriptor-T-224 (starting from V3 checkpoint)")
    if not V3_MODEL.exists():
        log(f"V3 model not found: {V3_MODEL}", "ERR"); sys.exit(1)

    v3_ckpt = torch.load(str(V3_MODEL), map_location=DEVICE, weights_only=False)
    backbone = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224",
                                  pretrained=False, num_classes=0)
    backbone.load_state_dict(v3_ckpt["backbone_state"])
    backbone = backbone.to(DEVICE)
    with torch.no_grad():
        dummy  = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        emb_dim = backbone(dummy).shape[1]
    log(f"  Embedding dim : {emb_dim}")
    log(f"  Params        : {sum(p.numel() for p in backbone.parameters())/1e6:.1f}M")
    log(f"  Starting from : V3 epoch {v3_ckpt.get('epoch','?')}, val_acc={v3_ckpt.get('val_acc',0)*100:.1f}%")

    k_per_class = [K_KNOWN] * len(all_classes) + [K_UNKNOWN]
    arc_loss    = SubCenterArcFace(emb_dim, n_total_cls, k_per_class,
                                   ARC_SCALE, ARC_MARGIN).to(DEVICE)

    optimizer = optim.AdamW([
        {"params": backbone.parameters(), "lr": LR_BACKBONE},
        {"params": arc_loss.parameters(), "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    # Cosine annealing scheduler (total steps estimated)
    steps_per_epoch = (len(all_tr_p) // BATCH_SIZE) + (WILD_PER_EPOCH // BATCH_SIZE)
    total_steps     = MAX_EPOCHS * steps_per_epoch
    warmup_steps    = WARMUP_EPOCHS * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * prog)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Resume if possible
    # ------------------------------------------------------------------
    start_epoch    = 1
    best_val       = 0.0
    patience_count = 0
    history        = {"train_loss": [], "zoo_val_acc": [], "bos_val_acc": []}

    if RESUME_CKP.exists() and STATE_FILE.exists():
        log("\n  Resume checkpoint detected - loading...")
        try:
            ckpt = torch.load(str(RESUME_CKP), map_location=DEVICE, weights_only=False)
            backbone.load_state_dict(ckpt["backbone"])
            arc_loss.load_state_dict(ckpt["arc_loss"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch    = ckpt["epoch"] + 1
            best_val       = ckpt["best_val"]
            history        = ckpt["history"]
            np.random.set_state(ckpt["rng_np"])
            torch.set_rng_state(torch.tensor(ckpt["rng_torch"], dtype=torch.uint8))
            state          = json.loads(STATE_FILE.read_text())
            patience_count = state.get("patience_count", 0)
            log(f"  Resumed from epoch {start_epoch-1}. Best val={best_val*100:.2f}%")
        except Exception as e:
            log(f"  Resume failed ({e}), starting fresh.", "WARN")
            start_epoch = 1
    else:
        log("\n  No checkpoint found - fresh training.")

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    section("Building DataLoaders")

    # Weighted sampler for known classes (compensate imbalance)
    counts    = Counter(all_tr_l)
    weights   = [1.0 / counts[l] for l in all_tr_l]
    sampler   = WeightedRandomSampler(weights, len(weights), replacement=True)
    zoo_tf    = get_train_tf_zoo()
    bos_tf    = get_train_tf_bos()

    # Split zoo and BOS paths for per-transform handling
    combined_paths  = all_tr_p
    combined_labels = all_tr_l
    combined_tfs    = [zoo_tf if l < n_zoo else bos_tf for l in combined_labels]

    known_ds  = MultiTfDataset(combined_paths, combined_labels, combined_tfs)
    known_dl  = DataLoader(known_ds, batch_size=BATCH_SIZE, sampler=sampler,
                           num_workers=4, pin_memory=True, persistent_workers=True)

    wild_ds   = WildDataset(WILD_DIR, get_train_tf_wild(), WILD_PER_EPOCH, wild_class)
    # NOTE: wild_dl is recreated each epoch after _resample() so workers
    # always see the freshly sampled file list. persistent_workers=False is
    # mandatory here — with persistent workers, the worker processes keep
    # their initial copy of wild_ds.files and never see the resampled list.

    log(f"  Known loader : {len(known_ds)} crops, {len(known_dl)} batches/epoch")
    log(f"  Wild dataset : {len(wild_ds.all_files)} total crops, {WILD_PER_EPOCH} sampled/epoch")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    title("TRAINING — press Ctrl+C to stop and save")
    t_start = time.time()

    zoo_class_range = list(range(n_zoo))
    bos_class_range = list(range(n_zoo, n_zoo + n_bos))

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        if _interrupt:
            log("\n  Interrupt flag set, stopping training loop."); break

        elapsed = time.time() - t_start
        log(f"\n  Epoch {epoch}/{MAX_EPOCHS}  |  best_zoo_val={best_val*100:.2f}%  "
            f"|  patience={patience_count}/{PATIENCE}  |  elapsed={timedelta(seconds=int(elapsed))}")

        wild_ds._resample()
        wild_dl = DataLoader(wild_ds, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=4, pin_memory=True, persistent_workers=False)
        log(f"  Wild: {len(wild_ds.files)} crops resampled for this epoch")

        log("  [1/3] Training on known individuals...")
        loss1 = train_epoch(backbone, arc_loss, known_dl, optimizer, scheduler, epoch)

        log("  [2/3] Training on wild background...")
        loss2 = train_epoch(backbone, arc_loss, wild_dl, optimizer, scheduler, epoch)
        avg_loss = (loss1 + loss2) / 2

        log("  [3/3] Validating...")
        zoo_val_acc = validate(backbone, zoo_va_p, zoo_va_l, all_classes, zoo_class_range)
        bos_val_acc = validate(backbone, bos_va_p, bos_va_l, all_classes, bos_class_range)

        log(f"  Loss={avg_loss:.4f} | Zoo_val={zoo_val_acc*100:.2f}% | BOS_val={bos_val_acc*100:.2f}%")
        history["train_loss"].append(avg_loss)
        history["zoo_val_acc"].append(zoo_val_acc)
        history["bos_val_acc"].append(bos_val_acc)

        if zoo_val_acc > best_val:
            best_val       = zoo_val_acc
            patience_count = 0
            save_best(backbone, arc_loss, all_classes, emb_dim, epoch, zoo_val_acc)
        else:
            if epoch >= PATIENCE_START:
                patience_count += 1
                if patience_count >= PATIENCE:
                    log(f"  Early stopping at epoch {epoch} (patience={PATIENCE})"); break

        save_resume(backbone, arc_loss, optimizer, scheduler,
                    epoch, best_val, history, all_classes)
        save_state(epoch, best_val, patience_count)

        eta = (time.time() - t_start) / epoch * (MAX_EPOCHS - epoch)
        log(f"  ETA: {timedelta(seconds=int(eta))}")

    # ------------------------------------------------------------------
    # Post-training
    # ------------------------------------------------------------------
    title("POST-TRAINING — loading best model")
    if BEST_MODEL.exists():
        best_ckpt = torch.load(str(BEST_MODEL), map_location=DEVICE, weights_only=False)
        backbone.load_state_dict(best_ckpt["backbone_state"])
        log(f"  Best: epoch={best_ckpt['epoch']}, zoo_val={best_ckpt['val_acc']*100:.2f}%")

    section("Building gallery (all 40 individuals)")
    opt_t, sep_gap, pos_m, neg_m = build_gallery(
        backbone, all_tr_p, all_tr_l, all_classes, emb_dim)

    # ------------------------------------------------------------------
    # Stress test comparison
    # ------------------------------------------------------------------
    section("Stress test comparison V3 vs V4")
    v3_backbone = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224",
                                     pretrained=False, num_classes=0)
    v3_backbone.load_state_dict(v3_ckpt["backbone_state"])
    v3_backbone = v3_backbone.eval().to(DEVICE)

    log("  Running comparison (4 degradations x 5 severities x 6 crops x 10 classes)...")
    stress_res, deg_types = stress_compare(v3_backbone, backbone,
                                            ZOO_DIR, zoo_classes, zoo_classes)

    log("\n  COMPARISON TABLE (zoo accuracy %):")
    sev_labels = ["None", "Light", "Moderate", "Strong", "Extreme"]
    log(f"  {'Degradation':<16}  {'Model':<5}  " +
        "  ".join(f"{s:<9}" for s in sev_labels))
    for dtype in deg_types:
        for mname in ["V3", "V4"]:
            if mname not in stress_res or dtype not in stress_res[mname]: continue
            vals = [f"{stress_res[mname][dtype][s]*100:>8.1f}%" for s in range(5)]
            log(f"  {dtype:<16}  {mname:<5}  " + "  ".join(vals))

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    section("Generating diagnostic plots")
    gallery_data = json.loads(GALLERY_OUT.read_text(encoding="utf-8"))
    # Rebuild pos/neg sims for plot
    pos_sims_v4, neg_sims_v4 = [], []
    with torch.no_grad():
        backbone.eval()
        all_protos  = torch.zeros(len(all_classes), emb_dim)
        for i, name in enumerate(all_classes):
            if name in gallery_data["individuals"]:
                all_protos[i] = torch.tensor(
                    gallery_data["individuals"][name]["embedding"])
        for i, name in enumerate(all_classes):
            d = (ZOO_DIR / name) if i < n_zoo else (BOS_DIR / name)
            if not d.exists(): continue
            imgs = sorted([f for f in d.iterdir() if f.suffix in exts])[:30]
            for p in imgs:
                try:
                    t   = val_tf(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE)
                    e   = F.normalize(backbone(t), dim=1).cpu()
                    own = float((e @ all_protos[i]).item())
                    pos_sims_v4.append(own)
                    others = [j for j in range(len(all_classes)) if j != i]
                    neg = float((e @ all_protos[others].T).max().item())
                    neg_sims_v4.append(neg)
                except: pass

    make_plots(history, stress_res, deg_types, all_classes,
               gallery_data, pos_sims_v4, neg_sims_v4)

    # ------------------------------------------------------------------
    # JSON final report
    # ------------------------------------------------------------------
    section("Writing final report")
    v3_baseline = {}
    for dtype in deg_types:
        if "V3" in stress_res and dtype in stress_res["V3"]:
            v3_baseline[dtype] = {str(s): round(stress_res["V3"][dtype][s]*100, 1)
                                  for s in range(5)}
    v4_results = {}
    for dtype in deg_types:
        if "V4" in stress_res and dtype in stress_res["V4"]:
            v4_results[dtype]  = {str(s): round(stress_res["V4"][dtype][s]*100, 1)
                                  for s in range(5)}

    report = {
        "generated":       datetime.now().isoformat(),
        "training_epochs": len(history["train_loss"]),
        "best_epoch":      int(np.argmax(history["zoo_val_acc"])) + 1,
        "v3_baseline": {
            "val_acc":        round(v3_ckpt.get("val_acc", 0)*100, 2),
            "epoch":          v3_ckpt.get("epoch"),
            "stress_results": v3_baseline,
        },
        "v4_improved": {
            "best_zoo_val_acc": round(best_val*100, 2),
            "separability_gap": round(sep_gap, 4),
            "pos_sim_mean":     round(pos_m, 4),
            "neg_sim_mean":     round(neg_m, 4),
            "calibrated_threshold": round(opt_t, 4),
            "num_classes":      len(all_classes),
            "zoo_classes":      zoo_classes,
            "bos_classes":      bos_classes,
            "stress_results":   v4_results,
        },
        "improvements": {
            dtype: {
                str(s): round((stress_res.get("V4",{}).get(dtype,{}).get(s,0) -
                               stress_res.get("V3",{}).get(dtype,{}).get(s,0))*100, 1)
                for s in range(5)
            }
            for dtype in deg_types
        },
        "hyperparameters": {
            "img_size": IMG_SIZE, "batch_size": BATCH_SIZE,
            "arc_scale": ARC_SCALE, "arc_margin": ARC_MARGIN,
            "k_known": K_KNOWN, "k_unknown": K_UNKNOWN,
            "lr_backbone": LR_BACKBONE, "lr_head": LR_HEAD,
            "weight_decay": WEIGHT_DECAY, "patience": PATIENCE,
            "patience_start": PATIENCE_START,
            "wild_per_epoch": WILD_PER_EPOCH,
        },
        "outputs": {
            "best_model":  str(BEST_MODEL),
            "gallery":     str(GALLERY_OUT),
            "plot":        str(RESULTS_DIR / "v4_diagnostic.png"),
            "log":         str(LOG_FILE),
        }
    }

    report_path = RESULTS_DIR / "final_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    log(f"  Report: {report_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    title("DONE")
    total_time = time.time() - t_start
    log(f"""
  V3 zoo_val_acc      : {v3_ckpt.get('val_acc',0)*100:.2f}%
  V4 best zoo_val_acc : {best_val*100:.2f}%  (epoch {int(np.argmax(history['zoo_val_acc']))+1})
  V4 separability gap : {sep_gap:.4f}
  V4 threshold        : {opt_t:.4f}
  V4 classes          : {len(all_classes)} individuals ({n_zoo} zoo + {n_bos} BOS)

  Outputs:
    Model   : {BEST_MODEL}
    Gallery : {GALLERY_OUT}
    Plot    : {RESULTS_DIR / 'v4_diagnostic.png'}
    Report  : {RESULTS_DIR / 'final_report.json'}
    Logs    : {LOG_FILE}

  Total training time : {timedelta(seconds=int(total_time))}
  Finished at         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")

if __name__ == "__main__":
    main()
