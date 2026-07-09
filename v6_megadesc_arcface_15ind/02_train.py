"""
V6_3_train.py — OrangIdentifier V6
====================================
Étape 3 : Entraînement ZOO-UNIQUEMENT (pas de BOS).

  - Individus zoo existants : v1/data/crops_dataset/raw  (excl. _a_verifier)
  - Nouveaux individus zoo  : v6/data/new_zoo_crops      (excl. dossiers _*)
  - Wild (régularisation)   : v3/data/wild_crops
  - Backbone de départ      : V3 (zoo-only, cohérent avec l'objectif)
  - Techniques V5 conservées : SubCenterArcFace, curriculum 4 phases,
    L_invariance, hard mining wild, crash-safety
  - Bugs V5 corrigés : scheduler (double-step), hard mining (proto_matrix),
    WildDataset._resample (replace=True)

Architecture :
  MegaDescriptor-T-224 + SubCenterArcFace (K_ZOO=2, K_WILD=5)
  4 phases de curriculum progressif
  L_arcface + L_invariance (robustesse dégradations)
  Crash-safe : checkpoint après chaque epoch

CRASH-SAFETY :
  Fermer la fenêtre = relancer = reprend exactement où ça s'était arrêté.

RUN :
    conda activate orangs
    python v6_megadesc_arcface_15ind/02_train.py
    python v6_megadesc_arcface_15ind/02_train.py --dry-run   # 4 epochs test
"""

import os
os.environ["HF_HOME"]              = r"D:\HuggingFaceCache"
os.environ["TORCH_HOME"]           = r"D:\TorchCache"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys, io, json, math, time, signal, random, warnings, argparse, multiprocessing as _mp
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter

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

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]  # repository root (portable)

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    RICH = True
except ImportError:
    RICH = False

# ══════════════════════════════════════════════════════════════════════════════
# ARGS
# ══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true", help="3 epochs de test")
ARGS = parser.parse_args()
DRY  = ARGS.dry_run

# ══════════════════════════════════════════════════════════════════════════════
# CHEMINS
# ══════════════════════════════════════════════════════════════════════════════
# Backbone de départ : V3 (zoo-only — cohérent avec l'objectif V6)
# ArcFace sera réinitialisé (nouvelles classes → poids incompatibles)
V3_CKPT   = (REPO / "models" / "megadesc_T_arcface_final_epoch21_acc99.pt")

ZOO_DIR   = (REPO / "data" / "crops" / "known")   # individus zoo existants
NEW_ZOO   = (REPO / "data" / "crops" / "new")        # nouveaux individus zoo
WILD_DIR  = (REPO / "data" / "crops" / "wild")            # crops sauvages (régularisation)

V6_BASE   = (REPO / "output" / "v6")
MODELS    = V6_BASE / "models";  MODELS.mkdir(parents=True, exist_ok=True)
RESULTS   = V6_BASE / "results"; RESULTS.mkdir(parents=True, exist_ok=True)
LOGS      = V6_BASE / "logs";    LOGS.mkdir(parents=True, exist_ok=True)

BEST_PT     = MODELS / "v6_best.pt"
RESUME_PT   = MODELS / "v6_resume.pt"
BACKBONE_PT = MODELS / "v6_backbone_only.pt"
GALLERY_JS  = MODELS / "v6_gallery.json"
REPORT_JS   = RESULTS / "v6_report.json"
CURVES_PNG  = RESULTS / "v6_curves.png"
LOG_FILE    = LOGS    / "v6_training.log"

# ══════════════════════════════════════════════════════════════════════════════
# HYPER-PARAMÈTRES
# ══════════════════════════════════════════════════════════════════════════════
IMG_SIZE   = 224
BATCH      = 32
SEED       = 42
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = STD = [0.5, 0.5, 0.5]
EXTS       = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

ARC_SCALE  = 64
ARC_MARGIN = 0.35
K_ZOO      = 2    # sous-centres par individu zoo (intra-class variation)
K_WILD     = 5    # sous-centres pour la classe "inconnu"

# Galerie
K_EXEMPLARS_ZOO = 25
QUALITY_ZOO     = 0.60

VAL_RATIO  = 0.15
WILD_HARD_MINING_FROM_PHASE = 2

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# 4 PHASES (identiques à V5)
# ══════════════════════════════════════════════════════════════════════════════
PHASES = [
    dict(name="Phase 0 — Initialisation", epochs=3 if not DRY else 1,
         freeze=True,  lr_bb=0.0,  lr_h=1e-3,
         lam_inv=0.00, severity=0.00, wild=0),
    dict(name="Phase 1 — Warmup",         epochs=15 if not DRY else 1,
         freeze=False, lr_bb=1e-7, lr_h=5e-4,
         lam_inv=0.10, severity=0.40, wild=0),
    dict(name="Phase 2 — Apprentissage",  epochs=20 if not DRY else 1,
         freeze=False, lr_bb=5e-6, lr_h=3e-4,
         lam_inv=0.30, severity=0.70, wild=1200),
    dict(name="Phase 3 — Consolidation", epochs=15 if not DRY else 1,
         freeze=False, lr_bb=2e-6, lr_h=1e-4,
         lam_inv=0.50, severity=1.00, wild=1500,
         early_stop=True, patience=15),
]
TOTAL_EPOCHS = sum(p["epochs"] for p in PHASES)

# ══════════════════════════════════════════════════════════════════════════════
# LOGGER
# ══════════════════════════════════════════════════════════════════════════════
_IS_MAIN = _mp.current_process().name == "MainProcess"
_log_fh  = open(LOG_FILE if _IS_MAIN else os.devnull, "a", encoding="utf-8", buffering=1)
_console = Console() if (RICH and _IS_MAIN) else None

def log(msg="", level="INFO"):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}][{level}] {msg}"
    _log_fh.write(line + "\n"); _log_fh.flush()
    if _console:
        _console.print(line, markup=False, highlight=False)
    else:
        print(line)

def section(t):
    bar = "─" * 68
    log(""); log(bar); log(f"  {t}"); log(bar)

# ══════════════════════════════════════════════════════════════════════════════
# CRASH SAFETY
# ══════════════════════════════════════════════════════════════════════════════
_interrupt = False
_save_fn   = None

def _emergency_save():
    if _save_fn is not None:
        try: _save_fn(reason="interrupt")
        except Exception as e: log(f"Emergency save échouée : {e}", "ERROR")

def _sigint(sig, frame):
    global _interrupt
    if not _interrupt:
        log("[Ctrl+C] Arrêt propre après le batch courant...", "WARN")
        _interrupt = True
    else:
        _emergency_save(); _log_fh.close(); sys.exit(1)

signal.signal(signal.SIGINT, _sigint)

if _IS_MAIN:
    try:
        import win32api
        def _win32_handler(ct):
            if ct in (2, 5, 6):
                log(f"[Win32 CTRL={ct}] Fermeture — sauvegarde urgence...", "WARN")
                global _interrupt; _interrupt = True
                _emergency_save(); time.sleep(1)
            return False
        win32api.SetConsoleCtrlHandler(_win32_handler, True)
        log("Handler Win32 installé")
    except ImportError:
        log("pywin32 absent — checkpoint par epoch uniquement", "WARN")

# ══════════════════════════════════════════════════════════════════════════════
# SUB-CENTER ARCFACE
# ══════════════════════════════════════════════════════════════════════════════
class SubCenterArcFace(nn.Module):
    def __init__(self, emb_dim, num_classes, k_per_class, scale=64.0, margin=0.35):
        super().__init__()
        self.scale, self.margin = scale, margin
        self.num_classes = num_classes
        total_k = sum(k_per_class)
        self.weight = nn.Parameter(torch.FloatTensor(total_k, emb_dim))
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer("cls_start",
            torch.tensor([sum(k_per_class[:i]) for i in range(num_classes)], dtype=torch.long))
        self.register_buffer("cls_k", torch.tensor(k_per_class, dtype=torch.long))
        self.cos_m = math.cos(margin); self.sin_m = math.sin(margin)
        self.th    = math.cos(math.pi - margin)
        self.mm    = math.sin(math.pi - margin) * margin

    def forward(self, emb, labels):
        w      = F.normalize(self.weight, dim=1)
        ca     = emb @ w.T
        logits = torch.zeros(emb.size(0), self.num_classes, device=emb.device)
        for c in range(self.num_classes):
            s = self.cls_start[c].item(); k = self.cls_k[c].item()
            logits[:, c] = ca[:, s:s+k].max(dim=1).values
        cos = logits.clamp(-1.0, 1.0)
        sin = (1.0 - cos**2).clamp(0.0, 1.0).sqrt()
        phi = cos * self.cos_m - sin * self.sin_m
        phi = torch.where(cos > self.th, phi, cos - self.mm)
        oh  = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1.0)
        out = (oh * phi + (1 - oh) * logits) * self.scale
        return F.cross_entropy(out, labels, label_smoothing=0.05)

# ══════════════════════════════════════════════════════════════════════════════
# LOSSES AUXILIAIRES
# ══════════════════════════════════════════════════════════════════════════════
def loss_invariance(ec, ed):
    return 1.0 - (ec * ed).sum(dim=1).mean()

def loss_bos_spread(emb, labels, n_zoo, n_known):
    mask = (labels >= n_zoo) & (labels < n_known)
    if mask.sum() < 4:
        return torch.tensor(0.0, device=emb.device)
    be   = emb[mask]; bl = labels[mask]
    sim  = be @ be.T
    diff = (bl.unsqueeze(0) != bl.unsqueeze(1))
    if not diff.any():
        return torch.tensor(0.0, device=emb.device)
    return sim[diff].mean()

# ══════════════════════════════════════════════════════════════════════════════
# AUGMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════
class _LowRes:
    def __init__(self, min_f, max_f, p): self.min_f=min_f; self.max_f=max_f; self.p=p
    def __call__(self, img):
        if random.random() > self.p: return img
        w, h = img.size; f = random.uniform(self.min_f, self.max_f)
        s = max(int(w*f), 8), max(int(h*f), 8)
        return img.resize(s, Image.BILINEAR).resize((w,h), Image.BICUBIC)

class _JPEG:
    def __init__(self, min_q, max_q, p): self.min_q=min_q; self.max_q=max_q; self.p=p
    def __call__(self, img):
        if random.random() > self.p: return img
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=random.randint(self.min_q, self.max_q))
        buf.seek(0); return Image.open(buf).copy()

class _CropJitter:
    def __init__(self, px=2): self.px=px
    def __call__(self, t):
        p = T.functional.pad(t, self.px, padding_mode="reflect")
        i = random.randint(0, 2*self.px); j = random.randint(0, 2*self.px)
        return p[:, i:i+t.shape[1], j:j+t.shape[2]]

_norm = T.Normalize(MEAN, STD)

def get_clean_tf():
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(0.2, 0.2, 0.1, 0.05),
        T.ToTensor(), _norm,
    ])

def get_degraded_tf(severity: float):
    s = severity
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(max(0.50, 0.75-0.25*s), 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(int(10 + 25*s)),
        T.ColorJitter(0.2+0.4*s, 0.2+0.4*s, 0.1+0.3*s, 0.05+0.1*s),
        T.RandomGrayscale(p=0.05*s),
        T.Lambda(_LowRes(max(0.06, 0.45-0.39*s), 0.90, p=0.10+0.35*s)),
        T.RandomApply([T.GaussianBlur(11, sigma=(0.5, 0.5+6.5*s))], p=0.10+0.40*s),
        T.Lambda(_JPEG(max(5, int(40-35*s)), max(40, int(75-35*s)), p=0.10+0.30*s)),
        T.ToTensor(),
        T.RandomErasing(p=0.10+0.20*s, scale=(0.02, 0.02+0.23*s), value="random"),
        T.Lambda(_CropJitter(px=2)),
        _norm,
    ])

def get_val_tf():
    return T.Compose([T.Resize(IMG_SIZE), T.CenterCrop(IMG_SIZE), T.ToTensor(), _norm])

# ══════════════════════════════════════════════════════════════════════════════
# DATASETS
# ══════════════════════════════════════════════════════════════════════════════
class PairDataset(Dataset):
    def __init__(self, paths, labels, severity=1.0):
        self.paths=paths; self.labels=labels; self.severity=severity
    def __len__(self): return len(self.paths)
    def _load(self, idx):
        try:    return Image.open(self.paths[idx]).convert("RGB")
        except: return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
    def __getitem__(self, idx):
        img = self._load(idx)
        return get_clean_tf()(img), get_degraded_tf(self.severity)(img), int(self.labels[idx])

class PlainDataset(Dataset):
    def __init__(self, paths, labels, tf=None):
        self.paths=paths; self.labels=labels; self.tf=tf or get_val_tf()
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        try:    img = Image.open(self.paths[idx]).convert("RGB")
        except: img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        return self.tf(img), int(self.labels[idx])

class WildDataset(Dataset):
    def __init__(self, wild_dir, n, unknown_label):
        self.all  = sorted([f for f in wild_dir.iterdir() if f.suffix in EXTS])
        self.lbl  = unknown_label; self._resample(n)
    def _resample(self, n, weights=None):
        n = min(n, len(self.all))
        if weights is not None and len(weights) == len(self.all):
            # replace=True obligatoire : seuls ~512 items ont poids > 0,
            # et n peut être 1200-1500 → replace=False lèverait ValueError.
            # Sémantiquement correct : on veut sur-échantillonner les hard negatives.
            idx = np.random.choice(len(self.all), n, replace=True, p=weights)
            self.files = [self.all[i] for i in idx]
        else:
            self.files = random.sample(self.all, n)
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        try:    img = Image.open(self.files[idx]).convert("RGB")
        except: img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (100, 80, 60))
        return get_clean_tf()(img), get_degraded_tf(1.0)(img), self.lbl

# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DONNÉES — fusion ZOO + NEW_ZOO triés alphabétiquement
# ══════════════════════════════════════════════════════════════════════════════
def load_zoo_merged(zoo_dirs, offset=0, exclude=None):
    """Charge plusieurs dossiers zoo, fusionne et trie les individus par nom."""
    excl = set(exclude or [])
    all_ind_dirs = []
    for base in zoo_dirs:
        if not base.exists():
            continue
        for d in base.iterdir():
            if d.is_dir() and not d.name.startswith("_") and d.name not in excl:
                all_ind_dirs.append(d)

    # Tri alphabétique par nom d'individu (reproductible)
    all_ind_dirs.sort(key=lambda d: d.name.lower())

    paths, labels, names = [], [], []
    for i, d in enumerate(all_ind_dirs):
        imgs = sorted([f for f in d.iterdir() if f.suffix in EXTS])
        for f in imgs:
            paths.append(f); labels.append(i + offset)
        names.append(d.name)
    return paths, labels, names

def load_dir(base, offset=0, exclude=None):
    excl = set(exclude or [])
    dirs = sorted([d for d in base.iterdir()
                   if d.is_dir() and not d.name.startswith("_") and d.name not in excl])
    paths, labels, names = [], [], []
    for i, d in enumerate(dirs):
        for f in sorted([f for f in d.iterdir() if f.suffix in EXTS]):
            paths.append(f); labels.append(i + offset)
        names.append(d.name)
    return paths, labels, names

# ══════════════════════════════════════════════════════════════════════════════
# BACKBONE
# ══════════════════════════════════════════════════════════════════════════════
def load_backbone():
    bb = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224", pretrained=False, num_classes=0)

    # Priorité : V3 (zoo-only — cohérent avec l'objectif V6 sans BOS)
    if V3_CKPT.exists():
        ck    = torch.load(str(V3_CKPT), map_location="cpu", weights_only=False)
        state = ck.get("backbone_state") or ck.get("model_state_dict") or ck
        miss, unex = bb.load_state_dict(state, strict=False)
        log(f"  Backbone V3 — {len(miss)} miss, {len(unex)} unexpected")
        src = "V3"
    else:
        bb  = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224", pretrained=True, num_classes=0)
        log("  Backbone HuggingFace (V3 introuvable)", "WARN")
        src = "HuggingFace"

    with torch.no_grad():
        emb_dim = bb(torch.randn(1, 3, IMG_SIZE, IMG_SIZE)).shape[1]
    log(f"  {src} — {emb_dim}D — {sum(p.numel() for p in bb.parameters())/1e6:.1f}M params")
    return bb, emb_dim, src

# ══════════════════════════════════════════════════════════════════════════════
# ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════════════════
def train_epoch(bb, arc, loaders, opt, sched, pc, n_zoo, live_state, refresh_fn=None):
    bb.train(); arc.train()
    lam_inv = pc["lam_inv"]
    tot_arc = tot_inv = tot_n = 0.0
    all_loaders  = [l for l in loaders if l is not None]
    total_batches = sum(len(l) for l in all_loaders)
    batch_idx = 0; t0 = time.time()

    for loader in all_loaders:
        for clean, deg, labels in loader:
            if _interrupt: break
            clean  = clean.to(DEVICE,  non_blocking=True).float()
            deg    = deg.to(DEVICE,    non_blocking=True).float()
            labels = labels.to(DEVICE, non_blocking=True).long()
            both   = torch.cat([clean, deg], dim=0)
            ea     = F.normalize(bb(both), dim=1); B = clean.size(0)
            ec     = ea[:B]; ed = ea[B:]
            l_arc  = arc(ec, labels) + arc(ed, labels)
            l_inv  = loss_invariance(ec, ed)
            loss   = l_arc + lam_inv*l_inv
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(bb.parameters()) + list(arc.parameters()), 1.0)
            opt.step()
            bs = labels.size(0)
            tot_arc += l_arc.item()*bs; tot_inv += l_inv.item()*bs
            tot_n   += bs
            batch_idx += 1
            eta_b = (time.time()-t0)/batch_idx * (total_batches-batch_idx)
            live_state.update({"batch": batch_idx, "total_batches": total_batches,
                               "l_arc": tot_arc/tot_n, "l_inv": tot_inv/tot_n,
                               "eta_batch": int(eta_b)})
            if refresh_fn: refresh_fn()
        if _interrupt: break

    N = max(tot_n, 1)
    return tot_arc/N, tot_inv/N

@torch.no_grad()
def validate(bb, zoo_tr_p, zoo_tr_l, zoo_va_p, zoo_va_l, n_zoo, emb_dim):
    bb.eval()
    proto = torch.zeros(n_zoo, emb_dim); cnt = torch.zeros(n_zoo)
    for imgs, lbs in DataLoader(PlainDataset(zoo_tr_p, zoo_tr_l), 64, num_workers=0):
        e = F.normalize(bb(imgs.to(DEVICE).float()), dim=1).cpu()
        for ei, li in zip(e, lbs.tolist()):
            if li < n_zoo: proto[li] += ei; cnt[li] += 1
    for c in range(n_zoo):
        if cnt[c] > 0: proto[c] = F.normalize(proto[c], dim=0)

    ok = tot = 0
    for imgs, lbs in DataLoader(PlainDataset(zoo_va_p, zoo_va_l), 64, num_workers=0):
        e = F.normalize(bb(imgs.to(DEVICE).float()), dim=1).cpu()
        ok += (e @ proto.T).argmax(1).eq(lbs).sum().item(); tot += lbs.size(0)
    acc_c = ok / max(tot, 1)

    ok = tot = 0
    dtf = get_degraded_tf(0.70)
    for imgs, lbs in DataLoader(PlainDataset(zoo_va_p, zoo_va_l, dtf), 64, num_workers=0):
        e = F.normalize(bb(imgs.to(DEVICE).float()), dim=1).cpu()
        ok += (e @ proto.T).argmax(1).eq(lbs).sum().item(); tot += lbs.size(0)
    acc_d = ok / max(tot, 1)
    return acc_c, acc_d

@torch.no_grad()
def bos_discrimination(bb, bos_p, bos_l, n_zoo):
    if not bos_p: return 0.0
    bb.eval()
    embs, labs = [], []
    for imgs, lbs in DataLoader(PlainDataset(bos_p, bos_l), 64, num_workers=0):
        embs.append(F.normalize(bb(imgs.to(DEVICE).float()), dim=1).cpu())
        labs.extend(lbs.tolist())
    embs = torch.cat(embs).numpy(); labs = np.array(labs)
    intra, inter = [], []
    for i in sorted(set(labs)):
        mi = embs[labs==i]
        if len(mi) < 2: continue
        for a in range(len(mi)):
            for b in range(a+1, min(len(mi), 10)): intra.append(float(mi[a]@mi[b]))
        oth = embs[labs!=i]
        if len(oth): c = oth.mean(0); c /= (np.linalg.norm(c)+1e-8); inter.append(float(mi.mean(0)@c))
    if not intra or not inter: return 0.0
    return float(np.mean(intra) - np.mean(inter))

@torch.no_grad()
def build_proto_np(bb, paths, labels, n_cls, emb_dim):
    """Matrice de centroïdes par classe (float32 numpy) — pour hard mining wild."""
    bb.eval()
    proto = np.zeros((n_cls, emb_dim), dtype=np.float32)
    cnt   = np.zeros(n_cls)
    for imgs, lbs in DataLoader(PlainDataset(paths, labels), 64, num_workers=0):
        e = F.normalize(bb(imgs.to(DEVICE).float()), dim=1).cpu().numpy()
        for ei, li in zip(e, lbs.tolist()):
            if li < n_cls:
                proto[li] += ei; cnt[li] += 1
    for c in range(n_cls):
        if cnt[c] > 0:
            nrm = np.linalg.norm(proto[c])
            if nrm > 1e-8: proto[c] /= nrm
    return proto

@torch.no_grad()
def compute_wild_weights(bb, all_wild_files, proto_matrix):
    if len(all_wild_files) < 512: return None
    sample = random.sample(all_wild_files, 512)
    ds = PlainDataset(sample, [0]*512)
    bb.eval(); embs = []
    for imgs, _ in DataLoader(ds, 64, num_workers=0):
        embs.append(F.normalize(bb(imgs.to(DEVICE).float()), dim=1).cpu().numpy())
    embs = np.concatenate(embs)
    max_sims = (embs @ proto_matrix.T).max(1)
    w = np.exp(max_sims / 0.3); w /= w.sum()
    full = np.zeros(len(all_wild_files))
    idx  = {str(f): i for i, f in enumerate(all_wild_files)}
    for j, p in enumerate(sample):
        if str(p) in idx: full[idx[str(p)]] = w[j]
    return full / full.sum() if full.sum() > 1e-8 else None

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINTS
# ══════════════════════════════════════════════════════════════════════════════
def _atomic(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp); tmp.replace(path)

def save_resume(bb, arc, opt, sched, phase_idx, ep_in, global_ep,
                best_val, best_ep, history, classes):
    _atomic({
        "backbone_state": bb.state_dict(), "arc_loss_state": arc.state_dict(),
        "optimizer_state": opt.state_dict(),
        "scheduler_state": sched.state_dict() if sched else None,
        "phase_idx": phase_idx, "ep_in_phase": ep_in, "global_ep": global_ep,
        "best_val": best_val, "best_ep": best_ep, "history": history,
        "classes": classes, "saved_at": datetime.now().isoformat(),
    }, RESUME_PT)

def save_best(bb, arc, classes, emb_dim, ep, val, src):
    _atomic({
        "backbone_state": bb.state_dict(), "arc_loss_state": arc.state_dict(),
        "classes": classes, "emb_dim": emb_dim, "epoch": ep,
        "val_composite": val, "source": src, "version": "v6",
        "normalization": {"mean": MEAN, "std": STD},
    }, BEST_PT)
    _atomic({"backbone_state": bb.state_dict(), "emb_dim": emb_dim,
             "classes": classes, "version": "v6",
             "normalization": {"mean": MEAN, "std": STD}}, BACKBONE_PT)

# ══════════════════════════════════════════════════════════════════════════════
# GALERIE
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def build_gallery(bb, all_paths, all_labels, all_names, n_zoo, emb_dim):
    section("Construction de la galerie (exemplaires filtrés qualité)")
    bb.eval()
    ds = PlainDataset(all_paths, all_labels)
    dl = DataLoader(ds, 64, num_workers=0)
    embs, labs = [], []
    for imgs, lbs in dl:
        embs.append(F.normalize(bb(imgs.to(DEVICE).float()), dim=1).cpu().numpy())
        labs.extend(lbs.tolist())
    embs = np.concatenate(embs).astype(np.float32)
    labs = np.array(labs)
    n_known      = len(all_names)
    proto_matrix = np.zeros((n_known, emb_dim), dtype=np.float32)
    individuals  = {}
    all_exemplars = {}   # name → (K, emb_dim) — construit en passe 1

    # ── Passe 1 : exemplaires + métadonnées ───────────────────────────────────
    for i, name in enumerate(all_names):
        mask     = labs == i; ei = embs[mask]
        if len(ei) == 0: log(f"  [WARN] {name}: 0 crops", "WARN"); continue
        centroid = ei.mean(0); centroid /= (np.linalg.norm(centroid)+1e-8)
        proto_matrix[i] = centroid

        sims  = ei @ centroid
        q     = QUALITY_ZOO
        k_max = K_EXEMPLARS_ZOO
        good  = ei[sims >= q]
        if len(good) < 3: good = ei[np.argsort(-sims)[:max(3, k_max//2)]]

        gs    = good @ centroid
        top_k = min(k_max, len(good))
        best  = good[np.argsort(-gs)[:top_k]]
        norms = np.linalg.norm(best, axis=1, keepdims=True)
        best  = best / np.where(norms > 1e-8, norms, 1)
        all_exemplars[name] = best

        tag = "ZOO"
        individuals[name] = {
            "class_index": i,
            "is_zoo":      True,
            "num_crops":     int(len(ei)),
            "num_exemplars": len(best),
            "mean_intra":    round(float(np.mean(sims)), 4),
            "embedding":     centroid.tolist(),   # ancre lue par l'app Android
            "exemplars":     best.tolist(),       # exemplaires pour scoring max Python
        }
        log(f"  [{tag}] {name:<14}: {len(ei):3d} crops → {len(best):2d} exemplaires")

    # ── Passe 2 : calibration seuil avec scoring max-over-exemplaires ─────────
    # Même logique que l'app : score = max(dot(query, exemplaire))
    pos_sims = []; neg_sims = []
    name_list = list(all_exemplars.keys())
    for i, name in enumerate(all_names):
        if name not in all_exemplars: continue
        mask = labs == i; ei = embs[mask]
        if len(ei) == 0: continue
        own_ex = all_exemplars[name]

        # Positifs : max sim sur ses propres exemplaires
        pos_sims.extend((ei @ own_ex.T).max(1).tolist())

        # Négatifs : max sim sur les exemplaires de tous les autres individus
        other_ex_list = [v for k2, v in all_exemplars.items() if k2 != name]
        if other_ex_list:
            other_ex = np.vstack(other_ex_list)
            neg_sims.extend((ei @ other_ex.T).max(1).tolist())

    pos = np.array(pos_sims); neg = np.array(neg_sims) if neg_sims else np.zeros(1)
    gap = float(pos.mean() - neg.mean())
    log(f"\n  Pos (max-over-exemplaires) : {pos.mean():.4f} ± {pos.std():.4f}")
    log(f"  Neg (max-over-exemplaires) : {neg.mean():.4f} ± {neg.std():.4f}")
    log(f"  Gap : {gap:.4f}")

    thresholds = np.linspace(0, 1, 500); f1s = []
    for t in thresholds:
        tp = int((pos >= t).sum()); fp = int((neg >= t).sum()); fn = int((pos < t).sum())
        p  = tp/(tp+fp+1e-9); r = tp/(tp+fn+1e-9)
        f1s.append(2*p*r/(p+r+1e-9))
    opt_t = float(thresholds[np.argmax(f1s)])
    log(f"  Seuil optimal : {opt_t:.4f}  (F1={max(f1s):.4f})")

    gallery = {
        "version": "v6", "created": datetime.now().isoformat(),
        "model": "MegaDescriptor-T-224 + SubCenterArcFace V6",
        "embedding_dim": emb_dim, "similarity_metric": "cosine",
        "normalization": "megadescriptor",
        "unknown_threshold": round(opt_t, 4),
        "separability_gap":  round(gap, 4),
        "num_individuals": len(individuals),
        "n_zoo": n_zoo,
        "inference_note": "score = max(dot(query, exemplar)) — app: max(anchor, field)",
        "individuals": individuals,
    }
    GALLERY_JS.write_text(json.dumps(gallery, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    log(f"  Galerie → {GALLERY_JS.name} ({GALLERY_JS.stat().st_size/1024:.0f} KB)")
    return opt_t, gap, proto_matrix

# ══════════════════════════════════════════════════════════════════════════════
# RICH DISPLAY
# ══════════════════════════════════════════════════════════════════════════════
def make_rich_panel(state, pc, global_ep, best_ep, best_score):
    t = Table.grid(padding=1)
    t.add_column(style="bold cyan"); t.add_column()
    t.add_row("Phase",          pc["name"])
    t.add_row("Epoch",          f"{state.get('ep_in_phase','?')}/{pc['epochs']}  (global {global_ep}/{TOTAL_EPOCHS})")
    t.add_row("Batch",          f"{state.get('batch',0)}/{state.get('total_batches',1)}")
    t.add_row("L_arcface",      f"{state.get('l_arc',0):.4f}")
    t.add_row("L_inv",          f"{state.get('l_inv',0):.4f}  (lam={pc['lam_inv']:.2f})")
    t.add_row("─"*20,           "─"*30)
    t.add_row("Zoo propre",     f"{state.get('acc_c',0)*100:.2f}%")
    t.add_row("Zoo dégradé",    f"{state.get('acc_d',0)*100:.2f}%")
    t.add_row("Composite",      f"{state.get('composite',0):.4f}" + (" * BEST" if state.get("is_best") else ""))
    t.add_row("─"*20,           "─"*30)
    t.add_row("LR backbone",    f"{pc['lr_bb']:.1e}")
    t.add_row("LR head",        f"{pc['lr_h']:.1e}")
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated()/1e9
        tot  = torch.cuda.get_device_properties(0).total_memory/1e9
        t.add_row("GPU VRAM",   f"{used:.1f}/{tot:.1f} GB")
    t.add_row("ETA batch",      str(timedelta(seconds=state.get("eta_batch", 0))))
    t.add_row("ETA total",      str(timedelta(seconds=state.get("eta_total", 0))))
    t.add_row("Meilleur",       f"epoch {best_ep}  score {best_score:.4f}")
    return Panel(t, title="[bold green]OrangIdentifier V6[/bold green]", border_style="green")

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_benchmark(bb, zoo_va_p, zoo_va_l, zoo_names, gallery_data, threshold, emb_dim):
    """Benchmark sur le val-set zoo avec scoring max-over-exemplaires (= logique app)."""
    section("Benchmark V6 (val-set zoo — scoring max-over-exemplaires)")
    bb.eval()

    # Embeddings requêtes
    ds = PlainDataset(zoo_va_p, zoo_va_l)
    dl = DataLoader(ds, 64, num_workers=0)
    q_embs, q_labs = [], []
    for imgs, lbs in dl:
        q_embs.append(F.normalize(bb(imgs.to(DEVICE).float()), dim=1).cpu().numpy())
        q_labs.extend(lbs.tolist())
    q_embs = np.concatenate(q_embs).astype(np.float32)
    q_labs = np.array(q_labs)

    # Matrices d'exemplaires par classe (depuis la galerie construite)
    class_exemplars = {}
    for info in gallery_data["individuals"].values():
        ci = info["class_index"]
        ex = np.array(info["exemplars"], dtype=np.float32)
        class_exemplars[ci] = ex

    n_zoo_cls = len(zoo_names)
    scores = np.full((len(q_embs), n_zoo_cls), -1.0, dtype=np.float32)
    for ci, ex in class_exemplars.items():
        if ci < n_zoo_cls:
            scores[:, ci] = (q_embs @ ex.T).max(1)

    predicted = scores.argmax(1)
    own_scores = scores[np.arange(len(q_labs)), q_labs]
    accuracy   = float((predicted == q_labs).mean())

    # Wild FP rate
    wild_files_bm = []
    if WILD_DIR.exists():
        wild_files_bm = sorted([f for f in WILD_DIR.iterdir() if f.suffix in EXTS])
    fp_rate = float("nan")
    if wild_files_bm:
        wild_sample = random.sample(wild_files_bm, min(500, len(wild_files_bm)))
        ds_w = PlainDataset(wild_sample, [0]*len(wild_sample))
        dl_w = DataLoader(ds_w, 64, num_workers=0)
        w_embs = []
        for imgs, _ in dl_w:
            w_embs.append(F.normalize(bb(imgs.to(DEVICE).float()), dim=1).cpu().numpy())
        w_embs = np.concatenate(w_embs)
        all_ex = np.vstack([ex for ex in class_exemplars.values() if ex.shape[0] > 0])
        wild_max_scores = (w_embs @ all_ex.T).max(1)
        fp_rate = float((wild_max_scores >= threshold).mean())

    # Résumé terminal
    log(f"\n  Accuracy zoo (val) : {accuracy*100:.2f}%")
    log(f"  Confiance moyenne  : {own_scores.mean():.4f}  (min {own_scores.min():.4f})")
    log(f"  Seuil galerie      : {threshold:.4f}")
    if not math.isnan(fp_rate):
        log(f"  Wild FP rate       : {fp_rate*100:.2f}%  (≤5% recommandé)")
    log("")
    log(f"  {'Individu':<14} {'N req':>6} {'Acc':>7} {'Conf µ':>8} {'Conf min':>9}")
    log(f"  {'─'*50}")
    per_ind = {}
    for i, name in enumerate(zoo_names):
        mask_i = q_labs == i
        if mask_i.sum() == 0: continue
        own_i  = own_scores[mask_i]
        acc_i  = float((predicted[mask_i] == i).mean())
        warn   = " !" if acc_i < 0.80 else ""
        log(f"  {name:<14} {mask_i.sum():>6} {acc_i*100:>6.1f}% {own_i.mean():>8.4f} {own_i.min():>8.4f}{warn}")
        per_ind[name] = {"n_queries": int(mask_i.sum()), "accuracy": round(acc_i, 4),
                         "mean_conf": round(float(own_i.mean()), 4),
                         "min_conf":  round(float(own_i.min()), 4)}

    result = {
        "accuracy": round(accuracy, 4),
        "mean_confidence": round(float(own_scores.mean()), 4),
        "wild_fp_rate": round(fp_rate, 4) if not math.isnan(fp_rate) else None,
        "threshold": round(threshold, 4),
        "n_val_queries": len(q_labs),
        "per_individual": per_ind,
    }
    return result

# ══════════════════════════════════════════════════════════════════════════════
# COURBES
# ══════════════════════════════════════════════════════════════════════════════
def make_plots(history, best_ep=None):
    n = len(history["l_arc"])
    if n == 0:
        return
    xs = list(range(1, n + 1))

    # Délimiteurs de phases (fin de chaque phase sauf la dernière)
    bdry = []
    cum  = 0
    for p in PHASES[:-1]:
        cum += p["epochs"]
        if cum < n:
            bdry.append(cum)

    def _fmt(ax, title, ylabel=None):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        for b in bdry:
            ax.axvline(b + 0.5, color="#666", lw=1.0, ls="--", alpha=0.6)
        if best_ep and 1 <= best_ep <= n:
            ax.axvline(best_ep, color="#facc15", lw=1.5, ls=":", alpha=0.9,
                       label=f"* meilleur ep.{best_ep}")
        ax.legend(fontsize=9)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("OrangIdentifier V6 — Courbes d'entraînement (Zoo uniquement)\n"
                 "(tirets gris = séparations de phases · pointillé jaune = meilleur epoch)",
                 fontsize=13)

    axes[0, 0].plot(xs, history["l_arc"], color="#e74c3c", lw=1.5, label="L_arcface")
    _fmt(axes[0, 0], "Loss ArcFace")

    axes[0, 1].plot(xs, history["l_inv"], color="#3498db", lw=1.5, label="L_invariance")
    _fmt(axes[0, 1], "Loss Invariance")

    axes[1, 0].plot(xs, [v*100 for v in history["acc_clean"]],
                    color="#2ecc71", lw=1.5, label="Zoo propre")
    axes[1, 0].plot(xs, [v*100 for v in history["acc_deg"]],
                    color="#f39c12", lw=1.5, label="Zoo dégradé")
    axes[1, 0].set_ylim(0, 105)
    _fmt(axes[1, 0], "Accuracy zoo (validation)", ylabel="%")

    axes[1, 1].plot(xs, history["composite"], color="#1abc9c", lw=1.5, label="Composite")
    if best_ep and 1 <= best_ep <= n:
        bv = history["composite"][best_ep - 1]
        axes[1, 1].scatter([best_ep], [bv], color="#facc15", s=90, zorder=5)
    _fmt(axes[1, 1], "Score composite (0.45·acc_propre + 0.55·acc_dégradé)")

    plt.tight_layout()
    plt.savefig(CURVES_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Courbes → {CURVES_PNG.name}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()
    section(f"V6 TRAIN {'[DRY RUN] ' if DRY else ''}— {datetime.now():%Y-%m-%d %H:%M}")
    log(f"  Device : {DEVICE}")

    # ── Données ───────────────────────────────────────────────────────────────
    section("Chargement des données")
    zoo_p, zoo_l, zoo_names = load_zoo_merged(
        [ZOO_DIR, NEW_ZOO], offset=0, exclude=["_a_verifier"]
    )
    n_zoo = len(zoo_names)
    log(f"  Zoo : {n_zoo} individus (10 existants + nouveaux), {len(zoo_p):,} crops")
    log(f"  Individus zoo : {zoo_names}")

    all_names   = zoo_names
    n_known     = n_zoo
    unknown_l   = n_zoo    # classe wild = index juste après les zoo
    all_known_p = zoo_p
    all_known_l = zoo_l

    wild_files = []
    if WILD_DIR.exists():
        wild_files = sorted([f for f in WILD_DIR.iterdir() if f.suffix in EXTS])
        log(f"  Wild : {len(wild_files):,} crops")

    from sklearn.model_selection import train_test_split
    idx = list(range(len(zoo_p)))
    try:
        tr_idx, va_idx = train_test_split(
            idx, test_size=VAL_RATIO, stratify=zoo_l, random_state=SEED)
    except ValueError:
        log("  [WARN] Stratify impossible (individu avec < 2 crops) — split aléatoire", "WARN")
        tr_idx, va_idx = train_test_split(idx, test_size=VAL_RATIO, random_state=SEED)
    zoo_tr_p = [zoo_p[i] for i in tr_idx]; zoo_tr_l = [zoo_l[i] for i in tr_idx]
    zoo_va_p = [zoo_p[i] for i in va_idx]; zoo_va_l = [zoo_l[i] for i in va_idx]

    # ── Backbone + ArcFace ────────────────────────────────────────────────────
    section("Backbone + ArcFace")
    backbone, emb_dim, ckpt_src = load_backbone()
    backbone = backbone.to(DEVICE)

    k_per_class = [K_ZOO]*n_zoo + [K_WILD]
    arc_loss    = SubCenterArcFace(emb_dim, n_zoo+1, k_per_class,
                                   ARC_SCALE, ARC_MARGIN).to(DEVICE)
    log(f"  ArcFace : {n_zoo} zoo + 1 wild = {n_zoo+1} classes")
    log(f"  Sous-centres : zoo×{K_ZOO} + wild×{K_WILD}")

    # ── Reprise checkpoint ────────────────────────────────────────────────────
    start_phase = 0; start_ep_in = 0; global_ep = 0
    best_val = -999.0; best_ep = 0
    history = {"l_arc":[],"l_inv":[],"acc_clean":[],"acc_deg":[],"composite":[]}

    if RESUME_PT.exists():
        section("Reprise depuis checkpoint")
        ck = torch.load(str(RESUME_PT), map_location=DEVICE, weights_only=False)
        backbone.load_state_dict(ck["backbone_state"])
        arc_loss.load_state_dict(ck["arc_loss_state"])
        start_phase = ck["phase_idx"]; start_ep_in = ck["ep_in_phase"] + 1
        global_ep   = ck["global_ep"]; best_val = ck["best_val"]; best_ep = ck["best_ep"]
        history     = ck.get("history", history)
        log(f"  Reprise phase {start_phase}, epoch {start_ep_in}, meilleur={best_val:.4f}")
        if start_ep_in >= PHASES[start_phase]["epochs"]:
            start_phase += 1; start_ep_in = 0
            if start_phase >= len(PHASES):
                log("  Toutes les phases terminées — galerie uniquement")
                backbone.load_state_dict(
                    torch.load(str(BEST_PT), map_location=DEVICE, weights_only=False)["backbone_state"])
                build_gallery(backbone, all_known_p, all_known_l, all_names, n_zoo, emb_dim)
                make_plots(history); return

    # ── Crash handler ─────────────────────────────────────────────────────────
    _cur = {"bb": backbone, "arc": arc_loss, "opt": None, "sched": None,
            "phase": start_phase, "ep_in": start_ep_in, "global": global_ep}

    def _emergency(reason="interrupt"):
        log(f"  [SAVE-URGENCE] phase={_cur['phase']} ep={_cur['ep_in']}", "WARN")
        opt_  = _cur["opt"] or _mk_opt(PHASES[_cur["phase"]])
        save_resume(_cur["bb"], _cur["arc"], opt_, _cur["sched"],
                    _cur["phase"], _cur["ep_in"]-1, _cur["global"]-1,
                    best_val, best_ep, history, all_names)
    global _save_fn; _save_fn = _emergency

    def _mk_opt(pc):
        pg = [{"params": arc_loss.parameters(), "lr": pc["lr_h"]}]
        if not pc["freeze"]:
            pg.insert(0, {"params": backbone.parameters(), "lr": pc["lr_bb"]})
        return optim.AdamW(pg, weight_decay=1e-4)

    def _mk_sched(opt, pc, ep_done):
        total = pc["epochs"]; warmup = min(2, total//4)
        def _lr(step):
            if step < warmup: return (step+1)/max(warmup,1)
            prog = (step-warmup)/max(total-warmup,1)
            return max(0.01, 0.5*(1+math.cos(math.pi*prog)))
        sched = optim.lr_scheduler.LambdaLR(opt, _lr)
        for _ in range(ep_done): sched.step()
        return sched

    # ── Boucle de phases ──────────────────────────────────────────────────────
    live_state = {}; proto_matrix_np = None

    for phase_idx in range(start_phase, len(PHASES)):
        pc           = PHASES[phase_idx]
        start_ep_i   = start_ep_in if phase_idx == start_phase else 0
        patience_c   = 0
        section(pc["name"])

        for p in backbone.parameters(): p.requires_grad_(not pc["freeze"])
        opt   = _mk_opt(pc)
        sched = _mk_sched(opt, pc, start_ep_i)
        _cur["opt"] = opt; _cur["sched"] = sched

        if phase_idx == start_phase and RESUME_PT.exists() and start_ep_in > 0:
            ck2 = torch.load(str(RESUME_PT), map_location="cpu", weights_only=False)
            try:
                opt.load_state_dict(ck2["optimizer_state"])
                if ck2.get("scheduler_state") and sched:
                    sched.load_state_dict(ck2["scheduler_state"])
                log("  Optimizer/scheduler restaurés")
            except Exception as e:
                log(f"  Optimizer reparti de zéro ({e})", "WARN")

        live_ctx = Live(console=_console, refresh_per_second=4) if RICH else None
        if live_ctx: live_ctx.start()

        _bv = [best_val]; _be = [best_ep]; _ge = [global_ep]
        refresh_fn = (lambda: live_ctx.update(make_rich_panel(
            live_state, pc, _ge[0], _be[0], _bv[0]))) if live_ctx else None

        for ep in range(start_ep_i, pc["epochs"]):
            if _interrupt: break
            global_ep += 1
            _cur["phase"] = phase_idx; _cur["ep_in"] = ep; _cur["global"] = global_ep
            live_state["ep_in_phase"] = ep + 1
            _ge[0] = global_ep; _bv[0] = best_val; _be[0] = best_ep

            # Wild loader (avec hard mining optionnel)
            wild_dl_cur = None
            if pc["wild"] > 0 and wild_files:
                wild_ds = WildDataset(WILD_DIR, pc["wild"], unknown_l)
                if phase_idx >= WILD_HARD_MINING_FROM_PHASE and proto_matrix_np is not None:
                    ww = compute_wild_weights(backbone, wild_files, proto_matrix_np)
                    if ww is not None: wild_ds._resample(pc["wild"], ww)
                wild_dl_cur = DataLoader(wild_ds, BATCH, shuffle=True,
                                         num_workers=2, pin_memory=True)

            zoo_ds   = PairDataset(zoo_tr_p, zoo_tr_l, pc["severity"])
            zoo_w    = [1.0/Counter(zoo_tr_l)[l] for l in zoo_tr_l]
            zoo_dl   = DataLoader(zoo_ds, BATCH,
                                  sampler=WeightedRandomSampler(zoo_w, len(zoo_w), True),
                                  num_workers=2, pin_memory=True)

            elapsed = time.time() - t_start
            ep_done = sum(PHASES[i]["epochs"] for i in range(phase_idx)) + ep
            live_state["eta_total"] = int(elapsed/max(ep_done,1)*(TOTAL_EPOCHS-ep_done))

            l_arc, l_inv = train_epoch(
                backbone, arc_loss, [zoo_dl, wild_dl_cur],
                opt, sched, pc, n_zoo, live_state, refresh_fn)
            if sched: sched.step()

            acc_c, acc_d = validate(backbone, zoo_tr_p, zoo_tr_l,
                                     zoo_va_p, zoo_va_l, n_zoo, emb_dim)
            # Mise à jour des centroïdes pour le hard mining wild de la prochaine epoch.
            if phase_idx >= WILD_HARD_MINING_FROM_PHASE:
                proto_matrix_np = build_proto_np(
                    backbone, zoo_tr_p, zoo_tr_l, n_zoo, emb_dim)

            composite = 0.45*acc_c + 0.55*acc_d

            is_best = composite > best_val
            if is_best:
                best_val = composite; best_ep = global_ep
                save_best(backbone, arc_loss, all_names, emb_dim, global_ep, composite, ckpt_src)

            for k, v in zip(["l_arc","l_inv","acc_clean","acc_deg","composite"],
                             [l_arc, l_inv, acc_c, acc_d, composite]):
                history[k].append(v)

            live_state.update({"acc_c": acc_c, "acc_d": acc_d,
                               "composite": composite, "is_best": is_best})
            save_resume(backbone, arc_loss, opt, sched,
                        phase_idx, ep, global_ep, best_val, best_ep, history, all_names)

            star = " *" if is_best else ""
            log(f"  Ep {global_ep:3d} | arc={l_arc:.3f} inv={l_inv:.3f} "
                f"| zoo={acc_c*100:.1f}%/dég={acc_d*100:.1f}% "
                f"| composite={composite:.4f}{star}")
            if RICH and live_ctx:
                live_ctx.update(make_rich_panel(live_state, pc, global_ep, best_ep, best_val))

            if pc.get("early_stop") and ep >= min(10, pc["epochs"]//3):
                if not is_best: patience_c += 1
                else:           patience_c  = 0
                if patience_c >= pc["patience"]:
                    log(f"  Early stopping epoch {global_ep}"); break

        if live_ctx:
            try: live_ctx.stop()
            except Exception: pass
        if _interrupt: break

    # ── Post-training ─────────────────────────────────────────────────────────
    section("Post-training")
    if BEST_PT.exists():
        ck = torch.load(str(BEST_PT), map_location=DEVICE, weights_only=False)
        backbone.load_state_dict(ck["backbone_state"])
        log(f"  Meilleur chargé — epoch {ck['epoch']} score {ck['val_composite']:.4f}")

    opt_t, gap, proto_matrix_np = build_gallery(
        backbone, all_known_p, all_known_l, all_names, n_zoo, emb_dim)

    # Benchmark sur le val-set zoo (scoring max-over-exemplaires = logique app)
    gallery_data = json.loads(GALLERY_JS.read_text())
    bench = run_benchmark(backbone, zoo_va_p, zoo_va_l, zoo_names,
                          gallery_data, opt_t, emb_dim)

    if history["l_arc"]: make_plots(history, best_ep=best_ep)

    total_time = time.time() - t_start
    report = {
        "version": "v6", "generated": datetime.now().isoformat(),
        "training_min": round(total_time/60, 1), "dry_run": DRY,
        "best_epoch": best_ep, "best_composite": round(best_val, 4),
        "gallery_threshold": round(opt_t, 4), "separability_gap": round(gap, 4),
        "n_zoo": n_zoo, "zoo_classes": zoo_names,
        "backbone_src": ckpt_src,
        "benchmark": bench,
        "hyperparams": {"arc_scale": ARC_SCALE, "arc_margin": ARC_MARGIN,
                        "k_zoo": K_ZOO, "k_wild": K_WILD,
                        "k_exemplars_zoo": K_EXEMPLARS_ZOO,
                        "batch": BATCH, "total_epochs": TOTAL_EPOCHS},
    }
    REPORT_JS.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    section("DONE")
    log(f"""
  Score composite meilleur : {best_val:.4f}  (epoch {best_ep})
  Séparabilité gap         : {gap:.4f}
  Seuil galerie            : {opt_t:.4f}
  Individus zoo            : {n_zoo}
  Durée totale             : {total_time/60:.1f} min

  Modèle   : {BEST_PT.name}
  Galerie  : {GALLERY_JS.name}
  Rapport  : {REPORT_JS.name}
  Courbes  : {CURVES_PNG.name}

  ÉTAPE SUIVANTE :
    python v6_megadesc_arcface_15ind/03_tune_threshold.py
""")
    _log_fh.close()

if __name__ == "__main__":
    main()