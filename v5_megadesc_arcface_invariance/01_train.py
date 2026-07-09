"""
01_train.py — OrangIdentifier V5
=============================================
Départ : V3 (meilleure base — séparabilité 0.96, ROC 0.998)

Innovations vs tous les scripts précédents :
  [1] L_invariance : force le modèle à produire des embeddings
      stables sous dégradation (fix direct du bug screenshot 97%→0%)
      → chaque image passe en double (propre + dégradée), les deux
        embeddings sont poussés à être similaires
  [2] L_bos_spread : pousse les individus BOS à s'éloigner les uns
      des autres dans l'espace d'embedding
      → adresse directement la figure 1 (bloc BOS rouge uniforme)
  [3] 4 phases avec curriculum dégradation progressif
      → severity 0.0→0.4→0.7→1.0 selon la phase
  [4] Sauvegarde checkpoint après CHAQUE epoch, inconditionnellement
      → fermer la fenêtre = relancer = ça reprend exactement où ça s'est arrêté
  [5] Handler Win32 CTRL_CLOSE_EVENT pour fermeture fenêtre terminal
  [6] Galerie full-exemplaires filtrés par qualité (pas de prototype moyen)
  [7] Stats live rich : losses, val, GPU, ETA batch, ETA total

CRASH-SAFETY
  Peu importe comment le process s'arrête (fermeture fenêtre, crash, coupure
  courant), le script sauvegarde v5_resume.pt à la fin de chaque epoch.
  Relancer = reprise automatique depuis la dernière epoch complète.

CHEMINS
  Starting brain    : models/megadesc_T_arcface_final_epoch21_acc99.pt (V3)
  Outputs           : output/v5/

INSTALLATION (si rich/pywin32 manquants)
  pip install rich pywin32

RUN
  conda activate orangs
  python v5_megadesc_arcface_invariance/01_train.py
  python v5_megadesc_arcface_invariance/01_train.py --dry-run
"""

# ── Cache avant tout import HuggingFace/torch ─────────────────────────────────
import os
os.environ["HF_HOME"]              = r"D:\HuggingFaceCache"
os.environ["TORCH_HOME"]           = r"D:\TorchCache"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys, io, json, math, time, signal, random, shutil, warnings, argparse
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

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TimeRemainingColumn, TextColumn
    from rich.layout import Layout
    from rich.text import Text
    RICH = True
except ImportError:
    RICH = False

# ══════════════════════════════════════════════════════════════════════════════
# ARGS
# ══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true", help="3 epochs de test")
ARGS   = parser.parse_args()
DRY    = ARGS.dry_run

# ══════════════════════════════════════════════════════════════════════════════
# CHEMINS
# ══════════════════════════════════════════════════════════════════════════════
# Portable paths: everything is relative to the repository root.
REPO      = Path(__file__).resolve().parents[1]
V3_CKPT   = REPO / "models" / "megadesc_T_arcface_final_epoch21_acc99.pt"
ZOO_DIR   = REPO / "data" / "crops" / "known"   # the 10 zoo individuals
BOS_DIR   = REPO / "data" / "crops" / "bos"     # the 30 rescue-center individuals
WILD_DIR  = REPO / "data" / "crops" / "wild"    # background (unknown) crops

OUT       = REPO / "output" / "v5"
MODELS    = OUT / "models";  MODELS.mkdir(parents=True, exist_ok=True)
RESULTS   = OUT / "results"; RESULTS.mkdir(parents=True, exist_ok=True)
LOGS      = OUT / "logs";    LOGS.mkdir(parents=True, exist_ok=True)

BEST_PT     = MODELS / "v5_best.pt"
RESUME_PT   = MODELS / "v5_resume.pt"
BACKBONE_PT = MODELS / "v5_backbone_only.pt"
GALLERY_JS  = MODELS / "v5_gallery.json"
REPORT_JS   = RESULTS / "v5_report.json"
CURVES_PNG  = RESULTS / "v5_curves.png"
LOG_FILE    = LOGS    / "training.log"

# ══════════════════════════════════════════════════════════════════════════════
# HYPER-PARAMÈTRES
# ══════════════════════════════════════════════════════════════════════════════
IMG_SIZE   = 224
BATCH      = 32
SEED       = 42
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = STD = [0.5, 0.5, 0.5]
EXTS       = {".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"}

ARC_SCALE  = 64
ARC_MARGIN = 0.35      # réduit vs 0.50 — plus stable avec peu de données BOS
K_ZOO      = 2         # sous-centres par individu zoo
K_BOS      = 1         # 1 seul : BOS a trop peu de crops pour en justifier plus
K_WILD     = 5

# Galerie
K_EXEMPLARS_ZOO = 20
K_EXEMPLARS_BOS = 15
QUALITY_ZOO     = 0.60   # seuil sim vs centroïde pour garder un crop
QUALITY_BOS     = 0.50

VAL_RATIO  = 0.15
WILD_HARD_MINING_FROM_PHASE = 2   # mining démo. actif seulement à partir de la phase 2

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION DES 4 PHASES
# ══════════════════════════════════════════════════════════════════════════════
PHASES = [
    dict(name="Phase 0 — Initialisation", epochs=3 if not DRY else 1,
         freeze=True,  lr_bb=0.0,   lr_h=1e-3,
         lam_inv=0.00, lam_bos=0.00, severity=0.00, wild=0),
    dict(name="Phase 1 — Warmup",         epochs=15 if not DRY else 1,
         freeze=False, lr_bb=1e-7,  lr_h=5e-4,
         lam_inv=0.10, lam_bos=0.00, severity=0.40, wild=0),
    dict(name="Phase 2 — Apprentissage",  epochs=20 if not DRY else 1,
         freeze=False, lr_bb=5e-6,  lr_h=3e-4,
         lam_inv=0.30, lam_bos=0.10, severity=0.70, wild=1200),
    dict(name="Phase 3 — Consolidation", epochs=15 if not DRY else 1,
         freeze=False, lr_bb=2e-6,  lr_h=1e-4,
         lam_inv=0.50, lam_bos=0.20, severity=1.00, wild=1500,
         early_stop=True, patience=15),
]
TOTAL_EPOCHS = sum(p["epochs"] for p in PHASES)

# ══════════════════════════════════════════════════════════════════════════════
# LOGGER
# ══════════════════════════════════════════════════════════════════════════════
_log_fh = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
_console = Console() if RICH else None

def log(msg="", level="INFO"):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}][{level}] {msg}"
    _log_fh.write(line + "\n"); _log_fh.flush()
    if not RICH or level in ("ERROR", "WARN"):
        print(line)

def section(t):
    bar = "─" * 68
    log(""); log(bar); log(f"  {t}"); log(bar)

# ══════════════════════════════════════════════════════════════════════════════
# CRASH SAFETY — fermeture fenêtre + Ctrl+C
# ══════════════════════════════════════════════════════════════════════════════
_interrupt   = False
_save_fn     = None   # sera défini dans main()

def _emergency_save():
    if _save_fn is not None:
        try:
            _save_fn(reason="interrupt")
        except Exception as e:
            log(f"Emergency save failed: {e}", "ERROR")

def _sigint(sig, frame):
    global _interrupt
    if not _interrupt:
        log("\n  [Ctrl+C] Arrêt propre après le batch courant...", "WARN")
        _interrupt = True
    else:
        _emergency_save()
        _log_fh.close()
        sys.exit(1)

signal.signal(signal.SIGINT, _sigint)

try:
    import win32api
    def _win32_handler(ctrl_type):
        # CTRL_C=0 CTRL_BREAK=1 CTRL_CLOSE=2 LOGOFF=5 SHUTDOWN=6
        if ctrl_type in (2, 5, 6):
            log(f"  [Win32 CTRL_CLOSE={ctrl_type}] Fenêtre fermée — sauvegarde urgence...", "WARN")
            global _interrupt; _interrupt = True
            _emergency_save()
            time.sleep(1)    # laisser le temps à la sauvegarde
        return False         # False = laisser le handler par défaut agir aussi
    win32api.SetConsoleCtrlHandler(_win32_handler, True)
    log("  Handler Win32 CTRL_CLOSE installé — fermeture fenêtre = sauvegarde auto")
except ImportError:
    log("  pywin32 absent — sauvegarde par epoch uniquement (pip install pywin32)", "WARN")

# ══════════════════════════════════════════════════════════════════════════════
# SUB-CENTER ARCFACE
# ══════════════════════════════════════════════════════════════════════════════
class SubCenterArcFace(nn.Module):
    def __init__(self, emb_dim, num_classes, k_per_class, scale=64.0, margin=0.35):
        super().__init__()
        self.scale, self.margin = scale, margin
        self.num_classes  = num_classes
        self.k_per_class  = k_per_class
        total_k           = sum(k_per_class)
        self.weight       = nn.Parameter(torch.FloatTensor(total_k, emb_dim))
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer("cls_start",
            torch.tensor([sum(k_per_class[:i]) for i in range(num_classes)], dtype=torch.long))
        self.register_buffer("cls_k",
            torch.tensor(k_per_class, dtype=torch.long))
        self.cos_m = math.cos(margin); self.sin_m = math.sin(margin)
        self.th    = math.cos(math.pi - margin)
        self.mm    = math.sin(math.pi - margin) * margin

    def forward(self, emb, labels):
        w    = F.normalize(self.weight, dim=1)
        ca   = emb @ w.T                                          # [B, total_K]
        logits = torch.zeros(emb.size(0), self.num_classes, device=emb.device)
        for c in range(self.num_classes):
            s = self.cls_start[c].item(); k = self.cls_k[c].item()
            logits[:, c] = ca[:, s:s+k].max(dim=1).values
        cos  = logits.clamp(-1.0, 1.0)
        sin  = (1.0 - cos**2).clamp(0.0, 1.0).sqrt()
        phi  = cos * self.cos_m - sin * self.sin_m
        phi  = torch.where(cos > self.th, phi, cos - self.mm)
        one_hot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1.0)
        out  = (one_hot * phi + (1 - one_hot) * logits) * self.scale
        return F.cross_entropy(out, labels, label_smoothing=0.05)

# ══════════════════════════════════════════════════════════════════════════════
# LOSSES AUXILIAIRES
# ══════════════════════════════════════════════════════════════════════════════
def loss_invariance(emb_clean, emb_deg):
    """Force same individual clean/dégradé → embedding similaire.
    Fix direct du bug screenshot : Sinta_propre et Sinta_screenshot
    produiront des embeddings proches."""
    return 1.0 - (emb_clean * emb_deg).sum(dim=1).mean()

def loss_bos_spread(emb, labels, n_zoo, n_known):
    """Repousse les individus BOS les uns des autres.
    Adresse directement le bloc rouge de la figure 1."""
    bos_mask = (labels >= n_zoo) & (labels < n_known)
    if bos_mask.sum() < 4:
        return torch.tensor(0.0, device=emb.device)
    be   = emb[bos_mask]
    bl   = labels[bos_mask]
    sim  = be @ be.T
    diff = (bl.unsqueeze(0) != bl.unsqueeze(1))
    if not diff.any():
        return torch.tensor(0.0, device=emb.device)
    return sim[diff].mean()   # minimiser = repousser les BOS différents

# ══════════════════════════════════════════════════════════════════════════════
# AUGMENTATIONS — 3 niveaux
# ══════════════════════════════════════════════════════════════════════════════
class _LowRes:
    def __init__(self, min_f, max_f, p): self.min_f=min_f; self.max_f=max_f; self.p=p
    def __call__(self, img):
        if random.random() > self.p: return img
        w,h = img.size; f = random.uniform(self.min_f, self.max_f)
        s   = max(int(w*f),8), max(int(h*f),8)
        return img.resize(s, Image.BILINEAR).resize((w,h), Image.BICUBIC)

class _JPEG:
    def __init__(self, min_q, max_q, p): self.min_q=min_q; self.max_q=max_q; self.p=p
    def __call__(self, img):
        if random.random() > self.p: return img
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=random.randint(self.min_q, self.max_q))
        buf.seek(0); return Image.open(buf).copy()

class _CropJitter:
    """Décalage ±px avec padding reflect — simule variabilité YOLO bbox."""
    def __init__(self, px=2): self.px=px
    def __call__(self, tensor):
        pad = T.functional.pad(tensor, self.px, padding_mode="reflect")
        i   = random.randint(0, 2*self.px); j = random.randint(0, 2*self.px)
        return pad[:, i:i+tensor.shape[1], j:j+tensor.shape[2]]

_norm = T.Normalize(MEAN, STD)

def get_clean_tf():
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(10),
        T.ColorJitter(0.2, 0.2, 0.1, 0.05),
        T.ToTensor(), _norm,
    ])

def get_degraded_tf(severity: float):
    """severity ∈ [0,1] — curriculum : 0.4 → 0.7 → 1.0 selon la phase."""
    s = severity
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(max(0.50, 0.75-0.25*s), 1.0)),
        T.RandomHorizontalFlip(p=0.5),
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
    """Retourne (clean, degraded, label) — un seul forward pass suffit via cat."""
    def __init__(self, paths, labels, severity=1.0):
        self.paths=paths; self.labels=labels; self.severity=severity
    def __len__(self): return len(self.paths)
    def _load(self, idx):
        try:    return Image.open(self.paths[idx]).convert("RGB")
        except: return Image.new("RGB",(IMG_SIZE,IMG_SIZE),(128,128,128))
    def __getitem__(self, idx):
        img = self._load(idx)
        return get_clean_tf()(img), get_degraded_tf(self.severity)(img), int(self.labels[idx])

class PlainDataset(Dataset):
    """Dataset simple (val, galerie)."""
    def __init__(self, paths, labels, tf=None):
        self.paths=paths; self.labels=labels; self.tf=tf or get_val_tf()
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        try:    img = Image.open(self.paths[idx]).convert("RGB")
        except: img = Image.new("RGB",(IMG_SIZE,IMG_SIZE),(128,128,128))
        return self.tf(img), int(self.labels[idx])

class WildDataset(Dataset):
    def __init__(self, wild_dir, n, unknown_label):
        self.all = sorted([f for f in wild_dir.iterdir() if f.suffix in EXTS])
        self.lbl = unknown_label; self._resample(n)
    def _resample(self, n, weights=None):
        n = min(n, len(self.all))
        if weights is not None and len(weights)==len(self.all):
            idx = np.random.choice(len(self.all), n, replace=False, p=weights)
            self.files = [self.all[i] for i in idx]
        else:
            self.files = random.sample(self.all, n)
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        try:    img = Image.open(self.files[idx]).convert("RGB")
        except: img = Image.new("RGB",(IMG_SIZE,IMG_SIZE),(100,80,60))
        # wild reçoit seulement la version dégradée (plus de diversité)
        deg = get_degraded_tf(1.0)(img)
        cln = get_clean_tf()(img)
        return cln, deg, self.lbl

# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════════════════════════════════════
def load_dir(base, offset=0, exclude=None):
    excl  = set(exclude or [])
    dirs  = sorted([d for d in base.iterdir()
                    if d.is_dir() and not d.name.startswith("_") and d.name not in excl])
    paths, labels, names = [], [], []
    for i, d in enumerate(dirs):
        imgs = [f for f in d.iterdir() if f.suffix in EXTS]
        for f in sorted(imgs):
            paths.append(f); labels.append(i+offset)
        names.append(d.name)
    return paths, labels, names

def build_loaders(zoo_tr_p, zoo_tr_l, bos_p, bos_l, unknown_lbl, severity, wild_n):
    from sklearn.model_selection import train_test_split

    # Zoo
    zoo_ds   = PairDataset(zoo_tr_p, zoo_tr_l, severity)
    zoo_w    = [1.0/Counter(zoo_tr_l)[l] for l in zoo_tr_l]
    zoo_dl   = DataLoader(zoo_ds, BATCH,
                          sampler=WeightedRandomSampler(zoo_w, len(zoo_w), True),
                          num_workers=2, pin_memory=True)
    # BOS
    bos_dl = None
    if bos_p:
        bos_ds = PairDataset(bos_p, bos_l, severity)
        bos_w  = [1.0/Counter(bos_l)[l] for l in bos_l]
        bos_dl = DataLoader(bos_ds, BATCH,
                            sampler=WeightedRandomSampler(bos_w, len(bos_w), True),
                            num_workers=2, pin_memory=True)
    # Wild
    wild_dl = None
    if wild_n > 0 and WILD_DIR.exists():
        wild_ds = WildDataset(WILD_DIR, wild_n, unknown_lbl)
        wild_dl = DataLoader(wild_ds, BATCH, shuffle=True,
                             num_workers=2, pin_memory=True)
    return zoo_dl, bos_dl, wild_dl

# ══════════════════════════════════════════════════════════════════════════════
# BACKBONE
# ══════════════════════════════════════════════════════════════════════════════
def load_backbone():
    bb = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224", pretrained=False, num_classes=0)
    if V3_CKPT.exists():
        ckpt  = torch.load(str(V3_CKPT), map_location="cpu", weights_only=False)
        state = None
        for k in ("backbone_state","model_state_dict","state_dict"):
            if isinstance(ckpt,dict) and k in ckpt: state=ckpt[k]; break
        if state is None and isinstance(ckpt,dict): state=ckpt
        miss,unex = bb.load_state_dict(state, strict=False)
        log(f"  V3 chargé — {len(miss)} manquantes, {len(unex)} inattendues")
        src = "V3"
    else:
        log("  V3 introuvable — poids HuggingFace bruts", "WARN")
        bb  = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224", pretrained=True, num_classes=0)
        src = "HuggingFace"
    with torch.no_grad():
        emb_dim = bb(torch.randn(1,3,IMG_SIZE,IMG_SIZE)).shape[1]
    log(f"  Backbone {src} — {emb_dim}D — {sum(p.numel() for p in bb.parameters())/1e6:.1f}M params")
    return bb, emb_dim, src

# ══════════════════════════════════════════════════════════════════════════════
# ENTRAÎNEMENT UN EPOCH
# ══════════════════════════════════════════════════════════════════════════════
def train_epoch(bb, arc, loaders, opt, sched, phase_cfg, n_zoo, n_known, live_state, refresh_fn=None):
    bb.train(); arc.train()
    lam_inv = phase_cfg["lam_inv"]
    lam_bos = phase_cfg["lam_bos"]
    tot_arc, tot_inv, tot_bos, tot_n = 0.0, 0.0, 0.0, 0

    all_loaders = [l for l in loaders if l is not None]
    total_batches = sum(len(l) for l in all_loaders)
    batch_idx = 0

    t0 = time.time()
    for loader in all_loaders:
        for clean, deg, labels in loader:
            if _interrupt: break
            clean  = clean.to(DEVICE, non_blocking=True).float()
            deg    = deg.to(DEVICE,   non_blocking=True).float()
            labels = labels.to(DEVICE, non_blocking=True).long()

            # Un seul forward pass pour les deux flux
            both   = torch.cat([clean, deg], dim=0)       # [2B, 3, H, W]
            emb_all = F.normalize(bb(both), dim=1)         # [2B, 768]
            B       = clean.size(0)
            emb_c   = emb_all[:B]
            emb_d   = emb_all[B:]

            l_arc = arc(emb_c, labels) + arc(emb_d, labels)
            l_inv = loss_invariance(emb_c, emb_d)
            l_bos = loss_bos_spread(emb_c, labels, n_zoo, n_known)
            loss  = l_arc + lam_inv*l_inv + lam_bos*l_bos

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(bb.parameters()) + list(arc.parameters()), 1.0)
            opt.step()
            if sched: sched.step()

            bs  = labels.size(0)
            tot_arc += l_arc.item()*bs; tot_inv += l_inv.item()*bs
            tot_bos += l_bos.item()*bs; tot_n   += bs
            batch_idx += 1

            # Stats live
            eta_b = (time.time()-t0)/batch_idx * (total_batches-batch_idx)
            live_state.update({
                "batch": batch_idx, "total_batches": total_batches,
                "l_arc": tot_arc/tot_n, "l_inv": tot_inv/tot_n,
                "l_bos": tot_bos/tot_n, "eta_batch": int(eta_b),
            })
            if refresh_fn:
                refresh_fn()
        if _interrupt: break

    N = max(tot_n, 1)
    return tot_arc/N, tot_inv/N, tot_bos/N

# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def validate(bb, zoo_tr_p, zoo_tr_l, zoo_va_p, zoo_va_l, n_zoo, emb_dim):
    bb.eval()
    val_tf = get_val_tf()
    deg_tf = get_degraded_tf(0.70)  # dégradation modérée fixe pour validation

    # Prototypes depuis les crops d'entraînement
    proto = torch.zeros(n_zoo, emb_dim)
    cnt   = torch.zeros(n_zoo)
    tr_dl = DataLoader(PlainDataset(zoo_tr_p, zoo_tr_l), 64, num_workers=0)
    for imgs, lbs in tr_dl:
        e = F.normalize(bb(imgs.to(DEVICE).float()), dim=1).cpu()
        for ei, li in zip(e, lbs.tolist()):
            if li < n_zoo: proto[li]+=ei; cnt[li]+=1
    for c in range(n_zoo):
        if cnt[c]>0: proto[c] = F.normalize(proto[c], dim=0)

    # Accuracy propre
    va_dl  = DataLoader(PlainDataset(zoo_va_p, zoo_va_l), 64, num_workers=0)
    ok, tot = 0, 0
    for imgs, lbs in va_dl:
        e    = F.normalize(bb(imgs.to(DEVICE).float()), dim=1).cpu()
        pred = (e @ proto.T).argmax(1)
        ok  += (pred==lbs).sum().item(); tot += lbs.size(0)
    acc_clean = ok / max(tot,1)

    # Accuracy dégradée — même val set avec dégradation
    va_deg = DataLoader(PlainDataset(zoo_va_p, zoo_va_l, deg_tf), 64, num_workers=0)
    ok, tot = 0, 0
    for imgs, lbs in va_deg:
        e    = F.normalize(bb(imgs.to(DEVICE).float()), dim=1).cpu()
        pred = (e @ proto.T).argmax(1)
        ok  += (pred==lbs).sum().item(); tot += lbs.size(0)
    acc_deg = ok / max(tot,1)

    return acc_clean, acc_deg

@torch.no_grad()
def bos_discrimination(bb, bos_p, bos_l, n_zoo):
    """Écart moyen intra-BOS moins inter-BOS — > 0 = le modèle discrimine."""
    if not bos_p: return 0.0
    bb.eval()
    ds = PlainDataset(bos_p, bos_l)
    dl = DataLoader(ds, 64, num_workers=0)
    embs, labs = [], []
    for imgs, lbs in dl:
        embs.append(F.normalize(bb(imgs.to(DEVICE).float()),dim=1).cpu())
        labs.extend(lbs.tolist())
    embs = torch.cat(embs).numpy(); labs = np.array(labs)

    intra, inter = [], []
    names_bos = sorted(set(labs))
    for i in names_bos:
        mi = embs[labs==i]
        if len(mi)<2: continue
        # intra
        for a in range(len(mi)):
            for b in range(a+1, min(len(mi),10)):
                intra.append(float(mi[a]@mi[b]))
        # inter (vs centroid des autres)
        oth = embs[labs!=i]
        if len(oth)>0:
            c_oth = oth.mean(0); c_oth /= (np.linalg.norm(c_oth)+1e-8)
            inter.append(float(mi.mean(0)@c_oth))

    if not intra or not inter: return 0.0
    return float(np.mean(intra) - np.mean(inter))

# ══════════════════════════════════════════════════════════════════════════════
# SAUVEGARDES
# ══════════════════════════════════════════════════════════════════════════════
def _atomic(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp); tmp.replace(path)

def save_resume(bb, arc, opt, sched, phase_idx, ep_in_phase, global_ep,
                best_val, best_ep, history, classes):
    _atomic({
        "backbone_state": bb.state_dict(),
        "arc_loss_state": arc.state_dict(),
        "optimizer_state": opt.state_dict(),
        "scheduler_state": sched.state_dict() if sched else None,
        "phase_idx": phase_idx, "ep_in_phase": ep_in_phase,
        "global_ep": global_ep, "best_val": best_val,
        "best_ep": best_ep, "history": history,
        "classes": classes, "saved_at": datetime.now().isoformat(),
    }, RESUME_PT)

def save_best(bb, arc, classes, emb_dim, ep, val, src):
    _atomic({
        "backbone_state": bb.state_dict(), "arc_loss_state": arc.state_dict(),
        "classes": classes, "emb_dim": emb_dim, "epoch": ep,
        "val_composite": val, "source": src, "version": "v5",
        "normalization": {"mean": MEAN, "std": STD},
    }, BEST_PT)
    _atomic({"backbone_state": bb.state_dict(), "emb_dim": emb_dim,
             "classes": classes, "version": "v5",
             "normalization": {"mean": MEAN, "std": STD}}, BACKBONE_PT)

# ══════════════════════════════════════════════════════════════════════════════
# WILD HARD NEGATIVE MINING
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def compute_wild_weights(bb, all_wild_files, proto_matrix):
    """Sample 512 wilds aléatoires, calcule leur sim max vs individus connus,
    retourne weights pour le WildDataset (plus difficile = plus souvent tiré)."""
    if len(all_wild_files) < 512: return None
    sample = random.sample(all_wild_files, 512)
    ds = PlainDataset(sample, [0]*512)
    dl = DataLoader(ds, 64, num_workers=0)
    bb.eval()
    embs = []
    for imgs, _ in dl:
        embs.append(F.normalize(bb(imgs.to(DEVICE).float()),dim=1).cpu().numpy())
    embs = np.concatenate(embs)
    max_sims = (embs @ proto_matrix.T).max(1)
    w = np.exp(max_sims / 0.3); w /= w.sum()
    # Retourne un poids pour TOUS les fichiers wilds (les 512 ont un poids, le reste ~0)
    full_weights = np.zeros(len(all_wild_files))
    wild_names   = {str(f): i for i, f in enumerate(all_wild_files)}
    for j, p in enumerate(sample):
        if str(p) in wild_names:
            full_weights[wild_names[str(p)]] = w[j]
    if full_weights.sum() < 1e-8: return None
    full_weights /= full_weights.sum()
    return full_weights

# ══════════════════════════════════════════════════════════════════════════════
# GALERIE
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def build_gallery(bb, all_paths, all_labels, all_names, n_zoo, emb_dim):
    section("Construction de la galerie (full-exemplaires)")
    bb.eval()
    ds = PlainDataset(all_paths, all_labels)
    dl = DataLoader(ds, 64, num_workers=0)
    embs, labs = [], []
    for imgs, lbs in dl:
        embs.append(F.normalize(bb(imgs.to(DEVICE).float()),dim=1).cpu().numpy())
        labs.extend(lbs.tolist())
    embs = np.concatenate(embs).astype(np.float32)
    labs = np.array(labs)
    n_known = len(all_names)
    proto_matrix = np.zeros((n_known, emb_dim), dtype=np.float32)

    individuals = {}
    pos_sims, neg_sims = [], []

    for i, name in enumerate(all_names):
        mask   = labs == i
        ei     = embs[mask]
        if len(ei) == 0:
            log(f"  [WARN] {name}: 0 crops", "WARN"); continue

        centroid = ei.mean(0); centroid /= (np.linalg.norm(centroid)+1e-8)
        proto_matrix[i] = centroid

        # Filtre qualité
        sims    = ei @ centroid
        q       = QUALITY_ZOO if i < n_zoo else QUALITY_BOS
        k_max   = K_EXEMPLARS_ZOO if i < n_zoo else K_EXEMPLARS_BOS
        good    = ei[sims >= q]
        if len(good) < 3: good = ei[np.argsort(-sims)[:max(3,k_max//2)]]

        # Top-k par similarité au centroïde
        gs    = good @ centroid
        top_k = min(k_max, len(good))
        best  = good[np.argsort(-gs)[:top_k]]
        # Normalisation
        norms = np.linalg.norm(best, axis=1, keepdims=True)
        best  = best / np.where(norms>1e-8, norms, 1)

        # Stats intra
        own_sims = (ei @ centroid).tolist()
        pos_sims.extend(own_sims)
        oth = np.delete(proto_matrix, i, axis=0)
        if len(oth): neg_sims.extend((ei @ oth.T).max(1).tolist())

        individuals[name] = {
            "class_index": i, "is_zoo": (i < n_zoo),
            "num_crops": int(len(ei)), "num_exemplars": len(best),
            "mean_intra": round(float(np.mean(own_sims)),4),
            "prototype": centroid.tolist(),   # rétrocompat app actuelle
            "exemplars": best.tolist(),        # nouvelle inférence max-sim
        }
        log(f"  {name:<14}: {len(ei):3d} crops → {len(best):2d} exemplaires")

    pos = np.array(pos_sims); neg = np.array(neg_sims) if neg_sims else np.zeros(1)
    gap = float(pos.mean()-neg.mean())
    log(f"\n  Positive : {pos.mean():.4f} ± {pos.std():.4f}")
    log(f"  Négative : {neg.mean():.4f} ± {neg.std():.4f}")
    log(f"  Gap      : {gap:.4f}")

    # Seuil global
    thresholds = np.linspace(0,1,500)
    f1s = []
    for t in thresholds:
        tp=int((pos>=t).sum()); fp=int((neg>=t).sum()); fn=int((pos<t).sum())
        p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9)
        f1s.append(2*p*r/(p+r+1e-9))
    opt_t = float(thresholds[np.argmax(f1s)])
    log(f"  Seuil optimal : {opt_t:.4f}  (F1={max(f1s):.4f})")

    gallery = {
        "version": "5-final", "created": datetime.now().isoformat(),
        "model": "MegaDescriptor-T-224 + SubCenterArcFace V5-final",
        "embedding_dim": emb_dim, "similarity_metric": "cosine",
        "normalization": "megadescriptor",
        "unknown_threshold": round(opt_t,4),
        "separability_gap": round(gap,4),
        "num_individuals": len(individuals),
        "inference_note": "score = max(dot(query, exemplar)) over all exemplars",
        "individuals": individuals,
    }
    GALLERY_JS.write_text(json.dumps(gallery, separators=(",",":"), ensure_ascii=False))
    log(f"  Galerie sauvegardée : {GALLERY_JS.name} ({GALLERY_JS.stat().st_size/1024:.0f} KB)")
    return opt_t, gap, proto_matrix

# ══════════════════════════════════════════════════════════════════════════════
# AFFICHAGE LIVE (rich ou fallback)
# ══════════════════════════════════════════════════════════════════════════════
def make_rich_table(state, phase_cfg, global_ep, best_ep, best_score):
    t = Table.grid(padding=1)
    t.add_column(style="bold cyan"); t.add_column()
    phase_name = phase_cfg["name"]
    t.add_row("Phase", phase_name)
    t.add_row("Epoch", f"{state.get('ep_in_phase','?')}/{phase_cfg['epochs']}"
              f"  (global {global_ep}/{TOTAL_EPOCHS})")
    t.add_row("Batch", f"{state.get('batch',0)}/{state.get('total_batches',1)}")
    t.add_row("L_arcface", f"{state.get('l_arc',0):.4f}")
    t.add_row("L_inv    ", f"{state.get('l_inv',0):.4f}  (λ={phase_cfg['lam_inv']:.2f})")
    t.add_row("L_bos    ", f"{state.get('l_bos',0):.4f}  (λ={phase_cfg['lam_bos']:.2f})")
    t.add_row("─"*20, "─"*30)
    t.add_row("Zoo acc (propre)", f"{state.get('acc_c',0)*100:.2f}%")
    t.add_row("Zoo acc (dégradé)", f"{state.get('acc_d',0)*100:.2f}%")
    t.add_row("BOS discrim", f"{state.get('bos_disc',0):.4f}")
    t.add_row("Score composite", f"{state.get('composite',0):.4f}"
              + (" * BEST" if state.get("is_best") else ""))
    t.add_row("─"*20, "─"*30)
    t.add_row("LR backbone", f"{phase_cfg['lr_bb']:.1e}")
    t.add_row("LR head",     f"{phase_cfg['lr_h']:.1e}")
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated()/1e9
        total = torch.cuda.get_device_properties(0).total_memory/1e9
        t.add_row("GPU VRAM", f"{used:.1f}/{total:.1f} GB")
    t.add_row("ETA batch", str(timedelta(seconds=state.get("eta_batch",0))))
    t.add_row("ETA total", str(timedelta(seconds=state.get("eta_total",0))))
    t.add_row("Best", f"epoch {best_ep}  score {best_score:.4f}")
    return Panel(t, title="[bold green]OrangIdentifier V5[/bold green]",
                 border_style="green")

# ══════════════════════════════════════════════════════════════════════════════
# COURBES
# ══════════════════════════════════════════════════════════════════════════════
def make_plots(history):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("V5 — Courbes d'entraînement", fontsize=13)
    axes[0,0].plot(history["l_arc"], color="#e74c3c", label="L_arcface")
    axes[0,0].set_title("Loss ArcFace"); axes[0,0].grid(0.3); axes[0,0].legend()
    axes[0,1].plot(history["l_inv"], color="#3498db", label="L_invariance")
    axes[0,1].set_title("Loss Invariance"); axes[0,1].grid(0.3); axes[0,1].legend()
    axes[1,0].plot([v*100 for v in history["acc_clean"]], color="#2ecc71", label="Zoo propre")
    axes[1,0].plot([v*100 for v in history["acc_deg"]],   color="#f39c12", label="Zoo dégradé")
    axes[1,0].set_title("Accuracy validation"); axes[1,0].grid(0.3)
    axes[1,0].set_ylabel("%"); axes[1,0].legend()
    axes[1,1].plot(history["bos_disc"], color="#9b59b6", label="BOS discrimination")
    axes[1,1].set_title("BOS discrimination (>0=bon)"); axes[1,1].grid(0.3); axes[1,1].legend()
    plt.tight_layout(); plt.savefig(CURVES_PNG, dpi=150, bbox_inches="tight"); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()
    section(f"V5 {'[DRY RUN] ' if DRY else ''}— démarrage {datetime.now():%Y-%m-%d %H:%M}")
    log(f"  Device : {DEVICE}")
    log(f"  Sorties: {OUT}")

    # ── Données ───────────────────────────────────────────────────────────────
    section("Chargement des données")
    zoo_p, zoo_l_local, zoo_names = load_dir(ZOO_DIR, exclude=["_a_verifier"])
    n_zoo = len(zoo_names)
    log(f"  Zoo : {n_zoo} individus, {len(zoo_p):,} crops")
    bos_p, bos_l_local, bos_names = ([], [], [])
    if BOS_DIR.exists():
        bos_p, bos_l_local, bos_names = load_dir(BOS_DIR, offset=n_zoo)
        log(f"  BOS : {len(bos_names)} individus, {len(bos_p):,} crops")
    all_names  = zoo_names + bos_names
    n_known    = len(all_names)
    unknown_l  = n_known
    bos_l_flat = bos_l_local   # déjà offset

    from sklearn.model_selection import train_test_split
    idx = list(range(len(zoo_p)))
    tr_idx, va_idx = train_test_split(idx, test_size=VAL_RATIO, stratify=zoo_l_local, random_state=SEED)
    zoo_tr_p = [zoo_p[i] for i in tr_idx]; zoo_tr_l = [zoo_l_local[i] for i in tr_idx]
    zoo_va_p = [zoo_p[i] for i in va_idx]; zoo_va_l = [zoo_l_local[i] for i in va_idx]
    all_known_p = zoo_p + bos_p
    all_known_l = zoo_l_local + bos_l_flat

    wild_files = []
    if WILD_DIR.exists():
        wild_files = sorted([f for f in WILD_DIR.iterdir() if f.suffix in EXTS])
        log(f"  Wild: {len(wild_files):,} crops")

    # ── Backbone ──────────────────────────────────────────────────────────────
    section("Backbone + ArcFace")
    backbone, emb_dim, ckpt_src = load_backbone()
    backbone = backbone.to(DEVICE)

    k_per_class = [K_ZOO]*n_zoo + [K_BOS]*len(bos_names) + [K_WILD]
    arc_loss    = SubCenterArcFace(emb_dim, n_known+1, k_per_class,
                                   ARC_SCALE, ARC_MARGIN).to(DEVICE)
    log(f"  ArcFace : {n_known} connus + 1 wild = {n_known+1} classes")
    log(f"  Sous-centres : zoo×{K_ZOO} + BOS×{K_BOS} + wild×{K_WILD}")

    # ── Reprise ───────────────────────────────────────────────────────────────
    start_phase  = 0
    start_ep_in  = 0
    global_ep    = 0
    best_val     = -999.0
    best_ep      = 0
    history      = {"l_arc":[],"l_inv":[],"l_bos":[],"acc_clean":[],"acc_deg":[],"bos_disc":[],"composite":[]}

    if RESUME_PT.exists():
        section("Reprise depuis checkpoint")
        ck = torch.load(str(RESUME_PT), map_location=DEVICE, weights_only=False)
        backbone.load_state_dict(ck["backbone_state"])
        arc_loss.load_state_dict(ck["arc_loss_state"])
        start_phase = ck["phase_idx"]
        start_ep_in = ck["ep_in_phase"] + 1
        global_ep   = ck["global_ep"]
        best_val    = ck["best_val"]
        best_ep     = ck["best_ep"]
        history     = ck.get("history", history)
        log(f"  Reprise phase {start_phase}, epoch {start_ep_in}, "
            f"meilleur score={best_val:.4f}")
        # Si ep_in_phase dépasse les epochs de la phase → passer à la suivante
        if start_ep_in >= PHASES[start_phase]["epochs"]:
            start_phase += 1; start_ep_in = 0
            if start_phase >= len(PHASES):
                log("  Toutes les phases terminées — passage direct à la galerie")
                backbone.load_state_dict(
                    torch.load(str(BEST_PT), map_location=DEVICE, weights_only=False)["backbone_state"])
                build_gallery(backbone, all_known_p, all_known_l, all_names, n_zoo, emb_dim)
                make_plots(history)
                return

    # ── Enregistrement de la fonction de sauvegarde d'urgence ─────────────────
    _cur = {"bb": backbone, "arc": arc_loss, "opt": None, "sched": None,
            "phase": start_phase, "ep_in": start_ep_in, "global": global_ep}

    def _emergency(reason="interrupt"):
        log(f"  [SAVE-URGENCE] raison={reason} phase={_cur['phase']} ep={_cur['ep_in']}", "WARN")
        save_resume(_cur["bb"], _cur["arc"], _cur["opt"] or _mk_opt(PHASES[_cur["phase"]]),
                    _cur["sched"], _cur["phase"], _cur["ep_in"]-1,
                    _cur["global"]-1, best_val, best_ep, history, all_names)
        log("  [SAVE-URGENCE] checkpoint sauvegardé ✓", "WARN")

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
        # Avancer le scheduler si on reprend en milieu de phase
        for _ in range(ep_done): sched.step()
        return sched

    # ── Boucle de phases ──────────────────────────────────────────────────────
    live_state = {}
    console    = Console() if RICH else None

    proto_matrix_np = None   # sera rempli après la première validation complète

    for phase_idx in range(start_phase, len(PHASES)):
        pc         = PHASES[phase_idx]
        start_ep_i = start_ep_in if phase_idx == start_phase else 0
        patience_c = 0

        section(pc["name"])

        # Freeze / unfreeze backbone
        for p in backbone.parameters():
            p.requires_grad_(not pc["freeze"])

        opt   = _mk_opt(pc)
        sched = _mk_sched(opt, pc, start_ep_i)
        _cur["opt"] = opt; _cur["sched"] = sched

        # Restore optimizer si reprise dans cette phase
        if phase_idx == start_phase and RESUME_PT.exists() and start_ep_in > 0:
            ck2 = torch.load(str(RESUME_PT), map_location="cpu", weights_only=False)
            try:
                opt.load_state_dict(ck2["optimizer_state"])
                if ck2.get("scheduler_state") and sched:
                    sched.load_state_dict(ck2["scheduler_state"])
                log("  Optimizer/scheduler restaurés")
            except Exception as e:
                log(f"  [WARN] Restauration optimizer échouée ({e}) — reparti de zéro", "WARN")

        live_ctx = Live(console=console, refresh_per_second=4) if RICH else None
        if live_ctx:
            live_ctx.start()
            live_ctx.update(make_rich_table(live_state, pc, global_ep, best_ep, best_val))

        # Refresh callback — update à chaque batch (pas seulement fin d'epoch)
        _best_val_box  = [best_val]
        _best_ep_box   = [best_ep]
        _global_ep_box = [global_ep]
        if live_ctx:
            def _refresh():
                live_ctx.update(make_rich_table(
                    live_state, pc, _global_ep_box[0], _best_ep_box[0], _best_val_box[0]))
            refresh_fn = _refresh
        else:
            refresh_fn = None

        for ep in range(start_ep_i, pc["epochs"]):
            if _interrupt: break
            global_ep += 1
            _cur["phase"] = phase_idx; _cur["ep_in"] = ep; _cur["global"] = global_ep
            live_state["ep_in_phase"] = ep+1

            # Wild weights
            if pc["wild"] > 0 and wild_files:
                wild_dl_cur = DataLoader(WildDataset(WILD_DIR, pc["wild"], unknown_l),
                                         BATCH, shuffle=True, num_workers=2, pin_memory=True)
                if phase_idx >= WILD_HARD_MINING_FROM_PHASE and proto_matrix_np is not None:
                    ww = compute_wild_weights(backbone, wild_files, proto_matrix_np)
                    if ww is not None:
                        wild_ds_new = WildDataset(WILD_DIR, pc["wild"], unknown_l)
                        wild_ds_new._resample(pc["wild"], ww)
                        wild_dl_cur = DataLoader(wild_ds_new, BATCH, shuffle=True,
                                                 num_workers=2, pin_memory=True)
            else:
                wild_dl_cur = None

            zoo_dl, bos_dl, _ = build_loaders(
                zoo_tr_p, zoo_tr_l, bos_p, bos_l_flat, unknown_l, pc["severity"], 0)
            loaders = [zoo_dl, bos_dl, wild_dl_cur]

            # ETA total
            elapsed  = time.time() - t_start
            ep_done  = sum(PHASES[i]["epochs"] for i in range(phase_idx)) + ep
            ep_left  = TOTAL_EPOCHS - ep_done
            eta_ep   = (elapsed/max(ep_done,1)) * ep_left
            live_state["eta_total"] = int(eta_ep)

            t0ep = time.time()
            _global_ep_box[0] = global_ep
            _best_ep_box[0]   = best_ep
            _best_val_box[0]  = best_val
            l_arc, l_inv, l_bos = train_epoch(
                backbone, arc_loss, loaders, opt, sched, pc,
                n_zoo, n_known, live_state, refresh_fn=refresh_fn)
            if sched: sched.step()
            ep_time = time.time() - t0ep

            # Validation
            acc_c, acc_d = validate(backbone, zoo_tr_p, zoo_tr_l,
                                     zoo_va_p, zoo_va_l, n_zoo, emb_dim)
            bos_disc = bos_discrimination(backbone, bos_p, bos_l_flat, n_zoo)
            # Score composite : zoo propre + zoo dégradé + discrimination BOS
            composite = 0.35*acc_c + 0.40*acc_d + 0.25*min(max(bos_disc+0.1,0),1)

            is_best = composite > best_val
            if is_best:
                best_val = composite; best_ep = global_ep
                save_best(backbone, arc_loss, all_names, emb_dim,
                          global_ep, composite, ckpt_src)

            # History
            for k, v in zip(["l_arc","l_inv","l_bos","acc_clean","acc_deg","bos_disc","composite"],
                            [l_arc, l_inv, l_bos, acc_c, acc_d, bos_disc, composite]):
                history[k].append(v)

            live_state.update({
                "acc_c": acc_c, "acc_d": acc_d, "bos_disc": bos_disc,
                "composite": composite, "is_best": is_best,
            })

            # Sauvegarde après chaque epoch (crash-safe)
            save_resume(backbone, arc_loss, opt, sched,
                        phase_idx, ep, global_ep, best_val, best_ep, history, all_names)

            # Rich display — le live display se met à jour en continu via refresh_fn
            # On force un update final à la fin de l'epoch pour afficher les métriques de val
            if RICH and live_ctx:
                live_ctx.update(make_rich_table(live_state, pc, global_ep, best_ep, best_val))
            else:
                star = " *" if is_best else ""
                log(f"  Ep {global_ep:3d} | L={l_arc:.3f}+{l_inv:.3f}+{l_bos:.3f} "
                    f"| zoo={acc_c*100:.1f}%/dég={acc_d*100:.1f}% "
                    f"| bos={bos_disc:.3f} | composite={composite:.4f}{star} "
                    f"| {ep_time:.0f}s")

            # Early stopping (phase 3 seulement)
            if pc.get("early_stop") and ep >= min(10, pc["epochs"]//3):
                if not is_best:
                    patience_c += 1
                    if patience_c >= pc["patience"]:
                        log(f"  Early stopping à l'epoch {global_ep}"); break
                else:
                    patience_c = 0

        if live_ctx: live_ctx.stop()
        if _interrupt: break

    # ── Post-training ─────────────────────────────────────────────────────────
    section("Post-training")
    if BEST_PT.exists():
        best_ck = torch.load(str(BEST_PT), map_location=DEVICE, weights_only=False)
        backbone.load_state_dict(best_ck["backbone_state"])
        log(f"  Meilleur modèle chargé — epoch {best_ck['epoch']} "
            f"score {best_ck['val_composite']:.4f}")

    opt_t, gap, proto_matrix_np = build_gallery(
        backbone, all_known_p, all_known_l, all_names, n_zoo, emb_dim)

    if history["l_arc"]: make_plots(history)

    total_time = time.time() - t_start
    report = {
        "version": "v5", "generated": datetime.now().isoformat(),
        "training_min": round(total_time/60,1), "dry_run": DRY,
        "best_epoch": best_ep, "best_composite": round(best_val,4),
        "gallery_threshold": round(opt_t,4), "separability_gap": round(gap,4),
        "n_zoo": n_zoo, "n_bos": len(bos_names), "n_total": n_known,
        "zoo_classes": zoo_names, "bos_classes": bos_names,
        "hyperparams": {"arc_scale": ARC_SCALE, "arc_margin": ARC_MARGIN,
                        "k_zoo": K_ZOO, "k_bos": K_BOS, "k_wild": K_WILD,
                        "batch": BATCH, "total_epochs": TOTAL_EPOCHS},
    }
    REPORT_JS.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    section("DONE")
    log(f"""
  Score composite meilleur : {best_val:.4f}  (epoch {best_ep})
  Séparabilité gap         : {gap:.4f}
  Seuil galerie            : {opt_t:.4f}
  Individus                : {n_known}  ({n_zoo} zoo + {len(bos_names)} BOS)
  Durée totale             : {total_time/60:.1f} min

  Sorties dans {OUT} :
    Backbone seul  : {BACKBONE_PT.name}
    Modèle complet : {BEST_PT.name}
    Galerie        : {GALLERY_JS.name}
    Rapport        : {REPORT_JS.name}
    Courbes        : {CURVES_PNG.name}
    Log            : {LOG_FILE.name}

  Prochaine étape :
    Exporter backbone TFLite, puis swap Android
""")
    _log_fh.close()

if __name__ == "__main__":
    main()