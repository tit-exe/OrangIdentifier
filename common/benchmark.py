# benchmark_all_versions.py
# Benchmark comparatif V1 / V2 / V3 / V4
# READ-ONLY sur tous les modeles existants
# Outputs : D:\OrangIdentifier\benchmark\
#
# RUN:
#   conda activate orangs
#   python D:\OrangIdentifier\benchmark_all_versions.py

import os, sys, json, random, io, warnings, time
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime
from collections import defaultdict

os.environ["HF_HOME"]    = r"D:\HuggingFaceCache"
os.environ["TORCH_HOME"] = r"D:\TorchCache"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm
from PIL import Image, ImageFile, ImageFilter, ImageEnhance
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# ==============================================================================
# CONFIG
# ==============================================================================

SEED       = 42
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR    = Path(r"D:\OrangIdentifier\benchmark")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI        = 220   # haute résolution pour zoom sans flou
N_CROPS    = 12    # crops par individu pour le benchmark
N_WILD     = 400   # wild crops à tester

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Normalizations
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
MEGA_MEAN     = [0.5, 0.5, 0.5]
MEGA_STD      = [0.5, 0.5, 0.5]

# ==============================================================================
# PATHS
# ==============================================================================

ZOO_DIR    = Path(r"D:\OrangIdentifier\DATASET_CLASSIFICATION\raw")
BOS_DIR    = Path(r"D:\OrangIdentifier\V2\NEW_ORANGS_CROPS")
WILD_DIR   = Path(r"D:\OrangIdentifier\V2\WILD_CROPS\crops")

V1_MODEL   = Path(r"D:\OrangIdentifier\MODELS\resnet_orangs.pt")
V2_BACKBONE= Path(r"D:\OrangIdentifier\MODELS\backbone_orangs.pt")
V2_GALLERY = Path(r"D:\OrangIdentifier\MODELS\embeddings.json")
V3_MODEL   = Path(r"D:\OrangIdentifier\V2\MODELS\megadesc_T_arcface.pt")
V4_MODEL   = Path(r"D:\OrangIdentifier\V2\V4_improved\models\v4_best.pt")

exts = {".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"}

# ==============================================================================
# LOGGING
# ==============================================================================

log_path = OUT_DIR / "benchmark_log.txt"
_fh      = open(log_path, "w", encoding="utf-8", buffering=1)

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    _fh.write(line + "\n")

def title(t):
    sep = "=" * 70
    log(""); log(sep); log(f"  {t}"); log(sep)

def section(t):
    log(f"\n  --- {t} ---")

# ==============================================================================
# DEGRADATIONS
# ==============================================================================

import io as _io

def degrade(img, kind, sev):
    if kind == "clean":
        return img
    elif kind == "blur":
        r = [0, 1.5, 3.0, 5.0, 8.0][sev]
        return img.filter(ImageFilter.GaussianBlur(radius=r))
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
    elif kind == "exposure":
        f = [1.0, 0.4, 0.2, 2.5, 4.0][sev]
        return ImageEnhance.Brightness(img).enhance(f)
    elif kind == "rotation":
        a = [0, 15, 30, 45, 75][sev]
        return img.rotate(a, expand=False, fillcolor=(128,100,80))
    elif kind == "occlusion":
        img = img.copy(); w,h = img.size
        frac = [0,0.15,0.30,0.45,0.60][sev]
        bh = int(h*frac)
        ys = random.randint(0, max(1, h-bh))
        px = img.load()
        for y in range(ys, min(ys+bh,h)):
            for x in range(w):
                px[x,y] = (20,60,20)
        return img
    elif kind == "combined":
        if sev == 0: return img
        img = degrade(img, "blur",  min(sev,3))
        img = degrade(img, "jpeg",  min(sev,3))
        img = degrade(img, "lowres",min(sev,2))
        return img
    return img

DEG_TYPES  = ["clean","blur","lowres","jpeg","exposure","rotation","occlusion","combined"]
DEG_LABELS = ["Propre","Flou","Basse Res","JPEG","Exposition","Rotation","Occlusion","Combiné"]
SEV_LABELS = ["Aucune","Légère","Modérée","Forte","Extrême"]

# ==============================================================================
# MODEL WRAPPERS
# ==============================================================================

class ModelWrapper:
    """Unified interface for all model versions."""
    def __init__(self, name, threshold):
        self.name      = name
        self.threshold = threshold
        self.classes   = []
        self.emb_dim   = 0

    def embed(self, imgs_tensor):
        """Return L2-normalized embeddings. Override in subclasses."""
        raise NotImplementedError

    def build_protos(self, zoo_paths_by_class):
        """Build prototype dict from zoo paths."""
        self.protos = {}
        for cls, paths in zoo_paths_by_class.items():
            embs = []
            for p in paths:
                try:
                    t = self.preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        embs.append(self.embed(t).cpu())
                except: pass
            if embs:
                self.protos[cls] = F.normalize(torch.cat(embs).mean(0), dim=0)
        self.proto_names  = list(self.protos.keys())
        self.proto_matrix = torch.stack([self.protos[c] for c in self.proto_names])

    def predict(self, img_pil):
        """Return (predicted_class, max_sim, accepted)."""
        try:
            t    = self.preprocess(img_pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                emb  = self.embed(t).cpu()
            sims     = (emb @ self.proto_matrix.T)[0]
            max_sim  = float(sims.max())
            pred_cls = self.proto_names[int(sims.argmax())]
            accepted = max_sim >= self.threshold
            return pred_cls, max_sim, accepted
        except Exception as e:
            return None, 0.0, False

# --------------------------------------------------------------------------
# V1 — ResNet50 classifieur fermé
# --------------------------------------------------------------------------

class V1Model(ModelWrapper):
    def __init__(self):
        super().__init__("V1 ResNet50\nClassifieur fermé", threshold=0.50)
        if not V1_MODEL.exists():
            log(f"  [SKIP] V1 model not found: {V1_MODEL}")
            self.available = False
            return
        self.available = True
        import torchvision.models as tvm
        ckpt   = torch.load(str(V1_MODEL), map_location=DEVICE, weights_only=False)
        # Try to get classes
        self.classes = ckpt.get("classes", [])
        n_cls        = len(self.classes) if self.classes else 10
        model        = tvm.resnet50(weights=None)
        model.fc     = nn.Linear(2048, n_cls)
        try:
            model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        except:
            try: model.load_state_dict(ckpt)
            except: pass
        self.model   = model.eval().to(DEVICE)
        # Remove fc to get backbone
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.backbone = self.backbone.eval().to(DEVICE)
        self.emb_dim  = 2048
        self._tf      = T.Compose([
            T.Resize(224), T.CenterCrop(224), T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        log(f"  V1 loaded: {len(self.classes)} classes")

    def preprocess(self, img): return self._tf(img)

    def embed(self, t):
        with torch.no_grad():
            e = self.backbone(t).squeeze(-1).squeeze(-1)
        return F.normalize(e, dim=1)

# --------------------------------------------------------------------------
# V2 — ResNet50 backbone + galerie JSON
# --------------------------------------------------------------------------

class V2Model(ModelWrapper):
    def __init__(self):
        super().__init__("V2 ResNet50\nEmbeddings open-set", threshold=0.4885)
        if not V2_BACKBONE.exists() or not V2_GALLERY.exists():
            log(f"  [SKIP] V2 files not found")
            self.available = False
            return
        self.available = True
        import torchvision.models as tvm
        ckpt = torch.load(str(V2_BACKBONE), map_location=DEVICE, weights_only=False)
        model = tvm.resnet50(weights=None)
        model.fc = nn.Identity()
        try:
            model.load_state_dict(ckpt.get("backbone_state", ckpt), strict=False)
        except: pass
        self.backbone = model.eval().to(DEVICE)
        self.emb_dim  = 2048
        # Load gallery
        gallery       = json.loads(Path(V2_GALLERY).read_text(encoding="utf-8"))
        self.threshold= gallery.get("unknown_threshold", 0.4885)
        self.classes  = list(gallery.get("individuals", {}).keys())
        # Pre-load prototypes from JSON
        self.protos   = {}
        for name, info in gallery.get("individuals", {}).items():
            if "embedding" in info:
                self.protos[name] = F.normalize(
                    torch.tensor(info["embedding"], dtype=torch.float32), dim=0)
        self.proto_names  = list(self.protos.keys())
        self.proto_matrix = torch.stack([self.protos[c] for c in self.proto_names])
        self._tf = T.Compose([
            T.Resize(224), T.CenterCrop(224), T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        log(f"  V2 loaded: {len(self.classes)} classes, threshold={self.threshold}")

    def preprocess(self, img): return self._tf(img)

    def embed(self, t):
        with torch.no_grad():
            e = self.backbone(t)
        return F.normalize(e, dim=1)

    def build_protos(self, zoo_paths_by_class):
        pass  # V2 uses preloaded gallery

# --------------------------------------------------------------------------
# V3 — MegaDescriptor + ArcFace
# --------------------------------------------------------------------------

class V3Model(ModelWrapper):
    def __init__(self):
        super().__init__("V3 MegaDescriptor\n+ArcFace", threshold=0.22)
        if not V3_MODEL.exists():
            log(f"  [SKIP] V3 model not found: {V3_MODEL}")
            self.available = False
            return
        self.available = True
        ckpt = torch.load(str(V3_MODEL), map_location=DEVICE, weights_only=False)
        self.classes   = ckpt.get("classes", [])
        self.threshold = ckpt.get("threshold", 0.22)
        self.emb_dim   = ckpt.get("emb_dim", 768)
        model = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224",
                                   pretrained=False, num_classes=0)
        model.load_state_dict(ckpt["backbone_state"])
        self.backbone = model.eval().to(DEVICE)
        self._tf = T.Compose([
            T.Resize(224), T.CenterCrop(224), T.ToTensor(),
            T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
        ])
        log(f"  V3 loaded: {len(self.classes)} classes, threshold={self.threshold}")

    def preprocess(self, img): return self._tf(img)

    def embed(self, t):
        with torch.no_grad():
            e = self.backbone(t)
        return F.normalize(e, dim=1)

# --------------------------------------------------------------------------
# V4 — MegaDescriptor + ArcFace improved
# --------------------------------------------------------------------------

class V4Model(ModelWrapper):
    def __init__(self):
        super().__init__("V4 MegaDescriptor\n+ArcFace amélioré", threshold=0.22)
        if not V4_MODEL.exists():
            log(f"  [SKIP] V4 model not found: {V4_MODEL}")
            self.available = False
            return
        self.available = True
        ckpt = torch.load(str(V4_MODEL), map_location=DEVICE, weights_only=False)
        self.classes   = ckpt.get("classes", [])
        self.threshold = ckpt.get("threshold", 0.22)
        self.emb_dim   = ckpt.get("emb_dim", 768)
        # V4 has 40 classes — zoo are first 10
        self.zoo_classes = self.classes[:10] if len(self.classes) >= 10 else self.classes
        model = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224",
                                   pretrained=False, num_classes=0)
        model.load_state_dict(ckpt["backbone_state"])
        self.backbone = model.eval().to(DEVICE)
        self._tf = T.Compose([
            T.Resize(224), T.CenterCrop(224), T.ToTensor(),
            T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
        ])
        log(f"  V4 loaded: {len(self.classes)} classes "
            f"(zoo={len(self.zoo_classes)}), threshold={self.threshold}")

    def preprocess(self, img): return self._tf(img)

    def embed(self, t):
        with torch.no_grad():
            e = self.backbone(t)
        return F.normalize(e, dim=1)

    def build_protos(self, zoo_paths_by_class):
        """V4 builds protos only on zoo classes (first 10)."""
        self.protos = {}
        for cls, paths in zoo_paths_by_class.items():
            if cls not in self.zoo_classes:
                continue
            embs = []
            for p in paths:
                try:
                    t = self.preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        embs.append(self.embed(t).cpu())
                except: pass
            if embs:
                self.protos[cls] = F.normalize(torch.cat(embs).mean(0), dim=0)
        self.proto_names  = list(self.protos.keys())
        self.proto_matrix = torch.stack([self.protos[c] for c in self.proto_names])

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_zoo_paths(n_per_class=N_CROPS):
    dirs = sorted([d for d in ZOO_DIR.iterdir() if d.is_dir()
                   and not d.name.startswith("_")])
    zoo  = {}
    for d in dirs:
        imgs = sorted([f for f in d.iterdir() if f.suffix in exts])
        zoo[d.name] = random.sample(imgs, min(n_per_class, len(imgs)))
    return zoo

def load_bos_paths(n_per_indiv=8):
    bos = {}
    for d in sorted(BOS_DIR.iterdir()):
        if not d.is_dir(): continue
        imgs = sorted([f for f in d.iterdir() if f.suffix.lower() in exts])
        if imgs:
            bos[d.name] = random.sample(imgs, min(n_per_indiv, len(imgs)))
    return bos

def load_wild_paths(n=N_WILD):
    imgs = sorted([f for f in WILD_DIR.iterdir() if f.suffix.lower() in exts])
    return random.sample(imgs, min(n, len(imgs)))

# ==============================================================================
# BENCHMARK FUNCTIONS
# ==============================================================================

def run_clean_benchmark(model, zoo_paths):
    """Per-individual accuracy on clean crops."""
    results = {}
    for cls, paths in zoo_paths.items():
        accs, sims = [], []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                pred, sim, accepted = model.predict(img)
                accs.append(pred == cls)
                sims.append(sim)
            except: pass
        results[cls] = {
            "accuracy":  float(np.mean(accs))  if accs else 0.0,
            "sim_mean":  float(np.mean(sims))  if sims else 0.0,
            "sim_std":   float(np.std(sims))   if sims else 0.0,
            "n":         len(accs),
        }
    return results

def run_stress_benchmark(model, zoo_paths, deg_types, severities=[0,1,2,3,4]):
    """Accuracy per degradation type and severity."""
    results = {}
    for dtype in deg_types:
        results[dtype] = {}
        for sev in severities:
            accs, sims = [], []
            for cls, paths in zoo_paths.items():
                for p in paths:
                    try:
                        img = degrade(Image.open(p).convert("RGB"), dtype, sev)
                        pred, sim, accepted = model.predict(img)
                        accs.append(pred == cls)
                        sims.append(sim)
                    except: pass
            results[dtype][sev] = {
                "accuracy": float(np.mean(accs)) if accs else 0.0,
                "sim_mean": float(np.mean(sims)) if sims else 0.0,
            }
    return results

def run_bos_benchmark(model, bos_paths):
    """Open-set rejection on BOS individuals."""
    per_indiv, all_sims = {}, []
    for name, paths in bos_paths.items():
        sims, rejected = [], []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                _, sim, accepted = model.predict(img)
                sims.append(sim)
                rejected.append(not accepted)
            except: pass
        per_indiv[name] = {
            "rejection_rate": float(np.mean(rejected)) if rejected else 0.0,
            "sim_mean":       float(np.mean(sims))     if sims else 0.0,
            "n":              len(sims),
        }
        all_sims.extend(sims)
    overall = float(np.mean([v["rejection_rate"] for v in per_indiv.values()]))
    return overall, per_indiv, all_sims

def run_wild_benchmark(model, wild_paths):
    """Rejection on wild internet crops."""
    rejected, sims = [], []
    for p in wild_paths:
        try:
            img = Image.open(p).convert("RGB")
            _, sim, accepted = model.predict(img)
            rejected.append(not accepted)
            sims.append(sim)
        except: pass
    return float(np.mean(rejected)) if rejected else 0.0, sims

def run_separability(model, zoo_paths):
    """Compute separability metrics from zoo crops."""
    pos_sims, neg_sims = [], []
    for cls, paths in zoo_paths.items():
        if cls not in model.protos: continue
        proto  = model.protos[cls]
        others = torch.stack([model.protos[c] for c in model.proto_names if c != cls])
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                t   = model.preprocess(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    e = model.embed(t).cpu()
                pos_sims.append(float((e @ proto).item()))
                if len(others) > 0:
                    neg_sims.append(float((e @ others.T).max().item()))
            except: pass
    return (float(np.mean(pos_sims)) if pos_sims else 0.0,
            float(np.mean(neg_sims)) if neg_sims else 0.0,
            pos_sims, neg_sims)

# ==============================================================================
# PLOTTING — style dark, haute résolution
# ==============================================================================

DARK_BG  = "#0d0d0d"
DARK_AX  = "#111111"
COLORS   = ["#4488ff", "#00cc88", "#ff6644", "#ffcc00"]  # V1 V2 V3 V4

def style_ax(ax):
    ax.set_facecolor(DARK_AX)
    ax.tick_params(colors="#888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.grid(color="#222", linewidth=0.5, zorder=0)

def save_fig(fig, name):
    path = OUT_DIR / name
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    log(f"  Saved: {path.name}")
    return path

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    title("BENCHMARK COMPARATIF V1 / V2 / V3 / V4")
    log(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Device  : {DEVICE}")
    log(f"  Output  : {OUT_DIR}")
    log(f"  DPI     : {DPI}")

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    section("Loading models")
    models_all = [V1Model(), V2Model(), V3Model(), V4Model()]
    models     = [m for m in models_all if m.available]
    names      = [m.name for m in models]
    colors     = [COLORS[i] for i, m in enumerate(models_all) if m.available]
    short_names= ["V1","V2","V3","V4"]
    short_names= [short_names[i] for i, m in enumerate(models_all) if m.available]

    log(f"  Available models: {short_names}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    section("Loading data")
    zoo_paths_all = load_zoo_paths(n_per_class=50)   # all for protos
    zoo_paths_bm  = {cls: random.sample(paths, min(N_CROPS, len(paths)))
                     for cls, paths in zoo_paths_all.items()}
    bos_paths     = load_bos_paths(n_per_indiv=8)
    wild_paths    = load_wild_paths(N_WILD)
    zoo_classes   = list(zoo_paths_all.keys())
    log(f"  Zoo: {len(zoo_classes)} classes, {sum(len(v) for v in zoo_paths_bm.values())} benchmark crops")
    log(f"  BOS: {len(bos_paths)} individuals")
    log(f"  Wild: {len(wild_paths)} crops")

    # ------------------------------------------------------------------
    # Build prototypes
    # ------------------------------------------------------------------
    section("Building prototypes")
    for m in models:
        m.build_protos(zoo_paths_all)
        log(f"  {m.name.split(chr(10))[0]}: {len(m.protos)} prototypes")

    # ------------------------------------------------------------------
    # Run all benchmarks
    # ------------------------------------------------------------------
    section("Running benchmarks")
    all_results = {}

    for i, m in enumerate(models):
        sn = short_names[i]
        log(f"\n  [{sn}] {m.name.replace(chr(10),' ')}")
        t0 = time.time()

        log(f"    Clean accuracy...")
        clean = run_clean_benchmark(m, zoo_paths_bm)

        log(f"    Stress test ({len(DEG_TYPES)} types x 5 severities)...")
        stress = run_stress_benchmark(m, zoo_paths_bm, DEG_TYPES)

        log(f"    BOS rejection...")
        bos_rate, bos_per, bos_sims = run_bos_benchmark(m, bos_paths)

        log(f"    Wild rejection...")
        wild_rate, wild_sims = run_wild_benchmark(m, wild_paths)

        log(f"    Separability...")
        pos_m, neg_m, pos_sims, neg_sims = run_separability(m, zoo_paths_bm)

        all_results[sn] = {
            "clean":      clean,
            "stress":     stress,
            "bos_rate":   bos_rate,
            "bos_per":    bos_per,
            "bos_sims":   bos_sims,
            "wild_rate":  wild_rate,
            "wild_sims":  wild_sims,
            "pos_mean":   pos_m,
            "neg_mean":   neg_m,
            "pos_sims":   pos_sims,
            "neg_sims":   neg_sims,
            "sep_gap":    pos_m - neg_m,
            "threshold":  m.threshold,
        }
        log(f"    Done in {time.time()-t0:.0f}s | "
            f"clean={np.mean([v['accuracy'] for v in clean.values()])*100:.1f}% | "
            f"bos_rej={bos_rate*100:.1f}% | wild_rej={wild_rate*100:.1f}% | "
            f"gap={pos_m-neg_m:.3f}")

    # ------------------------------------------------------------------
    # FIGURE 1 — Vue d'ensemble : 4 métriques principales
    # ------------------------------------------------------------------
    title("Generating figures")
    section("Figure 1: Vue d'ensemble")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor(DARK_BG)

    metrics = [
        ("Accuracy zoo (clean)", [np.mean([v["accuracy"] for v in all_results[sn]["clean"].values()])*100
                                   for sn in short_names]),
        ("Rejet BOS inconnus (%)", [all_results[sn]["bos_rate"]*100 for sn in short_names]),
        ("Rejet wild internet (%)", [all_results[sn]["wild_rate"]*100 for sn in short_names]),
        ("Gap séparabilité", [all_results[sn]["sep_gap"] for sn in short_names]),
    ]

    for ax, (title_str, vals) in zip(axes, metrics):
        style_ax(ax)
        bars = ax.bar(short_names, vals, color=colors, alpha=0.85,
                      edgecolor="#555", linewidth=0.8, zorder=3)
        ax.set_title(title_str, color="white", fontsize=10, pad=8)
        ax.set_ylim(0, max(vals)*1.15 if max(vals) > 0 else 1)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                    f"{val:.1f}", ha="center", va="bottom", color="white", fontsize=9,
                    fontweight="bold")
        ax.tick_params(colors="#aaa")

    fig.suptitle("Vue d'ensemble — Comparaison V1 / V2 / V3 / V4",
                 color="white", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, "01_overview.png")

    # ------------------------------------------------------------------
    # FIGURE 2 — Stress test : accuracy par dégradation et sévérité
    # ------------------------------------------------------------------
    section("Figure 2: Stress test")

    n_deg = len(DEG_TYPES)
    fig   = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor(DARK_BG)
    gs    = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    for di, (dtype, dlabel) in enumerate(zip(DEG_TYPES, DEG_LABELS)):
        row, col = di // 4, di % 4
        ax = fig.add_subplot(gs[row, col])
        style_ax(ax)
        x = np.arange(5)
        for sn, color in zip(short_names, colors):
            ys = [all_results[sn]["stress"][dtype][sev]["accuracy"]*100 for sev in range(5)]
            ax.plot(x, ys, color=color, marker="o", linewidth=2,
                    markersize=5, label=sn, zorder=3)
        ax.axhline(80, color="#555", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(90, color="#444", linestyle=":",  linewidth=0.8, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(SEV_LABELS, rotation=25, fontsize=7)
        ax.set_ylim(0, 108)
        ax.set_title(dlabel, color="white", fontsize=11, pad=6)
        ax.set_ylabel("Accuracy (%)", color="#888", fontsize=8)
        if di == 0:
            leg = ax.legend(fontsize=8, framealpha=0.3, labelcolor="white",
                            facecolor="#111", loc="lower left")

    fig.suptitle("Stress test — Accuracy par type et sévérité de dégradation",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    save_fig(fig, "02_stress_test.png")

    # ------------------------------------------------------------------
    # FIGURE 3 — Heatmap des ruptures
    # ------------------------------------------------------------------
    section("Figure 3: Heatmap ruptures")

    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 7))
    fig.patch.set_facecolor(DARK_BG)
    if len(models) == 1: axes = [axes]

    cmap = LinearSegmentedColormap.from_list("rg", ["#cc2200","#ffaa00","#00cc44"])

    for ax, sn, color in zip(axes, short_names, colors):
        data = np.zeros((len(DEG_TYPES), 5))
        for di, dtype in enumerate(DEG_TYPES):
            for sev in range(5):
                data[di, sev] = all_results[sn]["stress"][dtype][sev]["accuracy"]*100
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(5))
        ax.set_xticklabels(SEV_LABELS, rotation=30, fontsize=8, color="#aaa")
        ax.set_yticks(range(len(DEG_LABELS)))
        ax.set_yticklabels(DEG_LABELS, fontsize=9, color="#aaa")
        ax.set_title(sn, color=color, fontsize=13, fontweight="bold", pad=8)
        for di in range(len(DEG_TYPES)):
            for sev in range(5):
                val = data[di, sev]
                tc  = "black" if val > 50 else "white"
                ax.text(sev, di, f"{val:.0f}", ha="center", va="center",
                        color=tc, fontsize=8, fontweight="bold")
        ax.spines[:].set_color("#333")

    plt.colorbar(im, ax=axes[-1], label="Accuracy (%)",
                 fraction=0.04, pad=0.04).ax.yaxis.label.set_color("white")
    fig.suptitle("Heatmap — Accuracy (%) par dégradation et sévérité",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(fig, "03_stress_heatmap.png")

    # ------------------------------------------------------------------
    # FIGURE 4 — Par individu zoo
    # ------------------------------------------------------------------
    section("Figure 4: Per individual accuracy")

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor(DARK_BG)
    style_ax(ax)

    n_cls = len(zoo_classes)
    x     = np.arange(n_cls)
    w     = 0.18
    offs  = np.linspace(-(len(models)-1)*w/2, (len(models)-1)*w/2, len(models))

    for j, (sn, color) in enumerate(zip(short_names, colors)):
        accs = [all_results[sn]["clean"].get(cls, {}).get("accuracy", 0)*100
                for cls in zoo_classes]
        ax.bar(x + offs[j], accs, width=w, color=color, alpha=0.8,
               label=sn, edgecolor="#333", linewidth=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(zoo_classes, rotation=35, ha="right", fontsize=9, color="#ccc")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)", color="#888")
    ax.set_title("Accuracy par individu zoo — images propres", color="white",
                 fontsize=12, pad=10)
    ax.legend(fontsize=9, framealpha=0.3, labelcolor="white", facecolor="#111")
    ax.axhline(100, color="#333", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    save_fig(fig, "04_per_individual.png")

    # ------------------------------------------------------------------
    # FIGURE 5 — Similarité distributions
    # ------------------------------------------------------------------
    section("Figure 5: Similarity distributions")

    fig, axes = plt.subplots(2, len(models), figsize=(5*len(models), 10))
    fig.patch.set_facecolor(DARK_BG)
    if len(models) == 1: axes = axes.reshape(2, 1)

    bins = np.linspace(-0.3, 1.05, 60)

    for j, (sn, color) in enumerate(zip(short_names, colors)):
        r = all_results[sn]
        # Top: positive vs negative
        ax = axes[0, j]
        style_ax(ax)
        ax.hist(r["pos_sims"], bins=bins, alpha=0.7, color="#00cc44",
                density=True, label=f"Positif (n={len(r['pos_sims'])})")
        ax.hist(r["neg_sims"], bins=bins, alpha=0.7, color="#cc2200",
                density=True, label=f"Négatif (n={len(r['neg_sims'])})")
        ax.axvline(r["threshold"], color="white", linestyle="--",
                   linewidth=1.5, label=f"Seuil={r['threshold']:.2f}")
        ax.set_title(f"{sn} — Séparabilité\ngap={r['sep_gap']:.3f}",
                     color=color, fontsize=10)
        if j == 0: ax.legend(fontsize=7, framealpha=0.3, labelcolor="white", facecolor="#111")

        # Bottom: BOS vs wild
        ax = axes[1, j]
        style_ax(ax)
        ax.hist(r["bos_sims"],  bins=bins, alpha=0.7, color="#4488ff",
                density=True, label=f"BOS inconnus (n={len(r['bos_sims'])})")
        ax.hist(r["wild_sims"], bins=bins, alpha=0.7, color="#ffaa00",
                density=True, label=f"Wild internet (n={len(r['wild_sims'])})")
        ax.axvline(r["threshold"], color="white", linestyle="--", linewidth=1.5)
        bos_rej  = np.mean(np.array(r["bos_sims"]) < r["threshold"])*100
        wild_rej = np.mean(np.array(r["wild_sims"]) < r["threshold"])*100
        ax.set_title(f"BOS rej={bos_rej:.1f}%  Wild rej={wild_rej:.1f}%",
                     color="#aaa", fontsize=9)
        if j == 0: ax.legend(fontsize=7, framealpha=0.3, labelcolor="white", facecolor="#111")

    fig.suptitle("Distributions de similarité cosinus",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(fig, "05_similarity_distributions.png")

    # ------------------------------------------------------------------
    # FIGURE 6 — Radar chart toutes métriques
    # ------------------------------------------------------------------
    section("Figure 6: Radar chart")

    categories = ["Acc zoo\n(clean)","Rejet BOS","Rejet wild",
                  "Séparabilité\n(norm.)","Robustesse\nflou","Robustesse\nrésolution"]
    N_cat = len(categories)
    angles = [n/N_cat * 2*np.pi for n in range(N_cat)] + [0]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_AX)
    ax.spines["polar"].set_color("#333")

    for j, (sn, color) in enumerate(zip(short_names, colors)):
        r   = all_results[sn]
        acc = np.mean([v["accuracy"] for v in r["clean"].values()])
        sep = min(r["sep_gap"] / 1.0, 1.0)   # normalize to 0-1
        rob_blur = np.mean([r["stress"]["blur"][s]["accuracy"] for s in [1,2,3]])
        rob_res  = np.mean([r["stress"]["lowres"][s]["accuracy"] for s in [1,2,3]])
        vals = [acc, r["bos_rate"], r["wild_rate"], sep, rob_blur, rob_res]
        vals += [vals[0]]
        ax.plot(angles, vals, color=color, linewidth=2, label=sn, zorder=3)
        ax.fill(angles, vals, color=color, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color="white", fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["25%","50%","75%","100%"], color="#666", fontsize=8)
    ax.grid(color="#333", linewidth=0.8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=11, framealpha=0.3, labelcolor="white", facecolor="#111")
    ax.set_title("Radar — Performance globale", color="white",
                 fontsize=14, fontweight="bold", pad=25)

    save_fig(fig, "06_radar.png")

    # ------------------------------------------------------------------
    # FIGURE 7 — BOS rejection par individu
    # ------------------------------------------------------------------
    section("Figure 7: BOS rejection per individual")

    bos_names = sorted(bos_paths.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 9))
    fig.patch.set_facecolor(DARK_BG)
    if len(models) == 1: axes = [axes]

    for j, (sn, color) in enumerate(zip(short_names, colors)):
        ax    = axes[j]
        style_ax(ax)
        r     = all_results[sn]["bos_per"]
        rates = [r.get(name, {}).get("rejection_rate", 0)*100 for name in bos_names]
        bar_colors = ["#00cc44" if v >= 90 else "#ffaa00" if v >= 70 else "#cc2200"
                      for v in rates]
        bars  = ax.barh(range(len(bos_names)), rates, color=bar_colors, alpha=0.85,
                        edgecolor="#333", linewidth=0.4, zorder=3)
        ax.set_yticks(range(len(bos_names)))
        ax.set_yticklabels(bos_names, fontsize=7, color="#ccc")
        ax.set_xlim(0, 110)
        ax.axvline(90, color="white", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{sn}\nMoy={np.mean(rates):.1f}%", color=color,
                     fontsize=11, pad=8)
        ax.set_xlabel("Taux de rejet (%)", color="#888", fontsize=8)
        for bar, val in zip(bars, rates):
            ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                    f"{val:.0f}%", va="center", fontsize=6, color="white")

    fig.suptitle("Taux de rejet BOS par individu inconnu",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(fig, "07_bos_per_individual.png")

    # ------------------------------------------------------------------
    # FIGURE 8 — Dégradation combinée : comparaison V3 vs V4
    # ------------------------------------------------------------------
    section("Figure 8: V3 vs V4 focus")

    if "V3" in short_names and "V4" in short_names:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor(DARK_BG)
        v3i = short_names.index("V3"); v4i = short_names.index("V4")
        v3c = colors[v3i]; v4c = colors[v4i]

        focus_types = ["blur","lowres","combined"]
        focus_labels= ["Flou","Basse résolution","Combiné"]

        for k, (dtype, dlabel) in enumerate(zip(focus_types, focus_labels)):
            ax = axes[k]
            style_ax(ax)
            x  = np.arange(5)
            v3_y = [all_results["V3"]["stress"][dtype][s]["accuracy"]*100 for s in range(5)]
            v4_y = [all_results["V4"]["stress"][dtype][s]["accuracy"]*100 for s in range(5)]
            ax.plot(x, v3_y, color=v3c, marker="o", linewidth=2.5,
                    markersize=8, label="V3", zorder=3)
            ax.plot(x, v4_y, color=v4c, marker="s", linewidth=2.5,
                    markersize=8, label="V4", zorder=3)
            # Delta
            for xi, (y3, y4) in enumerate(zip(v3_y, v4_y)):
                delta = y4 - y3
                col   = "#00cc44" if delta > 0 else "#cc2200"
                ax.annotate(f"{delta:+.1f}%", xy=(xi, max(y3,y4)+2),
                            ha="center", fontsize=8, color=col)
            ax.axhline(80, color="#444", linestyle="--", linewidth=0.8)
            ax.set_xticks(x); ax.set_xticklabels(SEV_LABELS, rotation=20, fontsize=8)
            ax.set_ylim(0, 115)
            ax.set_title(f"V3 vs V4 — {dlabel}", color="white", fontsize=11, pad=8)
            ax.set_ylabel("Accuracy (%)", color="#888")
            ax.legend(fontsize=9, framealpha=0.3, labelcolor="white", facecolor="#111")

        fig.suptitle("Focus V3 vs V4 — Dégradations critiques identifiées",
                     color="white", fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        save_fig(fig, "08_v3_vs_v4.png")

    # ------------------------------------------------------------------
    # FIGURE 9 — Tableau récapitulatif complet
    # ------------------------------------------------------------------
    section("Figure 9: Summary table")

    rows = []
    row_labels = []

    for sn in short_names:
        r   = all_results[sn]
        acc = np.mean([v["accuracy"] for v in r["clean"].values()])*100
        row_labels.append(sn)
        row = [
            f"{acc:.1f}%",
            f"{r['bos_rate']*100:.1f}%",
            f"{r['wild_rate']*100:.1f}%",
            f"{r['sep_gap']:.3f}",
            f"{r['pos_mean']:.3f}",
            f"{r['neg_mean']:.3f}",
            f"{r['stress']['blur'][1]['accuracy']*100:.1f}%",
            f"{r['stress']['blur'][2]['accuracy']*100:.1f}%",
            f"{r['stress']['lowres'][1]['accuracy']*100:.1f}%",
            f"{r['stress']['lowres'][2]['accuracy']*100:.1f}%",
            f"{r['stress']['combined'][1]['accuracy']*100:.1f}%",
            f"{r['stress']['combined'][2]['accuracy']*100:.1f}%",
        ]
        rows.append(row)

    col_labels = ["Acc zoo","Rejet BOS","Rejet wild",
                  "Sep gap","Pos sim","Neg sim",
                  "Flou\nléger","Flou\nmodéré",
                  "Res\nlégère","Res\nmodérée",
                  "Combiné\nléger","Combiné\nmodéré"]

    fig, ax = plt.subplots(figsize=(len(col_labels)*1.6, len(models)*0.9 + 1.5))
    fig.patch.set_facecolor(DARK_BG)
    ax.axis("off")

    tbl = ax.table(
        cellText  = rows,
        rowLabels = row_labels,
        colLabels = col_labels,
        cellLoc   = "center",
        loc       = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#333")
        if row == 0:
            cell.set_facecolor("#222")
            cell.set_text_props(color="white", fontweight="bold", fontsize=8)
        elif col == -1:
            idx = row - 1
            cell.set_facecolor(colors[idx] + "44")
            cell.set_text_props(color=colors[idx], fontweight="bold")
        else:
            cell.set_facecolor("#111" if row%2==0 else "#161616")
            cell.set_text_props(color="#ddd")

    ax.set_title("Tableau récapitulatif complet", color="white",
                 fontsize=13, fontweight="bold", pad=20)
    save_fig(fig, "09_summary_table.png")

    # ------------------------------------------------------------------
    # FIGURE 10 — Throughput / temps d'inférence estimé
    # ------------------------------------------------------------------
    section("Figure 10: Inference speed")

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DARK_BG)
    style_ax(ax)

    speeds = []
    for m in models:
        try:
            img = Image.new("RGB", (224,224))
            t0  = time.time()
            for _ in range(50):
                m.predict(img)
            ms_per = (time.time()-t0) / 50 * 1000
        except:
            ms_per = 0
        speeds.append(ms_per)

    bars = ax.bar(short_names, speeds, color=colors, alpha=0.85,
                  edgecolor="#555", linewidth=0.8, zorder=3)
    for bar, val in zip(bars, speeds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.0f}ms", ha="center", va="bottom", color="white",
                fontsize=10, fontweight="bold")
    ax.set_ylabel("Temps d'inférence (ms / image)", color="#888")
    ax.set_title("Vitesse d'inférence GPU (RTX 3050)\n"
                 "YOLO non inclus — backbone + similarité uniquement",
                 color="white", fontsize=11, pad=10)
    plt.tight_layout()
    save_fig(fig, "10_inference_speed.png")

    # ------------------------------------------------------------------
    # JSON FINAL REPORT
    # ------------------------------------------------------------------
    section("Saving JSON report")

    report = {
        "generated":    datetime.now().isoformat(),
        "device":       str(DEVICE),
        "n_crops_bench": N_CROPS,
        "n_wild":       N_WILD,
        "models":       {},
    }

    for sn in short_names:
        r = all_results[sn]
        report["models"][sn] = {
            "threshold":      all_results[sn]["threshold"],
            "zoo_accuracy":   round(np.mean([v["accuracy"] for v in r["clean"].values()])*100, 2),
            "bos_rejection":  round(r["bos_rate"]*100, 2),
            "wild_rejection": round(r["wild_rate"]*100, 2),
            "sep_gap":        round(r["sep_gap"], 4),
            "pos_sim_mean":   round(r["pos_mean"], 4),
            "neg_sim_mean":   round(r["neg_mean"], 4),
            "per_individual_zoo": {
                cls: round(v["accuracy"]*100, 1)
                for cls, v in r["clean"].items()
            },
            "stress_summary": {
                dtype: {
                    str(sev): round(r["stress"][dtype][sev]["accuracy"]*100, 1)
                    for sev in range(5)
                }
                for dtype in DEG_TYPES
            },
        }

    json_path = OUT_DIR / "benchmark_report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    log(f"  JSON saved: {json_path}")

    # ------------------------------------------------------------------
    # TERMINAL SUMMARY
    # ------------------------------------------------------------------
    title("RÉSULTATS FINAUX")
    log(f"\n  {'Modèle':<20}  {'Zoo acc':>8}  {'BOS rej':>8}  {'Wild rej':>9}  {'Sep gap':>8}  {'Flou mod':>9}  {'Res mod':>8}")
    log(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*8}")
    for sn in short_names:
        r = all_results[sn]
        log(f"  {sn:<20}  "
            f"{np.mean([v['accuracy'] for v in r['clean'].values()])*100:>7.1f}%  "
            f"{r['bos_rate']*100:>7.1f}%  "
            f"{r['wild_rate']*100:>8.1f}%  "
            f"{r['sep_gap']:>8.3f}  "
            f"{r['stress']['blur'][2]['accuracy']*100:>8.1f}%  "
            f"{r['stress']['lowres'][2]['accuracy']*100:>7.1f}%")

    log(f"\n  Outputs:")
    for f in sorted(OUT_DIR.glob("*.png")):
        log(f"    {f.name}")
    log(f"    benchmark_report.json")
    log(f"\n  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _fh.close()

if __name__ == "__main__":
    main()