"""
V6_field_sim.py — OrangIdentifier V6
=======================================
Simulation terrain "ajout d'un nouvel individu avec N photos".

Protocole (réplique exacte du comportement de l'app) :
  Pour chaque individu test (auto-sélectionnés : 1 facile, 1 moyen, 1 difficile) :
    1. Charger la galerie V6 (gallery.json) et RETIRER l'individu test en mémoire
       → galerie simulée = 14 autres individus (exemplars précomputes, pas de reforward)
    2. Collecter les crops de l'individu test sur disque, extraire les embeddings
    3. Split fixe 75% TRAIN (photos terrain) / 25% TEST (photos non vues)
    4. Pour N = 1, 2, 3, 4, 5, 7, 10, 15, 20 — répéter K_MC=50 fois :
         - Tirer N crops au hasard dans TRAIN
         - prototype = L2_normalize( mean( embs[N tirés] ) )   ← identique à l'app
         - Galerie_sim = {14 autres} ∪ {individu_test : [prototype]}
         - TP_score[N]   = mean cosine_sim(prototype, TEST crops de l'individu)
         - TP_rate[N]    = % TEST reconnus comme l'individu correct (top match ≥ seuil)
         - FP_rate[N]    = % crops wild reconnus comme QUELQU'UN dans Galerie_sim
    5. Figures :
         - grille 3×3 par individu (TP score, TP rate, FP rate)
         - synthèse comparant les 3 individus

Auto-sélection : 1 individu haute confiance, 1 médiane, 1 faible confiance
  (d'après la similarité intra-classe des exemplars dans gallery.json)
  → override : TEST_INDIVIDUALS = ["Auti", "Rosa", "Kembali"]

Durée estimée : ~3-5 min CPU / ~30 s GPU (embeddings cachés après 1er run)

RUN :
    conda activate orangs
    python v6_megadesc_arcface_15ind/06_field_simulation.py

    # Recalculer embeddings (si nouveaux crops) :
    python v6_megadesc_arcface_15ind/06_field_simulation.py --recache
"""

import argparse
import json
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import timm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

REPO = Path(__file__).resolve().parents[1]  # repository root (portable)

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

os.environ["HF_HOME"]    = r"D:\HuggingFaceCache"
os.environ["TORCH_HOME"] = r"D:\TorchCache"

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
V6_BEST   = (REPO / "output" / "v6" / "models" / "v6_best.pt")
GALLERY   = (REPO / "output" / "v6" / "models" / "v6_gallery.json")
ZOO_DIR   = (REPO / "data" / "crops" / "known")
NEW_ZOO   = (REPO / "data" / "crops" / "new")
WILD_DIR  = (REPO / "data" / "crops" / "wild")
RESULTS   = (REPO / "output" / "v6" / "results")
CACHE_F   = RESULTS / "emb_cache_field_sim.npz"

THRESHOLD = 0.5371      # seuil galerie V6

SEED      = 42
TEST_FRAC = 0.25        # fraction crops réservés pour le test
TEST_MIN  = 5           # minimum crops test
TRAIN_MIN = 3           # minimum crops train pool
MIN_CROPS = 12          # individus avec moins sont ignorés

N_VALUES  = [1, 2, 3, 4, 5, 7, 10, 15, 20, 25]   # inclut 25 (demandé par Cédric)
K_MC      = 50          # Monte Carlo repetitions (50 = good balance speed/variance)
N_WILD    = 200         # max wild crops for FP analysis
IMG_SIZE  = 224
BATCH     = 64
EXTS      = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

# Set to a list to test specific individuals only, e.g. ["Kembali", "Auti"]
# None = run ALL gallery individuals
TEST_INDIVIDUALS: list | None = None

# Colors
C_MEAN  = "#27500A"   # dark green — mean line
C_BAND  = "#639922"   # light green — ±std band
C_FP    = "#c4501b"   # orange — false positives
C_THR   = "#A32D2D"   # red — threshold

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════
ap = argparse.ArgumentParser()
ap.add_argument("--recache", action="store_true",
                help="Recalculer les embeddings (ignorer le cache)")
args = ap.parse_args()

# ══════════════════════════════════════════════════════════════
# CHARGEMENT GALERIE (exemplars précomputes depuis gallery.json)
# ══════════════════════════════════════════════════════════════
def load_gallery():
    if not GALLERY.exists():
        raise FileNotFoundError(f"gallery.json introuvable : {GALLERY}")
    with open(GALLERY, encoding="utf-8") as f:
        gal = json.load(f)
    exemplars = {}
    for name, data in gal.get("individuals", {}).items():
        exs = data.get("exemplars", [])
        if exs:
            exemplars[name] = np.array(exs, dtype=np.float32)  # (K, 768)
    print(f"Galerie chargée : {len(exemplars)} individus, threshold={THRESHOLD}")
    return exemplars

# ══════════════════════════════════════════════════════════════
# BACKBONE + EMBEDDINGS
# ══════════════════════════════════════════════════════════════
_tf = T.Compose([
    T.Resize(IMG_SIZE), T.CenterCrop(IMG_SIZE),
    T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3),
])

class _PathDS(Dataset):
    def __init__(self, paths):
        self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        try:   img = Image.open(self.paths[i]).convert("RGB")
        except: img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128,)*3)
        return _tf(img)

def load_backbone():
    if not V6_BEST.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {V6_BEST}")
    print("Chargement backbone V6...")
    bb = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224", pretrained=False, num_classes=0)
    ck = torch.load(str(V6_BEST), map_location="cpu", weights_only=False)
    state = ck.get("backbone_state") or ck.get("model_state_dict") or ck
    bb.load_state_dict(state, strict=False)
    return bb.to(DEVICE).eval()

@torch.no_grad()
def embed(backbone, paths):
    if not paths:
        return np.zeros((0, 768), dtype=np.float32)
    dl  = DataLoader(_PathDS(paths), batch_size=BATCH, num_workers=0)
    out = []
    for imgs in dl:
        out.append(F.normalize(backbone(imgs.to(DEVICE)), dim=1).cpu().numpy())
    return np.concatenate(out, 0).astype(np.float32)

def collect_individual(name):
    """Collecte tous les crops d'un individu depuis ZOO_DIR et NEW_ZOO."""
    paths = []
    for base in [ZOO_DIR, NEW_ZOO]:
        d = base / name
        if d.exists():
            paths.extend([f for f in d.iterdir() if f.suffix in EXTS])
    return sorted(set(paths))

def collect_wild():
    """Collecte max N_WILD crops wild (individus inconnus)."""
    if not WILD_DIR.exists():
        return []
    all_w = sorted([f for f in WILD_DIR.rglob("*") if f.suffix in EXTS])
    return random.Random(SEED).sample(all_w, min(N_WILD, len(all_w)))

# ══════════════════════════════════════════════════════════════
# CACHE EMBEDDINGS (pour ne pas tout recalculer à chaque run)
# ══════════════════════════════════════════════════════════════
def build_or_load_cache(all_gallery_names, recache=False):
    """
    Calcule les embeddings pour TOUS les individus de la galerie.
    Nécessaire pour que l'auto-sélection repose sur la variété réelle des crops
    (et non sur les exemplaires cherry-pickés de gallery.json).
    """
    RESULTS.mkdir(parents=True, exist_ok=True)

    if CACHE_F.exists() and not recache:
        data       = np.load(CACHE_F, allow_pickle=True)
        cached_for = set(data["cached_individuals"].tolist())
        if set(all_gallery_names).issubset(cached_for):
            print(f"Cache complet ({len(cached_for)} individus)  (--recache pour recalculer)")
            zoo_embs  = {n: data[f"zoo_{n}"] for n in all_gallery_names if f"zoo_{n}" in data}
            wild_embs = data["wild_embs"]
            total = sum(len(v) for v in zoo_embs.values())
            print(f"  {len(zoo_embs)} individus · {total} crops zoo · {len(wild_embs)} wild")
            return zoo_embs, wild_embs

    backbone = load_backbone()

    print(f"\nExtraction embeddings pour les {len(all_gallery_names)} individus...")
    zoo_embs = {}
    for name in sorted(all_gallery_names):
        paths = collect_individual(name)
        if not paths:
            print(f"  {name:<14}: IGNORÉ (aucun crop trouvé dans ZOO_DIR / NEW_ZOO)")
            continue
        zoo_embs[name] = embed(backbone, paths)
        print(f"  {name:<14}: {len(zoo_embs[name]):4d} embeddings")

    print(f"\nCrops wild (max {N_WILD})...")
    wild_embs = embed(backbone, collect_wild())
    print(f"  {len(wild_embs):4d} embeddings wild")

    save = {f"zoo_{n}": v for n, v in zoo_embs.items()}
    save["wild_embs"]          = wild_embs
    save["cached_individuals"] = np.array(sorted(zoo_embs.keys()))
    np.savez_compressed(CACHE_F, **save)
    print(f"Cache sauvegardé : {CACHE_F}")

    return zoo_embs, wild_embs

# ══════════════════════════════════════════════════════════════
# AUTO-SÉLECTION DES 3 INDIVIDUS
# ══════════════════════════════════════════════════════════════
def auto_select(zoo_embs, n_select=5):
    """
    Sélectionne n_select individus couvrant le spectre de difficulté.
    Critère : variété réelle des crops terrain (similarité cosinus au centroïde).
    Faible varieté = individu facile (ses photos se ressemblent).
    Forte variété  = individu difficile (apparence variable, hard case).

    Différent de la self-sim des EXEMPLAIRES de gallery.json qui sont
    cherry-pickés et donnent toujours une haute similarité.
    """
    variety = {}
    for name, embs in zoo_embs.items():
        centroid = embs.mean(0)
        centroid /= np.linalg.norm(centroid) + 1e-8
        # Similarité moyenne des crops au centroïde (haute = peu varié = facile)
        variety[name] = float((embs @ centroid).mean())

    # Trier du plus facile (haute sim) au plus difficile (faible sim)
    ranked = sorted(variety, key=variety.get, reverse=True)
    n = len(ranked)

    if n_select >= n:
        chosen = ranked
    else:
        # Échantillonner uniformément du plus facile au plus difficile
        indices = np.round(np.linspace(0, n - 1, n_select)).astype(int)
        chosen  = [ranked[i] for i in indices]

    print(f"  {'Individu':<14}  {'Sim au centroïde':>18}  {'Difficulté'}")
    print(f"  {'─'*45}")
    for name in ranked:
        tag    = "<── sélectionné" if name in chosen else ""
        level  = "facile" if variety[name] > 0.85 else ("moyen" if variety[name] > 0.70 else "difficile")
        print(f"  {name:<14}  {variety[name]:>18.4f}  {level:<10}  {tag}")

    return chosen, variety

# ══════════════════════════════════════════════════════════════
# SPLIT TRAIN / TEST — déterministe, sans fuite
# ══════════════════════════════════════════════════════════════
def make_split(n_total, seed):
    rng    = np.random.default_rng(seed)
    idx    = rng.permutation(n_total)
    n_test = max(TEST_MIN, int(np.ceil(n_total * TEST_FRAC)))
    n_test = min(n_test, n_total - TRAIN_MIN)
    if n_test <= 0:
        return idx, np.array([], dtype=int)
    return idx[n_test:], idx[:n_test]   # train, test

# ══════════════════════════════════════════════════════════════
# SCORING (logique identique à l'app)
# ══════════════════════════════════════════════════════════════
def score_against_gallery(query_embs, gallery_exemplars, proto_test=None, test_name=None):
    """
    query_embs : (N_query, 768)
    gallery_exemplars : {name: (K, 768)}  — SANS l'individu test
    proto_test : (768,)   — prototype de l'individu test (ou None)
    test_name  : nom de l'individu test

    Retourne pour chaque query :
      best_name, best_score, score_vs_test (None si proto_test=None)
    """
    results = []
    for qemb in query_embs:
        scores = {n: float((qemb @ exs.T).max()) for n, exs in gallery_exemplars.items()}
        if proto_test is not None:
            scores[test_name] = float(np.dot(qemb, proto_test))
        best_name  = max(scores, key=scores.get)
        best_score = scores[best_name]
        s_test     = scores.get(test_name, None)
        results.append((best_name, best_score, s_test))
    return results

# ══════════════════════════════════════════════════════════════
# SIMULATION MONTE CARLO PAR INDIVIDU
# ══════════════════════════════════════════════════════════════
def simulate_individual(name, embs, gallery_exemplars, wild_embs, n_values, k_mc, seed):
    """
    Retourne dict :
      {N: {"tp_score_mean", "tp_score_std",
           "tp_rate_mean",  "tp_rate_std",
           "fp_rate_mean",  "fp_rate_std"}}
    """
    # Galerie sans l'individu test
    gallery_without = {k: v for k, v in gallery_exemplars.items() if k != name}

    train_idx, test_idx = make_split(len(embs), seed)
    if len(test_idx) < 1 or len(train_idx) < 1:
        return None

    train_embs = embs[train_idx]
    test_embs  = embs[test_idx]
    n_train    = len(train_embs)

    rng     = np.random.default_rng(seed + 77)
    results = {}

    for N in n_values:
        if N > n_train and n_train < TRAIN_MIN:
            continue
        replace = N > n_train

        tp_scores_mc = np.empty(k_mc)
        tp_rates_mc  = np.empty(k_mc)
        fp_rates_mc  = np.empty(k_mc)

        for k in range(k_mc):
            # Construire le prototype (identique à l'app)
            chosen = rng.choice(n_train, size=N, replace=replace)
            proto  = train_embs[chosen].mean(axis=0)
            proto /= np.linalg.norm(proto) + 1e-8

            # ── TEST CROPS (individu test, photos non vues) ──
            tp_res   = score_against_gallery(test_embs, gallery_without, proto, name)
            scores_vs_proto = np.array([r[2] for r in tp_res])
            tp_scores_mc[k] = float(scores_vs_proto.mean())

            # TP correct = top match est l'individu ET score ≥ seuil
            tp_rates_mc[k] = float(np.mean(
                [(r[0] == name and r[1] >= THRESHOLD) for r in tp_res]
            ))

            # ── WILD CROPS (inconnus) ──
            if len(wild_embs) > 0:
                fp_res = score_against_gallery(wild_embs, gallery_without, proto, name)
                fp_rates_mc[k] = float(np.mean([r[1] >= THRESHOLD for r in fp_res]))
            else:
                fp_rates_mc[k] = 0.0

        results[N] = {
            "tp_score_mean": float(tp_scores_mc.mean()),
            "tp_score_std":  float(tp_scores_mc.std()),
            "tp_rate_mean":  float(tp_rates_mc.mean()),
            "tp_rate_std":   float(tp_rates_mc.std()),
            "fp_rate_mean":  float(fp_rates_mc.mean()),
            "fp_rate_std":   float(fp_rates_mc.std()),
            "with_replacement": replace,
        }

    return results

# ══════════════════════════════════════════════════════════════
# FIGURES — all labels in English
# ══════════════════════════════════════════════════════════════

def fig_main(all_results, n_values, out_path):
    """
    Main figure: recognition rate (%) vs N training photos.
    - Faded lines per individual (shows honest spread).
    - Bold mean line + ±1 std band.
    - Key numbers annotated on the mean curve.
    This is the graph to send to Cédric / Indonesian colleagues.
    """
    names   = sorted(all_results.keys())
    n_ind   = len(names)
    palette = plt.cm.tab20(np.linspace(0, 1, max(n_ind, 2)))

    fig, ax = plt.subplots(figsize=(12, 7))

    # Per-individual faded lines
    rate_by_n = {n: [] for n in n_values}
    for ci, name in enumerate(names):
        res = all_results[name]
        ns  = sorted(res.keys())
        rates = [res[n]["tp_rate_mean"] * 100 for n in ns]
        ax.plot(ns, rates, color=palette[ci], lw=1.2, alpha=0.45,
                marker="o", markersize=3, label=name)
        for n in ns:
            rate_by_n[n].append(res[n]["tp_rate_mean"] * 100)

    # Mean ± std across individuals
    ns_shared = [n for n in n_values if len(rate_by_n[n]) >= 2]
    grand_mean = np.array([np.mean(rate_by_n[n]) for n in ns_shared])
    grand_std  = np.array([np.std( rate_by_n[n]) for n in ns_shared])

    ax.plot(ns_shared, grand_mean,
            color=C_MEAN, lw=3, marker="o", markersize=7, zorder=8,
            label=f"Mean ({n_ind} individuals)")
    ax.fill_between(ns_shared,
                     grand_mean - grand_std,
                     grand_mean + grand_std,
                     color=C_BAND, alpha=0.18, zorder=7, label="±1 std")

    # Annotate key N values on the mean curve
    annotate_at = [n for n in [1, 5, 10, 20, 25] if n in ns_shared]
    for n in annotate_at:
        idx = ns_shared.index(n)
        y   = grand_mean[idx]
        ax.annotate(f"{y:.1f}%",
                    xy=(n, y), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=8.5, color=C_MEAN, fontweight="bold",
                    zorder=9)

    # Plateau detection on mean curve
    for i in range(1, len(ns_shared)):
        if abs(grand_mean[i] - grand_mean[i-1]) < 0.5:   # < 0.5% gain
            p_n = ns_shared[i]
            ax.axvline(p_n, color="gray", lw=1.3, linestyle="--", alpha=0.65, zorder=6)
            ax.text(p_n + 0.3, 20, f"plateau ≈ N={p_n}",
                    rotation=90, va="bottom", fontsize=9, color="gray")
            break

    ax.set_xlabel("Number of training photos (N)", fontsize=13)
    ax.set_ylabel("Recognition rate (%)", fontsize=13)
    ax.set_title(
        f"OrangIdentifier V6 — Recognition rate vs. number of training photos\n"
        f"({n_ind} individuals · 25% hold-out · Monte Carlo K={K_MC} · field simulation)",
        fontsize=12,
    )
    ax.set_xlim(0, max(n_values) + 1)
    ax.set_ylim(0, 108)
    ax.xaxis.set_major_locator(mticker.FixedLocator(n_values))
    ax.legend(fontsize=7.5, loc="lower right", ncol=2, framealpha=0.9)
    ax.grid(alpha=0.22)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path.name}")


def fig_per_individual(all_results, n_values, out_path):
    """
    5×3 grid (or ceil(N/3)×3): recognition rate per individual.
    Shows which individuals are easy vs hard — honest, no smoothing.
    """
    names = sorted(all_results.keys())
    ncols = 3
    nrows = -(-len(names) // ncols)   # ceil division
    palette = plt.cm.tab20(np.linspace(0, 1, max(len(names), 2)))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 3.8 * nrows), squeeze=False)
    fig.suptitle(
        f"OrangIdentifier V6 — Recognition rate per individual\n"
        f"(field simulation: individual removed from gallery, N random photos → prototype)",
        fontsize=12, y=1.01,
    )

    for ci, name in enumerate(names):
        row, col = divmod(ci, ncols)
        ax  = axes[row][col]
        res = all_results[name]
        ns  = sorted(res.keys())

        rates = np.array([res[n]["tp_rate_mean"] * 100 for n in ns])
        stds  = np.array([res[n]["tp_rate_std"]  * 100 for n in ns])

        ax.plot(ns, rates, color=palette[ci], lw=2, marker="o", markersize=4)
        ax.fill_between(ns, rates - stds, rates + stds,
                         color=palette[ci], alpha=0.18)
        ax.axhline(100, color="lightgray", lw=0.8, linestyle="--")

        # Plateau marker
        p_n = ns[-1]
        for i in range(1, len(ns)):
            if abs(rates[i] - rates[i-1]) < 0.5:
                p_n = ns[i]; break
        ax.axvline(p_n, color="gray", lw=1, linestyle="--", alpha=0.55)

        flag = "  !" if max(rates) < 90 else ""
        ax.set_title(f"{name}{flag}", fontsize=10, fontweight="bold")
        ax.set_xlabel("N training photos", fontsize=8)
        ax.set_ylabel("Recognition rate (%)", fontsize=8)
        ax.set_ylim(0, 108)
        ax.xaxis.set_major_locator(mticker.FixedLocator(
            [n for n in n_values if n <= max(ns)]))
        ax.grid(alpha=0.2)

    # Hide empty subplots
    for idx in range(len(names), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path.name}")


def fig_table(all_results, n_values, out_path):
    """
    Summary table exactly matching Cédric's requested format:
      Rows = N photos (1, 5, 10, 15, 20, 25)
      Cols = individual names + MEAN column
      Values = recognition rate %
    """
    table_ns = [n for n in [1, 5, 10, 15, 20, 25] if n in n_values]
    names    = sorted(all_results.keys())

    # Build data matrix
    rows = []
    for n in table_ns:
        row = [f"{n}"]
        rates = []
        for name in names:
            res = all_results[name]
            r   = res.get(n, {}).get("tp_rate_mean", float("nan")) * 100
            row.append(f"{r:.1f}%" if not np.isnan(r) else "—")
            if not np.isnan(r):
                rates.append(r)
        mean_r = np.mean(rates) if rates else float("nan")
        row.append(f"{mean_r:.1f}%")
        rows.append(row)

    cols = ["N photos"] + names + ["MEAN"]

    fig_w = max(10, 0.9 * len(cols))
    fig_h = 0.55 * len(table_ns) + 2.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tab = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1.0, 2.0)

    # Header style
    for j in range(len(cols)):
        cell = tab[0, j]
        cell.set_facecolor("#1e3a5f")
        cell.set_text_props(color="white", fontweight="bold")

    # MEAN column highlighted
    mean_col = len(cols) - 1
    for i in range(1, len(rows) + 1):
        tab[i, mean_col].set_facecolor("#dbeafe")
        tab[i, mean_col].set_text_props(fontweight="bold")

    # Row alternating
    for i in range(1, len(rows) + 1):
        for j in range(len(cols) - 1):
            tab[i, j].set_facecolor("#f0fdf4" if i % 2 == 0 else "white")

    ax.set_title(
        f"OrangIdentifier V6 — Recognition rate by number of training photos\n"
        f"({len(names)} individuals · 25% hold-out test set · field simulation)",
        fontsize=10, fontweight="bold", pad=16,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path.name}")


# ══════════════════════════════════════════════════════════════
# TERMINAL SUMMARY
# ══════════════════════════════════════════════════════════════
def print_summary(all_results):
    names = sorted(all_results.keys())
    table_ns = [n for n in [1, 5, 10, 15, 20, 25]
                if any(n in r for r in all_results.values())]

    print(f"\n{'═'*72}")
    print(f"  V6 FIELD SIMULATION — Recognition rate by N training photos")
    print(f"  (K={K_MC} Monte Carlo · 25% hold-out · {len(names)} individuals)")
    print(f"{'═'*72}")
    header = f"  {'Individual':<14}" + "".join(f"  N={n:>2}" for n in table_ns)
    print(header)
    print(f"  {'─'*60}")

    for name in names:
        res  = all_results[name]
        line = f"  {name:<14}"
        for n in table_ns:
            r = res.get(n, {}).get("tp_rate_mean", float("nan")) * 100
            line += f"  {r:>5.1f}%" if not np.isnan(r) else "     —"
        print(line)

    # Mean row
    print(f"  {'─'*60}")
    mean_line = f"  {'MEAN':<14}"
    for n in table_ns:
        rates = [all_results[nm].get(n, {}).get("tp_rate_mean", float("nan")) * 100
                 for nm in names]
        rates = [r for r in rates if not np.isnan(r)]
        mean_line += f"  {np.mean(rates):>5.1f}%" if rates else "     —"
    print(mean_line)
    print(f"{'═'*72}\n")

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import time
    t0 = time.time()
    random.seed(SEED); np.random.seed(SEED)

    print(f"Device : {DEVICE}")
    print(f"Config : hold-out {int(TEST_FRAC*100)}%, Monte Carlo K={K_MC}, N_WILD={N_WILD}\n")

    # 1. Charger galerie (exemplars précomputes, pas de reforward)
    gallery_exemplars = load_gallery()
    all_gallery_names = list(gallery_exemplars.keys())

    # 2. Embeddings pour TOUS les individus (nécessaire pour la vraie auto-sélection)
    zoo_embs, wild_embs = build_or_load_cache(all_gallery_names, recache=args.recache)

    # 3. Sélectionner les individus à simuler
    if TEST_INDIVIDUALS:
        test_names = list(TEST_INDIVIDUALS)
        for n in test_names:
            if n not in zoo_embs:
                raise ValueError(f"'{n}' not found in cache (check ZOO_DIR / NEW_ZOO)")
        print(f"\nTest individuals (manual): {test_names}")
    else:
        # Run on ALL gallery individuals — honest, no cherry-picking
        test_names = sorted(zoo_embs.keys())
        print(f"\nRunning on all {len(test_names)} gallery individuals...")

    # Filter individuals with too few crops
    valid_names = []
    print()
    for name in test_names:
        n = len(zoo_embs.get(name, []))
        if n < MIN_CROPS:
            print(f"  {name}: skipped ({n} crops < minimum {MIN_CROPS})")
        else:
            valid_names.append(name)

    if not valid_names:
        raise RuntimeError("No valid individuals found. Check ZOO_DIR / NEW_ZOO paths.")

    # 4. Monte Carlo simulation
    print(f"\nMonte Carlo simulation (K={K_MC} repetitions per N)...")
    all_results = {}
    for ci, name in enumerate(valid_names):
        embs = zoo_embs[name]
        res  = simulate_individual(
            name, embs, gallery_exemplars, wild_embs,
            N_VALUES, K_MC, seed=SEED + ci,
        )
        if res is None:
            print(f"  {name}: skipped (not enough crops for split)")
            continue
        all_results[name] = res

        n_train = len(embs) - max(TEST_MIN, int(np.ceil(len(embs) * TEST_FRAC)))
        n_test  = len(embs) - n_train
        r5  = res.get(5,  {}).get("tp_rate_mean", float("nan")) * 100
        r10 = res.get(10, {}).get("tp_rate_mean", float("nan")) * 100
        flag = "  ! LOW" if r10 < 80 else ""
        print(f"  {name:<14}: train={n_train:3d} test={n_test:3d} | "
              f"rate@5={r5:.0f}%  rate@10={r10:.0f}%{flag}")

    # 5. Figures
    RESULTS.mkdir(parents=True, exist_ok=True)
    print("\nGenerating figures...")
    fig_main(all_results, N_VALUES,
             RESULTS / "04_recognition_vs_nPhotos.png")
    fig_per_individual(all_results, N_VALUES,
                       RESULTS / "04_recognition_per_individual.png")
    fig_table(all_results, N_VALUES,
              RESULTS / "04_recognition_table.png")

    # 6. Résumé
    print_summary(all_results)
    elapsed = time.time() - t0
    print(f"Total time: {elapsed/60:.1f} min")
    print(f"Figures saved to: {RESULTS}")