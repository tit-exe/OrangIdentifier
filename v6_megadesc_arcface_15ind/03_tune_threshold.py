"""
V6_plateau_v2.py — OrangIdentifier V6
=======================================
Graph "number of training photos -> recognition confidence".

Protocol (no data leakage):
  For each zoo individual with >= MIN_TOTAL crops:
    1. Split TRAIN 75% / TEST 25%, deterministic (seed=42+i), BEFORE any
       quality or centroid computation.
    2. For N in N_VALUES, repeat K_MC times:
         - draw N random crops from TRAIN (without replacement*)
         - prototype = L2_normalize( mean( embs[drawn] ) )
         - score_TP  = mean( dot(prototype, TEST embs) )
       -> mean_TP[N], std_TP[N]  (over K_MC draws = real variance)
  *with replacement if N > len(train), equivalent to reusing the full
  centroid, reported in the terminal summary.

Guarantees:
  - test set frozen BEFORE any computation      [no leak]
  - no embedding information in the split        [no leak]
  - random sampling (not top-N quality) = realistic field scenario
  - K_MC=200 draws -> honest variance            [robust]
  - embeddings cached on disk -> reproducible    [audit]

FP rate: fraction of wild crops whose score exceeds the threshold against
         AT LEAST ONE individual of the simulated gallery -> field metric.

Sorties (dans results/) :
  03a_v2_plateau_confidence.png  — figure principale (confiance vs N)
  03b_v2_plateau_tradeoff.png    — variance MC + FP rate vs N
  03c_v2_plateau_table.png       - summary table

RUN :
    conda activate orangs
    python v6_megadesc_arcface_15ind/03_tune_threshold.py

    # Recompute embeddings if new crops were added:
    python v6_megadesc_arcface_15ind/03_tune_threshold.py --recache
"""

import argparse
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

# ══════════════════════════════════════════════════════════════════
# CONFIG — modifier ici si besoin
# ══════════════════════════════════════════════════════════════════
V6_BEST  = (REPO / "output" / "v6" / "models" / "v6_best.pt")
ZOO_DIR  = (REPO / "data" / "crops" / "known")
NEW_ZOO  = (REPO / "data" / "crops" / "new")
WILD_DIR = (REPO / "data" / "crops" / "wild")
RESULTS  = (REPO / "output" / "v6" / "results")
CACHE_F  = RESULTS / "emb_cache_plateau_v2.npz"

THRESHOLD = 0.5371      # V6 gallery threshold

SEED      = 42
TEST_FRAC = 0.25        # hold-out fraction for the queries
TEST_MIN  = 6           # minimum absolu de crops dans le test set
TRAIN_MIN = 5           # minimum absolu de crops dans le train pool
MIN_TOTAL = 15          # ignorer individus avec < MIN_TOTAL crops

N_VALUES  = [1, 2, 3, 4, 5, 7, 10, 15, 20, 25]
K_MC      = 200         # Monte Carlo repetitions per N
N_WILD    = 500         # max crops wild pour l'analyse FP
BATCH     = 64
IMG_SIZE  = 224
EXTS      = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Couleurs projet
C_MEAN      = "#27500A"   # green_800 (moyenne)
C_FILL      = "#639922"   # green_400 (bande ±std)
C_THRESHOLD = "#A32D2D"   # confidence_low (threshold)
C_FP        = "#c4501b"   # dark orange (FP)

# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════
ap = argparse.ArgumentParser(description="V6 plateau confidence script")
ap.add_argument("--recache", action="store_true",
                help="Recalculer les embeddings (ignorer le cache existant)")
args = ap.parse_args()

# ══════════════════════════════════════════════════════════════════
# DATASET / TRANSFORMS
# ══════════════════════════════════════════════════════════════════
_tf = T.Compose([
    T.Resize(IMG_SIZE),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

class _PathDS(Dataset):
    def __init__(self, paths):
        self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        return _tf(img)


def collect_images(base_dirs, exclude=None):
    """Collects images per individual from several source folders."""
    excl = set(exclude or [])
    per_individual = {}
    for base in base_dirs:
        if not Path(base).exists():
            continue
        for d in sorted(Path(base).iterdir()):
            if not d.is_dir() or d.name.startswith("_") or d.name in excl:
                continue
            imgs = [f for f in d.iterdir() if f.suffix in EXTS]
            if imgs:
                per_individual.setdefault(d.name, []).extend(imgs)
    # Deduplicate and sort so the order is deterministic
    return {k: sorted(set(v)) for k, v in per_individual.items()}


@torch.no_grad()
def embed_paths(backbone, paths):
    """Returns L2-normalised embeddings, shape (len(paths), 768)."""
    if not paths:
        return np.zeros((0, 768), dtype=np.float32)
    dl  = DataLoader(_PathDS(paths), batch_size=BATCH, num_workers=0)
    out = []
    for imgs in dl:
        out.append(F.normalize(backbone(imgs.to(DEVICE)), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)

# ══════════════════════════════════════════════════════════════════
# BACKBONE
# ══════════════════════════════════════════════════════════════════
def load_backbone():
    print("Loading V6 backbone...")
    if not V6_BEST.exists():
        raise FileNotFoundError(f"Checkpoint not found : {V6_BEST}")
    bb = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224", pretrained=False, num_classes=0)
    ck = torch.load(str(V6_BEST), map_location="cpu", weights_only=False)
    state = ck.get("backbone_state") or ck.get("model_state_dict") or ck
    bb.load_state_dict(state, strict=False)
    return bb.to(DEVICE).eval()

# ══════════════════════════════════════════════════════════════════
# CACHE EMBEDDINGS
# ══════════════════════════════════════════════════════════════════
def build_or_load_cache(recache=False):
    RESULTS.mkdir(parents=True, exist_ok=True)

    if CACHE_F.exists() and not recache:
        print(f"Cache found : {CACHE_F.name}  (--recache to recompute)")
        data = np.load(CACHE_F, allow_pickle=True)
        names     = list(data["individuals"])
        zoo_embs  = {n: data[f"zoo_{n}"] for n in names}
        wild_embs = data["wild_embs"]
        print(f"  {len(zoo_embs)} individus | {len(wild_embs)} crops wild")
        return zoo_embs, wild_embs

    backbone = load_backbone()

    print("\nCollecte images zoo...")
    zoo_images = collect_images([ZOO_DIR, NEW_ZOO], exclude=["_a_verifier"])
    for name, imgs in sorted(zoo_images.items()):
        print(f"  {name:<14}: {len(imgs):4d} images")

    print("\nExtraction embeddings zoo...")
    zoo_embs = {}
    for name, paths in sorted(zoo_images.items()):
        zoo_embs[name] = embed_paths(backbone, paths)
        print(f"  {name:<14}: {zoo_embs[name].shape[0]:4d} embeddings")

    print(f"\nCollecte wild (max {N_WILD})...")
    wild_paths = []
    if WILD_DIR.exists():
        all_wild = sorted([f for f in WILD_DIR.rglob("*") if f.suffix in EXTS])
        rng = random.Random(SEED)
        wild_paths = rng.sample(all_wild, min(N_WILD, len(all_wild)))
    wild_embs = embed_paths(backbone, wild_paths)
    print(f"  {len(wild_embs):4d} embeddings wild")

    save = {f"zoo_{k}": v for k, v in zoo_embs.items()}
    save["wild_embs"]    = wild_embs
    save["individuals"]  = np.array(list(zoo_embs.keys()))
    np.savez_compressed(CACHE_F, **save)
    print(f"  Cache saved : {CACHE_F}")

    return zoo_embs, wild_embs

# ══════════════════════════════════════════════════════════════════
# SPLIT TRAIN / TEST — sans fuite
# ══════════════════════════════════════════════════════════════════
def make_split(n_total, seed):
    """
    Deterministic TRAIN/TEST split based only on n_total and seed.
    Aucune info des embeddings n'influence le split.
    Retourne (train_idx, test_idx) en numpy int arrays.
    """
    rng    = np.random.default_rng(seed)
    idx    = rng.permutation(n_total)
    n_test = max(TEST_MIN, int(np.ceil(n_total * TEST_FRAC)))
    # S'assurer qu'il reste au moins TRAIN_MIN dans le train
    n_test = min(n_test, n_total - TRAIN_MIN)
    if n_test <= 0:
        return idx, np.array([], dtype=int)
    return idx[n_test:], idx[:n_test]   # train, test

# ══════════════════════════════════════════════════════════════════
# MONTE CARLO PAR INDIVIDU
# ══════════════════════════════════════════════════════════════════
def analyse_individual(name, embs, splits, n_values, k_mc):
    """
    Retourne dict {N: {"mean": float, "std": float, "q25": float, "q75": float}}.
    splits : (train_idx, test_idx) precomputed.
    """
    train_idx, test_idx = splits

    if len(test_idx) < 1 or len(train_idx) < 1:
        return None

    train_embs = embs[train_idx]  # (n_train, 768)
    test_embs  = embs[test_idx]   # (n_test,  768)

    rng     = np.random.default_rng(SEED + hash(name) % (2**31))
    results = {}
    n_train = len(train_embs)

    for N in n_values:
        if N > n_train and n_train < TRAIN_MIN:
            continue

        with_replacement = N > n_train
        scores_mc = np.empty(k_mc, dtype=np.float32)

        for k in range(k_mc):
            chosen = rng.choice(n_train, size=N, replace=with_replacement)
            proto  = train_embs[chosen].mean(axis=0)
            norm   = np.linalg.norm(proto)
            if norm < 1e-8:
                scores_mc[k] = 0.0
                continue
            proto /= norm
            scores_mc[k] = float((test_embs @ proto).mean())

        results[N] = {
            "mean": float(scores_mc.mean()),
            "std":  float(scores_mc.std()),
            "q25":  float(np.percentile(scores_mc, 25)),
            "q75":  float(np.percentile(scores_mc, 75)),
            "with_replacement": with_replacement,
        }

    return results

# ══════════════════════════════════════════════════════════════════
# FP ANALYSIS (wild against the full gallery)
# ══════════════════════════════════════════════════════════════════
def analyse_fp(wild_embs, zoo_embs, all_splits, n_values, k_mc):
    """
    For each N: fraction of wild crops whose score exceeds the threshold
    against AT LEAST ONE individual of the simulated gallery.
    Uses the same splits as the TP analysis.
    """
    if len(wild_embs) == 0:
        print("  [WARNING] No wild crops, FP rate not computed.")
        return {}

    # Valid individuals for the simulated gallery
    valid = [(n, zoo_embs[n], all_splits[n][0])   # name, embs, train_idx
             for n in sorted(all_splits)
             if len(all_splits[n][0]) >= 1]

    if not valid:
        return {}

    rng     = np.random.default_rng(SEED + 9999)
    fp_res  = {}

    for N in n_values:
        fp_rates = np.empty(k_mc, dtype=np.float32)

        for k in range(k_mc):
            protos = []
            for _, embs, t_idx in valid:
                n_train = len(t_idx)
                replace = N > n_train
                chosen  = rng.choice(n_train, size=N, replace=replace)
                proto   = embs[t_idx[chosen]].mean(axis=0)
                norm    = np.linalg.norm(proto)
                if norm > 1e-8:
                    protos.append(proto / norm)

            if not protos:
                fp_rates[k] = 0.0
                continue

            # Score of each wild crop against the best individual of the gallery
            protos_mat  = np.stack(protos, axis=0)   # (n_ind, 768)
            best_scores = (wild_embs @ protos_mat.T).max(axis=1)  # (n_wild,)
            fp_rates[k] = float((best_scores >= THRESHOLD).mean())

        fp_res[N] = {
            "mean": float(fp_rates.mean()),
            "std":  float(fp_rates.std()),
        }

    return fp_res

# ══════════════════════════════════════════════════════════════════
# DÉTECTION PLATEAU
# ══════════════════════════════════════════════════════════════════
def detect_plateau(res, tol=0.005):
    """First N where the absolute gain on the mean is < tol."""
    ns    = sorted(res.keys())
    means = [res[n]["mean"] for n in ns]
    for i in range(1, len(ns)):
        if abs(means[i] - means[i-1]) < tol:
            return ns[i]
    return ns[-1]

# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — Confiance vs N (figure principale)
# ══════════════════════════════════════════════════════════════════
def fig_confidence(all_results, n_values, out_path):
    palette = plt.cm.tab20(np.linspace(0, 1, max(len(all_results), 2)))
    fig, ax = plt.subplots(figsize=(12, 7))

    all_means_by_n = {N: [] for N in n_values}

    for ci, (name, res) in enumerate(sorted(all_results.items())):
        ns    = sorted(res.keys())
        means = [res[n]["mean"] for n in ns]

        ax.plot(ns, means,
                color=palette[ci % len(palette)], lw=1.3, alpha=0.55,
                marker="o", markersize=3.5,
                label=f"{name}")

        # Marquer le plateau
        p = detect_plateau(res)
        if p in res:
            pi = ns.index(p)
            ax.scatter([p], [means[pi]],
                       color=palette[ci % len(palette)], s=80, zorder=5,
                       edgecolors="white", linewidths=1)

        for n in ns:
            all_means_by_n[n].append(res[n]["mean"])

    # Bande moyenne inter-individus
    ns_shared = [n for n in n_values if len(all_means_by_n[n]) >= 2]
    if ns_shared:
        grand_mean = np.array([np.mean(all_means_by_n[n]) for n in ns_shared])
        grand_std  = np.array([np.std( all_means_by_n[n]) for n in ns_shared])

        ax.plot(ns_shared, grand_mean,
                color=C_MEAN, lw=3, linestyle="--", zorder=8,
                label=f"Moyenne ({len(all_results)} individus)")
        ax.fill_between(ns_shared,
                         grand_mean - grand_std,
                         grand_mean + grand_std,
                         color=C_FILL, alpha=0.15, zorder=7,
                         label="±1 std inter-individus")

        # Annotation plateau
        deltas = np.diff(grand_mean)
        for i, d in enumerate(deltas):
            if abs(d) < 0.005:
                p_n = ns_shared[i + 1]
                p_y = grand_mean[i + 1]
                ax.axvline(p_n, color="gray", lw=1.3, linestyle="--", alpha=0.6, zorder=6)
                ax.annotate(
                    f"plateau ≈ N={p_n}",
                    xy=(p_n, p_y),
                    xytext=(p_n + 0.6, max(0.35, p_y - 0.12)),
                    fontsize=9, color="gray",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
                )
                break

    # Rejection threshold
    ax.axhline(THRESHOLD, color=C_THRESHOLD, lw=1.8, linestyle=":",
               zorder=9, label=f"Rejection threshold ({THRESHOLD:.4f})")

    ax.set_xlabel("Number of training photos (N)", fontsize=12)
    ax.set_ylabel("Recognition confidence (cosine similarity)", fontsize=12)
    ax.set_title(
        f"V6 - Recognition confidence vs. number of training photos\n"
        f"({len(all_results)} individuals - hold-out 25% - Monte Carlo K={K_MC} - random sampling)",
        fontsize=11,
    )
    ax.set_xlim(0, max(n_values) + 1)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(mticker.FixedLocator(n_values))
    ax.legend(fontsize=7.5, loc="lower right", ncol=2, framealpha=0.9)
    ax.grid(alpha=0.22)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path.name}")


# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — Variance MC + FP rate
# ══════════════════════════════════════════════════════════════════
def fig_tradeoff(all_results, fp_res, n_values, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f"V6 - Prototype stability (Monte Carlo variance) and false positive risk",
        fontsize=12,
    )

    ns_avail = [n for n in n_values
                if any(n in r for r in all_results.values())]

    # ── Gauche : variance MC (std) en fonction de N ──
    ax = axes[0]
    tp_stds_per_ind = {n: [] for n in ns_avail}
    for res in all_results.values():
        for n in ns_avail:
            if n in res:
                tp_stds_per_ind[n].append(res[n]["std"])

    std_means = [np.mean(tp_stds_per_ind[n]) * 100 if tp_stds_per_ind[n] else 0
                 for n in ns_avail]
    std_maxs  = [np.max( tp_stds_per_ind[n]) * 100 if tp_stds_per_ind[n] else 0
                 for n in ns_avail]

    ax.plot(ns_avail, std_means, color=C_MEAN, lw=2.5, marker="o", markersize=6,
            label="Mean MC Std (all individuals)")
    ax.plot(ns_avail, std_maxs,  color=C_FILL, lw=1.5, marker="s", markersize=4,
            linestyle="--", label="Std MC max (pire individu)")
    ax.fill_between(ns_avail, std_means, std_maxs, alpha=0.15, color=C_FILL)
    ax.set_xlabel("N exemplaires", fontsize=11)
    ax.set_ylabel("Écart-type Monte Carlo (×100)", fontsize=11)
    ax.set_title(
        "Prototype variance vs N photos\n(lower = stable prototype = reproducible in the field)",
        fontsize=10,
    )
    ax.set_xlim(0, max(n_values) + 1)
    ax.xaxis.set_major_locator(mticker.FixedLocator(ns_avail))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # ── Droite : FP rate en fonction de N ──
    ax = axes[1]
    if fp_res:
        ns_fp   = [n for n in ns_avail if n in fp_res]
        fp_mean = [fp_res[n]["mean"] * 100 for n in ns_fp]
        fp_std  = [fp_res[n]["std"]  * 100 for n in ns_fp]

        ax.plot(ns_fp, fp_mean, color=C_FP, lw=2.5, marker="o", markersize=6,
                label="Taux FP moyen")
        ax.fill_between(ns_fp,
                         [m - s for m, s in zip(fp_mean, fp_std)],
                         [m + s for m, s in zip(fp_mean, fp_std)],
                         alpha=0.2, color=C_FP, label="±1 std")
        ax.set_ylim(0)
    else:
        ax.text(0.5, 0.5, "Wild data unavailable",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)

    ax.set_xlabel("N exemplaires", fontsize=11)
    ax.set_ylabel("Wild crops accepted as known (%)", fontsize=11)
    ax.set_title(
        "Taux de faux positifs en terrain vs N\n"
        "(wild crops scored against the whole simulated gallery)",
        fontsize=10,
    )
    ax.set_xlim(0, max(n_values) + 1)
    ax.xaxis.set_major_locator(mticker.FixedLocator(ns_avail))
    if fp_res:
        ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path.name}")


# ══════════════════════════════════════════════════════════════════
# FIGURE 3 - Summary table
# ══════════════════════════════════════════════════════════════════
def fig_table(all_results, n_values, out_path):
    rows = []
    for name, res in sorted(all_results.items()):
        ns    = sorted(res.keys())
        means = [res[n]["mean"] for n in ns]
        cmax  = max(means)

        def fmt(n):
            return f"{res[n]['mean']:.3f}" if n in res else "—"

        plateau = detect_plateau(res)
        flag    = " !" if cmax < THRESHOLD else ""

        rows.append([
            f"{name}{flag}",
            fmt(5), fmt(10), fmt(20),
            f"{cmax:.3f}",
            str(plateau),
            f"{res[ns[0]]['std']*100:.2f}",
        ])

    cols = [
        "Individu", "Conf\nN=5", "Conf\nN=10", "Conf\nN=20",
        "Conf\nmax", "Plateau\nN≈", "Std MC\nN=1 (×100)",
    ]

    fig, ax = plt.subplots(figsize=(13, 0.58 * len(rows) + 1.8))
    ax.axis("off")

    tab = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1.0, 1.95)

    for j in range(len(cols)):
        tab[0, j].set_facecolor("#1e3a5f")
        tab[0, j].set_text_props(color="white", fontweight="bold")

    for i, row in enumerate(rows):
        under_thresh = "!" in row[0]
        for j in range(len(cols)):
            tab[i + 1, j].set_facecolor("#fef3c7" if under_thresh else "#f0fdf4")

    ax.set_title(
        f"V6 — Plateau de confiance (Monte Carlo K={K_MC}, hold-out 25%)\n"
        f"! = max conf. < threshold {THRESHOLD:.4f}   -   MC Std N=1 = variance with a single photo",
        fontsize=10, fontweight="bold", pad=16,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path.name}")


# ══════════════════════════════════════════════════════════════════
# RÉSUMÉ TERMINAL
# ══════════════════════════════════════════════════════════════════
def print_summary(all_results):
    print(f"\n{'═'*72}")
    print("  RÉSUMÉ PLATEAU — V6  (Monte Carlo K={}, hold-out {}%)".format(K_MC, int(TEST_FRAC*100)))
    print(f"{'═'*72}")
    print(f"  {'Individu':<14} {'Plateau≈':>10} {'Conf@5':>8} {'Conf@10':>9} {'Conf max':>10}")
    print(f"  {'─'*55}")

    plateaux = []
    for name, res in sorted(all_results.items()):
        ns    = sorted(res.keys())
        means = [res[n]["mean"] for n in ns]
        cmax  = max(means)
        p     = detect_plateau(res)
        plateaux.append(p)

        c5  = res.get(5,  {}).get("mean", float("nan"))
        c10 = res.get(10, {}).get("mean", float("nan"))
        flag = "  ! BELOW THRESHOLD" if cmax < THRESHOLD else ""

        wr_ns = [n for n in ns if res[n].get("with_replacement")]
        wr_note = f"  [sampling avec remise pour N>={min(wr_ns)}]" if wr_ns else ""

        print(f"  {name:<14} {p:>10}  {c5:>8.3f}  {c10:>9.3f}  {cmax:>9.3f}{flag}{wr_note}")

    print(f"\n  Median plateau            : N = {int(np.median(plateaux))}")
    print(f"  Plateau P75 (pire 25%)    : N = {int(np.percentile(plateaux, 75))}")
    print(f"\n  Recommandation terrain    : ≥ {int(np.percentile(plateaux, 75))} photos")
    print(f"  (beyond that the confidence gain is < 0.5 points)")
    print(f"{'═'*72}\n")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    random.seed(SEED); np.random.seed(SEED)

    print(f"Device    : {DEVICE}")
    print(f"Protocol : hold-out {int(TEST_FRAC*100)}%, Monte Carlo K={K_MC}, random sampling (field scenario)")

    # 1. Embeddings
    zoo_embs, wild_embs = build_or_load_cache(recache=args.recache)

    # 2. Filtrer individus insuffisants
    zoo_valid = {k: v for k, v in zoo_embs.items() if len(v) >= MIN_TOTAL}
    skipped   = sorted(set(zoo_embs) - set(zoo_valid))
    if skipped:
        print(f"\nSkipped (< {MIN_TOTAL} crops) : {', '.join(skipped)}")
    print(f"\n{len(zoo_valid)} individuals analysed\n")

    # 3. Precompute the splits (one call per individual -> TP / FP consistency)
    all_splits = {}
    for ci, name in enumerate(sorted(zoo_valid)):
        train_idx, test_idx = make_split(len(zoo_valid[name]), seed=SEED + ci)
        all_splits[name] = (train_idx, test_idx)
        n_t = len(train_idx); n_e = len(test_idx)
        print(f"  Split {name:<14}: train={n_t:3d}  test={n_e:3d}")

    # 4. Monte Carlo par individu
    print(f"\nAnalyse Monte Carlo (K={K_MC})...")
    all_results = {}
    for ci, name in enumerate(sorted(zoo_valid)):
        res = analyse_individual(
            name, zoo_valid[name], all_splits[name], N_VALUES, K_MC
        )
        if res:
            all_results[name] = res
            p    = detect_plateau(res)
            cmax = max(r["mean"] for r in res.values())
            c5   = res.get(5, {}).get("mean", float("nan"))
            flag = " !" if cmax < THRESHOLD else ""
            print(f"  {name:<14}: conf@5={c5:.3f}  conf_max={cmax:.3f}  plateau≈N={p}{flag}")
        else:
            print(f"  {name:<14}: skipped (insufficient split)")

    # 5. Analyse FP
    print("\nAnalyse FP wild...")
    fp_results = analyse_fp(wild_embs, zoo_valid, all_splits, N_VALUES, K_MC)
    if fp_results:
        fp_mean_at_10 = fp_results.get(10, {}).get("mean", float("nan"))
        print(f"  Mean FP rate at N=10 : {fp_mean_at_10*100:.2f}%")
    else:
        print("  (no wild data)")

    # 6. Figures
    RESULTS.mkdir(parents=True, exist_ok=True)
    print("\nGenerating figures...")
    fig_confidence(all_results, N_VALUES,
                   RESULTS / "03a_v2_plateau_confidence.png")
    fig_tradeoff(all_results, fp_results, N_VALUES,
                 RESULTS / "03b_v2_plateau_tradeoff.png")
    fig_table(all_results, N_VALUES,
              RESULTS / "03c_v2_plateau_table.png")

    # 7. Summary
    print_summary(all_results)
    print(f"Figures dans : {RESULTS}\n")