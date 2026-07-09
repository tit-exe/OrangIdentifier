# V3_visualize_model.py
# Orang-outan V2 pipeline
# 
#
# Comprehensive visual diagnostic of the current model.
# Generates multiple PNG files in output/v3_reports/
#
# What it produces:
#   1. umap_embedding_space.png  - 2D map of ALL individuals (zoo + BOS + wild)
#   2. similarity_distributions.png - histograms showing separability
#   3. individual_grid.png       - sample crops per individual with their scores
#   4. confusion_heatmap.png     - who gets confused with who
#   5. wild_crops_rejection.png  - how wild internet crops are handled
#
# RUN:
#   conda activate orangs
#   pip install umap-learn --break-system-packages  (if not installed)
#   python v3_megadesc_arcface_10ind/07_visualize.py

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
import random
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime
from collections import defaultdict




import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import timm
from PIL import Image, ImageFile, ImageDraw, ImageFont
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

# ==============================================================================
# PATHS
# ==============================================================================

MODEL_PATH = V3_PT
ZOO_DIR = CROPS_KNOWN_DIR
BOS_DIR = CROPS_KNOWN_DIR
WILD_DIR = CROPS_WILD_DIR
OUT_DIR = OUTPUT_DIR / "v3_reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE    = 224
BATCH_SIZE  = 32
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEGA_MEAN   = [0.5, 0.5, 0.5]
MEGA_STD    = [0.5, 0.5, 0.5]
THRESHOLD   = 0.22

TRANSFORM = T.Compose([
    T.Resize(IMG_SIZE),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(MEGA_MEAN, MEGA_STD),
])

random.seed(42)
np.random.seed(42)

# ==============================================================================
# HELPERS
# ==============================================================================

def title(text):
    print(f"\n{'='*70}\n  {text}\n{'='*70}")

def step(text):
    print(f"  -> {text}")

@torch.no_grad()
def embed_paths(model, paths, label=""):
    embs = []
    n = len(paths)
    for i in range(0, n, BATCH_SIZE):
        batch = []
        for p in paths[i:i+BATCH_SIZE]:
            try:
                img = Image.open(p).convert("RGB")
                batch.append(TRANSFORM(img))
            except:
                batch.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))
        t = torch.stack(batch).to(DEVICE, non_blocking=True)
        embs.append(F.normalize(model(t), dim=1).cpu())
        done = min(i + BATCH_SIZE, n)
        print(f"\r    {label}: {done}/{n} ({done*100//n}%)", end="", flush=True)
    print()
    return torch.cat(embs, dim=0)

def load_img_rgb(path, size=112):
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        return np.array(img)
    except:
        return np.zeros((size, size, 3), dtype=np.uint8)

# ==============================================================================
# LOAD MODEL
# ==============================================================================

title("Loading model")
ckpt    = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
classes = ckpt["classes"]
emb_dim = ckpt.get("emb_dim", 768)

model = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224",
                           pretrained=False, num_classes=0)
model.load_state_dict(ckpt["backbone_state"])
model = model.eval().to(DEVICE)
step(f"Model loaded - epoch {ckpt.get('epoch','?')} - val_acc {ckpt.get('val_acc',0)*100:.1f}%")

# ==============================================================================
# COLLECT PATHS
# ==============================================================================

title("Collecting image paths")

exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

zoo_paths  = {}   # class -> [paths]
zoo_labels = []   # per-image label index
zoo_plist  = []   # flat path list

for i, cls in enumerate(classes):
    d = ZOO_DIR / cls
    imgs = sorted([f for f in d.iterdir() if f.suffix in exts]) if d.exists() else []
    zoo_paths[cls] = imgs
    zoo_plist.extend(imgs)
    zoo_labels.extend([i] * len(imgs))
    step(f"Zoo {cls:<12}: {len(imgs)} crops")

bos_paths  = {}
bos_names  = sorted([d.name for d in BOS_DIR.iterdir() if d.is_dir()])
for name in bos_names:
    imgs = sorted([f for f in (BOS_DIR/name).iterdir() if f.suffix.lower() in exts])
    if imgs:
        bos_paths[name] = imgs
        step(f"BOS {name:<12}: {len(imgs)} crops")

# Sample wild crops (max 300 for speed)
wild_all  = sorted([f for f in WILD_DIR.iterdir() if f.suffix.lower() in exts])
wild_sample = random.sample(wild_all, min(300, len(wild_all)))
step(f"Wild internet : {len(wild_all)} total, sampling {len(wild_sample)}")

# ==============================================================================
# COMPUTE EMBEDDINGS
# ==============================================================================

title("Computing embeddings")

step("Zoo individuals...")
zoo_embs = embed_paths(model, zoo_plist, "Zoo")

step("BOS individuals...")
bos_embs_dict = {}
for name in sorted(bos_paths.keys()):
    embs = embed_paths(model, bos_paths[name], f"BOS {name}")
    bos_embs_dict[name] = embs

step("Wild internet crops...")
wild_embs = embed_paths(model, wild_sample, "Wild")

# ==============================================================================
# BUILD ZOO PROTOTYPES
# ==============================================================================

title("Building zoo prototypes")
proto = {}
for i, cls in enumerate(classes):
    mask = torch.tensor([l == i for l in zoo_labels])
    proto[cls] = F.normalize(zoo_embs[mask].mean(0), dim=0)
proto_matrix = torch.stack([proto[c] for c in classes])
step(f"Prototype matrix: {tuple(proto_matrix.shape)}")

# ==============================================================================
# FIGURE 1 : UMAP 2D EMBEDDING SPACE
# ==============================================================================

title("Figure 1: UMAP 2D embedding space")

try:
    from umap import UMAP
    has_umap = True
    step("UMAP available")
except ImportError:
    from sklearn.manifold import TSNE
    has_umap = False
    step("UMAP not found, using t-SNE (install umap-learn for better results)")

# Gather all embeddings for projection
all_embs_list  = []
all_plot_labels = []
all_plot_colors = []
all_plot_markers = []
all_plot_alpha  = []
all_plot_sizes  = []

# Color palette - 10 zoo + 30 BOS + wild
zoo_colors  = plt.cm.tab10(np.linspace(0, 1, 10))
bos_colors  = plt.cm.Set3(np.linspace(0, 1, len(bos_paths)))

# Zoo individuals
for i, cls in enumerate(classes):
    mask = [l == i for l in zoo_labels]
    e    = zoo_embs[[j for j,m in enumerate(mask) if m]]
    for ee in e:
        all_embs_list.append(ee.numpy())
        all_plot_labels.append(cls)
        all_plot_colors.append(zoo_colors[i])
        all_plot_markers.append("o")
        all_plot_alpha.append(0.8)
        all_plot_sizes.append(40)

# BOS individuals
for j, name in enumerate(sorted(bos_paths.keys())):
    embs = bos_embs_dict[name]
    for ee in embs:
        all_embs_list.append(ee.numpy())
        all_plot_labels.append(f"BOS_{name}")
        all_plot_colors.append(bos_colors[j])
        all_plot_markers.append("^")
        all_plot_alpha.append(0.5)
        all_plot_sizes.append(25)

# Wild crops
for ee in wild_embs:
    all_embs_list.append(ee.numpy())
    all_plot_labels.append("wild_internet")
    all_plot_colors.append((0.7, 0.7, 0.7, 0.3))
    all_plot_markers.append(".")
    all_plot_alpha.append(0.3)
    all_plot_sizes.append(10)

X = np.array(all_embs_list, dtype=np.float32)

step(f"Running dimensionality reduction on {len(X)} points...")
if has_umap:
    reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                   random_state=42, verbose=False)
else:
    # Use PCA first to speed up t-SNE
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50, random_state=42)
    X   = pca.fit_transform(X)
    reducer = TSNE(n_components=2, perplexity=30, random_state=42,
                   n_iter=500, verbose=0)

X2d = reducer.fit_transform(X)
step("Reduction done")

fig, axes = plt.subplots(1, 2, figsize=(22, 10))
fig.patch.set_facecolor("#0f0f0f")

# LEFT: all points (zoo + BOS + wild)
ax = axes[0]
ax.set_facecolor("#0f0f0f")

n_zoo = sum(len(zoo_paths[c]) for c in classes)
n_bos = sum(len(v) for v in bos_paths.values())

# Wild first (background)
ax.scatter(X2d[n_zoo+n_bos:, 0], X2d[n_zoo+n_bos:, 1],
           c="gray", alpha=0.2, s=8, marker=".", label="Wild internet", zorder=1)

# BOS
bos_x = X2d[n_zoo:n_zoo+n_bos, 0]
bos_y = X2d[n_zoo:n_zoo+n_bos, 1]
ax.scatter(bos_x, bos_y, c="cyan", alpha=0.3, s=20, marker="^",
           label="BOS (30 indivs, unknown)", zorder=2)

# Zoo - colored by individual
for i, cls in enumerate(classes):
    n_cls = len(zoo_paths[cls])
    start = sum(len(zoo_paths[c]) for c in classes[:i])
    x = X2d[start:start+n_cls, 0]
    y = X2d[start:start+n_cls, 1]
    ax.scatter(x, y, c=[zoo_colors[i]], s=60, marker="o",
               label=cls, zorder=3, edgecolors="white", linewidths=0.3)
    # Label centroid
    ax.annotate(cls, xy=(x.mean(), y.mean()), fontsize=7,
                color="white", fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=zoo_colors[i], alpha=0.7))

ax.set_title("Espace d'embedding 2D - TOUS les individus",
             color="white", fontsize=13, pad=12)
ax.tick_params(colors="gray")
ax.spines[:].set_color("#333")
legend = ax.legend(loc="upper right", fontsize=7, framealpha=0.3,
                   labelcolor="white", facecolor="#111")

# RIGHT: zoo only, colored + with prototype markers
ax2 = axes[1]
ax2.set_facecolor("#0f0f0f")

for i, cls in enumerate(classes):
    n_cls = len(zoo_paths[cls])
    start = sum(len(zoo_paths[c]) for c in classes[:i])
    x = X2d[start:start+n_cls, 0]
    y = X2d[start:start+n_cls, 1]
    ax2.scatter(x, y, c=[zoo_colors[i]], s=70, marker="o",
                label=cls, alpha=0.7, edgecolors="white", linewidths=0.3)
    # Prototype marker (star)
    cx, cy = x.mean(), y.mean()
    ax2.scatter([cx], [cy], c="white", s=200, marker="*", zorder=5)
    ax2.annotate(cls, xy=(cx, cy+0.3), fontsize=8,
                color="white", fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=zoo_colors[i], alpha=0.8))

# Add a few BOS points as triangles to show separation
ax2.scatter(bos_x[:50], bos_y[:50], c="cyan", alpha=0.4, s=30, marker="^",
            label="BOS samples", zorder=2)

ax2.set_title("Zoom sur les 10 individus connus\n(étoiles = prototypes, triangles = BOS inconnus)",
              color="white", fontsize=11, pad=12)
ax2.tick_params(colors="gray")
ax2.spines[:].set_color("#333")
ax2.legend(loc="upper right", fontsize=7, framealpha=0.3,
           labelcolor="white", facecolor="#111")

plt.tight_layout()
out = OUT_DIR / "1_umap_embedding_space.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
plt.close()
step(f"Saved: {out}")

# ==============================================================================
# FIGURE 2 : SIMILARITY DISTRIBUTIONS
# ==============================================================================

title("Figure 2: Similarity distributions")

# Compute all needed similarities
# Positive pairs: within same zoo individual
pos_sims = []
for i, cls in enumerate(classes):
    mask = [l == i for l in zoo_labels]
    e    = zoo_embs[[j for j,m in enumerate(mask) if m]]
    p    = proto[cls]
    sims = (e @ p).tolist()
    pos_sims.extend(sims)

# Negative pairs: zoo individual vs other prototypes
neg_sims = []
for i, cls in enumerate(classes):
    mask = [l == i for l in zoo_labels]
    e    = zoo_embs[[j for j,m in enumerate(mask) if m]]
    for j, cls2 in enumerate(classes):
        if i == j: continue
        sims = (e @ proto[cls2]).tolist()
        neg_sims.extend(sims)

# BOS vs zoo (should all be low)
bos_all_embs = torch.cat(list(bos_embs_dict.values()))
bos_vs_zoo   = (bos_all_embs @ proto_matrix.T).max(dim=1).values.tolist()

# Wild vs zoo
wild_vs_zoo  = (wild_embs @ proto_matrix.T).max(dim=1).values.tolist()

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.patch.set_facecolor("#0f0f0f")
fig.suptitle("Distributions de similarité cosinus", color="white",
             fontsize=15, fontweight="bold", y=1.01)

bins = np.linspace(-0.2, 1.0, 60)

def hist_ax(ax, data_dict, title_text, threshold=None):
    ax.set_facecolor("#111")
    colors = ["#00ff88", "#ff4444", "#4488ff", "#ffaa00", "#cc88ff"]
    for (label, data), color in zip(data_dict.items(), colors):
        ax.hist(data, bins=bins, alpha=0.6, label=label,
                color=color, edgecolor="none", density=True)
    if threshold is not None:
        ax.axvline(threshold, color="white", linewidth=2,
                   linestyle="--", label=f"threshold={threshold}")
    ax.set_title(title_text, color="white", fontsize=11, pad=8)
    ax.tick_params(colors="gray")
    ax.spines[:].set_color("#333")
    legend = ax.legend(fontsize=8, framealpha=0.3, labelcolor="white",
                       facecolor="#111")

hist_ax(axes[0,0], {
    "Même individu (positif)": pos_sims,
    "Individus différents (négatif)": neg_sims,
}, "Séparabilité des 10 individus connus\n(plus le gap est grand, mieux c'est)", threshold=THRESHOLD)

hist_ax(axes[0,1], {
    "BOS unknowns vs zoo": bos_vs_zoo,
    "Même individu (ref)": pos_sims[:500],
}, "BOS unknowns vs zoo prototypes\n(should be < threshold)", threshold=THRESHOLD)

hist_ax(axes[1,0], {
    "Wild internet vs zoo": wild_vs_zoo,
    "BOS unknowns vs zoo": bos_vs_zoo,
}, "Wild internet vs zoo prototypes\n(should be < threshold)", threshold=THRESHOLD)

# Per-individual separability bar chart
ax = axes[1,1]
ax.set_facecolor("#111")
sep_per_class = []
for i, cls in enumerate(classes):
    mask  = [l == i for l in zoo_labels]
    e     = zoo_embs[[j for j,m in enumerate(mask) if m]]
    pos   = float((e @ proto[cls]).mean())
    neg_m = float(
        torch.cat([e @ proto[c2] for j2, c2 in enumerate(classes) if j2 != i], dim=0).mean()
    ) if len(classes) > 1 else 0
    sep_per_class.append((cls, pos, neg_m, pos - neg_m))

classes_sorted  = [x[0] for x in sep_per_class]
sep_values      = [x[3] for x in sep_per_class]
pos_values      = [x[1] for x in sep_per_class]
neg_values      = [x[2] for x in sep_per_class]

x_pos = np.arange(len(classes))
bars  = ax.bar(x_pos, sep_values, color=zoo_colors[:len(classes)],
               alpha=0.8, edgecolor="white", linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(classes_sorted, rotation=35, ha="right", color="gray", fontsize=8)
ax.set_title("Gap de séparabilité par individu\n(pos_sim - neg_sim, plus haut = mieux)", 
             color="white", fontsize=11, pad=8)
ax.tick_params(colors="gray")
ax.spines[:].set_color("#333")
ax.axhline(0, color="white", linewidth=0.5, alpha=0.5)
for bar, val in zip(bars, sep_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.2f}", ha="center", va="bottom", color="white", fontsize=7)

plt.tight_layout()
out = OUT_DIR / "2_similarity_distributions.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
plt.close()
step(f"Saved: {out}")

# ==============================================================================
# FIGURE 3 : SAMPLE CROPS GRID (5 zoo individuals + 3 BOS + 5 wild)
# ==============================================================================

title("Figure 3: Sample crops with similarity scores")

def img_grid_with_score(paths, proto_mat, classes_list, threshold, n_samples=8):
    selected = random.sample(paths, min(n_samples, len(paths)))
    rows = []
    for p in selected:
        img_arr = load_img_rgb(p, size=112)
        try:
            t   = TRANSFORM(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                emb = F.normalize(model(t), dim=1).cpu()
            sims = (emb @ proto_mat.T)[0]
            max_sim, max_idx = sims.max().item(), sims.argmax().item()
            best_cls = classes_list[max_idx]
            label    = f"{best_cls}\n{max_sim:.3f}"
            color    = (0, 200, 80) if max_sim < threshold else (255, 60, 60)
        except:
            label, color = "error", (200, 200, 200)
        rows.append((img_arr, label, color))
    return rows

fig = plt.figure(figsize=(22, 14))
fig.patch.set_facecolor("#0f0f0f")

# Show 5 zoo individuals (2 samples each)
n_zoo_show  = 5
n_bos_show  = 3
n_wild_show = 5
samples_per = 6

rows_data = []

# Zoo
zoo_show = random.sample(classes, n_zoo_show)
for cls in zoo_show:
    paths = zoo_paths[cls]
    data  = img_grid_with_score(paths, proto_matrix, classes, threshold=-1, n_samples=samples_per)
    rows_data.append((f"ZOO: {cls}", data, "zoo"))

# BOS
bos_show_names = random.sample(list(bos_paths.keys()), n_bos_show)
for name in bos_show_names:
    data = img_grid_with_score(bos_paths[name], proto_matrix, classes, THRESHOLD, samples_per)
    rows_data.append((f"BOS: {name} (unknown)", data, "bos"))

# Wild
wild_data = img_grid_with_score(wild_sample, proto_matrix, classes, THRESHOLD, samples_per)
rows_data.append(("WILD internet (unknown)", wild_data, "wild"))

total_rows = len(rows_data)
gs = GridSpec(total_rows, samples_per + 1, figure=fig, hspace=0.15, wspace=0.05)

for row_i, (row_label, data, row_type) in enumerate(rows_data):
    color_map = {"zoo": "#00ff88", "bos": "#4488ff", "wild": "#ffaa00"}
    lax = fig.add_subplot(gs[row_i, 0])
    lax.set_facecolor("#0f0f0f")
    lax.axis("off")
    lax.text(0.5, 0.5, row_label, ha="center", va="center",
             color=color_map[row_type], fontsize=8, fontweight="bold",
             rotation=0, wrap=True, transform=lax.transAxes)

    for col_i, (img_arr, label, color) in enumerate(data):
        iax = fig.add_subplot(gs[row_i, col_i + 1])
        iax.imshow(img_arr)
        iax.axis("off")
        parts = label.split("\n")
        cls_label = parts[0] if parts else ""
        sim_label = parts[1] if len(parts) > 1 else ""
        rgb_norm  = tuple(c/255 for c in color)
        iax.set_title(sim_label, fontsize=7, color=rgb_norm, pad=2)
        iax.text(0.5, -0.02, cls_label, ha="center", va="top",
                 transform=iax.transAxes, fontsize=6.5, color="white")

plt.suptitle(
    "Sample crops — green/red = below/above threshold\n"
    "Zoo: sim avec le vrai individu | BOS/Wild: sim max avec n'importe quel zoo connu",
    color="white", fontsize=11, y=1.01
)

out = OUT_DIR / "3_sample_crops_grid.png"
plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="#0f0f0f")
plt.close()
step(f"Saved: {out}")

# ==============================================================================
# FIGURE 4 : CONFUSION HEATMAP (zoo vs zoo)
# ==============================================================================

title("Figure 4: Confusion heatmap between known individuals")

n_cls = len(classes)
conf_matrix = np.zeros((n_cls, n_cls))

for i, cls in enumerate(classes):
    mask  = [l == i for l in zoo_labels]
    e     = zoo_embs[[j for j,m in enumerate(mask) if m]]
    sims  = (e @ proto_matrix.T).numpy()   # (n, 10)
    conf_matrix[i] = sims.mean(axis=0)

fig, ax = plt.subplots(figsize=(11, 9))
fig.patch.set_facecolor("#0f0f0f")
ax.set_facecolor("#0f0f0f")

im = ax.imshow(conf_matrix, cmap="RdYlGn", vmin=-0.1, vmax=1.0, aspect="auto")
cbar = plt.colorbar(im, ax=ax)
cbar.ax.tick_params(colors="white")
cbar.set_label("Similarité cosinus", color="white")

ax.set_xticks(range(n_cls))
ax.set_yticks(range(n_cls))
ax.set_xticklabels(classes, rotation=40, ha="right", color="white", fontsize=9)
ax.set_yticklabels(classes, color="white", fontsize=9)
ax.set_xlabel("Prototype le plus proche", color="white", fontsize=10)
ax.set_ylabel("Individu réel (crops moyennés)", color="white", fontsize=10)

for i in range(n_cls):
    for j in range(n_cls):
        val   = conf_matrix[i, j]
        tcolor = "black" if val > 0.5 else "white"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                color=tcolor, fontsize=8,
                fontweight="bold" if i == j else "normal")

ax.set_title(
    "Matrice de confusion — similarité cosinus\n"
    "Diagonale = individu reconnu lui-même (doit être haute)\n"
    "Hors diagonale = confusion entre individus (doit être basse)",
    color="white", fontsize=11, pad=12
)

out = OUT_DIR / "4_confusion_heatmap.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
plt.close()
step(f"Saved: {out}")

# ==============================================================================
# FIGURE 5 : WILD CROPS DEEP DIVE
# ==============================================================================

title("Figure 5: Wild internet crops analysis")

wild_max_sims, wild_max_idx = (wild_embs @ proto_matrix.T).max(dim=1)
wild_max_sims = wild_max_sims.numpy()

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.patch.set_facecolor("#0f0f0f")

# LEFT: histogram of max sim for wild crops
ax = axes[0]
ax.set_facecolor("#111")
n_rejected = (wild_max_sims < THRESHOLD).sum()
n_total    = len(wild_max_sims)

bins = np.linspace(0, 1, 50)
hist_below = np.zeros(len(bins)-1)
hist_above = np.zeros(len(bins)-1)
for i in range(len(bins)-1):
    mask_b = (wild_max_sims >= bins[i]) & (wild_max_sims < bins[i+1]) & (wild_max_sims < THRESHOLD)
    mask_a = (wild_max_sims >= bins[i]) & (wild_max_sims < bins[i+1]) & (wild_max_sims >= THRESHOLD)
    hist_below[i] = mask_b.sum()
    hist_above[i] = mask_a.sum()

bw = bins[1] - bins[0]
ax.bar(bins[:-1], hist_below, width=bw, color="#00ff88", alpha=0.8,
       label=f"Rejeté ({n_rejected} crops)", align="edge")
ax.bar(bins[:-1], hist_above, width=bw, color="#ff4444", alpha=0.8,
       label=f"Faux positif ({n_total-n_rejected} crops)", align="edge")
ax.axvline(THRESHOLD, color="white", linewidth=2, linestyle="--",
           label=f"Seuil={THRESHOLD}")
ax.set_title("Wild crops: distribution de similarité\n avec individus zoo",
             color="white", fontsize=11)
ax.tick_params(colors="gray")
ax.spines[:].set_color("#333")
ax.set_xlabel("Max similarity avec zoo", color="gray")
ax.legend(fontsize=8, framealpha=0.3, labelcolor="white", facecolor="#111")

pct = 100 * n_rejected / n_total
ax.text(0.05, 0.95, f"{pct:.1f}% rejetés correctement",
        transform=ax.transAxes, color="white", fontsize=11,
        fontweight="bold", va="top")

# MIDDLE: which zoo class captures most wild crops above threshold
ax = axes[1]
ax.set_facecolor("#111")
false_pos_idx = wild_max_idx.numpy()[wild_max_sims >= THRESHOLD]
if len(false_pos_idx) > 0:
    from collections import Counter
    counts = Counter(int(i) for i in false_pos_idx)
    cls_names = [classes[i] for i in range(n_cls)]
    cls_counts = [counts.get(i, 0) for i in range(n_cls)]
    bars = ax.barh(cls_names, cls_counts,
                   color=zoo_colors[:n_cls], alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.set_title(f"Wild crops above threshold?\n({len(false_pos_idx)} false positifs)",
                 color="white", fontsize=11)
    for bar, val in zip(bars, cls_counts):
        if val > 0:
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    str(val), va="center", color="white", fontsize=9)
else:
    ax.text(0.5, 0.5, "0 false positifs!", ha="center", va="center",
            color="#00ff88", fontsize=16, fontweight="bold",
            transform=ax.transAxes)
    ax.set_title("Wild crops: false positifs par classe", color="white", fontsize=11)
ax.set_facecolor("#111")
ax.tick_params(colors="gray")
ax.spines[:].set_color("#333")

# RIGHT: sample wild crops above threshold (the problematic ones)
ax = axes[2]
ax.axis("off")
ax.set_facecolor("#111")

false_pos_paths = [wild_sample[i] for i, s in enumerate(wild_max_sims) if s >= THRESHOLD]
false_pos_sims  = wild_max_sims[wild_max_sims >= THRESHOLD]
false_pos_cls   = [classes[wild_max_idx[i].item()] for i,s in enumerate(wild_max_sims) if s >= THRESHOLD]

if false_pos_paths:
    n_show  = min(6, len(false_pos_paths))
    sorted_fp = sorted(zip(false_pos_sims, false_pos_paths, false_pos_cls), reverse=True)[:n_show]

    n_cols = 3
    n_rows = (n_show + n_cols - 1) // n_cols
    thumb_h = 1.0 / (n_rows + 0.5)
    thumb_w = 1.0 / n_cols

    for idx, (sim, p, pred) in enumerate(sorted_fp):
        row = idx // n_cols
        col = idx % n_cols
        img_arr = load_img_rgb(p, size=80)
        subax = fig.add_axes([
            axes[2].get_position().x0 + col * thumb_w * axes[2].get_position().width,
            axes[2].get_position().y0 + (n_rows - row - 1) * thumb_h * axes[2].get_position().height,
            thumb_w * axes[2].get_position().width * 0.9,
            thumb_h * axes[2].get_position().height * 0.85,
        ])
        subax.imshow(img_arr)
        subax.axis("off")
        subax.set_title(f"{pred}\n{sim:.3f}", fontsize=6, color="#ff4444", pad=2)

    ax.set_title(f"Wild crops above threshold (worst cases)\n= false positives to investiguer",
                 color="white", fontsize=11)
else:
    ax.text(0.5, 0.5, "Aucun false positif!\nTous les wild crops\nsont rejetés.",
            ha="center", va="center", color="#00ff88", fontsize=14,
            fontweight="bold", transform=ax.transAxes)
    ax.set_title("Wild crops: aucun false positif!", color="white", fontsize=11)

plt.tight_layout()
out = OUT_DIR / "5_wild_crops_analysis.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
plt.close()
step(f"Saved: {out}")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

title("SUMMARY - all outputs")

pos_arr = np.array(pos_sims)
neg_arr = np.array(neg_sims)
bos_arr = np.array(bos_vs_zoo)

print(f"""
  {'='*68}
  DIAGNOSTIC SUMMARY
  {'='*68}

  MODEL:
    File      : {MODEL_PATH}
    Val acc   : {ckpt.get('val_acc', 0)*100:.2f}%
    Epoch     : {ckpt.get('epoch', '?')}

  SEPARABILITY (zoo individuals):
    Mean positive similarity  : {pos_arr.mean():.4f}
    Mean negative similarity  : {neg_arr.mean():.4f}
    Gap (higher = better)     : {pos_arr.mean() - neg_arr.mean():.4f}
    Separability ratio        : {pos_arr.mean() / max(neg_arr.mean(), 1e-6):.2f}x

  OPEN-SET REJECTION (BOS - 1622 crops, 30 unseen individuals):
    Threshold                 : {THRESHOLD}
    Correctly rejected        : {(bos_arr < THRESHOLD).mean()*100:.1f}%
    Mean max similarity       : {bos_arr.mean():.4f}  (should be << threshold)

  WILD INTERNET ({len(wild_sample)} sampled crops):
    Correctly rejected        : {(wild_max_sims < THRESHOLD).sum()}/{len(wild_max_sims)} ({(wild_max_sims < THRESHOLD).mean()*100:.1f}%)
    False positives           : {(wild_max_sims >= THRESHOLD).sum()} crops above threshold

  OUTPUTS:
    {OUT_DIR}
    1_umap_embedding_space.png    <- 2D map of all individuals
    2_similarity_distributions.png <- separability histograms
    3_sample_crops_grid.png       <- example crops with scores
    4_confusion_heatmap.png       <- who confuses with who
    5_wild_crops_analysis.png     <- wild internet analysis

  {'='*68}
""")

print(f"  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")