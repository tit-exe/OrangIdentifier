"""
V6_5_test_rejection.py — OrangIdentifier V6
=============================================
Tests the ability of the V6 model to REJECT unknowns:
  - BOS (30 sanctuary individuals, real orangutans, but not trained on)
  - Wild (5429 crops sauvages)

A good model must:
  - Accepter  les individus zoo  (score ≥ seuil)
  - Rejeter   les BOS et wild    (score < seuil)

RUN :
    python v6_megadesc_arcface_15ind/04_test_open_set.py
"""

import os, json, random, warnings
os.environ["HF_HOME"]    = r"D:\HuggingFaceCache"
os.environ["TORCH_HOME"] = r"D:\TorchCache"

import numpy as np
import torch
import torch.nn.functional as F
import timm
from pathlib import Path
from PIL import Image, ImageFile
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]  # repository root (portable)

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
BACKBONE_PT = (REPO / "output" / "v6" / "models" / "v6_backbone_only.pt")
GALLERY_JS  = (REPO / "output" / "v6" / "models" / "v6_gallery.json")
BOS_DIR     = (REPO / "data" / "crops" / "bos")
WILD_DIR    = (REPO / "data" / "crops" / "wild")
RESULTS     = (REPO / "output" / "v6" / "results")
RESULTS.mkdir(exist_ok=True)

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
EXTS     = {".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"}
MEAN = STD = [0.5, 0.5, 0.5]
WILD_SAMPLE = 1000   # number of wild to test

random.seed(42); np.random.seed(42)

# ── Transforms ───────────────────────────────────────────────────────────────
val_tf = T.Compose([T.Resize(IMG_SIZE), T.CenterCrop(IMG_SIZE),
                    T.ToTensor(), T.Normalize(MEAN, STD)])

class DS(Dataset):
    def __init__(self, paths):
        self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        try:    img = Image.open(self.paths[idx]).convert("RGB")
        except: img = Image.new("RGB",(IMG_SIZE,IMG_SIZE),(128,128,128))
        return val_tf(img)

# ── Backbone ─────────────────────────────────────────────────────────────────
print(f"Device : {DEVICE}")
print("Loading V6 backbone...")
bb = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224", pretrained=False, num_classes=0)
ck = torch.load(str(BACKBONE_PT), map_location="cpu", weights_only=False)
state = ck.get("backbone_state") or ck
bb.load_state_dict(state, strict=False)
bb = bb.to(DEVICE).eval()
print("  OK")

# ── Galerie ───────────────────────────────────────────────────────────────────
print("Loading V6 gallery...")
gal      = json.loads(GALLERY_JS.read_text(encoding="utf-8"))
THRESHOLD = gal["unknown_threshold"]
zoo_names = list(gal["individuals"].keys())

# Matrice exemplaires : (N_zoo, K_max, 768)
ex_list = []
for name in zoo_names:
    ex = np.array(gal["individuals"][name]["exemplars"], dtype=np.float32)
    ex_list.append(ex)

print(f"  {len(zoo_names)} individuals, threshold={THRESHOLD:.4f}")

@torch.no_grad()
def embed(paths):
    dl  = DataLoader(DS(paths), 64, num_workers=0)
    out = []
    for imgs in dl:
        out.append(F.normalize(bb(imgs.to(DEVICE).float()), dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)

def score_against_gallery(embs):
    """For each embedding, return (best_score, best_class_idx)."""
    n = len(embs)
    scores = np.full((n, len(zoo_names)), -1.0, dtype=np.float32)
    for ci, ex in enumerate(ex_list):
        scores[:, ci] = (embs @ ex.T).max(1)
    best_score = scores.max(1)
    best_class = scores.argmax(1)
    return best_score, best_class

# ══════════════════════════════════════════════════════════════════════════════
# TEST BOS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  BOS TEST (sanctuary individuals, must be REJECTED)")
print("═"*65)

bos_results = {}
bos_all_scores = []
bos_confusion = {name: 0 for name in zoo_names}

for ind_dir in sorted(BOS_DIR.iterdir()):
    if not ind_dir.is_dir(): continue
    paths = sorted([f for f in ind_dir.iterdir() if f.suffix in EXTS])
    if not paths: continue
    name = ind_dir.name

    embs = embed(paths)
    best_score, best_class = score_against_gallery(embs)

    accepted   = best_score >= THRESHOLD
    fp_rate    = accepted.mean()
    fp_scores  = best_score[accepted]
    fp_targets = [zoo_names[i] for i in best_class[accepted]]

    for t in fp_targets:
        bos_confusion[t] += 1
    bos_all_scores.extend(best_score.tolist())

    bos_results[name] = {
        "n": len(paths),
        "fp_rate": float(fp_rate),
        "fp_count": int(accepted.sum()),
        "mean_score": float(best_score.mean()),
        "max_score":  float(best_score.max()),
        "fp_targets": fp_targets,
    }
    flag = " !" if fp_rate > 0.05 else ""
    print(f"  {name:<14}: {len(paths):3d} crops | rejected={100*(1-fp_rate):5.1f}%"
          f" | score µ={best_score.mean():.3f} max={best_score.max():.3f}{flag}")

bos_scores_arr = np.array(bos_all_scores)
bos_total_n    = len(bos_scores_arr)
bos_total_fp   = (bos_scores_arr >= THRESHOLD).sum()
print(f"\n  TOTAL BOS : {bos_total_n} crops"
      f" | FP global = {bos_total_fp} ({100*bos_total_fp/bos_total_n:.2f}%)"
      f" | score µ={bos_scores_arr.mean():.4f}")

# Confusion BOS → zoo
top_conf = sorted(bos_confusion.items(), key=lambda x: -x[1])[:5]
print(f"\n  Individus zoo les plus confondus avec BOS :")
for zn, cnt in top_conf:
    if cnt > 0: print(f"    {zn:<14}: {cnt} FP")

# ══════════════════════════════════════════════════════════════════════════════
# TEST WILD
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print(f"  WILD TEST ({WILD_SAMPLE} wild crops, must be REJECTED)")
print("═"*65)

wild_files  = sorted([f for f in WILD_DIR.iterdir() if f.suffix in EXTS])
wild_sample = random.sample(wild_files, min(WILD_SAMPLE, len(wild_files)))
wild_embs   = embed(wild_sample)
wild_score, wild_class = score_against_gallery(wild_embs)

wild_accepted = wild_score >= THRESHOLD
wild_fp_rate  = wild_accepted.mean()
wild_confusion = {name: 0 for name in zoo_names}
for i in wild_class[wild_accepted]:
    wild_confusion[zoo_names[i]] += 1

print(f"  {len(wild_sample)} crops tested")
print(f"  FP rate   : {100*wild_fp_rate:.2f}%  ({wild_accepted.sum()} accepted)")
print(f"  Score µ   : {wild_score.mean():.4f}  ± {wild_score.std():.4f}")
print(f"  Score max : {wild_score.max():.4f}")
print(f"  Score p95 : {np.percentile(wild_score, 95):.4f}")

top_wild_conf = sorted(wild_confusion.items(), key=lambda x: -x[1])[:5]
print(f"\n  Individus zoo les plus confondus avec wild :")
for zn, cnt in top_wild_conf:
    if cnt > 0: print(f"    {zn:<14}: {cnt} FP")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Score distributions
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("V6 — Score distributions: BOS & Wild vs zoo gallery\n"
             "(scores must stay below the rejection threshold)", fontsize=13)

ax = axes[0]
ax.hist(bos_scores_arr, bins=40, color="#dc2626", alpha=0.7, label=f"BOS ({bos_total_n} crops)")
ax.axvline(THRESHOLD, color="black", lw=2, linestyle="--", label=f"Threshold {THRESHOLD:.3f}")
ax.set_title("BOS individuals — score distribution")
ax.set_xlabel("Score (cosine max-over-exemplars)")
ax.set_ylabel("Number of crops"); ax.set_xlim(0,1); ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.hist(wild_score, bins=40, color="#f97316", alpha=0.7, label=f"Wild ({len(wild_sample)} crops)")
ax.axvline(THRESHOLD, color="black", lw=2, linestyle="--", label=f"Threshold {THRESHOLD:.3f}")
ax.set_title("Wild crops — score distribution")
ax.set_xlabel("Score (cosine max-over-exemplars)")
ax.set_xlim(0,1); ax.legend(); ax.grid(alpha=0.3)

ax = axes[2]
ax.hist(bos_scores_arr, bins=40, color="#dc2626", alpha=0.6, label="BOS", density=True)
ax.hist(wild_score,     bins=40, color="#f97316", alpha=0.6, label="Wild", density=True)
ax.axvline(THRESHOLD, color="black", lw=2, linestyle="--", label=f"Threshold {THRESHOLD:.3f}")
ax.set_title("BOS + Wild overlaid (density)")
ax.set_xlabel("Score"); ax.set_xlim(0,1); ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
out = RESULTS / "04_rejection_test.png"
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
print(f"\n  Figure 1 -> {out.name}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — FP rate per BOS individual
# ══════════════════════════════════════════════════════════════════════════════
names_bos = list(bos_results.keys())
fp_rates  = [bos_results[n]["fp_rate"]*100 for n in names_bos]
means_bos = [bos_results[n]["mean_score"]   for n in names_bos]
maxes_bos = [bos_results[n]["max_score"]    for n in names_bos]

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(names_bos))
ax.bar(x, fp_rates, color=["#dc2626" if r > 5 else "#fca5a5" for r in fp_rates])
ax.axhline(5, color="black", lw=1.5, linestyle="--", label="5% limit")
ax.set_xticks(x); ax.set_xticklabels(names_bos, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("False positive rate (% accepted as known individual)")
ax.set_title("V6 — False positive rate per BOS individual\n(zero = perfect rejection)")
ax.set_ylim(0, max(max(fp_rates)+5, 10))
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out2 = RESULTS / "04b_bos_fp_per_individual.png"
plt.savefig(out2, dpi=150, bbox_inches="tight"); plt.close()
print(f"  Figure 2 -> {out2.name}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Pie charts summary
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle("V6 — Unknown rejection summary (pie charts)\n"
             "Green = correctly rejected  |  Red = wrongly accepted (false positive)",
             fontsize=13)

PIE_COLORS_OK = ["#16a34a", "#dc2626"]   # green / red

# ── Pie 1 : BOS ──────────────────────────────────────────────────────────────
bos_rejected = bos_total_n - int(bos_total_fp)
wedge_bos    = [bos_rejected, int(bos_total_fp)]
labels_bos   = [f"Correctly rejected\n{bos_rejected} crops ({100*bos_rejected/bos_total_n:.1f}%)",
                f"False positive\n{int(bos_total_fp)} crops ({100*bos_total_fp/bos_total_n:.2f}%)"]
explode_bos  = [0, 0.12] if bos_total_fp > 0 else [0, 0]
axes[0].pie(wedge_bos, labels=labels_bos, colors=PIE_COLORS_OK,
            autopct=lambda p: f"{p:.2f}%" if p < 99 else f"{p:.1f}%",
            startangle=90, explode=explode_bos,
            textprops={"fontsize": 10}, pctdistance=0.75)
axes[0].set_title(f"BOS individuals\n({bos_total_n} crops — 30 sanctuary orangutans)",
                  fontsize=11, fontweight="bold", pad=15)

# ── Pie 2 : Wild ─────────────────────────────────────────────────────────────
wild_n        = len(wild_sample)
wild_rej      = wild_n - int(wild_accepted.sum())
wedge_wild    = [wild_rej, int(wild_accepted.sum())]
labels_wild   = [f"Correctly rejected\n{wild_rej} crops ({100*wild_rej/wild_n:.1f}%)",
                 f"False positive\n{int(wild_accepted.sum())} crops ({100*wild_fp_rate:.2f}%)"]
explode_wild  = [0, 0.12] if wild_accepted.sum() > 0 else [0, 0]
axes[1].pie(wedge_wild, labels=labels_wild, colors=PIE_COLORS_OK,
            autopct=lambda p: f"{p:.2f}%" if p < 99 else f"{p:.1f}%",
            startangle=90, explode=explode_wild,
            textprops={"fontsize": 10}, pctdistance=0.75)
axes[1].set_title(f"Wild crops\n({wild_n} sampled from {len(wild_files)} total)",
                  fontsize=11, fontweight="bold", pad=15)

# ── Pie 3 : Combined unknowns (BOS + Wild) ───────────────────────────────────
total_unk    = bos_total_n + wild_n
total_unk_fp = int(bos_total_fp) + int(wild_accepted.sum())
total_unk_ok = total_unk - total_unk_fp
wedge_comb   = [total_unk_ok, total_unk_fp]
labels_comb  = [f"Correctly rejected\n{total_unk_ok} ({100*total_unk_ok/total_unk:.1f}%)",
                f"False positive\n{total_unk_fp} ({100*total_unk_fp/total_unk:.2f}%)"]
explode_comb = [0, 0.12] if total_unk_fp > 0 else [0, 0]
axes[2].pie(wedge_comb, labels=labels_comb, colors=PIE_COLORS_OK,
            autopct=lambda p: f"{p:.2f}%" if p < 99 else f"{p:.1f}%",
            startangle=90, explode=explode_comb,
            textprops={"fontsize": 10}, pctdistance=0.75)
axes[2].set_title(f"All unknowns combined\n(BOS + Wild — {total_unk} total crops)",
                  fontsize=11, fontweight="bold", pad=15)

plt.tight_layout()
out3 = RESULTS / "04c_rejection_pie_charts.png"
plt.savefig(out3, dpi=150, bbox_inches="tight"); plt.close()
print(f"  Figure 3 -> {out3.name}")

# ── Pie 4 : Zoo benchmark recall ─────────────────────────────────────────────
from json import loads as _jloads
_rep = _jloads(((REPO / "output" / "v6" / "results" / "v6_report.json")).read_text())
n_val     = _rep["benchmark"]["n_val_queries"]
n_correct = round(_rep["benchmark"]["accuracy"] * n_val)
n_wrong   = n_val - n_correct

fig, axes4 = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("V6 — Zoo identification benchmark vs Unknown rejection\n"
             "(left: known individuals correctly identified, right: unknowns correctly rejected)",
             fontsize=12)

axes4[0].pie([n_correct, n_wrong],
             labels=[f"Correctly identified\n{n_correct}/{n_val} ({100*n_correct/n_val:.1f}%)",
                     f"Misidentified\n{n_wrong}/{n_val} ({100*n_wrong/n_val:.2f}%)"],
             colors=["#16a34a","#dc2626"], startangle=90,
             explode=[0, 0.12],
             autopct=lambda p: f"{p:.2f}%" if p < 99 else f"{p:.1f}%",
             textprops={"fontsize": 10}, pctdistance=0.75)
axes4[0].set_title(f"Known zoo individuals\n({n_val} validation crops)", fontsize=11,
                   fontweight="bold", pad=15)

axes4[1].pie([total_unk_ok, total_unk_fp],
             labels=[f"Correctly rejected\n{total_unk_ok}/{total_unk} ({100*total_unk_ok/total_unk:.1f}%)",
                     f"False positive\n{total_unk_fp}/{total_unk} ({100*total_unk_fp/total_unk:.2f}%)"],
             colors=["#16a34a","#dc2626"], startangle=90,
             explode=[0, 0.12],
             autopct=lambda p: f"{p:.2f}%" if p < 99 else f"{p:.1f}%",
             textprops={"fontsize": 10}, pctdistance=0.75)
axes4[1].set_title(f"Unknown orangutans (BOS + Wild)\n({total_unk} total crops)",
                   fontsize=11, fontweight="bold", pad=15)

plt.tight_layout()
out4 = RESULTS / "04d_id_vs_rejection_summary.png"
plt.savefig(out4, dpi=150, bbox_inches="tight"); plt.close()
print(f"  Figure 4 -> {out4.name}")

# ══════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ FINAL
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*65}")
print("  RÉSUMÉ REJET — V6")
print(f"{'═'*65}")
print(f"  Gallery threshold : {THRESHOLD:.4f}")
print(f"  BOS FP global   : {100*bos_total_fp/bos_total_n:.2f}%  "
      f"({bos_total_fp}/{bos_total_n} crops accepted)")
print(f"  Wild FP global  : {100*wild_fp_rate:.2f}%  "
      f"({int(wild_accepted.sum())}/{len(wild_sample)} crops accepted)")
print(f"  Score BOS µ/max : {bos_scores_arr.mean():.4f} / {bos_scores_arr.max():.4f}")
print(f"  Score Wild µ/max: {wild_score.mean():.4f} / {wild_score.max():.4f}")

worst_bos = sorted(bos_results.items(), key=lambda x: -x[1]["fp_rate"])[:3]
if worst_bos[0][1]["fp_rate"] > 0:
    print(f"\n  BOS individuals hardest to reject :")
    for n, r in worst_bos:
        if r["fp_rate"] > 0:
            print(f"    {n:<14}: {r['fp_rate']*100:.1f}% FP → confondu avec {r['fp_targets'][:3]}")
print(f"{'═'*65}\n")