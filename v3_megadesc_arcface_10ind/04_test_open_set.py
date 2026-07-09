# V3_test_bos_baseline.py
# Orang-outan V2 pipeline
# 
#
# READ-ONLY diagnostic: tests current model against 1622 BOS crops.
# The 30 BOS individuals were NEVER seen during training.
# Goal: measure if the current model correctly rejects them as "unknown".
#
# RUN:
#   conda activate orangs
#   python v3_megadesc_arcface_10ind/06_test_open_set.py

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
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime




import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import timm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==============================================================================
# PATHS (all read-only)
# ==============================================================================

MODEL_PATH = V3_PT
ZOO_DIR = CROPS_KNOWN_DIR
BOS_DIR = CROPS_KNOWN_DIR
BOS_JSON      = BOS_DIR / "boxes_new_orangs.json"

IMG_SIZE      = 224
BATCH_SIZE    = 32
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEGA_MEAN     = [0.5, 0.5, 0.5]
MEGA_STD      = [0.5, 0.5, 0.5]

# ==============================================================================
# TERMINAL HELPERS
# ==============================================================================

def title(text):
    bar = "=" * 78
    print(f"\n{bar}\n  {text}\n{bar}")

def section(text):
    print(f"\n  {text}\n  {'-' * 76}")

def bar_visual(value, max_value, width=40, char="#"):
    if max_value <= 0: return " " * width
    fill = int(width * value / max_value)
    fill = max(0, min(fill, width))
    return char * fill + "." * (width - fill)

# ==============================================================================
# LOAD MODEL
# ==============================================================================

def load_model():
    section("Loading current model")

    if not MODEL_PATH.exists():
        print(f"  [ERROR] Model not found: {MODEL_PATH}")
        sys.exit(1)

    ckpt        = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    classes     = ckpt["classes"]
    emb_dim     = ckpt.get("emb_dim", 768)
    epoch       = ckpt.get("epoch", "?")
    val_acc     = ckpt.get("val_acc", 0.0)
    arc_scale   = ckpt.get("arc_scale", "?")
    arc_margin  = ckpt.get("arc_margin", "?")
    save_time   = ckpt.get("save_time", "?")

    print(f"  Model file       : {MODEL_PATH}")
    print(f"  Model size       : {MODEL_PATH.stat().st_size / 1e6:.1f} MB")
    print(f"  Saved at         : {save_time}")
    print(f"  Best epoch       : {epoch}")
    print(f"  Val accuracy     : {val_acc * 100:.2f}%")
    print(f"  Embedding dim    : {emb_dim}")
    print(f"  ArcFace scale    : {arc_scale}")
    print(f"  ArcFace margin   : {arc_margin}")
    print(f"  Known classes ({len(classes)}):")
    for i, name in enumerate(classes):
        print(f"    [{i:2d}] {name}")

    model = timm.create_model(
        "hf-hub:BVRA/MegaDescriptor-T-224",
        pretrained=False,
        num_classes=0,
    )
    model.load_state_dict(ckpt["backbone_state"])
    model = model.eval().to(DEVICE)
    print(f"\n  Backbone loaded on {DEVICE}")

    return model, classes, emb_dim, ckpt

# ==============================================================================
# IMAGE LOADING
# ==============================================================================

TRANSFORM = T.Compose([
    T.Resize(IMG_SIZE),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(MEGA_MEAN, MEGA_STD),
])

@torch.no_grad()
def embed_images(model, paths):
    embs = []
    for i in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[i:i+BATCH_SIZE]
        batch_imgs  = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_imgs.append(TRANSFORM(img))
            except Exception as e:
                print(f"  [WARN] Failed to load {p.name}: {e}")
                batch_imgs.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))
        batch = torch.stack(batch_imgs).to(DEVICE, non_blocking=True)
        emb   = F.normalize(model(batch), dim=1)
        embs.append(emb.cpu())
    return torch.cat(embs, dim=0)

# ==============================================================================
# BUILD ZOO PROTOTYPES (the 10 known individuals)
# ==============================================================================

def build_zoo_prototypes(model, classes):
    section("Building prototypes for 10 known individuals (zoo crops)")

    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    protos     = {}
    proto_counts = {}

    for cls in classes:
        cls_dir = ZOO_DIR / cls
        if not cls_dir.exists():
            print(f"  [WARN] Class folder missing: {cls_dir}")
            continue
        imgs = sorted([f for f in cls_dir.iterdir() if f.suffix in exts])
        if not imgs:
            print(f"  [WARN] No images for {cls}")
            continue
        embs  = embed_images(model, imgs)
        proto = F.normalize(embs.mean(dim=0), dim=0)
        protos[cls]       = proto
        proto_counts[cls] = len(imgs)
        print(f"  {cls:<12} : {len(imgs):4d} crops -> prototype built")

    return protos, proto_counts

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    title("BOS BASELINE TEST - READ-ONLY DIAGNOSTIC")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device    : {DEVICE}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU       : {props.name}  ({props.total_memory / 1e9:.1f} GB)")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model, classes, emb_dim, ckpt = load_model()

    # Get threshold from checkpoint or set default
    threshold = ckpt.get("threshold", None)
    if threshold is None:
        threshold = 0.22  # from previous calibration mentioned in transcripts
        print(f"\n  No threshold in checkpoint, using default: {threshold}")
    else:
        print(f"\n  Threshold from checkpoint: {threshold}")

    # ------------------------------------------------------------------
    # Build prototypes for 10 known individuals
    # ------------------------------------------------------------------
    protos, proto_counts = build_zoo_prototypes(model, classes)
    if not protos:
        print("  [ERROR] No prototypes built. Exiting.")
        sys.exit(1)

    proto_names  = list(protos.keys())
    proto_matrix = torch.stack([protos[n] for n in proto_names])

    print(f"\n  Prototype matrix : {tuple(proto_matrix.shape)}")
    print(f"  Total zoo crops  : {sum(proto_counts.values())}")

    # ------------------------------------------------------------------
    # Embed all BOS crops
    # ------------------------------------------------------------------
    section("Loading BOS individuals (1622 crops, 30 individuals - NEVER seen)")

    bos_individuals = sorted([d.name for d in BOS_DIR.iterdir() if d.is_dir()])
    if not bos_individuals:
        print(f"  [ERROR] No BOS individuals found in {BOS_DIR}")
        sys.exit(1)

    bos_data = {}  # name -> dict(paths, embeddings, ...)

    for bos_name in bos_individuals:
        bos_subdir = BOS_DIR / bos_name
        imgs = sorted([
            f for f in bos_subdir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])
        if not imgs:
            continue
        print(f"  {bos_name:<14} : embedding {len(imgs):3d} crops...", end="", flush=True)
        embs = embed_images(model, imgs)
        bos_data[bos_name] = {
            "paths":      imgs,
            "embeddings": embs,
            "count":      len(imgs),
        }
        print(f" done")

    total_bos = sum(d["count"] for d in bos_data.values())
    print(f"\n  Total BOS embeddings computed: {total_bos}")

    # ------------------------------------------------------------------
    # FOR EACH BOS INDIVIDUAL: compute statistics
    # ------------------------------------------------------------------
    title("ANALYSIS: similarity of each BOS individual to the 10 known")

    # Aggregate stats
    per_individual = {}     # bos_name -> stats dict
    all_max_sims   = []     # per-crop max similarity (across all BOS crops)
    all_predictions = []    # which zoo class each BOS crop is closest to
    all_correctly_rejected = 0  # crops where max_sim < threshold
    all_total = 0

    for bos_name in sorted(bos_data.keys()):
        embs = bos_data[bos_name]["embeddings"]

        # Cosine similarities: (n_crops, 10_classes)
        sims = embs @ proto_matrix.T  # already L2 normalized

        # For each crop: max similarity + which class
        max_sims, max_idx = sims.max(dim=1)
        max_sims_np = max_sims.numpy()
        max_idx_np  = max_idx.numpy()

        # Mean similarity to each known class (over all crops)
        mean_per_class = sims.mean(dim=0).numpy()

        # Std of similarities (variance of identity in this individual)
        std_per_class  = sims.std(dim=0).numpy()

        # Overall stats for this BOS individual
        n_crops          = len(embs)
        max_sim_mean     = float(max_sims_np.mean())
        max_sim_max      = float(max_sims_np.max())
        max_sim_min      = float(max_sims_np.min())
        max_sim_median   = float(np.median(max_sims_np))
        max_sim_std      = float(max_sims_np.std())

        # Rejection rate (good = correctly identified as unknown)
        rejected = int((max_sims_np < threshold).sum())
        rejection_rate = rejected / n_crops

        # Most confused class (highest mean similarity)
        most_confused_idx = int(np.argmax(mean_per_class))
        most_confused    = proto_names[most_confused_idx]
        confusion_value  = float(mean_per_class[most_confused_idx])

        # Predicted class for each crop (which zoo individual it would be tagged as)
        predicted_counts = Counter()
        for idx, sim in zip(max_idx_np, max_sims_np):
            if sim >= threshold:
                predicted_counts[proto_names[int(idx)]] += 1
            else:
                predicted_counts["<unknown>"] += 1

        per_individual[bos_name] = {
            "n_crops":          n_crops,
            "max_sim_mean":     max_sim_mean,
            "max_sim_max":      max_sim_max,
            "max_sim_min":      max_sim_min,
            "max_sim_median":   max_sim_median,
            "max_sim_std":      max_sim_std,
            "rejected":         rejected,
            "rejection_rate":   rejection_rate,
            "most_confused":    most_confused,
            "confusion_value":  confusion_value,
            "mean_per_class":   {proto_names[i]: float(mean_per_class[i]) for i in range(len(proto_names))},
            "predicted_counts": dict(predicted_counts),
        }

        all_max_sims.extend(max_sims_np.tolist())
        all_predictions.extend([proto_names[int(i)] for i in max_idx_np])
        all_correctly_rejected += rejected
        all_total              += n_crops

    # ------------------------------------------------------------------
    # PER-INDIVIDUAL TABLE
    # ------------------------------------------------------------------
    section("Per-BOS-individual breakdown")
    print(f"  {'Individual':<14} {'N':>4}  {'max_mean':>9}  {'rejected':>9}  {'rej_rate':>9}  {'confused_with':>14}  {'conf':>6}")
    print(f"  {'-'*14}  {'-'*3}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*14}  {'-'*6}")

    sorted_by_rate = sorted(per_individual.items(),
                             key=lambda kv: kv[1]["rejection_rate"])

    for name, st in sorted_by_rate:
        rate_pct = st["rejection_rate"] * 100
        if rate_pct >= 90:    flag = "OK "
        elif rate_pct >= 70:  flag = "?  "
        else:                 flag = "BAD"
        print(
            f"  {name:<14} {st['n_crops']:>4}  "
            f"{st['max_sim_mean']:>9.4f}  "
            f"{st['rejected']:>4}/{st['n_crops']:<4}  "
            f"{rate_pct:>8.1f}%  "
            f"{st['most_confused']:>14}  "
            f"{st['confusion_value']:>6.3f}  {flag}"
        )

    # ------------------------------------------------------------------
    # GLOBAL STATISTICS
    # ------------------------------------------------------------------
    title("GLOBAL STATISTICS - all 1622 BOS crops vs 10 known individuals")

    all_max_sims_np = np.array(all_max_sims)
    overall_rejection_rate = all_correctly_rejected / all_total

    section("Distribution of max similarity to nearest known prototype")
    pct_below_threshold = (all_max_sims_np < threshold).mean() * 100
    print(f"  Threshold used        : {threshold}")
    print(f"  Total BOS crops       : {all_total}")
    print(f"  Correctly rejected    : {all_correctly_rejected:>5}  ({overall_rejection_rate*100:.2f}%)")
    print(f"  Falsely identified    : {all_total - all_correctly_rejected:>5}  ({(1-overall_rejection_rate)*100:.2f}%)")
    print()
    print(f"  max_sim mean          : {all_max_sims_np.mean():.4f}")
    print(f"  max_sim median        : {np.median(all_max_sims_np):.4f}")
    print(f"  max_sim std           : {all_max_sims_np.std():.4f}")
    print(f"  max_sim min           : {all_max_sims_np.min():.4f}")
    print(f"  max_sim max           : {all_max_sims_np.max():.4f}")
    print(f"  5th percentile        : {np.percentile(all_max_sims_np, 5):.4f}")
    print(f"  25th percentile       : {np.percentile(all_max_sims_np, 25):.4f}")
    print(f"  50th percentile       : {np.percentile(all_max_sims_np, 50):.4f}")
    print(f"  75th percentile       : {np.percentile(all_max_sims_np, 75):.4f}")
    print(f"  95th percentile       : {np.percentile(all_max_sims_np, 95):.4f}")

    section("Histogram of max similarity (BOS vs known)")
    bins = np.linspace(0, 1, 21)
    hist, _ = np.histogram(all_max_sims_np, bins=bins)
    max_h = max(hist) if hist.max() > 0 else 1
    for i in range(len(hist)):
        lo, hi = bins[i], bins[i+1]
        marker = ""
        if lo <= threshold <= hi:
            marker = " <-- threshold"
        bar = bar_visual(hist[i], max_h, width=50)
        print(f"  [{lo:.2f}-{hi:.2f}]  {hist[i]:5d}  {bar}{marker}")

    section("How often is each KNOWN class falsely matched?")
    falsely_matched = Counter()
    for name, st in per_individual.items():
        for cls, cnt in st["predicted_counts"].items():
            if cls != "<unknown>":
                falsely_matched[cls] += cnt

    total_falsely_matched = sum(falsely_matched.values())
    if total_falsely_matched > 0:
        for cls, cnt in falsely_matched.most_common():
            pct  = 100 * cnt / total_falsely_matched
            bar = bar_visual(cnt, max(falsely_matched.values()), width=40)
            print(f"  {cls:<14} {cnt:5d} ({pct:5.1f}%)  {bar}")
    else:
        print("  No false matches - all BOS individuals correctly rejected!")

    # ------------------------------------------------------------------
    # PER-INDIVIDUAL DETAILED CONFUSION TABLE
    # ------------------------------------------------------------------
    section("Most confused BOS individuals (lowest rejection rates)")
    worst_5 = sorted_by_rate[:5]
    for bos_name, st in worst_5:
        print(f"\n  >>> {bos_name}  ({st['n_crops']} crops, rejection rate {st['rejection_rate']*100:.1f}%)")
        sorted_means = sorted(
            st["mean_per_class"].items(), key=lambda kv: kv[1], reverse=True
        )
        for cls, mv in sorted_means:
            bar = bar_visual(mv, 0.5, width=30)
            marker = " <-- main confusion" if cls == st["most_confused"] else ""
            print(f"      {cls:<12}  mean_sim={mv:>7.4f}  {bar}{marker}")

    section("Best behaved BOS individuals (highest rejection rates)")
    best_5 = sorted_by_rate[-5:]
    for bos_name, st in best_5:
        print(f"  {bos_name:<14} {st['n_crops']:3d} crops "
              f"rejection={st['rejection_rate']*100:5.1f}% "
              f"max_mean={st['max_sim_mean']:.4f}")

    # ------------------------------------------------------------------
    # SEPARABILITY ANALYSIS
    # ------------------------------------------------------------------
    section("Within-individual identity coherence (BOS)")
    print("  If crops of the same BOS individual cluster tightly, the model")
    print("  generalizes well to unseen individuals.")
    print("  Computed as: mean cosine similarity within each BOS individual.")
    print()
    print(f"  {'Individual':<14}  {'within_sim':>11}  {'std':>7}  bar")
    print(f"  {'-'*14}  {'-'*11}  {'-'*7}  {'-'*30}")

    within_sims = []
    for bos_name in sorted(bos_data.keys()):
        embs = bos_data[bos_name]["embeddings"]
        if len(embs) < 2: continue
        proto_bos = F.normalize(embs.mean(0), dim=0)
        sims = embs @ proto_bos
        within_sim = float(sims.mean())
        within_std = float(sims.std())
        within_sims.append(within_sim)
        bar = bar_visual(within_sim, 1.0, width=30)
        print(f"  {bos_name:<14}  {within_sim:>11.4f}  {within_std:>7.4f}  {bar}")

    print()
    print(f"  Mean within-individual coherence : {np.mean(within_sims):.4f}")
    print(f"  Std across individuals           : {np.std(within_sims):.4f}")
    print(f"  Min                              : {np.min(within_sims):.4f}")
    print(f"  Max                              : {np.max(within_sims):.4f}")

    # ------------------------------------------------------------------
    # SEPARABILITY: BOS vs ZOO (key metric)
    # ------------------------------------------------------------------
    section("Cross-domain separability: BOS individuals vs zoo individuals")
    print("  This is the KEY test. If BOS prototypes are FAR from zoo prototypes,")
    print("  the model has learned visual identity features that distinguish")
    print("  individuals across domains. If they're CLOSE, the model has")
    print("  learned spurious cues.")
    print()

    bos_protos     = {}
    for bos_name in sorted(bos_data.keys()):
        embs = bos_data[bos_name]["embeddings"]
        bos_protos[bos_name] = F.normalize(embs.mean(0), dim=0)
    bos_proto_matrix = torch.stack([bos_protos[n] for n in sorted(bos_protos.keys())])

    # bos_to_zoo (30 x 10)
    bos_to_zoo = bos_proto_matrix @ proto_matrix.T
    # max similarity of each BOS prototype to any zoo prototype
    bos_max_zoo, bos_max_idx = bos_to_zoo.max(dim=1)
    bos_max_zoo_np = bos_max_zoo.numpy()

    print(f"  For each BOS individual, similarity to NEAREST zoo individual:")
    print(f"    Mean   : {bos_max_zoo_np.mean():.4f}")
    print(f"    Median : {np.median(bos_max_zoo_np):.4f}")
    print(f"    Std    : {bos_max_zoo_np.std():.4f}")
    print(f"    Min    : {bos_max_zoo_np.min():.4f}")
    print(f"    Max    : {bos_max_zoo_np.max():.4f}")
    print(f"    >= 0.30: {(bos_max_zoo_np >= 0.30).sum()} / {len(bos_max_zoo_np)} BOS individuals")
    print(f"    >= 0.50: {(bos_max_zoo_np >= 0.50).sum()} / {len(bos_max_zoo_np)} BOS individuals")

    # ------------------------------------------------------------------
    # FINAL VERDICT
    # ------------------------------------------------------------------
    title("VERDICT")

    rejection_pct = overall_rejection_rate * 100

    print(f"  At threshold {threshold}:")
    print(f"  -> {rejection_pct:.1f}% of BOS crops correctly identified as UNKNOWN")
    print()

    if rejection_pct >= 90:
        print("  EXCELLENT - the model handles unknowns very well.")
        print("  The current model is probably good enough for field deployment.")
        print("  Going further (MoCo, SupCon, etc.) is NOT necessary.")
    elif rejection_pct >= 75:
        print("  GOOD - the model rejects most unknowns but ~25% slip through.")
        print("  Worth investigating WHICH BOS individuals are confused with WHICH")
        print("  zoo individuals (see tables above).")
        print("  Consider raising the threshold, then re-test.")
    elif rejection_pct >= 50:
        print("  AVERAGE - the model has trouble rejecting many unknowns.")
        print("  A re-training phase (SupCon or ArcFace fine-tuning) is justified.")
    else:
        print("  POOR - the model has clearly overfit to its training conditions.")
        print("  A full re-training pipeline is justified.")

    print()
    print(f"  Threshold suggestions based on this run:")
    # Optimal threshold to maximize true unknown rejection
    # while keeping known individuals identifiable
    for t_try in [0.10, 0.15, 0.20, 0.22, 0.25, 0.30, 0.35, 0.40]:
        rej = (all_max_sims_np < t_try).mean() * 100
        print(f"    threshold={t_try:.2f} -> {rej:5.1f}% rejection rate")

    # ------------------------------------------------------------------
    # SAMPLE INDIVIDUAL CROPS (worst case for diagnostic)
    # ------------------------------------------------------------------
    section("Top 20 most-falsely-confident BOS crops (highest max_sim)")
    crop_records = []
    for bos_name, d in bos_data.items():
        sims = (d["embeddings"] @ proto_matrix.T).max(dim=1)
        for i, (s, idx) in enumerate(zip(sims.values.numpy(), sims.indices.numpy())):
            crop_records.append({
                "bos":          bos_name,
                "crop":         d["paths"][i].name,
                "max_sim":      float(s),
                "predicted_as": proto_names[int(idx)],
            })
    crop_records.sort(key=lambda x: x["max_sim"], reverse=True)

    print(f"  {'BOS_indiv':<14}  {'crop_name':<28}  {'pred_as':<12}  max_sim")
    print(f"  {'-'*14}  {'-'*28}  {'-'*12}  -------")
    for rec in crop_records[:20]:
        print(f"  {rec['bos']:<14}  {rec['crop'][:28]:<28}  "
              f"{rec['predicted_as']:<12}  {rec['max_sim']:.4f}")

    title("DONE - no files written, everything was read-only")
    print(f"  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()