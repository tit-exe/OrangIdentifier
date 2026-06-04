# stress_test.py
# Robustness test for the V3/V4 model under simulated degraded conditions.
# READ-ONLY — writes nothing, modifies nothing.
#
# Simulates what a field ranger sees on a camera:
#   - Motion/focus blur
#   - Under/over-exposure (backlight, shadow)
#   - Low resolution (photo taken from far away)
#   - Heavy JPEG compression (WhatsApp transfer)
#   - Rotation / angle (animal moving)
#   - Partial occlusion (branch in front of face)
#   - Combined degradations
#
# RUN:
#   conda activate wildlife-id
#   python common/stress_test.py

import os, sys, random, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import io

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.config_loader import (
    apply_cache_env,
    CROPS_KNOWN_DIR, V4_PT, UNKNOWN_THRESHOLD,
)
apply_cache_env()

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import timm
from PIL import Image, ImageFile, ImageFilter, ImageEnhance
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==============================================================================
# PATHS (read-only — stress test never writes)
# ==============================================================================

MODEL_PATH = V4_PT
ZOO_DIR    = CROPS_KNOWN_DIR

IMG_SIZE   = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEGA_MEAN  = [0.5, 0.5, 0.5]
MEGA_STD   = [0.5, 0.5, 0.5]
THRESHOLD  = UNKNOWN_THRESHOLD
SEED       = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ==============================================================================
# HELPERS
# ==============================================================================

def title(t):
    print(f"\n{'='*70}\n  {t}\n{'='*70}")

def section(t):
    print(f"\n  --- {t} ---")

def bar(v, width=30):
    if v < 0: v = 0
    if v > 1: v = 1
    f = int(width * v)
    return "█" * f + "░" * (width - f)

CLEAN_TF = T.Compose([
    T.Resize(IMG_SIZE),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(MEGA_MEAN, MEGA_STD),
])

def to_tensor(img):
    return CLEAN_TF(img).unsqueeze(0).to(DEVICE)

@torch.no_grad()
def embed(model, img_tensor):
    return F.normalize(model(img_tensor), dim=1).cpu()

# ==============================================================================
# DEGRADATION FUNCTIONS — simulate field conditions
# ==============================================================================

def degrade_blur_motion(img, severity):
    """Motion blur — animal moving or shaky camera."""
    radius = [0, 1.5, 3.0, 5.0, 8.0][severity]
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def degrade_blur_focus(img, severity):
    """Focus blur — autofocus failure."""
    radius = [0, 2.0, 4.0, 7.0, 12.0][severity]
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def degrade_exposure(img, severity):
    """Under/over-exposure — backlight or deep shadow in forest."""
    factors = [1.0, 0.4, 0.2, 2.5, 4.0][severity]
    return ImageEnhance.Brightness(img).enhance(factors)

def degrade_low_res(img, severity):
    """Low resolution — distant shot then cropped."""
    scales = [1.0, 0.5, 0.25, 0.15, 0.08][severity]
    w, h = img.size
    small_w = max(int(w * scales), 8)
    small_h = max(int(h * scales), 8)
    return img.resize((small_w, small_h), Image.BILINEAR).resize((w, h), Image.BILINEAR)

def degrade_jpeg(img, severity):
    """JPEG compression — WhatsApp transfer or compressed storage."""
    qualities = [95, 50, 25, 10, 3][severity]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=qualities)
    buf.seek(0)
    return Image.open(buf).copy()

def degrade_rotation(img, severity):
    """Rotation — animal tilted or camera at an angle."""
    angles = [0, 15, 30, 45, 75][severity]
    return img.rotate(angle=angles, expand=False, fillcolor=(128, 100, 80))

def degrade_occlusion(img, severity):
    """Partial occlusion — branch, leaf, or hand in front of face."""
    img = img.copy()
    w, h = img.size
    fractions = [0, 0.15, 0.30, 0.45, 0.60][severity]
    pixels = img.load()
    bar_h = int(h * fractions)
    y_start = random.randint(0, max(1, h - bar_h))
    for y in range(y_start, min(y_start + bar_h, h)):
        for x in range(w):
            pixels[x, y] = (20, 60, 20)  # forest green
    return img

def degrade_noise(img, severity):
    """Digital noise — low-end sensor, high ISO."""
    arr = np.array(img, dtype=np.float32)
    stds = [0, 10, 25, 45, 70][severity]
    noise = np.random.randn(*arr.shape) * stds
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def degrade_combined(img, severity):
    """Realistic combination: blur + under-exposure + compression."""
    if severity == 0: return img
    img = degrade_blur_motion(img, min(severity, 3))
    img = degrade_exposure(img, 1 if severity >= 2 else 0)
    img = degrade_jpeg(img, min(severity, 3))
    return img

DEGRADATIONS = {
    "Motion blur":    degrade_blur_motion,
    "Focus blur":     degrade_blur_focus,
    "Exposure":       degrade_exposure,
    "Low resolution": degrade_low_res,
    "JPEG compress":  degrade_jpeg,
    "Rotation":       degrade_rotation,
    "Occlusion":      degrade_occlusion,
    "Digital noise":  degrade_noise,
    "Combined":       degrade_combined,
}

SEVERITY_LABELS = ["None", "Mild", "Moderate", "Strong", "Extreme"]

# ==============================================================================
# LOAD MODEL + BUILD PROTOTYPES
# ==============================================================================

title("STRESS TEST — simulated field conditions (READ-ONLY)")
print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Device  : {DEVICE}")

ckpt    = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
classes = ckpt["classes"]
model   = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224",
                             pretrained=False, num_classes=0)
model.load_state_dict(ckpt["backbone_state"])
model   = model.eval().to(DEVICE)
print(f"  Model   : epoch {ckpt.get('epoch','?')}, val_acc {ckpt.get('val_acc',0)*100:.1f}%")

exts = {".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"}

# Load a sample of crops per individual
N_SAMPLES = 8
zoo_crops = {}

for cls in classes:
    d = ZOO_DIR / cls
    if not d.exists(): continue
    imgs = sorted([f for f in d.iterdir() if f.suffix in exts])
    selected = random.sample(imgs, min(N_SAMPLES, len(imgs)))
    zoo_crops[cls] = [Image.open(p).convert("RGB") for p in selected]

# Build prototypes from ALL crops
print("\n  Building prototypes from all zoo crops...")
proto = {}
for cls in classes:
    d = ZOO_DIR / cls
    if not d.exists(): continue
    all_imgs = sorted([f for f in d.iterdir() if f.suffix in exts])
    embs = []
    for p in all_imgs:
        try:
            t = to_tensor(Image.open(p).convert("RGB"))
            embs.append(embed(model, t))
        except: pass
    proto[cls] = F.normalize(torch.stack(embs).squeeze(1).mean(0), dim=0)
    print(f"    {cls:<12}: {len(all_imgs)} crops")

proto_matrix = torch.stack([proto[c] for c in classes])  # (N, emb_dim)

# ==============================================================================
# BASELINE (no degradation)
# ==============================================================================

title("BASELINE — clean crops (no degradation)")

baseline_sims = defaultdict(list)
for cls in classes:
    for img in zoo_crops[cls]:
        t    = to_tensor(img)
        emb  = embed(model, t)
        sims = (emb @ proto_matrix.T)[0]
        max_sim, max_idx = sims.max().item(), sims.argmax().item()
        correct = classes[max_idx] == cls
        baseline_sims[cls].append((max_sim, correct))

print(f"\n  {'Individual':<12}  {'sim_mean':>9}  {'accuracy':>9}")
print(f"  {'-'*12}  {'-'*9}  {'-'*9}")
all_correct = []
for cls in classes:
    data    = baseline_sims[cls]
    sim_m   = np.mean([d[0] for d in data])
    acc     = np.mean([d[1] for d in data])
    all_correct.append(acc)
    flag = "OK" if acc == 1.0 else ("?" if acc >= 0.75 else "BAD")
    print(f"  {cls:<12}  {sim_m:>9.4f}  {acc*100:>8.1f}%  {flag}")

print(f"\n  Overall baseline accuracy: {np.mean(all_correct)*100:.1f}%")

# ==============================================================================
# STRESS TEST BY DEGRADATION TYPE
# ==============================================================================

title("STRESS TEST BY DEGRADATION TYPE")
print(f"  {N_SAMPLES} crops x {len(classes)} individuals x 5 severities x {len(DEGRADATIONS)} types")
print(f"  = {N_SAMPLES * len(classes) * 5 * len(DEGRADATIONS)} total inferences\n")

results = {}

for degrad_name, degrad_fn in DEGRADATIONS.items():
    results[degrad_name] = {}
    section(degrad_name)
    print(f"  {'Severity':<18}  {'accuracy':>9}  {'sim_mean':>9}  {'sim_std':>8}  {'rejection':>9}  bar")

    for sev in range(5):
        accs, sims, rejections = [], [], []

        for cls in classes:
            for img in zoo_crops[cls]:
                try:
                    deg_img  = degrad_fn(img, sev)
                    t        = to_tensor(deg_img)
                    emb      = embed(model, t)
                    sim_vec  = (emb @ proto_matrix.T)[0]
                    max_sim  = sim_vec.max().item()
                    pred_cls = classes[sim_vec.argmax().item()]

                    accs.append(pred_cls == cls)
                    sims.append(max_sim)
                    rejections.append(max_sim < THRESHOLD)
                except:
                    pass

        acc      = np.mean(accs)       if accs else 0
        sim_mean = np.mean(sims)       if sims else 0
        sim_std  = np.std(sims)        if sims else 0
        rej_rate = np.mean(rejections) if rejections else 0

        results[degrad_name][sev] = {
            "acc": acc, "sim_mean": sim_mean,
            "sim_std": sim_std, "rejection_rate": rej_rate
        }

        b    = bar(acc, width=25)
        flag = "✓" if acc >= 0.90 else ("~" if acc >= 0.70 else "✗")
        print(f"  {SEVERITY_LABELS[sev]:<18}  {acc*100:>8.1f}%  "
              f"{sim_mean:>9.4f}  {sim_std:>8.4f}  "
              f"{rej_rate*100:>8.1f}%  {b} {flag}")

# ==============================================================================
# SUMMARY TABLE — breaking point by type
# ==============================================================================

title("SUMMARY — breaking point per degradation type")
print("  For each degradation type: at what severity does accuracy drop below 80%?\n")

print(f"  {'Degradation type':<22}  {'Mild':>8}  {'Moderate':>8}  {'Strong':>8}  {'Extreme':>8}  Break")
print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

rupture_results = {}
for degrad_name, sev_data in results.items():
    row = []
    rupture = "None"
    for sev in [1, 2, 3, 4]:
        acc = sev_data[sev]["acc"]
        row.append(f"{acc*100:>7.1f}%")
        if acc < 0.80 and rupture == "None":
            rupture = SEVERITY_LABELS[sev]
    rupture_results[degrad_name] = rupture
    print(f"  {degrad_name:<22}  {'  '.join(row)}  {rupture}")

# ==============================================================================
# PER-INDIVIDUAL FRAGILITY — moderate severity
# ==============================================================================

title("MOST FRAGILE INDIVIDUALS — severity 3 (strong) across all degradation types")

fragility = defaultdict(list)

for degrad_name, degrad_fn in DEGRADATIONS.items():
    for cls in classes:
        accs = []
        for img in zoo_crops[cls]:
            try:
                deg_img  = degrad_fn(img, 3)
                t        = to_tensor(deg_img)
                emb      = embed(model, t)
                sim_vec  = (emb @ proto_matrix.T)[0]
                pred_cls = classes[sim_vec.argmax().item()]
                accs.append(pred_cls == cls)
            except:
                pass
        if accs:
            fragility[cls].append(np.mean(accs))

print(f"\n  {'Individual':<12}  {'acc_mean':>9}  {'acc_min':>9}  {'acc_max':>9}  fragility")
print(f"  {'-'*12}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*20}")

sorted_frag = sorted(fragility.items(), key=lambda kv: np.mean(kv[1]))
for cls, accs in sorted_frag:
    m   = np.mean(accs)
    mn  = np.min(accs)
    mx  = np.max(accs)
    b_s = bar(m, width=20)
    flag = "ROBUST" if m >= 0.85 else ("MEDIUM" if m >= 0.65 else "FRAGILE")
    print(f"  {cls:<12}  {m*100:>8.1f}%  {mn*100:>8.1f}%  {mx*100:>8.1f}%  {b_s} {flag}")

# ==============================================================================
# FINAL VERDICT
# ==============================================================================

title("VERDICT — is retraining justified?")

worst_degrad  = min(results.items(),
    key=lambda kv: np.mean([kv[1][s]["acc"] for s in [2,3,4]]))
worst_name    = worst_degrad[0]
worst_acc_mod = worst_degrad[1][2]["acc"]
worst_acc_str = worst_degrad[1][3]["acc"]

print(f"""
  Baseline (no degradation)      : {np.mean(all_correct)*100:.1f}%
  Worst degradation type         : {worst_name}
  Accuracy at moderate severity  : {worst_acc_mod*100:.1f}%
  Accuracy at strong severity    : {worst_acc_str*100:.1f}%

  Types with break point at mild or moderate:""")

critical = [(n, r) for n, r in rupture_results.items() if r in ["Mild", "Moderate"]]
if critical:
    for name, rup in critical:
        print(f"    - {name}: breaks at '{rup}'  <- NEEDS ATTENTION")
else:
    print("    None — model holds up to moderate severity on all degradation types")

print(f"""
  Fragile individuals (acc < 65% under degradation):""")

fragile_inds = [(cls, np.mean(accs)) for cls, accs in sorted_frag if np.mean(accs) < 0.65]
if fragile_inds:
    for cls, acc in fragile_inds:
        print(f"    - {cls}: {acc*100:.1f}% <- retrain with stronger augmentation")
else:
    print("    None — all individuals are robust")

n_critical = len(critical)
n_fragile  = len(fragile_inds)

if n_critical == 0 and n_fragile == 0:
    print("""
  CONCLUSION: Model is robust under simulated degraded conditions.
  Retraining is NOT justified for robustness alone.

  What could justify retraining:
  -> Add new labeled individuals to expand coverage
  -> Test on real field photos (the true judge)
    """)
elif n_critical > 0 or n_fragile > 2:
    print(f"""
  CONCLUSION: RETRAINING JUSTIFIED.
  {n_critical} critical degradation type(s), {n_fragile} fragile individual(s).

  Recommended actions:
  -> Use stronger augmentation during training
  -> Include these degradation types in training augmentations
  -> Add more crops for fragile individuals
    """)
else:
    print(f"""
  CONCLUSION: RETRAINING OPTIONAL.
  Minor weaknesses only — nothing critical.
  Test on real field photos before deciding.
    """)

print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
