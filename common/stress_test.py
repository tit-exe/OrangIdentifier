# V3_stress_test.py
# Test de robustesse du modele V3 en conditions degrades simulees
# READ-ONLY - n'ecrit rien, ne modifie rien
#
# Simule ce qu'un ranger voit sur le terrain :
#   - Flou (bougé, mise au point ratée)
#   - Sous-exposition / sur-exposition (contre-jour, ombre)
#   - Faible résolution (photo prise de loin)
#   - Compression JPEG agressive (envoi WhatsApp)
#   - Rotation / angle (orang qui bouge)
#   - Occlusion partielle (branche devant le visage)
#   - Combinaison de plusieurs dégradations
#
# RUN:
#   conda activate orangs
#   python D:\OrangIdentifier\V2\scripts\V3_stress_test.py

import os, sys, random, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import io

os.environ["HF_HOME"]    = r"D:\HuggingFaceCache"
os.environ["TORCH_HOME"] = r"D:\TorchCache"

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import timm
from PIL import Image, ImageFile, ImageFilter, ImageEnhance
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==============================================================================
# PATHS (tous en lecture seule)
# ==============================================================================

MODEL_PATH = Path(r"D:\OrangIdentifier\V2\MODELS\megadesc_T_arcface.pt")
ZOO_DIR    = Path(r"D:\OrangIdentifier\DATASET_CLASSIFICATION\raw")

IMG_SIZE   = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEGA_MEAN  = [0.5, 0.5, 0.5]
MEGA_STD   = [0.5, 0.5, 0.5]
THRESHOLD  = 0.22
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
# DEGRADATIONS — simulent les conditions terrain
# ==============================================================================

def degrade_blur_motion(img, severity):
    """Flou de bougé — orang qui se déplace ou ranger qui bouge."""
    radius = [0, 1.5, 3.0, 5.0, 8.0][severity]
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def degrade_blur_focus(img, severity):
    """Flou de mise au point — autofocus raté."""
    radius = [0, 2.0, 4.0, 7.0, 12.0][severity]
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def degrade_exposure(img, severity):
    """Sous/sur-exposition — contre-jour en forêt."""
    factors = [1.0, 0.4, 0.2, 2.5, 4.0][severity]
    return ImageEnhance.Brightness(img).enhance(factors)

def degrade_low_res(img, severity):
    """Faible résolution — photo prise de loin puis recadrée."""
    scales = [1.0, 0.5, 0.25, 0.15, 0.08][severity]
    w, h = img.size
    small_w = max(int(w * scales), 8)
    small_h = max(int(h * scales), 8)
    return img.resize((small_w, small_h), Image.BILINEAR).resize((w, h), Image.BILINEAR)

def degrade_jpeg(img, severity):
    """Compression JPEG — envoi WhatsApp ou stockage compressé."""
    qualities = [95, 50, 25, 10, 3][severity]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=qualities)
    buf.seek(0)
    return Image.open(buf).copy()

def degrade_rotation(img, severity):
    """Rotation — orang penché ou ranger qui filme de côté."""
    angles = [0, 15, 30, 45, 75][severity]
    return img.rotate(angle=angles, expand=False, fillcolor=(128, 100, 80))

def degrade_occlusion(img, severity):
    """Occlusion partielle — branche, feuille, main devant le visage."""
    img = img.copy()
    w, h = img.size
    fractions = [0, 0.15, 0.30, 0.45, 0.60][severity]
    pixels = img.load()
    # Barre horizontale aléatoire (branche)
    bar_h = int(h * fractions)
    y_start = random.randint(0, max(1, h - bar_h))
    for y in range(y_start, min(y_start + bar_h, h)):
        for x in range(w):
            pixels[x, y] = (20, 60, 20)  # vert forêt
    return img

def degrade_noise(img, severity):
    """Bruit numérique — capteur bas de gamme, ISO élevé."""
    arr = np.array(img, dtype=np.float32)
    stds = [0, 10, 25, 45, 70][severity]
    noise = np.random.randn(*arr.shape) * stds
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def degrade_combined(img, severity):
    """Combinaison réaliste : flou + sous-expo + compression."""
    if severity == 0: return img
    img = degrade_blur_motion(img, min(severity, 3))
    img = degrade_exposure(img, 1 if severity >= 2 else 0)
    img = degrade_jpeg(img, min(severity, 3))
    return img

DEGRADATIONS = {
    "Flou bougé":       degrade_blur_motion,
    "Flou focus":       degrade_blur_focus,
    "Exposition":       degrade_exposure,
    "Faible résolution": degrade_low_res,
    "Compression JPEG": degrade_jpeg,
    "Rotation angle":   degrade_rotation,
    "Occlusion":        degrade_occlusion,
    "Bruit numérique":  degrade_noise,
    "Combiné (réaliste)": degrade_combined,
}

SEVERITY_LABELS = ["Aucune", "Légère", "Modérée", "Forte", "Extrême"]

# ==============================================================================
# LOAD MODEL + BUILD PROTOTYPES
# ==============================================================================

title("V3 STRESS TEST — simulation conditions terrain (READ-ONLY)")
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

# Charger quelques crops par individu
N_SAMPLES = 8   # crops par individu pour le test
zoo_crops = {}  # class -> list of PIL images

for cls in classes:
    d = ZOO_DIR / cls
    if not d.exists(): continue
    imgs = sorted([f for f in d.iterdir() if f.suffix in exts])
    selected = random.sample(imgs, min(N_SAMPLES, len(imgs)))
    zoo_crops[cls] = [Image.open(p).convert("RGB") for p in selected]

# Build prototypes from ALL crops (not just the sample)
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

proto_matrix = torch.stack([proto[c] for c in classes])  # (10, 768)

# ==============================================================================
# BASELINE (pas de dégradation)
# ==============================================================================

title("BASELINE — crops sans dégradation")

baseline_sims = defaultdict(list)
for cls in classes:
    for img in zoo_crops[cls]:
        t    = to_tensor(img)
        emb  = embed(model, t)
        sims = (emb @ proto_matrix.T)[0]
        max_sim, max_idx = sims.max().item(), sims.argmax().item()
        correct = classes[max_idx] == cls
        baseline_sims[cls].append((max_sim, correct))

print(f"\n  {'Individu':<12}  {'sim_mean':>9}  {'accuracy':>9}")
print(f"  {'-'*12}  {'-'*9}  {'-'*9}")
all_correct = []
for cls in classes:
    data    = baseline_sims[cls]
    sim_m   = np.mean([d[0] for d in data])
    acc     = np.mean([d[1] for d in data])
    all_correct.append(acc)
    flag = "OK" if acc == 1.0 else ("?" if acc >= 0.75 else "BAD")
    print(f"  {cls:<12}  {sim_m:>9.4f}  {acc*100:>8.1f}%  {flag}")

print(f"\n  Accuracy globale baseline : {np.mean(all_correct)*100:.1f}%")

# ==============================================================================
# STRESS TEST PAR TYPE DE DEGRADATION
# ==============================================================================

title("STRESS TEST PAR TYPE DE DEGRADATION")
print(f"  {N_SAMPLES} crops × {len(classes)} individus × 5 sévérités × {len(DEGRADATIONS)} types")
print(f"  = {N_SAMPLES * len(classes) * 5 * len(DEGRADATIONS)} inférences totales\n")

results = {}  # degrad_name -> {severity -> {acc, sim_mean, rejection_rate}}

for degrad_name, degrad_fn in DEGRADATIONS.items():
    results[degrad_name] = {}
    section(degrad_name)
    print(f"  {'Sévérité':<18}  {'accuracy':>9}  {'sim_mean':>9}  {'sim_std':>8}  {'rejet':>8}  bar")

    for sev in range(5):
        accs, sims, rejections = [], [], []

        for cls in classes:
            for img in zoo_crops[cls]:
                try:
                    deg_img = degrad_fn(img, sev)
                    t       = to_tensor(deg_img)
                    emb     = embed(model, t)
                    sim_vec = (emb @ proto_matrix.T)[0]
                    max_sim = sim_vec.max().item()
                    pred_cls = classes[sim_vec.argmax().item()]

                    accs.append(pred_cls == cls)
                    sims.append(max_sim)
                    rejections.append(max_sim < THRESHOLD)
                except:
                    pass

        acc      = np.mean(accs)   if accs else 0
        sim_mean = np.mean(sims)   if sims else 0
        sim_std  = np.std(sims)    if sims else 0
        rej_rate = np.mean(rejections) if rejections else 0

        results[degrad_name][sev] = {
            "acc": acc, "sim_mean": sim_mean,
            "sim_std": sim_std, "rejection_rate": rej_rate
        }

        # Visual bar colored by severity drop
        b = bar(acc, width=25)
        flag = "✓" if acc >= 0.90 else ("~" if acc >= 0.70 else "✗")
        print(f"  {SEVERITY_LABELS[sev]:<18}  {acc*100:>8.1f}%  "
              f"{sim_mean:>9.4f}  {sim_std:>8.4f}  "
              f"{rej_rate*100:>7.1f}%  {b} {flag}")

# ==============================================================================
# TABLEAU RECAPITULATIF — seuil de rupture par type
# ==============================================================================

title("TABLEAU RÉCAPITULATIF — seuil de rupture")
print("  Pour chaque type de dégradation, à quelle sévérité l'accuracy tombe < 80% ?\n")

print(f"  {'Type dégradation':<22}  {'Légère':>8}  {'Modérée':>8}  {'Forte':>8}  {'Extrême':>8}  Rupture")
print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*7}")

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
# ANALYSE PAR INDIVIDU SOUS CONDITIONS DIFFICILES
# ==============================================================================

title("INDIVIDUS LES PLUS FRAGILES — sévérité modérée (3) sur toutes dégradations")

fragility = defaultdict(list)  # cls -> list of accuracies

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

print(f"\n  {'Individu':<12}  {'acc_mean':>9}  {'acc_min':>9}  {'acc_max':>9}  fragilité")
print(f"  {'-'*12}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*20}")

sorted_frag = sorted(fragility.items(), key=lambda kv: np.mean(kv[1]))
for cls, accs in sorted_frag:
    m = np.mean(accs)
    mn = np.min(accs)
    mx = np.max(accs)
    b_str = bar(m, width=20)
    flag = "ROBUSTE" if m >= 0.85 else ("MOYEN" if m >= 0.65 else "FRAGILE")
    print(f"  {cls:<12}  {m*100:>8.1f}%  {mn*100:>8.1f}%  {mx*100:>8.1f}%  {b_str} {flag}")

# ==============================================================================
# VERDICT FINAL
# ==============================================================================

title("VERDICT — est-ce qu'un réentraînement est justifié ?")

# Trouver la dégradation la plus critique
worst_degrad = min(results.items(),
    key=lambda kv: np.mean([kv[1][s]["acc"] for s in [2,3,4]]))
worst_name   = worst_degrad[0]
worst_acc_mod = worst_degrad[1][2]["acc"]
worst_acc_str = worst_degrad[1][3]["acc"]

baseline_acc = np.mean([np.mean([d[0] for d in baseline_sims[cls]]) for cls in classes])

print(f"""
  Baseline (pas de dégradation)     : {np.mean(all_correct)*100:.1f}%
  Pire dégradation                  : {worst_name}
  Accuracy à sévérité modérée       : {worst_acc_mod*100:.1f}%
  Accuracy à sévérité forte         : {worst_acc_str*100:.1f}%

  Types où rupture < sévérité modérée :""")

critical = [(n, r) for n, r in rupture_results.items() if r in ["Légère", "Modérée"]]
if critical:
    for name, rup in critical:
        print(f"    - {name} : rupture dès '{rup}'  ← PROBLÈME À RÉGLER")
else:
    print("    Aucun — le modèle tient jusqu'à sévérité modérée sur tous les types")

print(f"""
  Individus fragiles (acc < 65% en conditions dégradées) :""")

fragile_inds = [(cls, np.mean(accs)) for cls, accs in sorted_frag if np.mean(accs) < 0.65]
if fragile_inds:
    for cls, acc in fragile_inds:
        print(f"    - {cls} : {acc*100:.1f}% ← réentraîner avec plus d'augmentation")
else:
    print("    Aucun — tous les individus sont robustes")

print(f"""
  CONCLUSION :""")

n_critical = len(critical)
n_fragile  = len(fragile_inds)

if n_critical == 0 and n_fragile == 0:
    print("""
    Le modèle est robuste aux conditions dégradées simulées.
    Un réentraînement N'EST PAS justifié pour la robustesse.
    
    Ce qui pourrait justifier un réentraînement :
    → Intégrer les 30 BOS comme individus connus (gain mesurable)
    → Tester sur de vraies photos terrain (vrai juge)
    """)
elif n_critical > 0 or n_fragile > 2:
    print(f"""
    RÉENTRAÎNEMENT JUSTIFIÉ.
    {n_critical} type(s) de dégradation critique(s), {n_fragile} individu(s) fragile(s).
    
    Ce qu'il faut faire :
    → Augmentation plus agressive pendant l'entraînement
    → Inclure ces types de dégradation dans les augmentations
    → Potentiellement plus de crops par individu fragile
    """)
else:
    print(f"""
    RÉENTRAÎNEMENT OPTIONNEL.
    Quelques faiblesses mineures mais rien de critique.
    Tester sur de vraies photos terrain avant de décider.
    """)

print(f"  Finished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")