# generate_v4_gallery.py
# Génère embeddings_v4.json depuis v4_best.pt
# Run this BEFORE reorganize_final.py --run
#
# RUN:
#   conda activate orangs
#   python v4_megadesc_arcface_40ind/03_export_gallery.py

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


import os, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime
from collections import defaultdict




import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import timm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Paths
V4_MODEL = V4_PT
ZOO_DIR = CROPS_KNOWN_DIR
BOS_DIR = CROPS_KNOWN_DIR
OUT_JSON = OUTPUT_DIR / "embeddings_v4.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

IMG_SIZE   = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN, STD  = [0.5,0.5,0.5], [0.5,0.5,0.5]
exts       = {".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"}

tf = T.Compose([T.Resize(IMG_SIZE), T.CenterCrop(IMG_SIZE),
                T.ToTensor(), T.Normalize(MEAN, STD)])

print(f"Device : {DEVICE}")
print(f"Loading {V4_MODEL.name}...")
ckpt    = torch.load(str(V4_MODEL), map_location=DEVICE, weights_only=False)
classes = ckpt["classes"]   # 40 individus
emb_dim = ckpt.get("emb_dim", 768)

model = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224",
                           pretrained=False, num_classes=0)
model.load_state_dict(ckpt["backbone_state"])
model = model.eval().to(DEVICE)
print(f"Model OK — {len(classes)} classes, emb_dim={emb_dim}")

@torch.no_grad()
def embed_dir(d: Path):
    imgs = sorted([f for f in d.iterdir() if f.suffix in exts])
    if not imgs: return None, 0
    embs = []
    for p in imgs:
        try:
            t = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE)
            embs.append(F.normalize(model(t), dim=1).cpu())
        except: pass
    if not embs: return None, 0
    e = torch.cat(embs)
    return F.normalize(e.mean(0), dim=0), len(embs)

# Build prototypes
gallery = {}
pos_sims, neg_protos_names = [], []

print("\nBuilding prototypes...")
for i, cls in enumerate(classes):
    # Chercher dans zoo ou BOS
    d = ZOO_DIR / cls
    if not d.exists():
        d = BOS_DIR / cls
    if not d.exists():
        print(f"  WARN: {cls} not found"); continue
    proto, n = embed_dir(d)
    if proto is None:
        print(f"  WARN: {cls} no images"); continue
    gallery[cls] = {"class_index": i, "num_crops": n, "embedding": proto.tolist()}
    print(f"  {cls:<16}: {n:3d} crops → prototype OK")

# Calibration simple
print("\nCalibrating threshold...")
protos = {c: torch.tensor(v["embedding"]) for c, v in gallery.items()}
names  = list(protos.keys())
pm     = torch.stack([protos[c] for c in names])

pos_sims, neg_sims = [], []
for i, cls in enumerate(names):
    d = (ZOO_DIR/cls) if (ZOO_DIR/cls).exists() else (BOS_DIR/cls)
    if not d.exists(): continue
    imgs = sorted([f for f in d.iterdir() if f.suffix in exts])[:20]
    for p in imgs:
        try:
            t   = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE)
            e   = F.normalize(model(t), dim=1).cpu()
            own = float((e @ protos[cls]).item())
            neg = float((e @ pm[[j for j in range(len(names)) if j!=i]].T).max().item())
            pos_sims.append(own); neg_sims.append(neg)
        except: pass

pos_m  = float(np.mean(pos_sims))
neg_m  = float(np.mean(neg_sims))
gap    = pos_m - neg_m
# Seuil = milieu entre neg 95e percentile et pos 5e percentile
thresh = float((np.percentile(neg_sims, 95) + np.percentile(pos_sims, 5)) / 2)
thresh = round(max(0.15, min(0.40, thresh)), 4)

print(f"  pos_mean={pos_m:.4f}  neg_mean={neg_m:.4f}  gap={gap:.4f}")
print(f"  Threshold calibré : {thresh}")

out = {
    "version":           "4.0-arcface-improved",
    "created":           datetime.now().isoformat(),
    "model":             "MegaDescriptor-T-224 + SubCenterArcFace V4",
    "embedding_dim":     emb_dim,
    "normalization":     {"mean": MEAN, "std": STD},
    "similarity_metric": "cosine",
    "unknown_threshold": thresh,
    "separability_gap":  round(gap, 4),
    "pos_sim_mean":      round(pos_m, 4),
    "neg_sim_mean":      round(neg_m, 4),
    "num_individuals":   len(gallery),
    "individuals":       gallery,
}

tmp = OUT_JSON.with_suffix(".tmp")
tmp.write_text(json.dumps(out, separators=(",",":")), encoding="utf-8")
tmp.replace(OUT_JSON)

print(f"\nSaved: {OUT_JSON}")
print(f"  {len(gallery)} individuals, {OUT_JSON.stat().st_size/1024:.1f} KB")
print(f"  Threshold : {thresh}")
print("Done.")
