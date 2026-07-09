"""
V6_6_export_tflite.py — OrangIdentifier V6
============================================
Exporte le backbone V6 en TFLite (float32) et déploie dans les assets Android.
Ce script tourne sous WSL2/Linux — litert_torch n'existe qu'en Linux.

RUN (depuis PowerShell) :
  wsl -d Ubuntu -- bash -c "/root/miniconda3/bin/conda run -n orangs_export --no-capture-output python <repo>/v6_megadesc_arcface_15ind/05_export_tflite.py"

Fichiers produits :
  output/v6/models/v6_backbone.tflite
  .../assets/megadesc_v6_backbone.tflite          ← remplace V3
  .../assets/gallery.json                        ← remplace V3
  .../assets/yolo_v2_detector.tflite             ← inchangé
"""

import sys, shutil, json
from pathlib import Path
from datetime import datetime

# ── Logging ───────────────────────────────────────────────────────────────────
def log(msg="", level=""):
    tag = f"[{level}] " if level else ""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {tag}{msg}", flush=True)

def hr(title=""):
    bar = "─" * 68
    log()
    log(bar)
    if title:
        log(f"  {title}")
        log(bar)

# ── Portable paths (Path(__file__) resolves under /mnt/<drive> inside WSL too) ──
REPO         = Path(__file__).resolve().parents[1]
BACKBONE_PT  = REPO / "output" / "v6" / "models" / "v6_backbone_only.pt"
GALLERY_SRC  = REPO / "output" / "v6" / "models" / "v6_gallery.json"
TFLITE_OUT   = REPO / "output" / "v6" / "models" / "v6_backbone.tflite"

# The Android app lives in a SEPARATE repository (OrangIdentifier-Android).
# This script writes the .tflite and gallery.json into output/ and prints where
# to copy them. It does NOT write into the app repo directly.
ANDROID_ASSETS = REPO / "output" / "v6" / "android_assets"
ANDROID_MODEL  = ANDROID_ASSETS / "megadesc_v6_backbone.tflite"
ANDROID_GAL    = ANDROID_ASSETS / "gallery.json"
ANDROID_YOLO   = ANDROID_ASSETS / "yolo_v2_detector.tflite"  # do not touch in the app

IMG_SIZE = 224

# ══════════════════════════════════════════════════════════════════════════════
# 1. PRE-FLIGHT
# ══════════════════════════════════════════════════════════════════════════════
hr("Pre-flight checks")

errors = []
for p, label in [(BACKBONE_PT, "v6_backbone_only.pt"),
                 (GALLERY_SRC, "v6_gallery.json"),
                 (ANDROID_ASSETS, "android assets dir")]:
    if p.exists():
        sz = p.stat().st_size if p.is_file() else 0
        log(f"  ✓  {label:<35} {sz/1e6:6.1f} MB" if sz else f"  ✓  {label}/")
    else:
        log(f"  ✗  NOT FOUND: {p}", "ERR")
        errors.append(str(p))

if ANDROID_YOLO.exists():
    log(f"  ✓  yolo_v2_detector.tflite   (will NOT be touched)")
else:
    log(f"  !  yolo_v2_detector.tflite not found in assets", "WARN")

if errors:
    log(f"Aborting — missing: {errors}", "ERR")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD V6 BACKBONE
# ══════════════════════════════════════════════════════════════════════════════
hr("Loading V6 backbone")

import torch
import timm

log(f"  torch   : {torch.__version__}")
log(f"  timm    : {timm.__version__}")

# MegaDescriptor-T-224 = swin_tiny_patch4_window7_224 (num_classes=0 → 768-dim features)
# Try HuggingFace hub first (likely cached from training), fall back to local timm name
# to avoid hard dependency on network access.
log("  Creating model architecture (MegaDescriptor-T-224 = swin_tiny_patch4_window7_224)...")
try:
    bb = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224", pretrained=False, num_classes=0)
    log("  Architecture loaded via hf-hub:BVRA/MegaDescriptor-T-224")
except Exception as e_hf:
    log(f"  HF hub failed ({e_hf}) — falling back to swin_tiny_patch4_window7_224")
    bb = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=0)
    log("  Architecture loaded via timm (swin_tiny_patch4_window7_224)")

log(f"\n  Loading weights from {BACKBONE_PT.name} "
    f"({BACKBONE_PT.stat().st_size/1e6:.0f} MB)...")
ck = torch.load(str(BACKBONE_PT), map_location="cpu", weights_only=False)

log(f"  Checkpoint keys : {list(ck.keys())}")

state = ck["backbone_state"]   # confirmed key from training script
missing, unexpected = bb.load_state_dict(state, strict=True)
# strict=True — we want to know immediately if the architecture diverges
log(f"  Weights loaded  : strict=True, {len(missing)} missing, {len(unexpected)} unexpected")

if missing or unexpected:
    log(f"  Missing  : {missing[:5]}", "WARN")
    log(f"  Unexpected: {unexpected[:5]}", "WARN")

bb = bb.eval().cpu()

# Sanity check: one forward pass
log("  Running forward pass [1, 3, 224, 224]...")
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
with torch.no_grad():
    out = bb(dummy)
log(f"  Output shape    : {out.shape}  (expected torch.Size([1, 768]))")

if tuple(out.shape) != (1, 768):
    log(f"Wrong output shape {out.shape} — aborting", "ERR")
    sys.exit(1)

# Print normalization from checkpoint so it can be documented
norm = ck.get("normalization", {})
log(f"  Normalization   : mean={norm.get('mean')}  std={norm.get('std')}")
log(f"  Embedding dim   : {ck.get('emb_dim', 768)}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. LOAD CONVERTER
# ══════════════════════════════════════════════════════════════════════════════
hr("Loading TFLite converter")

converter = None
for pkg_name in ("litert_torch", "ai_edge_torch"):
    try:
        mod = __import__(pkg_name)
        if hasattr(mod, "convert"):
            converter = mod
            ver = getattr(mod, "__version__", "?")
            log(f"  Using {pkg_name}  (version {ver})")
            break
        else:
            log(f"  {pkg_name} found but no .convert() — skipping")
    except ImportError:
        log(f"  {pkg_name} not installed")

if converter is None:
    log("No converter found. Install: pip install litert-torch", "ERR")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# 4. CONVERT TO TFLITE
# ══════════════════════════════════════════════════════════════════════════════
hr("Converting to TFLite (float32)")
log("  Input : NCHW float32 [1, 3, 224, 224]")
log("  Output: float32 [1, 768]  (L2-normalize in the app after inference)")
log("  This may take 3–8 minutes on CPU...")
log()

TFLITE_OUT.parent.mkdir(parents=True, exist_ok=True)

try:
    edge_model = converter.convert(bb, (dummy,))
    edge_model.export(str(TFLITE_OUT))
except Exception as e:
    log(f"Conversion failed: {e}", "ERR")
    import traceback; traceback.print_exc()
    sys.exit(1)

tflite_size_mb = TFLITE_OUT.stat().st_size / 1e6
log(f"  Saved : {TFLITE_OUT}")
log(f"  Size  : {tflite_size_mb:.1f} MB")

# Validate TFLite magic bytes (FlatBuffers identifier "TFL3" at bytes 4–7)
with open(TFLITE_OUT, "rb") as f:
    f.seek(4)
    magic = f.read(4)
if magic == b"TFL3":
    log("  Magic : TFL3 ✓  (valid TFLite FlatBuffer)")
else:
    log(f"  Magic : {magic!r}  ← unexpected, file may be corrupt", "WARN")

# ══════════════════════════════════════════════════════════════════════════════
# 5. DEPLOY TO ANDROID ASSETS
# ══════════════════════════════════════════════════════════════════════════════
hr("Deploying to Android assets")

# Snapshot before
before = {p.name: p.stat().st_size for p in ANDROID_ASSETS.iterdir() if p.is_file()}
log("  Before:")
for name, sz in sorted(before.items()):
    log(f"    {name:<52} {sz/1e6:6.1f} MB")

log()

# Replace backbone
old_bb = ANDROID_MODEL.stat().st_size if ANDROID_MODEL.exists() else 0
shutil.copy2(str(TFLITE_OUT), str(ANDROID_MODEL))
new_bb = ANDROID_MODEL.stat().st_size
log(f"  Backbone : {old_bb/1e6:.1f} MB → {new_bb/1e6:.1f} MB")
log(f"    {ANDROID_MODEL}")

# Replace gallery
old_gal = ANDROID_GAL.stat().st_size if ANDROID_GAL.exists() else 0
shutil.copy2(str(GALLERY_SRC), str(ANDROID_GAL))
new_gal = ANDROID_GAL.stat().st_size
log(f"  Gallery  : {old_gal/1e6:.1f} MB → {new_gal/1e6:.1f} MB")
log(f"    {ANDROID_GAL}")

# Artifact check
after = {p.name: p.stat().st_size for p in ANDROID_ASSETS.iterdir() if p.is_file()}
expected = {"megadesc_v6_backbone.tflite", "gallery.json", "yolo_v2_detector.tflite"}
unexpected_files = set(after.keys()) - expected

log()
if unexpected_files:
    log(f"  !  Unexpected files in assets: {unexpected_files}", "WARN")
    log("     Delete them manually to keep assets clean.", "WARN")
else:
    log("  Assets folder clean — no unexpected files ✓")

log()
log("  After:")
for name, sz in sorted(after.items()):
    tag = " ← UPDATED" if name in {ANDROID_MODEL.name, ANDROID_GAL.name} else ""
    log(f"    {name:<52} {sz/1e6:6.1f} MB{tag}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. V6 FORMAT REFERENCE  (what to implement in the Android app)
# ══════════════════════════════════════════════════════════════════════════════
hr("V6 FORMAT REFERENCE — implement this in the Android app")

# Read gallery for accurate stats
with open(GALLERY_SRC, "r", encoding="utf-8") as f:
    gal = json.load(f)

n_ind      = gal["num_individuals"]
emb_dim    = gal["embedding_dim"]
threshold  = gal["unknown_threshold"]
norm_name  = gal["normalization"]
n_exemplars = gal["individuals"][next(iter(gal["individuals"]))]["num_exemplars"]

log(f"""
  ┌─ TFLite model (megadesc_v6_backbone.tflite) ─────────────────────────┐
  │  Input  : float32 NCHW  [1, 3, 224, 224]                            │
  │  Output : float32       [1, 768]  (raw, NOT L2-normalized)          │
  │  → L2-normalize the output in the app before computing scores        │
  └───────────────────────────────────────────────────────────────────────┘

  ┌─ Preprocessing ───────────────────────────────────────────────────────┐
  │  Resize image to 224×224                                             │
  │  pixel_float = pixel / 255.0                                         │
  │  pixel_norm  = (pixel_float - 0.5) / 0.5                            │
  │  → same for R, G, B channels  (mean=0.5, std=0.5)                   │
  │  normalization key in gallery.json: "{norm_name}"          │
  └───────────────────────────────────────────────────────────────────────┘

  ┌─ Gallery (gallery.json) ──────────────────────────────────────────────┐
  │  individuals    : {n_ind}                                              │
  │  embedding_dim  : {emb_dim}                                            │
  │  unknown_threshold: {threshold}  (cosine similarity)                 │
  │                                                                       │
  │  Per-individual structure:                                            │
  │    "Auti": {{                                                          │
  │      "class_index" : int,                                            │
  │      "num_exemplars": {n_exemplars},                                          │
  │      "embedding"  : [768 floats],   // single prototype (centroid)  │
  │      "exemplars"  : [[768 floats],  // list of {n_exemplars} exemplar vectors  │
  │                      [768 floats],  //   each L2-normalised          │
  │                      ...]                                            │
  │    }}                                                                 │
  └───────────────────────────────────────────────────────────────────────┘

  ┌─ Inference formula (V6 — max-over-exemplars) ─────────────────────────┐
  │  query_emb = l2_normalize( backbone(crop) )                          │
  │                                                                       │
  │  for each individual:                                                 │
  │    score = max( dot(query_emb, exemplar) for exemplar in exemplars ) │
  │                                                                       │
  │  best = argmax(score)                                                 │
  │  if score[best] < unknown_threshold → UNKNOWN                        │
  └───────────────────────────────────────────────────────────────────────┘

  Note: "embedding" (centroid) can be used as a fast single-probe fallback.
  The full max-over-exemplars over {n_exemplars} × {n_ind} individuals = {n_exemplars*n_ind} dot products of 768 floats
  per inference — negligible on a modern phone (~0.1 ms).
""")

# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════
hr("DONE")
log(f"""
  TFLite  : {TFLITE_OUT.name}  ({tflite_size_mb:.1f} MB)
  Gallery : {ANDROID_GAL.name}  ({new_gal/1e6:.1f} MB)
  Deployed: {ANDROID_ASSETS}

  YOLO detector : unchanged ✓
  Artefacts     : {"none ✓" if not unexpected_files else str(unexpected_files)}

  NEXT STEPS:
    1. Update Android app to use max-over-exemplars (see FORMAT REFERENCE above)
    2. Android Studio → Build → Clean Project → Rebuild Project
    3. Test on device
""")
