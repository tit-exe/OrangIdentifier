"""
export_to_tflite.py  --  OrangIdentifier maintenance
====================================================
OPTION B, final step: turn the new brain (a .pt file) into the phone file
(a .tflite file), and copy both the brain and the gallery into the app.

You only need this after retraining with train_brain.py. If you used Option A
(add individuals with build_gallery.py), you do NOT need this script, because
the brain did not change.

IMPORTANT: this script MUST run inside WSL (a small Linux inside Windows). The
tool that makes the .tflite file only exists on Linux. The first-time WSL setup
is explained in 00_first_time_setup. Once WSL is ready, run this from PowerShell:

  wsl -d Ubuntu -- bash -c "/root/miniconda3/bin/conda run -n orangs_export --no-capture-output python /mnt/d/<path-to-your-repo>/maintenance/scripts/export_to_tflite.py"

Files written:
  new_brain/models/new_backbone.tflite            (the new phone brain)
  app/.../assets/megadesc_v6_backbone.tflite      (replaced with the new brain)
  app/.../assets/gallery.json                     (replaced with the new gallery)
  app/.../assets/yolo_v2_detector.tflite          (NOT touched)

Note: the app looks for a file named "megadesc_v6_backbone.tflite", so we keep
that exact name even though the brain inside is your new one.
"""

import sys, shutil, json
from pathlib import Path
from datetime import datetime
REPO = Path(__file__).resolve().parents[2]  # repository root (portable)

# ---- Logging -----------------------------------------------------------------
def log(msg="", level=""):
    tag = f"[{level}] " if level else ""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {tag}{msg}", flush=True)

def hr(title=""):
    bar = "-" * 68
    log()
    log(bar)
    if title:
        log(f"  {title}")
        log(bar)

# ---- Paths (inside WSL, the D: drive is seen as /mnt/d) -----------------------
# These point at the new brain produced by train_brain.py.
BACKBONE_PT  = (REPO / "maintenance" / "new_brain" / "models" / "new_backbone_only.pt")
GALLERY_SRC  = (REPO / "maintenance" / "new_brain" / "models" / "new_gallery.json")
TFLITE_OUT   = (REPO / "maintenance" / "new_brain" / "models" / "new_backbone.tflite")

# The Android app is a SEPARATE repository (OrangIdentifier-Android).
# Set this to the assets folder of your local copy of the app repo,
# or leave it as output/ and copy the two files by hand afterwards.
ANDROID_ASSETS = REPO / "maintenance" / "new_brain" / "app_assets"
ANDROID_MODEL  = ANDROID_ASSETS / "megadesc_v6_backbone.tflite"
ANDROID_GAL    = ANDROID_ASSETS / "gallery.json"
ANDROID_YOLO   = ANDROID_ASSETS / "yolo_v2_detector.tflite"  # DO NOT TOUCH

IMG_SIZE = 224

# ==============================================================================
# 1. PRE-FLIGHT
# ==============================================================================
hr("Pre-flight checks")

errors = []
for p, label in [(BACKBONE_PT, "new_backbone_only.pt"),
                 (GALLERY_SRC, "new_gallery.json"),
                 (ANDROID_ASSETS, "android assets dir")]:
    if p.exists():
        sz = p.stat().st_size if p.is_file() else 0
        log(f"  [ok]  {label:<35} {sz/1e6:6.1f} MB" if sz else f"  [ok]  {label}/")
    else:
        log(f"  [MISSING]  NOT FOUND: {p}", "ERR")
        errors.append(str(p))

if ANDROID_YOLO.exists():
    log(f"  [ok]  yolo_v2_detector.tflite   (will NOT be touched)")
else:
    log(f"  [WARNING]  yolo_v2_detector.tflite not found in assets", "WARN")

if errors:
    log(f"Stopping, missing files: {errors}", "ERR")
    log("If new_backbone_only.pt or new_gallery.json is missing, run train_brain.py first.")
    sys.exit(1)

# ==============================================================================
# 2. LOAD THE NEW BRAIN
# ==============================================================================
hr("Loading the new brain")

# -- Dependency guard: on a missing package, show the exact install command -----
import importlib.util as _ilu
_missing = [m for m in ("torch", "timm") if _ilu.find_spec(m) is None]
if _missing:
    log("Missing Python package(s): " + ", ".join(_missing), "STOP")
    log("This script runs inside WSL, in the 'orangs_export' environment.")
    log("Easiest fix: re-run maintenance/00_first_time_setup/setup_wsl.ps1")
    log("Or install by hand inside WSL:")
    log("  conda run -n orangs_export pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
    log("  conda run -n orangs_export pip install timm litert-torch pillow numpy huggingface_hub")
    sys.exit(1)

import torch
import timm

log(f"  torch   : {torch.__version__}")
log(f"  timm    : {timm.__version__}")

# MegaDescriptor-T-224 is the architecture. num_classes=0 gives 768-dim vectors.
# Try the HuggingFace hub first (usually cached), otherwise use the plain timm
# name so the script still works without internet.
log("  Building the architecture (MegaDescriptor-T-224)...")
try:
    bb = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224", pretrained=False, num_classes=0)
    log("  Architecture loaded via hf-hub:BVRA/MegaDescriptor-T-224")
except Exception as e_hf:
    log(f"  HF hub failed ({e_hf}), using swin_tiny_patch4_window7_224 instead")
    bb = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=0)
    log("  Architecture loaded via timm (swin_tiny_patch4_window7_224)")

log(f"\n  Loading weights from {BACKBONE_PT.name} "
    f"({BACKBONE_PT.stat().st_size/1e6:.0f} MB)...")
ck = torch.load(str(BACKBONE_PT), map_location="cpu", weights_only=False)

log(f"  Checkpoint keys : {list(ck.keys())}")

state = ck["backbone_state"]   # the key used by train_brain.py
missing, unexpected = bb.load_state_dict(state, strict=True)
# strict=True so a mismatch shows up straight away.
log(f"  Weights loaded  : {len(missing)} missing, {len(unexpected)} unexpected")

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
    log(f"Wrong output shape {out.shape}, stopping", "ERR")
    sys.exit(1)

# Print normalization from checkpoint so it can be documented
norm = ck.get("normalization", {})
log(f"  Normalization   : mean={norm.get('mean')}  std={norm.get('std')}")
log(f"  Embedding dim   : {ck.get('emb_dim', 768)}")

# ==============================================================================
# 3. LOAD CONVERTER
# ==============================================================================
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
            log(f"  {pkg_name} found but has no .convert(), skipping")
    except ImportError:
        log(f"  {pkg_name} not installed")

if converter is None:
    log("No converter found. Install: pip install litert-torch", "ERR")
    sys.exit(1)

# ==============================================================================
# 4. CONVERT TO TFLITE
# ==============================================================================
hr("Converting to TFLite (float32)")
log("  Input : NCHW float32 [1, 3, 224, 224]")
log("  Output: float32 [1, 768]  (L2-normalize in the app after inference)")
log("  This may take 3 to 8 minutes on CPU...")
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

# Check the file really is a TFLite file (it starts with "TFL3" at byte 4).
with open(TFLITE_OUT, "rb") as f:
    f.seek(4)
    magic = f.read(4)
if magic == b"TFL3":
    log("  Magic : TFL3 [ok]  (valid TFLite file)")
else:
    log(f"  Magic : {magic!r}  [WARNING] unexpected, file may be corrupt", "WARN")

# ==============================================================================
# 5. DEPLOY TO ANDROID ASSETS
# ==============================================================================
hr("Deploying to Android assets")

# Snapshot before
before = {p.name: p.stat().st_size for p in ANDROID_ASSETS.iterdir() if p.is_file()}
log("  Before:")
for name, sz in sorted(before.items()):
    log(f"    {name:<52} {sz/1e6:6.1f} MB")

log()

# Replace the brain
old_bb = ANDROID_MODEL.stat().st_size if ANDROID_MODEL.exists() else 0
shutil.copy2(str(TFLITE_OUT), str(ANDROID_MODEL))
new_bb = ANDROID_MODEL.stat().st_size
log(f"  Brain    : {old_bb/1e6:.1f} MB -> {new_bb/1e6:.1f} MB")
log(f"    {ANDROID_MODEL}")

# Replace the gallery
old_gal = ANDROID_GAL.stat().st_size if ANDROID_GAL.exists() else 0
shutil.copy2(str(GALLERY_SRC), str(ANDROID_GAL))
new_gal = ANDROID_GAL.stat().st_size
log(f"  Gallery  : {old_gal/1e6:.1f} MB -> {new_gal/1e6:.1f} MB")
log(f"    {ANDROID_GAL}")

# Check no extra files ended up in the assets folder.
after = {p.name: p.stat().st_size for p in ANDROID_ASSETS.iterdir() if p.is_file()}
expected = {"megadesc_v6_backbone.tflite", "gallery.json", "yolo_v2_detector.tflite"}
unexpected_files = set(after.keys()) - expected

log()
if unexpected_files:
    log(f"  [WARNING] Unexpected files in assets: {unexpected_files}", "WARN")
    log("     Delete them by hand to keep the assets folder clean.", "WARN")
else:
    log("  Assets folder clean, no unexpected files.")

log()
log("  After:")
for name, sz in sorted(after.items()):
    tag = "  <- UPDATED" if name in {ANDROID_MODEL.name, ANDROID_GAL.name} else ""
    log(f"    {name:<52} {sz/1e6:6.1f} MB{tag}")

# ==============================================================================
# 6. FORMAT REFERENCE  (only useful if a developer changes the app code)
# ==============================================================================
hr("FORMAT REFERENCE (for a developer, not needed for a normal update)")

# Read the gallery to print accurate numbers.
with open(GALLERY_SRC, "r", encoding="utf-8") as f:
    gal = json.load(f)

n_ind      = gal["num_individuals"]
emb_dim    = gal["embedding_dim"]
threshold  = gal["unknown_threshold"]
norm_name  = gal["normalization"]
n_exemplars = gal["individuals"][next(iter(gal["individuals"]))]["num_exemplars"]

log(f"""
  TFLite model (megadesc_v6_backbone.tflite)
    Input  : float32 NCHW  [1, 3, 224, 224]
    Output : float32       [1, 768]  (raw, NOT L2-normalized)
    The app must L2-normalize the output before scoring.

  Preprocessing (must match training)
    Resize the image to 224 x 224
    pixel_float = pixel / 255.0
    pixel_norm  = (pixel_float - 0.5) / 0.5
    Same for the R, G, B channels (mean=0.5, std=0.5)
    normalization key in gallery.json: "{norm_name}"

  Gallery (gallery.json)
    individuals       : {n_ind}
    embedding_dim     : {emb_dim}
    unknown_threshold : {threshold}  (cosine similarity)
    Each individual has:
      class_index   : a number
      num_exemplars : {n_exemplars}
      embedding     : one average vector of 768 numbers
      exemplars     : a list of {n_exemplars} vectors of 768 numbers each

  How the app decides who it is (max over exemplars)
    query = l2_normalize( brain(crop) )
    for each individual:
      score = max( dot(query, exemplar) for each exemplar )
    best = the individual with the highest score
    if that score is below unknown_threshold, the answer is UNKNOWN
""")

# ==============================================================================
# DONE
# ==============================================================================
hr("DONE")
log(f"""
  New brain (phone) : {ANDROID_MODEL.name}  ({new_bb/1e6:.1f} MB)
  New gallery       : {ANDROID_GAL.name}  ({new_gal/1e6:.1f} MB)
  Copied into       : {ANDROID_ASSETS}

  Head detector : unchanged
  Extra files   : {"none" if not unexpected_files else str(unexpected_files)}

  NEXT STEP:
    Open the app in Android Studio, then Build > Clean Project > Rebuild Project,
    and test it on a phone.
""")
