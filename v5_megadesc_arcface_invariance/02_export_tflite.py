# export_backbone_wsl.py
# Converts MegaDescriptor backbone to TFLite using litert-torch (Linux only).
# Run this from WSL2 with the orangs_export conda env.
#
# RUN (in PowerShell):
#   wsl -d Ubuntu -- bash -c "/root/miniconda3/bin/conda run -n orangs_export --no-capture-output python <repo>/v5_megadesc_arcface_invariance/02_export_tflite.py"

import sys
from pathlib import Path
from datetime import datetime

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Portable paths (works from WSL too: Path(__file__) resolves under /mnt/<drive>) ──
REPO        = Path(__file__).resolve().parents[1]
BACKBONE_PT = REPO / "output" / "v5" / "v5_backbone_only.pt"
FULL_PT     = REPO / "output" / "v5" / "v5_best.pt"
# The .tflite is written to output/, then copied by hand into the Android app repo
# (app/src/main/assets/megadesc_v6_backbone.tflite). The app repo is separate.
OUT_PATH    = REPO / "output" / "v5" / "v5_backbone.tflite"

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Imports ─────────────────────────────────────────────────────────────────
log("Importing torch / timm...")
import torch
import timm

log(f"torch: {torch.__version__}")

# Try litert_torch first (new name), fall back to ai_edge_torch (old name)
converter = None
try:
    import litert_torch
    converter = litert_torch
    log("Using litert_torch (new package name)")
except ImportError:
    pass

if converter is None:
    try:
        import ai_edge_torch
        if hasattr(ai_edge_torch, 'convert'):
            converter = ai_edge_torch
            log("Using ai_edge_torch (legacy)")
        else:
            log("[ERR] ai_edge_torch imported but has no 'convert' attribute.")
            log("      The package was renamed. Installing litert-torch...")
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "litert-torch", "--quiet"], check=True)
            import litert_torch
            converter = litert_torch
            log("litert_torch installed and imported OK")
    except ImportError:
        log("[ERR] Neither litert_torch nor ai_edge_torch found.")
        log("      Run: pip install litert-torch")
        sys.exit(1)

if not hasattr(converter, 'convert'):
    log(f"[ERR] converter module has no 'convert'. Available: {[x for x in dir(converter) if not x.startswith('_')]}")
    sys.exit(1)

# ── Load backbone ────────────────────────────────────────────────────────────
log("Creating MegaDescriptor-T-224 architecture...")
backbone = timm.create_model(
    "hf-hub:BVRA/MegaDescriptor-T-224",
    pretrained=False,
    num_classes=0,
)

if BACKBONE_PT.exists():
    log(f"Loading {BACKBONE_PT.name} ({BACKBONE_PT.stat().st_size/1e6:.1f} MB)...")
    state = torch.load(str(BACKBONE_PT), map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for key in ("backbone_state", "model_state_dict", "state_dict"):
            if key in state:
                backbone.load_state_dict(state[key])
                log(f"Loaded weights from key '{key}'")
                break
        else:
            backbone.load_state_dict(state)
            log("Loaded weights directly")
    else:
        backbone = state
elif FULL_PT.exists():
    log(f"backbone_only.pt not found — loading from full checkpoint...")
    state = torch.load(str(FULL_PT), map_location="cpu", weights_only=False)
    if "backbone_state" not in state:
        log(f"[ERR] 'backbone_state' key not found. Keys: {list(state.keys())}")
        sys.exit(1)
    backbone.load_state_dict(state["backbone_state"])
    log("Loaded from 'backbone_state'")
else:
    log(f"[ERR] No backbone found at {BACKBONE_PT}")
    sys.exit(1)

backbone = backbone.eval().cpu()

with torch.no_grad():
    dummy = torch.randn(1, 3, 224, 224)
    out   = backbone(dummy)
    log(f"Backbone output shape: {out.shape}  (expected: [1, 768])")

# ── Convert ──────────────────────────────────────────────────────────────────
log("")
log("Converting to TFLite (float32) — 2-5 min on CPU...")

sample_inputs = (dummy,)

try:
    edge_model = converter.convert(backbone, sample_inputs)
    log("Conversion complete — saving...")
    edge_model.export(str(OUT_PATH))
    log(f"Saved : {OUT_PATH}")
    log(f"Size  : {OUT_PATH.stat().st_size / 1e6:.1f} MB")
except Exception as e:
    log(f"[ERR] Conversion failed: {e}")
    sys.exit(1)

# ── Validate ─────────────────────────────────────────────────────────────────
with open(OUT_PATH, "rb") as f:
    f.seek(4); magic = f.read(4)

if magic == b"TFL3":
    log("TFLite magic bytes OK — file is valid.")
    log("")
    log("Done. Rebuild the Android app in Android Studio.")
else:
    log(f"[WARN] Unexpected magic bytes: {magic} — file may be corrupt.")
    sys.exit(1)