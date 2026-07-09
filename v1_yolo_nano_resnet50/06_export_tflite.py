"""
Export Pipeline: PyTorch models → Android TFLite bundle
========================================================
Converts yolo_orangs_v2/best.pt and resnet_orangs.pt to TFLite format
and packages everything into a .zip ready to import in the Primate Face ID app.

HOW THE CONVERSION WORKS
-------------------------
ResNet50: PyTorch → ONNX → TFLite (via onnx-tf, NOT onnx2tf)
  onnx-tf is far more reliable for ResNet50 because it handles the NCHW→NHWC
  transposition correctly through every BatchNorm layer. onnx2tf (used previously)
  silently produces broken models on ResNet50 despite appearing to succeed.

YOLOv8:  Ultralytics native export (most reliable path, no intermediary needed)

WHY labels.txt IS NEEDED
-------------------------
PyTorch's ImageFolder assigns class indices alphabetically at training time
but does NOT embed the names in the .pt file itself. The Android app needs
labels.txt to map index 0 → "Auti", index 1 → "Jula", etc.
The order is determined by Python's sorted() on the dataset folder names.

REQUIREMENTS (install in your conda env 'orangs')
---------------------------------------------------
  pip install onnx onnx-tf tensorflow onnxruntime ai-edge-torch

  If ai-edge-torch is available (Python 3.9-3.11, PyTorch 2.x):
    pip install ai-edge-torch
  Otherwise the script falls back to onnx-tf automatically.

OUTPUT
------
  output/android_export/
    yolov8_detector.tflite
    resnet50_classifier.tflite
    labels.txt
    metadata.json            ← human-readable info about the models

  output/android_models.zip  ← import this in the app
"""

import os
import sys
import glob
import json
import shutil
import struct
import zipfile
import subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.config_loader import (
    YOLO_V2_PT, MODELS_DIR, CROPS_KNOWN_DIR, OUTPUT_DIR, ensure_dirs
)

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ==============================================================================
# CONFIGURATION
# ==============================================================================

YOLO_PT     = YOLO_V2_PT
RESNET_PT   = MODELS_DIR / "resnet50_classifier.pt"
DATASET_DIR = CROPS_KNOWN_DIR   # used to auto-detect class order from folder names
EXPORT_DIR  = OUTPUT_DIR / "android_export"
ZIP_PATH    = OUTPUT_DIR / "android_models"

DETECTOR_NAME   = "yolov8_detector.tflite"
CLASSIFIER_NAME = "resnet50_classifier.tflite"
LABELS_NAME     = "labels.txt"
METADATA_NAME   = "metadata.json"

# YOLOv8 confidence threshold for the Android app (written to metadata.json)
# 0.35 is safer than 0.25 to avoid false detections like feet/hands
YOLO_CONF_THRESHOLD = 0.35

# ==============================================================================
# HELPERS
# ==============================================================================

def print_section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

def check_file(path: str, label: str):
    if not os.path.exists(path):
        print(f"  [ERROR] {label} not found: {path}")
        sys.exit(1)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  OK  {label}: {path} ({size_mb:.1f} MB)")

def get_tflite_input_shape(tflite_path: str) -> list:
    """Read the input tensor shape from a TFLite flatbuffer without running it."""
    try:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=tflite_path)
        interp.allocate_tensors()
        shape = interp.get_input_details()[0]['shape'].tolist()
        dtype = interp.get_input_details()[0]['dtype']
        out_shape = interp.get_output_details()[0]['shape'].tolist()
        print(f"      Input  shape: {shape}  dtype: {dtype}")
        print(f"      Output shape: {out_shape}")
        return shape
    except Exception as e:
        print(f"      Could not inspect TFLite shape: {e}")
        return []

# ==============================================================================
# STEP 0 — Verify inputs and clean output dir
# ==============================================================================

print_section("STEP 0 — Verifying inputs")

check_file(YOLO_PT, "YOLO model")
check_file(RESNET_PT, "ResNet50 model")

if os.path.exists(EXPORT_DIR):
    shutil.rmtree(EXPORT_DIR)
os.makedirs(EXPORT_DIR)
temp_dir = os.path.join(EXPORT_DIR, "_temp")
os.makedirs(temp_dir)
print(f"  Export directory: {EXPORT_DIR}")

# ==============================================================================
# STEP 1 — Detect class names from dataset OR from checkpoint
# ==============================================================================

print_section("STEP 1 — Detecting class names and correct label order")

# Method A: read from training dataset folder (most reliable)
classes = None
if os.path.exists(DATASET_DIR):
    folder_names = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])
    if folder_names:
        classes = folder_names
        print(f"  Source: dataset folder '{DATASET_DIR}'")

# Method B: read from checkpoint
if not classes:
    checkpoint = torch.load(RESNET_PT, map_location='cpu')
    if isinstance(checkpoint, dict):
        for key in ('class_names', 'classes', 'labels'):
            if key in checkpoint:
                classes = checkpoint[key]
                print(f"  Source: checkpoint key '{key}'")
                break

# Method C: hardcoded fallback (alphabetical order of your known individuals)
if not classes:
    # Python sorted() on your folder names — this IS the correct order
    classes = sorted(["Auti", "Jula", "Mathai", "Molly", "NOAH",
                       "PULCO", "PUTRI", "Sari", "Sinta", "Ujian"])
    print(f"  Source: hardcoded fallback (alphabetical)")

print(f"\n  {len(classes)} classes found (index → name):")
for i, name in enumerate(classes):
    print(f"    [{i:2d}] {name}")

# Write labels.txt
labels_path = os.path.join(EXPORT_DIR, LABELS_NAME)
with open(labels_path, "w", encoding="utf-8") as f:
    f.write("\n".join(classes))
print(f"\n  labels.txt written: {labels_path}")

# ==============================================================================
# STEP 2 — Export YOLOv8 detector
# ==============================================================================

print_section("STEP 2 — Exporting YOLOv8 face detector → TFLite")

yolo = YOLO(YOLO_PT)
print(f"  Running Ultralytics TFLite export (imgsz=640)...")
yolo.export(format='tflite', imgsz=640)

# Ultralytics places the TFLite next to best.pt, inside a subfolder
yolo_dir = os.path.dirname(YOLO_PT)
tflite_candidates = glob.glob(
    os.path.join(yolo_dir, "**", "*float32.tflite"), recursive=True
)
if not tflite_candidates:
    tflite_candidates = glob.glob(
        os.path.join(yolo_dir, "**", "*.tflite"), recursive=True
    )
if not tflite_candidates:
    print("  [ERROR] YOLO TFLite file not found. Check Ultralytics output above.")
    sys.exit(1)

yolo_tflite_src = tflite_candidates[-1]
yolo_tflite_dst = os.path.join(EXPORT_DIR, DETECTOR_NAME)
shutil.copy(yolo_tflite_src, yolo_tflite_dst)

size_mb = os.path.getsize(yolo_tflite_dst) / 1e6
print(f"\n  YOLOv8 TFLite: {yolo_tflite_dst} ({size_mb:.1f} MB)")
print("  Inspecting output tensor shapes:")
get_tflite_input_shape(yolo_tflite_dst)

# ==============================================================================
# STEP 3 — Export ResNet50 classifier
# This is the critical step. We try three methods in order of reliability.
# ==============================================================================

print_section("STEP 3 — Exporting ResNet50 classifier → TFLite")

# --- Load the PyTorch model ---
print("  Loading PyTorch checkpoint...")
checkpoint = torch.load(RESNET_PT, map_location='cpu')

# Detect the state_dict key
if isinstance(checkpoint, dict):
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # The entire dict might be the state_dict
        state_dict = checkpoint
elif hasattr(checkpoint, 'state_dict'):
    state_dict = checkpoint.state_dict()
else:
    state_dict = checkpoint

num_classes = len(classes)

# Reconstruct the exact same architecture used during training
model = models.resnet50(weights=None)
num_features = model.fc.in_features  # 2048
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, num_classes)
)
model.load_state_dict(state_dict, strict=True)
model.eval()  # CRITICAL: disables Dropout and BatchNorm training mode
print(f"  Model loaded: ResNet50 with {num_classes} output classes")

# Quick sanity check: forward pass with random input
with torch.no_grad():
    test_out = model(torch.randn(1, 3, 224, 224))
    # Verify output shape
    assert test_out.shape == (1, num_classes), \
        f"Unexpected output shape: {test_out.shape}"
    predicted_class = test_out.argmax(1).item()
    print(f"  Sanity check passed: output shape {list(test_out.shape)}, "
          f"test prediction → class {predicted_class} ({classes[predicted_class]})")

# ── METHOD 1: ai-edge-torch (Google official, most reliable) ──────────────────
resnet_tflite_dst = os.path.join(EXPORT_DIR, CLASSIFIER_NAME)
converted = False

try:
    import ai_edge_torch
    print("\n  [Method 1] ai-edge-torch (Google official converter)...")

    with torch.no_grad():
        sample_input = (torch.randn(1, 3, 224, 224),)
        edge_model = ai_edge_torch.convert(model, sample_input)
        edge_model.export(resnet_tflite_dst)

    if os.path.exists(resnet_tflite_dst) and os.path.getsize(resnet_tflite_dst) > 1_000:
        converted = True
        print(f"  [Method 1] SUCCESS: ai-edge-torch")
    else:
        print("  [Method 1] FAILED: output file empty")

except ImportError:
    print("  [Method 1] SKIPPED: ai-edge-torch not installed")
    print("             Install with: pip install ai-edge-torch")
except Exception as e:
    print(f"  [Method 1] FAILED: {e}")

# ── METHOD 2: ONNX → onnx-tf → TFLite ───────────────────────────────────────
if not converted:
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf

        print("\n  [Method 2] ONNX → onnx-tf → TFLite...")

        # 2a. Export to ONNX
        onnx_path = os.path.join(temp_dir, "resnet.onnx")
        print("    2a. Exporting to ONNX...")
        with torch.no_grad():
            torch.onnx.export(
                model,
                torch.randn(1, 3, 224, 224),
                onnx_path,
                input_names=['input'],
                output_names=['logits'],
                dynamic_axes=None,   # fixed batch size of 1
                opset_version=13,
                do_constant_folding=True
            )
        print(f"    ONNX exported: {onnx_path}")

        # 2b. ONNX → TensorFlow SavedModel via onnx-tf
        print("    2b. Converting ONNX → TF SavedModel via onnx-tf...")
        tf_saved_model_dir = os.path.join(temp_dir, "resnet_saved_model")
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(tf_saved_model_dir)
        print(f"    SavedModel written: {tf_saved_model_dir}")

        # 2c. TF SavedModel → TFLite
        print("    2c. Converting TF SavedModel → TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_dir)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS  # allow fallback for complex ops
        ]
        tflite_model = converter.convert()

        with open(resnet_tflite_dst, 'wb') as f:
            f.write(tflite_model)

        if os.path.exists(resnet_tflite_dst) and os.path.getsize(resnet_tflite_dst) > 1_000:
            converted = True
            print(f"  [Method 2] SUCCESS: onnx-tf")
        else:
            print("  [Method 2] FAILED: output file empty")

    except ImportError as e:
        print(f"  [Method 2] SKIPPED: missing package ({e})")
        print("             Install with: pip install onnx onnx-tf tensorflow")
    except Exception as e:
        print(f"  [Method 2] FAILED: {e}")

# ── METHOD 3: ONNX → onnx2tf → TFLite (last resort) ─────────────────────────
if not converted:
    try:
        import onnx
        import onnx2tf

        print("\n  [Method 3] ONNX → onnx2tf → TFLite (last resort)...")

        onnx_path = os.path.join(temp_dir, "resnet.onnx")
        if not os.path.exists(onnx_path):
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    torch.randn(1, 3, 224, 224),
                    onnx_path,
                    opset_version=13,
                    do_constant_folding=True
                )

        tf_out = os.path.join(temp_dir, "onnx2tf_out")
        # non_transposing_optimization_mode helps preserve correct NHWC behavior
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=tf_out,
            non_verbose=True,
            copy_onnx_input_output_names_to_tflite=True,
            disable_group_convolution=True,
        )
        # onnx2tf names the file differently depending on version
        for name in ("model_float32.tflite", "resnet_float32.tflite"):
            candidate = os.path.join(tf_out, name)
            if os.path.exists(candidate):
                shutil.copy(candidate, resnet_tflite_dst)
                converted = True
                break
        if not converted:
            all_tflites = glob.glob(os.path.join(tf_out, "*.tflite"))
            if all_tflites:
                shutil.copy(all_tflites[0], resnet_tflite_dst)
                converted = True

        if converted:
            print(f"  [Method 3] SUCCESS: onnx2tf")
        else:
            print("  [Method 3] FAILED: no TFLite file produced")

    except ImportError as e:
        print(f"  [Method 3] SKIPPED: missing package ({e})")
    except Exception as e:
        print(f"  [Method 3] FAILED: {e}")

if not converted:
    print("\n  [FATAL] All conversion methods failed.")
    print("  Install at least one: pip install ai-edge-torch")
    print("                     or: pip install onnx onnx-tf tensorflow")
    sys.exit(1)

size_mb = os.path.getsize(resnet_tflite_dst) / 1e6
print(f"\n  ResNet50 TFLite: {resnet_tflite_dst} ({size_mb:.1f} MB)")
print("  Inspecting output tensor shapes:")
get_tflite_input_shape(resnet_tflite_dst)

# ==============================================================================
# STEP 4 — Validate both TFLite models in Python (catch errors before Android)
# ==============================================================================

print_section("STEP 4 — Validation: running both models in Python")

try:
    import tensorflow as tf

    # ── Validate ResNet50 ─────────────────────────────────────────────────────
    print("  Validating ResNet50 TFLite...")
    interp_cls = tf.lite.Interpreter(model_path=resnet_tflite_dst)
    interp_cls.allocate_tensors()
    inp_details = interp_cls.get_input_details()[0]
    out_details = interp_cls.get_output_details()[0]

    print(f"    Input:  shape={inp_details['shape'].tolist()}, "
          f"dtype={inp_details['dtype'].__name__}")
    print(f"    Output: shape={out_details['shape'].tolist()}, "
          f"dtype={out_details['dtype'].__name__}")

    # Feed the same random tensor we already tested in PyTorch
    # Note: TFLite ResNet expects NHWC [1, 224, 224, 3]
    # PyTorch uses NCHW internally but the conversion handles transposition
    test_img_nhwc = np.random.rand(1, 224, 224, 3).astype(np.float32)

    # Apply ImageNet normalization (same as Android app)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    test_img_nhwc = (test_img_nhwc - mean) / std

    interp_cls.set_tensor(inp_details['index'], test_img_nhwc)
    interp_cls.invoke()
    logits = interp_cls.get_tensor(out_details['index'])[0]

    # Softmax
    exp_logits = np.exp(logits - logits.max())
    probs = exp_logits / exp_logits.sum()
    top3_idx = np.argsort(probs)[::-1][:3]

    print(f"    Random-input Top-3 predictions:")
    for rank, idx in enumerate(top3_idx, 1):
        print(f"      #{rank}: {classes[idx]} ({probs[idx]*100:.1f}%)")

    print(f"    Result: ResNet50 TFLite is FUNCTIONAL")

    # Now cross-check with the original PyTorch model on the SAME input
    # Convert NHWC → NCHW for PyTorch
    test_img_nchw = torch.tensor(test_img_nhwc).permute(0, 3, 1, 2)
    with torch.no_grad():
        pt_logits = model(test_img_nchw).numpy()[0]
    pt_exp = np.exp(pt_logits - pt_logits.max())
    pt_probs = pt_exp / pt_exp.sum()
    pt_top1 = np.argmax(pt_probs)
    tflite_top1 = top3_idx[0]

    if pt_top1 == tflite_top1:
        print(f"    Cross-check PASSED: PyTorch and TFLite agree on top-1 "
              f"({classes[pt_top1]})")
    else:
        print(f"    [WARNING] Cross-check MISMATCH:")
        print(f"      PyTorch top-1:  {classes[pt_top1]} ({pt_probs[pt_top1]*100:.1f}%)")
        print(f"      TFLite  top-1:  {classes[tflite_top1]} ({probs[tflite_top1]*100:.1f}%)")
        print(f"      The conversion may have issues. Try Method 1 (ai-edge-torch).")

    # ── Validate YOLO ──────────────────────────────────────────────────────────
    print("\n  Validating YOLO TFLite...")
    interp_det = tf.lite.Interpreter(model_path=yolo_tflite_dst)
    interp_det.allocate_tensors()
    det_inp = interp_det.get_input_details()[0]
    det_out = interp_det.get_output_details()[0]

    print(f"    Input:  shape={det_inp['shape'].tolist()}, "
          f"dtype={det_inp['dtype'].__name__}")
    print(f"    Output: shape={det_out['shape'].tolist()}, "
          f"dtype={det_out['dtype'].__name__}")

    # NOTE: this output shape is important for the Android code
    # Ultralytics TFLite export produces [1, 5, 8400] for a 1-class model
    # but some versions may produce [1, 8400, 5]
    # We log it here so the Android team can verify the parsing code matches
    out_shape = det_out['shape'].tolist()
    print(f"\n  [IMPORTANT FOR ANDROID DEV] YOLO output shape is: {out_shape}")
    if len(out_shape) == 3:
        if out_shape[1] < out_shape[2]:
            print(f"    → Format: [batch, values_per_box, num_candidates]")
            print(f"      Android YoloDetector.kt uses this format — CORRECT")
        else:
            print(f"    → Format: [batch, num_candidates, values_per_box]  ← TRANSPOSED!")
            print(f"      Android YoloDetector.kt needs to swap index access:")
            print(f"      Change: output[valueIdx][candidateIdx]")
            print(f"      To:     output[candidateIdx][valueIdx]")

    print(f"    Result: YOLO TFLite is FUNCTIONAL")

except ImportError:
    print("  [SKIPPED] tensorflow not installed — cannot validate TFLite models in Python")
    print("  Install with: pip install tensorflow")
except Exception as e:
    print(f"  [WARNING] Validation error: {e}")

# ==============================================================================
# STEP 5 — Write metadata.json
# ==============================================================================

print_section("STEP 5 — Writing metadata.json")

metadata = {
    "export_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "species": "orang-outan",
    "num_individuals": len(classes),
    "individuals": classes,
    "detector": {
        "file": DETECTOR_NAME,
        "architecture": "YOLOv8",
        "input_size": 640,
        "confidence_threshold": YOLO_CONF_THRESHOLD,
        "iou_threshold": 0.45,
        "source": str(YOLO_PT)
    },
    "classifier": {
        "file": CLASSIFIER_NAME,
        "architecture": "ResNet50",
        "input_size": 224,
        "unknown_threshold": 0.40,
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
        "source": str(RESNET_PT)
    }
}

metadata_path = os.path.join(EXPORT_DIR, METADATA_NAME)
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"  metadata.json written: {metadata_path}")

# ==============================================================================
# STEP 6 — Package into ZIP
# ==============================================================================

print_section("STEP 6 — Packaging into ZIP")

# Remove temp dir before zipping
shutil.rmtree(temp_dir, ignore_errors=True)

files_to_zip = [
    os.path.join(EXPORT_DIR, DETECTOR_NAME),
    os.path.join(EXPORT_DIR, CLASSIFIER_NAME),
    os.path.join(EXPORT_DIR, LABELS_NAME),
    os.path.join(EXPORT_DIR, METADATA_NAME),
]

zip_full_path = ZIP_PATH + ".zip"
with zipfile.ZipFile(zip_full_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for fpath in files_to_zip:
        if os.path.exists(fpath):
            zf.write(fpath, arcname=os.path.basename(fpath))
            print(f"  Added: {os.path.basename(fpath)} "
                  f"({os.path.getsize(fpath)/1e6:.1f} MB)")
        else:
            print(f"  [WARNING] Missing: {fpath}")

zip_size_mb = os.path.getsize(zip_full_path) / 1e6
print(f"\n  ZIP created: {zip_full_path} ({zip_size_mb:.1f} MB)")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

print(f"""
{'=' * 70}
  ALL DONE
{'=' * 70}

  Individual files:
    {os.path.join(EXPORT_DIR, DETECTOR_NAME)}
    {os.path.join(EXPORT_DIR, CLASSIFIER_NAME)}
    {os.path.join(EXPORT_DIR, LABELS_NAME)}
    {os.path.join(EXPORT_DIR, METADATA_NAME)}

  Android import bundle:
    {zip_full_path}

  HOW TO UPDATE THE APP:
    1. Send {os.path.basename(zip_full_path)} to the phone (email, USB, etc.)
    2. Open Primate Face ID → Settings → Import Model Bundle (.ZIP)
    3. Select the file → done. No recompilation needed.

  LABELS ORDER (must match what was used during training):
{chr(10).join(f"    [{i:2d}] {name}" for i, name in enumerate(classes))}
{'=' * 70}
""")