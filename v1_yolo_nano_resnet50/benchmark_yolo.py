# =============================================================================
# benchmark_yolo.py
# Compare YOLO v1 et v2 sur le val set de v2.
# Genere des graphiques, des stats et des exemples visuels.
# N'ecrase rien, ne modifie pas les crops existants.
#
# Usage : python scripts/benchmark_yolo.py
# =============================================================================

import sys
import json
import time
import random
import shutil
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    BASE_DIR, MODELS_DIR, RESULTS_DIR,
    DATASET_CLASSIF_DIR
)

# =============================================================================
# CONFIGURATION
# =============================================================================

YOLO_V1      = BASE_DIR / "runs" / "orang_face_detector"    / "weights" / "best.pt"
YOLO_V2      = BASE_DIR / "runs" / "orang_face_detector_v2" / "weights" / "best.pt"
VAL_DIR      = BASE_DIR / "DATASET_YOLO_V2" / "val"  # val set de v2 avec labels corriges
BENCH_DIR    = RESULTS_DIR / "benchmark_yolo"
CONF_SEUIL   = 0.25
IOU_SEUIL    = 0.50
N_EXEMPLES   = 12   # nombre d'images d'exemple affichees dans le rapport visuel

RANDOM_SEED  = 42
random.seed(RANDOM_SEED)

# =============================================================================
# CHARGEMENT DES MODELES
# =============================================================================

def charger_modeles():
    from ultralytics import YOLO

    modeles = {}

    if YOLO_V1.exists():
        print(f"  Chargement YOLO v1 : {YOLO_V1}")
        modeles['v1'] = YOLO(str(YOLO_V1))
    else:
        print(f"  AVERTISSEMENT : YOLO v1 introuvable ({YOLO_V1})")

    if YOLO_V2.exists():
        print(f"  Chargement YOLO v2 : {YOLO_V2}")
        modeles['v2'] = YOLO(str(YOLO_V2))
    else:
        print(f"  ERREUR : YOLO v2 introuvable ({YOLO_V2})")
        print("  L'entrainement est-il termine ?")
        sys.exit(1)

    return modeles

# =============================================================================
# LECTURE DES LABELS GROUND TRUTH
# =============================================================================

def lire_label(label_path, img_w, img_h):
    """Retourne une liste de [x1,y1,x2,y2] en pixels depuis un label YOLO."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            _, xc, yc, w, h = map(float, parts)
            x1 = int((xc - w/2) * img_w)
            y1 = int((yc - h/2) * img_h)
            x2 = int((xc + w/2) * img_w)
            y2 = int((yc + h/2) * img_h)
            boxes.append([x1, y1, x2, y2])
    return boxes

# =============================================================================
# CALCUL DE L'IOU
# =============================================================================

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter)

# =============================================================================
# EVALUATION SUR LE VAL SET
# =============================================================================

def evaluer(model, images, version):
    """
    Evalue un modele sur la liste d'images du val set.
    Retourne un dict de metriques et les resultats par image.
    """
    print(f"\n  Evaluation YOLO {version} sur {len(images)} images...")

    tp_total = 0   # vrais positifs
    fp_total = 0   # faux positifs
    fn_total = 0   # faux negatifs
    temps_inference = []
    resultats = []  # pour les exemples visuels

    for i, img_path in enumerate(images):
        pct  = (i + 1) / len(images)
        fill = int(40 * pct)
        print(f"\r    [{('='*fill)+('-'*(40-fill))}] {i+1}/{len(images)}",
              end="", flush=True)

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Ground truth
        lbl_path = VAL_DIR / "labels" / (img_path.stem + ".txt")
        gt_boxes = lire_label(lbl_path, w, h)

        # Inference
        t0 = time.perf_counter()
        results = model.predict(
            source=str(img_path),
            conf=CONF_SEUIL,
            verbose=False,
            device=0
        )
        t1 = time.perf_counter()
        temps_inference.append((t1 - t0) * 1000)

        pred_boxes = []
        pred_confs = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                pred_boxes.append([x1, y1, x2, y2])
                pred_confs.append(float(box.conf[0]))

        # Calcul TP/FP/FN avec IOU
        matched_gt = set()
        matched_pred = set()

        for pi, pb in enumerate(pred_boxes):
            best_iou = 0
            best_gi  = -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                score = iou(pb, gb)
                if score > best_iou:
                    best_iou = score
                    best_gi  = gi
            if best_iou >= IOU_SEUIL:
                tp_total += 1
                matched_gt.add(best_gi)
                matched_pred.add(pi)
            else:
                fp_total += 1

        fn_total += len(gt_boxes) - len(matched_gt)

        resultats.append({
            "img_path":   img_path,
            "gt_boxes":   gt_boxes,
            "pred_boxes": pred_boxes,
            "pred_confs": pred_confs,
            "tp": len(matched_pred),
            "fp": fp_total,
            "fn": len(gt_boxes) - len(matched_gt),
            "temps_ms": temps_inference[-1],
        })

    print()

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall    = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    moy_temps = np.mean(temps_inference)
    fps       = 1000 / moy_temps if moy_temps > 0 else 0

    metriques = {
        "version":   version,
        "tp":        tp_total,
        "fp":        fp_total,
        "fn":        fn_total,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "temps_ms":  round(moy_temps, 2),
        "fps":       round(fps, 1),
        "n_images":  len(images),
    }

    return metriques, resultats

# =============================================================================
# LECTURE DU mAP DEPUIS results.csv
# =============================================================================

def lire_map_depuis_csv(version):
    """Lit le mAP50 officiel Ultralytics depuis le fichier results.csv."""
    run_name = "orang_face_detector_v2" if version == "v2" else "orang_face_detector"
    csv_path = BASE_DIR / "runs" / run_name / "results.csv"
    if not csv_path.exists():
        return None
    import csv
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return None
    best = max(rows, key=lambda r: float(r.get('metrics/mAP50(B)', 0) or 0))
    return {
        "map50":     float(best.get('metrics/mAP50(B)', 0)),
        "map50_95":  float(best.get('metrics/mAP50-95(B)', 0)),
        "precision": float(best.get('metrics/precision(B)', 0)),
        "recall":    float(best.get('metrics/recall(B)', 0)),
        "epoch":     best.get('                  epoch', '?').strip(),
        "n_epochs":  len(rows),
    }

# =============================================================================
# GRAPHIQUES
# =============================================================================

def generer_graphiques(m_v1, m_v2, map_v1, map_v2):
    print("\n  Generation des graphiques...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Comparaison YOLO v1 vs v2 — Orangs-outangs CNRS IPHC",
                 fontsize=14, fontweight='bold')

    versions = ['YOLO v1\n(520 imgs annotées)', 'YOLO v2\n(1986 imgs corrigées)']
    couleurs = ['#4499ff', '#00ff88']

    # --- Graphique 1 : Precision / Recall / F1 ---
    ax = axes[0]
    metriques_noms = ['Precision', 'Recall', 'F1']
    vals_v1 = [m_v1['precision'], m_v1['recall'], m_v1['f1']]
    vals_v2 = [m_v2['precision'], m_v2['recall'], m_v2['f1']]

    x = np.arange(len(metriques_noms))
    w = 0.35
    bars1 = ax.bar(x - w/2, vals_v1, w, label='YOLO v1', color='#4499ff', alpha=0.85)
    bars2 = ax.bar(x + w/2, vals_v2, w, label='YOLO v2', color='#00ff88', alpha=0.85)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylim(0, 1.15)
    ax.set_xticks(x); ax.set_xticklabels(metriques_noms)
    ax.set_title('Precision / Recall / F1\n(val set v2)')
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    # --- Graphique 2 : mAP50 officiel (depuis CSV Ultralytics) ---
    ax = axes[1]
    maps = []
    labels_map = []
    if map_v1:
        maps.append(map_v1['map50'])
        labels_map.append(f"v1\n({map_v1['n_epochs']} epochs)")
    if map_v2:
        maps.append(map_v2['map50'])
        labels_map.append(f"v2\n({map_v2['n_epochs']} epochs)")

    bars = ax.bar(labels_map, maps,
                  color=couleurs[:len(maps)], alpha=0.85, width=0.4)
    for bar, val in zip(bars, maps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='Objectif 0.90')
    ax.axhline(y=0.95, color='red',    linestyle='--', alpha=0.7, label='Excellent 0.95')
    ax.set_ylim(0, 1.05)
    ax.set_title('mAP@0.5 officiel\n(val set Ultralytics)')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    # --- Graphique 3 : Vitesse d'inference ---
    ax = axes[2]
    vitesses = [m_v1['fps'], m_v2['fps']]
    temps    = [m_v1['temps_ms'], m_v2['temps_ms']]

    bars = ax.bar(versions, vitesses, color=couleurs, alpha=0.85, width=0.4)
    for bar, t in zip(bars, temps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f} FPS\n({t:.1f}ms)',
                ha='center', va='bottom', fontsize=10)

    ax.set_title('Vitesse d\'inference\n(GPU RTX 3050)')
    ax.set_ylabel('FPS'); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = BENCH_DIR / "comparaison_metriques.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Graphique sauvegarde : {out.name}")

# =============================================================================
# EXEMPLES VISUELS
# =============================================================================

def generer_exemples(res_v1, res_v2):
    """Genere une image montrant des exemples cote a cote v1 vs v2."""
    print("  Generation des exemples visuels...")

    # Choisir des images variees : bonnes detections, erreurs, cas difficiles
    indices = random.sample(range(len(res_v2)), min(N_EXEMPLES, len(res_v2)))

    n_cols = 4
    n_rows = (len(indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows * 2, n_cols,
                              figsize=(n_cols * 4, n_rows * 2 * 3))
    fig.suptitle("Exemples : YOLO v1 (haut) vs YOLO v2 (bas)",
                 fontsize=13, fontweight='bold')

    def dessiner(ax, img_bgr, gt_boxes, pred_boxes, pred_confs, titre):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        # Ground truth en bleu
        for gb in gt_boxes:
            rect = patches.Rectangle(
                (gb[0], gb[1]), gb[2]-gb[0], gb[3]-gb[1],
                linewidth=2, edgecolor='#4499ff', facecolor='none', linestyle='--'
            )
            ax.add_patch(rect)
        # Predictions en vert ou rouge
        for pb, conf in zip(pred_boxes, pred_confs):
            matched = any(iou(pb, gb) >= IOU_SEUIL for gb in gt_boxes)
            col = '#00ff88' if matched else '#ff4466'
            rect = patches.Rectangle(
                (pb[0], pb[1]), pb[2]-pb[0], pb[3]-pb[1],
                linewidth=2, edgecolor=col, facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(pb[0], pb[1]-5, f'{conf:.2f}',
                    color=col, fontsize=7, fontweight='bold')
        ax.set_title(titre, fontsize=8, pad=2)
        ax.axis('off')

    for plot_i, idx in enumerate(indices):
        row = (plot_i // n_cols) * 2
        col = plot_i % n_cols
        r2  = res_v2[idx]
        img = cv2.imread(str(r2['img_path']))
        if img is None:
            continue

        # Trouver le meme fichier dans res_v1
        r1 = None
        for r in res_v1:
            if r['img_path'].name == r2['img_path'].name:
                r1 = r; break

        # Ligne 1 : v1
        ax_v1 = axes[row][col] if n_rows > 1 else axes[0][col]
        if r1:
            dessiner(ax_v1, img, r2['gt_boxes'], r1['pred_boxes'], r1['pred_confs'],
                     f"v1  P={len(r1['pred_boxes'])} GT={len(r2['gt_boxes'])}")
        else:
            ax_v1.axis('off')

        # Ligne 2 : v2
        ax_v2 = axes[row+1][col] if n_rows > 1 else axes[1][col]
        dessiner(ax_v2, img, r2['gt_boxes'], r2['pred_boxes'], r2['pred_confs'],
                 f"v2  P={len(r2['pred_boxes'])} GT={len(r2['gt_boxes'])}")

    # Masquer les axes vides
    for i in range(len(indices), n_rows * n_cols):
        row = (i // n_cols) * 2; col = i % n_cols
        if n_rows > 1:
            axes[row][col].axis('off')
            axes[row+1][col].axis('off')

    plt.tight_layout()
    out = BENCH_DIR / "exemples_detections.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Exemples sauvegardes : {out.name}")

# =============================================================================
# RAPPORT TEXTE
# =============================================================================

def rapport_texte(m_v1, m_v2, map_v1, map_v2):
    lines = [
        "=" * 60,
        "BENCHMARK YOLO v1 vs v2",
        f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        "=" * 60,
        "",
        f"Val set : {m_v2['n_images']} images (DATASET_YOLO_V2/val/)",
        f"IOU seuil : {IOU_SEUIL}   Confiance seuil : {CONF_SEUIL}",
        "",
        f"{'Metrique':<25} {'v1':>10} {'v2':>10} {'Delta':>10}",
        "-" * 55,
    ]

    metriques = [
        ("Precision",     m_v1['precision'], m_v2['precision']),
        ("Recall",        m_v1['recall'],    m_v2['recall']),
        ("F1 score",      m_v1['f1'],        m_v2['f1']),
        ("Inference (ms)", m_v1['temps_ms'], m_v2['temps_ms']),
        ("FPS (GPU)",     m_v1['fps'],       m_v2['fps']),
    ]

    if map_v1 and map_v2:
        metriques.insert(3, ("mAP@0.5 (officiel)", map_v1['map50'], map_v2['map50']))
        metriques.insert(4, ("mAP@0.5-0.95",       map_v1['map50_95'], map_v2['map50_95']))

    for nom, v1, v2 in metriques:
        delta = v2 - v1
        signe = "+" if delta >= 0 else ""
        lines.append(f"  {nom:<23} {v1:>10.4f} {v2:>10.4f} {signe+f'{delta:.4f}':>10}")

    lines += [
        "",
        "=" * 60,
        "CONCLUSION",
        "=" * 60,
    ]

    if map_v2:
        if map_v2['map50'] >= 0.95:
            lines.append("  Excellent — pret pour deploiement terrain.")
        elif map_v2['map50'] >= 0.90:
            lines.append("  Tres bon — utilisable sur le terrain.")
        else:
            lines.append("  Correct — peut etre ameliore avec plus de donnees.")

    rapport = "\n".join(lines)
    print("\n" + rapport)

    out = BENCH_DIR / "rapport_benchmark.txt"
    out.write_text(rapport, encoding='utf-8')

    # Aussi en JSON
    json_out = BENCH_DIR / "rapport_benchmark.json"
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "val_images": m_v2['n_images'],
            "iou_seuil": IOU_SEUIL,
            "conf_seuil": CONF_SEUIL,
            "v1": {**m_v1, **(map_v1 or {})},
            "v2": {**m_v2, **(map_v2 or {})},
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  Rapport texte : {out.name}")
    print(f"  Rapport JSON  : {json_out.name}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BENCHMARK YOLO v1 vs v2")
    print("Orangs-outangs | CNRS IPHC Strasbourg")
    print("=" * 60)

    # Preparer le dossier de sortie
    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    # Verifier le val set
    images_val = sorted((VAL_DIR / "images").glob("*.jpg"))
    if not images_val:
        images_val = sorted((VAL_DIR / "images").glob("*.JPG"))
    if not images_val:
        print(f"ERREUR : aucune image dans {VAL_DIR / 'images'}")
        print("Lance d'abord 2b_train_yolo_v2.py")
        sys.exit(1)

    print(f"\n  Val set : {len(images_val)} images")

    # Charger les modeles
    print("\nChargement des modeles...")
    modeles = charger_modeles()

    # Lire les mAP officiels depuis les CSV Ultralytics
    print("\nLecture des metriques officielles Ultralytics...")
    map_v1 = lire_map_depuis_csv("v1")
    map_v2 = lire_map_depuis_csv("v2")

    if map_v1:
        print(f"  YOLO v1 : mAP50={map_v1['map50']:.4f}  ({map_v1['n_epochs']} epochs)")
    if map_v2:
        print(f"  YOLO v2 : mAP50={map_v2['map50']:.4f}  ({map_v2['n_epochs']} epochs)")
    else:
        print("  YOLO v2 : entrainement en cours ou pas encore termine")

    # Evaluation sur le val set
    m_v1, res_v1 = evaluer(modeles.get('v1'), images_val, "v1") \
        if 'v1' in modeles else ({
            "version":"v1","tp":0,"fp":0,"fn":0,
            "precision":0,"recall":0,"f1":0,
            "temps_ms":0,"fps":0,"n_images":len(images_val)
        }, [])

    m_v2, res_v2 = evaluer(modeles['v2'], images_val, "v2")

    # Graphiques
    generer_graphiques(m_v1, m_v2, map_v1, map_v2)

    # Exemples visuels
    if res_v1 and res_v2:
        generer_exemples(res_v1, res_v2)

    # Rapport texte + JSON
    rapport_texte(m_v1, m_v2, map_v1, map_v2)

    print(f"\n  Tous les fichiers dans : {BENCH_DIR}")
    print("=" * 60)
