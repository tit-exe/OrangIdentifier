# =============================================================================
# 3_extract_faces.py
# Extraction automatique des visages depuis toutes les photos originales.
# Utilise le modele YOLO entraine (best.pt).
#
# IMPORTANT : Supprime et recrée DATASET_CLASSIFICATION/raw/ a chaque lancement.
#
# Structure de sortie :
#   DATASET_CLASSIFICATION/
#       raw/
#           Auti/              <- 1 crop 224x224 par photo, identite certaine
#           Molly/
#           ...
#           _a_verifier/
#               Auti/          <- photo originale + JSON boites si 2+ visages
#       boxes_cache.json       <- coordonnees YOLO de chaque crop, pour app revision
#
# Usage : python scripts/3_extract_faces.py
# =============================================================================

import sys
import time
import shutil
import cv2
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PHOTOS_DIR, CLASSIF_RAW_DIR, DATASET_CLASSIF_DIR,
    YOLO_BEST, RESULTS_DIR,
    RESNET_IMG_SIZE,
    creer_dossiers, log_action
)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONF_EXTRACTION = 0.25    # Seuil confiance YOLO
MARGE_CROP      = 0.05    # Marge autour de la boite (5% de la taille de la boite)
TAILLE_FACE     = RESNET_IMG_SIZE   # 224

# Fichier cache : coordonnees YOLO de chaque crop extrait
BOXES_CACHE     = DATASET_CLASSIF_DIR / "boxes_cache.json"

# =============================================================================
# VERIFICATION
# =============================================================================

def verifier_prerequisites():
    if not YOLO_BEST.exists():
        print(f"ERREUR : Modele YOLO introuvable : {YOLO_BEST}")
        sys.exit(1)
    if not PHOTOS_DIR.exists():
        print(f"ERREUR : Dossier PHOTOS introuvable : {PHOTOS_DIR}")
        sys.exit(1)

    individus = sorted([d.name for d in PHOTOS_DIR.iterdir() if d.is_dir()])
    if not individus:
        print(f"ERREUR : Aucun sous-dossier dans {PHOTOS_DIR}")
        sys.exit(1)

    print("=" * 60)
    print("ETAPE 3 - EXTRACTION DES VISAGES")
    print("=" * 60)
    print(f"  Modele   : {YOLO_BEST.name}")
    print(f"  Photos   : {PHOTOS_DIR}")
    print(f"  Sortie   : {CLASSIF_RAW_DIR}")
    print(f"  Cache    : {BOXES_CACHE.name}")
    print(f"  Conf     : {CONF_EXTRACTION}")
    print(f"  Marge    : {MARGE_CROP*100:.0f}%")
    print(f"  Taille   : {TAILLE_FACE}x{TAILLE_FACE}px")
    print(f"  Individus: {individus}")
    print("=" * 60)
    return individus

# =============================================================================
# NETTOYAGE ET PREPARATION
# =============================================================================

def nettoyer_et_preparer(individus):
    # Supprimer raw/ entier si existant
    if CLASSIF_RAW_DIR.exists():
        print(f"\n  Suppression de raw/ existant...")
        shutil.rmtree(CLASSIF_RAW_DIR)
        print(f"  Supprime.")

    # Supprimer cache existant
    if BOXES_CACHE.exists():
        BOXES_CACHE.unlink()

    # Recreer la structure
    for ind in individus:
        (CLASSIF_RAW_DIR / ind).mkdir(parents=True, exist_ok=True)

    verif_dir = CLASSIF_RAW_DIR / "_a_verifier"
    for ind in individus:
        (verif_dir / ind).mkdir(parents=True, exist_ok=True)

    DATASET_CLASSIF_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Dossiers crees.")
    return verif_dir

# =============================================================================
# CROP
# =============================================================================

def extraire_crop(img, x1, y1, x2, y2):
    """
    Extrait un crop depuis une image OpenCV avec marge,
    redimensionne en TAILLE_FACE x TAILLE_FACE.
    Retourne (face_resizee, x1_final, y1_final, x2_final, y2_final)
    pour enregistrer les vraies coordonnees utilisees dans le cache.
    """
    h, w = img.shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    mx = int(bw * MARGE_CROP)
    my = int(bh * MARGE_CROP)

    x1f = max(0, x1 - mx)
    y1f = max(0, y1 - my)
    x2f = min(w, x2 + mx)
    y2f = min(h, y2 + my)

    crop = img[y1f:y2f, x1f:x2f]
    if crop.size == 0:
        return None, x1f, y1f, x2f, y2f

    face = cv2.resize(crop, (TAILLE_FACE, TAILLE_FACE), interpolation=cv2.INTER_LANCZOS4)
    return face, x1f, y1f, x2f, y2f

# =============================================================================
# BARRE DE PROGRESSION
# =============================================================================

def barre(current, total, debut, largeur=40):
    pct    = current / total if total > 0 else 0
    fill   = int(largeur * pct)
    b      = "=" * fill + "-" * (largeur - fill)
    elapsed = time.time() - debut
    eta    = str(timedelta(seconds=int(elapsed / current * (total - current)))) if current > 0 else "?"
    print(f"\r  [{b}] {current}/{total} ({pct*100:.1f}%)  ETA: {eta}   ", end="", flush=True)

# =============================================================================
# EXTRACTION PRINCIPALE
# =============================================================================

def extraire(individus, verif_dir):
    from ultralytics import YOLO

    print("\n  Chargement du modele YOLO...")
    model = YOLO(str(YOLO_BEST))
    print("  Modele charge.\n")

    # Collecter toutes les photos
    toutes = []
    for ind in individus:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in (PHOTOS_DIR / ind).glob(ext):
                toutes.append((ind, p))
    total = len(toutes)
    print(f"  {total} photos | {len(individus)} individus\n")

    debut = time.time()

    # Cache boxes : cle = "Individu/stem"
    # Contient toutes les infos pour retrouver les coordonnees originales
    boxes_cache = {}

    # Statistiques
    stats = defaultdict(lambda: {
        "photos": 0, "faces": 0, "multi": 0, "rates": 0, "erreurs": 0
    })
    gs = {"total": total, "faces": 0, "multi": 0, "rates": 0, "erreurs": 0}
    journal_multi = []

    for i, (individu, photo_path) in enumerate(toutes):
        barre(i + 1, total, debut)

        img = cv2.imread(str(photo_path))
        if img is None:
            stats[individu]["erreurs"] += 1
            gs["erreurs"] += 1
            continue

        h, w = img.shape[:2]

        try:
            results = model.predict(
                source=str(photo_path),
                conf=CONF_EXTRACTION,
                verbose=False,
                device=0,
            )
        except Exception:
            stats[individu]["erreurs"] += 1
            gs["erreurs"] += 1
            continue

        boxes = results[0].boxes
        stats[individu]["photos"] += 1
        nom   = photo_path.stem
        cle   = f"{individu}/{nom}"

        if boxes is None or len(boxes) == 0:
            stats[individu]["rates"] += 1
            gs["rates"] += 1
            continue

        nb = len(boxes)

        if nb == 1:
            # --- Cas simple : 1 visage ---
            x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[0].tolist()]
            conf = float(boxes.conf[0])
            face, x1f, y1f, x2f, y2f = extraire_crop(img, x1, y1, x2, y2)

            if face is not None:
                dest = CLASSIF_RAW_DIR / individu / f"{nom}.jpg"
                cv2.imwrite(str(dest), face)

                # Enregistrer dans le cache
                boxes_cache[cle] = {
                    "individu":      individu,
                    "stem":          nom,
                    "photo_source":  str(photo_path),
                    "crop_dest":     str(dest),
                    "img_w":         w,
                    "img_h":         h,
                    # Coordonnees YOLO brutes (avant marge)
                    "yolo_x1": x1, "yolo_y1": y1,
                    "yolo_x2": x2, "yolo_y2": y2,
                    "yolo_conf": round(conf, 3),
                    # Coordonnees finales utilisees pour le crop (avec marge)
                    "crop_x1": x1f, "crop_y1": y1f,
                    "crop_x2": x2f, "crop_y2": y2f,
                    "marge_pct":  MARGE_CROP,
                    "taille_face": TAILLE_FACE,
                    "nb_detections": 1,
                    "statut": "auto",   # auto | valide | supprime | passe
                    "ts_extraction": datetime.now().isoformat(),
                }

                stats[individu]["faces"] += 1
                gs["faces"] += 1

        else:
            # --- Cas multi : 2+ visages ---
            stats[individu]["multi"] += 1
            gs["multi"] += 1

            dest_verif = verif_dir / individu / f"{nom}.jpg"
            shutil.copy(str(photo_path), str(dest_verif))

            boites_info = []
            for j, box in enumerate(boxes):
                bx1, by1, bx2, by2 = [int(v) for v in box.xyxy[0].tolist()]
                bconf = float(box.conf[0])
                boites_info.append({
                    "index": j,
                    "x1": bx1, "y1": by1, "x2": bx2, "y2": by2,
                    "conf": round(bconf, 3),
                })

            # JSON cote de la photo dans _a_verifier
            json_path = verif_dir / individu / f"{nom}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "individu":      individu,
                    "stem":          nom,
                    "photo_source":  str(photo_path),
                    "img_w":         w,
                    "img_h":         h,
                    "nb_detections": nb,
                    "boites":        boites_info,
                }, f, indent=2, ensure_ascii=False)

            # Aussi dans le cache global avec statut "multi"
            boxes_cache[cle] = {
                "individu":      individu,
                "stem":          nom,
                "photo_source":  str(photo_path),
                "crop_dest":     None,  # pas encore extrait
                "img_w":         w,
                "img_h":         h,
                "nb_detections": nb,
                "boites":        boites_info,
                "statut":        "multi_a_verifier",
                "ts_extraction": datetime.now().isoformat(),
            }

            journal_multi.append({
                "individu": individu,
                "stem":     nom,
                "nb":       nb,
            })

    print()

    # Sauvegarder le cache
    with open(BOXES_CACHE, 'w', encoding='utf-8') as f:
        json.dump(boxes_cache, f, indent=2, ensure_ascii=False)
    print(f"\n  Cache sauvegarde : {BOXES_CACHE}")
    print(f"  Entrees dans le cache : {len(boxes_cache)}")

    return stats, gs, journal_multi, boxes_cache

# =============================================================================
# RAPPORT
# =============================================================================

def rapport(stats, gs, journal_multi, debut):
    duree = timedelta(seconds=int(time.time() - debut))

    print("\n" + "=" * 60)
    print("RAPPORT FINAL")
    print("=" * 60)
    print(f"  Duree              : {duree}")
    print(f"  Photos traitees    : {gs['total']}")
    print(f"  Faces extraites    : {gs['faces']}")
    print(f"  Multi-detections   : {gs['multi']}  -> _a_verifier/")
    print(f"  Photos sans face   : {gs['rates']}")
    print(f"  Erreurs            : {gs['erreurs']}")

    print(f"\n  {'Individu':<28} {'Photos':>7} {'Faces':>6} {'Multi':>6} {'Rates':>6}")
    print(f"  {'-'*28} {'-'*7} {'-'*6} {'-'*6} {'-'*6}")
    for ind in sorted(stats.keys()):
        s = stats[ind]
        print(f"  {ind:<28} {s['photos']:>7} {s['faces']:>6} {s['multi']:>6} {s['rates']:>6}")

    taux = gs['faces'] / max(gs['total'], 1) * 100
    print(f"\n  Taux auto : {taux:.1f}%")
    print("=" * 60)
    print("ETAPE 3 TERMINEE")
    print("=" * 60)
    print("\nProchaines etapes :")
    print("  1. Lance 3b_reviser_faces.py pour corriger/valider les crops")
    print("  2. Lance 4_train_resnet.py quand le dataset est propre")

    metriques = {
        "faces": gs["faces"], "multi": gs["multi"],
        "rates": gs["rates"], "taux": round(taux, 2),
        "duree_s": int(time.time() - debut)
    }

    # Sauvegarder rapport JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rp = RESULTS_DIR / "extraction_rapport.json"
    with open(rp, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp":    datetime.now().isoformat(),
            "global":       gs,
            "metriques":    metriques,
            "multi_detail": journal_multi,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Rapport : {rp}")

    return metriques

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    debut = time.time()

    individus = verifier_prerequisites()
    verif_dir = nettoyer_et_preparer(individus)

    confirmation = input("\nSupprimer raw/ existant et relancer l'extraction ? (o/n) : ").strip().lower()
    if confirmation != 'o':
        print("Annule.")
        sys.exit(0)

    stats, gs, journal_multi, cache = extraire(individus, verif_dir)
    metriques = rapport(stats, gs, journal_multi, debut)
    log_action("3_extract_faces", "extraction_terminee", metriques)