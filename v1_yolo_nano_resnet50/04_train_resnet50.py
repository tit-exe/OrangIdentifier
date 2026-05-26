#!/usr/bin/env python3
# =============================================================================
# 4_train_resnet.py
# Identification individuelle des orangs-outangs — ResNet50 Fine-tuning
# CNRS IPHC Strasbourg | Stage Titouane CPI2
#
# Architecture : ResNet50 (ImageNet) → AvgPool → Dropout(0.5) → FC(N)
#
# Anti-overfit (PRIORITÉ sur petit dataset) :
#   ▸ Fine-tuning progressif : freeze backbone → unfreeze avec LR différentiel
#   ▸ Augmentation forte (crop, flip, jitter, perspective, blur, erasing)
#   ▸ Mixup (α=0.2) — mélange paires d'images, lisse la frontière de décision
#   ▸ Label smoothing (ε=0.1) — empêche l'overconfidence
#   ▸ WeightedRandomSampler — compense Molly×408 vs PUTRI×56
#   ▸ Early stopping patience=20 sur val accuracy
#   ▸ Weight decay (AdamW)
#
# Détection "individu inconnu" (open-set recognition) :
#   ▸ Seuil calibré sur val set (5e percentile des prédictions correctes)
#   ▸ Embeddings train sauvegardés → kNN fallback possible
#   ▸ Si conf < seuil → "INDIVIDU NON RECONNU"
#
# Scalabilité — Ajouter un nouvel individu :
#   ▸ Backbone sauvegardé séparément → fine-tune uniquement la tête (~10 min)
#   ▸ Voir instructions en bas de ce fichier
# =============================================================================

import sys
import json
import time
import random
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')

# Dépendances optionnelles
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("ERREUR : pip install scikit-learn matplotlib seaborn")
    sys.exit(1)

# Config projet
sys.path.insert(0, str(Path(__file__).parent))
from config import BASE_DIR, DATASET_CLASSIF_DIR, MODELS_DIR, RESULTS_DIR

# =============================================================================
# CONFIGURATION — modifie ici si besoin
# =============================================================================

RAW_DIR       = DATASET_CLASSIF_DIR / "raw"
OUT_DIR       = RESULTS_DIR / "resnet_training"
MODEL_SAVE    = MODELS_DIR / "resnet_orangs.pt"
BACKBONE_SAVE = MODELS_DIR / "backbone_orangs.pt"
META_SAVE     = MODELS_DIR / "resnet_metadata.json"
EMBED_SAVE    = MODELS_DIR / "embeddings_train.pt"

IMG_SIZE      = 224
BATCH_SIZE    = 32
SEED          = 42

# Phases d'entraînement
EPOCHS_FREEZE   = 10     # Phase 1 : backbone gelé, train la tête seulement
EPOCHS_UNFREEZE = 90     # Phase 2 : fine-tuning complet (early stop)
LR_HEAD         = 1e-3   # Learning rate tête (Phase 1 et 2)
LR_BACKBONE     = 5e-5   # Learning rate backbone (Phase 2 seulement, très bas)
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTH    = 0.10   # Label smoothing — réduit overconfidence
MIXUP_ALPHA     = 0.20   # Mixup — 0 pour désactiver
PATIENCE        = 20     # Early stopping : epochs sans amélioration
DROPOUT         = 0.50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =============================================================================
# AFFICHAGE — helpers
# =============================================================================

SEP  = "=" * 64
SEP2 = "─" * 64

def titre(texte):
    print(f"\n{SEP}")
    print(f"  {texte}")
    print(SEP)

def sous_titre(texte):
    print(f"\n  {texte}")
    print(f"  {SEP2[:len(texte)+2]}")

def barre(current, total, width=36, suffix=''):
    """Barre de progression inline."""
    pct  = current / max(total, 1)
    fill = int(width * pct)
    bar  = '█' * fill + '░' * (width - fill)
    print(f"\r  [{bar}] {current:>4}/{total}  {suffix:<40}", end='', flush=True)

def fmt_temps(secondes):
    return str(timedelta(seconds=int(secondes)))

# =============================================================================
# CHARGEMENT DU DATASET
# =============================================================================

def charger_dataset():
    """Scan RAW_DIR, ignore les dossiers commençant par '_'."""
    print(f"\n  Scan de : {RAW_DIR}")

    if not RAW_DIR.exists():
        print(f"  ERREUR : dossier introuvable → {RAW_DIR}")
        sys.exit(1)

    exts = {'.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG'}
    classes_dirs = sorted([
        d for d in RAW_DIR.iterdir()
        if d.is_dir() and not d.name.startswith('_')
    ])

    if not classes_dirs:
        print(f"  ERREUR : aucun sous-dossier individu dans {RAW_DIR}")
        sys.exit(1)

    classes = [d.name for d in classes_dirs]
    c2i = {c: i for i, c in enumerate(classes)}
    chemins, labels = [], []

    for d in classes_dirs:
        imgs = [f for f in d.iterdir() if f.suffix in exts]
        for img in sorted(imgs):
            chemins.append(img)
            labels.append(c2i[d.name])

    return chemins, labels, classes

# =============================================================================
# AFFICHAGE DISTRIBUTION
# =============================================================================

def afficher_distribution(labels, classes, nom="Dataset"):
    counts = Counter(labels)
    total  = len(labels)
    max_n  = max(counts.values())

    print(f"\n  {'Individu':<22} {'N':>6}  {'%':>6}  Distribution")
    print(f"  {'─'*22}  {'─'*6}  {'─'*6}  {'─'*25}")
    for i, cls in enumerate(classes):
        n   = counts[i]
        pct = n / total * 100
        bar = '█' * int(n / max_n * 25)
        print(f"  {cls:<22} {n:>6}  {pct:>5.1f}%  {bar}")
    print(f"  {'─'*22}  {'─'*6}")
    print(f"  {'TOTAL':<22} {total:>6}")

    ratio = max_n / min(counts.values())
    print(f"\n  Ratio max/min : {ratio:.1f}×")
    if ratio > 4:
        print("  → Déséquilibre important → WeightedRandomSampler activé")

# =============================================================================
# SPLIT STRATIFIÉ 70 / 15 / 15
# =============================================================================

def split_dataset(chemins, labels, classes):
    X = list(range(len(chemins)))

    X_tv, X_te, y_tv, y_te = train_test_split(
        X, labels, test_size=0.15, stratify=labels, random_state=SEED
    )
    val_ratio = 0.15 / 0.85
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tv, y_tv, test_size=val_ratio, stratify=y_tv, random_state=SEED
    )

    total = len(chemins)
    print(f"\n  Split stratifié (par individu) :")
    print(f"    Train : {len(X_tr):>4} images  ({len(X_tr)/total*100:.1f}%)")
    print(f"    Val   : {len(X_va):>4} images  ({len(X_va)/total*100:.1f}%)")
    print(f"    Test  : {len(X_te):>4} images  ({len(X_te)/total*100:.1f}%)")

    # Vérifier que toutes les classes sont dans le train
    classes_train = set(y_tr)
    if len(classes_train) < len(classes):
        absents = [classes[i] for i in range(len(classes)) if i not in classes_train]
        print(f"\n  ATTENTION : {absents} absents du train (images insuffisantes)")

    return (
        [chemins[i] for i in X_tr], y_tr,
        [chemins[i] for i in X_va], y_va,
        [chemins[i] for i in X_te], y_te,
    )

# =============================================================================
# DATASET PYTORCH
# =============================================================================

class OrangDataset(Dataset):
    def __init__(self, chemins, labels, transform):
        self.chemins   = chemins
        self.labels    = labels
        self.transform = transform

    def __len__(self): return len(self.chemins)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.chemins[idx]).convert('RGB')
        except Exception:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        return self.transform(img), self.labels[idx]

# =============================================================================
# AUGMENTATIONS
# =============================================================================

def get_transforms():
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Augmentation forte — clé pour la généralisation sur petit dataset
    train_tf = T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.60, 1.0), ratio=(0.85, 1.15)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.05),
        T.RandomRotation(degrees=20),
        # Simule conditions zoo : éclairage variable, vitre, reflets
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.25, hue=0.08),
        T.RandomGrayscale(p=0.05),
        T.RandomPerspective(distortion_scale=0.25, p=0.3),
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        # Cache aléatoirement une petite zone (simule occlusion partielle)
        T.ToTensor(),
        T.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3), value='random'),
        norm,
    ])

    val_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        norm,
    ])

    return train_tf, val_tf

# =============================================================================
# WEIGHTED SAMPLER — équilibre les classes sous-représentées
# =============================================================================

def make_sampler(labels):
    counts = Counter(labels)
    poids  = {c: 1.0 / counts[c] for c in counts}
    sample_weights = [poids[l] for l in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

# =============================================================================
# MODÈLE — ResNet50 avec tête personnalisée
# =============================================================================

def creer_modele(n_classes):
    # IMAGENET1K_V2 = meilleure accuracy que V1, important pour transfer learning
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    in_feat = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(in_feat, n_classes)
    )
    return model.to(DEVICE)

def geler_backbone(model):
    """Gèle tout sauf fc (tête)."""
    for name, param in model.named_parameters():
        param.requires_grad = ('fc' in name)

def degeler_backbone(model):
    for p in model.parameters():
        p.requires_grad = True

def get_optimizer_phase2(model):
    """Optimiseur à learning rates différentiels backbone/tête."""
    backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n and p.requires_grad]
    head_params     = [p for n, p in model.named_parameters() if 'fc' in n and p.requires_grad]
    return optim.AdamW([
        {'params': backbone_params, 'lr': LR_BACKBONE},
        {'params': head_params,     'lr': LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

# =============================================================================
# MIXUP — mélange paires d'images + labels
# =============================================================================

def mixup_data(x, y, alpha=0.2):
    """Applique Mixup sur un batch. Retourne x_mix, ya, yb, lambda."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1 - lam)  # toujours >= 0.5 → image dominante préservée
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, y, y[idx], lam

def mixup_loss(logits, ya, yb, lam):
    """Perte Mixup = somme pondérée sur les deux labels."""
    loss_a = F.cross_entropy(logits, ya, label_smoothing=LABEL_SMOOTH)
    loss_b = F.cross_entropy(logits, yb, label_smoothing=LABEL_SMOOTH)
    return lam * loss_a + (1 - lam) * loss_b

# =============================================================================
# EPOCH D'ENTRAÎNEMENT
# =============================================================================

def train_epoch(model, loader, optimizer, use_mixup):
    model.train()
    total_loss = correct = total = 0

    t0 = time.time()
    for i, (imgs, labs) in enumerate(loader):
        imgs = imgs.to(DEVICE, non_blocking=True)
        labs = labs.to(DEVICE, non_blocking=True)

        if use_mixup and MIXUP_ALPHA > 0:
            imgs, ya, yb, lam = mixup_data(imgs, labs, MIXUP_ALPHA)
            logits = model(imgs)
            loss   = mixup_loss(logits, ya, yb, lam)
            # Accuracy sur le label dominant (ya)
            preds_acc = logits.argmax(1)
            correct  += (preds_acc == ya).sum().item()
        else:
            logits = model(imgs)
            loss   = F.cross_entropy(logits, labs, label_smoothing=LABEL_SMOOTH)
            correct += (logits.argmax(1) == labs).sum().item()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)

        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(loader) - i - 1)
        barre(i + 1, len(loader),
              suffix=f"loss={total_loss/total:.4f}  acc={correct/total*100:.1f}%  ETA={fmt_temps(eta)}")

    print()
    return total_loss / total, correct / total

# =============================================================================
# EPOCH DE VALIDATION
# =============================================================================

@torch.no_grad()
def val_epoch(model, loader):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labs, all_confs = [], [], []

    for i, (imgs, labs) in enumerate(loader):
        imgs, labs = imgs.to(DEVICE, non_blocking=True), labs.to(DEVICE, non_blocking=True)
        logits = model(imgs)
        loss   = F.cross_entropy(logits, labs, label_smoothing=LABEL_SMOOTH)
        probs  = torch.softmax(logits, dim=1)
        confs, preds = probs.max(1)

        total_loss += loss.item() * imgs.size(0)
        correct    += (preds == labs).sum().item()
        total      += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labs.extend(labs.cpu().numpy())
        all_confs.extend(confs.cpu().numpy())

        barre(i + 1, len(loader),
              suffix=f"loss={total_loss/total:.4f}  acc={correct/total*100:.1f}%")

    print()
    return (total_loss / total, correct / total,
            np.array(all_preds), np.array(all_labs), np.array(all_confs))

# =============================================================================
# BOUCLE D'ENTRAÎNEMENT PRINCIPALE
# =============================================================================

def entrainer(model, train_loader, val_loader, n_classes):
    historique = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc   = 0.0
    best_state     = None
    patience_count = 0

    # ──────────────────────────────────────────────
    # PHASE 1 — Backbone gelé, tête seulement
    # ──────────────────────────────────────────────
    titre(f"PHASE 1 / 2 — TÊTE SEULEMENT  ({EPOCHS_FREEZE} epochs)")
    print(f"  Backbone ImageNet gelé — LR tête = {LR_HEAD}")
    print(f"  Pas de Mixup en phase 1 (stabilise la convergence initiale)")

    geler_backbone(model)
    optimizer  = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS_FREEZE, eta_min=1e-5
    )

    t_phase1 = time.time()
    for epoch in range(1, EPOCHS_FREEZE + 1):
        sous_titre(f"Epoch {epoch}/{EPOCHS_FREEZE}  [Phase 1 — Freeze]")
        t_loss, t_acc = train_epoch(model, train_loader, optimizer, use_mixup=False)
        v_loss, v_acc, _, _, _ = val_epoch(model, val_loader)
        scheduler.step()

        historique['train_loss'].append(t_loss)
        historique['val_loss'].append(v_loss)
        historique['train_acc'].append(t_acc)
        historique['val_acc'].append(v_acc)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  ★ Nouveau meilleur val acc : {best_val_acc*100:.2f}%")

        elapsed = time.time() - t_phase1
        eta_p1  = elapsed / epoch * (EPOCHS_FREEZE - epoch)
        print(f"  Elapsed : {fmt_temps(elapsed)}  |  ETA phase 1 : {fmt_temps(eta_p1)}")

    # ──────────────────────────────────────────────
    # PHASE 2 — Fine-tuning complet, LR différentiel
    # ──────────────────────────────────────────────
    titre(f"PHASE 2 / 2 — FINE-TUNING COMPLET  (max {EPOCHS_UNFREEZE} epochs)")
    print(f"  LR backbone = {LR_BACKBONE}  |  LR tête = {LR_HEAD}")
    print(f"  Mixup α={MIXUP_ALPHA}  |  Early stop patience={PATIENCE}")
    print(f"  La phase 2 s'arrête automatiquement si pas d'amélioration\n")

    degeler_backbone(model)
    optimizer = get_optimizer_phase2(model)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS_UNFREEZE, eta_min=1e-6
    )

    t_phase2 = time.time()
    for epoch in range(1, EPOCHS_UNFREEZE + 1):
        sous_titre(
            f"Epoch {epoch}/{EPOCHS_UNFREEZE}  [Phase 2]  "
            f"patience {patience_count}/{PATIENCE}"
        )
        t_loss, t_acc = train_epoch(model, train_loader, optimizer, use_mixup=True)
        v_loss, v_acc, preds_v, labs_v, confs_v = val_epoch(model, val_loader)
        scheduler.step()

        historique['train_loss'].append(t_loss)
        historique['val_loss'].append(v_loss)
        historique['train_acc'].append(t_acc)
        historique['val_acc'].append(v_acc)

        if v_acc > best_val_acc:
            best_val_acc   = v_acc
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
            print(f"  ★ Nouveau meilleur : {best_val_acc*100:.2f}%  [sauvegardé]")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\n  Early stopping déclenché à l'epoch {epoch + EPOCHS_FREEZE}")
                break

        elapsed_p2 = time.time() - t_phase2
        remaining  = EPOCHS_UNFREEZE - epoch
        if epoch > 0:
            eta = elapsed_p2 / epoch * remaining
        else:
            eta = 0
        print(f"  Best = {best_val_acc*100:.2f}%  |  ETA : {fmt_temps(eta)}")

    # Recharger le meilleur modèle
    model.load_state_dict(best_state)
    print(f"\n  Meilleur val accuracy : {best_val_acc*100:.2f}%")
    return model, historique, best_val_acc, preds_v, labs_v, confs_v

# =============================================================================
# CALIBRATION DU SEUIL "INDIVIDU INCONNU"
# =============================================================================

def calibrer_seuil(confs, preds, labs):
    """
    Calcule un seuil de rejet basé sur la distribution des confidences.

    Logique :
      - On cherche le seuil T tel que si conf < T → "inconnu"
      - On le fixe au 5e percentile des prédictions CORRECTES :
        * 95% des bonnes prédictions passent → très peu de faux rejets
        * Toutes les prédictions trop incertaines sont rejetées
      - En pratique, de vrais "inconnus" auront une conf << ce seuil

    Note : Sans images d'inconnus dans le val set, la calibration est
    conservative. Ajuste CONF_SEUIL dans metadata.json si besoin terrain.
    """
    corrects = confs[preds == labs]
    errors   = confs[preds != labs]

    print(f"\n  Distribution des confidences sur val set :")
    print(f"    Prédictions correctes — médiane : {np.median(corrects):.3f}  "
          f"| min : {corrects.min():.3f}  | p5 : {np.percentile(corrects, 5):.3f}")
    if len(errors):
        print(f"    Prédictions erronées  — médiane : {np.median(errors):.3f}  "
              f"| max : {errors.max():.3f}")
        pct_erreurs_capturees = (errors < np.percentile(corrects, 5)).mean()
        print(f"    Erreurs capturées par le seuil p5 : {pct_erreurs_capturees*100:.1f}%")

    seuil = float(np.percentile(corrects, 5))
    seuil = max(seuil, 0.50)  # jamais en dessous de 50%
    seuil = round(seuil, 2)
    print(f"\n  → Seuil calibré : {seuil:.2f}")
    print(f"    Si conf < {seuil:.2f} → afficher 'Individu non reconnu'")
    return seuil

# =============================================================================
# ÉVALUATION FINALE — TEST SET
# =============================================================================

def evaluer_test(model, test_loader, classes, seuil):
    titre("ÉVALUATION — TEST SET (données jamais vues)")

    _, acc, preds, labs, confs = val_epoch(model, test_loader)

    print(f"\n  Accuracy globale (sans rejet) : {acc*100:.2f}%")

    # Avec rejet "inconnu"
    mask = confs >= seuil
    if mask.sum() > 0:
        acc_filtre = (preds[mask] == labs[mask]).mean()
        print(f"  Accuracy avec rejet (≥{seuil:.2f}) : {acc_filtre*100:.2f}%  "
              f"({(1-mask.mean())*100:.1f}% images rejetées)")
    else:
        acc_filtre = acc

    # Détail par individu
    print(f"\n  {'Individu':<22} {'OK':>5}  {'Tot':>5}  {'Acc':>7}  Barre")
    print(f"  {'─'*22}  {'─'*5}  {'─'*5}  {'─'*7}  {'─'*20}")
    for i, cls in enumerate(classes):
        mask_cls = labs == i
        if not mask_cls.any():
            continue
        n_tot = mask_cls.sum()
        n_ok  = (preds[mask_cls] == i).sum()
        a     = n_ok / n_tot
        bar   = '█' * int(a * 20) + '░' * (20 - int(a * 20))
        print(f"  {cls:<22} {n_ok:>5}  {n_tot:>5}  {a*100:>6.1f}%  {bar}")

    # Rapport sklearn
    print(f"\n  Classification report (test set) :\n")
    print(classification_report(labs, preds, target_names=classes, digits=3))

    return preds, labs, confs, acc

# =============================================================================
# EXTRACTION EMBEDDINGS (pour kNN / détection inconnu avancée)
# =============================================================================

@torch.no_grad()
def extraire_embeddings(model, loader):
    """
    Extrait les vecteurs d'embedding (2048-dim) avant la couche FC.
    Utile pour :
      1. Détection d'inconnus par distance cosinus (plus robuste que seuil softmax)
      2. Fine-tuning avec ajout nouveaux individus (kNN dans l'espace d'embedding)
    """
    model.eval()
    all_embeds, all_labs = [], []
    feats_buf = []

    # Hook sur avgpool (sortie = vecteur 2048-dim)
    handle = model.avgpool.register_forward_hook(
        lambda m, inp, out: feats_buf.append(out.flatten(1).cpu())
    )

    for imgs, labs in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        _    = model(imgs)
        all_embeds.append(feats_buf.pop())
        all_labs.extend(labs.numpy())

    handle.remove()
    return torch.cat(all_embeds, 0), np.array(all_labs)

# =============================================================================
# GRAPHIQUES
# =============================================================================

def sauvegarder_graphiques(historique, classes, preds_test, labs_test):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ep = range(1, len(historique['train_loss']) + 1)
    sep_phase = EPOCHS_FREEZE

    # ── Courbes loss / accuracy ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Entraînement ResNet50 — Orangs-outangs (CNRS IPHC Strasbourg)",
        fontsize=13, fontweight='bold'
    )

    for ax, key_tr, key_va, ylabel, title in [
        (axes[0], 'train_loss', 'val_loss',  'Loss',         'Loss'),
        (axes[1], 'train_acc',  'val_acc',   'Accuracy (%)', 'Accuracy'),
    ]:
        y_tr = historique[key_tr]
        y_va = historique[key_va]
        if 'acc' in key_tr:
            y_tr = [v * 100 for v in y_tr]
            y_va = [v * 100 for v in y_va]

        ax.plot(ep, y_tr, label='Train', color='#4488ff', linewidth=1.5)
        ax.plot(ep, y_va, label='Val',   color='#ff6644', linewidth=1.5)
        ax.axvline(x=sep_phase, color='#888888', linestyle='--',
                   alpha=0.6, label='Unfreeze backbone')
        ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.25)

    plt.tight_layout()
    path = OUT_DIR / 'courbes_entrainement.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Courbes    → {path.name}")

    # ── Matrice de confusion ─────────────────────────────────────────
    cm     = confusion_matrix(labs_test, preds_test)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm_pct, annot=True, fmt='.1f', cmap='Blues',
        xticklabels=classes, yticklabels=classes,
        linewidths=0.5, vmin=0, vmax=100, ax=ax
    )
    ax.set_xlabel('Prédit', fontsize=11)
    ax.set_ylabel('Réel',   fontsize=11)
    ax.set_title('Matrice de confusion — Test set (%)', fontsize=13)
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    path = OUT_DIR / 'confusion_matrix.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion  → {path.name}")

# =============================================================================
# SAUVEGARDE
# =============================================================================

def sauvegarder_modele(model, classes, seuil, acc_test,
                       embed_train, labs_embed, historique):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Modèle complet
    torch.save({
        'model_state':  model.state_dict(),
        'classes':      classes,
        'n_classes':    len(classes),
        'img_size':     IMG_SIZE,
        'seuil':        float(seuil),
        'acc_test':     float(acc_test),
        'dropout':      DROPOUT,
    }, MODEL_SAVE)

    # Backbone seul (pour fine-tuning avec nouveaux individus)
    backbone_state = {
        k: v for k, v in model.state_dict().items() if 'fc' not in k
    }
    torch.save({'backbone_state': backbone_state, 'classes': classes}, BACKBONE_SAVE)

    # Embeddings train (2048-dim par image, pour kNN)
    torch.save({
        'embeddings': embed_train,
        'labels':     labs_embed,
        'classes':    classes,
    }, EMBED_SAVE)

    # Metadata JSON (lisible par les autres scripts)
    meta = {
        "timestamp":       datetime.now().isoformat(timespec='seconds'),
        "classes":         classes,
        "n_classes":       len(classes),
        "img_size":        IMG_SIZE,
        "seuil_inconnu":   float(seuil),
        "acc_test_pct":    round(float(acc_test) * 100, 2),
        "architecture":    "ResNet50 IMAGENET1K_V2",
        "batch_size":      BATCH_SIZE,
        "label_smoothing": LABEL_SMOOTH,
        "mixup_alpha":     MIXUP_ALPHA,
        "dropout":         DROPOUT,
        "best_val_acc":    round(max(historique['val_acc']) * 100, 2),
        "epochs_trained":  len(historique['val_acc']),
        "note_inconnu":    f"Si softmax_max < {seuil:.2f} → individu inconnu",
        "ajouter_individu": "Voir instructions en bas de 4_train_resnet.py",
    }
    META_SAVE.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8'
    )

    print(f"\n  {MODEL_SAVE.name:<40} ({MODEL_SAVE.stat().st_size/1e6:.1f} MB)")
    print(f"  {BACKBONE_SAVE.name:<40} ({BACKBONE_SAVE.stat().st_size/1e6:.1f} MB)")
    print(f"  {EMBED_SAVE.name}")
    print(f"  {META_SAVE.name}")

# =============================================================================
# RAPPORT FINAL
# =============================================================================

def rapport_final(classes, acc_test, seuil, historique, t_total):
    best_val = max(historique['val_acc']) * 100
    n_epochs  = len(historique['val_acc'])

    titre("RAPPORT FINAL")
    print(f"  Individus    : {len(classes)}")
    print(f"  Epochs       : {n_epochs} ({EPOCHS_FREEZE} freeze + {n_epochs-EPOCHS_FREEZE} unfreeze)")
    print(f"  Meilleure val acc : {best_val:.2f}%")
    print(f"  Accuracy test     : {acc_test*100:.2f}%")
    print(f"  Seuil inconnu     : {seuil:.2f}")
    print(f"  Temps total       : {fmt_temps(t_total)}")
    print()

    if acc_test >= 0.92:
        note = "EXCELLENT — Prêt pour déploiement Android/TFLite"
    elif acc_test >= 0.85:
        note = "BON — Utilisable, quelques confusions possibles"
    elif acc_test >= 0.75:
        note = "CORRECT — Ajouter images pour Jula/PUTRI si possible"
    else:
        note = "INSUFFISANT — Dataset trop petit pour certaines classes"
    print(f"  {note}")

    print(f"""
  ┌──────────────────────────────────────────────────────────┐
  │  PROCHAINES ÉTAPES                                        │
  │                                                          │
  │  1. Exporter en TFLite                                   │
  │     → python scripts/5_export_tflite.py                  │
  │                                                          │
  │  2. Pipeline vidéo (YOLO + ResNet50)                     │
  │     → python scripts/5_video_pipeline.py                 │
  │                                                          │
  │  3. Ajouter un nouvel individu (ex: Kawan)               │
  │     → Voir section AJOUT INDIVIDU ci-dessous             │
  └──────────────────────────────────────────────────────────┘
""")

    print(f"""  ═══════════════════════════════════════════════════════════
  AJOUT D'UN NOUVEL INDIVIDU — MARCHE À SUIVRE
  ═══════════════════════════════════════════════════════════

  Quand un nouvel orang arrive au zoo :

  1. Photos → PHOTOS/NouvelIndividu/*.jpg
  2. Lance 3_extract_faces.py   (YOLO v2 détecte les faces)
  3. Révise les crops manuellement (3b_reviser_faces.py)
  4. Lance 4b_ajouter_individu.py  ← script à créer

     Ce script :
       a) Charge backbone_orangs.pt  (backbone déjà entraîné)
       b) Remplace la tête : FC(10) → FC(11)
       c) Initialise le nouveau neurone proprement
       d) Fine-tune UNIQUEMENT la tête (backbone gelé)
          → Durée : ~5-10 minutes seulement
          → Pas besoin de réentraîner tout le modèle

  5. Génère nouveau resnet_orangs.pt + metadata.json

  DÉTECTION "INCONNU" sur le terrain :
    Si confidence < {seuil:.2f} → afficher "Individu non reconnu"
    (Ajuster dans metadata.json si trop/pas assez de rejets)

  ═══════════════════════════════════════════════════════════
""")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    t_debut = time.time()

    titre("IDENTIFICATION ORANGS-OUTANGS — ResNet50")
    print(f"  CNRS IPHC Strasbourg | Stage Titouane CPI2")
    print(f"  Device : {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  Batch  : {BATCH_SIZE}  |  Seed : {SEED}")

    # ── 1. Dataset ─────────────────────────────────────────────────
    titre("CHARGEMENT DU DATASET")
    chemins, labels, classes = charger_dataset()
    n_classes = len(classes)
    afficher_distribution(labels, classes)
    print(f"\n  {len(chemins)} images  |  {n_classes} individus")

    # ── 2. Split ────────────────────────────────────────────────────
    titre("SPLIT TRAIN / VAL / TEST")
    tr_p, tr_l, va_p, va_l, te_p, te_l = split_dataset(chemins, labels, classes)

    # ── 3. Transforms & DataLoaders ─────────────────────────────────
    train_tf, val_tf = get_transforms()

    train_ds = OrangDataset(tr_p, tr_l, train_tf)
    val_ds   = OrangDataset(va_p, va_l, val_tf)
    test_ds  = OrangDataset(te_p, te_l, val_tf)

    sampler      = make_sampler(tr_l)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    # ── 4. Modèle ───────────────────────────────────────────────────
    titre("MODÈLE")
    model  = creer_modele(n_classes)
    n_par  = sum(p.numel() for p in model.parameters())
    n_tr   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ResNet50 : {n_par/1e6:.1f}M paramètres total")
    print(f"  Phase 1  : {sum(p.numel() for n,p in model.named_parameters() if 'fc' in n)/1e6:.2f}M entraînables (tête)")
    print(f"  Phase 2  : {n_tr/1e6:.1f}M entraînables (tout)")

    # ── 5. Entraînement ─────────────────────────────────────────────
    model, historique, best_val, preds_v, labs_v, confs_v = entrainer(
        model, train_loader, val_loader, n_classes
    )

    # ── 6. Calibration seuil inconnu ────────────────────────────────
    titre("CALIBRATION SEUIL INCONNU")
    seuil = calibrer_seuil(confs_v, preds_v, labs_v)

    # ── 7. Évaluation test ──────────────────────────────────────────
    preds_test, labs_test, confs_test, acc_test = evaluer_test(
        model, test_loader, classes, seuil
    )

    # ── 8. Embeddings (kNN / future détection inconnu avancée) ──────
    titre("EXTRACTION EMBEDDINGS TRAIN")
    print("  (vecteurs 2048-dim pour détection d'inconnus par distance cosinus)")
    embed_train, labs_embed = extraire_embeddings(model, train_loader)
    print(f"  Tenseur : {embed_train.shape}  ({embed_train.shape[0]} images × 2048 dims)")

    # ── 9. Graphiques ───────────────────────────────────────────────
    titre("GRAPHIQUES")
    sauvegarder_graphiques(historique, classes, preds_test, labs_test)
    print(f"  Dossier : {OUT_DIR}")

    # ── 10. Sauvegarde ──────────────────────────────────────────────
    titre("SAUVEGARDE")
    sauvegarder_modele(model, classes, seuil, acc_test,
                       embed_train, labs_embed, historique)

    # ── 11. Rapport ─────────────────────────────────────────────────
    t_total = time.time() - t_debut
    rapport_final(classes, acc_test, seuil, historique, t_total)

# =============================================================================
# FIN DU SCRIPT
# =============================================================================
#
# PERFORMANCES ATTENDUES (sur tes 1986 images corrigées à la main) :
#
#   Individus > 150 images (Auti, Mathai, Molly, PULCO, Sari, Sinta, Ujian) :
#     → Accuracy individuelle : 88-95%
#
#   Individus < 100 images (Jula=63, NOAH=76, PUTRI=56) :
#     → Accuracy individuelle : 70-85%
#     → Ce n'est pas un bug : le modèle est limité par le dataset
#     → Solution : photographier davantage Jula/PUTRI/NOAH
#
#   Accuracy globale test set : 85-93% (vs YOLO 99.4%)
#
#   Pourquoi pas 99% comme YOLO ?
#     YOLO détecte UNE classe (visage) sur 1986 images → tâche simple
#     ResNet distingue 10 individus → tâche bien plus difficile
#     Pour comparaison : Face ID Apple = ~95% en conditions idéales
#
#   DURÉE D'ENTRAÎNEMENT (RTX 3050, 1986 images) :
#     Phase 1 (10 epochs)  : ~2 minutes
#     Phase 2 (~30 epochs) : ~15 minutes  (early stop avant 90)
#     Total estimé         : 20-35 minutes
#
# =============================================================================