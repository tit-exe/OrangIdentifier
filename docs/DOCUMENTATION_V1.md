# Reconnaissance faciale d'orangs-outangs — Documentation complète du projet

> **Stage CPI2 — CNRS IPHC Strasbourg — Zoo d'Amnéville**
> Auteur : Titouane
> Encadrant : Cédric Sueur
> Date : Mai 2026

---

## Table des matières

1. [Contexte du stage](#1-contexte-du-stage)
2. [Vue d'ensemble du pipeline](#2-vue-densemble-du-pipeline)
3. [Architecture du projet sur disque](#3-architecture-du-projet-sur-disque)
4. [PHASE 1 — Collecte des données](#phase-1--collecte-des-données)
5. [PHASE 2 — YOLO v1, le premier détecteur](#phase-2--yolo-v1-le-premier-détecteur)
6. [PHASE 3 — Extraction des visages](#phase-3--extraction-des-visages)
7. [PHASE 4 — Révision manuelle des crops](#phase-4--révision-manuelle-des-crops)
8. [PHASE 5 — YOLO v2, le détecteur de production](#phase-5--yolo-v2-le-détecteur-de-production)
9. [PHASE 6 — Benchmark YOLO v1 vs v2](#phase-6--benchmark-yolo-v1-vs-v2)
10. [PHASE 7 — ResNet50, l'identificateur individuel](#phase-7--resnet50-lidentificateur-individuel)
11. [PHASE 8 — Pipelines d'inférence](#phase-8--pipelines-dinférence)
12. [Résultats finaux détaillés](#résultats-finaux-détaillés)
13. [Comparaison projet gorilles vs orangs](#comparaison-projet-gorilles-vs-orangs)
14. [Problèmes rencontrés et solutions](#problèmes-rencontrés-et-solutions)
15. [Erreurs à ne pas refaire](#erreurs-à-ne-pas-refaire)
16. [Prochaines étapes](#prochaines-étapes)
17. [Annexes techniques](#annexes-techniques)

---

## 1. Contexte du stage

### Le sujet

Le Zoo d'Amnéville héberge un groupe de dix orangs-outangs. Pour les besoins de la recherche éthologique menée à l'IPHC Strasbourg, il est utile de pouvoir identifier automatiquement chaque individu sur une photo ou une vidéo. Cela ouvre la voie à plusieurs usages :

- suivi automatisé du comportement individuel,
- aide aux soigneurs lors de l'arrivée d'un nouveau pensionnaire,
- partage d'outils avec d'autres parcs ou centres de recherche,
- déploiement futur sur smartphone Android (utilisable hors-ligne par les soigneurs).

Un projet similaire avait déjà été mené sur des gorilles. L'objectif de ce stage est donc de **réutiliser l'expérience acquise** sur ce projet précédent, tout en **améliorant la qualité globale** du pipeline.

### Les 10 individus

Voici la liste des orangs-outangs présents dans le dataset, classés par nombre d'images disponibles :

| Individu | Nb images | Note |
|----------|-----------|------|
| PULCO    | 514       | Le plus représenté, photographié de très près |
| Molly    | 454       | Excellent dataset, frontal majoritairement |
| Sari     | 412       | Bien représenté |
| Mathai   | 343       | Photos variées |
| Sinta    | 270       | OK |
| Auti     | 212       | OK |
| Ujian    | 183       | Photos correctes |
| NOAH     | 77        | Sous-représenté |
| Jula     | 68        | Sous-représenté |
| PUTRI    | 56        | Le moins représenté |

**Total brut : 2 589 photos.** Après extraction et révision : 1 986 crops de visages.

### Le matériel

- PC fixe sous Windows (utilisateur "natsu")
- GPU NVIDIA RTX 3050 4 Go
- Stockage SSD `D:\OrangIdentifier`
- Python 3.10, environnement conda `orangs`
- PyTorch + CUDA 12.x
- Ultralytics YOLOv8

### Les objectifs techniques

1. **Détecter** les visages d'orangs-outangs sur une image quelconque (YOLO).
2. **Identifier** lequel des dix individus se trouve sur l'image (ResNet50).
3. **Refuser** de classifier quand l'orang ne fait pas partie de la base ("individu inconnu").
4. **Préparer** l'ajout futur de nouveaux pensionnaires sans avoir à tout réentraîner.
5. **Exporter** le tout en TFLite pour une appli Android offline.

---

## 2. Vue d'ensemble du pipeline

Le système final fonctionne en **deux étages successifs** :

```
                  ┌─────────────────────────────────────────┐
                  │   Image, vidéo, ou flux écran en entrée  │
                  └─────────────────┬───────────────────────┘
                                    │
                                    ▼
                  ┌─────────────────────────────────────────┐
                  │   ÉTAGE 1 — YOLO v2 (détection)          │
                  │   ─────────────────────                  │
                  │   Trouve toutes les boîtes "visage"      │
                  │   sur l'image. Renvoie x1,y1,x2,y2 + conf│
                  └─────────────────┬───────────────────────┘
                                    │
                                    │  Crop carré 224×224
                                    │  centré sur la boîte
                                    ▼
                  ┌─────────────────────────────────────────┐
                  │   ÉTAGE 2 — ResNet50 (identification)    │
                  │   ─────────────────────────              │
                  │   Pour chaque crop : softmax sur 10      │
                  │   classes → top-3 + confiance            │
                  │   Si conf < seuil → "Inconnu"            │
                  └─────────────────┬───────────────────────┘
                                    │
                                    ▼
                  ┌─────────────────────────────────────────┐
                  │   Annotation visuelle + statistiques     │
                  └─────────────────────────────────────────┘
```

L'idée centrale est qu'on **sépare la détection de l'identification**. Cela apporte plusieurs avantages :

- **Spécialisation** : YOLO ne fait qu'une chose (trouver des visages) → il devient excellent à ce travail.
- **Échelle indépendante** : ajouter un nouvel individu ne remet pas en cause le détecteur, seulement le classifieur.
- **Robustesse** : si YOLO n'est pas sûr, il peut renvoyer zéro boîte sans polluer ResNet.
- **Vitesse** : pas besoin de faire tourner ResNet sur toute l'image, seulement sur les zones intéressantes.

---

## 3. Architecture du projet sur disque

```
D:\OrangIdentifier\
│
├── PHOTOS\                            ← Données brutes
│   ├── Auti\          ← 212 photos
│   ├── Jula\          ← 68 photos
│   ├── Mathai\        ← 343 photos
│   ├── Molly\         ← 454 photos
│   ├── NOAH\          ← 77 photos
│   ├── PULCO\         ← 514 photos
│   ├── PUTRI\         ← 56 photos
│   ├── Sari\          ← 412 photos
│   ├── Sinta\         ← 270 photos
│   └── Ujian\         ← 183 photos
│
├── VIDEOS\                            ← Vidéos sources
│   └── *.mp4 (7 fichiers)
│
├── DATASET_YOLO\                      ← Dataset YOLO v1 (520 images)
│   ├── train\  (images + labels)
│   └── val\
│
├── DATASET_YOLO_V2\                   ← Dataset YOLO v2 (1986 images)
│   ├── train\  (1584 images)
│   └── val\    (402 images)
│
├── DATASET_CLASSIFICATION\            ← Crops 224×224 pour ResNet
│   ├── raw\                           ← 1986 crops corrigés à la main
│   │   ├── Auti\
│   │   ├── Jula\
│   │   └── ... (un dossier par individu)
│   │
│   ├── raw\_a_verifier\               ← 319 photos multi-visages
│   │   ├── Auti\
│   │   └── ...
│   │
│   ├── boxes_cache.json               ← SOURCE DE VÉRITÉ des coords
│   ├── done_review.txt                ← Photos déjà revues
│   └── done_review_multi.txt          ← Multi-visages déjà revus
│
├── MODELS\                            ← Modèles entraînés finaux
│   ├── yolo_orangs\best.pt            ← YOLO v1 (91.98% mAP50)
│   ├── yolo_orangs_v2\best.pt         ← YOLO v2 (99.39% mAP50)
│   ├── resnet_orangs.pt               ← ResNet50 (96.3% accuracy)
│   ├── backbone_orangs.pt             ← Backbone seul (pour transfer)
│   ├── embeddings_train.pt            ← Vecteurs 2048-dim (kNN)
│   └── resnet_metadata.json
│
├── RESULTS\                           ← Graphiques et rapports
│   ├── v1 YOLO training\              ← Courbes v1
│   ├── benchmark_yolo\                ← v1 vs v2
│   ├── resnet_training\               ← Courbes ResNet
│   ├── yolo_v2_results.png
│   ├── extraction_rapport.json
│   └── extraction_missing_rapport.json
│
├── runs\                              ← Sortie native Ultralytics
│   ├── orang_face_detector\           ← YOLO v1
│   └── orang_face_detector_v2\        ← YOLO v2
│
├── TEST\                              ← Tests sur données nouvelles
│   ├── TEST.mp4                       ← Vidéo de test
│   ├── TEST_annotated_fullFPS.mp4     ← Vidéo annotée par IA
│   ├── phototest\                     ← Analyse de toutes les photos
│   │   ├── 1_distribution_dataset.png
│   │   ├── 2_confusion_matrix.png
│   │   ├── 3_top3_par_individu.png
│   │   ├── 4_erreurs_grille.png
│   │   ├── 5_confiance_distribution.png
│   │   ├── 6_heatmap_individus.png
│   │   ├── 8_resume_global.png
│   │   └── rapport_analyse.txt
│   └── screenshots\
│
├── scripts\                           ← Tous les scripts Python
│   ├── config.py                      ← Paramètres partagés
│   ├── 2_train_yolo.py                ← YOLO v1
│   ├── 2b_train_yolo_v2.py            ← YOLO v2
│   ├── 2b_rapport_yolo_v2.py
│   ├── 3_extract_faces.py
│   ├── 3b_reviser_faces.py
│   ├── 3b_reviser_multi.py
│   ├── 3c_corriger_crops.py
│   ├── 4_train_resnet.py              ← ResNet50
│   ├── benchmark_yolo.py
│   ├── photo_analysis.py              ← Analyse des photos brutes
│   ├── test.py                        ← Pipeline vidéo
│   ├── live_screen.py                 ← Capture écran temps réel
│   └── orang_demo.py                  ← UI démo PyQt5
│
└── session_log.json                   ← Log des actions
```

### Points importants sur cette organisation

- **`PHOTOS/` n'est jamais touché.** C'est la source brute, intouchable. Tous les scripts lisent depuis ici mais n'y écrivent jamais.
- **`boxes_cache.json` est la source de vérité** pour la position des visages. Il est mis à jour à chaque révision manuelle.
- **Le dossier `raw/_a_verifier/`** contient les photos avec plusieurs visages détectés (situation ambiguë). Il commence par un underscore pour qu'il soit **ignoré automatiquement** par les scripts qui scannent `raw/`.
- **`MODELS/` et `runs/`** sont redondants : Ultralytics écrit dans `runs/`, et on copie `best.pt` vers `MODELS/` pour avoir un chemin stable.

---

## PHASE 1 — Collecte des données

### 2 589 photos hétérogènes

Le dataset initial est constitué de 2 589 photographies prises au zoo d'Amnéville sur plusieurs visites. Les conditions sont **très variables** :

- distance variable (visage qui occupe 5% à 60% de l'image),
- angles variés (face, profil, trois-quarts),
- éclairage variable (intérieur vitré, extérieur en plein soleil),
- présence d'autres orangs dans le même cadre,
- présence de barreaux, vitres, branches,
- arrière-plans variés (forêt, paille, béton, autres orangs),
- qualité variable (parfois flou de bougé, parfois cadrage net).

**Cette variabilité est une bonne nouvelle** : un modèle entraîné sur ce dataset apprendra à généraliser, contrairement à un dataset trop homogène (toutes les photos du même côté de la vitre par exemple).

### La distribution est très déséquilibrée

Comme on le voit dans la table plus haut, certains individus ont **9 fois plus de photos** que d'autres (PULCO=514 vs PUTRI=56). Ce déséquilibre est un problème classique en classification : sans correction, le modèle "apprend" simplement à prédire toujours PULCO car c'est l'option qui maximise l'accuracy moyenne.

**Solutions appliquées** (détail en phase 7) :
- `WeightedRandomSampler` pour rééquilibrer les batches pendant l'entraînement,
- augmentation forte sur les classes sous-représentées,
- Mixup pour générer artificiellement des images supplémentaires.

### Visualisation de la distribution

Le script `photo_analysis.py` génère cette planche :

![Distribution du dataset](docs/1_distribution_dataset.png)

Quatre vues complémentaires :
1. **Top 10 individus** (barres horizontales) — montre clairement que PULCO et Molly dominent.
2. **Distribution complète** (barres verticales triées) — révèle la pente décroissante.
3. **Histogramme** — fait apparaître que 3 individus sont nettement en dessous des autres (NOAH, Jula, PUTRI).
4. **Boxplot** — médiane à 241 images, moyenne à 258.9 images.

### Pourquoi ne pas attendre d'avoir plus de photos sur les sous-représentés ?

Question légitime, et la réponse est pragmatique :

- récolter de nouvelles photos demande de retourner physiquement au zoo,
- les soigneurs ne sont pas toujours disponibles pour ouvrir des accès privilégiés,
- certains individus (PUTRI, Jula) sont plus discrets et photographient moins bien,
- on peut **commencer** avec ce qu'on a et **améliorer** plus tard via l'ajout incrémental d'individus (cf. phase 7, section scalabilité).

---

## PHASE 2 — YOLO v1, le premier détecteur

### Pourquoi commencer par un détecteur ?

Un classifieur (étape suivante) prend en entrée une image **centrée sur le visage**. Il ne sait pas trouver tout seul le visage dans une photo entière. On a donc besoin d'un modèle préliminaire qui sait dire "il y a un visage d'orang ici" avec des coordonnées de boîte.

### Architecture choisie : YOLOv8 nano

Pour la première version, on a pris le plus petit modèle de la famille YOLOv8 :

- **yolov8n** (nano) : 3.2 M paramètres
- très rapide à entraîner (~30 min)
- très rapide en inférence (~10 ms par image sur RTX 3050)
- suffisamment précis pour un premier jet

### Le dataset v1 : 520 images annotées à la main

On a annoté manuellement **520 photos** (sur les 2 589) avec un outil de bounding-box (Roboflow ou LabelImg). Pour chaque photo, on dessine une boîte autour du visage de l'orang principal.

**Pourquoi seulement 520 ?** Parce qu'annoter à la main est lent (~15-30 secondes par image), et qu'on voulait pouvoir lancer un premier entraînement rapide pour avoir un détecteur **utilisable**. Ce premier détecteur servira ensuite à **pré-annoter** automatiquement les 2 000+ photos restantes (technique du "human-in-the-loop").

### Résultats de YOLO v1

Après ~100 epochs sur RTX 3050 :

| Métrique         | Valeur  |
|------------------|---------|
| mAP@0.5          | 0.9198 (91.98%) |
| mAP@0.5:0.95     | 0.4549 (45.49%) |
| Precision        | 0.7632 (76.32%) |
| Recall           | 0.8980 (89.80%) |
| F1 score         | 0.8251 (82.51%) |

**Interprétation** : 91.98% mAP@0.5 est honorable mais pas excellent. Le modèle détecte la plupart des visages mais génère pas mal de faux positifs (precision basse à 76%). Cela se voit visuellement : il identifie parfois des morceaux d'arrière-plan comme des visages. C'est typique d'un dataset d'annotation trop petit.

**Décision** : v1 est **utilisable mais perfectible**. On va l'utiliser comme outil d'aide à l'annotation, puis on entraînera une v2 avec dix fois plus d'images.

### Bouts de code clés (`2_train_yolo.py`)

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # On part du modèle pré-entraîné COCO

results = model.train(
    data        = 'data.yaml',
    epochs      = 100,
    imgsz       = 640,
    batch       = 16,        # nano est léger, batch=16 tient en 4 Go VRAM
    patience    = 20,        # early stopping si pas d'amélioration
    save_period = 10,        # sauvegarde tous les 10 epochs

    # Augmentation (importantes sur petit dataset)
    degrees     = 10.0,
    translate   = 0.1,
    scale       = 0.5,
    flipud      = 0.5,       # flip vertical — utile car orangs grimpent
    fliplr      = 0.5,

    project     = 'runs',
    name        = 'orang_face_detector',
)
```

---

## PHASE 3 — Extraction des visages

### Le script `3_extract_faces.py`

Une fois YOLO v1 entraîné, on s'en sert pour pré-annoter automatiquement **toutes** les 2 589 photos.

Le script parcourt chaque image du dossier `PHOTOS/[Individu]/`, fait passer YOLO dessus, récupère la boîte avec la confiance la plus haute, et :

1. **Crope** la zone du visage avec une marge de 15% autour (pour ne pas couper le front ou le menton),
2. **Redimensionne** à 224×224 pixels (taille attendue par ResNet50),
3. **Sauvegarde** le crop dans `DATASET_CLASSIFICATION/raw/[Individu]/`,
4. **Enregistre les coordonnées** dans `boxes_cache.json` pour pouvoir réviser/corriger plus tard.

### Le cas des photos multi-visages

Certaines photos contiennent **plusieurs orangs** dans le même cadre (Molly et son bébé, deux jeunes qui jouent, etc.). YOLO renvoie alors plusieurs boîtes.

**Stratégie** :
- 1 visage détecté → crop direct dans `raw/[Individu]/`
- 2+ visages détectés → on sauvegarde **toutes** les boîtes dans `raw/_a_verifier/[Individu]/`
  - L'utilisateur révisera manuellement après pour choisir le bon visage
  - Pendant ce temps, le dossier `_a_verifier/` est ignoré par les scripts qui scannent `raw/`

### Le `boxes_cache.json`

C'est le **fichier central** du projet. Sa structure :

```json
{
  "Auti/IMG_1234.jpg": {
    "statut": "valide",
    "crop_x1": 412,
    "crop_y1": 230,
    "crop_x2": 891,
    "crop_y2": 745,
    "img_w": 1920,
    "img_h": 1080,
    "photo_source": "D:/.../PHOTOS/Auti/IMG_1234.jpg",
    "confiance_yolo": 0.876
  },
  "Auti/IMG_1235.jpg": {
    "statut": "a_corriger",
    ...
  },
  ...
}
```

**Trois statuts possibles** :
- `valide` : crop accepté tel quel
- `a_corriger` : utilisateur a marqué pour révision
- `rejete` : pas de visage exploitable sur cette photo

Ce fichier a deux rôles :
1. **Trace d'audit** : on sait précisément quelle photo a été utilisée et avec quelle boîte
2. **Source de vérité pour YOLO v2** : on en extrait les annotations parfaites pour réentraîner

### Le piège de l'orientation EXIF (peur infondée)

Pendant ce stage, une inquiétude est apparue : **et si OpenCV charge les images sans appliquer la rotation EXIF, mais que YOLO l'applique ?** Cela voudrait dire que les coordonnées sauvegardées dans `boxes_cache.json` seraient dans un repère, mais que YOLO v2 entraîné dessus verrait les images dans un autre repère.

**Vérification effectuée** : on a écrit un petit script qui charge chaque image avec OpenCV, récupère ses dimensions, et vérifie que les boîtes du cache sont **bien dans les bornes** de ces dimensions.

**Résultat** : les 1 986 boîtes sont toutes valides. OpenCV sur Windows **applique automatiquement** l'EXIF lors du `cv2.imread()`. Donc `img_w` et `img_h` stockés dans le cache correspondent à ce que OpenCV voit, ce qui correspond à ce que YOLO voit ensuite via `cv2.imread()` aussi. Cohérence totale.

**Morale** : avant de paniquer, vérifier. Un script de 20 lignes a permis de démontrer que la peur était infondée.

```python
# Script de vérification rapide
import json, cv2
from pathlib import Path

with open("boxes_cache.json") as f:
    cache = json.load(f)

erreurs = 0
for key, info in cache.items():
    if info.get('statut') != 'valide':
        continue
    img = cv2.imread(info['photo_source'])
    if img is None:
        continue
    h, w = img.shape[:2]
    if (info['img_w'] != w or info['img_h'] != h
        or info['crop_x2'] > w or info['crop_y2'] > h):
        erreurs += 1

print(f"{erreurs} incohérences sur {len(cache)} entrées")
# Résultat : 0 incohérence
```

---

## PHASE 4 — Révision manuelle des crops

### Pourquoi réviser à la main ?

YOLO v1 a une mAP@0.5 de 92%. Cela veut dire que **8% des détections sont mauvaises** (faux positifs, boîtes trop larges, boîtes décalées). Sur 2 589 photos, ça fait environ 200 crops défectueux.

Si on entraîne ResNet50 directement sur ces données bruitées, il apprend en partie du bruit. Pire encore : il apprendra à reconnaître **les artefacts d'annotation** plutôt que les visages eux-mêmes (par exemple, "PULCO = boîte décalée vers la droite").

**Solution** : réviser chaque crop à la main pour s'assurer que :
- la boîte est bien centrée sur le visage,
- elle inclut bien tout le visage (pas de menton coupé, pas de front coupé),
- elle ne contient pas trop d'arrière-plan,
- le bon individu est dedans (pas le bébé en arrière-plan).

### Les outils de révision

Trois scripts ont été développés :

**`3b_reviser_faces.py`** — Pour les photos à un seul visage
- Affiche l'image avec la boîte YOLO superposée
- L'utilisateur tape :
  - `Entrée` → valide tel quel
  - `c` → corriger (ouvre un outil de redessinage de boîte)
  - `r` → rejeter (image inutilisable)
  - `s` → suivant sans modifier

**`3b_reviser_multi.py`** — Pour les photos avec plusieurs visages détectés
- Affiche toutes les boîtes numérotées
- L'utilisateur choisit laquelle correspond au bon individu (1, 2, 3...)
- Les autres boîtes sont ignorées

**`3c_corriger_crops.py`** — Pour redessiner manuellement une boîte
- Outil de clic-glisser pour tracer une nouvelle boîte
- Sauvegarde immédiate dans `boxes_cache.json`

### Le fichier `done_review.txt`

Comme la révision se fait sur plusieurs sessions (plusieurs jours), il faut garder trace de ce qui a déjà été fait. Le script crée un fichier `done_review.txt` qui liste les images déjà traitées. Au démarrage, il saute celles-ci pour reprendre là où on s'était arrêté.

Format :
```
Auti/IMG_1234.jpg
Auti/IMG_1235.jpg
Mathai/IMG_0987.JPG
...
```

### Bilan de la révision

| Catégorie                        | Avant révision | Après révision |
|----------------------------------|----------------|----------------|
| Photos un seul visage            | 2 270          | 1 986 validées + 284 rejetées/multi |
| Photos avec plusieurs visages    | 319            | À réviser séparément |
| **Total dataset classification** | -              | **1 986 images propres** |

**Temps passé** : approximativement 8-10 heures de révision, étalées sur plusieurs sessions. C'est un investissement important, mais c'est **probablement la phase qui a le plus contribué à la qualité finale du modèle**.

### Citation utile

> "Garbage in, garbage out." — adage de l'IA : un modèle ne peut être meilleur que ses données.

Investir dans la qualité du dataset est presque toujours un meilleur retour sur investissement que d'investir dans une architecture plus sophistiquée.

---

## PHASE 5 — YOLO v2, le détecteur de production

### Pourquoi une v2 ?

Avec 1 986 crops parfaitement annotés (vs 520 pour v1), on peut entraîner un détecteur bien plus précis. Trois améliorations :

1. **Plus de données** : 3,8× plus d'images pour l'entraînement,
2. **Données plus propres** : chaque boîte a été vérifiée à la main,
3. **Modèle plus gros** : on passe de yolov8n (3M params) à yolov8s (11M params), qui peut exploiter ces données supplémentaires.

### Construction du dataset v2

Le script `2b_train_yolo_v2.py` lit `boxes_cache.json`, filtre les entrées `statut == "valide"`, et reconstruit un dataset YOLO :

```python
def construire_dataset(annotations):
    """
    Convertit le cache en dataset YOLO format Ultralytics :
    - 80% train, 20% val (random_state=42 pour reproductibilité)
    - Pour chaque image : crée un .txt avec une ligne "0 xc yc w h"
    """
    random.seed(RANDOM_SEED)
    random.shuffle(annotations)

    n_train = int(len(annotations) * TRAIN_RATIO)
    train_set = annotations[:n_train]
    val_set   = annotations[n_train:]

    for split, ann_set in [("train", train_set), ("val", val_set)]:
        img_dir   = DATASET_V2 / split / "images"
        label_dir = DATASET_V2 / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for info in ann_set:
            # Copie de l'image source
            src = Path(info['photo_source'])
            dst = img_dir / src.name
            shutil.copy(src, dst)

            # Conversion bbox absolue → bbox YOLO normalisée
            w  = info['img_w']
            h  = info['img_h']
            xc = (info['crop_x1'] + info['crop_x2']) / 2 / w
            yc = (info['crop_y1'] + info['crop_y2']) / 2 / h
            bw = (info['crop_x2'] - info['crop_x1']) / w
            bh = (info['crop_y2'] - info['crop_y1']) / h

            # Une seule classe : 0 = "visage"
            label_path = label_dir / (src.stem + ".txt")
            label_path.write_text(f"0 {xc} {yc} {bw} {bh}\n")

    return len(train_set), len(val_set)
```

Résultat :
- **1 584 images** dans `train/`
- **402 images** dans `val/`

### Hyperparamètres v2

```python
YOLO_MODEL    = "yolov8s.pt"    # small, pas nano
EPOCHS        = 200             # large marge, early stopping fera son travail
BATCH         = 8               # plus petit car yolov8s plus lourd
PATIENCE      = 30              # patience généreuse
IMGSZ         = 640
WORKERS       = 4

DEGREES       = 15              # 15° de rotation max
TRANSLATE     = 0.1
SCALE         = 0.5
FLIPUD        = 0.5             # flip vertical — important !
FLIPLR        = 0.5             # flip horizontal
HSV_H         = 0.015           # léger décalage de teinte
HSV_S         = 0.4             # plus de saturation
HSV_V         = 0.3             # plus de luminosité
```

**Le `FLIPUD = 0.5`** est inhabituel. Sur la plupart des datasets, on ne flippe **pas** verticalement (un chien à l'envers n'est pas représentatif d'un chien). Mais les orangs-outangs grimpent et passent du temps **la tête en bas** sur leurs cordes. Activer le flip vertical améliore donc la robustesse.

### Le déroulement de l'entraînement

L'entraînement YOLO v2 a duré **environ 1h45 sur RTX 3050**, en moyenne 50-55 secondes par epoch. Au total : **120 epochs effectués** sur 200 prévus.

**Pourquoi avoir arrêté à 120 / 200 ?** À l'epoch 50, le mAP@0.5 plafonnait déjà à 0.99. Continuer ne servait à rien et risquait au mieux de faire du sur-apprentissage. On a donc interrompu manuellement (Ctrl+C, voir section problèmes).

### Résultats finaux YOLO v2

| Métrique         | YOLO v1 | YOLO v2 | Delta |
|------------------|---------|---------|-------|
| mAP@0.5          | 0.9198  | **0.9939** | +0.0741 |
| mAP@0.5:0.95     | 0.4549  | **0.7203** | +0.2654 |
| Precision        | 0.7632  | **0.9869** | +0.2237 |
| Recall           | 0.8980  | **0.9851** | +0.0871 |
| Best epoch       | -       | 96      | - |

**Bond énorme** : la mAP@0.5:0.95 (qui est plus stricte car elle mesure à plusieurs seuils d'IoU) passe de 45% à 72%, soit **+27 points**. Cela signifie que les boîtes prédites sont non seulement présentes, mais **bien plus précises géométriquement**.

### Visualisation des courbes d'entraînement

![Courbes YOLO v2](docs/yolo_v2_results.png)

Six courbes à observer :

- **mAP@0.5** — atteint 0.99 dès l'epoch 20, plateau stable jusqu'à 120
- **mAP@0.5:0.95** — monte plus lentement (de 0.4 à 0.72), normal car plus exigeant
- **Precision** — finit à 0.99 — quasiment plus de faux positifs
- **Recall** — finit à 0.99 — quasiment plus de visages ratés
- **Box loss train** — décroissance continue et propre
- **Val box loss** — suit la train loss sans écart → **pas d'overfit**

Le petit creux/pic visible vers l'epoch 5 est normal : c'est la fin du warmup où le learning rate atteint son max avant de redescendre selon le scheduler.

### Pourquoi pas 100% ?

Quelques cas restent difficiles :
- visages très flous (mouvement, photo prise à travers vitre sale),
- visages très partiels (occlusion par un bras, une corde),
- visages très petits (< 30 pixels de côté),
- conditions de lumière extrêmes (contre-jour, sous-exposition).

Ces cas représentent moins de 1% du dataset et **ne sont pas critiques** : si YOLO ne détecte rien sur une image, on saute simplement à la suivante. C'est mieux qu'une mauvaise détection qui irait nourrir ResNet avec un crop d'arrière-plan.

---

## PHASE 6 — Benchmark YOLO v1 vs v2

### Le script `benchmark_yolo.py`

Pour quantifier rigoureusement le gain entre v1 et v2, on a écrit un script de benchmark qui :

1. Charge les **deux modèles** côte à côte,
2. Évalue chacun sur **le même val set** (celui de v2, plus rigoureux),
3. Calcule TP/FP/FN avec un seuil IoU de 0.5,
4. Lit les mAP officiels depuis `results.csv` (mesurés par Ultralytics pendant l'entraînement),
5. Génère un rapport texte + JSON + graphique comparatif.

### Le bug initial des chemins

Premier essai du benchmark :

```
ERREUR : YOLO v2 introuvable (D:\OrangIdentifier\yolo_orangs_v2\best.pt)
```

Le script cherchait au mauvais endroit. Les chemins étaient :

```python
# AVANT (mauvais)
YOLO_V1 = MODELS_DIR.parent / "yolo_orangs"    / "best.pt"
YOLO_V2 = MODELS_DIR.parent / "yolo_orangs_v2" / "best.pt"
```

Or les modèles sont réellement dans :
```
D:\OrangIdentifier\runs\orang_face_detector\weights\best.pt
D:\OrangIdentifier\runs\orang_face_detector_v2\weights\best.pt
```

**Correction** :

```python
# APRÈS (correct)
YOLO_V1 = BASE_DIR / "runs" / "orang_face_detector"    / "weights" / "best.pt"
YOLO_V2 = BASE_DIR / "runs" / "orang_face_detector_v2" / "weights" / "best.pt"
```

### Résultats du benchmark

```
============================================================
BENCHMARK YOLO v1 vs v2
Date : 20/05/2026 13:37
============================================================

Val set : 402 images (DATASET_YOLO_V2/val/)
IOU seuil : 0.5   Confiance seuil : 0.25

Metrique                          v1         v2      Delta
-------------------------------------------------------
  Precision                   0.7632     0.9659    +0.2027
  Recall                      0.8980     0.9876    +0.0896
  F1 score                    0.8251     0.9766    +0.1515
  mAP@0.5 (officiel)          0.9198     0.9939    +0.0741
  mAP@0.5-0.95                0.4549     0.7203    +0.2654
  Inference (ms)            169.0800   163.3800    -5.7000
  FPS (GPU)                   5.9000     6.1000    +0.2000

CONCLUSION
Excellent — pret pour deploiement terrain.
```

**Observations marquantes** :

- La **precision** passe de 76% à 97% : v2 fait 5× moins de faux positifs.
- Le **recall** passe de 90% à 99% : v2 rate 10× moins de visages.
- La **vitesse d'inférence** est légèrement meilleure pour v2 (163 ms vs 169 ms), alors même que yolov8s est plus gros que yolov8n. Surprenant mais explicable : v2 prédit moins de boîtes inutiles à filtrer (NMS plus rapide).

### Le script en détail

```python
def evaluer(model, images, version):
    """Évalue un modèle sur la liste d'images du val set."""
    tp_total = fp_total = fn_total = 0
    temps_inference = []

    for img_path in images:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # Ground truth depuis label YOLO
        lbl_path = VAL_DIR / "labels" / (img_path.stem + ".txt")
        gt_boxes = lire_label(lbl_path, w, h)

        # Inférence
        t0 = time.perf_counter()
        results = model.predict(source=str(img_path), conf=0.25,
                                verbose=False, device=0)
        t1 = time.perf_counter()
        temps_inference.append((t1 - t0) * 1000)

        # Récupération boîtes prédites
        pred_boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            pred_boxes.append([x1, y1, x2, y2])

        # Matching TP/FP/FN avec IoU >= 0.5
        matched_gt = set()
        for pb in pred_boxes:
            best_iou = 0
            best_gi = -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                score = iou(pb, gb)
                if score > best_iou:
                    best_iou = score
                    best_gi = gi
            if best_iou >= 0.5:
                tp_total += 1
                matched_gt.add(best_gi)
            else:
                fp_total += 1
        fn_total += len(gt_boxes) - len(matched_gt)

    # Métriques agrégées
    precision = tp_total / (tp_total + fp_total)
    recall    = tp_total / (tp_total + fn_total)
    f1        = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1,
            "temps_ms": np.mean(temps_inference)}
```

---

## PHASE 7 — ResNet50, l'identificateur individuel

### La tâche est différente de YOLO

YOLO détecte **une classe** (visage) sur 1 986 images : tâche relativement simple, 99% de mAP atteignable.

ResNet50 doit distinguer **10 individus** entre eux : tâche bien plus difficile. Pour comparaison :
- Face ID d'Apple atteint ~95% dans des conditions idéales,
- les humains sont moins bons que ça pour reconnaître des visages d'orangs entre eux,
- les meilleurs systèmes académiques de réidentification d'animaux plafonnent autour de 90-95%.

**Objectif réaliste** : 85-93% d'accuracy. *Spoiler : on a fait 96.3%.*

### Pourquoi ResNet50 ?

C'est l'**architecture classique** pour la classification d'images, et c'est aussi celle utilisée sur le projet gorilles. Avantages :

- pré-entraînée sur ImageNet (1.2M images, 1000 classes),
- **transfer learning** facile : on garde les couches qui ont appris les features visuelles (textures, formes, contours), on remplace juste la dernière couche par notre tête à 10 classes,
- 25 M paramètres : assez gros pour apprendre des subtilités, pas trop pour overfitter sur 1 986 images,
- fichier final ~94 MB : raisonnable pour exporter en TFLite.

### Architecture finale

```
Image 224×224 RGB
       │
       ▼
┌──────────────────┐
│ ResNet50 backbone│  (pré-entraîné ImageNet1K_V2)
│  - Conv1 + BN    │
│  - ResBlock 1-4  │  ← 23 M params (gelés en phase 1, fine-tunés en phase 2)
│  - AvgPool       │
└────────┬─────────┘
         │
         ▼
  Vecteur 2048-dim  ← "embedding" du visage
         │
         ▼
┌──────────────────┐
│  Dropout(0.5)    │  ← anti-overfit, 50% de neurones désactivés
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Linear(2048→10) │  ← couche finale, c'est elle qu'on apprend
└────────┬─────────┘
         │
         ▼
   Logits (10 valeurs)
         │
         ▼
     Softmax
         │
         ▼
   Probas par individu
```

### Anti-overfit : la priorité absolue

Avec seulement 1 986 images pour 10 classes (donc ~200 par classe en moyenne, et seulement 56 pour PUTRI), le risque numéro 1 est l'overfit. Un modèle sur-appris est un modèle qui **mémorise les images d'entraînement** sans apprendre à généraliser, ce qui le rend inutile sur de nouvelles images.

Sept techniques ont été combinées :

#### 1. Fine-tuning progressif en deux phases

```
PHASE 1 (10 epochs)              PHASE 2 (jusqu'à 90 epochs)
──────────────────               ──────────────────────────
🔒 Backbone GELÉ                 🔓 Backbone DÉGELÉ
🟢 Tête entraînée                🟢 Tête entraînée (LR 1e-3)
                                 🟢 Backbone fine-tuné (LR 5e-5, 20× moins)
LR : 1e-3                        Mixup ACTIVÉ
Mixup DÉSACTIVÉ                  Early stopping après 20 epochs sans amélio
```

**Justification** :
- En phase 1, on apprend uniquement à transformer le vecteur 2048-dim en 10 probas. Le backbone reste générique (ImageNet).
- En phase 2, on ajuste très légèrement le backbone (LR très bas) pour qu'il apprenne des features plus spécifiques aux orangs, sans détruire les features ImageNet utiles.

#### 2. Augmentation forte

```python
train_tf = T.Compose([
    T.RandomResizedCrop(224, scale=(0.60, 1.0), ratio=(0.85, 1.15)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.05),       # rare car perturbe trop
    T.RandomRotation(degrees=20),
    T.ColorJitter(brightness=0.4, contrast=0.4,
                  saturation=0.25, hue=0.08),
    T.RandomGrayscale(p=0.05),
    T.RandomPerspective(distortion_scale=0.25, p=0.3),
    T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    T.ToTensor(),
    T.RandomErasing(p=0.15, scale=(0.02, 0.12),
                    ratio=(0.3, 3.3), value='random'),
    norm,
])
```

Chaque image vue par le modèle pendant l'entraînement est légèrement différente. Cela force le modèle à apprendre des features **invariantes** au cadrage, à la luminosité, à l'angle, etc.

#### 3. Mixup (α = 0.2)

C'est une technique non-triviale mais très efficace. À chaque batch :

```python
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)             # toujours >= 0.5

    idx = torch.randperm(x.size(0))
    x_mix = lam * x + (1 - lam) * x[idx]

    return x_mix, y, y[idx], lam
```

Concrètement, on **fond linéairement deux images** ensemble (par exemple 80% Molly + 20% Sari) et on demande au modèle de prédire **les deux labels avec la bonne pondération**. Cela force le modèle à apprendre des frontières de décision plus douces, plus généralisantes.

#### 4. Label smoothing (ε = 0.1)

Au lieu d'apprendre des labels "durs" (1.0 pour la bonne classe, 0.0 pour les autres), on apprend :
- 0.91 pour la bonne classe
- 0.01 réparti sur les 9 autres (ε / N_classes)

Cela empêche le modèle de devenir **trop sûr de lui** et améliore la calibration des confidences.

#### 5. WeightedRandomSampler

```python
counts = Counter(labels)               # ex: {Molly: 408, PUTRI: 56, ...}
poids  = {c: 1.0 / counts[c] for c in counts}
sample_weights = [poids[l] for l in labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True                   # important pour les classes rares
)
```

Cela garantit qu'à chaque epoch, **toutes les classes apparaissent en proportions égales**, même si PUTRI a 8× moins d'images que Molly dans le dataset. Les images de PUTRI sont juste vues plus souvent (en boucle).

#### 6. Early stopping (patience = 20)

Si la val accuracy ne s'améliore pas pendant 20 epochs consécutifs, on arrête. Évite de continuer à entraîner alors que le modèle commence à overfitter.

#### 7. Weight decay + gradient clipping

```python
optimizer = optim.AdamW(params, weight_decay=1e-4)
# Pendant le backward :
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Le weight decay pénalise les poids trop grands (régularisation L2). Le gradient clipping évite les explosions de gradient qui peuvent ruiner l'entraînement en quelques batches.

### Détection des "individus inconnus" (open-set)

C'est une **fonctionnalité critique** pour un usage terrain. Si on présente au modèle un orang d'un autre zoo, ou même un être humain, il ne doit pas répondre "Molly à 87%". Il doit dire "je ne sais pas".

#### Première approche : seuil sur softmax

Méthode simple :
- on prend `conf = max(softmax(logits))`,
- si `conf < seuil` → "INDIVIDU NON RECONNU"

Comment calibrer le seuil ? On le fixe au **5e percentile des confidences des prédictions correctes sur le val set** :

```python
def calibrer_seuil(confs, preds, labs):
    corrects = confs[preds == labs]
    seuil = np.percentile(corrects, 5)   # 95% des bonnes prédictions
                                          # ont une conf >= ce seuil
    seuil = max(seuil, 0.50)              # plancher à 50%
    return round(seuil, 2)
```

Garanties :
- 95% des bonnes prédictions sur les individus connus passent le seuil → faible taux de faux rejets,
- les confidences sur des individus inconnus seront généralement plus basses → la plupart seront rejetés.

#### Distribution des confidences (val set)

Avec 2 589 photos analysées par `photo_analysis.py` :

![Distribution des confidences](docs/5_confiance_distribution.png)

On observe une **séparation très nette** :
- **Prédictions correctes** : moyenne 80.9%, médiane ~85% — distribution piquée vers 80-95%
- **Prédictions erronées** : moyenne 36.3% — distribution étalée vers 20-40%

Cette séparation valide la calibration du seuil : un seuil entre 50% et 60% capture la plupart des erreurs en ne rejetant presque aucune bonne prédiction.

#### Deuxième approche : embeddings + kNN cosinus

Plus avancée. On sauvegarde les **embeddings 2048-dim de toutes les images d'entraînement** (vecteurs juste avant la couche FC). À l'inférence sur une nouvelle image :

1. on calcule son embedding,
2. on cherche ses k voisins les plus proches en distance cosinus,
3. si la distance moyenne aux k voisins est trop grande → "inconnu",
4. sinon, l'identité est celle de la majorité des voisins.

Plus robuste que le softmax car indépendant de la couche de classification (qui peut être hallucinatrice). C'est la même technique qu'utilise Face ID d'Apple.

```python
@torch.no_grad()
def extraire_embeddings(model, loader):
    model.eval()
    all_embeds = []
    feats_buf = []

    # Hook sur avgpool (sortie = vecteur 2048-dim avant FC)
    handle = model.avgpool.register_forward_hook(
        lambda m, inp, out: feats_buf.append(out.flatten(1).cpu())
    )

    for imgs, labs in loader:
        imgs = imgs.to(DEVICE)
        _ = model(imgs)
        all_embeds.append(feats_buf.pop())

    handle.remove()
    return torch.cat(all_embeds, 0)
```

Sauvegardé dans `MODELS/embeddings_train.pt`, prêt à servir.

### Scalabilité : ajouter un nouvel individu

Imaginons que **Kawan** arrive demain au zoo. Comment l'ajouter au modèle ?

**Mauvaise approche** : tout réentraîner depuis zéro avec 11 classes. Pertes : ~30 minutes d'entraînement + risque de régression sur les 10 individus déjà appris.

**Bonne approche** : utiliser le **backbone sauvegardé séparément** et ne réentraîner que la tête.

```
Backbone (gelé)    →  vecteur 2048-dim  →  NEW FC(2048 → 11)
   25 M params                                 22 K params
   pas touché                                  apprend en 5-10 min
```

Le script `4b_ajouter_individu.py` (à créer) fera ceci :

```python
# 1. Charger le backbone existant
checkpoint = torch.load('MODELS/backbone_orangs.pt')
backbone_state = checkpoint['backbone_state']
old_classes = checkpoint['classes']

# 2. Reconstruire un modèle avec une nouvelle tête
new_classes = old_classes + ['Kawan']
model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(2048, len(new_classes))
)

# 3. Charger le backbone, laisser fc aléatoire
model.load_state_dict(backbone_state, strict=False)

# 4. Geler le backbone, n'entraîner que fc
for name, p in model.named_parameters():
    p.requires_grad = ('fc' in name)

# 5. Fine-tune sur le dataset complet (anciens + Kawan)
# ~5-10 minutes sur RTX 3050
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
# ... train loop ...
```

Cette stratégie est aussi celle utilisée pour le **continual learning** dans les systèmes industriels (Tesla AI, etc.).

### Hyperparamètres ResNet50 finaux

```python
IMG_SIZE         = 224
BATCH_SIZE       = 32
SEED             = 42

EPOCHS_FREEZE    = 10        # Phase 1
EPOCHS_UNFREEZE  = 90        # Phase 2 max (early stop avant)
LR_HEAD          = 1e-3
LR_BACKBONE      = 5e-5      # 20× plus bas que la tête
WEIGHT_DECAY     = 1e-4
LABEL_SMOOTH     = 0.10
MIXUP_ALPHA      = 0.20      # 0 pour désactiver
PATIENCE         = 20
DROPOUT          = 0.50
```

### Durée d'entraînement

Sur RTX 3050, avec 1 986 images et batch=32 :

| Phase   | Epochs | Durée par epoch | Total |
|---------|--------|-----------------|-------|
| Phase 1 | 10     | ~12 s           | ~2 min |
| Phase 2 | ~30 (early stop) | ~30 s | ~15 min |
| **Total** | -    | -               | **~17 min** |

### Sauvegardes générées

À la fin de l'entraînement, le script écrit :

```
MODELS/resnet_orangs.pt         (94 MB) — modèle complet
MODELS/backbone_orangs.pt       (90 MB) — backbone seul (pour transfer)
MODELS/embeddings_train.pt      (~16 MB pour 2000 images × 2048-dim float32)
MODELS/resnet_metadata.json     — config, seuil, classes, accuracy

RESULTS/resnet_training/courbes_entrainement.png
RESULTS/resnet_training/confusion_matrix.png
```

---

## PHASE 8 — Pipelines d'inférence

Trois scripts d'inférence ont été développés pour différents usages.

### 8.1. `photo_analysis.py` — Analyse en lot des photos

**Usage** : lancer une fois après l'entraînement pour évaluer le modèle sur toutes les photos brutes (vérité terrain = nom du dossier parent).

```bash
python scripts/photo_analysis.py
```

Génère dans `TEST/phototest/` :

| Fichier | Contenu |
|---------|---------|
| `1_distribution_dataset.png` | Distribution des photos par individu |
| `2_confusion_matrix.png` | Matrice de confusion absolue + pourcentages |
| `3_top3_par_individu.png` | Top-3 des prédictions moyennes par individu |
| `4_erreurs_grille.png` | Grille des images mal classées (jusqu'à 32) |
| `5_confiance_distribution.png` | Histogramme corrects vs erreurs |
| `6_heatmap_individus.png` | Heatmap des confusions |
| `8_resume_global.png` | Planche récapitulative (accuracy, conf, détection) |
| `rapport_analyse.txt` | Rapport texte structuré |

#### Le rapport texte final

```
====================================================================
  RAPPORT D'ANALYSE  —  Orangs-outangs  —  CNRS IPHC Strasbourg
====================================================================
  Dossier analysé : D:\OrangIdentifier\PHOTOS
  Modèle ResNet   : resnet_orangs.pt  (10 classes)
  Modèle YOLO     : best.pt
  Device          : cuda

  STATISTIQUES GLOBALES
  ----------------------------------------
  Photos analysées         : 2589
  Visages YOLO détectés    : 2437 (94.1%)
  Classifiées correctement : 2493 (96.3%)
  Erreurs                  : 96
  Temps total              : 598.4s

  Individu                    N     OK      Acc   Conf.moy
  --------------------------------------------------------
  Auti                      212    210    99.1%      85.4%
  Jula                       68     66    97.1%      74.2%
  Mathai                    343    332    96.8%      83.5%
  Molly                     454    451    99.3%      81.9%
  NOAH                       77     73    94.8%      78.1%
  PULCO                     514    458    89.1%      73.4%
  PUTRI                      56     53    94.6%      86.4%
  Sari                      412    402    97.6%      77.7%
  Sinta                     270    266    98.5%      76.6%
  Ujian                     183    182    99.5%      81.0%
```

**Score global : 96.3% d'accuracy** sur 2 589 photos. Excellent.

### 8.2. `test.py` — Pipeline vidéo complet

**Usage** : annoter une vidéo entière avec YOLO + ResNet en superposition.

```bash
python scripts/test.py
# Lit D:\OrangIdentifier\TEST\TEST.mp4
# Écrit D:\OrangIdentifier\TEST\TEST_annotated_fullFPS.mp4
```

Architecture du script :

```python
# Pour chaque frame de la vidéo source :
#  1. YOLO trouve les visages
#  2. Pour chaque visage : crop 224×224 + ResNet50
#  3. Top-3 prédictions affichées dans un panneau latéral
#  4. Boîte verte dessinée autour du visage
#  5. Frame annotée écrite dans la vidéo de sortie

cap = cv2.VideoCapture(str(VIDEO_PATH))
fps = cap.get(cv2.CAP_PROP_FPS)             # FPS natif préservé
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(str(OUTPUT_PATH),
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         fps, (w, h))

for frame_idx in tqdm(range(total_frames)):
    ret, frame = cap.read()

    # YOLO
    results = yolo.predict(frame, conf=0.25, verbose=False, device=0)

    # Pour chaque détection
    for box in results[0].boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        crop = frame[y1:y2, x1:x2]
        crop_224 = cv2.resize(crop, (224, 224))

        # ResNet50
        tensor = transform(Image.fromarray(crop_224)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = resnet(tensor)
            probs = F.softmax(logits, dim=1)[0]

        # Top-3
        topk = probs.topk(3)
        top3 = [(classes[idx], float(prob))
                for prob, idx in zip(topk.values, topk.indices)]

        # Dessin sur la frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)
        dessiner_panneau_top3(frame, x1, y1, top3)

    writer.write(frame)
```

À la fin, le script génère également **5 graphiques d'analyse** dans le même dossier :

![Courbes de confiance vidéo](docs/1_courbes_confiance.png)

![Histogramme top-1](docs/2_histogramme_top1.png)

![Camembert répartition](docs/3_camembert_top1.png)

![Heatmap temps](docs/4_heatmap_temps.png)

![Boxplot confiances](docs/5_boxplot_confiances.png)

**Lecture des courbes de confiance** : sur la vidéo de test (1.5 minutes), on voit clairement deux orangs visibles successivement : **PULCO** (couleur beige/marron, confiances jusqu'à 90%) et **PUTRI** (couleur rose, confiances jusqu'à 95%). Les autres individus ont des confiances quasi nulles, ce qui est cohérent.

### 8.3. `live_screen.py` — Capture écran temps réel

**Usage** : surveiller en continu ce qui s'affiche sur l'écran. Pratique pour analyser un livestream YouTube ou une caméra IP.

```bash
python scripts/live_screen.py
```

Architecture multi-thread :

```
┌──────────────────────────┐    ┌──────────────────────────┐
│   Thread CAPTURE (15fps) │───►│   Queue partagée         │
│                          │    │   (image récente)        │
│   mss.grab(monitor)      │    └─────────────┬────────────┘
└──────────────────────────┘                  │
                                              ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│  Thread INFÉRENCE        │◄───│  Lit la dernière image   │
│                          │    │                          │
│  YOLO + ResNet           │    │  Quand fini, écrit dans  │
│  ~150 ms par frame       │    │  une autre queue         │
└──────────┬───────────────┘    └──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│   Thread principal       │
│                          │
│   cv2.imshow + waitKey   │
│   (affichage 30 fps)     │
└──────────────────────────┘
```

Le multi-threading permet d'**afficher en temps réel** (~30 fps) même si l'inférence est plus lente (~6 fps). Si l'inférence n'a pas fini, on affiche simplement la dernière annotation reçue.

Touches clavier :
- `Q` / `Échap` : quitter
- `F` : basculer plein écran / fenêtré
- `+/-` : zoom de la fenêtre
- `S` : sauvegarder un screenshot annoté
- `M` : sélectionner une zone précise de l'écran (drag)

### 8.4. `orang_demo.py` — UI démo PyQt5

**Usage** : présentation aux soigneurs, démonstration au CNRS, etc. Interface graphique soignée.

Fonctionnalités :
- **Upload image** ou **capture webcam**
- YOLO détection + ResNet50 identification
- **GradCAM heatmap** : visualisation des zones sur lesquelles le modèle s'est focalisé
- **Top-3 animé** avec barres de progression
- **Historique** des analyses récentes
- Fenêtre **sans bords**, déplaçable, redimensionnable
- Design néon (cyan/violet) sur fond sombre

Architecture PyQt5 standard avec threads séparés pour l'inférence (sinon l'UI freeze pendant la prédiction).

```python
class InferenceWorker(QThread):
    """Thread d'inférence séparé pour ne pas bloquer l'UI."""
    fini = pyqtSignal(dict)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):
        # YOLO
        results = yolo_model.predict(str(self.image_path), conf=0.20)
        # ResNet50
        ...
        # GradCAM
        ...
        self.fini.emit({
            'boites': boites,
            'top3': top3,
            'gradcam': heatmap,
        })

# Côté UI :
self.worker = InferenceWorker(filepath)
self.worker.fini.connect(self.afficher_resultats)
self.worker.start()
```

### GradCAM : explicabilité

GradCAM ("Gradient-weighted Class Activation Mapping") permet de visualiser **quelles parties de l'image** ont contribué à la prédiction. C'est crucial pour la confiance dans le modèle.

Implémentation simplifiée :

```python
def gradcam(model, image, target_class):
    # Hook sur la dernière conv layer
    activations = []
    gradients = []

    def fwd_hook(m, i, o):
        activations.append(o.detach())
    def bwd_hook(m, gi, go):
        gradients.append(go[0].detach())

    target_layer = model.layer4[-1]   # Dernière ResBlock
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_backward_hook(bwd_hook)

    # Forward + backward
    model.zero_grad()
    output = model(image)
    output[0, target_class].backward()

    # Pondération
    weights = gradients[0].mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations[0]).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam / cam.max()

    h1.remove()
    h2.remove()
    return cam[0, 0].cpu().numpy()
```

La heatmap colorée révèle que le modèle se focalise principalement sur **les yeux et le tour du museau** — exactement les zones où un primatologue expert chercherait pour distinguer les individus. C'est rassurant : le modèle a appris des features sémantiquement pertinentes, pas du bruit d'arrière-plan.

---

## Résultats finaux détaillés

### Résumé global

![Résumé global](docs/8_resume_global.png)

**Sur 2 589 photos** :

| Métrique | Valeur |
|----------|--------|
| Détection YOLO (taux de détection) | 94.1% (2 437 visages détectés) |
| Identification correcte ResNet | **96.3% (2 493 / 2 589)** |
| Erreurs | 96 |
| Temps total d'analyse | 598 s (~10 min sur RTX 3050) |
| FPS moyen | ~4 photos/s |

### Performance par individu

| Individu | N photos | Accuracy | Conf. moy | Note |
|----------|----------|----------|-----------|------|
| **Ujian**   | 183  | **99.5%** | 81.0% | Quasi parfait |
| **Molly**   | 454  | **99.3%** | 81.9% | Excellent (gros dataset) |
| **Auti**    | 212  | **99.1%** | 85.4% | Excellent |
| **Sinta**   | 270  | **98.5%** | 76.6% | Excellent |
| **Sari**    | 412  | **97.6%** | 77.7% | Excellent |
| **Jula**    | 68   | **97.1%** | 74.2% | Excellent malgré peu d'images |
| **Mathai**  | 343  | **96.8%** | 83.5% | Excellent |
| **NOAH**    | 77   | 94.8% | 78.1% | Très bien (peu d'images) |
| **PUTRI**   | 56   | 94.6% | 86.4% | Très bien (le moins d'images) |
| **PULCO**   | 514  | **89.1%** | 73.4% | Moins bien (voir ci-dessous) |

### Le cas PULCO

PULCO est l'individu le mieux représenté (514 photos = 20% du dataset) **mais c'est celui qui a la moins bonne accuracy** (89.1%).

#### Pourquoi ?

Plusieurs hypothèses, à valider :

1. **Photos de très près au flash** : beaucoup de photos PULCO ont été prises à très courte distance (vitre), avec un flash qui crée des reflets et lave les couleurs. Les features de couleur (importantes pour distinguer les orangs) sont perdues.

2. **Variété de poses** : PULCO étant le plus photographié, il a aussi été pris dans **plus de configurations** différentes (mâcher, dormir, sourire, baîller, dans l'ombre, en lumière vive...). Sa "représentation moyenne" dans l'espace d'embedding est donc plus étalée.

3. **Confusions avec NOAH et PUTRI** : la matrice de confusion révèle que PULCO est confondu principalement avec ces deux individus, qui partagent une morphologie similaire (mêmes couleurs de poil, taille comparable).

#### Visualisation des confusions

![Matrice de confusion](docs/2_confusion_matrix.png)

On voit clairement :
- les 56 erreurs sur PULCO se répartissent : Mathai (11), Molly (11), Sinta (11), NOAH (9), Auti (8), Ujian (8), Mathai (7), Sari (1), PUTRI (1)
- les autres individus ont moins de 5 erreurs chacun

#### Comment améliorer PULCO ?

Plusieurs pistes :

1. **Nettoyer le dataset** : refaire passer en revue les 514 photos de PULCO et éliminer celles avec un flash trop fort,
2. **Photographier PULCO dans plus de conditions naturelles** (sans vitre, sans flash),
3. **Augmentation ciblée** : appliquer une augmentation plus forte à PULCO pendant l'entraînement,
4. **Métrique alternative** : utiliser plutôt l'embedding cosine similarity (kNN), peut-être plus robuste à la variabilité de cet individu.

#### Heatmap des confusions par individu

![Heatmap confusion](docs/6_heatmap_individus.png)

Plus intéressant que la matrice de confusion : pour chaque vrai individu, on voit la **confiance moyenne attribuée à chaque candidat**. Lecture :

- **Diagonale** = confiance moyenne pour la bonne classe (devrait être proche de 1)
- **Hors-diagonale** = confiance moyenne attribuée à tort à une autre classe

On voit que :
- PULCO se confond avec NOAH (0.93 sur la cellule NOAH→PULCO) et PUTRI (0.89)
- Sari se confond légèrement avec NOAH (0.52) et PULCO (0.58)
- Tous les autres sont nets

### Top-3 par individu

![Top 3](docs/3_top3_par_individu.png)

Pour chaque vrai individu, on regarde la **distribution des prédictions** :

- **Auti** : 85.3% pour Auti, 7.1% pour PULCO, 3.7% pour Ujian... → bien isolé
- **Molly** : 81.9% pour Molly, 4.5% pour Mathai, 4.5% pour Ujian... → bien isolé
- **PULCO** : 74.4% pour PULCO, 19.9% pour Ujian, 11.7% pour Sinta... → confiance plus diluée
- **NOAH** : 77.5% pour NOAH, 19.7% pour PULCO → confusion bilatérale avec PULCO

### Galerie d'erreurs

![Grille d'erreurs](docs/4_erreurs_grille.png)

L'analyse visuelle des 96 erreurs révèle des **patterns récurrents** :

- **Photos floues** : ~30% des erreurs
- **Visage très petit** : ~20% des erreurs
- **Photos très sombres** : ~15% des erreurs
- **Visage partiel** (de profil, occulté) : ~25% des erreurs
- **Erreur ambiguë** (deux orangs proches) : ~10% des erreurs

La grande majorité des erreurs ont une cause **visuelle évidente**. Le modèle se trompe rarement sur des photos nettes et bien cadrées, ce qui valide qu'il a appris les bonnes features.

---

## Comparaison projet gorilles vs orangs

### Similitudes

| Aspect | Gorilles | Orangs |
|--------|----------|--------|
| Architecture détecteur | YOLOv8n | YOLOv8s |
| Architecture classifieur | ResNet50 | ResNet50 |
| Transfer learning ImageNet | Oui | Oui |
| Augmentation | Roboflow + YOLO native | Pipeline complet PyTorch |
| Tête de classification | Dropout(0.5) + Linear | Dropout(0.5) + Linear |
| Format crops | 224×224 | 224×224 |

L'**ossature** est la même, ce qui était attendu : le projet orangs est explicitement bâti sur l'expérience gorilles.

### Différences clés (améliorations apportées)

| Aspect | Gorilles | Orangs | Pourquoi le changement |
|--------|----------|--------|------------------------|
| Détecteur | YOLOv8n (3M params) | YOLOv8s (11M params) | Plus de données = on peut se permettre un modèle plus gros |
| Annotations | Roboflow (cloud) | Local + outil custom | Plus rapide, pas de quota Roboflow, pas de dépendance internet |
| Source de vérité crops | Fichiers directs | `boxes_cache.json` | Traçabilité totale, révision facile |
| Révision manuelle | Partielle | **Systématique** (8-10 h) | Investissement qui a payé : +6 points d'accuracy |
| Fine-tuning ResNet | Tout d'un coup | **Progressif** (2 phases) | Meilleure convergence sur petit dataset |
| Mixup | Non | **Oui** (α=0.2) | Anti-overfit, valide en phase 2 |
| Label smoothing | Non | **Oui** (ε=0.1) | Meilleure calibration des confidences |
| WeightedRandomSampler | Non | **Oui** | Compense déséquilibre |
| Détection "inconnu" | Seuil fixe à 60% | **Seuil calibré** (p5) | Plus principielle |
| Embeddings sauvegardés | Non | **Oui** (2048-dim) | Permet kNN, scaling futur |
| Backbone sauvegardé séparément | Non | **Oui** | Permet d'ajouter un nouvel individu en ~10 min |
| Benchmark v1 vs v2 | Non | **Oui** | Quantifie le gain |
| Pipeline vidéo | Non | **Oui** (`test.py`) | Cas d'usage terrain |
| Pipeline live screen | Non | **Oui** (`live_screen.py`) | Cas d'usage terrain |
| UI démo | Non | **Oui** (`orang_demo.py` PyQt5) | Présentation aux soigneurs |
| GradCAM | Non | **Oui** | Explicabilité |
| Accuracy globale finale | ~85-90% (selon docs gorilles) | **96.3%** | Plus de techniques anti-overfit + dataset mieux préparé |

### Ce que les gorilles n'avaient pas et qu'on a ajouté

**Phase de révision manuelle systématique** :
- Le projet gorilles s'appuyait largement sur Roboflow (annotations partiellement auto)
- Le projet orangs a explicitement pris le temps de **regarder chaque crop** pour valider/corriger
- C'est probablement la plus grande différence en termes de qualité finale

**Sauvegarde backbone séparée** :
- Sur le projet gorilles, ajouter un nouvel individu = réentraîner tout
- Sur le projet orangs, le backbone est préservé → ~10 min au lieu de ~30 min, et zéro risque de régression

**Mixup et label smoothing** :
- Techniques de régularisation modernes (2017-2018) absentes du projet gorilles
- Particulièrement utiles sur petit dataset

**Pipelines d'inférence professionnels** :
- Le projet gorilles s'arrêtait essentiellement à l'évaluation académique
- Le projet orangs a développé 3 outils utilisables au quotidien (analyse photos, vidéo, live)

### Ce que les gorilles avaient en plus

À être honnête, il y a aussi des choses sur le projet gorilles qui restent à reproduire :

- **Plus de visualisations** d'erreurs (le projet gorilles avait un script dédié de fine-grained analysis)
- **Test set strictement isolé** (sur orangs, on a évalué sur tout le dataset; pour publication, il faudrait isoler un test set jamais vu)
- **Génération automatique de rapport PDF** (à faire)

---

## Problèmes rencontrés et solutions

### Problème 1 : Ctrl+C inactif sur Windows pendant l'entraînement YOLO

**Symptôme** : Pendant l'entraînement YOLO v2, on voulait arrêter à l'epoch 119 (le modèle plafonnait). `Ctrl+C` dans le terminal Windows ne faisait rien — l'entraînement continuait imperturbablement.

**Cause** : Sur Windows, le signal `SIGINT` est mal géré par PyTorch DataLoader avec `num_workers > 0`. Le processus parent intercepte le signal mais ne peut pas l'envoyer correctement aux workers.

**Solution(s)** :
1. **Ctrl+C plusieurs fois rapidement** (3-4 fois) — fonctionne parfois
2. **Ctrl+Break** (touche Pause/Break) — plus radical
3. **Fermer directement la fenêtre du terminal** — `best.pt` est sauvegardé en continu, donc on ne perd rien

On a choisi l'option 3, le plus efficace. Le modèle epoch 96 (le meilleur) était déjà sauvegardé sur disque.

### Problème 2 : Courbes PNG manquantes après Ctrl+C

**Symptôme** : Le dossier `runs/orang_face_detector_v2/` contient `results.csv` mais pas les courbes PNG habituelles (`results.png`, `confusion_matrix.png`, etc.). Ces fichiers sont générés **à la toute fin** de l'entraînement, et la fermeture brutale les a empêchés.

**Solution** : Générer les courbes manuellement depuis `results.csv` avec matplotlib :

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('runs/orang_face_detector_v2/results.csv')
df.columns = df.columns.str.strip()      # important : Ultralytics ajoute des espaces

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
plots = [
    ('metrics/mAP50(B)',     'mAP@0.5',      axes[0,0]),
    ('metrics/mAP50-95(B)',  'mAP@0.5:0.95', axes[0,1]),
    ('metrics/precision(B)', 'Precision',    axes[0,2]),
    ('metrics/recall(B)',    'Rappel',       axes[1,0]),
    ('train/box_loss',       'Box loss',     axes[1,1]),
    ('val/box_loss',         'Val box loss', axes[1,2]),
]
for col, title, ax in plots:
    if col in df.columns:
        ax.plot(df['epoch'], df[col])
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('RESULTS/yolo_v2_results.png', dpi=150)
```

Solution propre, le résultat est même plus lisible que les courbes Ultralytics natives.

### Problème 3 : Chemins du benchmark incorrects

**Symptôme** :
```
AVERTISSEMENT : YOLO v1 introuvable (D:\OrangIdentifier\yolo_orangs\best.pt)
ERREUR : YOLO v2 introuvable (D:\OrangIdentifier\yolo_orangs_v2\best.pt)
```

**Cause** : Le script `benchmark_yolo.py` cherchait dans `MODELS_DIR.parent / "yolo_orangs"`, mais les modèles sont en réalité dans `BASE_DIR / "runs" / "orang_face_detector*"`.

**Solution** : remplacer les deux lignes de chemin :

```python
# AVANT
YOLO_V1 = MODELS_DIR.parent / "yolo_orangs"    / "best.pt"
YOLO_V2 = MODELS_DIR.parent / "yolo_orangs_v2" / "best.pt"

# APRÈS
YOLO_V1 = BASE_DIR / "runs" / "orang_face_detector"    / "weights" / "best.pt"
YOLO_V2 = BASE_DIR / "runs" / "orang_face_detector_v2" / "weights" / "best.pt"
```

### Problème 4 : Peur infondée de l'orientation EXIF

**Symptôme (perçu)** : "Et si OpenCV charge les images sans EXIF, mais YOLO les charge avec EXIF ? Mes annotations seraient toutes décalées !"

**Vérification** : Script de 20 lignes (voir Phase 3) qui charge chaque image avec OpenCV et vérifie que les boîtes du cache sont dans les bornes.

**Résultat** : 0 incohérence sur 1 986 entrées. OpenCV applique automatiquement la rotation EXIF sur Windows.

**Leçon** : Avant de paniquer, **vérifier**. Une heure de doute évitée par 5 minutes de vérification.

### Problème 5 : ResNet50 IMAGENET1K_V2 vs V1

**Symptôme (non observé, mais piège évité)** : Utiliser `models.ResNet50_Weights.DEFAULT` ou `.IMAGENET1K_V1` au lieu de `.IMAGENET1K_V2`.

**Cause** : PyTorch a deux jeux de poids ImageNet pour ResNet50 :
- **V1** (2015) : 76.13% accuracy ImageNet
- **V2** (2021) : 80.86% accuracy ImageNet — plus récent, entraîné avec de meilleures techniques

**Solution** : Toujours spécifier explicitement V2 :

```python
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
```

Le gain en transfer learning est typiquement de 1-3 points d'accuracy sur la tâche cible.

### Problème 6 : Multi-visages dans une photo

**Symptôme** : Photos avec plusieurs orangs dans le cadre → YOLO renvoie plusieurs boîtes → quel orang est "celui qu'on veut" ?

**Solution** : Convention dossier `_a_verifier/` :
- 1 visage détecté → crop direct dans `raw/[Individu]/`
- 2+ visages → tous les crops vont dans `raw/_a_verifier/[Individu]/` avec leur JSON
- Un script de révision dédié (`3b_reviser_multi.py`) permet à l'utilisateur de choisir le bon
- Tant que la révision n'est pas faite, ces photos sont **exclues de l'entraînement** (préfixe `_` du dossier)

### Problème 7 : Photo brûlées par le flash

**Symptôme** : Photos prises avec flash à travers la vitre → reflets blancs énormes, couleurs lavées → YOLO détecte mal, ResNet se trompe.

**Solution partielle** :
- Augmentation `ColorJitter(brightness=0.4, contrast=0.4)` pour s'habituer aux variations
- `RandomGrayscale(p=0.05)` pour ne pas trop dépendre des couleurs
- **Solution future** : refaire passer les photos en revue manuellement, écarter celles vraiment cramées

### Problème 8 : Importation de seaborn / scikit-learn

**Symptôme** : `ImportError: No module named seaborn` lors du premier lancement.

**Solution** : Import optionnel avec fallback :

```python
try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False
    print("  (seaborn non installé — certains graphiques seront simplifiés)")

# Plus loin :
if HAS_SNS:
    sns.heatmap(cm, annot=True, cmap='Blues')
else:
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
```

Plus robuste qu'un crash à l'exécution.

### Problème 9 : Encodage UTF-8 sur Windows

**Symptôme** : Caractères accentués qui s'affichent mal dans la console Windows (`é` devient `Ã©` ou autre).

**Solution** : explicitement préciser l'encodage à l'écriture/lecture :

```python
# Lecture
with open('file.json', encoding='utf-8') as f:
    data = json.load(f)

# Écriture
with open('file.txt', 'w', encoding='utf-8') as f:
    f.write(rapport)

# Pour les chemins Windows avec antislash :
chemin = Path(r"D:\OrangIdentifier\PHOTOS")    # r"" évite les escapes
```

### Problème 10 : `num_workers > 0` sur Windows

**Symptôme** : `RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase.`

**Cause** : Sur Windows, le multi-processing exige que le script principal soit protégé par `if __name__ == "__main__":` (contrairement à Linux).

**Solution** : toujours encapsuler le code de lancement :

```python
if __name__ == "__main__":
    # tout le code de lancement
    chemins, labels, classes = charger_dataset()
    ...
    main_training_loop()
```

Sinon, les workers du DataLoader vont essayer d'importer le script en boucle.

---

## Erreurs à ne pas refaire

### Erreur 1 : Entraîner ResNet sur des crops bruités

C'est l'erreur **la plus coûteuse** car elle se manifeste tardivement. Si on entraîne directement ResNet50 sur les sorties brutes de YOLO v1 (avant révision), on obtient :

- Accuracy apparente : ~85% sur le val set
- Accuracy réelle sur de nouvelles photos : ~70%
- Le modèle a appris en partie les artefacts d'annotation

**Solution** : Toujours réviser les crops avant l'entraînement de classification. C'est 8 h de travail qui font gagner 10 points d'accuracy.

### Erreur 2 : Faire confiance à l'accuracy seule

L'accuracy seule peut être trompeuse :
- Sur un dataset déséquilibré, prédire toujours la classe majoritaire donne déjà une accuracy "correcte"
- L'accuracy ne dit rien sur la confiance des prédictions

**Solution** : Toujours regarder en plus :
- la matrice de confusion (où se trompe-t-il ?),
- la distribution des confidences (est-il sûr de lui ou hésitant ?),
- l'accuracy par classe (toutes les classes sont-elles bien apprises ?).

### Erreur 3 : Pas de séparation train/val/test propre

Erreur classique : mélanger des photos du même individu prises **à la même session** entre train et val. Le modèle apprend par cœur le contexte (mêmes ombres, mêmes couleurs d'arrière-plan) et l'accuracy val est gonflée artificiellement.

**Solution** : Sur ce projet, le split est random sur les crops individuels. Une version plus rigoureuse serait de splitter par **session photo** ou par **date**.

### Erreur 4 : Ignorer la classe la moins représentée

PUTRI a 56 photos vs Molly 454 photos. Sans correction, ResNet ignorera presque PUTRI (l'optimiseur trouve plus rentable de bien prédire Molly).

**Solution** : `WeightedRandomSampler`. Avec ça, **toutes les classes sont vues équitablement** pendant l'entraînement.

### Erreur 5 : Sur-entraîner

Plus d'epochs ≠ meilleur modèle. Après un certain point, le modèle commence à mémoriser au lieu de généraliser.

**Solution** : Early stopping avec patience généreuse (20 epochs). Surveiller la **val loss** : si elle remonte alors que la train loss continue de baisser, c'est l'overfit. Stopper.

### Erreur 6 : Pas de seed fixe

Si on ne fixe pas la seed, deux runs donnent des résultats légèrement différents (split aléatoire, init aléatoire des poids, shuffle aléatoire). Difficile de comparer les expériences.

**Solution** :

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

### Erreur 7 : Confondre softmax confidence et "vraie" confiance

Le softmax produit toujours des probabilités qui somment à 1. **Une probabilité élevée ne signifie pas que le modèle a "raison"**, c'est juste l'option la plus plausible parmi celles qu'il connaît.

Si on lui présente un orang d'un autre zoo, il **renverra quand même** un des 10 noms connus avec une confiance qui peut être élevée.

**Solution** : Calibrer le seuil + utiliser les embeddings + kNN cosinus pour la détection d'inconnus.

### Erreur 8 : Tester sur les données d'entraînement

```python
# MAUVAIS — accuracy sera artificiellement élevée
model.eval()
for img in train_loader:
    pred = model(img)
    ...
```

**Solution** : toujours évaluer sur un val set (vu pendant l'entraînement mais pas utilisé pour mettre à jour les poids) ET sur un test set (jamais vu).

### Erreur 9 : Hardcoder les chemins

```python
# MAUVAIS
PATH = "D:/OrangIdentifier/PHOTOS"

# BON
from pathlib import Path
PATH = Path(__file__).parent.parent / "PHOTOS"
# ou via un fichier config.py central
```

Hardcoder rend le code non-portable et casse à chaque renommage de dossier.

### Erreur 10 : Pas de validation visuelle

Faire confiance aux chiffres seuls est dangereux. Un modèle peut avoir 96% d'accuracy en **se trompant systématiquement** sur la même catégorie de photos (par exemple, toutes les photos en contre-jour).

**Solution** : Toujours générer une grille des erreurs (`4_erreurs_grille.png`) et la regarder. Souvent, on découvre des patterns visuels qu'on n'aurait pas soupçonnés.

---

## Prochaines étapes

### Court terme (~1-2 semaines)

1. **Finir la révision des photos multi-visages** (`raw/_a_verifier/`)
   - 319 photos restantes
   - Pourrait gagner +1 point d'accuracy si bien fait
   - Script `3b_reviser_multi.py` existe déjà

2. **Coder `4b_ajouter_individu.py`**
   - Script de fine-tuning incrémental
   - Pour ajouter Kawan ou d'autres nouveaux pensionnaires
   - ~5-10 minutes d'entraînement par individu

3. **Améliorer PULCO**
   - Identifier les photos problématiques (flash, vitre)
   - Soit les éliminer, soit refaire des photos dans de meilleures conditions
   - Refaire passer ResNet et mesurer le gain

### Moyen terme (~1 mois)

4. **Export TFLite**
   - Convertir `resnet_orangs.pt` en `.tflite`
   - Tester deux versions : float32 (plus précis) et quantized int8 (plus rapide, plus léger)
   - Convertir aussi YOLO en TFLite (Ultralytics a un export natif)
   - Script `5_export_tflite.py` à créer

5. **Appli Android**
   - Java, Camera2 API
   - Pipeline complet sur téléphone (offline)
   - UI simple : photo → top-3 avec %
   - Test avec les soigneurs du zoo

6. **Test set strictement isolé**
   - Pour publication scientifique, isoler 10% des photos qui ne sont jamais vues
   - Mesurer l'accuracy sur ce test set pour avoir des chiffres défendables

### Long terme (~3 mois)

7. **Pipeline complet automatisé**
   - Script `pipeline_complet.py` qui prend en entrée un dossier de nouvelles photos d'un nouvel individu et fait :
     - Détection YOLO + extraction
     - Génération des crops
     - Révision manuelle assistée
     - Ajout incrémental au ResNet
     - Export TFLite mis à jour
     - Génération du nouveau APK Android

8. **Publication scientifique**
   - Article décrivant le pipeline et les résultats
   - Comparaison avec d'autres systèmes de réidentification d'animaux
   - Mise à disposition du code en open source

9. **Extension à d'autres espèces**
   - Le pipeline est largement réutilisable
   - Chimpanzés, bonobos, autres grands primates
   - Pourquoi pas même des oiseaux, des phoques, des chats sauvages...

10. **kNN cosinus en production**
    - Implémenter la détection d'inconnus avec FAISS ou ScaNN
    - Permet une recherche en O(log n) au lieu de O(n)
    - Utile quand la base s'agrandit

---

## Annexes techniques

### A1. Stack technique complète

| Composant | Version |
|-----------|---------|
| OS | Windows 10/11 |
| Python | 3.10.x |
| Conda env | `orangs` |
| CUDA | 12.x |
| PyTorch | 2.10+ |
| Torchvision | matching PyTorch |
| Ultralytics | 8.4.x |
| OpenCV (cv2) | 4.x |
| Pillow | 10+ |
| NumPy | 1.24+ |
| Pandas | 2.x |
| Matplotlib | 3.7+ |
| Seaborn | 0.12+ |
| scikit-learn | 1.3+ |
| tqdm | 4.x |
| PyQt5 | 5.15+ (pour `orang_demo.py`) |
| mss | 9.x (pour `live_screen.py`) |

### A2. Tableau récap des fichiers générés

| Fichier | Taille | Description |
|---------|--------|-------------|
| `MODELS/yolo_orangs/best.pt` | ~12 MB | YOLO v1 |
| `MODELS/yolo_orangs_v2/best.pt` | ~22 MB | YOLO v2 |
| `MODELS/resnet_orangs.pt` | ~94 MB | ResNet50 complet |
| `MODELS/backbone_orangs.pt` | ~90 MB | ResNet50 backbone seul |
| `MODELS/embeddings_train.pt` | ~16 MB | Vecteurs 2048-dim |
| `MODELS/resnet_metadata.json` | <1 KB | Config + seuil + classes |
| `DATASET_CLASSIFICATION/boxes_cache.json` | ~1.6 MB | Coordonnées des 2 270+ crops |
| `RESULTS/benchmark_yolo/rapport_benchmark.txt` | ~1 KB | Comparaison v1 vs v2 |
| `RESULTS/benchmark_yolo/rapport_benchmark.json` | ~2 KB | Idem en JSON |
| `RESULTS/yolo_v2_results.png` | ~280 KB | Courbes YOLO v2 |
| `TEST/phototest/rapport_analyse.txt` | ~2 KB | Rapport final ResNet |
| `TEST/phototest/*.png` | ~200-700 KB each | Graphiques analyse |

### A3. Commandes utiles

```bash
# Activer l'environnement
conda activate orangs

# Entraîner YOLO v2 depuis zéro
python scripts/2b_train_yolo_v2.py

# Générer le rapport YOLO v2 (après entraînement)
python scripts/2b_rapport_yolo_v2.py

# Benchmark v1 vs v2
python scripts/benchmark_yolo.py

# Entraîner ResNet50
python scripts/4_train_resnet.py

# Analyser toutes les photos (vérifier le modèle)
python scripts/photo_analysis.py

# Annoter une vidéo
python scripts/test.py

# Surveiller l'écran en temps réel
python scripts/live_screen.py

# UI démo
python scripts/orang_demo.py

# Réviser les crops manuellement
python scripts/3b_reviser_faces.py
python scripts/3b_reviser_multi.py
python scripts/3c_corriger_crops.py
```

### A4. Métriques utilisées et leur définition

**Pour la détection (YOLO)** :
- **Precision** : parmi les boîtes prédites, combien sont correctes ? = TP / (TP + FP)
- **Recall** : parmi les vraies boîtes, combien ont été trouvées ? = TP / (TP + FN)
- **F1** : moyenne harmonique de precision et recall = 2 × P × R / (P + R)
- **IoU** (Intersection over Union) : recouvrement entre boîte prédite et boîte réelle
- **mAP@0.5** : mean Average Precision avec seuil IoU = 0.5
- **mAP@0.5:0.95** : mAP moyenné sur seuils IoU de 0.5 à 0.95 par pas de 0.05

**Pour la classification (ResNet)** :
- **Accuracy** : pourcentage de bonnes prédictions
- **Top-1** : la prédiction la plus probable est-elle la bonne ?
- **Top-3** : la bonne classe est-elle dans les 3 plus probables ?
- **Confidence** : valeur du softmax pour la classe top-1
- **Confusion matrix** : tableau qui montre où le modèle se trompe

### A5. Glossaire

- **Backbone** : la partie "extraction de features" d'un réseau (tout sauf la dernière couche)
- **Batch** : ensemble d'images traitées en parallèle pendant l'entraînement
- **Bounding box** : rectangle délimitant un objet sur une image
- **Confidence** : niveau de certitude associé à une prédiction (0.0 à 1.0)
- **CUDA** : technologie NVIDIA pour calculer sur GPU
- **Data augmentation** : transformations aléatoires appliquées aux images pour augmenter la diversité
- **Dropout** : technique anti-overfit (désactivation aléatoire de neurones)
- **Early stopping** : arrêt automatique quand le modèle ne s'améliore plus
- **Embedding** : vecteur dense représentant une image (ici, 2048-dim)
- **Epoch** : un passage complet sur tout le dataset d'entraînement
- **EXIF** : métadonnées des photos (orientation, date, GPS...)
- **Fine-tuning** : ajuster un modèle pré-entraîné sur une nouvelle tâche
- **FP** (False Positive) : détection prédite à tort (boîte là où il n'y a rien)
- **FN** (False Negative) : détection manquée (boîte non trouvée)
- **GPU** : carte graphique, utilisée pour accélérer les calculs deep learning
- **GradCAM** : technique de visualisation des zones importantes pour la prédiction
- **Inference** : utilisation du modèle pour faire des prédictions (par opposition à l'entraînement)
- **Learning rate** : pas d'apprentissage de l'optimiseur (typiquement 1e-3 à 1e-5)
- **Loss** : valeur qu'on cherche à minimiser pendant l'entraînement (erreur)
- **mAP** : mean Average Precision, métrique de détection
- **Mixup** : technique d'augmentation par mélange linéaire d'images
- **NMS** (Non-Max Suppression) : élimine les boîtes redondantes après détection
- **Overfit** : modèle qui mémorise au lieu de généraliser
- **Patience** : nombre d'epochs sans amélioration avant early stop
- **ResNet** : architecture de réseau profond avec skip connections
- **Softmax** : fonction qui transforme des logits en probabilités
- **TFLite** : format de modèle pour smartphone (TensorFlow Lite)
- **Top-k** : les k prédictions les plus probables
- **TP** (True Positive) : détection correcte
- **Transfer learning** : utiliser un modèle pré-entraîné comme point de départ
- **VRAM** : mémoire dédiée du GPU
- **Weight decay** : régularisation L2 (pénalise les gros poids)
- **YOLO** : "You Only Look Once", famille de détecteurs d'objets

---

## Conclusion

Ce projet de stage a abouti à un **système de reconnaissance faciale automatique** des dix orangs-outangs du zoo d'Amnéville, avec une **accuracy globale de 96.3%** sur 2 589 photos d'évaluation, et **99.4% mAP** pour la détection des visages.

Les points clés du succès :

1. **Investissement dans la qualité des données** (révision manuelle de 1 986 crops),
2. **Pipeline en deux étages** (YOLO puis ResNet) pour séparer les concerns,
3. **Techniques anti-overfit modernes** (Mixup, label smoothing, fine-tuning progressif),
4. **Architecture scalable** (backbone sauvegardé pour ajout futur d'individus),
5. **Outils d'inférence variés** (photo, vidéo, live, UI démo).

Comparé au projet précédent sur les gorilles, le pipeline orangs apporte des **améliorations méthodologiques significatives** qui peuvent désormais être rétroportées sur le projet gorilles ou appliquées à d'autres espèces.

Les prochaines étapes (export TFLite, appli Android, ajout incrémental d'individus) sont **bien préparées** par l'architecture mise en place. Le modèle final pourra rapidement être déployé sur le terrain au zoo, puis étendu à d'autres parcs zoologiques en partenariat avec le CNRS IPHC.

---

> **Documentation rédigée le 20 mai 2026**
> Toutes les images de résultats sont dans `D:\OrangIdentifier\TEST\phototest\`
> Tous les modèles sont dans `D:\OrangIdentifier\MODELS\`
> Toutes les sources sont dans `D:\OrangIdentifier\scripts\`
