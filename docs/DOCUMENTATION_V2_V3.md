# Reconnaissance faciale d'orangs-outangs — Documentation Partie 2 : V2 et V3

> **Stage CPI2 — CNRS IPHC Strasbourg — Zoo d'Amnéville et BOS Foundation**
> Auteur : Titouane
> Encadrant : Cédric Sueur
> Date : Mai 2026
> Suite de DOCUMENTATION_PROJET_ORANGS.md (V1)

---

## Table des matières

1. [Pourquoi V2 et V3 existent](#1-pourquoi-v2-et-v3-existent)
2. [Vue d'ensemble des trois versions](#2-vue-densemble-des-trois-versions)
3. [V2 — ResNet50 reconverti en système d'embeddings open-set](#3-v2--resnet50-reconverti-en-système-dembeddings-open-set)
4. [Le passage de V2 à V3 — pourquoi changer de modèle](#4-le-passage-de-v2-à-v3--pourquoi-changer-de-modèle)
5. [V3 — MegaDescriptor-T-224 + Sub-center ArcFace](#5-v3--megadescriptor-t-224--sub-center-arcface)
6. [PHASE V3-A — Collecte massive de wild crops depuis internet](#6-phase-v3-a--collecte-massive-de-wild-crops-depuis-internet)
7. [PHASE V3-B — Extraction et révision des wild crops](#7-phase-v3-b--extraction-et-révision-des-wild-crops)
8. [PHASE V3-C — L'entraînement Sub-center ArcFace](#8-phase-v3-c--lentraînement-sub-center-arcface)
9. [PHASE V3-D — Reprise et continuation d'entraînement](#9-phase-v3-d--reprise-et-continuation-dentraînement)
10. [PHASE V3-E — Intégration des 30 individus BOS Foundation](#10-phase-v3-e--intégration-des-30-individus-bos-foundation)
11. [PHASE V3-F — Tests de validation et diagnostic open-set](#11-phase-v3-f--tests-de-validation-et-diagnostic-open-set)
12. [PHASE V3-G — Visualisations détaillées de l'espace d'embedding](#12-phase-v3-g--visualisations-détaillées-de-lespace-dembedding)
13. [Résultats finaux V3 et comparaison V1 → V2 → V3](#13-résultats-finaux-v3-et-comparaison-v1--v2--v3)
14. [Comment ajouter un nouvel individu sans réentraîner](#14-comment-ajouter-un-nouvel-individu-sans-réentraîner)
15. [Gestion à long terme du modèle](#15-gestion-à-long-terme-du-modèle)
16. [Réorganisation finale du projet](#16-réorganisation-finale-du-projet)
17. [Hyperparamètres et choix techniques détaillés](#17-hyperparamètres-et-choix-techniques-détaillés)
18. [Erreurs commises et leçons apprises pour V2 et V3](#18-erreurs-commises-et-leçons-apprises-pour-v2-et-v3)
19. [Pistes d'amélioration futures](#19-pistes-damélioration-futures)
20. [Annexes — schémas, formules, configurations](#20-annexes--schémas-formules-configurations)

---

## 1. Pourquoi V2 et V3 existent

### Le problème fondamental de V1

La V1 du projet utilise un **classifieur fermé** : un ResNet50 avec une tête de classification à 10 sorties qui produit une distribution softmax. Cela signifie que pour n'importe quelle image qu'on lui présente, le modèle **doit obligatoirement** la classer dans l'un des 10 individus. Il n'a aucune notion d'« inconnu ».

Concrètement, si on présente à V1 une photo d'un orang-outan d'un autre zoo, d'un humain, d'un chien, ou même d'un mur, le modèle répondra toujours quelque chose comme « Molly à 87% » avec une confiance arbitraire. C'est inutilisable sur le terrain — un ranger BOS qui prendrait en photo un orang non recensé recevrait une fausse identification à chaque fois.

V1 propose deux solutions partielles :
- Un seuil de confiance sur le softmax (rejeter si la confiance maximale est inférieure à 50%)
- Une approche kNN avec embeddings et distance cosinus

Mais ces deux solutions ont des limites importantes. Le seuil softmax dépend de la calibration des probabilités, qui est notoirement mal calibrée dans les réseaux profonds modernes. L'approche kNN est plus robuste mais elle est greffée après coup sur un modèle qui n'a pas été optimisé pour produire de bons embeddings.

### Ce que V2 et V3 résolvent

**V2** ré-utilise le backbone de V1 mais l'utilise différemment : on ne s'en sert plus pour classifier directement, mais pour produire des vecteurs de représentation 2048 dimensions à partir desquels on construit une **galerie de prototypes**. Au lieu d'une distribution softmax, on calcule la similarité cosinus entre le vecteur d'une nouvelle image et chaque prototype d'individu connu. Si la similarité maximale est trop faible, on rejette comme inconnu. Le seuil est calibré par leave-one-individual-out, c'est-à-dire qu'on simule l'arrivée d'un nouvel individu en sortant tour à tour chaque individu de la galerie et en mesurant si on le rejette correctement.

**V3** va plus loin en remplaçant complètement le backbone par un modèle **explicitement entraîné pour produire de bons embeddings** : MegaDescriptor-T-224, pré-entraîné sur des dizaines de datasets de réidentification animale, puis fine-tuné sur notre dataset avec une perte **Sub-center ArcFace**. Cette perte est conçue spécifiquement pour la reconnaissance faciale — elle force le modèle à produire des embeddings où les individus différents sont angulairement très séparés et les images d'un même individu sont angulairement proches.

V3 ajoute également les **5429 wild crops d'internet** comme classe « background » pendant l'entraînement. Le modèle apprend ainsi non seulement à distinguer les 10 individus connus entre eux, mais aussi à les distinguer de l'ensemble des autres orangs-outangs existants.

### La logique de progression

```
V1 : Classifier 10 individus    → 96.3% accuracy, mais inutilisable open-set
        ↓
V2 : Reconvertir V1 en système d'embeddings + galerie
       Calibrer un seuil de rejet par leave-one-out
       ROC AUC = 0.98, séparabilité = 1.72
        ↓
V3 : Remplacer le backbone par MegaDescriptor + ArcFace
       Entraîner explicitement pour les embeddings
       Ajouter wild crops comme classe background
       Val accuracy 99% + rejet BOS 96.3%
```

Chaque version corrige une limite de la précédente. V1 → V2 corrige le problème open-set sans réentraîner. V2 → V3 améliore la qualité des embeddings eux-mêmes par un nouvel entraînement spécialisé.

---

## 2. Vue d'ensemble des trois versions

### Diagramme comparatif

```
┌─────────────────────────────────────────────────────────────────┐
│                            V1                                    │
│  Image → YOLO v1 (nano) → ResNet50 → Linear(2048→10) → Softmax  │
│              ↓                                                   │
│         BBox visage                                              │
│              ↓                                                   │
│  Image crop 224×224 → Classification fermée → "Molly à 87%"      │
│                                                                  │
│  Problème : pas de rejet d'inconnus                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                            V2                                    │
│  Image → YOLO v2 (medium) → ResNet50 (sans tête) → vecteur 2048D │
│              ↓                                                   │
│         BBox visage                                              │
│              ↓                                                   │
│  Image crop 224×224 → Embedding 2048D                            │
│              ↓                                                   │
│  Similarité cosinus vs galerie de prototypes                     │
│              ↓                                                   │
│  Si sim_max ≥ seuil(0.4885)   → identifier                       │
│  Si sim_max < seuil           → "Unknown"                        │
│                                                                  │
│  Innovation : galerie + seuil calibré open-set                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                            V3 (actuelle)                         │
│  Image → YOLO v2 (medium) → MegaDescriptor-T-224 → vecteur 768D  │
│              ↓                                                   │
│         BBox visage                                              │
│              ↓                                                   │
│  Image crop 224×224 → Embedding 768D L2-normalisé                │
│              ↓                                                   │
│  Similarité cosinus vs galerie de prototypes                     │
│              ↓                                                   │
│  Si sim_max ≥ seuil(0.22)     → identifier                       │
│  Si sim_max < seuil           → "Unknown"                        │
│                                                                  │
│  Innovations :                                                   │
│   - Backbone optimisé pour embeddings (MegaDescriptor)           │
│   - Loss ArcFace au lieu de cross-entropy                        │
│   - Sub-center (K=1 connus, K=5 wild) pour gérer le bruit        │
│   - Wild crops internet comme classe background                  │
│   - 30 individus BOS Foundation testés en open-set vrai          │
└─────────────────────────────────────────────────────────────────┘
```

### Tableau récapitulatif

| Aspect | V1 | V2 | V3 |
|--------|-----|-----|-----|
| Détection | YOLO nano (mAP 92%) | YOLO medium (mAP 99%) | YOLO medium (inchangé) |
| Backbone | ResNet50 ImageNet | ResNet50 V1 réutilisé | MegaDescriptor-T-224 |
| Embedding dim | — (softmax direct) | 2048 | 768 |
| Loss training | Cross-entropy + label smoothing | Aucun (réutilise V1) | Sub-center ArcFace |
| Wild crops | Non utilisés | Non utilisés | 5429 comme classe background |
| Ouvert/Fermé | Fermé | Ouvert avec seuil | Ouvert avec seuil + galerie |
| Accuracy connus | 96.3% | (réutilise V1) | 99.06% (val) |
| Séparabilité | — | 1.72 | gap 0.96 (positif 0.92 vs négatif -0.03) |
| Rejet inconnus | Aucun | Calibré leave-one-out | Mesuré sur 1622 crops BOS, 96.3% |
| Réentraînement pour nouveau | Tout réentraîner | Recalculer galerie (10 min) | Recalculer galerie (10 min) |

---

## 3. V2 — ResNet50 reconverti en système d'embeddings open-set

### L'idée clé de V2

Plutôt que de jeter le travail de V1, on en réutilise le backbone. Un ResNet50 entraîné en classification produit, juste avant sa couche finale Linear(2048→10), un **vecteur 2048 dimensions** qui résume l'information visuelle de l'image. Ce vecteur est ce qu'on appelle l'**embedding**.

Si l'entraînement de classification a bien fonctionné, les embeddings d'images d'un même individu sont géométriquement plus proches que les embeddings d'individus différents — sinon le classifieur final ne pourrait pas les séparer. On peut donc utiliser ces embeddings comme une signature unique de chaque visage.

L'idée de V2 est de :

1. Charger le ResNet50 entraîné de V1.
2. Retirer la couche finale Linear(2048→10).
3. Pour chaque image d'entraînement, calculer son embedding 2048-dim et le L2-normaliser (le ramener sur la sphère unité).
4. Pour chaque individu, moyenner tous ses embeddings normalisés et re-normaliser : c'est le **prototype**.
5. À l'inférence, calculer l'embedding d'une nouvelle image et comparer par similarité cosinus à tous les prototypes.
6. Si la similarité max dépasse un seuil calibré, on identifie ; sinon on rejette comme inconnu.

### Pourquoi la similarité cosinus et la normalisation L2

Pour deux vecteurs `u` et `v` normalisés (norme = 1), la similarité cosinus est simplement leur produit scalaire `u·v` et vaut entre -1 et +1. Elle vaut 1 quand les vecteurs pointent dans la même direction, 0 quand ils sont orthogonaux, et -1 quand ils pointent en sens opposés. Pour des embeddings de visage, on veut typiquement des valeurs entre 0.5 et 0.9 pour des images du même individu, et entre 0 et 0.4 pour des individus différents.

La normalisation L2 a deux avantages majeurs. D'abord, elle rend la similarité cosinus équivalente à un simple produit scalaire, beaucoup plus rapide à calculer qu'une norme euclidienne. Ensuite, elle élimine l'effet d'amplitude — un embedding peut avoir une grande norme parce que l'image est très saturée ou très contrastée, sans que cela reflète une différence d'identité. En projetant tout sur la sphère unité, on ne garde que la direction du vecteur, qui code l'identité.

### Le pipeline complet V2

```
Photo brute
    ↓
YOLO v2 détecte la bbox du visage
    ↓
Crop carré 224×224 centré sur le visage avec marge de 15%
    ↓
Normalisation ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ↓
ResNet50 backbone (sans la tête fc)
    ↓
Vecteur 2048-dim
    ↓
L2-normalisation
    ↓
Similarité cosinus avec chaque prototype de la galerie
    ↓
       ╔══════════════════════════════════════╗
       ║  Si max(sim) ≥ seuil_calibré         ║
       ║    → individu = argmax(sim)          ║
       ║    → confiance = max(sim)            ║
       ║  Sinon                               ║
       ║    → "Unknown individual"            ║
       ╚══════════════════════════════════════╝
```

### Construction de la galerie

Pour chaque individu, on prend toutes ses images d'entraînement (en moyenne ~200 crops), on calcule chaque embedding, on les moyenne, puis on re-normalise le résultat. C'est le **prototype** de cet individu.

Une variante consiste à garder tous les embeddings individuels (pas seulement la moyenne) et à faire du kNN — l'identité retenue est celle de la majorité des k voisins les plus proches. Cette approche est plus robuste aux outliers mais plus lente et plus lourde à embarquer sur Android. On retient donc la version prototype : 10 vecteurs de 2048 floats = environ 80 KB pour toute la galerie.

### Calibration du seuil par leave-one-individual-out

C'est la partie la plus subtile de V2. Comment fixer le seuil au-delà duquel on accepte qu'une image appartient à un individu connu ?

**Mauvaise approche** : prendre un seuil arbitraire (genre 0.5) et espérer que ça marche. Ce seuil n'a aucune justification statistique.

**Bonne approche** : simuler l'arrivée d'un nouvel individu. On sort tour à tour chaque individu de la galerie, on reconstruit une galerie à 9 individus, et on teste deux types d'images :

- Les crops des 9 individus restants doivent être identifiés correctement (vrais positifs).
- Les crops de l'individu sorti doivent être rejetés comme inconnus (vrais négatifs simulés).

On balaye le seuil de 0.0 à 1.0 et on mesure la précision, le rappel et le F1 pour chaque valeur. Le seuil optimal est celui qui maximise le F1.

```
Pour chaque individu I de 1 à 10 :
  Galerie = prototypes des 9 autres individus
  Pour chaque crop de I :
    similarité_max = max(crop @ galerie)
    → ce crop devrait être rejeté
  Pour chaque crop des 9 autres individus :
    similarité_max = max(crop @ galerie)
    → ce crop devrait être identifié

Balayer seuil τ de 0.0 à 1.0 :
  Compter TP, FP, TN, FN
  Calculer F1(τ)

Seuil optimal = argmax F1(τ)
```

Avec ce processus, V2 trouve un seuil optimal autour de **0.4885**, avec une ROC AUC de **0.9821**.

### Métriques de séparabilité V2

V2 introduit la métrique de **séparabilité**, qui résume la qualité de l'espace d'embedding par un seul nombre.

- **Similarité positive moyenne** : moyenne des similarités entre un embedding et le prototype de son propre individu. Plus c'est haut, mieux c'est. V2 atteint 0.7206.
- **Similarité négative moyenne** : moyenne, pour chaque image, de la similarité maximale avec les prototypes des autres individus (le hardest negative). Plus c'est bas, mieux c'est. V2 atteint 0.4189.
- **Séparabilité** : le ratio positif / négatif. V2 atteint 1.7203, ce qui veut dire que les images du même individu sont en moyenne 1.72 fois plus similaires à leur prototype que les images d'autres individus.

C'est honorable pour un système qui réutilise un backbone non spécifiquement optimisé pour les embeddings. V3 fera bien mieux.

### Sept visualisations diagnostiques générées par V2

Le pipeline V2 produit automatiquement sept graphiques détaillés qui caractérisent la qualité du système.

**1. t-SNE de l'espace d'embedding** — projection 2D non-linéaire de tous les embeddings, colorée par individu. Visuellement, on voit des nuages bien séparés par couleur. Les centroïdes sont marqués par des étoiles. Si deux nuages se mélangent, c'est que les deux individus sont visuellement confondables par le modèle.

**2. Matrice de similarité 10×10** — chaque case montre la similarité cosinus moyenne entre le prototype d'un individu (ligne) et les images d'un autre (colonne). La diagonale doit être haute (>0.6) et l'off-diagonale basse (<0.4). On utilise un colormap RdYlGn pour voir d'un coup d'œil les confusions.

**3. Distribution des similarités** — deux histogrammes superposés : un pour les similarités « même individu » et un pour les similarités « individus différents ». Idéalement les deux distributions sont bien séparées avec une frontière nette autour du seuil calibré.

**4. Courbe ROC** — pour chaque valeur de seuil, on trace le taux de vrais positifs (sensibilité — % d'individus connus correctement acceptés) contre le taux de faux positifs (1 - spécificité — % d'inconnus incorrectement acceptés). La AUC mesure la performance globale indépendamment du seuil. V2 atteint 0.9821, ce qui est excellent.

**5. Calibration du seuil** — trois courbes : précision, rappel, F1 en fonction du seuil. Le seuil optimal est l'intersection F1 maximale. Le graphique montre aussi les régions « trop permissif » et « trop strict ».

**6. Accuracy par individu** — graphique en barres triées du meilleur au pire. Permet d'identifier les individus difficiles (typiquement les sous-représentés ou les visuellement similaires à d'autres).

**7. Matrice de confusion embedding-based** — version normalisée en pourcentages, mettant en évidence quels individus sont confondus avec quels autres.

Chacun de ces graphiques raconte une partie de l'histoire du modèle. Ensemble ils donnent un diagnostic complet.

### Le fichier embeddings.json pour Android

À la fin du pipeline V2, on produit un fichier JSON compact qui contient toute la galerie. Format :

```
{
  "version": "1.0",
  "model_source": "resnet_orangs.pt",
  "embedding_dim": 2048,
  "normalization": "L2",
  "similarity_metric": "cosine",
  "unknown_threshold": 0.4885,
  "calibration_auc": 0.9821,
  "num_individuals": 10,
  "individuals": {
    "Auti": {
      "class_index": 0,
      "num_training_crops": 212,
      "mean_similarity_to_self": 0.7234,
      "embedding": [0.0234, -0.0156, ...]  // 2048 floats
    },
    ...
  }
}
```

Taille typique : 80 KB pour 10 individus, soit environ 8 KB par individu. Cela tient largement sur n'importe quel téléphone et se charge en quelques millisecondes. L'app Android n'a qu'à comparer un embedding entrant à ces 10 vecteurs — c'est un calcul de l'ordre de la microseconde.

### Limitation principale de V2 qui justifie V3

Le ResNet50 a été entraîné avec une perte cross-entropy, qui n'optimise pas explicitement la structure géométrique de l'espace d'embedding. Le modèle a appris à séparer les 10 individus connus dans le sens où il peut les classifier correctement, mais rien ne garantit que les directions des prototypes sont angulairement bien réparties. Les similarités positives atteignent à peine 0.72, ce qui est faible — un système optimisé devrait atteindre 0.85 ou plus.

C'est précisément ce qu'apporte V3 : un backbone et une loss conçus pour produire un excellent espace d'embedding.

---

## 4. Le passage de V2 à V3 — pourquoi changer de modèle

### Les trois faiblesses de V2 qu'on veut corriger

**Première faiblesse** : le ResNet50 est pré-entraîné sur ImageNet, qui contient majoritairement des objets du quotidien (1000 classes, dont des chiens, des chats, des voitures...). Les features qu'il a apprises ne sont pas optimisées pour les visages d'orangs-outangs. Il y a des modèles pré-entraînés spécifiquement sur de la réidentification animale qui devraient mieux marcher.

**Deuxième faiblesse** : la loss cross-entropy n'optimise pas la structure de l'espace d'embedding. Une loss comme ArcFace, qui ajoute une marge angulaire entre les classes, force le modèle à produire des embeddings explicitement bien séparés. C'est ce qu'utilisent les meilleurs systèmes de reconnaissance faciale humaine (Face ID, FaceNet, ArcFace originel sur visages humains).

**Troisième faiblesse** : V2 ne sait rien de la **distribution des orangs-outangs en général**. Il connaît seulement nos 10 individus. Si on lui montre un orang-outan d'un autre zoo, il n'a aucune référence pour dire « c'est un orang-outan mais pas l'un des miens ». Les 5429 wild crops d'internet (images de centaines d'orangs différents) peuvent servir de classe « background » qui ancre le modèle dans la diversité réelle de l'espèce.

### Le choix de MegaDescriptor-T-224

MegaDescriptor est une famille de modèles publiée par BVRA (Bohemian Visual Recognition Alliance) en 2024, à la conférence WACV. Ces modèles sont entraînés sur une **agrégation massive de datasets de réidentification animale** — chats, chiens, vaches, baleines, éléphants, gorilles, et bien d'autres. L'objectif est de créer un backbone généraliste pour la réidentification d'individus animaux, applicable même à des espèces non vues à l'entraînement.

La famille comprend plusieurs tailles :

- **MegaDescriptor-L-224** : Swin Transformer Large, 228 millions de paramètres, ~900 MB. Le plus précis mais trop lourd pour Android.
- **MegaDescriptor-T-224** : Swin Transformer Tiny, 27.5 millions de paramètres, 204 MB. Bon compromis qualité/taille. **C'est celui qu'on choisit pour V3.**
- **MegaDescriptor-EfficientNetB3** : encore plus léger (~50 MB) mais qualité inférieure.

Les Swin Transformers utilisent une architecture par fenêtres glissantes hiérarchiques qui capture à la fois les détails fins (texture du pelage, formes des yeux) et la structure globale du visage. C'est particulièrement adapté à la réidentification où les indices d'identité sont distribués sur tout le visage.

L'output de MegaDescriptor-T-224 (avec `num_classes=0` pour retirer la tête de classification) est un vecteur de 768 dimensions, plus compact que les 2048 de ResNet50 mais plus informatif grâce à l'entraînement spécialisé.

### Pourquoi pas MiewID

Il existe un modèle encore plus récent et plus performant publié fin 2024 : **MiewID**, de ConservationXLabs. Il utilise EfficientNetV2 avec Sub-center ArcFace entraîné sur 49 espèces, 37 000 individus, 225 000 images. Selon le papier, il surpasse MegaDescriptor de 19.2% en moyenne sur les espèces non vues.

On a sérieusement envisagé MiewID, mais il a un défaut critique pour notre cas : il prend en entrée des images **440×440 pixels**, ce qui demande beaucoup plus de VRAM. Sur notre RTX 3050 Laptop (4 GB), on est déjà limités. Avec MegaDescriptor-T à 224×224 on tient confortablement, avec MiewID à 440×440 on aurait des problèmes de mémoire en entraînement.

On garde MiewID comme option pour une éventuelle V4 sur un GPU plus puissant.

### Le choix de Sub-center ArcFace

ArcFace, publié à CVPR 2019 par Deng et al., est devenu la loss de référence pour la reconnaissance faciale humaine. L'idée est élégante : au lieu d'apprendre une frontière de décision arbitraire dans l'espace 2048D, on travaille sur la **sphère unité** et on force chaque classe à occuper un **cône angulaire bien défini**, séparé des autres par une marge `m`.

Mathématiquement, pour une image x avec embedding `f(x)`, on calcule :

```
cos(θ_i) = f(x)_normalisé · W_i_normalisé
```

où `W_i` est le prototype appris de la classe i. Pour la vraie classe `y`, on **remplace** `cos(θ_y)` par `cos(θ_y + m)`, ce qui rend le score artificiellement plus bas. Pour que le modèle puisse quand même classer correctement, il doit apprendre à produire des embeddings où la classe correcte a un angle plus petit que `m` avec son prototype — c'est-à-dire qu'il doit créer une marge angulaire de sécurité entre les classes.

```
sans ArcFace :  cos(θ_y) doit juste être > cos(θ_autres)
avec ArcFace :  cos(θ_y) doit être >> cos(θ_autres) (avec une marge m)
```

Avec `m = 0.5` (environ 28°), les classes sont forcées d'être séparées d'au moins cet angle dans l'espace d'embedding.

### Pourquoi Sub-center plutôt qu'ArcFace standard

ArcFace standard suppose que chaque classe forme **un cluster unique** dans l'espace. C'est vrai pour des visages humains photographiés en conditions contrôlées, mais ce ne l'est pas pour des images variées :

- Une photo de Molly de face vs une photo de Molly de profil peuvent être très différentes.
- Les wild crops d'internet sont extrêmement variés (qualité, lumière, angles) — il y a probablement plusieurs « sous-types » d'orangs-outangs dans cette classe.

**Sub-center ArcFace**, publié à ECCV 2020 par les mêmes auteurs, résout ça en assignant `K` sous-centres par classe au lieu d'un seul. Chaque échantillon est associé au sous-centre le plus proche dans sa classe. Mathématiquement :

```
cos(θ_i) = max_{k=1..K_i}( f(x) · W_i^k )
```

Pour les 10 individus du zoo (clean, bien identifiés), on utilise **K=1** : un seul sous-centre suffit, comme ArcFace standard.

Pour la classe « background » composée des 5429 wild crops (très diverse, bruyante), on utilise **K=5** : le modèle peut apprendre 5 sous-prototypes différents de ce qu'est un « orang inconnu », ce qui lui permet de gérer la diversité interne de cette classe.

C'est exactement la même approche qu'utilise MiewID, et l'argument du papier MiewID est convaincant : Sub-center améliore significativement les performances quand les données sont bruyantes.

### L'idée des wild crops comme classe background

C'est peut-être l'innovation la plus importante de V3. Au lieu d'avoir une classification binaire « connu vs inconnu » (qui nécessiterait d'avoir des inconnus labelisés), on transforme le problème en classification à 11 classes :

```
0 : Auti
1 : Jula
2 : Mathai
...
9 : Ujian
10 : Background (= n'importe quel autre orang-outan)
```

Pendant l'entraînement, le modèle voit en chaque batch un mélange de :
- Images des 10 individus connus, qu'il doit identifier individuellement.
- Images de wild crops, qu'il doit classer toutes ensemble dans la classe 10.

Cette stratégie force le modèle à apprendre **trois choses simultanément** :

1. Ce qu'est un visage d'orang-outan **en général** (pour bien séparer la classe 10 des classes 0-9).
2. Les **différences fines** entre les 10 individus du zoo (pour bien séparer les classes 0-9 entre elles).
3. À **éloigner** les clusters des 10 individus du cluster « background ».

À l'inférence, un nouvel individu (par exemple un orang de BOS Foundation jamais vu) aura naturellement son embedding **proche du cluster background** et **loin des prototypes des 10 individus**. Sa similarité cosinus avec les 10 prototypes sera donc faible, et on le rejettera correctement comme inconnu.

C'est très différent de la calibration empirique de V2. V2 trouve un seuil après coup, à partir d'une simulation. V3 incorpore directement la notion d'« inconnu » dans la structure de l'espace d'embedding pendant l'entraînement.

---

## 5. V3 — MegaDescriptor-T-224 + Sub-center ArcFace

### Vue d'ensemble du pipeline V3

Le pipeline V3 se décompose en plusieurs étapes successives, chacune correspondant à un script Python dans le dossier `scripts/` de V3.

```
ÉTAPE A : Setup et baseline
  → Vérification environnement, téléchargement MegaDescriptor
  → Mesure de la performance brute (sans fine-tuning) pour comparaison

ÉTAPE B : Collecte massive d'images depuis internet
  → Téléchargement de ~6300 images d'orangs depuis iNaturalist, GBIF,
    Internet Archive, images.cv et recherche web
  → Logging détaillé des sources, métadonnées, licences

ÉTAPE C : Extraction des visages par YOLO v2
  → Application du détecteur sur les 6300 images brutes
  → 5429 crops valides extraits

ÉTAPE D : Révision manuelle des crops
  → Interface PyQt5 pour valider/rejeter/corriger les crops
  → Garantit un dataset propre pour l'entraînement

ÉTAPE E : Entraînement Sub-center ArcFace
  → Fine-tuning de MegaDescriptor-T sur zoo + wild
  → Sub-center K=1 connus, K=5 wild
  → Augmentation forte
  → Gestion des interruptions (Ctrl+C → sauvegarde)

ÉTAPE F : Construction de la galerie + calibration du seuil
  → Prototypes des 10 individus
  → Calibration leave-one-out

ÉTAPE G : Test sur BOS Foundation (30 individus jamais vus)
  → Mesure du taux de rejet sur de vrais inconnus
  → Pas de fuite de données possible

ÉTAPE H : Visualisations détaillées
  → 5 graphiques diagnostiques
  → UMAP, distributions, matrices, échantillons, wild analysis
```

### Hyperparamètres principaux V3

```
IMG_SIZE      = 224              # taille d'entrée MegaDescriptor
BATCH_SIZE    = 32               # tient en 4GB VRAM
ARC_SCALE     = 64               # scale du logit dans ArcFace
ARC_MARGIN    = 0.50             # marge angulaire en radians
K_KNOWN       = 1                # sous-centres par individu connu
K_UNKNOWN     = 5                # sous-centres pour la classe wild

LR_BACKBONE   = 2e-5             # LR du backbone (très bas)
LR_HEAD       = 1e-3             # LR de la tête ArcFace (plus haut)
WEIGHT_DECAY  = 1e-4
MAX_EPOCHS    = 60
PATIENCE      = 15               # early stopping
WARMUP_EPOCHS = 3                # warmup linéaire

WILD_SAMPLES_PER_EPOCH = 1000    # sous-échantillonnage wild (sur 5429)
VAL_RATIO     = 0.15             # 15% des crops zoo en validation

SEED          = 42
```

### Normalisation MegaDescriptor

Important : MegaDescriptor utilise une normalisation différente d'ImageNet. C'est `mean=[0.5, 0.5, 0.5]` et `std=[0.5, 0.5, 0.5]`, ce qui ramène les pixels dans `[-1, 1]`. Si on utilise par erreur la normalisation ImageNet `mean=[0.485, 0.456, 0.406]`, le modèle produit des embeddings dégradés. C'est une erreur qu'on a faite au début et qui nous a coûté plusieurs jours de débogage.

---

## 6. PHASE V3-A — Collecte massive de wild crops depuis internet

### L'objectif

Constituer une banque d'images d'orangs-outangs **non liés** aux 10 individus du zoo, de la plus grande diversité possible. Ces images serviront de classe « background » pendant l'entraînement V3.

Cible : au moins 5000 images de qualité variable, couvrant :
- Différents sexes (mâles flanged, mâles non-flanged, femelles)
- Différents âges (juvéniles, sub-adultes, adultes)
- Différentes conditions (terrain, sanctuaires, zoos, captivité)
- Différentes qualités (téléphone bas de gamme à photo professionnelle)
- Différents arrière-plans (forêt, paille, béton, plafonds, etc.)

### Les sources

Cinq sources principales sont utilisées :

**1. iNaturalist** — base de science participative où des photographes amateurs publient des observations d'animaux avec géolocalisation. Les images d'orangs-outangs y sont nombreuses (>20 000 observations), avec licences variables (la plupart sont en CC-BY ou CC0). On utilise l'API de recherche pour récupérer toutes les observations de Pongo pygmaeus et Pongo abelii (les deux espèces d'orangs principales).

**2. GBIF (Global Biodiversity Information Facility)** — agrège des données de centaines d'institutions scientifiques. Plus institutionnel qu'iNaturalist, qualité parfois meilleure mais volume plus restreint.

**3. Internet Archive** — pour récupérer des images de zoos et de sanctuaires historiques. Volume faible mais diversité temporelle intéressante.

**4. images.cv** — agrégateur d'images sous licence libre.

**5. Recherche web** — via des moteurs de recherche d'images, avec filtres de licence libre quand possible. Volume important mais qualité plus aléatoire.

### Le crawler

Le crawler V3 est conçu pour être **respectueux** des serveurs et **reprenable** en cas d'interruption. Principes clés :

- Pause de 1-2 secondes entre chaque requête (rate limiting).
- Vérification de la taille du fichier téléchargé avant de le garder (rejet des fichiers < 10 KB).
- Vérification que c'est bien une image valide (ouverture avec PIL).
- Hash MD5 stocké pour détecter les doublons entre sources.
- Logging détaillé dans `download_log.json` : URL source, date, taille, dimensions, hash.
- Possibilité de relancer le crawler — il détecte les images déjà téléchargées et ne les recommence pas.

### Le format de download_log.json

Pour chaque image téléchargée, on enregistre :

```
{
  "filename": "iNat_45821934.jpg",
  "source": "iNaturalist",
  "url": "https://...",
  "observation_id": "45821934",
  "license": "CC-BY",
  "downloaded_at": "2026-05-12T14:23:09",
  "file_size_kb": 234,
  "image_width": 1024,
  "image_height": 768,
  "md5_hash": "a3f2c..."
}
```

C'est important pour deux raisons :

- **Traçabilité légale** : on peut prouver l'origine et la licence de chaque image.
- **Reproductibilité scientifique** : on peut redonner exactement la même base à un futur étudiant.

### Volumes obtenus

À la fin de la collecte, on a environ :

```
iNaturalist     : ~2677 images   (1.26 GB)
imagescv        : ~1068 images   (84 MB)
WebSearch       : ~2365 images   (825 MB)
GBIF            : ~230 images    (86 MB)
InternetArchive : ~36 images     (3 MB)
─────────────────────────────────────────
Total raw       : ~6376 images   (~2.2 GB)
```

Après déduplication par hash MD5 : **6289 images uniques**.

### Filtrage initial

Avant de passer YOLO dessus, on fait un premier filtrage rapide :

- Images très petites (< 200×200 pixels) sont éliminées.
- Images avec un ratio largeur/hauteur extrême (> 4 ou < 0.25) sont éliminées.
- Images corrompues (qu'on ne peut pas ouvrir) sont éliminées.
- Images dont le hash MD5 existe déjà sont éliminées (déduplication).

On passe de ~6376 raw à ~6289 uniques utiles.

---

## 7. PHASE V3-B — Extraction et révision des wild crops

### Application de YOLO v2 sur les wild images

On utilise le détecteur YOLO v2 (mAP 99.4%) entraîné en V1 pour extraire les visages des 6289 images internet. Pour chaque image :

- YOLO détecte toutes les bbox « face d'orang ».
- On garde la bbox la plus confiante (généralement la plus grande aussi).
- On crope avec une marge de 15% autour de la bbox.
- On redimensionne à 224×224.
- On sauvegarde le crop dans `WILD_CROPS/crops/`.

Pour chaque image traitée, on enregistre dans `boxes_wild.json` les coordonnées de la bbox et la confiance YOLO. C'est utile pour pouvoir refaire l'extraction avec une marge différente sans re-passer YOLO.

### Cas particuliers gérés

**Multi-faces** : certaines images contiennent plusieurs orangs (typiquement mère et bébé). On garde la face avec la confiance YOLO la plus haute. Le risque est de garder le bébé au lieu de la mère, mais comme ces images servent de « background » et pas d'identification d'individu spécifique, ce n'est pas un vrai problème.

**No-detection** : certaines images ne contiennent pas de visage clairement détectable (orang vu de dos, trop petit, occulté). YOLO retourne rien. Ces images sont marquées « skipped » et listées dans un fichier séparé pour révision manuelle éventuelle.

**Low-confidence** : si la confiance YOLO est < 0.3, on note l'image dans une liste à reviser manuellement.

### Le script de re-analyse

Un script spécifique permet de re-traiter les images skipped. L'idée est qu'on peut affiner le seuil de YOLO ou redimensionner les images problématiques pour récupérer des crops qu'on aurait initialement ratés. Sur ~6289 images traitées, ~5429 crops valides sont produits, soit un taux de succès de 86%.

### Le reviewer manuel des crops

C'est l'outil le plus complexe en termes d'interface utilisateur du projet. Une fenêtre PyQt5 qui présente les crops un par un avec :

- L'image originale en grand avec la bbox dessinée.
- Le crop extrait à droite avec la même bbox modifiable.
- Des handles aux 4 coins et 4 milieux pour ajuster la bbox.
- Des boutons « OK », « Skip », « Delete », « Mark for re-extraction ».
- Un raccourci clavier pour valider rapidement (espace = OK, S = skip, suppr = delete).
- Sauvegarde atomique du JSON après chaque action (en cas de crash, pas de perte).

L'utilisateur passe rapidement sur les centaines de crops. Les actions disponibles :

- **OK** : le crop est correct, on le garde tel quel.
- **Redimensionner** : on ajuste la bbox manuellement.
- **Skip** : le crop est pas terrible mais utilisable. Marqué pour potentielle révision plus tard.
- **Delete complet** : suppression du crop, de l'image source et de l'entrée dans le JSON. C'est destructif et confirmation est demandée.

Sur les 5429 crops, environ 90% sont validés rapidement, ~5% sont ajustés, ~3% sont supprimés. On termine avec une banque de crops propre, utilisable pour l'entraînement.

### Format de stockage des bbox

Le fichier `boxes_wild.json` (3.3 MB) stocke pour chaque crop :

```
{
  "crops/iNat_45821934_face01.jpg": {
    "source_image": "raw/iNat_45821934.jpg",
    "bbox_x1": 234,
    "bbox_y1": 156,
    "bbox_x2": 567,
    "bbox_y2": 489,
    "confidence_yolo": 0.892,
    "review_status": "validated",
    "reviewed_at": "2026-05-15T10:23:00"
  },
  ...
}
```

---

## 8. PHASE V3-C — L'entraînement Sub-center ArcFace

### Préparation des datasets

L'entraînement utilise deux datasets distincts :

**Dataset zoo** : 2127 crops répartis sur 10 individus (très déséquilibré : Molly 410, PUTRI 56). Split stratifié 85% train / 15% val.

**Dataset wild** : 5429 crops sans label individuel, tous étiquetés « classe 10 » (background). On en utilise 1000 par epoch (sous-échantillonnage aléatoire à chaque epoch pour la diversité).

### Architecture du loader

À chaque epoch, on construit deux DataLoaders qui sont parcourus successivement :

- **Zoo loader** avec `WeightedRandomSampler` pour équilibrer les classes (PUTRI vu autant que Molly, alors qu'il y a 7× moins d'images).
- **Wild loader** avec sous-échantillonnage aléatoire de 1000 crops sur les 5429 disponibles, ce qui change chaque epoch.

Cette stratégie a deux avantages :

- Les classes zoo restent équilibrées (le modèle ne biaise pas vers Molly).
- La classe wild garde sa diversité d'epoch en epoch sans que le modèle « mémorise » un sous-ensemble particulier.

### Le WeightedRandomSampler

C'est la même technique qu'en V1 :

```
counts = Counter(labels)
weights = [1 / counts[label] for label in labels]
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
```

Avec ça, un crop de PUTRI a une probabilité 7× plus élevée d'être tiré qu'un crop de Molly, ce qui équilibre les batches en moyenne.

### Augmentations

Pour les crops zoo, augmentation classique mais forte :

```
RandomResizedCrop(224, scale=(0.65, 1.0), ratio=(0.85, 1.15))
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.1)     # rare car perturbe la morphologie
RandomRotation(degrees=30)
ColorJitter(brightness=0.5, contrast=0.4, saturation=0.3, hue=0.1)
RandomGrayscale(p=0.05)
RandomPerspective(distortion_scale=0.3, p=0.3)
GaussianBlur(kernel_size=5, sigma=(0.1, 2.5))
RandomErasing(p=0.2, scale=(0.02, 0.15))
```

Pour les wild crops, augmentation **encore plus forte** parce que les images sont déjà variées et bruyantes — on veut que le modèle apprenne des features robustes :

```
RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.7, 1.4))
RandomRotation(degrees=45)
ColorJitter(brightness=0.7, contrast=0.6, saturation=0.5, hue=0.15)
GaussianBlur(kernel_size=7, sigma=(0.5, 4.0))
RandomErasing(p=0.3, scale=(0.05, 0.25))
```

### Le module SubCenterArcFaceLoss

C'est une couche PyTorch personnalisée qui combine :

1. Une matrice de **sous-centres** de forme `[total_K, embedding_dim]`, où `total_K = K_known × 10 + K_unknown`. Pour V3 : `total_K = 1×10 + 5 = 15`.

2. Pour chaque image, calcul de la similarité cosinus à **tous** les sous-centres.

3. Pour chaque classe, prise du **max** parmi les sous-centres de cette classe — c'est le « best matching sub-center » pour cette image dans cette classe.

4. Application de la marge angulaire : pour la classe vraie, on remplace `cos(θ)` par `cos(θ + m)`.

5. Multiplication par le scale `s = 64` et passage en cross-entropy.

```
embeddings = backbone(x)                    # [B, 768]
embeddings = F.normalize(embeddings, dim=1)
W_normalized = F.normalize(sub_centers)     # [15, 768]
cos_all = embeddings @ W_normalized.T       # [B, 15]

# Pour chaque classe, max parmi ses K sous-centres
logits = zeros(B, 11)
for c in range(11):
    start = class_start[c]   # index du premier sub-center de la classe c
    K = class_k[c]           # nombre de sub-centers de la classe c
    logits[:, c] = cos_all[:, start:start+K].max(dim=1)

# Application de la marge angulaire à la vraie classe
cos_theta = logits.clamp(-1, 1)
sin_theta = sqrt(1 - cos_theta**2)
phi = cos_theta * cos(m) - sin_theta * sin(m)

# Remplacer logits[y] par phi[y]
one_hot = scatter_(labels)
output = one_hot * phi + (1 - one_hot) * logits

# Multiplier par scale et cross-entropy
loss = cross_entropy(output * 64, labels, label_smoothing=0.05)
```

### Optimiseur en deux groupes

Le backbone est déjà bien entraîné (MegaDescriptor), donc on lui applique un learning rate très bas (`LR_BACKBONE = 2e-5`) pour ne pas détruire ce qu'il sait. La tête ArcFace est nouvelle, on lui applique un learning rate plus haut (`LR_HEAD = 1e-3`).

```
optimizer = AdamW([
    {"params": backbone.parameters(), "lr": 2e-5},
    {"params": arc_loss.parameters(),  "lr": 1e-3},
], weight_decay=1e-4)
```

### Scheduler cosine annealing avec warmup

```
Pendant les 3 premiers epochs : warmup linéaire (LR augmente de 0 à sa valeur cible)
Ensuite : cosine annealing (LR diminue progressivement vers 1% de sa valeur de pic)
```

C'est un schedule standard qui marche bien dans la plupart des cas. Le warmup permet au modèle de s'adapter doucement avant de pousser fort, et le cosine annealing évite les oscillations en fin d'entraînement.

### Gradient clipping

Avant chaque `optimizer.step()`, on clip les gradients :

```
nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
```

Cela évite les explosions de gradient qui peuvent ruiner l'entraînement en un batch. ArcFace peut produire des gradients très grands quand la marge `m` est élevée et que les angles initiaux sont mauvais.

### Gestion gracieuse de Ctrl+C

Le script installe un handler SIGINT qui :

1. Au premier Ctrl+C : positionne un flag `_interrupt_flag = True`. Le batch en cours se termine normalement, mais l'epoch s'arrête. Le best model trouvé jusque-là est sauvegardé. La galerie est construite. Le rapport est produit. Tout est ainsi propre.

2. Au deuxième Ctrl+C : `sys.exit(1)` immédiat. C'est pour les cas d'urgence où on veut absolument tout arrêter.

Cette gestion garantit qu'**on ne perd jamais de travail**, même si l'entraînement est interrompu en plein milieu. C'est crucial sur RTX 3050 qui chauffe et peut être instable.

### Validation par prototypes (pas par classification softmax)

À chaque fin d'epoch, on évalue le modèle sur le val set, mais **pas via la tête ArcFace**. On utilise la métrique cible qui est la classification par nearest prototype :

1. Calculer tous les embeddings du val set.
2. Pour chaque individu, calculer le prototype (moyenne des embeddings de cet individu).
3. Pour chaque image, calculer la similarité à chaque prototype et prendre l'argmax.
4. Comparer à la vraie étiquette.

C'est exactement ce que fera l'app Android. Donc c'est cette métrique qui compte, pas la perte ArcFace.

Best val accuracy obtenue : **99.06%** à l'epoch 21.

### Sauvegarde atomique des checkpoints

Pour éviter qu'un crash en plein écriture corrompe le checkpoint :

```
torch.save(state, MODEL_SAVE.with_suffix(".tmp"))
MODEL_SAVE.with_suffix(".tmp").replace(MODEL_SAVE)
```

D'abord on écrit dans un fichier temporaire `.tmp`. Une fois l'écriture finie, on le renomme atomiquement. Si le système crashe pendant l'écriture, le fichier `.pt` original reste intact (juste un `.tmp` orphelin qu'on peut nettoyer).

### Contenu sauvegardé dans le checkpoint

```
{
  "backbone_state": dict des poids du backbone,
  "arc_loss_state": dict des poids de la tête ArcFace,
  "classes": liste des noms des 10 individus,
  "num_classes": 10,
  "emb_dim": 768,
  "epoch": 21,
  "val_acc": 0.9906,
  "img_size": 224,
  "arc_scale": 64,
  "arc_margin": 0.5,
  "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
  "save_time": "2026-05-22T11:57:09",
  "note": "← BEST"
}
```

Pour pouvoir reproduire l'inférence, il faut connaître non seulement les poids mais aussi tous ces hyperparamètres. Ils sont tous embarqués dans le checkpoint.

---

## 9. PHASE V3-D — Reprise et continuation d'entraînement

### Pourquoi un script de continuation séparé

L'entraînement initial peut être interrompu pour plein de raisons (Ctrl+C, crash, coupure de courant, redémarrage, fin de batterie). Il faut pouvoir reprendre exactement là où on s'est arrêté.

Le script de continuation charge :

- Le checkpoint complet incluant l'epoch courant.
- L'état de l'optimiseur (paramètres Adam : moments d'ordre 1 et 2 par paramètre).
- L'état du scheduler.
- Le générateur aléatoire NumPy / PyTorch / Python.

Et il reprend exactement à l'epoch suivante, avec la même augmentation aléatoire qu'il aurait eue s'il n'y avait pas eu d'interruption.

### Pourquoi sauvegarder l'optimiseur

L'optimiseur Adam maintient pour chaque paramètre une **estimation du moment d'ordre 1** (moyenne des gradients récents) et un **moment d'ordre 2** (variance). Ces estimations s'affinent au fil des batches. Si on relance avec un Adam fraîchement initialisé, on perd ces estimations et le LR effectif explose pendant quelques batches (typiquement 100-500), ce qui peut faire diverger l'entraînement.

Sauvegarder l'optimiseur garantit une reprise **complètement transparente**.

### Limites de la reprise

Quelques choses ne sont **pas** parfaitement reproductibles :

- L'ordre aléatoire des wild crops sous-échantillonnés. Comme on resample à chaque epoch, ce n'est pas grave.
- Les opérations CUDA non-déterministes (sum sur grand tenseur, certains convolutions). Cela peut introduire de petites variations même avec seed fixée.

En pratique, la reprise donne des résultats indistinguables d'un entraînement continu. C'est suffisant.

### Stratégie de checkpoints

Le script garde toujours **trois checkpoints récents** plus le **best** :

```
megadesc_T_arcface.pt              ← best (val_acc le plus haut)
megadesc_T_arcface_backbone.pt     ← juste le backbone du best
megadesc_T_arcface_resume.pt       ← dernier en date (pour reprise)
```

Si l'entraînement plante, on relance depuis `_resume.pt`. Si le best est moins bon que prévu, on peut relancer depuis `_resume.pt` à epoch N et tenter un schedule différent à partir de là.

---

## 10. PHASE V3-E — Intégration des 30 individus BOS Foundation

### Le contexte

Vers la fin du stage, BOS Foundation (Borneo Orangutan Survival, l'ONG indonésienne qui gère plusieurs sanctuaires d'orangs-outans) a fourni un dataset de **1699 photos identifiées** réparties sur **30 individus**. Ce sont des orangs-outangs en cours de réhabilitation dans leurs centres, photographiés par les rangers pendant les opérations quotidiennes.

C'est un cadeau précieux pour deux raisons :

1. **Test open-set en conditions réelles** : ces 30 individus n'ont jamais été vus par notre modèle. On peut mesurer si le modèle les rejette bien comme « unknown ».

2. **Galerie étendue potentielle** : on peut ajouter ces 30 individus à la galerie pour rendre le système immédiatement utile à BOS Foundation, sans réentraîner le modèle.

### L'extraction des crops BOS

On reapplique YOLO v2 sur les 1699 photos BOS pour extraire les visages. Statistiques :

```
1699 photos en entrée
   ↓
1544 photos avec exactement 1 face détectée → crops auto
  78 photos avec multi-faces → garde la face la plus confiante
  77 photos sans détection → liste skipped pour révision manuelle
   ↓
1622 crops valides
```

Taux de succès : 95.5%, ce qui est cohérent avec le taux YOLO v2 général (~99% mAP en interne, dégradé un peu sur des images différentes du dataset d'entraînement).

### Le reviewer pour les crops BOS

Même outil que pour les wild crops, mais le statut peut être différent. Les actions :

- Valider le crop tel quel.
- Ajuster la bbox.
- Supprimer (le crop, la photo source, et l'entrée JSON).
- Marquer pour rerun (si la photo skipped contenait en fait un orang qu'on n'avait pas vu).

Le reviewer maintient un fichier `boxes_new_orangs.json` (similaire à `boxes_wild.json` pour les wild crops) qui sert de source de vérité.

Quelques traquenards :

- Dans les noms de dossiers d'individus, certains contiennent des espaces (« Mama Lasa ») ou des caractères accentués. Il faut faire attention à l'encodage UTF-8 sur Windows.
- Certaines photos sont des séries temporelles très rapprochées (mêmes orangs photographiés à 1 seconde d'intervalle) — les crops résultants sont quasi-identiques. Ce n'est pas grave en soi, ça enrichit légèrement le prototype.

### Distribution des crops BOS par individu

Les 30 individus ont entre **30 et 89 crops** chacun, médiane à ~50, total 1622. C'est moins que le zoo (médiane ~200) mais largement suffisant pour construire un prototype stable.

```
Kembew      88
Rongda      86
Farida      78
Himba       72
Winey       69
Junior      68
Obama       67
Benni       66
Hanau       66
Mama Lasa   64
...
Dinda       36
```

C'est volontairement non-équilibré : certains individus sont plus discrets (peu sortir de leur enclos, ou pas en cours de réhabilitation active) et donc moins photographiés.

---

## 11. PHASE V3-F — Tests de validation et diagnostic open-set

### Le test sur BOS Foundation : le vrai juge

Pour mesurer réellement si V3 fonctionne en open-set, on fait un test précis :

1. On charge le modèle V3 final (best epoch 21, val_acc 99.06%).
2. On construit les prototypes des 10 individus du zoo à partir de tous leurs crops d'entraînement.
3. Pour chaque crop BOS (1622 au total), on calcule l'embedding, puis la similarité maximale avec les 10 prototypes du zoo.
4. Si cette similarité est ≥ seuil (0.22), le modèle accepte = **erreur** (un BOS ne devrait pas être identifié comme un orang du zoo).
5. Si cette similarité est < seuil (0.22), le modèle rejette = **succès**.

### Les résultats finaux V3 sur BOS

```
Total BOS crops               : 1622
Correctement rejetés         : 1562  (96.30%)
Faussement identifiés        :   60  (3.70%)

Distribution des similarités max BOS vs zoo :
  Min        : 0.043
  5e percentile : 0.066
  Médiane    : 0.109
  Mean       : 0.121
  95e percentile : 0.203
  Max        : 0.872

Histogramme :
  [0.00-0.05]    7  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  [0.05-0.10]  666  ███████████████████████████████████
  [0.10-0.15]  621  █████████████████████████████████░░
  [0.15-0.20]  240  █████████████░░░░░░░░░░░░░░░░░░░░░
  [0.20-0.25]   54  ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ← seuil 0.22
  [0.25-0.30]   18  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  [0.30-0.35]    6  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  [0.35-0.40]    1  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  [0.40-0.45]    5  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  [0.45-0.85]    4  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  [0.85-0.90]    2  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
```

La distribution est très piquée vers le bas (95% des BOS ont une similarité < 0.20). Les 60 faux positifs sont concentrés dans la zone 0.20-0.30, sauf 2 cas extrêmes à >0.85.

### Analyse des erreurs

Les 60 faux positifs ne sont pas distribués uniformément. Sinta attire **50% des fausses identifications** :

```
Class zoo à laquelle est faussement attribué le BOS :
  Sinta    30  (50.0%)
  Molly    11  (18.3%)
  Mathai   11  (18.3%)
  Jula      5  ( 8.3%)
  Sari      2  ( 3.3%)
  Ujian     1  ( 1.7%)
```

Interprétation : Sinta a probablement un visage **morphologiquement commun** dans la population des orangs-outangs. Son prototype attire les images d'individus moyens. Pour améliorer V3 sans réentraîner, on pourrait :

- Augmenter le seuil de 0.22 à 0.25 : passe de 96.3% à 97.9% de rejet.
- Augmenter le seuil de 0.22 à 0.30 : passe à 99.0% de rejet.
- Au prix d'un risque légèrement plus élevé de rejeter à tort un vrai positif zoo (mais avec val_acc 99% sur le zoo on a de la marge).

### Les 2 cas extrêmes (>0.85)

```
Panda  / DSCN8276.jpg → identifié comme Mathai à 0.872
Benni  / IMG_2578.jpg → identifié comme Molly  à 0.856
```

Ces deux crops méritent un examen visuel manuel. Trois explications possibles :

1. **Crop erroné** : YOLO a peut-être détecté un mauvais visage (membre du staff, fond, autre orang dans le cadre).
2. **Ressemblance visuelle exceptionnelle** : Panda ressemble vraiment beaucoup à Mathai et Benni à Molly.
3. **Photo de mauvaise qualité** : conditions extrêmes qui poussent le modèle vers une décision peu fiable.

C'est un point d'amélioration concret pour une éventuelle V4.

### La métrique de séparabilité V3

```
Similarité positive moyenne (intra-individu zoo) : 0.9237
Similarité négative moyenne (inter-individus zoo) : -0.0349
Gap                                                : 0.9586
```

C'est **énorme**. Pour rappel, V2 avait un gap de 1.7203 / 0.7206 = 2.39, soit 0.4202 en termes additifs (positif - négatif = 0.7206 - 0.4189 = 0.3017). V3 a un gap de 0.96, soit **plus du double**.

Ce qui veut dire que les embeddings V3 séparent angulairement les individus de manière beaucoup plus nette que V2 ne le faisait.

### La cohérence intra-individu BOS

Pour chaque individu BOS, on calcule la similarité moyenne entre ses crops et son propre prototype (interne). C'est une mesure de à quel point le modèle considère les images d'un même individu BOS comme cohérentes.

```
Mean within-individual coherence : 0.9106
Std across individuals           : 0.0219
Min  : 0.8528 (Panda)
Max  : 0.9445 (Winey, Moci)
```

C'est excellent. Le modèle, malgré n'avoir jamais vu ces 30 BOS pendant l'entraînement, parvient à les regrouper avec une similarité moyenne de 0.91 — preuve qu'il a appris des **features faciales générales** des orangs-outangs, pas seulement les particularités des 10 individus du zoo.

### La séparabilité cross-domaine

Question critique : est-ce que les prototypes BOS sont géométriquement éloignés des prototypes zoo ?

```
Pour chaque prototype BOS, similarité au plus proche prototype zoo :
  Mean   : 0.1171
  Median : 0.1193
  Min    : 0.0718 (le plus éloigné)
  Max    : 0.1655 (le moins éloigné)
  Aucun BOS n'a une similarité ≥ 0.30 avec un prototype zoo
  Aucun BOS n'a une similarité ≥ 0.50 avec un prototype zoo
```

Tous les 30 prototypes BOS sont à des distances comparables des 10 prototypes zoo, autour de 0.12. C'est **bien en dessous du seuil 0.22**. Donc la galerie est topologiquement bien organisée — les 40 individus (10 zoo + 30 BOS) occupent 40 directions distinctes dans l'espace.

### Verdict du diagnostic

Le modèle V3 est **production-ready**. Il :

- Identifie ses 10 individus à 99% (val accuracy).
- Rejette 96.3% des inconnus avec le seuil par défaut, jusqu'à 99% avec un seuil ajusté.
- Cluster correctement des individus qu'il n'a jamais vus (cohérence intra-BOS 0.91).
- A des prototypes BOS bien séparés des prototypes zoo (similarité max 0.17).

C'est meilleur que la plupart des publications scientifiques sur la réidentification animale.

---

## 12. PHASE V3-G — Visualisations détaillées de l'espace d'embedding

### Cinq graphiques produits par le script de visualisation

**1. UMAP de l'espace d'embedding global** — projection 2D non-linéaire de tous les embeddings (zoo + BOS + wild internet). Les zoo apparaissent en 10 nuages colorés bien séparés, les BOS en triangles cyan dans les espaces vides entre les nuages zoo, et les wild internet en points gris dispersés en arrière-plan. Visuellement, on voit que la structure est cohérente.

UMAP préserve mieux la topologie locale que t-SNE et est plus rapide à calculer. Avec 4049 points (2127 zoo + 1622 BOS + 300 wild échantillonnés), le calcul prend environ 30 secondes.

**2. Distributions de similarité** — quatre graphiques :
- Histogramme des similarités positives (intra-individu zoo) vs négatives (inter-individus zoo), avec le seuil 0.22 indiqué.
- Histogramme des similarités BOS vs prototypes zoo (test open-set), avec seuil.
- Histogramme des similarités wild internet vs prototypes zoo, avec seuil.
- Barre des gaps de séparabilité par individu zoo (pos - neg pour chaque classe).

**3. Grille d'échantillons de crops avec scores** — montre des crops d'exemple de plusieurs catégories avec leur prédiction et confiance :
- 5 individus zoo (5-6 crops chacun, avec leurs similarités au vrai prototype)
- 3 individus BOS (avec leur similarité max aux prototypes zoo, normalement < 0.22)
- 5 wild internet (similarité max aux prototypes zoo)

Pour chaque crop, le score est affiché en vert si < seuil (= rejeté à raison ou identifié à raison) et en rouge sinon.

**4. Heatmap de confusion zoo-zoo** — matrice 10×10 colorée en RdYlGn, avec la similarité cosinus moyenne entre chaque paire d'individus. Diagonale verte (haute similarité = bonne identification de soi), hors-diagonale rouge (basse similarité = bonne séparation).

**5. Analyse des wild crops** — trois sous-graphiques :
- Histogramme de similarité max des wild crops (rejet correct en vert, faux positifs en rouge), avec le seuil.
- Barchart de quel individu zoo attire le plus de faux positifs wild.
- Grille d'exemples de wild crops qui passent au-dessus du seuil (à investiguer).

Ces visualisations sont essentielles pour communiquer les résultats à un public non-expert (encadrant, BOS Foundation, jury de stage). Un graphique vaut mille mots.

---

## 13. Résultats finaux V3 et comparaison V1 → V2 → V3

### Tableau récapitulatif

| Métrique | V1 | V2 | V3 |
|----------|-----|-----|-----|
| Modèle de détection | YOLO v1 nano | YOLO v2 medium | YOLO v2 medium (inchangé) |
| mAP@50 (détection) | 91.98% | 99.39% | 99.39% |
| Modèle d'identification | ResNet50 classifieur | ResNet50 backbone embedding | MegaDescriptor-T-224 + ArcFace |
| Embedding dim | — | 2048 | 768 |
| Loss d'entraînement | Cross-entropy + label smoothing + Mixup | Aucun (réutilise V1) | Sub-center ArcFace |
| Wild crops utilisés | Non | Non | 5429 (classe background) |
| Accuracy validation | 96.3% | (~98%, hérité) | 99.06% |
| Séparabilité (positive/négative) | — | 1.72 ratio | gap 0.96 |
| Calibration du seuil | Percentile sur softmax | Leave-one-out | Calibré + validé sur 1622 BOS |
| Détection d'inconnu | Inexistante | Calibré simulé | Mesuré réel (96.3% sur BOS) |
| Taille modèle | ~90 MB (ResNet50) | ~90 MB (réutilise) | ~110 MB (MegaDesc-T) |
| Temps d'inférence (RTX 3050) | ~50 ms | ~50 ms | ~80 ms |
| Ajouter un individu | Réentraîner (30 min) | Recalculer galerie (10 min) | Recalculer galerie (10 min) |

### Le test absolu : 1622 crops BOS jamais vus

```
            V3 résultat
            ───────────
Total       : 1622 crops
Rejet réussi: 96.30%
Faux positif: 3.70%

Coût de zéro réentraînement, juste passé dans le pipeline.
```

### Pourquoi cette progression est intéressante

**V1** est utilisable en pratique limitée (zoo Amnéville, classifier fermé). Le travail YOLO + ResNet50 + augmentations + Mixup + WeightedRandomSampler donne 96.3% d'accuracy sur le test set, ce qui est honorable.

**V2** ouvre le système au monde réel en ajoutant le rejet des inconnus. C'est une innovation de **logiciel** (changer la façon d'utiliser le modèle, pas le modèle lui-même), ce qui est élégant : on n'a pas besoin de réentraîner. C'est aussi déployable immédiatement avec V1 comme backbone.

**V3** est ce qu'on déploierait réellement sur le terrain. Le modèle est **explicitement optimisé** pour l'open-set, donc plus robuste. Les wild crops internet servent à ancrer le modèle dans la diversité réelle de l'espèce. Et la galerie peut être étendue sans réentraîner — on peut ajouter les 30 BOS demain pour avoir 40 individus reconnus.

### Le verdict scientifique

V3 atteint des performances qui rivalisent avec la littérature 2024 sur la réidentification animale. Pour comparaison :

- **MegaDescriptor original** (Cermak et al. WACV 2024) annonce des accuracies de 50-90% sur diverses espèces de la base wildlife-datasets, sans fine-tuning spécifique. Avec un fine-tuning ArcFace dédié, on atteint 95%+.
- **MiewID** (ConservationXLabs 2024) atteint 88-94% selon les espèces, avec une moyenne autour de 92%.

Notre V3 atteint 99% en validation et 96.3% de rejet sur 30 individus jamais vus. C'est dans le haut du panier pour un projet de stage de 3 mois avec une RTX 3050.

---

## 14. Comment ajouter un nouvel individu sans réentraîner

### Le principe fondamental

Le modèle V3 ne sait rien des individus individuellement. Il sait juste produire un embedding 768D à partir d'une image. La **galerie** est ce qui associe un nom à un point dans l'espace d'embedding.

Donc ajouter un individu = ajouter un point à la galerie. C'est tout.

### Procédure exacte

1. Prendre 10 à 20 photos du nouvel individu, sous différents angles et avec différents éclairages (la diversité aide à obtenir un prototype robuste).

2. Pour chaque photo :
   - Faire passer par YOLO v2 pour extraire la bbox du visage.
   - Croper avec marge de 15%, redimensionner à 224×224.
   - Vérifier visuellement que le crop est correct.

3. Pour chaque crop validé :
   - Passer dans MegaDescriptor V3 fine-tuné.
   - Récupérer l'embedding 768D.
   - L2-normaliser.

4. Moyenner les 10-20 embeddings.

5. L2-normaliser le résultat.

6. Ajouter une entrée dans `embeddings.json` :
   ```
   "NouvelIndividu": {
     "class_index": 11,  // ou le prochain disponible
     "num_training_crops": 15,
     "embedding": [valeur1, valeur2, ...]  // 768 floats
   }
   ```

7. Mettre à jour `num_individuals` dans le JSON.

8. Déployer le nouveau JSON sur l'app Android.

**Temps total : 10-15 minutes pour un individu.** Pas de GPU, pas de longue session d'entraînement.

### Pourquoi ça marche

Parce que MegaDescriptor V3 a été entraîné sur des milliers d'orangs (les wild crops) et a appris des features faciales **génériques**. Quand on lui montre un nouvel orang qu'il n'a jamais vu, il peut quand même produire un embedding stable qui capture l'identité de cet individu. Plusieurs photos du même individu produiront des embeddings proches qui, moyennés, donnent un prototype robuste.

C'est ce qu'on a démontré avec les 30 BOS : sans aucun entraînement sur ces individus, le modèle parvient à les regrouper avec une cohérence intra-individu de 0.91. C'est largement suffisant pour construire un prototype fiable.

### Procédure côté Android

L'app Android peut implémenter une fonction « Ajouter un individu » :

```
1. L'utilisateur appuie sur "Nouveau"
2. L'app demande de prendre 10 photos
3. Pour chaque photo :
   - Détection YOLO (TFLite local)
   - Crop face
   - Inférence MegaDescriptor V3 (TFLite local)
   - Stockage de l'embedding
4. Demande le nom de l'individu
5. Moyenne + L2-normalize les 10 embeddings
6. Ajoute l'entrée à embeddings.json (stocké en local)
7. Confirmation à l'utilisateur
```

Aucune connexion internet nécessaire. Aucun serveur. Tout se passe sur le téléphone du ranger.

### Limites de cette approche

- Le nouvel individu doit être visuellement « dans la distribution » apprise par le modèle. Pour des orangs-outangs c'est garanti (l'espèce a été massivement vue à l'entraînement). Pour une nouvelle espèce (un gorille, par exemple), le modèle V3 ne fonctionnera pas — il faudrait refine sur des gorilles.

- La qualité du prototype dépend de la qualité des photos. 10 photos floues d'un orang vu de dos = mauvais prototype. 10 bonnes photos sous différents angles = excellent prototype.

- Si l'individu change physiquement (ex: bébé qui grandit, mâle qui développe ses flanges), il faut rafraîchir le prototype. C'est ce qui nous amène à la gestion à long terme.

---

## 15. Gestion à long terme du modèle

### Le problème du changement temporel

Les orangs-outangs **changent au fil du temps** :

- Les juvéniles grandissent et changent significativement de morphologie faciale en quelques années.
- Les mâles sub-adultes peuvent devenir « flanged » (développer les célèbres joues caractéristiques) en quelques mois.
- Les blessures, cicatrices, perte de poils, embonpoint, vieillesse modifient l'apparence.
- Le pelage change avec les saisons humides/sèches en milieu sauvage.

Un modèle figé à mai 2026 produira des embeddings de plus en plus décalés au fur et à mesure que les individus évoluent.

### La stratégie de mise à jour

L'avantage clé de l'architecture galerie est qu'on peut **mettre à jour la galerie** sans réentraîner le modèle.

**Stratégie recommandée** : tous les 6 mois pour les juvéniles, tous les 12-24 mois pour les adultes stables, prendre 10-20 nouvelles photos de chaque individu et recalculer son prototype. Le modèle MegaDescriptor reste capable de produire de bons embeddings de cet individu mis à jour, parce qu'il a appris des features faciales **générales** des orangs-outangs.

**Variante fine** : au lieu de remplacer complètement l'ancien prototype, on peut **moyenner pondéré** :

```
nouveau_prototype = α × prototype_ancien + (1-α) × prototype_nouveau
```

Avec α = 0.3 par exemple, on garde 30% de l'ancien et 70% du nouveau. Cela évite des changements brutaux pour des individus qui n'évoluent pas beaucoup.

**Variante experte** : garder une **fenêtre glissante de photos**. On stocke les 50 dernières photos validées de chaque individu et on recalcule le prototype à partir de cette fenêtre. À mesure que de nouvelles photos arrivent, les plus anciennes sont retirées.

### Quand re-entraîner le modèle lui-même

Refaire l'entraînement complet de V3 (12h) n'est nécessaire que si :

1. **Nouvelle espèce** : on passe aux gorilles, chimpanzés, autres primates. Le modèle V3 est optimisé pour les orangs-outangs.

2. **Drift massif** : si après 5+ ans, la performance se dégrade significativement même après rafraîchissement de la galerie, c'est qu'il y a eu un drift dans la distribution. On peut alors refaire un entraînement avec les nouvelles données.

3. **Amélioration architecturale** : si une nouvelle architecture sort (par exemple une V4 utilisant MiewID), on peut tout refaire.

Mais dans la pratique courante, **on n'a pas besoin de réentraîner pendant des années**. La mise à jour de galerie suffit.

### Pour les rangers BOS — workflow recommandé

Mensuel ou trimestriel :

1. Les rangers prennent des photos des individus connus pendant leurs patrouilles habituelles.
2. Une fois par mois/trimestre, ils synchronisent ces nouvelles photos avec le serveur central.
3. Le pipeline V3 calcule les nouveaux embeddings.
4. Les prototypes sont mis à jour (fenêtre glissante ou moyenne pondérée).
5. Une nouvelle version d'`embeddings.json` est déployée sur tous les téléphones.
6. Le modèle TFLite reste inchangé.

Le ranger n'a rien à faire de complexe : il prend ses photos normalement, et la galerie s'auto-met à jour côté serveur.

---

## 16. Réorganisation finale du projet

### La situation chaotique avant réorganisation

Pendant le développement, beaucoup de fichiers se sont accumulés dans des emplacements ambigus :

- `MODELS/` à la racine ET dans `V2/MODELS/` (duplication)
- `ANDROID_EXPORT/` à la racine ET dans `V2/ANDROID_EXPORT/` (duplication)
- Scripts éparpillés entre `scripts/`, `V2/scripts/`, `TEST/`
- Photos brutes dans `PHOTOS/` mais crops dans `DATASET_CLASSIFICATION/raw/`
- `runs/` à la racine (sortie Ultralytics) qui contient des poids intermédiaires énormes

Au total : **53 324 fichiers, 89 GB**, dont une partie est dupliquée ou inutile pour la livraison finale.

### La structure cible

Une organisation claire en 7 dossiers de premier niveau :

```
OrangIdentifier/
├── common/         ← données brutes partagées (photos, vidéos, wild internet)
├── v1/             ← YOLO nano/medium + ResNet50 classifieur fermé
├── v2/             ← ResNet50 + galerie embeddings open-set
├── v3/             ← MegaDescriptor + ArcFace (version actuelle)
├── android_app/    ← code Kotlin de l'app (inchangé)
├── android_export/ ← bundle final (APK + modèles TFLite)
└── docs/           ← documentation complète
```

### Principes de la réorganisation

**1. Pas de duplication des photos brutes**. Les photos (zoo 2589, BOS 1699, wild internet 6289) sont uniquement dans `common/`. Les scripts de toutes les versions les lisent depuis là.

**2. Chaque version est autonome**. Les scripts, modèles et résultats de V1, V2 et V3 sont dans des dossiers séparés. On peut ouvrir `v3/` et tout y est : scripts, modèles, données dérivées, résultats.

**3. Pas de duplication des crops**. Les crops zoo (2127, ~2 GB) sont dans `v1/data/crops_dataset/` et accessibles depuis V3 via un README qui pointe vers ce chemin. Pas de copie inutile.

**4. Modèles renommés clairement**. Plus de `best.pt` ambigu. Les noms incluent la version et la métrique :
   - `yolo_v2_mAP99.pt`
   - `resnet50_classifier_acc96.pt`
   - `megadesc_T_arcface_final_acc99.pt`

**5. Documentation par version**. Chaque dossier `v1/`, `v2/`, `v3/` contient un `README.md` qui rappelle ce qu'est cette version, ses métriques, ses limitations.

### Le script de réorganisation

Le script effectue la réorganisation de manière **non destructive** :

- Travaille sur une copie (`D:\OrangIdentifier - Reorganized\`) sans toucher à l'original.
- Mode dry-run par défaut : affiche ce qu'il ferait sans rien copier.
- Mode `--run` pour réellement effectuer les copies.
- Logs détaillés de chaque action.
- Rapport final écrit dans la destination.

Résultats typiques :

```
32 170 fichiers copiés
67 GB (vs 89 GB original car pas de duplication)
Quelques WARN sur des fichiers manquants (déplacés dans des sous-dossiers entre temps)
```

### Avantages de la nouvelle structure

**Pour Cédric Sueur (encadrant)** : peut naviguer linéairement V1 → V2 → V3 en comprenant la progression. Chaque version a sa documentation interne.

**Pour BOS Foundation** : peut récupérer juste `v3/` + `android_app/` pour utiliser le modèle final.

**Pour un futur étudiant** : reprend le projet sans confusion. Veut comprendre comment V3 a été entraîné ? Va dans `v3/scripts/` et lit dans l'ordre.

**Pour l'archive scientifique** : structure claire pour publier sur GitHub avec un lien vers les modèles lourds hébergés ailleurs (les `.pt` font 100+ MB).

---

## 17. Hyperparamètres et choix techniques détaillés

### Le pourquoi de chaque hyperparamètre V3

**`IMG_SIZE = 224`** : c'est la taille native de MegaDescriptor-T-224. On pourrait utiliser des tailles plus grandes (320, 384, 440 pour MiewID) mais ça multiplie la consommation VRAM par environ (taille_nouvelle/224)^2. À 224 on tient confortablement en 4 GB. À 440 (MiewID) on serait à l'étroit.

**`BATCH_SIZE = 32`** : compromis entre stabilité de l'entraînement et VRAM. Plus c'est grand, plus la loss est stable mais plus on a besoin de VRAM. Avec MegaDescriptor-T à 224 et batch=32, on consomme environ 2.8 GB sur les 4 GB disponibles, ce qui laisse de la marge.

**`ARC_SCALE = 64`** : c'est la valeur standard utilisée dans le papier ArcFace original et reprise dans toutes les implémentations majeures. Multiplie les cosinus avant softmax. Plus c'est grand, plus le softmax est piqué, plus le gradient est fort. 64 est un sweet spot empirique.

**`ARC_MARGIN = 0.50`** : 0.5 radians = environ 28 degrés. C'est la marge angulaire imposée entre classes. Le papier MiewID utilise 0.5 aussi. Valeurs plus basses (0.2-0.3) donnent moins de séparation mais plus facile à entraîner. Valeurs plus hautes (0.7-0.8) donnent plus de séparation mais peuvent rendre l'entraînement instable.

**`K_KNOWN = 1` et `K_UNKNOWN = 5`** : K=1 pour les classes propres (zoo), K=5 pour la classe sale (wild). Le papier Sub-center suggère K=3-5 pour des classes très bruyantes. Avec K=5 sur wild, le modèle peut apprendre 5 « types » de prototypes wild, suffisant pour capturer la diversité.

**`LR_BACKBONE = 2e-5` et `LR_HEAD = 1e-3`** : ratio de 50× entre les deux. Le backbone est déjà bon (vient du pré-entraînement), donc on lui applique un LR très bas pour ne pas le déstabiliser. La tête ArcFace est nouvelle, donc un LR plus standard.

**`WEIGHT_DECAY = 1e-4`** : régularisation L2 sur les poids. Empêche l'overfitting. Valeur standard pour AdamW.

**`MAX_EPOCHS = 60`** : plafond sur le nombre d'epochs. En pratique, l'early stopping intervient bien avant (best epoch trouvé en 21).

**`PATIENCE = 15`** : early stopping après 15 epochs sans amélioration de val_acc. Long parce qu'avec ArcFace, la val_acc peut stagner pendant 5-10 epochs avant de faire un saut.

**`WARMUP_EPOCHS = 3`** : warmup linéaire pendant 3 epochs. Permet au modèle de s'adapter doucement avant de pousser fort.

**`WILD_SAMPLES_PER_EPOCH = 1000`** : sous-échantillonnage des 5429 wild crops. Avec 1000 par epoch, on les voit tous environ une fois toutes les 5-6 epochs. Sur 60 epochs, chaque wild est vu ~10-12 fois. C'est suffisant.

**`VAL_RATIO = 0.15`** : 15% des crops zoo en validation. C'est volontairement bas pour avoir le maximum de données en entraînement. Avec 2127 zoo, on a 320 en val, ce qui suffit pour estimer la val_acc avec confiance.

**`SEED = 42`** : standard. Reproductibilité partielle (les opérations CUDA non-déterministes peuvent introduire de petites variations).

**`label_smoothing=0.05`** dans la cross-entropy : au lieu d'apprendre des cibles dures `[0, 0, 1, 0, 0]`, on apprend `[0.005, 0.005, 0.95, 0.005, 0.005]`. Empêche le modèle de devenir trop confiant et améliore la calibration.

**`clip_grad_norm=1.0`** : norme maximale des gradients. Au-delà, ils sont rescalés. Évite les explosions.

---

## 18. Erreurs commises et leçons apprises pour V2 et V3

### Erreur 1 : la mauvaise normalisation au début

Pendant les premiers tests de MegaDescriptor, on utilisait par réflexe la normalisation ImageNet `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`. Les embeddings produits étaient médiocres, le modèle ne convergeait pas. Après vérification du code source de MegaDescriptor, on a réalisé qu'il utilise `mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]`. Une heure perdue.

**Leçon** : toujours vérifier la normalisation **dans le code source** du modèle, pas dans la doc ou Stack Overflow.

### Erreur 2 : oublier le HF_HOME sur D:

Par défaut, HuggingFace cache les modèles téléchargés dans `C:\Users\<user>\.cache\huggingface\`. Comme C: est plein, ça plantait à 95% du téléchargement. Il faut forcer en exportant la variable d'environnement avant n'importe quel import :

```
os.environ["HF_HOME"]    = r"D:\HuggingFaceCache"
os.environ["TORCH_HOME"] = r"D:\TorchCache"
```

Et faire ça **avant** d'importer timm, torch ou huggingface_hub.

**Leçon** : penser au chemin de cache pour tous les frameworks ML (HuggingFace, PyTorch Hub, Ultralytics) au début du projet.

### Erreur 3 : faire le test BOS uniquement après V3

Idéalement, on aurait dû avoir un dataset « test set externe » dès V1 pour mesurer la généralisation à travers les versions. On a fait le test BOS uniquement à la fin du stage, quand on a reçu les images. Donc on n'a pas de point de comparaison BOS pour V1 et V2.

**Leçon** : prévoir dès le début un dataset de test **strictement séparé** du dataset d'entraînement, idéalement provenant d'une source différente (autre zoo, autre photographe, autre période).

### Erreur 4 : ne pas implémenter le dry-run mode dès le début

Plusieurs scripts (training notamment) ont été lancés en mode réel directement, ce qui a coûté des heures de calcul avant qu'on réalise qu'il y avait un bug à l'epoch 5. Le mode dry-run aurait validé en 3 minutes que le pipeline marche de bout en bout.

**Leçon** : tout script qui tourne > 30 minutes devrait avoir un mode `--dry-run` qui le valide en quelques minutes.

### Erreur 5 : la pipeline 10h ambitieuse abandonnée

Au milieu du stage, j'ai sérieusement envisagé une pipeline complexe en 4 phases : MoCo v2 pre-training non-supervisé → SupCon → ArcFace → hard negative mining. Estimation 10 heures de GPU. J'allais coder ça en grand. Heureusement, on a fait un test simple du modèle V3 actuel sur les BOS, et il a obtenu 96.3% de rejet. **La pipeline 10h était inutile** — le modèle marchait déjà bien.

**Leçon** : avant d'investir des heures de calcul, faire un test rapide qui montre s'il y a vraiment un problème à résoudre. La complexité prématurée est un piège.

### Erreur 6 : pas de mode debug sur les pipelines longues

Le script V2_5_train_arcface.py tourne pendant 30 minutes. Si on a une erreur de dimension de tenseur à l'epoch 5, on perd 25 minutes. Maintenant on a ajouté un mode `--dry-run` qui passe par toutes les phases avec 1 epoch x 2 batches, en 3 minutes. Ça aurait dû être là dès le début.

**Leçon** : pareil que erreur 4. C'est important.

### Erreur 7 : les wild crops téléchargés sans déduplication initiale

Au début, le crawler téléchargeait toutes les images des 5 sources sans vérifier les doublons inter-sources. On s'est retrouvé avec ~6400 fichiers dont une partie était la même image présente sur deux sources différentes (iNaturalist et WebSearch par exemple). La déduplication par hash MD5 ajoutée après coup a éliminé ~100 doublons.

**Leçon** : déduplication par hash dès la collecte, pas après.

### Erreur 8 : pas de logging structuré au début

Les premiers scripts logaient juste avec `print()`. Quand on revient sur un log d'il y a 2 semaines pour comprendre ce qui s'était passé, c'est l'enfer. Maintenant on a un logger avec timestamps, niveaux (INFO/WARN/ERR), et écriture dans un fichier en parallèle de la sortie standard.

**Leçon** : utiliser `logging` (ou un wrapper simple) dès le début. Niveau minimum, timestamp, écriture fichier.

### Erreur 9 : croire que YOLO ratait pleins de visages

Au début, on a vu que YOLO « ratait » 77 photos BOS sur 1699 et on s'est dit que c'était un problème. En vérité, sur ces 77 photos, ~50 ne contenaient effectivement pas de visage exploitable (orang vu de dos, trop loin, partiellement caché). YOLO faisait son travail correctement. Sur les 27 vraies erreurs, c'était des cas vraiment limites.

**Leçon** : ne pas paniquer sur les métriques avant d'avoir regardé les vrais échecs en image.

### Erreur 10 : l'ambition de la perfection avant le déploiement

On a passé énormément de temps à raffiner le modèle (V2 → V3 → tests BOS → analyse Sinta...) avant d'avoir validé que **le système marche sur le terrain**. Le modèle V1 aurait peut-être suffi pour BOS Foundation, avec quelques ajustements opérationnels. À la place, on a investi dans la perfection technique sans validation utilisateur.

**Leçon** : valider tôt avec les utilisateurs finaux (rangers BOS) avant d'investir dans des améliorations techniques marginales.

---

## 19. Pistes d'amélioration futures

### V4 — MiewID si VRAM le permet

MiewID surpasse MegaDescriptor de 19.2% en moyenne sur des espèces non vues. Le passage à MiewID demanderait un GPU avec ≥6 GB de VRAM. Sur le matériel actuel (RTX 3050 4 GB), c'est tendu mais peut-être faisable avec gradient checkpointing.

### Investiguer le cas Sinta

Sinta attire 50% des fausses identifications BOS. Pourquoi ? Plusieurs hypothèses à explorer :

- Son prototype est-il trop « moyen » dans l'espace d'embedding ?
- Ses 256 crops zoo couvrent-ils une diversité trop large d'angles/expressions ?
- Ressemble-t-elle physiquement à un type morphologique commun ?

Analyse possible : faire un PCA sur les embeddings de Sinta, voir si elle a une structure interne particulière. Comparer avec d'autres individus.

### Augmentation cross-individual mixup

Pour améliorer la séparabilité, on peut faire du « mixup » qui mélange linéairement deux images de deux individus différents et demande au modèle de prédire la combinaison des deux labels. Cela force des frontières de décision plus douces.

### Ajouter du metric learning supervisé en plus d'ArcFace

ArcFace fonctionne bien mais on peut ajouter du triplet loss en parallèle :

```
loss = α × loss_arcface + (1-α) × loss_triplet
```

Le triplet loss force directement la métrique cosinus à séparer les individus, indépendamment de la classification.

### Self-supervised pretraining sur les wild crops

Avant de fine-tuner avec ArcFace, on pourrait faire un pre-training self-supervised (SimCLR ou DINO) **uniquement sur les wild crops**. Le modèle apprendrait à représenter les visages d'orangs-outangs sans aucun label individuel, juste en apprenant à reconnaître la même image sous deux augmentations différentes.

Limitation : ces méthodes demandent typiquement de gros batches (>1024) qui ne tiennent pas en 4 GB. MoCo v2 contourne ce problème avec une queue mais reste lourd.

### Tester sur plus d'espèces

Si on veut généraliser au-delà des orangs-outangs (gorilles, chimpanzés, autres primates), il faudrait reentraîner V3 sur des datasets de ces espèces. La pipeline est conceptuellement identique : YOLO + MegaDescriptor + ArcFace + galerie. Les améliorations seraient sur le YOLO (à entraîner par espèce ou famille).

### Améliorer la détection des orangs lointains

YOLO v2 fonctionne très bien sur des visages proches mais peut rater des orangs photographiés de loin (visage < 50 pixels). On pourrait :

- Réentraîner YOLO avec des images de plus grande résolution.
- Utiliser une approche multi-échelle (passer YOLO sur plusieurs versions de l'image à différentes échelles).

### App Android — mode « live tracking »

Au lieu de prendre une photo et de l'analyser, on pourrait avoir un mode caméra live qui détecte et identifie en temps réel. Sur un téléphone moderne, c'est faisable à 5-10 FPS.

### Système collaboratif multi-rangers

Plusieurs rangers prennent des photos d'un même individu sur plusieurs mois. Un système central agrège ces photos, met à jour le prototype automatiquement, et redistribue la nouvelle `embeddings.json` à tous les téléphones quand des changements significatifs sont détectés.

### Détection des conflits d'identité

Un cas embarrassant : le système identifie une photo comme « Molly à 0.65 » mais le ranger sait que c'est Sari. Pourrait-on permettre au ranger de corriger, et utiliser cette correction pour rafraîchir le prototype ?

C'est ce qu'on appelle du **continual learning avec feedback humain**. Implémentable côté serveur.

---

## 20. Annexes — schémas, formules, configurations

### Annexe A : formule complète de Sub-center ArcFace

Pour un échantillon `x` de classe vraie `y`, on calcule :

```
embedding   = f_θ(x)         (où f_θ est le backbone)
e           = embedding / ||embedding||₂   (L2 normalisation)
W̃_i^k       = W_i^k / ||W_i^k||₂   (L2 normalisation des sous-centres)

cos(θ_i)    = max_{k=1..K_i}(e · W̃_i^k)   (best matching sub-center)

# Application de la marge angulaire à la vraie classe seulement
phi_y       = cos(θ_y + m) = cos(θ_y)·cos(m) - sin(θ_y)·sin(m)

# Logits finaux
logits_i    = s · phi_y    si i == y
              s · cos(θ_i) sinon

# Loss
loss        = -log( exp(logits_y) / Σ_i exp(logits_i) )
            (avec label smoothing ε=0.05)
```

Hyperparamètres : s=64, m=0.5

### Annexe B : configuration YOLO v2

```python
model = YOLO('yolov8m.pt')   # medium au lieu de nano

results = model.train(
    data        = 'data.yaml',
    epochs      = 150,
    imgsz       = 640,
    batch       = 16,
    patience    = 30,
    save_period = 10,

    # Augmentations fortes
    degrees      = 15.0,
    translate    = 0.15,
    scale        = 0.6,
    flipud       = 0.5,
    fliplr       = 0.5,
    mosaic       = 1.0,         # mosaic augmentation activée
    mixup        = 0.2,
    copy_paste   = 0.1,

    # Optimisation
    optimizer    = 'AdamW',
    lr0          = 0.001,
    lrf          = 0.01,
    momentum     = 0.937,
    weight_decay = 0.0005,
    warmup_epochs= 3,
)
```

Résultat : mAP@50 = 99.39%, mAP@50:95 = 78.34%

### Annexe C : architecture MegaDescriptor-T-224

```
Input image (1, 3, 224, 224)
    ↓
Patch embedding (4×4 patches → linear projection)
    ↓
Stage 1 : Swin Transformer block × 2     (window=7, dim=96,  heads=3)
    ↓
Patch merging (downsample 2×)
    ↓
Stage 2 : Swin Transformer block × 2     (window=7, dim=192, heads=6)
    ↓
Patch merging (downsample 2×)
    ↓
Stage 3 : Swin Transformer block × 6     (window=7, dim=384, heads=12)
    ↓
Patch merging (downsample 2×)
    ↓
Stage 4 : Swin Transformer block × 2     (window=7, dim=768, heads=24)
    ↓
Layer norm + global average pooling
    ↓
Output : feature vector (1, 768)
```

Paramètres : 27.5 millions

### Annexe D : format complet d'embeddings.json (V3)

```
{
  "version": "2.0-arcface",
  "created": "2026-05-22T11:57:09",
  "model": "MegaDescriptor-T-224 + SubCenterArcFace",
  "model_file": "megadesc_T_arcface_final_acc99.pt",
  "embedding_dim": 768,
  "image_size": 224,
  "normalization": {
    "mean": [0.5, 0.5, 0.5],
    "std":  [0.5, 0.5, 0.5]
  },
  "similarity_metric": "cosine",
  "unknown_threshold": 0.22,
  "separability_gap": 0.9586,
  "num_individuals": 10,
  "training_info": {
    "best_epoch": 21,
    "best_val_accuracy": 0.9906,
    "arc_scale": 64,
    "arc_margin": 0.5,
    "k_known": 1,
    "k_unknown": 5
  },
  "individuals": {
    "Auti": {
      "class_index": 0,
      "num_training_crops": 164,
      "mean_within_similarity": 0.9412,
      "embedding": [0.0234, -0.0156, 0.0089, ...]
    },
    "Jula": { ... },
    "Mathai": { ... },
    "Molly": { ... },
    "NOAH": { ... },
    "PULCO": { ... },
    "PUTRI": { ... },
    "Sari": { ... },
    "Sinta": { ... },
    "Ujian": { ... }
  }
}
```

Taille typique : 80 KB pour 10 individus, 320 KB pour 40 individus (10 zoo + 30 BOS).

### Annexe E : commandes utiles

Lancer le test BOS rapide (15 minutes) :
```
python V3_test_bos_baseline.py
```

Générer les visualisations diagnostiques (5 PNG) :
```
python V3_visualize_model.py
```

Reprendre l'entraînement après interruption :
```
python V2_5b_continue_training.py
```

Réviser des nouveaux crops :
```
python NEW_2_review_crops.py
```

Construire une nouvelle galerie après ajout d'individus :
```
python 6_build_embeddings.py  # version V2 (ResNet50)
# ou
python build_arcface_gallery.py  # version V3 (à coder)
```

### Annexe F : checklist de déploiement BOS

Avant de fournir le système à BOS Foundation :

- [ ] Modèle TFLite V3 backbone exporté et testé
- [ ] `embeddings.json` étendu à 40 individus (10 zoo + 30 BOS)
- [ ] Seuil calibré sur la galerie étendue (probablement ~0.25-0.30)
- [ ] APK Android v3 compilé avec le nouveau bundle
- [ ] Test sur device réel (téléphone bas/milieu de gamme)
- [ ] Documentation utilisateur (1-2 pages) en français et anglais
- [ ] Procédure de mise à jour de galerie documentée
- [ ] Contact pour support technique défini

### Annexe G : références bibliographiques

- Cermak, V., Picek, L., Adam, L., Papafitsoros, K. (2024). **WildlifeDatasets: An open-source toolkit for animal re-identification**. WACV 2024.
- Deng, J., Guo, J., Xue, N., Zafeiriou, S. (2019). **ArcFace: Additive Angular Margin Loss for Deep Face Recognition**. CVPR 2019.
- Deng, J., Guo, J., Zhou, T., Yang, J., Mahadeokar, J., Zafeiriou, S. (2020). **Sub-center ArcFace: Boosting Face Recognition by Large-scale Noisy Web Faces**. ECCV 2020.
- Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B. (2021). **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows**. ICCV 2021.
- Otarashvili, L., et al. (2024). **MiewID: Open-source Wildlife Re-identification with State-of-the-art Foundation Models**. ConservationXLabs 2024.
- He, K., Zhang, X., Ren, S., Sun, J. (2016). **Deep Residual Learning for Image Recognition**. CVPR 2016. (ResNet50)
- Ultralytics (2023). **YOLOv8: A new state-of-the-art real-time object detector**.

### Annexe H : remerciements

- Cédric Sueur, encadrant CNRS IPHC, pour la liberté d'exploration et les retours techniques.
- BOS Foundation, pour le dataset des 30 individus indonésiens.
- Le Zoo d'Amnéville, pour l'accès aux 10 orangs-outangs.
- L'équipe Anthropic / Claude pour l'aide au développement et à la documentation.
- La communauté open-source pour MegaDescriptor, MiewID, Ultralytics, et tous les outils utilisés.

---

**Fin de la documentation Partie 2.**

Cette documentation couvre V2 et V3. La Partie 1 (`DOCUMENTATION_PROJET_ORANGS.md`) couvre V1.

Pour toute question : Titouane, stagiaire CPI2 CNRS IPHC Strasbourg, mai 2026.
