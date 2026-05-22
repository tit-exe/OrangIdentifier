# OrangIdentifier

version 1

---

## Comment ça marche

Le pipeline se déroule en deux étapes :

1. **Détection (YOLO)** — localise et découpe le visage de l'orang-outan dans la photo
2. **Identification (ResNet50)** — compare le visage détecté à une galerie d'individus connus et retourne l'identité ou "inconnu"

Le modèle de reconnaissance produit un vecteur numérique (embedding) pour chaque visage. On compare ensuite ce vecteur à ceux des individus connus par similarité cosinus. En dessous d'un certain seuil, l'individu est déclaré inconnu.

---

## Performances (V1)

| Étape | Métrique | Résultat |
|---|---|---|
| Détection YOLO | mAP50 (test set) | 99,4 % |
| Détection YOLO | Sur 2 589 photos brutes | 94,1 % |
| Pipeline complet | Sur 2 589 photos brutes | 96,3 % |
| Identification ResNet50 | Val accuracy (10 individus) | 99,1 % |
| Séparabilité des clusters | | 1,72 |

---

## Dataset

- **2 127 photos de visages** — 10 individus connus, annotés manuellement
- Split entraînement / validation : 85 % / 15 %, stratifié par individu

Les images ne sont pas incluses dans ce dépôt.

## Modèles utilisés

- **Détection** : YOLOv8, fine-tuné sur ~500 images annotées manuellement, puis réentrainé avec les 2 217 photos annotées manuellement.
- **Identification** : ResNet50, fine-tuné avec une loss par embeddings sur les 10 individus connus