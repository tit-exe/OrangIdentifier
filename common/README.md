# common/ — Outils partagés entre toutes les versions

## review_crops.py — Reviewer unifié
Remplace tous les anciens reviewers (3b_reviser_faces.py, V2_4_review_crops.py, etc.)

```bash
python common/review_crops.py <dossier_crops>
# ou avec un JSON de boxes existant :
python common/review_crops.py <dossier_crops> --boxes boxes.json
```

**Contrôles :**
- `Espace` — valider le crop
- `S` — passer (skip)
- `Suppr` — supprimer (crop + entrée JSON)
- `←` / `→` — naviguer
- Handles sur la bbox pour redimensionner

## benchmark.py — Comparaison V1-V4

```bash
python common/benchmark.py --models-dir models/
```

## download_models.py — Téléchargement depuis HuggingFace

```bash
python models/download_models.py --version all
```
