# common/

## review_crops.py
Replaces all the old reviewers (3b_reviser_faces.py, V2_4_review_crops.py, etc.)

```bash
python common/review_crops.py <crops_folder>
# or with an existing boxes JSON:
python common/review_crops.py <crops_folder> --json boxes.json
```

**Controls:**
- `Enter` : validate the crop
- `S` : skip
- `Del` / `X` : delete (crop + JSON entry)
- `A` / `←` : previous
- Handles on the bounding box to resize

## benchmark.py

```bash
python common/benchmark.py --models-dir models/
```

## download_models.py

```bash
python models/download_models.py --version all
```
