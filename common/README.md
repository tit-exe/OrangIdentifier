# common/

Shared utilities used by every version of the pipeline.

## review_crops.py

Unified crop reviewer (replaces all the old per-version reviewers).

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

## config_loader.py

Single source of truth for all paths and settings. Every script imports from here
instead of hardcoding paths. Edit `config.yaml` (at the repository root) to adapt the
pipeline to a new machine or a new species.
