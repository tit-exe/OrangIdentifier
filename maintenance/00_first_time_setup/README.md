# First time setup

Do this once, before your first run. After that you can skip straight to
Option A or Option B.

## What you need for each option

| | Windows tools | WSL (Linux) |
|---|---|---|
| Option A (add individuals) | yes | no |
| Option B (retrain the brain) | yes | yes |

So:

- If you only want to add animals (Option A), do part 1 below.
- If you want to retrain the brain (Option B), do part 1 and part 2.

## Part 1: Windows training tools

This installs Python and the packages used to crop photos, build the gallery,
and train. See `1_install_training_tools.md`.

## Part 2: WSL, only for Option B

This installs a small Linux inside Windows, used only to turn the trained brain
into the `.tflite` phone file. See `2_install_wsl_for_export.md`. A script does
almost all of it for you.

## How do I know it worked

Each part ends with a check. If the check prints errors, open
`../reference/troubleshooting.md`, which lists the common problems and the fix
for each one.
