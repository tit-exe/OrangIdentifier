# Step 5: turn the new brain into a phone file (WSL)

After `train_brain.py` finishes, you have a new brain in `.pt` format at
`../new_brain/models/new_backbone_only.pt`. The phone cannot read that format.
This step turns it into a `.tflite` file and copies it into the app. It runs
inside WSL, because the converter only works on Linux.

Before this, WSL must be set up once. See
`../00_first_time_setup/2_install_wsl_for_export.md`.

## Run the export

1. Right click the Start button and choose "Windows PowerShell".
2. Copy and paste this one line and press Enter:

   ```
   wsl -d Ubuntu -- bash -c "/root/miniconda3/bin/conda run -n orangs_export --no-capture-output python /mnt/d/<path-to-your-repo>/maintenance/scripts/export_to_tflite.py"
   ```

It prints its progress. The conversion itself takes about 3 to 8 minutes. At the
end it prints "DONE" and shows the two files it copied into the app.

## What it does

- Reads `new_backbone_only.pt` (the new brain).
- Makes `new_backbone.tflite`.
- Copies it into the app as `megadesc_v6_backbone.tflite` (the app expects that
  exact name, even though the brain inside is your new one).
- Copies `new_gallery.json` into the app as `gallery.json`.
- Does not touch the head detector.

## Common problems

If you see `libGL.so.1: cannot open shared object file`, WSL is missing a
system library. Run:

```
wsl -d Ubuntu -- bash -c "apt-get install -y libgl1 libgl1-mesa-glx"
```

If you see `module 'ai_edge_torch' has no attribute 'convert'`, the converter
package was renamed. Install the current one:

```
wsl -d Ubuntu -- bash -c "/root/miniconda3/bin/conda run -n orangs_export pip install -U litert-torch"
```

More errors and fixes are in `../reference/troubleshooting.md`.

## After the export

Go back to `README.md`, step 6: rebuild the app in Android Studio and test it on
a phone.
