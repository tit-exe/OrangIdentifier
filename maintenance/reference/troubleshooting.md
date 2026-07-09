# Troubleshooting

This lists the problems we have actually hit, and the fix for each one. Find the
message you see and follow the fix.

## During WSL setup or export

### libGL.so.1: cannot open shared object file

A system graphics library is missing inside WSL. Install it:

```
wsl -d Ubuntu -- bash -c "apt-get install -y libgl1 libgl1-mesa-glx"
```

The current `setup_wsl.ps1` already installs this, so you should not see it if
you used that script.

### module 'ai_edge_torch' has no attribute 'convert'  (or 'no attribute __version__')

The converter package was renamed from `ai-edge-torch` to `litert-torch`. The
old one no longer works. Install the current one:

```
wsl -d Ubuntu -- bash -c "/root/miniconda3/bin/conda run -n orangs_export pip install -U litert-torch"
```

The export script already tries `litert_torch` first, so with the current setup
this is handled for you.

### unexpected EOF while looking for matching quote  (in PowerShell)

This happens when a command has quotes inside quotes. The safest way is to run
the export exactly as written in `step_5_export_tflite_wsl.md`, in one line,
without changing the quotes.

If you want to test a small Python snippet in WSL, put the code in single quotes
and avoid double quotes inside it. For example, use `print(42)` rather than
`print("OK")`.

### only X GB free on C:

WSL is created on C: by default and needs a few GB. If C: is almost full:

- The `setup_wsl.ps1` script puts WSL on D:, which usually avoids this.
- If the very first `wsl --install` fails, free some space on C: and try again.
- To move an existing Ubuntu to D:, run these four lines in PowerShell (this
  keeps your Ubuntu, it just moves it):

  ```
  wsl --shutdown
  wsl --export Ubuntu D:\WSL\ubuntu_backup.tar
  wsl --unregister Ubuntu
  wsl --import Ubuntu D:\WSL\Ubuntu D:\WSL\ubuntu_backup.tar --version 2
  ```

### The setup script printed some red lines

Run it again. It skips what is already installed, so a second run often finishes
the missing parts. If the same package keeps failing, check your internet
connection, then run the script once more.

## During cropping (crop_photos.py)

### No animal folder inside ...

Your photos are not arranged correctly. Inside
`new_animals/1_put_raw_photos_here/`, there must be one folder per animal, named
with the animal name, with the photos inside. Photos placed loose (not in a
named folder) are not seen.

### No head found on many photos

The detector could not find a head. This is normal for a few photos. If it
happens on most photos of an animal, the photos may be too far, too dark, or the
face may be hidden. Use closer, clearer photos.

### CUDA out of memory / very slow

Cropping falls back to the CPU if there is no graphics card, which is slower but
works. If you have a card and still run out of memory, close other programs that
use the card and try again.

## During training (train_brain.py)

### It looks stuck, no new line for a long time

Training is slow. Each epoch can take many minutes. As long as the live panel is
still updating (the batch number changes), it is working. If the whole window is
frozen and nothing changes for a very long time, close it and run the same
command again. It continues where it stopped.

### I closed the window by accident

Nothing is lost. Run the same command again. It reads the last checkpoint and
continues from the last finished epoch.

### Cannot stratify (an animal has < 2 crops)

An animal has only one crop. The script continues with a random split, but that
animal will barely be learned. Add more crops for that animal.

### The new model is worse than before

Training does not always improve things, especially for animals that look almost
identical. The old model is safe in the `models/` folder (the downloaded V6 model). To go back, copy
the old files back into the app. See `file_map.md` for the exact names.

## In the app

### A new animal is recognized as someone else

If you used Option A and this keeps happening for one animal, that animal
probably looks too similar to an existing one for the current brain. Try Option
B for a stronger result, but be aware it may still be hard if the animals truly
look the same.

### Everyone is recognized as "unknown"

The gallery threshold may be too high, or the wrong gallery was copied. Rebuild
the gallery (Option A, step 3) and make sure the new `gallery.json` reached the
app assets, then rebuild the app.
