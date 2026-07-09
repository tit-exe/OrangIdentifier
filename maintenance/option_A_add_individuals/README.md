# Option A: add individuals without retraining

Use this when the new animals look clearly different from each other. You will
add them to the gallery using the existing brain. The brain file does not
change, so at the end you only copy one small file into the app.

You need about 20 to 30 minutes. You do not need a graphics card.

Before you start, make sure the tools are installed once. See
`../00_first_time_setup/1_install_training_tools.md`. Option A does not need WSL.

## Step 1: put your photos in place

Go to `../new_animals/1_put_raw_photos_here/`.

Make one folder per animal, named exactly with the animal name. Put that
animal's photos inside. Example:

```
new_animals/1_put_raw_photos_here/
    Bella/
        photo1.jpg
        photo2.jpg
        ...
    Kodi/
        photo1.jpg
        ...
```

Tips:
- Use clear photos where the face is visible.
- 15 to 30 photos per animal is a good amount.
- The name of the folder is the name the app will show, so spell it the way you
  want it to appear.

## Step 2: cut the heads out of the photos

Open the Anaconda Prompt and run these two lines:

```
conda activate orangs
python maintenance\scripts\crop_photos.py
```

This finds the head in every photo and saves a clean square picture in
`../new_animals/2_crops_appear_here/`. Photos where no head is found are
reported at the end. You can run this again any time, already-done photos are
skipped.

Have a quick look in `2_crops_appear_here/`. If some crops are wrong (a hand, a
background, the wrong animal), just delete those individual crop files.

## Step 3: rebuild the gallery

Run:

```
conda activate orangs
python maintenance\scripts\build_gallery.py
```

This reads every animal (the original ones plus your new ones), computes their
reference vectors with the existing brain, and writes a new file
`new_animals/updated_gallery.json`. The original V6 gallery is left untouched as
a backup. If `DEPLOY_TO_APP` is left on (it is on by default), it also copies the
new gallery into the app, replacing the app's `gallery.json`.

Watch the numbers it prints:
- "Same animal" should be clearly higher than "Other animals".
- The "Separation gap" should be positive and not tiny. A larger gap means the
  animals are easier to tell apart.

If one new animal shows a very low separation from the others, it means that
animal looks too similar to an existing one. That is the sign you may need
Option B instead for that case.

## Step 4: put the new gallery in the app and rebuild

If `DEPLOY_TO_APP` was on, the gallery is already copied. Now:

1. Open the project in Android Studio.
2. Menu: Build, then Clean Project.
3. Menu: Build, then Rebuild Project.
4. Run it on a phone and test the new animals.

That is all. The brain file was not touched, only the gallery changed.

## If something goes wrong

See `../reference/troubleshooting.md`.
