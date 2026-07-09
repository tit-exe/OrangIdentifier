# OrangIdentifier maintenance

This folder explains how to update the OrangIdentifier app when you want to add
new animals or improve recognition. It is written for someone who is not a
computer specialist. Follow the steps in order and copy the commands exactly.

## What the app is made of

The app recognizes an animal in three steps:

1. It finds the head in the photo (a small detector called YOLO).
2. It turns that head into a list of 768 numbers (a model called the "brain").
3. It compares those numbers to a list of known animals (a file called the
   "gallery") and picks the closest match, or says "unknown".

So there are two things that can be changed:

- The **gallery**: the list of known animals. Small, quick to rebuild.
- The **brain**: the part that turns a head into numbers. Big, slow to retrain.

The head detector never changes.

## Which option do I need

Answer one question: **do the new animals look very similar to each other?**

```
   Do the new animals look almost identical to each other?
                      |
          +-----------+-----------+
          |                       |
         NO                      YES
          |                       |
     OPTION A                 OPTION B
  add to the gallery       retrain the brain
  (fast, no GPU)           (slow, needs a GPU)
```

### Option A: add individuals (the normal case)

Use this for new zoo animals, or any animals that look clearly different from
each other. You only rebuild the gallery. The brain stays the same, so you only
copy one small file into the app.

- No graphics card needed.
- Takes about 20 to 30 minutes.
- Folder: `option_A_add_individuals/`

### Option B: retrain the brain (the hard case)

Use this only when the animals look almost identical (for example the BOS
animals) and Option A gives too many confusions. Here the brain itself is
retrained to tell them apart.

- Needs a graphics card (GPU).
- Takes several hours.
- Needs WSL (a small Linux inside Windows) for the last step.
- Folder: `option_B_retrain_brain/`

## Important: you cannot "retrain the .tflite file"

The `.tflite` file inside the app is a finished, read-only version of the brain.
It cannot be trained. Training always happens on the PC in the `.pt` format, and
only at the very end is it turned into a `.tflite` file. So the real choice is
always "gallery only" (Option A) or "retrain the brain" (Option B).

## First time only

Before your very first run, install the tools once. See `00_first_time_setup/`.
Option A needs only the Windows tools. Option B also needs WSL.

## Folder map

```
maintenance/
  README.md                  you are here
  00_first_time_setup/       install the tools once
  new_animals/               where you drop your photos and where crops appear
  option_A_add_individuals/  steps to add animals without retraining
  option_B_retrain_brain/    steps to retrain the brain
  scripts/                   the programs the steps use
  reference/                 how it works, file map, history, troubleshooting
```

If anything goes wrong, open `reference/troubleshooting.md`. It lists the errors
we have already hit and how to fix each one.
