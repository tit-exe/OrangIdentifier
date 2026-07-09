# How it works

A short, plain explanation of what happens when the app recognizes an animal.
You do not need this to run the steps, but it helps to understand the choices.

## The three parts

1. **Head detector (YOLO).** It looks at the photo and finds where the head is.
   It draws a box around it. This part never changes.

2. **The brain (backbone).** It takes the head picture and turns it into a list
   of 768 numbers. These numbers are like a fingerprint of the face. Two photos
   of the same animal give almost the same numbers. Two different animals give
   different numbers. This is the part that is slow to train.

3. **The gallery.** It is a file that holds, for every known animal, a few of
   these fingerprints (called reference vectors). To recognize a new photo, the
   app makes its fingerprint and compares it to every animal in the gallery.

## How the match is decided

- The app makes the fingerprint of the new photo.
- For each known animal, it measures how close the new fingerprint is to that
  animal's reference vectors, and keeps the best match.
- The animal with the highest match wins.
- But if even the best match is below a limit (the "unknown threshold"), the app
  answers "unknown". This is what stops it from naming a random animal it has
  never seen.

## Why there are two options

- Adding an animal to the **gallery** (Option A) just adds new fingerprints. The
  brain already knows how to make good fingerprints for animals that look
  different, so this is enough most of the time, and it is fast.

- Retraining the **brain** (Option B) is only needed when animals look almost
  identical. Then the brain has to learn finer differences, which the fixed
  brain cannot see on its own. This is slow and not always successful, because
  if the animals truly look the same and there are few photos, there is only so
  much the brain can learn.

## Why the photos must be cropped first

The brain was trained on tight 224 by 224 head pictures, always prepared the
same way. If you feed it a whole photo, or a differently prepared picture, the
fingerprints are wrong. That is why every photo goes through the same cropping
step before anything else.

## The important numbers

- **Separation gap**: how far apart the animals are in fingerprint space. Higher
  is better. A small gap means animals are easy to confuse.
- **Unknown threshold**: the limit below which the app says "unknown". It is
  chosen automatically when the gallery is built.
- **Accuracy (clean and degraded)**: how often the model is right on good
  pictures and on hard pictures (dark, blurry, far).
