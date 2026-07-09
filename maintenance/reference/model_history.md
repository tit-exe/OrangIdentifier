# Model history

A short history of the versions, so you know what the current model is and why.
This matches the version folders in the repository (v1 to v6).

## The versions

- **V1.** YOLO face detector plus a ResNet50 closed classifier for the 10 zoo
  animals. It always returns a name, even for an unknown animal, so it cannot be
  used in the field on its own.

- **V2.** Same ResNet50 used as an embedding gallery instead of a classifier. It
  can now say "unknown", but it rejects unknown animals only about 27 percent of
  the time, which is too weak.

- **V3.** The first strong brain. It switches to MegaDescriptor-T-224 trained with
  Sub-center ArcFace, plus thousands of internet crops as an "unknown" background
  class. It recognizes the 10 zoo animals at 99 percent and rejects unknown
  animals about 96 percent of the time. Training for the later versions still
  starts from V3, because it is a clean, animal-only starting point.

- **V4.** Adds the 30 rescue-center (BOS) animals as known classes (40 animals in
  total) and stronger augmentations for blur and low resolution. It improves low
  resolution but not blur, and the look-alike BOS animals crowd the gallery.

- **V5.** The turning point for robustness. It trains the brain to ignore image
  damage (an "invariance" objective: each crop is shown clean and degraded and
  the two fingerprints are forced to match), with a degradation curriculum. These
  are the methods that make V6 robust.

- **V6.** The current model. Zoo animals only (no BOS). It has 15 animals (the 10
  original plus 5 new). It reuses the V5 methods and adds a crop quality step. It
  is the strongest zoo model: about 98 percent recognition on validation images,
  about 96 percent even on degraded images, and it correctly rejects unknown
  animals. This is the brain the maintenance scripts start from and the one in the
  app today.

## What this means for you

- The app today uses the **V6** brain and its gallery, with 15 zoo animals.
- To add more animals that look different, use **Option A**. It uses the V6 brain
  as is.
- To add animals that look almost identical, use **Option B**. It retrains a new
  brain from V3 with all the animals together, which is the approach that has the
  best chance for look-alikes. (This is the same lesson learned with the BOS
  animals: look-alikes have to be trained in from the start, not added on top of a
  fixed brain.)

## The key numbers of V6 (for reference)

- 15 animals in the gallery.
- Recognition about 98 percent on validation pictures.
- About 96 percent recognition even on degraded (blurred, low resolution) pictures.
- Unknown animals rejected correctly most of the time.
- Each animal is stored with up to 25 reference vectors of 768 numbers.
