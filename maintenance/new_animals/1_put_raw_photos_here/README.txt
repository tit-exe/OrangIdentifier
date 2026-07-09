PUT YOUR PHOTOS HERE
====================

Make one folder per animal, named exactly with the animal name, and put that
animal's photos inside.

Example:

    1_put_raw_photos_here/
        Bella/
            photo1.jpg
            photo2.jpg
            ...
        Kodi/
            photo1.jpg
            ...

Rules:
 - One folder per animal. The folder name is the name the app will show.
 - Use clear photos where the face is visible.
 - 15 to 30 photos per animal is a good amount (more is better for lookalikes).
 - Accepted picture types: .jpg .jpeg .png

Then run the cropping step:

    conda activate orangs
    python maintenance\scripts\crop_photos.py

The cropped heads will appear in the folder next to this one,
2_crops_appear_here.
