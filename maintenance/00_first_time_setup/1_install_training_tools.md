# Part 1: Windows training tools

These are the tools used on the Windows side: cropping photos, building the
gallery, and training. You need them for both Option A and Option B.

## Step 1: install Anaconda

Anaconda is a program that manages Python and its packages.

1. Go to https://www.anaconda.com/download
2. Download the Windows version and install it with the default choices.
3. After it is installed, open "Anaconda Prompt" from the Start menu. A black
   window opens. You type the commands there.

## Step 2: create the environment

An "environment" is a clean box that holds the right versions of everything. In
the Anaconda Prompt, type these lines one by one:

```
conda create -n orangs python=3.10 -y
conda activate orangs
```

After the second line, the start of the line should show `(orangs)`. That means
the box is active.

## Step 3: install the packages

Still in the Anaconda Prompt, with `(orangs)` showing:

```
pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install timm==0.9.16 ultralytics==8.2.0 opencv-python==4.9.0.80 Pillow==10.3.0 numpy==1.26.4 scikit-learn==1.4.2 matplotlib==3.9.0 tqdm==4.66.4 huggingface_hub==0.23.2 rich pywin32
```

These are the exact versions the pipeline was tested with (same as
`requirements.txt`). The first line installs the part that uses the graphics
card. If you do not have an NVIDIA graphics card, replace the first line with
this one instead:

```
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
```

Note: without a graphics card, Option A still works fine. Option B (training)
would be far too slow, so it really wants a graphics card.

## Step 4: check it worked

Type this single line:

```
python -c "import torch, timm, ultralytics, cv2, huggingface_hub, rich; print('all good, GPU =', torch.cuda.is_available())"
```

You should see `all good`. If it also says `GPU = True`, the graphics card is
ready. If it says `GPU = False`, only Option A will be practical.

If you see an error instead, open `../reference/troubleshooting.md`.

## Every time you open a new prompt

Whenever you open a fresh Anaconda Prompt to run a script, type this first:

```
conda activate orangs
```

That turns the box on. Then run the script.
