# Part 2: WSL, only for Option B

You only need this if you retrain the brain (Option B). The phone needs the
brain in a format called `.tflite`, and the tool that makes that format only
runs on Linux. WSL is a small Linux that runs inside Windows, just for this one
job.

If you only add animals (Option A), skip this page.

## Step 1: turn WSL on (once)

1. Right click the Start button and choose "Windows PowerShell (Admin)" or
   "Terminal (Admin)".
2. Type this and press Enter:

   ```
   wsl --install
   ```

3. Restart the computer when it asks.

This installs WSL and Ubuntu (a version of Linux). You may be asked to create a
Linux username and password the first time Ubuntu opens. Any name and password
are fine, just remember them.

## Step 2: run the setup script

This script installs everything the export needs. It fixes the two problems that
happened before (a missing graphics library, and a package that was renamed), so
you do not have to type anything by hand.

1. Right click the Start button and choose "Windows PowerShell".
2. Copy and paste this line and press Enter:

   ```
   powershell -NoExit -ExecutionPolicy Bypass -File maintenance\00_first_time_setup\setup_wsl.ps1
   ```

3. Wait. It downloads and installs several things and prints its progress. It
   ends with either "SETUP COMPLETE" or a list of what is missing.

## Step 3: check it worked

At the end the script checks every package and prints `ok` for each one, plus a
line saying the converter has a `convert()` function. If everything is `ok` and
you see "SETUP COMPLETE", you are done.

If some line is red, run the script again. It is safe to run several times: it
skips what is already installed. If it still fails, open
`../reference/troubleshooting.md`.

## What this installs (for reference)

Inside WSL, in an environment called `orangs_export`:

- PyTorch (CPU version), timm: to load the brain
- litert-torch: the tool that makes the `.tflite` file
- pillow, numpy, huggingface_hub: helpers

It does not install ultralytics or the head detector here, because cropping runs
on the Windows side. That is why the old `libGL.so.1` error does not happen with
this script.

## A note about disk space

WSL and these packages take a few gigabytes. The setup script puts WSL on the D:
drive. If your C: drive is very full, that is fine here, but if the very first
`wsl --install` fails for lack of space, free some room on C: and try again.
