# =============================================================================
# setup_wsl.ps1  --  OrangIdentifier maintenance
# =============================================================================
# One-time setup of WSL (a small Linux inside Windows) for ONE job only:
# turning the trained brain into the .tflite phone file. That conversion only
# works on Linux, which is why WSL is needed.
#
# You only need this for Option B (retrain the brain). Option A does not use WSL.
#
# HOW TO RUN (right click the Start button, choose "Windows PowerShell", then):
#   powershell -NoExit -ExecutionPolicy Bypass -File maintenance\00_first_time_setup\setup_wsl.ps1
#
# This script installs only what the export step needs. It does NOT install
# ultralytics or tensorflow-gpu, because the head detection (which needs those)
# runs on the Windows side, not here. That is why the old libGL error does not
# happen with this script.
# =============================================================================

$ErrorActionPreference = "Continue"
$distro    = "Ubuntu"
$wslDir    = "D:\WSL\Ubuntu"
$condaPath = "/root/miniconda3"
$envName   = "orangs_export"

function Log($msg, $col = "White") {
    $ts = Get-Date -Format "HH:mm:ss"
    Write-Host "[$ts] $msg" -ForegroundColor $col
}

Write-Host ""
Write-Host "  WSL setup for OrangIdentifier export" -ForegroundColor Cyan
Write-Host "  (no training happens here, only the .tflite conversion)" -ForegroundColor DarkCyan
Write-Host ""

# -----------------------------------------------------------------------------
# 0. Disk space. WSL and the packages need a few GB. They live on D: here.
# -----------------------------------------------------------------------------
Log "Checking free space on D:..." "Cyan"
$freeD = (Get-PSDrive D).Free / 1GB
if ($freeD -lt 8) {
    Log "WARNING: only $([math]::Round($freeD,1)) GB free on D: (8 GB recommended)" "Yellow"
} else {
    Log "Free space on D: is fine ($([math]::Round($freeD,1)) GB)" "Green"
}

# The C: drive is where WSL is created by default. If it is nearly full, the
# import below can fail. We warn but continue.
$freeC = (Get-PSDrive C).Free / 1GB
if ($freeC -lt 3) {
    Log "WARNING: only $([math]::Round($freeC,1)) GB free on C:. If the import" "Yellow"
    Log "         fails, free some space on C: and run this script again." "Yellow"
}

# -----------------------------------------------------------------------------
# 1. Is WSL available at all
# -----------------------------------------------------------------------------
Log "Checking WSL..." "Cyan"
wsl --version 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Log "WSL is not installed. Open PowerShell as administrator and run:" "Red"
    Log "  wsl --install" "Yellow"
    Log "Then restart the computer and run this script again." "Yellow"
    exit 1
}
Log "WSL is available" "Green"

# -----------------------------------------------------------------------------
# 2. Find Ubuntu, or import it if it is missing
# -----------------------------------------------------------------------------
Log "Looking for an Ubuntu distribution..." "Cyan"
$found = $false
$rawList = wsl -l -q 2>&1
foreach ($line in $rawList) {
    $clean = ($line -replace '[^\x20-\x7E]', '').Trim()
    if ($clean -like "*Ubuntu*") { $distro = $clean; $found = $true; break }
}

if (-not $found) {
    Log "Ubuntu not found, importing it from the Ubuntu cloud image..." "Yellow"
    $tarPath = "D:\ubuntu2204.tar.gz"
    if (-not (Test-Path $tarPath)) {
        Log "Downloading Ubuntu 22.04 (about 300 MB)..." "Yellow"
        $url = "https://cloud-images.ubuntu.com/wsl/releases/jammy/current/ubuntu-jammy-wsl-amd64-wsl.rootfs.tar.gz"
        $wc  = New-Object System.Net.WebClient
        $wc.DownloadFile($url, $tarPath)
        if (-not (Test-Path $tarPath)) { Log "Download failed." "Red"; exit 1 }
    }
    New-Item -Path $wslDir -ItemType Directory -Force | Out-Null
    Log "Importing Ubuntu into $wslDir ..." "Yellow"
    wsl --import Ubuntu $wslDir $tarPath
    if ($LASTEXITCODE -eq 0) { $distro = "Ubuntu"; Log "Ubuntu imported" "Green" }
    else { Log "Import failed. Try: wsl --install -d Ubuntu" "Red"; exit 1 }
}
Log "Using distribution: $distro" "Green"

# -----------------------------------------------------------------------------
# 3. Quick test that WSL answers
# -----------------------------------------------------------------------------
$testResult = wsl -d $distro -- bash -c "echo WSLOK" 2>&1
if ("$testResult" -match "WSLOK") { Log "WSL answers correctly" "Green" }
else { Log "WSL did not answer: $testResult" "Red"; exit 1 }

# -----------------------------------------------------------------------------
# 4. System packages. libgl1 and friends are needed by some Python packages.
#    Installing them here avoids the "libGL.so.1 not found" error.
# -----------------------------------------------------------------------------
Log "Updating the Ubuntu package list..." "Cyan"
wsl -d $distro -- bash -c "apt-get update -qq 2>&1 | tail -1"
Log "Installing system libraries (this avoids the libGL error)..." "Cyan"
wsl -d $distro -- bash -c "apt-get install -y -qq wget curl libgl1 libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev 2>&1 | tail -2"
Log "System libraries installed" "Green"

# -----------------------------------------------------------------------------
# 5. Miniconda (a Python manager)
# -----------------------------------------------------------------------------
Log "Checking Miniconda..." "Cyan"
$condaOk = wsl -d $distro -- bash -c "test -f $condaPath/bin/conda && echo FOUND || echo MISSING"
if ($condaOk -match "FOUND") {
    Log "Miniconda is already installed" "Green"
} else {
    $dirExists = wsl -d $distro -- bash -c "test -d $condaPath && echo EXISTS || echo NO"
    if ($dirExists -match "EXISTS") {
        Log "Removing an incomplete Miniconda folder..." "Yellow"
        wsl -d $distro -- bash -c "rm -rf $condaPath"
    }
    $installerWsl = "/mnt/d/miniconda.sh"
    Log "Downloading Miniconda (about 155 MB)..." "Yellow"
    wsl -d $distro -- bash -c "wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $installerWsl"
    Log "Installing Miniconda (2 to 4 minutes)..." "Yellow"
    wsl -d $distro -- bash -c "bash $installerWsl -b -p $condaPath 2>&1 | tail -2"
    $condaOk2 = wsl -d $distro -- bash -c "test -f $condaPath/bin/conda && echo FOUND || echo MISSING"
    if ($condaOk2 -match "FOUND") { Log "Miniconda installed" "Green" }
    else { Log "Miniconda install failed" "Red"; exit 1 }
}

# -----------------------------------------------------------------------------
# 6. The conda environment for the export
# -----------------------------------------------------------------------------
Log "Checking the conda environment '$envName'..." "Cyan"
$envCheck = wsl -d $distro -- bash -c "$condaPath/bin/conda env list 2>&1"
if ($envCheck -match $envName) {
    Log "Environment '$envName' already exists" "Green"
} else {
    Log "Creating the environment with Python 3.10 (1 to 2 minutes)..." "Yellow"
    wsl -d $distro -- bash -c "$condaPath/bin/conda create -n $envName python=3.10 -y 2>&1 | tail -3"
    $envOk = wsl -d $distro -- bash -c "test -d $condaPath/envs/$envName && echo OK || echo FAIL"
    if ($envOk -match "FAIL") { Log "Environment creation failed" "Red"; exit 1 }
    Log "Environment created" "Green"
}

# Helper to run a command inside the environment
$R = "$condaPath/bin/conda run -n $envName --no-capture-output"

# -----------------------------------------------------------------------------
# 7. Python packages. Only what the export needs.
#    litert-torch is the current name of the converter (it used to be
#    ai-edge-torch, which is why the old scripts failed).
# -----------------------------------------------------------------------------
Log "Installing PyTorch (CPU version)..." "Cyan"
$torchOk = wsl -d $distro -- bash -c "$R python -c 'import torch' 2>/dev/null && echo OK || echo MISSING"
if ($torchOk -match "MISSING") {
    wsl -d $distro -- bash -c "$R pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -2"
}
Log "Installing litert-torch (the .tflite converter)..." "Cyan"
$litertOk = wsl -d $distro -- bash -c "$R python -c 'import litert_torch' 2>/dev/null && echo OK || echo MISSING"
if ($litertOk -match "MISSING") {
    wsl -d $distro -- bash -c "$R pip install litert-torch 2>&1 | tail -2"
}
Log "Installing timm and helpers..." "Cyan"
wsl -d $distro -- bash -c "$R pip install timm pillow numpy huggingface_hub 2>&1 | tail -2"
Log "Packages installed" "Green"

# -----------------------------------------------------------------------------
# 8. Final check
# -----------------------------------------------------------------------------
Write-Host ""
Log "Checking that everything imports..." "Cyan"
$imports = @("torch", "torchvision", "litert_torch", "timm", "PIL", "numpy", "huggingface_hub")
$allOk = $true
foreach ($pkg in $imports) {
    $check = wsl -d $distro -- bash -c "$R python -c 'import $pkg' 2>/dev/null && echo OK || echo MISSING"
    if ($check -match "MISSING") { Log "  MISSING: $pkg" "Red"; $allOk = $false }
    else { Log "  ok: $pkg" "Green" }
}
# The converter must have a convert() function.
$convOk = wsl -d $distro -- bash -c "$R python -c 'import litert_torch; print(hasattr(litert_torch, ""convert""))' 2>/dev/null"
if ($convOk -match "True") { Log "  ok: litert_torch has convert()" "Green" }
else { Log "  PROBLEM: litert_torch has no convert(). Run: $R pip install -U litert-torch" "Red"; $allOk = $false }

Write-Host ""
if ($allOk) {
    Write-Host "  SETUP COMPLETE" -ForegroundColor Green
    Write-Host ""
    Write-Host "  You can now run the export (after training with train_brain.py):" -ForegroundColor White
    Write-Host "  wsl -d $distro -- bash -c `"$R python /mnt/d/<path-to-your-repo>/maintenance/scripts/export_to_tflite.py`"" -ForegroundColor DarkCyan
} else {
    Write-Host "  Some packages are missing. See the red lines above." -ForegroundColor Red
    Write-Host "  Then run this script again." -ForegroundColor Red
}
Write-Host ""
