# RDR2 AI - one-time environment setup (Windows PowerShell)
# Creates .venv, installs runtime + dev dependencies.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Root

Write-Host "== RDR2 AI setup ==" -ForegroundColor Cyan

$python = Get-Command py -ErrorAction SilentlyContinue
if ($python) {
    $ver = & py -3 -c "import sys; print(sys.version_info[:2])" 2>$null
    Write-Host "Using Python launcher: $ver"
}

$venv = Join-Path $Root ".venv"
$venvPython = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment at .venv ..."
    & py -3 -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
} else {
    Write-Host "Reusing existing .venv"
}

Write-Host "Upgrading pip ..."
& $venvPython -m pip install --upgrade pip --quiet

Write-Host "Installing requirements ..."
& $venvPython -m pip install -r (Join-Path $Root "requirements-dev.txt")
if ($LASTEXITCODE -ne 0) { throw "dependency installation failed" }

Write-Host "Verifying imports ..."
& $venvPython -c "import cv2, numpy, yaml, mss; print('opencv', cv2.__version__, '| numpy', numpy.__version__)"
if ($LASTEXITCODE -ne 0) { throw "import verification failed" }

& $venvPython -c "import dxcam; print('dxcam OK')"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "dxcam unavailable - the agent will fall back to mss (slower)."
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Next steps:"
Write-Host "  1. Launch RDR2 Story Mode"
Write-Host "  2. .\run.ps1            (start the agent)"
Write-Host "  3. .\run.ps1 -Demo      (run the Phase 1 input demonstration)"
Write-Host "  4. F12 = emergency stop, F11 = pause, F10 = human takeover"
