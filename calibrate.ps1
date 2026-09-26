# Capture/input calibration for RDR2 AI (Windows PowerShell)
#   .\calibrate.ps1                 - measure capture performance
#   .\calibrate.ps1 -Seconds 10
#   .\calibrate.ps1 -Backend mss
#   .\calibrate.ps1 -Inspect        - save sample frames instead

param(
    [string]$Backend = "",
    [double]$Seconds = 5,
    [switch]$Inspect,
    [switch]$Full
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Root

$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error ".venv not found - run .\setup.ps1 first"
    exit 1
}

if ($Inspect) {
    $argsList = @("tools\inspect_frames.py", "--frames", "5")
    if ($Backend -ne "") { $argsList += @("--backend", $Backend) }
    if ($Full) { $argsList += "--full" }
} else {
    $argsList = @("tools\calibrate_capture.py", "--seconds", "$Seconds")
    if ($Backend -ne "") { $argsList += @("--backend", $Backend) }
    if ($Full) { $argsList += @("--roi", "full") }
}

& $venvPython @argsList
exit $LASTEXITCODE
