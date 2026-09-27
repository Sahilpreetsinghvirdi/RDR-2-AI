# Start the RDR2 AI agent (Windows PowerShell)
# Examples:
#   .\run.ps1
#   .\run.ps1 -Demo
#   .\run.ps1 -Autopilot
#   .\run.ps1 -Headless -Record
#   .\run.ps1 -Backend mss -ExtraArgs @("--duration","30")
#   .\run.ps1 -Doctor
#   .\run.ps1 -ControlTest -ExtraArgs @("--dry-run")

param(
    [switch]$Demo,
    [switch]$Autopilot,
    [switch]$Mission,
    [switch]$Headless,
    [switch]$Record,
    [switch]$Doctor,
    [switch]$ControlTest,
    [string]$Config = "",
    [string]$Backend = "",
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Root

$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error ".venv not found - run .\setup.ps1 first"
    exit 1
}

if ($Doctor) {
    $argsList = @("-m", "src.doctor")
    if ($Config -ne "") { $argsList += @("--config", $Config) }
} elseif ($ControlTest) {
    $argsList = @("-m", "src.control_test")
    if ($Config -ne "") { $argsList += @("--config", $Config) }
    if ($ExtraArgs.Count -gt 0) { $argsList += $ExtraArgs }
} else {
    $argsList = @("-m", "src.main")
    if ($Config -ne "") { $argsList += @("--config", $Config) }
    if ($Backend -ne "") { $argsList += @("--backend", $Backend) }
    if ($Demo) { $argsList += "--demo" }
    if ($Autopilot) { $argsList += "--autopilot" }
    if ($Mission) { $argsList += "--mission" }
    if ($Headless) { $argsList += "--headless" }
    if ($Record) { $argsList += "--record" }
    if ($ExtraArgs.Count -gt 0) { $argsList += $ExtraArgs }
}

Write-Host "Starting RDR2 AI: python $($argsList -join ' ')" -ForegroundColor Cyan
if (-not $Doctor) {
    Write-Host "Emergency stop: F12 | Pause: F11 | Human takeover: F10" -ForegroundColor Yellow
}

& $venvPython $argsList
exit $LASTEXITCODE
