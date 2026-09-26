# Start the RDR2 AI agent (Windows PowerShell)
# Examples:
#   .\run.ps1
#   .\run.ps1 -Demo
#   .\run.ps1 -Autopilot
#   .\run.ps1 -Headless -Record
#   .\run.ps1 -Backend mss -ExtraArgs @("--duration","30")

param(
    [switch]$Demo,
    [switch]$Autopilot,
    [switch]$Headless,
    [switch]$Record,
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

$argsList = @("-m", "src.main")
if ($Config -ne "") { $argsList += @("--config", $Config) }
if ($Backend -ne "") { $argsList += @("--backend", $Backend) }
if ($Demo) { $argsList += "--demo" }
if ($Autopilot) { $argsList += "--autopilot" }
if ($Headless) { $argsList += "--headless" }
if ($Record) { $argsList += "--record" }
if ($ExtraArgs.Count -gt 0) { $argsList += $ExtraArgs }

Write-Host "Starting RDR2 AI: python $($argsList -join ' ')" -ForegroundColor Cyan
Write-Host "Emergency stop: F12 | Pause: F11 | Human takeover: F10" -ForegroundColor Yellow

& $venvPython $argsList
exit $LASTEXITCODE
