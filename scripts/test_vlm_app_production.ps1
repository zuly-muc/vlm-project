<#
    Production acceptance test (Windows / PowerShell wrapper).

    Runs the full end-to-end flow: build image, verify deps, obtain the real
    nuScenes v1.0-mini dataset, ingest real clips, and assert correctness.

    Examples:
      .\scripts\test_vlm_app_production.ps1                       # docker mode (needs Docker Desktop)
      .\scripts\test_vlm_app_production.ps1 -Mode local           # no Docker, uses the venv
      $env:NUSCENES_DIR="D:\nuscenes"; .\scripts\test_vlm_app_production.ps1
#>
[CmdletBinding()]
param(
    [ValidateSet("docker", "local")] [string]$Mode = "docker",
    [string]$NuscenesDir = $env:NUSCENES_DIR,
    [switch]$SkipBuild,
    [switch]$NoDownload
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

# Prefer the project venv's python if present, else system python.
$py = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$argsList = @("$repo\scripts\production_acceptance.py", "--mode", $Mode)
if ($NuscenesDir) { $argsList += @("--nuscenes-dir", $NuscenesDir) }
if ($SkipBuild)   { $argsList += "--skip-build" }
if ($NoDownload)  { $argsList += "--no-download" }

Write-Host "Running: $py $($argsList -join ' ')"
& $py @argsList
exit $LASTEXITCODE
