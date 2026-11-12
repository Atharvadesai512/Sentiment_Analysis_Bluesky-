# run_monitor.ps1
#
# Keeps stream_monitor.py running continuously, automatically restarting
# it if it ever exits - whether from a crash or any failure not already
# handled by its own internal websocket reconnect logic. Everything it
# prints is appended to logs/stream_monitor.log, so you don't need to
# keep a visible terminal open to check on it.
#
# This script itself does not detach from the terminal that launches
# it - use start_monitor_background.ps1 for a hidden, detached launch.

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$LogDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir "stream_monitor.log"

# Pin the Hugging Face model cache inside the project folder. Without
# this, the cache location depends on whose user profile is active -
# which differs when this runs as SYSTEM (Scheduled Task) vs. your own
# account, causing a redundant model re-download. Pinning it makes
# behavior identical regardless of which account launches this script.
$HFCacheDir = Join-Path $ProjectRoot ".cache\huggingface"
New-Item -ItemType Directory -Force -Path $HFCacheDir | Out-Null
$env:HF_HOME = $HFCacheDir

$PythonExe = Join-Path $ProjectRoot "venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Error "Could not find $PythonExe - make sure you've created the venv (see README step 1) and are running this from inside the project folder."
    exit 1
}

Add-Content -Path $LogFile -Value "=== run_monitor.ps1 started: $(Get-Date -Format o) ==="

while ($true) {
    Add-Content -Path $LogFile -Value "=== Launching stream_monitor.py: $(Get-Date -Format o) ==="

    & $PythonExe "$ProjectRoot\stream_monitor.py" *>> $LogFile

    Add-Content -Path $LogFile -Value "=== stream_monitor.py exited (code $LASTEXITCODE) at $(Get-Date -Format o). Restarting in 10s... ==="
    Start-Sleep -Seconds 10
}
