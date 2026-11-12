# start_monitor_background.ps1
#
# Launches run_monitor.ps1 as a separate, hidden, detached process, so
# it keeps running even after you close this terminal window. Run this
# once to start collecting in the background.

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RunScript = Join-Path $PSScriptRoot "run_monitor.ps1"

Start-Process -FilePath "powershell.exe" `
    -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$RunScript`"" `
    -WindowStyle Hidden

Write-Host "Started. It will keep running in the background even if you close this window."
Write-Host "Check progress any time with:  Get-Content -Wait '$ProjectRoot\logs\stream_monitor.log'"
Write-Host "Stop it later with:            .\scripts\stop_monitor.ps1"
