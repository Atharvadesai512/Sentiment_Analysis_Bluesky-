# stop_monitor.ps1
#
# Stops the background monitor: both the run_monitor.ps1 supervisor loop
# (if still running) and any stream_monitor.py process it launched.
#
# Safe to run even if nothing is currently running.

$found = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match "run_monitor\.ps1" -or $_.CommandLine -match "stream_monitor\.py"
}

if (-not $found) {
    Write-Host "Nothing running - no matching process found."
    exit 0
}

foreach ($proc in $found) {
    Write-Host "Stopping PID $($proc.ProcessId): $($proc.CommandLine)"
    Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
}

Write-Host "Done."
