#Requires -RunAsAdministrator
# remove_scheduled_task.ps1
#
# Removes the "SentimentMonitor" Scheduled Task registered by
# setup_scheduled_task.ps1. Run from an elevated (Administrator)
# PowerShell window.
#
# This only stops it from starting on future boots - if it's currently
# running, stop the live process too with:  .\stop_monitor.ps1

$TaskName = "SentimentMonitor"

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if (-not $existing) {
    Write-Host "No scheduled task named '$TaskName' found - nothing to remove."
    exit 0
}

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
Write-Host "Removed scheduled task '$TaskName'."
Write-Host "If it's currently running, also run .\stop_monitor.ps1 to stop the live process."
