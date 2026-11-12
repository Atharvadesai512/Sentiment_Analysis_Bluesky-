#Requires -RunAsAdministrator
# setup_scheduled_task.ps1
#
# Registers a Windows Scheduled Task that starts run_monitor.ps1
# automatically at system startup, running as the SYSTEM account. That
# means it starts on boot with no login required, and keeps running
# through logout - only a full shutdown stops it, and it restarts
# automatically on the next boot.
#
# MUST be run from an elevated (Administrator) PowerShell window.
# Safe to re-run - it replaces any existing task of the same name.

$TaskName = "SentimentMonitor"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RunScript = Join-Path $PSScriptRoot "run_monitor.ps1"

if (-not (Test-Path $RunScript)) {
    Write-Error "Could not find $RunScript - run this script from inside the project's scripts folder."
    exit 1
}

$Action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$RunScript`""

$Trigger = New-ScheduledTaskTrigger -AtStartup

# SYSTEM + ServiceAccount = runs with no user logged in at all, and
# keeps running if the user later logs out.
$Principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

# ExecutionTimeLimit = 0 disables Task Scheduler's default 72-hour
# cutoff, which would otherwise silently kill a multi-day collector.
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

Register-ScheduledTask -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Principal $Principal `
    -Settings $Settings `
    -Description "Runs the Bluesky sentiment monitor continuously, starting at boot, independent of user login." `
    | Out-Null

Write-Host "Scheduled task '$TaskName' registered."
Write-Host ""
Write-Host "It will start automatically on the next reboot. To start it right now:"
Write-Host "    Start-ScheduledTask -TaskName '$TaskName'"
Write-Host ""
Write-Host "Check its status:"
Write-Host "    Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
Write-Host ""
Write-Host "Watch it working (once started):"
Write-Host "    Get-Content -Wait '$ProjectRoot\logs\stream_monitor.log'"
Write-Host ""
Write-Host "Remove it later with:"
Write-Host "    .\remove_scheduled_task.ps1"
