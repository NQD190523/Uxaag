<#
.SYNOPSIS
  Stop the components started by start-project.bat (called by its "stop" mode).

.DESCRIPTION
  For each app port (8006, 8005, 8001, 8000, 3000) finds the listening process,
  checks it is the expected kind (python for services, node for the UI), then
  walks up its parent processes to the launcher's own "cmd /k" and ends that
  whole tree, which also closes its console window / Windows Terminal tab.

  Why not match by window title: on Windows 11 consoles are hosted by Windows
  Terminal, so cmd.exe owns no window and title filters never match.

  Safety: only process trees whose cmd command line is exactly what
  start-project.bat launches are ended. Anything else on these ports is
  reported and left running.

  Exit code: 0 if every port is free afterwards, 1 otherwise.
#>

$ErrorActionPreference = 'SilentlyContinue'

# port -> expected listener process-name prefix
$ports = [ordered]@{ 8006 = 'python'; 8005 = 'python'; 8001 = 'python'; 8000 = 'python'; 3000 = 'node' }
# Command-line fragments of the cmd /k windows start-project.bat opens.
$launcherMarkers = @('.venv\Scripts\python.exe src\main.py', 'npm run dev -- -p 3000')

function Get-Proc([int]$id) { Get-CimInstance Win32_Process -Filter "ProcessId=$id" }

$killed = @{}
foreach ($port in $ports.Keys) {
    $listeners = Get-NetTCPConnection -State Listen -LocalPort $port |
        Select-Object -ExpandProperty OwningProcess -Unique
    if (-not $listeners) { Write-Host "[  --  ] :$port already free"; continue }

    foreach ($procId in $listeners) {
        $proc = Get-Proc $procId
        if (-not $proc) { continue }
        if ($proc.Name -notlike "$($ports[$port])*") {
            Write-Host "[ SKIP ] :$port held by $($proc.Name) (PID $procId), not started by the launcher - left running"
            continue
        }
        # Walk up (max 6 levels) to the launcher's cmd /k window process.
        $root = $null; $cur = $proc
        for ($i = 0; $i -lt 6 -and $cur; $i++) {
            if ($cur.Name -eq 'cmd.exe') {
                foreach ($m in $launcherMarkers) {
                    if ($cur.CommandLine -and $cur.CommandLine.Contains($m)) { $root = $cur; break }
                }
                if ($root) { break }
            }
            $cur = Get-Proc $cur.ParentProcessId
        }
        if (-not $root) {
            Write-Host "[ SKIP ] :$port $($proc.Name) (PID $procId) was not started by start-project.bat - left running"
            continue
        }
        if (-not $killed.ContainsKey($root.ProcessId)) {
            # /T ends the whole tree (venv launcher, python / npm, node), /F forces it.
            taskkill /PID $root.ProcessId /T /F | Out-Null
            $killed[$root.ProcessId] = $true
        }
        Write-Host "[STOPPED] :$port $($proc.Name) (PID $procId) and its launcher window"
    }
}

# Ports can take a moment to be released after the processes exit.
Start-Sleep -Seconds 2
$still = @($ports.Keys | Where-Object { Get-NetTCPConnection -State Listen -LocalPort $_ })
if ($still.Count) { Write-Host "Still listening: $($still -join ', ')"; exit 1 }
Write-Host 'All app ports are free.'
exit 0
