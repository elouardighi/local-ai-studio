<#
.SYNOPSIS
  Stop Local AI Studio.

.DESCRIPTION
  Stops the backend recorded in data\backend.pid after verifying every process
  it kills really belongs to this installation (never kills by executable name
  alone):

  * the recorded PID is the *root* of a process tree - the venv python spawns
    a second python (uvicorn), which spawns llama-server engines - so the whole
    tree is stopped, leaving no orphaned server or engine behind;
  * python nodes are only touched when their command line references this
    project (uvicorn/backend.app/the project directory);
  * llama-server nodes are only touched when their executable lives in this
    installation's engine\ directory;
  * engine processes orphaned by an earlier crash (PID file already gone) are
    also cleaned up using the same executable-path check.
#>
[CmdletBinding()]
param()

$Root = Split-Path -Parent $PSScriptRoot
$PidFile = Join-Path $Root "data\backend.pid"
$EngineExe = [regex]::Escape((Join-Path $Root "engine"))   # our extracted llama-server path
$PyMarker  = "uvicorn|backend\.app|local-ai-studio"        # markers for our backend processes

function Write-Ok($msg)   { Write-Host "[ok] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[!] $msg" -ForegroundColor Yellow }

# Returns $true when the Win32_Process row is one of ours.
function Test-OwnProcess($p) {
    if (-not $p) { return $false }
    if ($p.Name -match '^python(3(\.\d+)?)?\.exe$' -or $p.Name -eq 'pythonw.exe') {
        return [bool]($p.CommandLine -and $p.CommandLine -match $PyMarker)
    }
    if ($p.Name -eq 'llama-server.exe') {
        return [bool]($p.CommandLine -and $p.CommandLine -match $EngineExe)
    }
    return $false
}

# All descendants of a process (iterative walk over a snapshot).
function Get-Descendants([int]$RootPid, $All) {
    $result = New-Object System.Collections.Generic.List[object]
    $queue = New-Object System.Collections.Generic.Queue[object]
    $queue.Enqueue($RootPid)
    $seen  = @{}
    while ($queue.Count -gt 0) {
        $current = $queue.Dequeue()
        foreach ($p in $All) {
            if ([int]$p.ParentProcessId -eq $current -and -not $seen.ContainsKey([int]$p.ProcessId)) {
                $seen[[int]$p.ProcessId] = $true
                $result.Add($p)
                $queue.Enqueue([int]$p.ProcessId)
            }
        }
    }
    # Emit rows individually; the call site wraps with @() so an empty result
    # still becomes an empty array instead of $null.
    foreach ($item in $result) { Write-Output $item }
}

$All = @(Get-CimInstance Win32_Process)
$toStop = New-Object System.Collections.Generic.List[object]
$rootPid = $null

if (Test-Path $PidFile) {
    $rootPid = (Get-Content $PidFile -Raw).Trim()
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue

    if (-not $rootPid) {
        Write-Warn "Empty PID file"
    } else {
        $root = $All | Where-Object { [int]$_.ProcessId -eq [int]$rootPid }
        if (-not $root) {
            Write-Warn "Backend PID $rootPid is not running."
        } elseif (-not (Test-OwnProcess $root)) {
            # Never touch an unrelated process that happens to reuse the PID.
            Write-Warn "PID $rootPid is not the Local AI Studio backend ($($root.CommandLine)). Not touching it."
            exit 1
        } else {
            $toStop.Add($root)
            foreach ($d in @(Get-Descendants ([int]$rootPid) $All)) {
                if (Test-OwnProcess $d) { $toStop.Add($d) }
            }
        }
    }
}

# Orphan cleanup: engines still running without (or beyond) the PID file.
$claimed = @{}
foreach ($p in $toStop) { $claimed[[int]$p.ProcessId] = $true }
foreach ($p in $All) {
    if ($p.Name -eq 'llama-server.exe' -and (Test-OwnProcess $p) -and -not $claimed[[int]$p.ProcessId]) {
        $toStop.Add($p)
        $claimed[[int]$p.ProcessId] = $true
        Write-Warn "Found orphaned engine PID $($p.ProcessId) - will stop it too."
    }
}

if ($toStop.Count -eq 0) {
    Write-Warn "Nothing to stop."
    exit 0
}

$names = ($toStop | ForEach-Object { "$($_.ProcessId) $($_.Name)" }) -join ', '
Write-Host "[*] Stopping: $names" -ForegroundColor Cyan

# Engines first (so they can flush), then the backend tree.
$engines  = @($toStop | Where-Object { $_.Name -eq 'llama-server.exe' })
$backends = @($toStop | Where-Object { $_.Name -ne 'llama-server.exe' })
foreach ($p in ($engines + $backends)) {
    Stop-Process -Id ([int]$p.ProcessId) -Force -ErrorAction SilentlyContinue
}

# The OS needs a moment to reap the processes.
$deadline = (Get-Date).AddSeconds(8)
$alive = $true
while ($alive -and (Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 400
    $alive = [bool]($toStop | Where-Object {
        Get-Process -Id ([int]$_.ProcessId) -ErrorAction SilentlyContinue
    })
}

$leftover = @($toStop | Where-Object { Get-Process -Id ([int]$_.ProcessId) -ErrorAction SilentlyContinue })
if ($leftover) {
    Write-Warn "Still running after stop: $($leftover.ProcessId -join ', ')"
    exit 1
}

$remaining = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $_.LocalPort -ge 18400 -and $_.LocalPort -le 18499 }
if ($remaining) {
    Write-Warn "Engine ports still listening: $($remaining.LocalPort -join ', ')"
    Write-Host "     Check Task Manager for llama-server.exe"
    exit 1
}
Write-Ok "Backend and all engine processes stopped; no engine ports remain open."
