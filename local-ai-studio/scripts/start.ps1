<#
.SYNOPSIS
  Start Local AI Studio (backend + dashboard).

.DESCRIPTION
  Validates the environment, refuses to fight another app for the port, starts
  the backend and opens the dashboard in your default browser.
  The backend PID is written to data\backend.pid so stop.ps1 can stop exactly
  the process this script started - never a random process with the same name.

.EXAMPLE
  .\scripts\start.ps1
  .\scripts\start.ps1 -Port 1235 -NoBrowser
#>
[CmdletBinding()]
param(
    [int]$Port,
    [switch]$NoBrowser,
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$SettingsFile = Join-Path $Root "data\settings.json"
$PidFile = Join-Path $Root "data\backend.pid"

function Write-Step($msg) { Write-Host "[*] $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "[ok] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[!] $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "[x] $msg" -ForegroundColor Red }

# --- 1. virtual environment -------------------------------------------------
if (-not (Test-Path $VenvPython)) {
    Write-Err "Virtual environment missing at $VenvPython"
    Write-Host "     Run .\scripts\setup.ps1 first."
    exit 1
}
Write-Ok "Python environment found"

# --- 2. engine ---------------------------------------------------------------
$EnginePath = Join-Path $Root "engine\llama.cpp\llama-server.exe"
if (-not (Test-Path $EnginePath)) {
    Write-Warn "Engine not found at $EnginePath"
    Write-Warn "Models cannot load until the engine exists. Run .\scripts\setup.ps1"
} else {
    Write-Ok "Engine found: $EnginePath"
}

# --- 3. determine the port (never steal one silently) -------------------------
if (-not $Port) {
    $Port = 1234
    if (Test-Path $SettingsFile) {
        try {
            $cfg = Get-Content $SettingsFile -Raw | ConvertFrom-Json
            if ($cfg.api_port) { $Port = [int]$cfg.api_port }
        } catch {
            Write-Warn "settings.json unreadable ($_); using default port 1234"
        }
    }
}

$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    $ownerPid = $listener.OwningProcess | Select-Object -First 1
    $owner = Get-CimInstance Win32_Process -Filter "ProcessId=$ownerPid" -ErrorAction SilentlyContinue
    Write-Warn "Port $Port is already used by $($owner.Name) (PID $ownerPid)"
    if ($owner.Name -match "python") {
        Write-Warn "That looks like an existing Local AI Studio backend. Open http://127.0.0.1:$Port instead, or run .\scripts\stop.ps1"
    } else {
        Write-Host "     Either close that application, or start on another port:"
        Write-Host "     .\scripts\start.ps1 -Port 1235" -ForegroundColor Yellow
        Write-Host "     (then update 'api_port' in data\settings.json to match)"
    }
    exit 2
}

# --- 4. stale pid cleanup ----------------------------------------------------
if (Test-Path $PidFile) {
    # stop.ps1 verifies identity and kills the whole process tree, so no
    # orphaned uvicorn/engine process can survive this cleanup.
    Write-Step "Stale backend PID file found - running stop.ps1 first..."
    & (Join-Path $PSScriptRoot "stop.ps1")
    if (Test-Path $PidFile) { Remove-Item $PidFile -Force -ErrorAction SilentlyContinue }
}

# --- 5. start -----------------------------------------------------------------
$LogsDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
$OutLog = Join-Path $LogsDir "backend.out.log"
$ErrLog = Join-Path $LogsDir "backend.err.log"

if ($Foreground) {
    Write-Step "Starting backend in the foreground (Ctrl+C to stop)..."
    & $VenvPython -m uvicorn backend.app.main:app --host 127.0.0.1 --port $Port --app-dir $Root
    exit $LASTEXITCODE
}

Write-Step "Starting backend on port $Port..."
# The child must not inherit our stdin (an empty file severs that handle), so
# nothing that waits on this console/pipe can be kept alive by the backend.
$StdInFile = Join-Path $LogsDir "stdin.empty"
if (-not (Test-Path $StdInFile)) { [System.IO.File]::WriteAllBytes($StdInFile, [byte[]]@()) }
$proc = Start-Process -FilePath $VenvPython `
    -ArgumentList @("-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "$Port", "--app-dir", $Root) `
    -WorkingDirectory $Root `
    -RedirectStandardInput $StdInFile `
    -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog `
    -PassThru -WindowStyle Hidden

$proc.Id | Out-File -FilePath $PidFile -Encoding ascii
Write-Ok "Backend started (PID $($proc.Id))"

# --- 6. wait for health --------------------------------------------------------
$health = "http://127.0.0.1:$Port/health"
$deadline = (Get-Date).AddSeconds(45)
$ready = $false
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 700
    if ($proc.HasExited) { break }
    try {
        $resp = Invoke-RestMethod -Uri $health -TimeoutSec 3
        if ($resp.status -eq "ok") { $ready = $true; break }
    } catch { }
}

if (-not $ready) {
    if ($proc.HasExited) {
        Write-Err "Backend exited during startup (code $($proc.ExitCode)). Last log lines:"
        if (Test-Path $ErrLog) { Get-Content $ErrLog -Tail 20 }
    } else {
        Write-Warn "Backend did not answer /health within 45s. Check $ErrLog"
    }
    exit 3
}

Write-Ok "Backend healthy"
Write-Host ""
Write-Host "  Dashboard:  http://127.0.0.1:$Port/" -ForegroundColor White
Write-Host "  OpenAI API: http://127.0.0.1:$Port/v1" -ForegroundColor White
Write-Host "  Health:     $health" -ForegroundColor White
Write-Host ""
Write-Host "  Stop with:  .\scripts\stop.ps1" -ForegroundColor DarkGray

if (-not $NoBrowser) {
    Start-Process "http://127.0.0.1:$Port/"
}
