<#
.SYNOPSIS
  Local AI Studio diagnostics.

.DESCRIPTION
  Reports the Python environment, GPU/Vulkan state, engine version, model
  directory access, port availability, memory, pagefile and recent errors.
  Read-only: changes nothing on the system.

.EXAMPLE
  .\scripts\diagnose.ps1
  .\scripts\diagnose.ps1 -OutFile logs\diagnostics.txt
#>
[CmdletBinding()]
param([string]$OutFile)

$ErrorActionPreference = "SilentlyContinue"
$Root = Split-Path -Parent $PSScriptRoot
$report = New-Object System.Collections.Generic.List[string]

function Section($title) {
    $report.Add("")
    $report.Add("=" * 70)
    $report.Add("  $title")
    $report.Add("=" * 70)
}
function Line($text) { $report.Add($text) }

Section "Local AI Studio diagnostics"
Line ("Generated: " + (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"))
Line ("Project:   $Root")

# --- OS / hardware -----------------------------------------------------------
Section "Operating system & CPU"
$os = Get-CimInstance Win32_OperatingSystem
$cs = Get-CimInstance Win32_ComputerSystem
$cpu = Get-CimInstance Win32_Processor
Line ("OS:         $($os.Caption) build $($os.BuildNumber) $($os.OSArchitecture)")
Line ("Model:      $($cs.Manufacturer) $($cs.Model)")
Line ("CPU:        $($cpu.Name)")
Line ("Cores/Thr:  $($cpu.NumberOfCores) / $($cpu.NumberOfLogicalProcessors)")
Line ("RAM total:  $([math]::Round($cs.TotalPhysicalMemory/1GB,1)) GB")
Line ("RAM free:   $([math]::Round($os.FreePhysicalMemory/1MB,2)) GB")

Section "Memory & pagefile (measured)"
$vm = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory -ErrorAction SilentlyContinue
if ($vm) {
    Line ("Commit limit:      $([math]::Round($vm.CommittedBytes/1GB,1)) GB")
    Line ("Commit used:       $([math]::Round($vm.CommittedKB/1KB/1024,1)) GB" )
    Line ("Commit peak:       $([math]::Round($vm.PeakCommittedBytes/1GB,1)) GB")
}
Get-CimInstance Win32_PageFileUsage | ForEach-Object {
    Line ("Pagefile: $($_.Name)  allocated $($_.AllocatedBaseSize) MB  current $($_.CurrentUsage) MB  peak $($_.PeakUsage) MB")
}

Section "GPU"
Get-CimInstance Win32_VideoController | ForEach-Object {
    Line ("GPU:      $($_.Name)")
    Line ("Driver:   $($_.DriverVersion)  ($($_.Status))")
}

Section "Vulkan"
$vulkaninfo = "vulkaninfo"
if (Get-Command $vulkaninfo -ErrorAction SilentlyContinue) {
    # Merge stderr: loader warnings must not abort the report.
    $out = (& $vulkaninfo --summary 2>&1 | Out-String) -split "`n"
    $summary = $out | Select-String -Pattern "deviceName|apiVersion|driverName|driverInfo" | Select-Object -First 8
    if ($summary) { $summary | ForEach-Object { Line ("  " + $_.Line.Trim()) } }
    else { Line "vulkaninfo present but produced no summary" }
} else {
    Line "vulkaninfo NOT found on PATH"
}

Section "Toolchain"
foreach ($tool in @("python", "git", "cmake", "ninja", "cl")) {
    $cmd = Get-Command $tool -ErrorAction SilentlyContinue
    if ($cmd) {
        $v = & $tool --version 2>$null | Select-Object -First 1
        Line ("{0,-8} {1}" -f $tool, $v)
    } else {
        Line ("{0,-8} NOT FOUND" -f $tool)
    }
}

Section "Python environment"
$venv = Join-Path $Root ".venv\Scripts\python.exe"
if (Test-Path $venv) {
    Line ("venv:     $venv")
    Line ("version:  " + (& $venv --version 2>$null))
    $pkgs = & $venv -m pip list --format=freeze 2>$null | Select-String "fastapi|uvicorn|pydantic|psutil|httpx"
    Line ("packages: " + ($pkgs -join ", "))
} else {
    Line "venv MISSING - run .\scripts\setup.ps1"
}

Section "Engine"
$engine = Join-Path $Root "engine\llama.cpp\llama-server.exe"
if (Test-Path $engine) {
    Line ("engine:   $engine")
    Line ("version:  " + (& $engine --version 2>&1 | Select-Object -First 1))
    $devices = & $engine --list-devices 2>&1
    Line "Vulkan devices as the engine sees them:"
    $devices | ForEach-Object { Line ("  " + $_) }
} else {
    Line "engine MISSING - run .\scripts\setup.ps1 to download the Vulkan build"
}

Section "Model directories"
$dirs = @("C:\Users\SAKISAKI\.lmstudio\models")
$settingsFile = Join-Path $Root "data\settings.json"
if (Test-Path $settingsFile) {
    try {
        $cfg = Get-Content $settingsFile -Raw | ConvertFrom-Json
        if ($cfg.model_dirs) { $dirs = $cfg.model_dirs }
    } catch { Line "settings.json unreadable: $_" }
}
foreach ($d in $dirs) {
    if (Test-Path $d) {
        $ggufs = Get-ChildItem -LiteralPath $d -Recurse -Filter *.gguf -ErrorAction SilentlyContinue
        $total = ($ggufs | Measure-Object -Property Length -Sum).Sum
        Line ("$d  ->  $($ggufs.Count) GGUF files, $([math]::Round($total/1GB,1)) GB")
    } else {
        Line "$d  ->  NOT ACCESSIBLE"
    }
}

Section "Ports"
foreach ($p in @(1234, 1235)) {
    $conn = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
    if ($conn) {
        $ownerPid = $conn.OwningProcess | Select-Object -First 1
        $ownerName = (Get-Process -Id $ownerPid -ErrorAction SilentlyContinue).Name
        Line ("port $p : IN USE by $ownerName (PID $ownerPid)")
    } else {
        Line ("port $p : free")
    }
}
$eng = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $_.LocalPort -ge 18400 -and $_.LocalPort -le 18499 }
if ($eng) { Line ("engine ports in use: " + (($eng.LocalPort | Sort-Object -Unique) -join ", ")) }
else { Line "engine ports (18400-18499): free" }

Section "Running engines / backend"
$procs = Get-Process -Name "llama-server", "python" -ErrorAction SilentlyContinue
if ($procs) {
    $procs | ForEach-Object {
        $cl = (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)").CommandLine
        if ($cl -match "llama-server|uvicorn|local-ai-studio") {
            Line ("PID $($_.Id)  $($_.Name)  working set $([math]::Round($_.WorkingSet64/1MB,0)) MB")
        }
    }
} else { Line "none" }

Section "Recent errors (backend log)"
$errLog = Join-Path $Root "logs\backend.err.log"
if (Test-Path $errLog) {
    Get-Content $errLog -Tail 15 | ForEach-Object { Line ("  " + $_) }
} else { Line "no backend error log yet" }

$text = $report -join "`r`n"
$text | Write-Host
if ($OutFile) {
    $target = if ([System.IO.Path]::IsPathRooted($OutFile)) { $OutFile } else { Join-Path $Root $OutFile }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
    Set-Content -Path $target -Value $text -Encoding utf8
    Write-Host "`nReport written to $target" -ForegroundColor Green
}
