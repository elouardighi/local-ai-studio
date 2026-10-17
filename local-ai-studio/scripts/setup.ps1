<#
.SYNOPSIS
  Set up Local AI Studio on Windows.

.DESCRIPTION
  Checks the OS, Python, GPU and Vulkan, downloads the official llama.cpp
  Vulkan build (or builds from source with -BuildFromSource), creates an
  isolated virtual environment, installs dependencies, writes a default
  configuration without overwriting your existing one, and verifies the result.

  Your model directory is treated as read-only: nothing here writes to,
  renames or deletes anything under C:\Users\SAKISAKI\.lmstudio\models.

.EXAMPLE
  .\scripts\setup.ps1
  .\scripts\setup.ps1 -EngineTag b11712        # pin a specific nightly
  .\scripts\setup.ps1 -BuildFromSource         # compile with GGML_VULKAN=ON
#>
[CmdletBinding()]
param(
    [string]$EngineTag = "b11712",
    [switch]$BuildFromSource,
    [switch]$SkipEngine
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$EngineDir = Join-Path $Root "engine\llama.cpp"
$DataDir = Join-Path $Root "data"
$LogsDir = Join-Path $Root "logs"

function Step($msg) { Write-Host "`n[*] $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "  [ok] $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "  [!] $msg" -ForegroundColor Yellow }
function Err($msg)  { Write-Host "  [x] $msg" -ForegroundColor Red }

function Invoke-Native {
    <#
      Run a native executable and capture combined output as strings.

      Native tools (vulkaninfo, llama-server) routinely write warnings to
      stderr. Under $ErrorActionPreference = 'Stop' PowerShell turns those
      into terminating errors, so the preference is relaxed just for the call.
    #>
    param([string]$Exe, [string[]]$Arguments = @())
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $lines = & $Exe @Arguments 2>&1
        return @($lines | ForEach-Object { "$_" })
    } finally {
        $ErrorActionPreference = $old
    }
}

# ---------------------------------------------------------------- 1. OS ------
Step "Checking Windows"
$os = Get-CimInstance Win32_OperatingSystem
if ($os.OSArchitecture -notmatch "64") {
    Err "64-bit Windows is required (found: $($os.OSArchitecture))"; exit 1
}
Ok "$($os.Caption) build $($os.BuildNumber) (64-bit)"

# ------------------------------------------------------------- 2. Python ------
Step "Checking Python (3.10+ required)"
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { Err "Python not found on PATH. Install Python 3.10+ from python.org"; exit 1 }
$pyVersion = & python -c "import sys; print('%d.%d' % sys.version_info[:2])"
$pyMajor, $pyMinor = $pyVersion.Split(".")
if ([int]$pyMajor -lt 3 -or ([int]$pyMajor -eq 3 -and [int]$pyMinor -lt 10)) {
    Err "Python $pyVersion is too old; 3.10+ required"; exit 1
}
Ok "Python $pyVersion at $py"

$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Step "Creating virtual environment"
    & python -m venv (Join-Path $Root ".venv")
    if (-not (Test-Path $venvPython)) { Err "venv creation failed"; exit 1 }
    Ok "venv created"
} else {
    Ok "venv already exists"
}

# --------------------------------------------------- 3. Python dependencies ---
Step "Installing Python dependencies"
& $venvPython -m pip install --quiet --upgrade pip
& $venvPython -m pip install --quiet -r (Join-Path $Root "requirements.txt")
if ($LASTEXITCODE -ne 0) { Err "dependency install failed"; exit 1 }
& $venvPython -c "import fastapi, uvicorn, pydantic, psutil, httpx"
if ($LASTEXITCODE -ne 0) { Err "dependency import check failed"; exit 1 }
Ok "fastapi, uvicorn, pydantic, psutil, httpx import cleanly"

# ------------------------------------------------------- 4. GPU & Vulkan ------
Step "Checking GPU and Vulkan"
$gpus = Get-CimInstance Win32_VideoController
foreach ($g in $gpus) { Ok "GPU: $($g.Name) (driver $($g.DriverVersion))" }
$vulkanOk = $false
if (Get-Command vulkaninfo -ErrorAction SilentlyContinue) {
    $summary = (Invoke-Native "vulkaninfo" @("--summary")) -join "`n"
    $devName = ($summary -split "`n") | Select-String "deviceName" | Select-Object -First 1
    if ($devName) { Ok "Vulkan device: $($devName.Line.Trim())"; $vulkanOk = $true }
    else { Warn "vulkaninfo produced no device summary" }
} else {
    Warn "vulkaninfo not on PATH (a Vulkan runtime is still required for the Vulkan backend)"
}

# ------------------------------------------------------------- 5. Engine ------
$engineExe = Join-Path $EngineDir "llama-server.exe"
if ($SkipEngine) {
    Warn "Skipping engine setup (-SkipEngine)"
} elseif ($BuildFromSource) {
    Step "Building llama.cpp from source with GGML_VULKAN=ON"
    foreach ($tool in @("git", "cmake")) {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
            Err "$tool is required for a source build"; exit 1
        }
    }
    $srcDir = Join-Path $Root "engine\llama.cpp-src"
    if (-not (Test-Path $srcDir)) {
        & git clone --depth 1 https://github.com/ggml-org/llama.cpp.git $srcDir
        if ($LASTEXITCODE -ne 0) { Err "git clone failed"; exit 1 }
    }
    & cmake -S $srcDir -B (Join-Path $srcDir "build-vulkan") -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
    if ($LASTEXITCODE -ne 0) { Err "cmake configure failed"; exit 1 }
    & cmake --build (Join-Path $srcDir "build-vulkan") --config Release --parallel
    if ($LASTEXITCODE -ne 0) { Err "cmake build failed"; exit 1 }
    New-Item -ItemType Directory -Force -Path $EngineDir | Out-Null
    Copy-Item -Path (Join-Path $srcDir "build-vulkan\bin\*") -Destination $EngineDir -Recurse -Force
    Ok "engine built and copied to $EngineDir"
} else {
    if (Test-Path $engineExe) {
        Ok "engine already present: $engineExe"
    } else {
        Step "Downloading official llama.cpp Vulkan build ($EngineTag)"
        $url = "https://github.com/ggml-org/llama.cpp/releases/download/$EngineTag/llama-$EngineTag-bin-win-vulkan-x64.zip"
        $zip = Join-Path $Root "engine\llama-vulkan.zip"
        New-Item -ItemType Directory -Force -Path $EngineDir, (Join-Path $Root "engine") | Out-Null
        try {
            $ProgressPreference = "SilentlyContinue"
            Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
            Ok "downloaded $([math]::Round((Get-Item $zip).Length/1MB,1)) MB"
        } catch {
            Err "download failed: $_"
            Write-Host "     Check the tag name at https://github.com/ggml-org/llama.cpp/releases"
            exit 1
        }
        Expand-Archive -Path $zip -DestinationPath $EngineDir -Force
        Remove-Item $zip -Force
        Ok "extracted to $EngineDir"
    }
}

# ----------------------------------------------------- 6. engine verification -
if (-not $SkipEngine) {
    Step "Verifying engine"
    if (-not (Test-Path $engineExe)) { Err "llama-server.exe missing at $engineExe"; exit 1 }
    $version = (Invoke-Native $engineExe @("--version") | Select-Object -First 1)
    Ok "version: $version"
    $devices = Invoke-Native $engineExe @("--list-devices")
    $vulkanLine = $devices | Select-String "Vulkan"
    if ($vulkanLine) { Ok "devices: $($vulkanLine.Line.Trim())" }
    else { Warn "no Vulkan device reported by the engine:`n$($devices -join "`n")" }
}

# ------------------------------------------------- 7. directories & settings --
Step "Creating application directories"
New-Item -ItemType Directory -Force -Path $DataDir, $LogsDir, (Join-Path $DataDir "converted") | Out-Null
Ok "data/, logs/, data/converted/"

$settingsFile = Join-Path $DataDir "settings.json"
if (Test-Path $settingsFile) {
    Ok "existing settings.json preserved (never overwritten)"
} else {
    $defaultSettings = @{
        model_dirs     = @("C:\Users\SAKISAKI\.lmstudio\models")
        converted_dir  = Join-Path $DataDir "converted"
        api_host       = "127.0.0.1"
        api_port       = 1234
        engine_path    = $engineExe
        engine_backend = "vulkan"
        default_ngl    = 99
        default_ctx    = 8192
    } | ConvertTo-Json
    Set-Content -Path $settingsFile -Value $defaultSettings -Encoding utf8
    Ok "default settings.json written"
}

# -------------------------------------------------------- 8. smoke tests ------
Step "Running smoke tests"
$tests = Join-Path $Root "tests"
if (Test-Path $tests) {
    & $venvPython -m pytest $tests -q --disable-warnings --no-header -x 2>$null
    if ($LASTEXITCODE -eq 0) { Ok "unit tests passed" }
    else { Warn "some tests failed - see output above (the app can still run)" }
} else {
    Warn "no tests directory found"
}

Write-Host "`n" -NoNewline
Write-Host "  Setup complete." -ForegroundColor Green
Write-Host "  Start with:   .\scripts\start.ps1" -ForegroundColor White
Write-Host "  Diagnose:     .\scripts\diagnose.ps1" -ForegroundColor White
if (-not $vulkanOk) {
    Write-Host "  Note: Vulkan was not confirmed. The engine may fall back to CPU." -ForegroundColor Yellow
}
