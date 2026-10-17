<#
.SYNOPSIS
  Rebuild llama.cpp from source with the Vulkan backend.

.DESCRIPTION
  Alternative to the prebuilt nightly used by setup.ps1. Use this when you want
  to pin a specific commit, add another backend, or when the official binary
  does not work on your machine.

  Requires: git, CMake 3.14+, Visual Studio Build Tools 2022 (MSVC) or clang,
  and the Vulkan SDK / a Vulkan loader (for vulkan-1.lib).

.EXAMPLE
  .\scripts\build-vulkan.ps1
  .\scripts\build-vulkan.ps1 -Tag b11712 -Clean
#>
[CmdletBinding()]
param(
    [string]$Tag = "b11712",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$SrcDir = Join-Path $Root "engine\llama.cpp-src"
$BuildDir = Join-Path $SrcDir "build-vulkan"
$EngineDir = Join-Path $Root "engine\llama.cpp"

Write-Host "[*] Building llama.cpp ($Tag) with GGML_VULKAN=ON" -ForegroundColor Cyan

foreach ($tool in @("git", "cmake")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Write-Host "[x] $tool is required" -ForegroundColor Red; exit 1
    }
}

if (-not (Test-Path $SrcDir)) {
    Write-Host "[*] Cloning ggml-org/llama.cpp" -ForegroundColor Cyan
    & git clone https://github.com/ggml-org/llama.cpp.git $SrcDir
    if ($LASTEXITCODE -ne 0) { Write-Host "[x] clone failed" -ForegroundColor Red; exit 1 }
}

Push-Location $SrcDir
try {
    & git fetch --tags --quiet
    & git checkout $Tag
    if ($LASTEXITCODE -ne 0) { Write-Host "[x] checkout $Tag failed" -ForegroundColor Red; exit 1 }

    if ($Clean -and (Test-Path $BuildDir)) {
        Write-Host "[*] Cleaning previous build" -ForegroundColor Cyan
        Remove-Item $BuildDir -Recurse -Force
    }

    Write-Host "[*] Configuring (this enables the Vulkan backend)" -ForegroundColor Cyan
    & cmake -S $SrcDir -B $BuildDir -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[x] CMake configure failed." -ForegroundColor Red
        Write-Host "    Is the Vulkan SDK installed? https://vulkan.lunarg.com/sdk/home"
        Write-Host "    On configure errors mentioning vulkan, install the SDK and reopen the terminal."
        exit 1
    }

    Write-Host "[*] Building (parallel)" -ForegroundColor Cyan
    & cmake --build $BuildDir --config Release --parallel
    if ($LASTEXITCODE -ne 0) { Write-Host "[x] build failed" -ForegroundColor Red; exit 1 }
} finally {
    Pop-Location
}

# Locate the produced binary rather than assuming a path.
$built = Get-ChildItem -Path $BuildDir -Recurse -Filter "llama-server.exe" |
    Select-Object -First 1
if (-not $built) { Write-Host "[x] llama-server.exe not found in $BuildDir" -ForegroundColor Red; exit 1 }

New-Item -ItemType Directory -Force -Path $EngineDir | Out-Null
Copy-Item -Path (Join-Path $built.DirectoryName "*") -Destination $EngineDir -Recurse -Force
Write-Host "[ok] Engine installed to $EngineDir" -ForegroundColor Green

$exe = Join-Path $EngineDir "llama-server.exe"
& $exe --version 2>&1 | Select-Object -First 1 | Write-Host -ForegroundColor Green
& $exe --list-devices 2>&1 | ForEach-Object { Write-Host "  $_" -ForegroundColor Green }
