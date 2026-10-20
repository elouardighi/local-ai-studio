<#
.SYNOPSIS
  Add the models from Local AI Studio to an OpenCode provider.

.DESCRIPTION
  OpenCode does not ask an OpenAI-compatible server what models it has. For
  a custom provider it reads the models straight out of its own config file,
  so a model that is downloaded after the config was written does not show
  up in the /models menu until the config is updated.

  This runs the sync script, which adds whatever is missing and writes the
  context length each model can really use - the one that fits in the free
  VRAM, not the number the model advertises. The config is edited as text,
  so comments, formatting and every other provider are left alone.

.EXAMPLE
  .\scripts\sync-opencode.ps1
  .\scripts\sync-opencode.ps1 -Provider bionic
  .\scripts\sync-opencode.ps1 -DryRun
#>
[CmdletBinding()]
param(
    [string]$Provider = "localai",
    [string]$BaseUrl = "http://127.0.0.1:1234/v1",
    [string]$Config,
    [switch]$Force,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Script = Join-Path $Root "scripts\sync_opencode_models.py"
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Host "[x] No virtual environment at $Python" -ForegroundColor Red
    Write-Host "    Run .\scripts\setup.ps1 first." -ForegroundColor Red
    exit 1
}

$arguments = @($Script, "--provider", $Provider, "--base-url", $BaseUrl)
if ($Config) { $arguments += @("--config", $Config) }
if ($Force)   { $arguments += "--force" }
if ($DryRun)  { $arguments += "--dry-run" }

& $Python @arguments
exit $LASTEXITCODE
