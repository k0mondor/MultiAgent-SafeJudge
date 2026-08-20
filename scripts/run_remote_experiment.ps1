[CmdletBinding()]
param(
    [string]$InputPath = "data/formal/jailbreakv-balanced-5.jsonl",
    [string]$MediaRoot = "",
    [ValidateSet("text", "text+image", "text+audio", "text+video")]
    [string]$Capability = "text+image",
    [ValidateRange(1, 1000000)]
    [int]$Limit = 5,
    [string]$TargetProfile = "glm-4.6v-target-v2",
    [string]$GroundingProfile = "glm-4.6v-grounding-v2",
    [string]$OutputDir = "",
    [switch]$Yes
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$EnvPath = Join-Path $RepoRoot ".env"
$ExampleEnvPath = Join-Path $RepoRoot ".env.example"
$PythonPath = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$AcceptanceScript = Join-Path $RepoRoot "scripts\run_e2e_acceptance.py"
$JuryPlan = Join-Path $RepoRoot "config\juries\m3-deepseek-llamaguard-remote-v1.toml"

Set-Location $RepoRoot

function Get-AbsoluteExperimentPath {
    param([Parameter(Mandatory = $true)][string]$PathValue)
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $PathValue))
}

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw ".venv was not found. Run: uv sync --extra dev --extra orchestration"
}

if (-not (Test-Path -LiteralPath $EnvPath -PathType Leaf)) {
    Copy-Item -LiteralPath $ExampleEnvPath -Destination $EnvPath
}

if ([string]::IsNullOrWhiteSpace($MediaRoot)) {
    $LocalJailBreakRoot = "D:\JailBreakV_28K\JailBreakV_28K"
    $MediaRoot = if (Test-Path -LiteralPath $LocalJailBreakRoot -PathType Container) {
        $LocalJailBreakRoot
    }
    else {
        "."
    }
}

$ResolvedInput = Get-AbsoluteExperimentPath $InputPath
$ResolvedMediaRoot = Get-AbsoluteExperimentPath $MediaRoot
if (-not (Test-Path -LiteralPath $ResolvedInput -PathType Leaf)) {
    throw "Input data was not found: $ResolvedInput`nPlace Canonical JSONL there or pass -InputPath."
}
if (-not (Test-Path -LiteralPath $ResolvedMediaRoot -PathType Container)) {
    throw "Media root was not found: $ResolvedMediaRoot"
}

$EnvText = Get-Content -LiteralPath $EnvPath -Raw
$KeyMatch = [regex]::Match($EnvText, '(?m)^OPENROUTER_API_KEY\s*=\s*(.+?)\s*$')
$ApiKey = if ($KeyMatch.Success) { $KeyMatch.Groups[1].Value.Trim() } else { "" }

if ([string]::IsNullOrWhiteSpace($ApiKey)) {
    Write-Host "First run: paste the NEW OpenRouter API key. It will not be displayed." -ForegroundColor Cyan
    $SecureKey = Read-Host "OpenRouter API Key" -AsSecureString
    $ApiKey = [System.Net.NetworkCredential]::new("", $SecureKey).Password.Trim()
    if ([string]::IsNullOrWhiteSpace($ApiKey)) {
        throw "No API key was entered. Stopped."
    }
    if ($EnvText -match '(?m)^OPENROUTER_API_KEY\s*=.*$') {
        $EnvText = [regex]::Replace(
            $EnvText,
            '(?m)^OPENROUTER_API_KEY\s*=.*$',
            "OPENROUTER_API_KEY=$ApiKey"
        )
    }
    else {
        $EnvText = "$EnvText`r`nOPENROUTER_API_KEY=$ApiKey`r`n"
    }
    [System.IO.File]::WriteAllText(
        $EnvPath,
        $EnvText,
        [System.Text.UTF8Encoding]::new($false)
    )
    Write-Host "The key was saved to local .env, which Git ignores." -ForegroundColor Green
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputDir = "runs/remote-$TargetProfile-$Timestamp"
}
$ResolvedOutput = Get-AbsoluteExperimentPath $OutputDir

Write-Host ""
Write-Host "Remote experiment configuration" -ForegroundColor Cyan
Write-Host "  Target model:       $TargetProfile"
Write-Host "  Grounding model:    $GroundingProfile"
Write-Host "  Main Judge:         DeepSeek V4 Flash (OpenRouter)"
Write-Host "  Safety sub-judge:   Llama Guard 4 12B (OpenRouter)"
Write-Host "  Taxonomy router:    GB/T 45654-2025 (enabled)"
Write-Host "  Sub-judge context:  compact"
Write-Host "  Sample limit:       $Limit"
Write-Host "  Input:              $ResolvedInput"
Write-Host "  Output:             $ResolvedOutput"
Write-Host ""

if (-not $Yes) {
    $Confirmation = Read-Host "This uses OpenRouter credits. Type RUN to continue"
    if ($Confirmation -cne "RUN") {
        Write-Host "Cancelled. No model request was sent."
        exit 0
    }
}

& $PythonPath $AcceptanceScript `
    --input $ResolvedInput `
    --media-root $ResolvedMediaRoot `
    --target-profile $TargetProfile `
    --grounding-profile $GroundingProfile `
    --jury-plan $JuryPlan `
    --output-dir $ResolvedOutput `
    --capability $Capability `
    --limit $Limit `
    --allow-unqualified-model

if ($LASTEXITCODE -ne 0) {
    throw "Experiment needs attention or failed with exit code $LASTEXITCODE. Read $ResolvedOutput\evaluation-report.md first."
}

Write-Host ""
Write-Host "Experiment complete. Check these files first:" -ForegroundColor Green
Write-Host "  $ResolvedOutput\evaluation-report.md"
Write-Host "  $ResolvedOutput\e2e-acceptance-report.json"
Write-Host "  $ResolvedOutput\evaluations.jsonl"
