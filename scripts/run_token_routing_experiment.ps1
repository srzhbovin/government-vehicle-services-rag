param(
    [switch]$Local
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$OutputDirectory = Join-Path $ProjectRoot "data\experiments\priority_gateway\token_aware"

if ($Local) {
    $Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $Python)) {
        throw "Virtual environment is missing. Run .\scripts\setup.ps1 first."
    }
    Push-Location $ProjectRoot
    try {
        $env:PYTHONPATH = Join-Path $ProjectRoot "src"
        & $Python -m priority_gateway.token_routing_benchmark `
            --output-dir $OutputDirectory
        if ($LASTEXITCODE -ne 0) {
            throw "Token-aware routing experiment failed."
        }
    }
    finally {
        Pop-Location
    }
}
else {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Docker is not installed or is unavailable in PATH."
    }
    $ComposeFile = Join-Path $ProjectRoot "deploy\priority_gateway\token_benchmark.compose.yaml"
    & docker compose -f $ComposeFile run --rm --build token-routing-benchmark
    if ($LASTEXITCODE -ne 0) {
        throw "Docker token-aware routing experiment failed."
    }
}

Write-Host "Results: $OutputDirectory"
