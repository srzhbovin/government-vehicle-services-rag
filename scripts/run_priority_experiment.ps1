param(
    [int[]]$Concurrency = @(1, 2, 4, 8),
    [int]$Repeats = 3,
    [double]$ServiceMs = 100
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Virtual environment is missing. Run .\scripts\setup.ps1 first."
}

Push-Location $ProjectRoot
try {
    $env:PYTHONPATH = Join-Path $ProjectRoot "src"
    & $Python -m priority_gateway.benchmark `
        --concurrency $Concurrency `
        --repeats $Repeats `
        --service-ms $ServiceMs
    if ($LASTEXITCODE -ne 0) {
        throw "Priority scheduling experiment failed."
    }
}
finally {
    Pop-Location
}
