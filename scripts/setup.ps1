$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Set-Location $ProjectRoot

if (-not (Test-Path $Python)) {
    py -3.13 -m venv .venv
}

& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "pip upgrade failed."
}
& $Python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed."
}

$EnvPath = Join-Path $ProjectRoot ".env"
$ExampleEnvPath = Join-Path $ProjectRoot ".env.example"
if (-not (Test-Path $EnvPath)) {
    Copy-Item -Path $ExampleEnvPath -Destination $EnvPath
    Write-Host "Created .env from .env.example. Fill in Yandex credentials before generation."
}

Write-Host "Environment is ready."
