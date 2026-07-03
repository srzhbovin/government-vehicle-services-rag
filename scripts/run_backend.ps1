$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Virtual environment not found. Follow the installation steps in README.md."
}

Set-Location $ProjectRoot
& $Python -m uvicorn backend.main:app --app-dir src --host 127.0.0.1 --port 8000

