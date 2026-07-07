$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$RunDir = Join-Path $ProjectRoot ".run"

if (-not (Test-Path $Python)) {
    throw "Virtual environment not found. Run scripts/setup.ps1 first."
}

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
Set-Location $ProjectRoot

$BackendLog = Join-Path $RunDir "backend.log"
$BackendErr = Join-Path $RunDir "backend.err.log"
$FrontendLog = Join-Path $RunDir "frontend.log"
$FrontendErr = Join-Path $RunDir "frontend.err.log"

$Backend = Start-Process `
    -FilePath $Python `
    -ArgumentList @("-m", "uvicorn", "backend.main:app", "--app-dir", "src", "--host", "127.0.0.1", "--port", "8000") `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $BackendLog `
    -RedirectStandardError $BackendErr `
    -WindowStyle Hidden `
    -PassThru

Start-Sleep -Seconds 4

$Frontend = Start-Process `
    -FilePath $Python `
    -ArgumentList @("gradio_app.py") `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $FrontendLog `
    -RedirectStandardError $FrontendErr `
    -WindowStyle Hidden `
    -PassThru

$Backend.Id | Set-Content -Path (Join-Path $RunDir "backend.pid")
$Frontend.Id | Set-Content -Path (Join-Path $RunDir "frontend.pid")

Write-Host "Backend:  http://127.0.0.1:8000/docs"
Write-Host "Gradio:   http://127.0.0.1:7860"
Write-Host "Logs:     .run/backend.log, .run/frontend.log"
Write-Host "Stop app: powershell -ExecutionPolicy Bypass -File scripts/stop_app.ps1"
