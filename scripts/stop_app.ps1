$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RunDir = Join-Path $ProjectRoot ".run"

if (-not (Test-Path $RunDir)) {
    Write-Host "No .run directory found."
    exit 0
}

Get-ChildItem -Path $RunDir -Filter "*.pid" | ForEach-Object {
    $ProcessIdText = Get-Content -Path $_.FullName -ErrorAction SilentlyContinue
    if (-not $ProcessIdText) {
        return
    }
    $ProcessId = [int]$ProcessIdText
    $Process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($Process) {
        Stop-Process -Id $ProcessId -Force
        Write-Host "Stopped process $ProcessId"
    }
}
