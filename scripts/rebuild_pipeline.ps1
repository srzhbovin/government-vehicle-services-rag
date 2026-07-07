$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Virtual environment not found. Run scripts/setup.ps1 first."
}

Set-Location $ProjectRoot
$env:PYTHONPATH = "src"

& $Python -m rag_pipeline.build_token_chunks
if ($LASTEXITCODE -ne 0) {
    throw "Chunk rebuild failed."
}
& $Python -m rag_pipeline.build_rag_index
if ($LASTEXITCODE -ne 0) {
    throw "Index rebuild failed."
}

Write-Host "RAG chunks and index were rebuilt."
