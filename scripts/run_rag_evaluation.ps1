$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Virtual environment not found. Create .venv and install requirements first."
}

Push-Location $projectRoot
try {
    & $python -m rag_pipeline.build_rag_test_set
    if ($LASTEXITCODE -ne 0) { throw "Test-set preparation failed." }

    & $python -m rag_pipeline.evaluate_rag
    if ($LASTEXITCODE -ne 0) { throw "RAG evaluation failed." }
}
finally {
    Pop-Location
}
