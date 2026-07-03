$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Virtual environment not found. Follow the installation steps in README.md."
}

Set-Location $ProjectRoot
& $Python -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501

