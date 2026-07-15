$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ComposeFile = Join-Path $ProjectRoot "deploy\priority_gateway\compose.yaml"
$EnvFile = Join-Path $ProjectRoot "deploy\priority_gateway\.env"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker is not installed or is unavailable in PATH."
}
if (-not (Test-Path $EnvFile)) {
    throw "Create deploy\priority_gateway\.env from .env.example and replace the placeholder secrets."
}

& docker compose --env-file $EnvFile -f $ComposeFile up -d --build
if ($LASTEXITCODE -ne 0) {
    throw "Priority Gateway stack failed to start."
}

$Deadline = (Get-Date).AddMinutes(3)
do {
    try {
        $Health = Invoke-RestMethod -Uri "http://127.0.0.1:4000/health" -TimeoutSec 3
        if ($Health.status -eq "ok") {
            Write-Host "Priority Gateway is ready at http://127.0.0.1:4000"
            exit 0
        }
    }
    catch {
        Start-Sleep -Seconds 2
    }
} while ((Get-Date) -lt $Deadline)

throw "The containers started, but Priority Gateway did not become healthy within 3 minutes."
