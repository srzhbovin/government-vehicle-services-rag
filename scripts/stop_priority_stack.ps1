$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ComposeFile = Join-Path $ProjectRoot "deploy\priority_gateway\compose.yaml"
$EnvFile = Join-Path $ProjectRoot "deploy\priority_gateway\.env"

if (-not (Test-Path $EnvFile)) {
    throw "deploy\priority_gateway\.env is missing."
}

& docker compose --env-file $EnvFile -f $ComposeFile down
if ($LASTEXITCODE -ne 0) {
    throw "Priority Gateway stack failed to stop cleanly."
}
