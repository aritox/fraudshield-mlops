$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repositoryRoot

$monitorId = & docker compose ps --quiet monitor
if ($LASTEXITCODE -ne 0 -or -not $monitorId) {
    throw "The monitoring service is not running. Start the stack first."
}

& docker compose exec --no-TTY monitor python -m fraudshield.monitoring.run --once
if ($LASTEXITCODE -ne 0) { throw "The one-shot monitoring calculation failed." }
