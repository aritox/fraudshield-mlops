$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repositoryRoot
& docker compose down
if ($LASTEXITCODE -ne 0) { throw "Docker Compose shutdown failed." }
Write-Host "FraudShield stack stopped. The postgres_data volume and audit records remain stored."
