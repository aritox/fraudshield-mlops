$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repositoryRoot
& docker compose ps --all
if ($LASTEXITCODE -ne 0) { throw "Could not read Docker Compose service status." }
