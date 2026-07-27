$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repositoryRoot
& docker compose ps --all
if ($LASTEXITCODE -ne 0) { throw "Could not read Docker Compose service status." }
& docker volume ls --filter name=fraudshield_postgres_data
if ($LASTEXITCODE -ne 0) { throw "Could not verify the PostgreSQL volume." }
& docker volume ls --filter name=fraudshield_prometheus_data
if ($LASTEXITCODE -ne 0) { throw "Could not verify the Prometheus volume." }
& docker volume ls --filter name=fraudshield_grafana_data
if ($LASTEXITCODE -ne 0) { throw "Could not verify the Grafana volume." }
