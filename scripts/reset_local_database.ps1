param([switch]$ConfirmReset)

$ErrorActionPreference = "Stop"
if (-not $ConfirmReset) {
    throw "Refusing to reset PostgreSQL. Re-run with -ConfirmReset to remove only the Phase 2C volume."
}
$repositoryRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repositoryRoot
& docker compose down
if ($LASTEXITCODE -ne 0) { throw "Could not stop the Phase 2C stack." }
$volumeName = "fraudshield_postgres_data"
$projectLabel = & docker volume inspect --format "{{index .Labels `"com.docker.compose.project`"}}" $volumeName 2>$null
$volumeLabel = & docker volume inspect --format "{{index .Labels `"com.docker.compose.volume`"}}" $volumeName 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Phase 2C PostgreSQL volume does not exist; nothing was removed."
    exit 0
}
if ($projectLabel -ne "fraudshield" -or $volumeLabel -ne "postgres_data") {
    throw "Refusing to remove a volume without the expected Phase 2C Compose labels."
}
& docker volume rm $volumeName
if ($LASTEXITCODE -ne 0) { throw "Could not remove the Phase 2C PostgreSQL volume." }
Write-Host "Removed only the Phase 2C postgres_data volume."
