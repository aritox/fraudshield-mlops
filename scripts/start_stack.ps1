$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$pythonExecutable = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
$environmentPath = Join-Path $repositoryRoot ".env"

Set-Location -LiteralPath $repositoryRoot
try {
    & docker info *> $null
    if ($LASTEXITCODE -ne 0) { throw "Docker Engine is unavailable." }
    if (-not (Test-Path -LiteralPath $environmentPath -PathType Leaf)) {
        & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "init_local_env.ps1")
    }
    & $pythonExecutable -m fraudshield.container.verify_package
    if ($LASTEXITCODE -ne 0) { throw "Container model package verification failed." }
    & docker compose config --quiet
    if ($LASTEXITCODE -ne 0) { throw "Docker Compose configuration is invalid." }
    & docker compose build
    if ($LASTEXITCODE -ne 0) { throw "Docker Compose build failed." }
    & docker compose up -d
    if ($LASTEXITCODE -ne 0) { throw "Docker Compose startup failed." }

    $healthy = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        $statusLines = & docker compose ps --all --format json
        $services = @($statusLines | ForEach-Object { $_ | ConvertFrom-Json })
        $postgres = $services | Where-Object { $_.Service -eq "postgres" }
        $api = $services | Where-Object { $_.Service -eq "api" }
        $monitor = $services | Where-Object { $_.Service -eq "monitor" }
        $prometheus = $services | Where-Object { $_.Service -eq "prometheus" }
        $grafana = $services | Where-Object { $_.Service -eq "grafana" }
        $migrate = $services | Where-Object { $_.Service -eq "migrate" }
        $postgresHealth = if ($postgres) { $postgres.Health } else { "missing" }
        $apiHealth = if ($api) { $api.Health } else { "missing" }
        $monitorHealth = if ($monitor) { $monitor.Health } else { "missing" }
        $prometheusHealth = if ($prometheus) { $prometheus.Health } else { "missing" }
        $grafanaHealth = if ($grafana) { $grafana.Health } else { "missing" }
        $migrateExit = if ($migrate) { [string]$migrate.ExitCode } else { "missing" }
        if (
            $postgresHealth -eq "healthy" -and
            $apiHealth -eq "healthy" -and
            $monitorHealth -eq "healthy" -and
            $prometheusHealth -eq "healthy" -and
            $grafanaHealth -eq "healthy" -and
            $migrateExit -eq "0"
        ) {
            $healthy = $true
            break
        }
        Start-Sleep -Seconds 2
    }
    if (-not $healthy) { throw "Services did not become healthy before the timeout." }

    $settings = @{}
    Get-Content -LiteralPath $environmentPath | ForEach-Object {
        if ($_ -match '^([^#=]+)=(.*)$') { $settings[$matches[1]] = $matches[2] }
    }
    $apiPort = if ($settings["FRAUDSHIELD_API_PORT"]) { $settings["FRAUDSHIELD_API_PORT"] } else { "8000" }
    $databasePort = if ($settings["FRAUDSHIELD_POSTGRES_PORT"]) { $settings["FRAUDSHIELD_POSTGRES_PORT"] } else { "5432" }
    $grafanaPort = if ($settings["FRAUDSHIELD_GRAFANA_PORT"]) { $settings["FRAUDSHIELD_GRAFANA_PORT"] } else { "3000" }
    Write-Host "FraudShield API: http://127.0.0.1:$apiPort"
    Write-Host "Swagger UI: http://127.0.0.1:$apiPort/docs"
    Write-Host "Prometheus: http://127.0.0.1:9090"
    Write-Host "Grafana: http://127.0.0.1:$grafanaPort"
    Write-Host "PostgreSQL: 127.0.0.1:$databasePort"
}
catch {
    Write-Error $_.Exception.Message
    & docker compose ps
    & docker compose logs --tail 100 postgres migrate api monitor prometheus grafana
    exit 1
}
