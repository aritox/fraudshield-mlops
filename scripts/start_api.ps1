param(
    [switch]$Reload
)

$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $repositoryRoot ".venv"
$pythonExecutable = Join-Path $venvPath "Scripts\python.exe"
$configPath = Join-Path $repositoryRoot "configs\api.yaml"
$mlflowDatabase = Join-Path $repositoryRoot "artifacts\mlflow\mlflow.db"
$mlflowArtifacts = Join-Path $repositoryRoot "artifacts\mlflow\artifacts"

if (-not (Test-Path -LiteralPath $venvPath -PathType Container)) {
    throw "Project virtual environment is missing: $venvPath"
}
if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) {
    throw "Project Python interpreter is missing: $pythonExecutable"
}
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "API configuration is missing: $configPath"
}
if (-not (Test-Path -LiteralPath $mlflowDatabase -PathType Leaf)) {
    throw "Local MLflow database is missing: $mlflowDatabase"
}
if (-not (Test-Path -LiteralPath $mlflowArtifacts -PathType Container)) {
    throw "Local MLflow artifact store is missing: $mlflowArtifacts"
}

$configText = Get-Content -LiteralPath $configPath -Raw
foreach ($setting in @("host: 127.0.0.1", "port: 8000")) {
    if (-not $configText.Contains($setting)) {
        throw "Unsupported or missing API setting in configs/api.yaml: $setting"
    }
}

Set-Location -LiteralPath $repositoryRoot
Write-Host "FraudShield API: http://127.0.0.1:8000"
Write-Host "Swagger UI: http://127.0.0.1:8000/docs"
Write-Host "ReDoc: http://127.0.0.1:8000/redoc"
Write-Host "Press Ctrl+C to stop the API."

$arguments = @(
    "-m",
    "uvicorn",
    "fraudshield.api.main:app",
    "--host",
    "127.0.0.1",
    "--port",
    "8000"
)
if ($Reload) {
    $arguments += "--reload"
}

& $pythonExecutable @arguments
