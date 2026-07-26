$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$configPath = Join-Path $repositoryRoot "configs\mlflow.yaml"
$mlflowExecutable = Join-Path $repositoryRoot ".venv\Scripts\mlflow.exe"

if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "MLflow configuration is missing: $configPath"
}
if (-not (Test-Path -LiteralPath $mlflowExecutable -PathType Leaf)) {
    throw "MLflow is not installed in the project virtual environment: $mlflowExecutable"
}

$configText = Get-Content -LiteralPath $configPath -Raw
$requiredSettings = @(
    "host: 127.0.0.1",
    "port: 5000",
    "backend_database: artifacts/mlflow/mlflow.db",
    "artifact_root: artifacts/mlflow/artifacts"
)
foreach ($setting in $requiredSettings) {
    if (-not $configText.Contains($setting)) {
        throw "Unsupported or missing MLflow setting in configs/mlflow.yaml: $setting"
    }
}

$databasePath = Join-Path $repositoryRoot "artifacts\mlflow\mlflow.db"
$artifactPath = Join-Path $repositoryRoot "artifacts\mlflow\artifacts"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $databasePath) | Out-Null
New-Item -ItemType Directory -Force -Path $artifactPath | Out-Null

$databaseUriPath = $databasePath.Replace("\", "/")
$artifactUri = ([System.Uri]$artifactPath).AbsoluteUri

Write-Host "MLflow UI: http://127.0.0.1:5000"
Write-Host "Press Ctrl+C to stop the server."

& $mlflowExecutable server `
    --host 127.0.0.1 `
    --port 5000 `
    --backend-store-uri "sqlite:///$databaseUriPath" `
    --default-artifact-root $artifactUri `
    --no-serve-artifacts
