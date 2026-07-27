$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$environmentPath = Join-Path $repositoryRoot ".env"
$gitignorePath = Join-Path $repositoryRoot ".gitignore"

function New-RandomSecret {
    $bytes = New-Object byte[] 32
    $generator = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $generator.GetBytes($bytes)
    }
    finally {
        $generator.Dispose()
    }
    return -join ($bytes | ForEach-Object { $_.ToString("x2") })
}

try {
    if (-not (Test-Path -LiteralPath $gitignorePath -PathType Leaf)) {
        throw "Repository .gitignore is missing."
    }
    $ignored = & git -C $repositoryRoot check-ignore ".env"
    if ($LASTEXITCODE -ne 0 -or $ignored -ne ".env") {
        throw ".env is not ignored by Git."
    }
    $created = -not (Test-Path -LiteralPath $environmentPath -PathType Leaf)
    $lines = New-Object System.Collections.Generic.List[string]
    if ($created) {
        $postgresPassword = New-RandomSecret
        $lines.Add("POSTGRES_DB=fraudshield")
        $lines.Add("POSTGRES_USER=fraudshield_app")
        $lines.Add("POSTGRES_PASSWORD=$postgresPassword")
        $lines.Add("FRAUDSHIELD_POSTGRES_PORT=5432")
        $lines.Add("FRAUDSHIELD_API_PORT=8000")
    }
    else {
        [System.IO.File]::ReadAllLines($environmentPath) | ForEach-Object {
            $lines.Add($_)
        }
    }

    $variables = @{}
    foreach ($line in $lines) {
        if ($line -match '^([^#=]+)=') {
            $variables[$matches[1]] = $true
        }
    }
    $updated = $false
    if (-not $variables.ContainsKey("GRAFANA_ADMIN_USER")) {
        $lines.Add("GRAFANA_ADMIN_USER=admin")
        $updated = $true
    }
    if (-not $variables.ContainsKey("GRAFANA_ADMIN_PASSWORD")) {
        $grafanaPassword = New-RandomSecret
        $lines.Add("GRAFANA_ADMIN_PASSWORD=$grafanaPassword")
        $updated = $true
    }
    if (-not $variables.ContainsKey("FRAUDSHIELD_GRAFANA_PORT")) {
        $lines.Add("FRAUDSHIELD_GRAFANA_PORT=3000")
        $updated = $true
    }

    if ($created -or $updated) {
        $content = $lines -join [Environment]::NewLine
        $encoding = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText(
            $environmentPath,
            $content + [Environment]::NewLine,
            $encoding
        )
    }
    if ($created) {
        Write-Host ".env created."
    }
    elseif ($updated) {
        Write-Host "Existing .env preserved and missing Grafana settings added."
    }
    else {
        Write-Host "Existing .env preserved."
    }
}
catch {
    throw "Local environment initialization failed safely: $($_.Exception.Message)"
}
