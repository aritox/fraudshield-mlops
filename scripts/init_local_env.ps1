$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$environmentPath = Join-Path $repositoryRoot ".env"
$gitignorePath = Join-Path $repositoryRoot ".gitignore"

try {
    if (-not (Test-Path -LiteralPath $gitignorePath -PathType Leaf)) {
        throw "Repository .gitignore is missing."
    }
    $ignored = & git -C $repositoryRoot check-ignore ".env"
    if ($LASTEXITCODE -ne 0 -or $ignored -ne ".env") {
        throw ".env is not ignored by Git."
    }
    if (Test-Path -LiteralPath $environmentPath -PathType Leaf) {
        Write-Host "Existing .env preserved."
        exit 0
    }
    $bytes = New-Object byte[] 32
    $generator = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $generator.GetBytes($bytes)
    }
    finally {
        $generator.Dispose()
    }
    $password = -join ($bytes | ForEach-Object { $_.ToString("x2") })
    $content = @(
        "POSTGRES_DB=fraudshield"
        "POSTGRES_USER=fraudshield_app"
        "POSTGRES_PASSWORD=$password"
        "FRAUDSHIELD_POSTGRES_PORT=5432"
        "FRAUDSHIELD_API_PORT=8000"
    ) -join [Environment]::NewLine
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($environmentPath, $content + [Environment]::NewLine, $encoding)
    Write-Host ".env created."
}
catch {
    throw "Local environment initialization failed safely: $($_.Exception.Message)"
}
