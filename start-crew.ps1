Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$EnvFile = Join-Path $Root ".env"

Set-Location -LiteralPath $Root

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtual environment not found: $Python"
}
if (-not (Test-Path -LiteralPath $EnvFile)) {
    throw ".env not found: $EnvFile"
}

$EnvText = [System.IO.File]::ReadAllText($EnvFile)
$KeyLine = (
    $EnvText -split "`r?`n" |
    Where-Object { $_ -match '^OPENAI_API_KEY=' } |
    Select-Object -First 1
)

if (
    [string]::IsNullOrWhiteSpace($KeyLine) -or
    $KeyLine -eq "OPENAI_API_KEY="
) {
    Write-Host "OPENAI_API_KEY is empty." -ForegroundColor Yellow
    Read-Host "Press Enter to close"
    exit 1
}

Write-Host ""
Write-Host "CREW v0.0.3" -ForegroundColor Cyan
Write-Host "Configurable Panel A / Panel B"
Write-Host "Three-call blind Sol review"
Write-Host "http://127.0.0.1:8000"
Write-Host "Ctrl+C to stop"
Write-Host ""

Start-Process "http://127.0.0.1:8000"
& $Python -m uvicorn app:app --host 127.0.0.1 --port 8000