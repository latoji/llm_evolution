# scripts/dev.ps1 — start backend + frontend for local development (Windows).
# Usage: pwsh scripts/dev.ps1
param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173
)

$Root = Split-Path -Parent $PSScriptRoot

# Activate venv if present
$VenvActivate = Join-Path $Root "venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    & $VenvActivate
}

Write-Host "Starting backend on http://localhost:$BackendPort ..."
$BackendProc = Start-Process -PassThru -FilePath "python" `
    -ArgumentList "-m", "uvicorn", "api.main:app", "--reload", "--port", $BackendPort `
    -WorkingDirectory $Root

Write-Host "Starting frontend on http://localhost:$FrontendPort ..."
$FrontendProc = Start-Process -PassThru -FilePath "npm" `
    -ArgumentList "run", "dev" `
    -WorkingDirectory (Join-Path $Root "frontend")

Write-Host "Press Ctrl+C to stop both servers."
try {
    Wait-Process -Id $BackendProc.Id, $FrontendProc.Id
} finally {
    Stop-Process -Id $BackendProc.Id -Force -ErrorAction SilentlyContinue
    Stop-Process -Id $FrontendProc.Id -Force -ErrorAction SilentlyContinue
}
