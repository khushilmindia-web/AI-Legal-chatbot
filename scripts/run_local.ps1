$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root "venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Virtual environment Python not found at $python"
    exit 1
}

& $python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
