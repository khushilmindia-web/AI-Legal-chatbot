$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root "venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Virtual environment Python not found at $python"
    exit 1
}

$staleProcesses = @(
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.CommandLine -and
            $_.CommandLine -like "*uvicorn backend.main:app*" -and
            $_.CommandLine -like "*--host 127.0.0.1*" -and
            $_.CommandLine -like "*--port 5000*"
        }
)

foreach ($process in $staleProcesses) {
    if ($process.ProcessId -ne $PID) {
        Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

& $python -m uvicorn backend.main:app --host 127.0.0.1 --port 5000 --reload
