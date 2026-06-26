$ErrorActionPreference = "Stop"

Set-Location "d:\model python"

$python = ".\.venv\Scripts\python.exe"

Write-Host "=== Fetch MT5 data ==="
& $python main.py fetch --symbol EURUSD --timeframe H1 --bars 1000
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Train model ==="
& $python main.py train
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Backtest ==="
& $python main.py backtest
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Current signal ==="
& $python main.py signal
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Pipeline completed ==="
