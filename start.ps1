# Astana Tree Detection - launcher (PowerShell)

if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv venv
    & .\venv\Scripts\Activate.ps1
    pip install -r requirements.txt
} else {
    & .\venv\Scripts\Activate.ps1
}

Write-Host "Starting server on http://localhost:8000" -ForegroundColor Green
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
