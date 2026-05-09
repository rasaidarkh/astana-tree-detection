@echo off
REM Astana Tree Detection - launcher (Windows)

if not exist venv\ (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate
)

echo Starting server on http://localhost:8000
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
