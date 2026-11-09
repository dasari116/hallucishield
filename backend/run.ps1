$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -m venv .venv
}

& .venv\Scripts\python.exe -m pip install -q -r requirements.txt
& .venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000