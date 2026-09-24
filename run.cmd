@echo off
rem Start the downloader, setting up the virtual environment on first run.
rem
rem Exists because the obvious command is wrong: `python scripts/serve.py` uses whatever
rem Python is on PATH, which is not the one the dependencies were installed into. That
rem produces a bare ModuleNotFoundError, which says nothing about the actual problem.
setlocal
cd /d "%~dp0"

set PY=.venv\Scripts\python.exe

if not exist "%PY%" (
    echo Creating the virtual environment...
    py -3 -m venv .venv || (echo Could not create .venv - is Python 3.12+ installed? & exit /b 1)
    echo Installing dependencies...
    "%PY%" -m pip install --upgrade pip --quiet
    "%PY%" -m pip install -e ".[desktop]" --quiet || (echo Dependency install failed. & exit /b 1)
    echo Done.
    echo.
)

"%PY%" scripts\serve.py %*
