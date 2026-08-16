@echo off
rem Build standalone ModelDL.exe using PyInstaller.
setlocal
cd /d "%~dp0"

set PY=.venv\Scripts\python.exe

if not exist "%PY%" (
    echo Creating the virtual environment...
    py -3 -m venv .venv || (echo Could not create .venv & exit /b 1)
    "%PY%" -m pip install --upgrade pip --quiet
    "%PY%" -m pip install -e ".[hf,build]" --quiet || (echo Dependency install failed. & exit /b 1)
)

"%PY%" scripts\build_exe.py %*
pause
