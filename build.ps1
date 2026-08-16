# Build standalone ModelDL.exe using PyInstaller.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

if (-not (Test-Path $py)) {
    Write-Host 'Creating the virtual environment...'
    & py -3 -m venv .venv
    & $py -m pip install --upgrade pip --quiet
    & $py -m pip install -e '.[hf,build]' --quiet
}

& $py scripts\build_exe.py @args
