# Start the downloader, setting up the virtual environment on first run.
#
# Exists because the obvious command is wrong: `python scripts/serve.py` uses whatever
# Python is on PATH, not the one the dependencies were installed into, and the result is a
# bare ModuleNotFoundError that says nothing about the real problem.

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

if (-not (Test-Path $py)) {
    Write-Host 'Creating the virtual environment...'
    # `py -3` rather than `python`: on Windows the latter can be the Store stub that opens
    # the Microsoft Store instead of running anything.
    & py -3 -m venv .venv
    if (-not (Test-Path $py)) { throw 'Could not create .venv — is Python 3.12+ installed?' }

    Write-Host 'Installing dependencies...'
    & $py -m pip install --upgrade pip --quiet
    & $py -m pip install -e '.[hf]' --quiet
    Write-Host "Done.`n"
}

& $py scripts\serve.py --open @args
