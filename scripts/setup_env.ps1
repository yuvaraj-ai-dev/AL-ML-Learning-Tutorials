# PowerShell helper: create venv and install requirements
param(
    [string]$venvName = '.venv'
)

Write-Host "Creating virtual environment $venvName"
python -m venv $venvName
Write-Host "Activating virtual environment"
& "$venvName\Scripts\Activate.ps1"
Write-Host "Installing dependencies from requirements.txt"
pip install -r requirements.txt
