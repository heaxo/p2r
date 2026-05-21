Set-Location (Join-Path $PSScriptRoot "..")

# Simple token authorization. The HTML page must use the same token.
$env:APP_TOKEN = "change-me-please"

# Change this to your real YOLO weight path.
$env:YOLO_MODEL_PATH = "best2.pt"

# Output files will be saved here and exposed through /files.
$env:OUTPUT_ROOT = "measure_out"

# SAM2 model name. Use the same value as your original console program.
$env:SAM_MODEL = "sam2"

# Default YOLO parameters.
$env:YOLO_IMGSZ = "1280"
$env:YOLO_CONF = "0.35"

# Set true to avoid concurrent GPU/SAM2 calls. Keep true first.
$env:SERIALIZE_PROCESSING = "true"
$env:PYTHONUTF8 = "1"

$venvPython = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (!(Test-Path $venvPython)) {
    Write-Host ".venv was not found. Please run scripts\install_windows.bat first."
    Read-Host "Press Enter to exit"
    exit 1
}

& $venvPython -c "import uvicorn" | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "uvicorn is not installed in .venv. Please run scripts\install_windows.bat again."
    Read-Host "Press Enter to exit"
    exit 1
}

& $venvPython -m uvicorn app.main:app --host 0.0.0.0 --port 8000
