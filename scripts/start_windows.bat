@echo off
setlocal
cd /d "%~dp0\.."

rem Simple token authorization. The HTML page must use the same token.
set APP_TOKEN=tk_c2VjcmV0LXJhbmRvbS10b2tlbi0xMjM0NTY3OA

rem Change this to your real YOLO weight path.
set YOLO_MODEL_PATH=best2.pt

rem Output files will be saved here and exposed through /files.
set OUTPUT_ROOT=measure_out

rem SAM2 model name. Use the same value as your original console program.
set SAM_MODEL=sam2

rem Default YOLO parameters.
set YOLO_IMGSZ=1280
set YOLO_CONF=0.35

rem Set true to avoid concurrent GPU/SAM2 calls. Keep true first.
set SERIALIZE_PROCESSING=true

rem Make Python use UTF-8 mode on Windows.
set PYTHONUTF8=1

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo .venv was not found.
    echo Please run scripts\install_windows.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -c "import uvicorn" 1>nul 2>nul
if errorlevel 1 (
    echo.
    echo uvicorn is not installed in .venv.
    echo Please run scripts\install_windows.bat again.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8000
pause
