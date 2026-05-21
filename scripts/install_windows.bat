@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0\.."

rem Optional: set PYTHON_EXE to a full Python path before running this script.
rem Example:
rem set PYTHON_EXE=C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe

if exist ".venv\Scripts\python.exe" (
    echo Virtual environment already exists: .venv
) else (
    echo Creating virtual environment...

    if defined PYTHON_EXE (
        "%PYTHON_EXE%" -m venv .venv
    ) else (
        py -3.11 -m venv .venv 2>nul
        if errorlevel 1 py -3.10 -m venv .venv 2>nul
        if errorlevel 1 python -m venv .venv
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Failed to create .venv.
    echo Please install the official Python 3.10 or 3.11 with pip and venv support.
    echo Do not use python-embed for this service.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo.
echo Install finished.
echo You can now run scripts\start_windows.bat
pause
