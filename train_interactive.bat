@echo off
echo Trainova Feedback Network - Interactive Training Mode
echo ====================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python and try again.
    exit /b 1
)

REM Run the training script
python train_interactive.py

echo.
pause