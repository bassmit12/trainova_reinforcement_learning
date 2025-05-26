@echo off
echo Trainova Feedback Network Model Reset Tool
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python and try again.
    exit /b 1
)

echo Available reset options:
echo 1. Reset all data (full reset)
echo 2. Reset only feedback history
echo 3. Reset only prediction weights
echo 4. Reset only trained scalers and models
echo.

set /p RESET_OPTION="Enter option number (1-4): "

set RESET_TYPE=all
if "%RESET_OPTION%"=="2" set RESET_TYPE=feedback
if "%RESET_OPTION%"=="3" set RESET_TYPE=weights
if "%RESET_OPTION%"=="4" set RESET_TYPE=scalers

echo.
echo You've selected to reset: %RESET_TYPE%
set /p CONFIRM="Are you sure you want to proceed? (y/n): "

if /i not "%CONFIRM%"=="y" (
    echo.
    echo Reset operation cancelled.
    exit /b 0
)

echo.
echo Resetting model data...

REM Create a temporary Python script to execute the reset operation
echo import sys > temp_reset.py
echo from models.feedback_prediction_model import FeedbackBasedPredictionModel >> temp_reset.py
echo. >> temp_reset.py
echo model = FeedbackBasedPredictionModel() >> temp_reset.py
echo result = model.clear_data('%RESET_TYPE%') >> temp_reset.py
echo print(result['message']) >> temp_reset.py

REM Execute the temporary Python script
python temp_reset.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: Failed to reset model data.
    del temp_reset.py
    exit /b 1
)

REM Delete the temporary Python script
del temp_reset.py

echo.
echo Reset operation completed successfully.
echo.
pause