@echo off
echo Starting Trainer Recommendation System...

:: Check if Python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    goto :end
)

:: Check if trainer_recommendation.py exists
if not exist trainer_recommendation.py (
    echo Error: trainer_recommendation.py not found in the current directory
    goto :end
)

:: Run the trainer recommendation system
python trainer_recommendation.py
if %errorlevel% neq 0 (
    echo Error: The trainer recommendation system encountered an error
)

:end
pause