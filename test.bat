@echo off
setlocal

:: If arguments provided, pass them directly to test_single.py
if not "%~1"=="" (
    .venv\Scripts\python.exe tests/test_single.py %*
    goto :eof
)

:: Interactive mode if no arguments are provided
echo ========================================================================
echo   SafeSight AI — Quick Moderation & Forensics Terminal Tester
echo ========================================================================
echo.
echo Drag and drop an image file here, or type text to moderate:
echo Examples:
echo   - Type: I hate everyone, death to them
echo   - Or drag: sample_images\nature_sample.jpg
echo.
set /p USER_INPUT="Input > "

if "%USER_INPUT%"=="" (
    echo [!] No input provided.
    pause
    exit /b 0
)

echo.
.venv\Scripts\python.exe tests/test_single.py %USER_INPUT%
pause
