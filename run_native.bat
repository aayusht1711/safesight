@echo off
setlocal

echo ========================================================================
echo   SafeSight AI — One-Click Native Local Runtime (No Docker Needed)
echo ========================================================================
echo.

:: 1. Verify Virtual Environment
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in .venv!
    echo Please make sure the .venv directory is present.
    pause
    exit /b 1
)

:: 2. Check Native Ollama Status
where ollama >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [OK] Native Ollama is installed on your computer.
    curl.exe -s http://localhost:11434/ >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        echo [OK] Ollama service is active on http://localhost:11434
    ) else (
        echo [*] Starting Ollama service...
        start "" "C:\Users\%USERNAME%\AppData\Local\Programs\Ollama\ollama.exe"
        timeout /t 3 /nobreak >nul
    )
) else (
    echo [INFO] Ollama not found in PATH — running with Fast-Path Forensics mode.
)

echo.
echo [*] Launching SafeSight AI Server...
echo.
echo ========================================================================
echo   SYSTEM READY
echo ========================================================================
echo   Swagger UI Docs:   http://localhost:8000/docs
echo   Health Check:      http://localhost:8000/health
echo   Ollama Endpoint:   http://localhost:11434
echo ========================================================================
echo.
echo Press CTRL+C at any time to stop the server.
echo.

.venv\Scripts\python.exe run_server.py
pause
