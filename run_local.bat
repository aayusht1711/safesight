@echo off
setlocal enabledelayedexpansion

echo ========================================================================
echo   SafeSight AI — One-Click Local Runtime Environment
echo ========================================================================
echo.

:: 1. Verify Docker CLI is available
where docker >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker is not installed or not in system PATH.
    echo Please install Docker Desktop from https://www.docker.com/products/docker-desktop
    exit /b 1
)

:: 2. Verify Docker Daemon is running
docker info >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker daemon is not running!
    echo Please open Docker Desktop, wait for the engine to start, and run this script again.
    exit /b 1
)

:: 3. Determine docker compose command (compose v2 vs v1)
docker compose version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set COMPOSE_CMD=docker compose
) else (
    set COMPOSE_CMD=docker-compose
)

echo [*] Starting SafeSight AI containers via %COMPOSE_CMD%...
%COMPOSE_CMD% up -d --build
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to start containers via Docker Compose.
    exit /b 1
)

:: 4. Poll for Ollama readiness
echo [*] Waiting for Ollama container to complete initialization...
set /a ATTEMPTS=0
:WAIT_OLLAMA
set /a ATTEMPTS+=1
powershell -Command "try { $res = Invoke-WebRequest -Uri 'http://localhost:11434' -UseBasicParsing -TimeoutSec 2; if ($res.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [OK] Ollama service is up and responsive on port 11434.
    goto PULL_MODEL
)
if %ATTEMPTS% geq 30 (
    echo [WARN] Ollama container initialization check timed out after 60s. Attempting model pull anyway...
    goto PULL_MODEL
)
timeout /t 2 /nobreak >nul
goto WAIT_OLLAMA

:PULL_MODEL
:: 5. Pull the Qwen2.5 quantized model into the persistent volume
echo [*] Pulling vision-language model weights (qwen2.5:3b)...
echo     (This runs once; weights are permanently saved to volume: safesight_ollama_cache)
docker exec -i safesight-ollama ollama pull qwen2.5:3b
if %ERRORLEVEL% neq 0 (
    echo [WARN] Model pull encountered a warning. You can retry with:
    echo        docker exec -it safesight-ollama ollama pull qwen2.5:3b
)

:: 6. Display Success Banner
echo.
echo ========================================================================
echo   SAFESIGHT AI — SYSTEM READY
echo ========================================================================
echo.
echo   Swagger UI Docs:    http://localhost:8000/docs
echo   Alternative Docs:   http://localhost:8000/redoc
echo   Health Check:       http://localhost:8000/health
echo   Ollama Endpoint:    http://localhost:11434
echo   Active VLM Model:   qwen2.5-vl:3b
echo.
echo   Test with sample CLI:
echo     python tests/test_single.py --text "Check this content" --url http://localhost:8000/moderate
echo.
echo ========================================================================
echo.
pause
