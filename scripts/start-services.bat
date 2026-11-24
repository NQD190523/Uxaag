@echo off
REM Startup script for RAG Chat Room Web microservices (Windows)

echo.
echo ========================================
echo   RAG Chat Room Web microservices Startup
echo ========================================
echo.

setlocal enabledelayedexpansion
REM This script lives in uxaag\scripts\; ROOT_DIR is the uxaag\ folder (one level up).
for %%I in ("%~dp0..") do set "ROOT_DIR=%%~fI\"

REM Create logs directory if it doesn't exist
if not exist "%ROOT_DIR%logs" mkdir "%ROOT_DIR%logs"

set /a started_count=0
set /a running_count=0
set /a failed_count=0

REM Canonical service list aligned with api_gateway/src/core/services.yaml + gateway itself
call :start_service "auth-service" "8006" "services\auth_service" "src\main.py"
call :start_service "chat-service" "8005" "services\chat_service" "src\main.py"
call :start_service "document-service" "8001" "services\document_service" "src\main.py"
call :start_service "rag-service" "8002" "services\rag_service" "src\main.py"
call :start_service "vector-store-service" "8003" "services\vector_store_service" "src\main.py"
call :start_service "web-crawler-service" "8004" "services\web_crawler_service" "src\main.py"
call :start_service "api-gateway" "8000" "services\api_gateway" "src\main.py"

echo.
echo ========================================
echo   Startup completed
echo ========================================
echo   Started : %started_count%
echo   Running : %running_count%
echo   Failed  : %failed_count%
echo.
echo Service URLs:
echo   API Gateway:          http://localhost:8000
echo   Document Service:     http://localhost:8001
echo   RAG Service:          http://localhost:8002
echo   Vector Store Service: http://localhost:8003
echo   Web Crawler Service:  http://localhost:8004
echo   Chat Service:         http://localhost:8005
echo   Auth Service:         http://localhost:8006
echo.
echo Each service is running in its own window.
echo Close any window to stop that service.
echo Logs are written to logs\*.log and logs\*.err.log
echo To stream all logs in one console:
echo   view-logs.bat
echo.
pause
exit /b 0

:start_service
set "name=%~1"
set "port=%~2"
set "svc_dir=%~3"
set "entrypoint=%~4"
set "abs_svc_dir=%ROOT_DIR%%svc_dir%"
set "venv_activate=%abs_svc_dir%\.venv\Scripts\Activate.ps1"
set "venv_python=%abs_svc_dir%\.venv\Scripts\python.exe"
set "entrypoint_path=%abs_svc_dir%\%entrypoint%"
set "pyproject_file=%abs_svc_dir%\pyproject.toml"
set "health_url=http://127.0.0.1:%port%/health"

powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri '%health_url%' -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }"
if %errorlevel%==0 (
    echo [RUNNING] %name% already healthy on port %port%.
    set /a running_count+=1
    exit /b 0
)

if not exist "%abs_svc_dir%" (
    echo [FAILED] %name% missing service folder: %abs_svc_dir%
    set /a failed_count+=1
    exit /b 1
)

if not exist "%venv_activate%" (
    echo [FAILED] %name% missing venv activate script: %venv_activate%
    echo          Create it first: cd /d "%abs_svc_dir%" ^&^& python -m venv .venv
    set /a failed_count+=1
    exit /b 1
)
if not exist "%venv_python%" (
    echo [FAILED] %name% missing venv python: %venv_python%
    set /a failed_count+=1
    exit /b 1
)

if not exist "%entrypoint_path%" (
    echo [FAILED] %name% missing entrypoint: %entrypoint_path%
    set /a failed_count+=1
    exit /b 1
)

set "mode=venv-python"

if exist "%pyproject_file%" (
    where poetry >nul 2>nul
    if errorlevel 1 (
        echo [FAILED] %name% has pyproject.toml but poetry is not in PATH.
        echo          Install poetry and ensure 'poetry' command is available.
        set /a failed_count+=1
        exit /b 1
    )
    set "mode=venv+poetry"
)

REM Dependency sanity check before launching a new window
"%venv_python%" -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo [FAILED] %name% venv missing required package 'fastapi'.
    echo          Fix: cd /d "%abs_svc_dir%" ^&^& poetry install
    set /a failed_count+=1
    exit /b 1
)
if /I "%name%"=="auth-service" (
    "%venv_python%" -c "import dotenv" >nul 2>&1
    if errorlevel 1 (
        echo [FAILED] %name% venv missing required package 'python-dotenv'.
        echo          Fix: cd /d "%abs_svc_dir%" ^&^& poetry install
        set /a failed_count+=1
        exit /b 1
    )
)

echo [STARTING] %name% on port %port% using %mode%...
if not exist "%ROOT_DIR%logs\%name%.log" type nul > "%ROOT_DIR%logs\%name%.log"
if not exist "%ROOT_DIR%logs\%name%.err.log" type nul > "%ROOT_DIR%logs\%name%.err.log"
if "%mode%"=="venv+poetry" (
    start "RAG Chat Room Web - %name%" powershell -NoExit -NoProfile -ExecutionPolicy Bypass -Command "Set-Location -Path '%abs_svc_dir%'; & '%venv_activate%'; & poetry run python '%entrypoint%' 1>>'%ROOT_DIR%logs\%name%.log' 2>>'%ROOT_DIR%logs\%name%.err.log'"
) else (
    start "RAG Chat Room Web - %name%" powershell -NoExit -NoProfile -ExecutionPolicy Bypass -Command "Set-Location -Path '%abs_svc_dir%'; & '%venv_activate%'; python '%entrypoint%' 1>>'%ROOT_DIR%logs\%name%.log' 2>>'%ROOT_DIR%logs\%name%.err.log'"
)

powershell -NoProfile -Command "$ok = $false; 1..10 | ForEach-Object { try { $r = Invoke-WebRequest -Uri '%health_url%' -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -eq 200) { $ok = $true; break } } catch {}; Start-Sleep -Seconds 1 }; if ($ok) { exit 0 } else { exit 1 }"
if %errorlevel%==0 (
    set /a started_count+=1
    exit /b 0
)

echo [FAILED] %name% did not become healthy on %health_url% within timeout.
echo          Check logs\%name%.log and logs\%name%.err.log
set /a failed_count+=1
exit /b 1
