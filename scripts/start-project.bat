@echo off
REM ==========================================================================
REM start-project.bat - local dev launcher for RAG Chat Room Web (Windows).
REM
REM Starts the core backend + frontend, each in its own window:
REM   auth-service :8006, chat-service :8005, document-service :8001,
REM   api-gateway :8000, Next.js UI :3000
REM Deliberately NOT started: rag-service (:8002) and Redis (:6379).
REM
REM Usage (from anywhere):
REM   uxaag\scripts\start-project.bat          start everything
REM   uxaag\scripts\start-project.bat setup    one-time: create service venvs
REM                                            (poetry install) + npm install
REM   uxaag\scripts\start-project.bat status   health of every service + UI
REM   uxaag\scripts\start-project.bat stop     close the windows this script opened
REM   Add "nopause" to any mode for unattended use (Claude, CI): no prompts,
REM   exit code 0 = success, 1 = something failed (e.g. "... start-project.bat nopause").
REM
REM Before starting it runs scripts\check_db.py, which tests the Supabase
REM DATABASE_URL step by step (DNS, TCP, login, tables) and explains failures.
REM It never prints the URL or password.
REM
REM Close a service window to stop that service. Logs: uxaag\logs\<service>\.
REM ==========================================================================

setlocal EnableExtensions EnableDelayedExpansion

REM This file lives in uxaag\scripts\; ROOT_DIR = uxaag\, UI_DIR = sibling uxaag_UI\.
for %%I in ("%~dp0..") do set "ROOT_DIR=%%~fI"
for %%I in ("%ROOT_DIR%\..\uxaag_UI") do set "UI_DIR=%%~fI"

REM Core services started by this script: name:port:folder (start order matters:
REM the gateway goes last so its upstreams are already up).
set "SERVICES=auth-service:8006:auth_service chat-service:8005:chat_service document-service:8001:document_service api-gateway:8000:api_gateway"

REM Args: one mode (start|setup|status|stop, default start) plus optional "nopause".
set "MODE=start"
set "NOPAUSE="
for %%A in (%*) do (
    if /i "%%~A"=="nopause" (set "NOPAUSE=1") else set "MODE=%%~A"
)

echo.
echo ==========================================
echo   RAG Chat Room Web - local %MODE%
echo ==========================================
echo   Backend : %ROOT_DIR%
echo   Frontend: %UI_DIR%
echo.

if /i "%MODE%"=="setup" goto :setup
if /i "%MODE%"=="status" goto :status
if /i "%MODE%"=="stop" goto :stop
if /i not "%MODE%"=="start" (
    echo [FAILED] Unknown mode "%MODE%". Use: start, setup, status, stop [nopause]
    goto :end_fail
)

REM ---------------------------------------------------------------- 1. tools
where python >nul 2>nul
if errorlevel 1 (
    echo [FAILED] python is not on PATH. Install Python 3.11+ and reopen this window.
    goto :end_fail
)

REM ------------------------------------------------- 2. venvs for each service
set /a missing=0
for %%S in (%SERVICES%) do (
    for /f "tokens=1-3 delims=:" %%a in ("%%S") do (
        if not exist "%ROOT_DIR%\services\%%c\.venv\Scripts\python.exe" (
            echo [MISSING] %%a has no venv: services\%%c\.venv
            set /a missing+=1
        )
    )
)
if !missing! gtr 0 (
    echo.
    echo One-time setup needed. Run:
    echo     "%~f0" setup
    goto :end_fail
)

REM ------------------------------------------------------- 3. database check
echo --- Database check (services\chat_service\.env) ---
"%ROOT_DIR%\services\chat_service\.venv\Scripts\python.exe" "%ROOT_DIR%\scripts\check_db.py"
set "db_rc=!errorlevel!"
echo.
if "!db_rc!"=="1" (
    echo [WARNING] Cannot connect to the database - see the reason above.
    echo           auth-service and chat-service will start but every login,
    echo           room or message request will fail until this is fixed.
    REM Unattended runs stop here: starting against a dead DB only hides the cause.
    if defined NOPAUSE goto :end_fail
    choice /c YN /n /m "Start the services anyway? [Y/N] "
    if errorlevel 2 goto :end_fail
    echo.
)

REM ------------------------------------------ 4. local routing + port ownership
REM services.yaml uses docker-compose hostnames (auth-service, chat-service...)
REM that do not resolve outside Docker. The gateway reads GATEWAY_UPSTREAM_HOST
REM (core/config.py) and swaps that hostname for 127.0.0.1. Windows `start`
REM passes this variable on to the gateway window.
set "GATEWAY_UPSTREAM_HOST=127.0.0.1"

REM Refuse to start when a port is held by the wrong kind of process. Why: on
REM Windows a second program can bind 0.0.0.0/[::] on a port where a service
REM already holds 127.0.0.1, and Node then reaches the wrong one via ::1 (a
REM Next.js dev server on :8000 silently swallowed every gateway call).
set /a port_conflicts=0
call :check_port 8006 python
call :check_port 8005 python
call :check_port 8001 python
call :check_port 8000 python
call :check_port 3000 node
if !port_conflicts! gtr 0 (
    echo.
    echo Close the process^(es^) above ^(or their window^), then run this script again.
    goto :end_fail
)

REM ---------------------------------------------------------- 5. start services
set /a started=0
set /a running=0
set /a failed=0
for %%S in (%SERVICES%) do (
    for /f "tokens=1-3 delims=:" %%a in ("%%S") do call :start_service "%%a" "%%b" "%%c"
)
call :start_frontend

REM ------------------------------------------------------------------ summary
echo.
echo ==========================================
echo   Started: !started!   Already running: !running!   Failed: !failed!
echo ==========================================
echo   Frontend         http://localhost:3000
echo   API Gateway      http://localhost:8000
echo   Document Service http://localhost:8001
echo   Chat Service     http://localhost:8005
echo   Auth Service     http://localhost:8006
echo.
echo   NOT started on purpose:
echo     - rag-service :8002  -^> "@model" AI replies in chat will return llm_error
echo     - Redis       :6379  -^> live room-list updates off; chat-service logs a
echo                            "Redis pub/sub ... reconnecting" warning every 2s
if defined GATEWAY_WAS_RUNNING (
    echo.
    echo   [NOTE] api-gateway was already running, so it was not restarted. If it
    echo          was started outside this script it may lack GATEWAY_UPSTREAM_HOST
    echo          and return 502. Close its window and run this script again.
)
echo.
echo   Logs: %ROOT_DIR%\logs\^<service^>\   (stream: scripts\view-logs.bat)
echo   Close a service window to stop it, or run: "%~f0" stop
echo.
if not defined NOPAUSE pause
if !failed! gtr 0 exit /b 1
exit /b 0

REM ==========================================================================
REM :status - health of every service and the UI; exit 0 only if all are up.
REM ==========================================================================
:status
set /a down=0
for %%S in (%SERVICES%) do (
    for /f "tokens=1-3 delims=:" %%a in ("%%S") do (
        call :wait_http "http://127.0.0.1:%%b/health" 1
        if errorlevel 1 (
            echo [ DOWN ] %%a :%%b
            set /a down+=1
        ) else (
            echo [  UP  ] %%a :%%b
        )
    )
)
call :wait_port 3000 1
if errorlevel 1 (
    echo [ DOWN ] frontend :3000
    set /a down+=1
) else (
    echo [  UP  ] frontend :3000  http://localhost:3000
)
echo.
if !down! gtr 0 (
    echo !down! component^(s^) down.
    goto :end_fail
)
echo All components up.
if not defined NOPAUSE pause
exit /b 0

REM ==========================================================================
REM :stop - stops only what this script started, via stop-project.ps1: it finds
REM   each port's listener and ends the launcher's own cmd /k tree (closing its
REM   window). Window-title matching is not used: on Windows 11 the consoles
REM   live in Windows Terminal, so cmd.exe has no title to match.
REM   Exit 0 = all app ports free.
REM ==========================================================================
:stop
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop-project.ps1"
set "stop_rc=!errorlevel!"
if not defined NOPAUSE pause
exit /b !stop_rc!

REM ==========================================================================
REM :start_service name port folder
REM   Skips the service if /health already answers 200; otherwise opens a new
REM   window running the service venv python and waits up to 30s for /health.
REM ==========================================================================
:start_service
set "name=%~1"
set "port=%~2"
set "svc_dir=%ROOT_DIR%\services\%~3"
set "health_url=http://127.0.0.1:%port%/health"

call :wait_http "%health_url%" 1
if not errorlevel 1 (
    echo [RUNNING ] %name% already healthy on :%port%
    if /i "%name%"=="api-gateway" set "GATEWAY_WAS_RUNNING=1"
    set /a running+=1
    exit /b 0
)

if not exist "%svc_dir%\src\main.py" (
    echo [FAILED  ] %name%: missing %svc_dir%\src\main.py
    set /a failed+=1
    exit /b 1
)

echo [STARTING] %name% on :%port% ...
if not exist "%ROOT_DIR%\logs\%name%" mkdir "%ROOT_DIR%\logs\%name%"
REM cmd /k keeps the window open after a crash so the traceback stays visible.
REM Working dir = service folder so each service loads its own .env.
start "RCRW - %name% :%port%" /D "%svc_dir%" cmd /k ".venv\Scripts\python.exe src\main.py"

call :wait_http "%health_url%" 30
if errorlevel 1 (
    echo [FAILED  ] %name% not healthy on %health_url% after 30s.
    echo            Look at its window, or logs\%name%\%name%.error.log
    set /a failed+=1
    exit /b 1
)
echo [   OK   ] %name% healthy
set /a started+=1
exit /b 0

REM ==========================================================================
REM :start_frontend
REM   Starts `npm run dev` in uxaag_UI on a fixed port 3000 and waits up to 90s
REM   for the port to accept. Port is forced (PORT + -p) because Next.js honours
REM   an inherited PORT variable, which once moved it onto the gateway's :8000.
REM   BACKEND_URL uses 127.0.0.1, not localhost: Node resolves localhost to ::1
REM   first, while the gateway listens on IPv4 127.0.0.1 only.
REM ==========================================================================
:start_frontend
call :wait_port 3000 1
if not errorlevel 1 (
    echo [RUNNING ] frontend already listening on :3000
    set /a running+=1
    exit /b 0
)
where npm >nul 2>nul
if errorlevel 1 (
    echo [FAILED  ] frontend: npm is not on PATH ^(install Node.js^).
    set /a failed+=1
    exit /b 1
)
if not exist "%UI_DIR%\node_modules" (
    echo [FAILED  ] frontend: no node_modules. Run: "%~f0" setup
    set /a failed+=1
    exit /b 1
)
echo [STARTING] frontend on :3000 ...
start "RCRW - frontend :3000" /D "%UI_DIR%" cmd /k "set PORT=3000&& set BACKEND_URL=http://127.0.0.1:8000&& set NEXT_PUBLIC_BACKEND_WS_URL=ws://localhost:8005&& npm run dev -- -p 3000"
REM First `next dev` compile can be slow; 90s budget.
call :wait_port 3000 90
if errorlevel 1 (
    echo [FAILED  ] frontend did not open :3000 within 90s - look at its window.
    set /a failed+=1
    exit /b 1
)
echo [   OK   ] frontend listening
set /a started+=1
exit /b 0

REM :check_port port expected_process_prefix
REM   Lists every process LISTENING on the port. Any process whose name does not
REM   start with the expected prefix (python / node) counts as a conflict and is
REM   printed with its PID; increments port_conflicts.
:check_port
powershell -NoProfile -Command "$bad=@(Get-NetTCPConnection -State Listen -LocalPort %~1 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue } | Where-Object { $_.ProcessName -notlike '%~2*' }); foreach ($p in $bad) { Write-Host ('[CONFLICT] port %~1 is held by ' + $p.ProcessName + ' (PID ' + $p.Id + '), expected %~2') }; if ($bad.Count) { exit 1 } else { exit 0 }"
if errorlevel 1 set /a port_conflicts+=1
exit /b 0

REM :wait_http url seconds -> errorlevel 0 when the URL returns HTTP 200.
:wait_http
powershell -NoProfile -Command "$end=(Get-Date).AddSeconds(%~2); do { try { if ((Invoke-WebRequest -Uri '%~1' -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) { exit 0 } } catch {}; Start-Sleep -Milliseconds 700 } while ((Get-Date) -lt $end); exit 1"
exit /b %errorlevel%

REM :wait_port port seconds -> errorlevel 0 when 127.0.0.1:port accepts TCP.
:wait_port
powershell -NoProfile -Command "$end=(Get-Date).AddSeconds(%~2); do { try { $c=New-Object Net.Sockets.TcpClient; $c.Connect('127.0.0.1',%~1); $c.Close(); exit 0 } catch {}; Start-Sleep -Milliseconds 700 } while ((Get-Date) -lt $end); exit 1"
exit /b %errorlevel%

REM ==========================================================================
REM :setup - one-time install. Creates services\<svc>\.venv via Poetry (from
REM each poetry.lock, no upgrades) and runs npm install for the UI if needed.
REM ==========================================================================
:setup
where poetry >nul 2>nul
if errorlevel 1 (
    echo [FAILED] Poetry is not installed. Install it once, then rerun setup:
    echo     python -m pip install --user pipx
    echo     python -m pipx install poetry
    echo     python -m pipx ensurepath   ^(then open a new terminal^)
    goto :end_fail
)
REM Put each venv inside its service folder (services\<svc>\.venv), which is
REM where this launcher and start-services.bat expect it.
set "POETRY_VIRTUALENVS_IN_PROJECT=true"
set /a setup_failed=0
for %%S in (%SERVICES%) do (
    for /f "tokens=1-3 delims=:" %%a in ("%%S") do (
        echo.
        echo --- %%a: poetry install ---
        pushd "%ROOT_DIR%\services\%%c"
        call poetry install
        if errorlevel 1 (
            echo [FAILED] %%a: poetry install failed
            set /a setup_failed+=1
        )
        popd
    )
)
if not exist "%UI_DIR%\node_modules" (
    echo.
    echo --- frontend: npm install ---
    pushd "%UI_DIR%"
    call npm install
    if errorlevel 1 set /a setup_failed+=1
    popd
)
echo.
if !setup_failed! gtr 0 (
    echo Setup finished with !setup_failed! failure^(s^). Fix them, then rerun setup.
    goto :end_fail
)
echo Setup done. Now run: "%~f0"
if not defined NOPAUSE pause
exit /b 0

:end_fail
echo.
if not defined NOPAUSE pause
exit /b 1
