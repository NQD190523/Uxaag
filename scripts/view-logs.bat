@echo off
REM Stream the logs of all RAG Chat Room Web services in one console.
REM   view-logs.bat          -> logs\<service>\<service>.log       (INFO and above)
REM   view-logs.bat errors   -> logs\<service>\<service>.error.log (WARNING and above)
REM Files are written and rotated by each service (shared.logger); this only reads them.

setlocal

REM This script lives in uxaag\scripts\; run from the uxaag\ folder so .\logs resolves.
cd /d "%~dp0.."

set "PATTERN=*.log"
if /i "%~1"=="errors" set "PATTERN=*.error.log"

if not exist "logs" (
    echo No logs folder yet. Start the services first: scripts\start-services.bat
    exit /b 0
)

echo.
echo Streaming %PATTERN% from logs\^<service^>\ ... Press Ctrl+C to stop.
echo.

REM Only current files (not rotated .log.1 ... .log.5); main logs exclude *.error.log.
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$files = Get-ChildItem -Path '.\logs' -Recurse -Filter '%PATTERN%' -File | Where-Object { '%PATTERN%' -eq '*.error.log' -or $_.Name -notlike '*.error.log' } | Sort-Object FullName; " ^
  "if (-not $files) { Write-Host 'No log files found under .\logs'; exit 0 }; " ^
  "Write-Host ('Watching: ' + (($files | ForEach-Object { $_.Name }) -join ', ')); " ^
  "Get-Content -Path ($files | ForEach-Object { $_.FullName }) -Tail 30 -Wait"
