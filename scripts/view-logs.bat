@echo off
REM Stream logs from all RAG Chat Room Web services in one console

setlocal

REM This script lives in uxaag\scripts\; run from the uxaag\ folder so .\logs resolves.
cd /d "%~dp0.."

if not exist "logs" mkdir logs

if not exist "logs\auth-service.log" type nul > "logs\auth-service.log"
if not exist "logs\auth-service.err.log" type nul > "logs\auth-service.err.log"
if not exist "logs\chat-service.log" type nul > "logs\chat-service.log"
if not exist "logs\chat-service.err.log" type nul > "logs\chat-service.err.log"
if not exist "logs\document-service.log" type nul > "logs\document-service.log"
if not exist "logs\document-service.err.log" type nul > "logs\document-service.err.log"
if not exist "logs\rag-service.log" type nul > "logs\rag-service.log"
if not exist "logs\rag-service.err.log" type nul > "logs\rag-service.err.log"
if not exist "logs\vector-store-service.log" type nul > "logs\vector-store-service.log"
if not exist "logs\vector-store-service.err.log" type nul > "logs\vector-store-service.err.log"
if not exist "logs\web-crawler-service.log" type nul > "logs\web-crawler-service.log"
if not exist "logs\web-crawler-service.err.log" type nul > "logs\web-crawler-service.err.log"
if not exist "logs\api-gateway.log" type nul > "logs\api-gateway.log"
if not exist "logs\api-gateway.err.log" type nul > "logs\api-gateway.err.log"

echo.
echo Streaming all service logs (stdout + stderr)...
echo Press Ctrl+C to stop log streaming.
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$files = Get-ChildItem -Path '.\logs' -Filter '*.log' -File | Sort-Object Name; " ^
  "if (-not $files) { Write-Host 'No log files found in .\logs'; exit 0 }; " ^
  "Write-Host ('Watching files: ' + (($files | ForEach-Object { $_.Name }) -join ', ')); " ^
  "Get-Content -Path ($files | ForEach-Object { $_.FullName }) -Tail 30 -Wait"
