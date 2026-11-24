@echo off
REM Start the RAG Chat Room Web backend with Docker Compose inside WSL.
REM Keep this window open while using the frontend; closing it can stop WSL
REM and make localhost:8000/8001/8002/8005/8006 unreachable from Windows.

set "WSL_PROJECT_DIR=/mnt/d/Duccc/Coding/UI_UX_AGENT_PROJECT/uxaag"

echo.
echo ========================================
echo   RAG Chat Room Web Docker Backend
echo ========================================
echo.
echo Starting Docker Compose in WSL...
echo.

wsl -- bash -lc "cd '%WSL_PROJECT_DIR%' && docker compose up -d redis api-gateway chat-service auth-service rag-service document-service && docker compose ps && echo && echo 'Backend URLs:' && echo '  API Gateway:      http://localhost:8000' && echo '  Document Service: http://localhost:8001' && echo '  RAG Service:      http://localhost:8002' && echo '  Chat Service:     http://localhost:8005' && echo '  Auth Service:     http://localhost:8006' && echo && echo 'Keep this WSL window open while developing.' && exec bash"
