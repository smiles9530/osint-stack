@echo off
setlocal EnableDelayedExpansion

echo.
echo ========================================================
echo 🚀 OSINT Stack Deployment with Ollama Model Initialization
echo ========================================================
echo.

REM Check if .env file exists
if not exist ".env" (
    echo ❌ .env file not found. Please create one from .env.example
    exit /b 1
)

echo 📋 Deployment Steps:
echo   1. Build and start core services
echo   2. Initialize Ollama models
echo   3. Verify all services are healthy
echo.

REM Step 1: Start core services
echo Step 1: Starting core services...
docker compose up -d --build

REM Wait a bit for services to stabilize
echo Waiting for services to stabilize...
timeout /t 30 /nobreak > nul

REM Step 2: Initialize Ollama models
echo Step 2: Initializing Ollama models...
docker compose --profile init up ollama-init

if %errorlevel% == 0 (
    echo ✅ Ollama model initialization completed successfully
) else (
    echo ❌ Ollama model initialization failed
    echo You can retry with: docker compose --profile init up ollama-init
)

REM Step 3: Verify services
echo Step 3: Verifying service health...

REM Check service status
echo Service Status:
docker compose ps

echo.
echo Health Checks:

REM Check each service (simplified for Windows)
set services=db:5432 qdrant:6333 meilisearch:7700 ollama:11434 redis:6379 api:8000

for %%s in (%services%) do (
    for /f "tokens=1,2 delims=:" %%a in ("%%s") do (
        curl -s -f "http://localhost:%%b" >nul 2>&1
        if !errorlevel! == 0 (
            echo   ✅ %%a: Healthy
        ) else (
            curl -s -f "http://localhost:%%b/health" >nul 2>&1
            if !errorlevel! == 0 (
                echo   ✅ %%a: Healthy
            ) else (
                curl -s -f "http://localhost:%%b/healthz" >nul 2>&1
                if !errorlevel! == 0 (
                    echo   ✅ %%a: Healthy
                ) else (
                    echo   ⚠️  %%a: Not responding ^(may still be starting^)
                )
            )
        )
    )
)

echo.
echo 🎉 Deployment completed!
echo ========================================================
echo Next steps:
echo   1. Access the frontend: http://localhost:3000
echo   2. Access the API: http://localhost:8000
echo   3. Access N8N: http://localhost:5678
echo   4. Login with your configured credentials
echo.
echo To check Ollama models:
echo   curl http://localhost:11434/api/tags
echo.

pause
