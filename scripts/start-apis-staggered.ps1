# Staggered API Startup Script for Windows PowerShell
# This script starts API services with delays to prevent resource conflicts

Write-Host "🚀 Starting OSINT Stack APIs with Load Balancing..." -ForegroundColor Green

# Function to wait for service to be healthy
function Wait-ForHealthy {
    param(
        [string]$ServiceName,
        [int]$MaxAttempts = 60
    )
    
    Write-Host "⏳ Waiting for $ServiceName to be healthy..." -ForegroundColor Yellow
    
    $attempt = 1
    while ($attempt -le $MaxAttempts) {
        $status = docker-compose ps $ServiceName | Select-String "healthy"
        if ($status) {
            Write-Host "✅ $ServiceName is healthy!" -ForegroundColor Green
            return $true
        }
        
        Write-Host "   Attempt $attempt/$MaxAttempts - $ServiceName still starting..." -ForegroundColor Gray
        Start-Sleep -Seconds 10
        $attempt++
    }
    
    Write-Host "❌ $ServiceName failed to become healthy after $MaxAttempts attempts" -ForegroundColor Red
    return $false
}

# Function to start a service and wait for it to be healthy
function Start-AndWait {
    param(
        [string]$ServiceName,
        [int]$DelaySeconds
    )
    
    Write-Host "⏳ Starting $ServiceName in $DelaySeconds seconds..." -ForegroundColor Yellow
    Start-Sleep -Seconds $DelaySeconds
    
    Write-Host "🚀 Starting $ServiceName..." -ForegroundColor Green
    docker-compose up -d $ServiceName
    
    Wait-ForHealthy $ServiceName
}

# Start services in sequence with delays
Write-Host "📋 Starting API services with staggered delays..." -ForegroundColor Cyan

# Start main API first (no delay)
Write-Host "🚀 Starting main API (api)..." -ForegroundColor Green
docker-compose up -d api
Wait-ForHealthy api

# Start api-2 after 30 seconds
Start-AndWait api-2 30

# Start api-3 after 60 seconds
Start-AndWait api-3 60

Write-Host "🎉 All API services started successfully!" -ForegroundColor Green
Write-Host "📊 API Status:" -ForegroundColor Cyan
docker-compose ps | Select-String api

Write-Host ""
Write-Host "🔗 Available API endpoints:" -ForegroundColor Cyan
Write-Host "   Main API:     http://localhost:8000" -ForegroundColor White
Write-Host "   API-2:        http://localhost:8001" -ForegroundColor White
Write-Host "   API-3:        http://localhost:8002" -ForegroundColor White
Write-Host "   Load Balancer: http://localhost/api/" -ForegroundColor White

Write-Host ""
Write-Host "✅ Load balancing setup complete!" -ForegroundColor Green


