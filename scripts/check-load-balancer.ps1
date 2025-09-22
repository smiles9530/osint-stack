# Load Balancer Health Check Script (PowerShell)
# This script checks the health of all API instances and load balancer

Write-Host "🔍 Load Balancer Health Check" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan

# Check if all API instances are running
$apiInstances = @("osint-api", "osint-api-2", "osint-api-3")
$healthyInstances = @()
$unhealthyInstances = @()

Write-Host ""
Write-Host "📊 API Instance Status:" -ForegroundColor Yellow
Write-Host "----------------------" -ForegroundColor Yellow

foreach ($instance in $apiInstances) {
    $container = docker ps --filter "name=$instance" --format "{{.Names}}"
    if ($container) {
        # Check health endpoint
        try {
            $port = if ($instance -eq "osint-api") { "8000" } 
                   elseif ($instance -eq "osint-api-2") { "8001" } 
                   else { "8002" }
            
            $response = Invoke-RestMethod -Uri "http://localhost:$port/healthz" -Method Get -TimeoutSec 5
            Write-Host "✅ $instance (port $port): Healthy" -ForegroundColor Green
            $healthyInstances += $instance
        } catch {
            Write-Host "❌ $instance (port $port): Unhealthy - $($_.Exception.Message)" -ForegroundColor Red
            $unhealthyInstances += $instance
        }
    } else {
        Write-Host "❌ ${instance}: Not running" -ForegroundColor Red
        $unhealthyInstances += $instance
    }
}

# Check Nginx load balancer
Write-Host ""
Write-Host "🔄 Load Balancer Status:" -ForegroundColor Yellow
Write-Host "-----------------------" -ForegroundColor Yellow

try {
    $lbResponse = Invoke-RestMethod -Uri "http://localhost/api/healthz" -Method Get -TimeoutSec 5
    Write-Host "✅ Nginx Load Balancer: Healthy" -ForegroundColor Green
} catch {
    Write-Host "❌ Nginx Load Balancer: Unhealthy - $($_.Exception.Message)" -ForegroundColor Red
}

# Check Redis queue status
Write-Host ""
Write-Host "📈 Redis Queue Status:" -ForegroundColor Yellow
Write-Host "---------------------" -ForegroundColor Yellow

try {
    $redisInfo = docker exec osint-redis redis-cli info
    $connectedClients = ($redisInfo | Select-String "connected_clients").Line
    $usedMemory = ($redisInfo | Select-String "used_memory_human").Line
    $keyspaceHits = ($redisInfo | Select-String "keyspace_hits").Line
    $keyspaceMisses = ($redisInfo | Select-String "keyspace_misses").Line
    
    Write-Host "✅ Redis: Connected" -ForegroundColor Green
    Write-Host "   $connectedClients" -ForegroundColor White
    Write-Host "   $usedMemory" -ForegroundColor White
    Write-Host "   $keyspaceHits" -ForegroundColor White
    Write-Host "   $keyspaceMisses" -ForegroundColor White
} catch {
    Write-Host "❌ Redis: Connection failed - $($_.Exception.Message)" -ForegroundColor Red
}

# Check N8N queue mode
Write-Host ""
Write-Host "🤖 N8N Queue Mode Status:" -ForegroundColor Yellow
Write-Host "-------------------------" -ForegroundColor Yellow

try {
    $n8nLogs = docker logs osint-n8n --tail=10
    if ($n8nLogs -match "queue|worker|runner") {
        Write-Host "✅ N8N Queue Mode: Active" -ForegroundColor Green
    } else {
        Write-Host "⚠️  N8N Queue Mode: Status unclear" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ N8N: Cannot check status" -ForegroundColor Red
}

# Performance summary
Write-Host ""
Write-Host "📊 Performance Summary:" -ForegroundColor Magenta
Write-Host "======================" -ForegroundColor Magenta
Write-Host "Healthy API Instances: $($healthyInstances.Count)/$($apiInstances.Count)" -ForegroundColor White
Write-Host "Unhealthy Instances: $($unhealthyInstances.Count)" -ForegroundColor White

if ($healthyInstances.Count -gt 1) {
    Write-Host "✅ Load Balancing: Active ($($healthyInstances.Count) instances)" -ForegroundColor Green
} else {
    Write-Host "⚠️  Load Balancing: Limited (only $($healthyInstances.Count) instance)" -ForegroundColor Yellow
}

# Recommendations
Write-Host ""
Write-Host "💡 Recommendations:" -ForegroundColor Cyan
Write-Host "===================" -ForegroundColor Cyan

if ($unhealthyInstances.Count -gt 0) {
    Write-Host "• Fix unhealthy instances:" -ForegroundColor Yellow
    foreach ($instance in $unhealthyInstances) {
        Write-Host "  - docker-compose restart $instance" -ForegroundColor White
    }
}

if ($healthyInstances.Count -eq 3) {
    Write-Host "• All instances healthy - optimal load balancing active" -ForegroundColor Green
    Write-Host "• Consider monitoring resource usage" -ForegroundColor White
    Write-Host "• Test failover scenarios" -ForegroundColor White
}

Write-Host ""
Write-Host "🔧 Load Testing Commands:" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan
Write-Host "Test load balancer: Invoke-WebRequest -Uri 'http://localhost/api/healthz' -Method Get" -ForegroundColor White
Write-Host "Test specific instance: Invoke-WebRequest -Uri 'http://localhost:8000/healthz' -Method Get" -ForegroundColor White
Write-Host "Monitor logs: docker-compose logs -f api" -ForegroundColor White
