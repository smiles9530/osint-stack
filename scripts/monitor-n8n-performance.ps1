# N8N Performance Monitoring Script (PowerShell)
# This script monitors N8N performance and provides optimization recommendations

Write-Host "🔍 N8N Performance Monitor" -ForegroundColor Cyan
Write-Host "==========================" -ForegroundColor Cyan

# Check if N8N container is running
$n8nContainer = docker ps --filter "name=osint-n8n" --format "{{.Names}}"
if (-not $n8nContainer) {
    Write-Host "❌ N8N container is not running" -ForegroundColor Red
    exit 1
}

Write-Host "✅ N8N container is running" -ForegroundColor Green

# Get container stats
Write-Host ""
Write-Host "📊 Container Resource Usage:" -ForegroundColor Yellow
Write-Host "----------------------------" -ForegroundColor Yellow
docker stats osint-n8n --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}"

# Check N8N logs for performance issues
Write-Host ""
Write-Host "🔍 Recent Performance Issues:" -ForegroundColor Yellow
Write-Host "-----------------------------" -ForegroundColor Yellow
$logs = docker logs osint-n8n --tail=50
$performanceIssues = $logs | Select-String -Pattern "(error|warning|timeout|memory|slow|performance)" -CaseSensitive:$false
if ($performanceIssues) {
    $performanceIssues | Select-Object -Last 10 | ForEach-Object { Write-Host $_.Line }
} else {
    Write-Host "No performance issues detected in recent logs" -ForegroundColor Green
}

# Check Redis queue status (if using queue mode)
Write-Host ""
Write-Host "📈 Redis Queue Status:" -ForegroundColor Yellow
Write-Host "---------------------" -ForegroundColor Yellow
try {
    $redisInfo = docker exec osint-redis redis-cli info
    $redisInfo | Select-String -Pattern "(used_memory|connected_clients|keyspace)" | Select-Object -First 5 | ForEach-Object { Write-Host $_.Line }
} catch {
    Write-Host "Could not connect to Redis" -ForegroundColor Red
}

# Check N8N metrics endpoint
Write-Host ""
Write-Host "📊 N8N Metrics:" -ForegroundColor Yellow
Write-Host "---------------" -ForegroundColor Yellow
try {
    $metrics = Invoke-RestMethod -Uri "http://localhost:5678/rest/metrics" -Method Get -TimeoutSec 5
    $metrics | Select-Object -First 10 | ForEach-Object { Write-Host $_ }
} catch {
    Write-Host "Metrics endpoint not available or not accessible" -ForegroundColor Red
}

# Performance recommendations
Write-Host ""
Write-Host "💡 Performance Recommendations:" -ForegroundColor Magenta
Write-Host "===============================" -ForegroundColor Magenta
Write-Host "1. Enable queue mode for better scalability"
Write-Host "2. Use Redis for queue management"
Write-Host "3. Monitor memory usage and adjust limits"
Write-Host "4. Use connection pooling for database operations"
Write-Host "5. Enable workflow caching for repeated operations"
Write-Host "6. Consider using N8N runners for heavy workloads"

# Check current configuration
Write-Host ""
Write-Host "⚙️  Current Configuration:" -ForegroundColor Yellow
Write-Host "-------------------------" -ForegroundColor Yellow
$envVars = docker exec osint-n8n env | Select-String -Pattern "(N8N_|NODE_|PUPPETEER_)"
$envVars | Sort-Object | ForEach-Object { Write-Host $_.Line }

Write-Host ""
Write-Host "🎯 Performance Tips:" -ForegroundColor Cyan
Write-Host "===================" -ForegroundColor Cyan
Write-Host "• Use 'Wait' nodes instead of 'Sleep' for better resource management"
Write-Host "• Implement error handling to prevent workflow failures"
Write-Host "• Use 'Set' nodes to cache frequently used data"
Write-Host "• Consider splitting large workflows into smaller ones"
Write-Host "• Enable workflow versioning for better debugging"
Write-Host "• Use webhook triggers instead of polling when possible"

Write-Host ""
Write-Host "🚀 To apply performance optimizations, restart N8N:" -ForegroundColor Green
Write-Host "docker-compose restart n8n" -ForegroundColor White
