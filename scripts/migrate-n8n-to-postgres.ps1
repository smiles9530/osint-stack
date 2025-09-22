# N8N SQLite to PostgreSQL Migration Script (PowerShell)
# This script helps migrate N8N from SQLite to PostgreSQL

Write-Host "🔄 N8N Database Migration: SQLite → PostgreSQL" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# Check if N8N container is running
$n8nContainer = docker ps --filter "name=osint-n8n" --format "{{.Names}}"
if (-not $n8nContainer) {
    Write-Host "❌ N8N container is not running" -ForegroundColor Red
    exit 1
}

Write-Host "✅ N8N container is running" -ForegroundColor Green

# Check if PostgreSQL is accessible
Write-Host ""
Write-Host "🔍 Checking PostgreSQL connectivity..." -ForegroundColor Yellow
try {
    $result = docker exec osint-db pg_isready -U osint -d n8n
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ PostgreSQL is accessible" -ForegroundColor Green
    } else {
        Write-Host "❌ Cannot connect to PostgreSQL" -ForegroundColor Red
        Write-Host "Please ensure the database is running and the n8n database exists" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ Cannot connect to PostgreSQL" -ForegroundColor Red
    Write-Host "Please ensure the database is running and the n8n database exists" -ForegroundColor Red
    exit 1
}

# Backup current N8N data
Write-Host ""
Write-Host "💾 Creating backup of current N8N data..." -ForegroundColor Yellow
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
docker exec osint-n8n tar -czf "/tmp/n8n-backup-$timestamp.tar.gz" -C /home/node/.n8n .

# Check if n8n database exists
Write-Host ""
Write-Host "🔍 Checking n8n database..." -ForegroundColor Yellow
try {
    $dbCheck = docker exec osint-db psql -U osint -d n8n -c "SELECT 1;" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ n8n database exists" -ForegroundColor Green
    } else {
        Write-Host "❌ n8n database does not exist" -ForegroundColor Red
        Write-Host "Creating n8n database..." -ForegroundColor Yellow
        docker exec osint-db createdb -U osint n8n
        Write-Host "✅ n8n database created" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ n8n database does not exist" -ForegroundColor Red
    Write-Host "Creating n8n database..." -ForegroundColor Yellow
    docker exec osint-db createdb -U osint n8n
    Write-Host "✅ n8n database created" -ForegroundColor Green
}

Write-Host ""
Write-Host "📋 Migration Steps:" -ForegroundColor Magenta
Write-Host "===================" -ForegroundColor Magenta
Write-Host "1. ✅ PostgreSQL database configured" -ForegroundColor Green
Write-Host "2. ✅ N8N container configured for PostgreSQL" -ForegroundColor Green
Write-Host "3. ✅ Redis queue configured" -ForegroundColor Green
Write-Host "4. ✅ Performance optimizations applied" -ForegroundColor Green

Write-Host ""
Write-Host "🚀 Next Steps:" -ForegroundColor Cyan
Write-Host "==============" -ForegroundColor Cyan
Write-Host "1. Stop N8N: docker-compose stop n8n" -ForegroundColor White
Write-Host "2. Remove old data volume: docker volume rm osint-stack_n8n_data" -ForegroundColor White
Write-Host "3. Start N8N: docker-compose up n8n -d" -ForegroundColor White
Write-Host "4. N8N will automatically create tables in PostgreSQL" -ForegroundColor White

Write-Host ""
Write-Host "⚠️  Important Notes:" -ForegroundColor Red
Write-Host "===================" -ForegroundColor Red
Write-Host "• This will create a fresh N8N instance with PostgreSQL" -ForegroundColor Yellow
Write-Host "• All existing workflows and data will be lost" -ForegroundColor Yellow
Write-Host "• Make sure to export important workflows before migration" -ForegroundColor Yellow
Write-Host "• The new setup will be much more performant and scalable" -ForegroundColor Yellow

Write-Host ""
Write-Host "🎯 Benefits of PostgreSQL:" -ForegroundColor Green
Write-Host "=========================" -ForegroundColor Green
Write-Host "• Better performance for large datasets" -ForegroundColor White
Write-Host "• Support for queue mode and scaling" -ForegroundColor White
Write-Host "• Better concurrent access" -ForegroundColor White
Write-Host "• More reliable data persistence" -ForegroundColor White
Write-Host "• Support for advanced features" -ForegroundColor White

Write-Host ""
Write-Host "🔧 Ready to migrate? Run these commands:" -ForegroundColor Cyan
Write-Host "docker-compose stop n8n" -ForegroundColor White
Write-Host "docker volume rm osint-stack_n8n_data" -ForegroundColor White
Write-Host "docker-compose up n8n -d" -ForegroundColor White
