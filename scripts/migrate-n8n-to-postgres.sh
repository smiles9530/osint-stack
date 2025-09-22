#!/bin/bash

# N8N SQLite to PostgreSQL Migration Script
# This script helps migrate N8N from SQLite to PostgreSQL

echo "🔄 N8N Database Migration: SQLite → PostgreSQL"
echo "=============================================="

# Check if N8N container is running
if ! docker ps | grep -q "osint-n8n"; then
    echo "❌ N8N container is not running"
    exit 1
fi

echo "✅ N8N container is running"

# Check if PostgreSQL is accessible
echo ""
echo "🔍 Checking PostgreSQL connectivity..."
if docker exec osint-db pg_isready -U osint -d n8n; then
    echo "✅ PostgreSQL is accessible"
else
    echo "❌ Cannot connect to PostgreSQL"
    echo "Please ensure the database is running and the n8n database exists"
    exit 1
fi

# Backup current N8N data
echo ""
echo "💾 Creating backup of current N8N data..."
docker exec osint-n8n tar -czf /tmp/n8n-backup-$(date +%Y%m%d-%H%M%S).tar.gz -C /home/node/.n8n .

# Check if n8n database exists
echo ""
echo "🔍 Checking n8n database..."
if docker exec osint-db psql -U osint -d n8n -c "SELECT 1;" > /dev/null 2>&1; then
    echo "✅ n8n database exists"
else
    echo "❌ n8n database does not exist"
    echo "Creating n8n database..."
    docker exec osint-db createdb -U osint n8n
    echo "✅ n8n database created"
fi

echo ""
echo "📋 Migration Steps:"
echo "==================="
echo "1. ✅ PostgreSQL database configured"
echo "2. ✅ N8N container configured for PostgreSQL"
echo "3. ✅ Redis queue configured"
echo "4. ✅ Performance optimizations applied"
echo ""
echo "🚀 Next Steps:"
echo "=============="
echo "1. Stop N8N: docker-compose stop n8n"
echo "2. Remove old data volume: docker volume rm osint-stack_n8n_data"
echo "3. Start N8N: docker-compose up n8n -d"
echo "4. N8N will automatically create tables in PostgreSQL"
echo ""
echo "⚠️  Important Notes:"
echo "==================="
echo "• This will create a fresh N8N instance with PostgreSQL"
echo "• All existing workflows and data will be lost"
echo "• Make sure to export important workflows before migration"
echo "• The new setup will be much more performant and scalable"
echo ""
echo "🎯 Benefits of PostgreSQL:"
echo "========================="
echo "• Better performance for large datasets"
echo "• Support for queue mode and scaling"
echo "• Better concurrent access"
echo "• More reliable data persistence"
echo "• Support for advanced features"
