#!/bin/bash

# N8N Performance Monitoring Script
# This script monitors N8N performance and provides optimization recommendations

echo "🔍 N8N Performance Monitor"
echo "=========================="

# Check if N8N container is running
if ! docker ps | grep -q "osint-n8n"; then
    echo "❌ N8N container is not running"
    exit 1
fi

echo "✅ N8N container is running"

# Get container stats
echo ""
echo "📊 Container Resource Usage:"
echo "----------------------------"
docker stats osint-n8n --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}"

# Check N8N logs for performance issues
echo ""
echo "🔍 Recent Performance Issues:"
echo "-----------------------------"
docker logs osint-n8n --tail=50 | grep -i -E "(error|warning|timeout|memory|slow|performance)" | tail -10

# Check Redis queue status (if using queue mode)
echo ""
echo "📈 Redis Queue Status:"
echo "---------------------"
docker exec osint-redis redis-cli info | grep -E "(used_memory|connected_clients|keyspace)" | head -5

# Check N8N metrics endpoint
echo ""
echo "📊 N8N Metrics:"
echo "---------------"
curl -s http://localhost:5678/rest/metrics 2>/dev/null | head -10 || echo "Metrics endpoint not available"

# Performance recommendations
echo ""
echo "💡 Performance Recommendations:"
echo "==============================="
echo "1. Enable queue mode for better scalability"
echo "2. Use Redis for queue management"
echo "3. Monitor memory usage and adjust limits"
echo "4. Use connection pooling for database operations"
echo "5. Enable workflow caching for repeated operations"
echo "6. Consider using N8N runners for heavy workloads"

# Check current configuration
echo ""
echo "⚙️  Current Configuration:"
echo "-------------------------"
docker exec osint-n8n env | grep -E "(N8N_|NODE_|PUPPETEER_)" | sort

echo ""
echo "🎯 Performance Tips:"
echo "==================="
echo "• Use 'Wait' nodes instead of 'Sleep' for better resource management"
echo "• Implement error handling to prevent workflow failures"
echo "• Use 'Set' nodes to cache frequently used data"
echo "• Consider splitting large workflows into smaller ones"
echo "• Enable workflow versioning for better debugging"
echo "• Use webhook triggers instead of polling when possible"
