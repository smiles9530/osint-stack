#!/bin/bash

# Staggered API Startup Script
# This script starts API services with delays to prevent resource conflicts

set -e

echo "🚀 Starting OSINT Stack APIs with Load Balancing..."

# Function to wait for service to be healthy
wait_for_healthy() {
    local service_name=$1
    local max_attempts=60
    local attempt=1
    
    echo "⏳ Waiting for $service_name to be healthy..."
    
    while [ $attempt -le $max_attempts ]; do
        if docker-compose ps $service_name | grep -q "healthy"; then
            echo "✅ $service_name is healthy!"
            return 0
        fi
        
        echo "   Attempt $attempt/$max_attempts - $service_name still starting..."
        sleep 10
        ((attempt++))
    done
    
    echo "❌ $service_name failed to become healthy after $max_attempts attempts"
    return 1
}

# Function to start a service and wait for it to be healthy
start_and_wait() {
    local service_name=$1
    local delay=$2
    
    echo "⏳ Starting $service_name in $delay seconds..."
    sleep $delay
    
    echo "🚀 Starting $service_name..."
    docker-compose up -d $service_name
    
    wait_for_healthy $service_name
}

# Start services in sequence with delays
echo "📋 Starting API services with staggered delays..."

# Start main API first (no delay)
echo "🚀 Starting main API (api)..."
docker-compose up -d api
wait_for_healthy api

# Start api-2 after 30 seconds
start_and_wait api-2 30

# Start api-3 after 60 seconds
start_and_wait api-3 60

echo "🎉 All API services started successfully!"
echo "📊 API Status:"
docker-compose ps | grep api

echo ""
echo "🔗 Available API endpoints:"
echo "   Main API:     http://localhost:8000"
echo "   API-2:        http://localhost:8001" 
echo "   API-3:        http://localhost:8002"
echo "   Load Balancer: http://localhost/api/"

echo ""
echo "✅ Load balancing setup complete!"


