#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 OSINT Stack Deployment with Ollama Model Initialization${NC}"
echo -e "${BLUE}=======================================================${NC}"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${RED}❌ .env file not found. Please create one from .env.example${NC}"
    exit 1
fi

echo -e "${YELLOW}📋 Deployment Steps:${NC}"
echo -e "  1. Build and start core services"
echo -e "  2. Initialize Ollama models"
echo -e "  3. Verify all services are healthy"
echo ""

# Step 1: Start core services
echo -e "${BLUE}Step 1: Starting core services...${NC}"
docker compose up -d --build

# Wait a bit for services to stabilize
echo -e "${YELLOW}Waiting for services to stabilize...${NC}"
sleep 30

# Step 2: Initialize Ollama models
echo -e "${BLUE}Step 2: Initializing Ollama models...${NC}"
docker compose --profile init up ollama-init

# Check if initialization was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Ollama model initialization completed successfully${NC}"
else
    echo -e "${RED}❌ Ollama model initialization failed${NC}"
    echo -e "${YELLOW}You can retry with: docker compose --profile init up ollama-init${NC}"
fi

# Step 3: Verify services
echo -e "${BLUE}Step 3: Verifying service health...${NC}"

# Check service status
echo -e "${YELLOW}Service Status:${NC}"
docker compose ps

echo ""
echo -e "${YELLOW}Health Checks:${NC}"

# Check each service
services=("db:5432" "qdrant:6333" "meilisearch:7700" "ollama:11434" "redis:6379" "api:8000")

for service in "${services[@]}"; do
    IFS=':' read -r name port <<< "$service"
    if curl -s -f "http://localhost:${port}" > /dev/null 2>&1 || \
       curl -s -f "http://localhost:${port}/health" > /dev/null 2>&1 || \
       curl -s -f "http://localhost:${port}/healthz" > /dev/null 2>&1 || \
       curl -s -f "http://localhost:${port}/api/tags" > /dev/null 2>&1; then
        echo -e "  ✅ ${name}: Healthy"
    else
        echo -e "  ⚠️  ${name}: Not responding (may still be starting)"
    fi
done

echo ""
echo -e "${GREEN}🎉 Deployment completed!${NC}"
echo -e "${BLUE}=======================================================${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. Access the frontend: http://localhost:3000"
echo -e "  2. Access the API: http://localhost:8000"
echo -e "  3. Access N8N: http://localhost:5678"
echo -e "  4. Login with default credentials: admin/admin123"
echo ""
echo -e "${YELLOW}To check Ollama models:${NC}"
echo -e "  curl http://localhost:11434/api/tags"
echo ""
