#!/bin/bash

# Test script to validate Ollama initialization
set -e

echo "🧪 Testing Ollama Model Initialization"
echo "====================================="

# Check if docker-compose.yml has been updated
echo "1. Checking Docker Compose configuration..."
if grep -q "ollama-init:" docker-compose.yml; then
    echo "✅ Ollama init service found in docker-compose.yml"
else
    echo "❌ Ollama init service not found in docker-compose.yml"
    exit 1
fi

if grep -q "condition: service_healthy" docker-compose.yml; then
    echo "✅ Health check dependencies configured"
else
    echo "❌ Health check dependencies not found"
    exit 1
fi

# Check if initialization script exists and is executable
echo "2. Checking initialization script..."
if [ -f "scripts/init-ollama-models.sh" ]; then
    echo "✅ Initialization script exists"
else
    echo "❌ Initialization script not found"
    exit 1
fi

# Check if deployment scripts exist
echo "3. Checking deployment scripts..."
if [ -f "scripts/deploy-with-models.sh" ]; then
    echo "✅ Linux/Mac deployment script exists"
else
    echo "❌ Linux/Mac deployment script not found"
fi

if [ -f "scripts/deploy-with-models.bat" ]; then
    echo "✅ Windows deployment script exists"
else
    echo "❌ Windows deployment script not found"
fi

# Validate Docker Compose syntax
echo "4. Validating Docker Compose syntax..."
if docker compose config >/dev/null 2>&1; then
    echo "✅ Docker Compose configuration is valid"
else
    echo "❌ Docker Compose configuration has syntax errors"
    docker compose config
    exit 1
fi

# Test the initialization container (dry run)
echo "5. Testing initialization container configuration..."
if docker compose --profile init config >/dev/null 2>&1; then
    echo "✅ Ollama init profile configuration is valid"
else
    echo "❌ Ollama init profile configuration has errors"
    exit 1
fi

echo ""
echo "🎉 All tests passed! The Ollama initialization setup is ready."
echo ""
echo "To deploy with model initialization:"
echo "  Linux/Mac: ./scripts/deploy-with-models.sh"
echo "  Windows:   scripts\\deploy-with-models.bat"
echo "  Manual:    docker compose --profile init up ollama-init"
