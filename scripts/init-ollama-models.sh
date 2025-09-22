#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🤖 Ollama Model Initialization Script${NC}"
echo -e "${BLUE}====================================${NC}"

# Configuration
OLLAMA_HOST=${OLLAMA_HOST:-"http://ollama:11434"}
OLLAMA_EMBED_MODEL=${OLLAMA_EMBED_MODEL:-"nomic-embed-text"}
MAX_RETRIES=30
RETRY_DELAY=10

echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Ollama Host: ${OLLAMA_HOST}"
echo -e "  Embed Model: ${OLLAMA_EMBED_MODEL}"
echo ""

# Function to check if Ollama is ready
check_ollama_ready() {
    local retries=0
    echo -e "${YELLOW}Waiting for Ollama to be ready...${NC}"
    
    while [ $retries -lt $MAX_RETRIES ]; do
        if curl -s -f "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Ollama is ready!${NC}"
            return 0
        fi
        
        retries=$((retries + 1))
        echo -e "  Attempt ${retries}/${MAX_RETRIES} - Ollama not ready yet, waiting ${RETRY_DELAY}s..."
        sleep $RETRY_DELAY
    done
    
    echo -e "${RED}❌ Ollama failed to become ready after ${MAX_RETRIES} attempts${NC}"
    return 1
}

# Function to check if a model exists
model_exists() {
    local model_name=$1
    curl -s "${OLLAMA_HOST}/api/tags" | grep -q "\"name\":\"${model_name}\""
}

# Function to pull a model
pull_model() {
    local model_name=$1
    echo -e "${YELLOW}📥 Pulling model: ${model_name}${NC}"
    
    # Use curl to call the pull API
    local response=$(curl -s -X POST "${OLLAMA_HOST}/api/pull" \
        -H "Content-Type: application/json" \
        -d "{\"name\":\"${model_name}\"}")
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Successfully initiated pull for model: ${model_name}${NC}"
        
        # Wait for the pull to complete by checking if model exists
        local wait_retries=0
        local max_wait=60  # 10 minutes max wait time
        
        echo -e "${YELLOW}Waiting for model download to complete...${NC}"
        while [ $wait_retries -lt $max_wait ]; do
            if model_exists "$model_name"; then
                echo -e "${GREEN}✅ Model ${model_name} is now available!${NC}"
                return 0
            fi
            
            wait_retries=$((wait_retries + 1))
            echo -e "  Waiting for download completion... (${wait_retries}/${max_wait})"
            sleep 10
        done
        
        echo -e "${RED}❌ Model download timed out${NC}"
        return 1
    else
        echo -e "${RED}❌ Failed to pull model: ${model_name}${NC}"
        return 1
    fi
}

# Function to verify model functionality
verify_model() {
    local model_name=$1
    echo -e "${YELLOW}🔍 Verifying model: ${model_name}${NC}"
    
    # Test embedding generation
    local response=$(curl -s -X POST "${OLLAMA_HOST}/api/embeddings" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${model_name}\",\"input\":[\"test text\"]}")
    
    if echo "$response" | grep -q "embeddings"; then
        echo -e "${GREEN}✅ Model ${model_name} is working correctly!${NC}"
        return 0
    else
        echo -e "${RED}❌ Model ${model_name} verification failed${NC}"
        echo -e "Response: $response"
        return 1
    fi
}

# Main initialization process
main() {
    echo -e "${BLUE}Starting Ollama model initialization...${NC}"
    
    # Check if Ollama is ready
    if ! check_ollama_ready; then
        echo -e "${RED}❌ Ollama initialization failed - service not ready${NC}"
        exit 1
    fi
    
    # Check if embedding model exists
    echo -e "${YELLOW}Checking if embedding model exists: ${OLLAMA_EMBED_MODEL}${NC}"
    
    if model_exists "$OLLAMA_EMBED_MODEL"; then
        echo -e "${GREEN}✅ Model ${OLLAMA_EMBED_MODEL} already exists${NC}"
    else
        echo -e "${YELLOW}Model ${OLLAMA_EMBED_MODEL} not found, pulling...${NC}"
        if ! pull_model "$OLLAMA_EMBED_MODEL"; then
            echo -e "${RED}❌ Failed to pull embedding model${NC}"
            exit 1
        fi
    fi
    
    # Verify the model works
    if ! verify_model "$OLLAMA_EMBED_MODEL"; then
        echo -e "${RED}❌ Model verification failed${NC}"
        exit 1
    fi
    
    # List all available models
    echo -e "${BLUE}📋 Available models:${NC}"
    curl -s "${OLLAMA_HOST}/api/tags" | grep '"name"' | sed 's/.*"name":"\([^"]*\)".*/  - \1/'
    
    echo -e "${GREEN}🎉 Ollama model initialization completed successfully!${NC}"
    echo -e "${BLUE}====================================${NC}"
}

# Run main function
main "$@"
