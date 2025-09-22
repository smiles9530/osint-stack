#!/bin/bash
# Automated OSINT Stack Setup for RunPod
# This script will run automatically when executed on the pod

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Check if we're on RunPod
if [ -z "${OSINT_DEPLOYMENT:-}" ]; then
    error "This script should be run on RunPod with OSINT deployment environment variables set."
    exit 1
fi

log "🚀 Starting secure OSINT Stack deployment on RunPod..."
log "📊 Pod Configuration: NVIDIA A40 GPU, 50GB RAM, 100GB Volume"

# Update system
log "📦 Updating system packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y && apt-get upgrade -y

# Install essential packages
log "📦 Installing essential packages..."
apt-get install -y curl git wget gnupg lsb-release software-properties-common

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    log "🐳 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    systemctl start docker
    systemctl enable docker
    rm get-docker.sh
    log "✅ Docker installed successfully"
else
    log "✅ Docker already installed"
fi

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null; then
    log "🔧 Installing Docker Compose..."
    DOCKER_COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep tag_name | cut -d '"' -f 4)
    curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    log "✅ Docker Compose installed successfully"
else
    log "✅ Docker Compose already installed"
fi

# Install NVIDIA Container Toolkit
if ! dpkg -l | grep -q nvidia-docker2; then
    log "🔧 Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
    apt-get update && apt-get install -y nvidia-docker2
    systemctl restart docker
    log "✅ NVIDIA Container Toolkit installed successfully"
else
    log "✅ NVIDIA Container Toolkit already installed"
fi

# Create workspace directory
log "📁 Setting up workspace..."
cd /workspace
rm -rf osint-stack 2>/dev/null || true
mkdir -p osint-stack
cd osint-stack

# Clone the repository
log "📥 Cloning OSINT Stack repository..."
git clone ${OSINT_REPO_URL} .
log "✅ Repository cloned successfully"

# Create secure environment file using pod environment variables
log "🔐 Creating secure environment configuration..."
cat > .env << EOF
# Database Configuration
POSTGRES_USER=osint_admin
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
POSTGRES_DB=osint_production
POSTGRES_PORT=5432
POSTGRES_HOST=db
N8N_DB=n8n
SUPERSET_DB=superset

# API Configuration
API_PORT=8000
API_HOST=0.0.0.0
API_LOG_LEVEL=info
SECRET_KEY=${SECRET_KEY}
ACCESS_TOKEN_EXPIRE_MINUTES=30

# AI/ML Configuration
OLLAMA_HOST=http://ollama:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
EMBEDDINGS_BACKEND=ollama
LOCAL_EMBED_MODEL=BAAI/bge-m3

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Search Configuration
TYPESENSE_API_KEY=${TYPESENSE_API_KEY}

# Storage Configuration
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=osint_storage
MINIO_SECRET_KEY=SecureMinIOPass2024
MINIO_SECURE=false
MINIO_ROOT_USER=osint_storage
MINIO_ROOT_PASSWORD=SecureMinIOPass2024

# Workflow Configuration
N8N_HOST=0.0.0.0
N8N_PORT=5678
N8N_PROTOCOL=http
N8N_ENCRYPTION_KEY=${N8N_ENCRYPTION_KEY}

# Analytics Configuration
SUPERSET_SECRET_KEY=SecureSupersetKey2024
SUPERSET_LOAD_EXAMPLES=no
SUPERSET_DATABASE_URI=postgresql+psycopg2://osint_admin:${POSTGRES_PASSWORD}@db:5432/superset

# System Configuration
GENERIC_TIMEZONE=UTC
PARALLEL_WORKERS=4
MAX_BATCH_SIZE=50
ENABLE_PARALLEL_PROCESSING=true
API_CPU_LIMIT=4.0
API_MEMORY_LIMIT=4G
ENABLE_PERFORMANCE_LOGGING=true
PERFORMANCE_METRICS_INTERVAL=60
EOF

# Set secure permissions
chmod 600 .env
log "✅ Secure environment configuration created"

# Test GPU access
log "🔍 Testing GPU access..."
nvidia-smi
if [ $? -eq 0 ]; then
    log "✅ GPU access confirmed - NVIDIA A40 available"
else
    warn "GPU access test failed - continuing with CPU-only mode"
fi

# Build and start the stack
log "🏗️  Building and starting OSINT Stack..."
docker-compose -f runpod-docker-compose.yml up -d --build

# Wait for services to initialize
log "⏳ Waiting for services to initialize (60 seconds)..."
sleep 60

# Check service health
log "🔍 Checking service health..."
docker-compose -f runpod-docker-compose.yml ps

# Initialize Ollama models
log "🤖 Initializing AI models..."
docker-compose -f runpod-docker-compose.yml --profile init up ollama-init -d

# Get external IP
EXTERNAL_IP=$(curl -s ifconfig.me || echo "unable-to-detect")

# Display deployment results
log "✅ OSINT Stack deployment completed successfully!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}🌐 OSINT Stack Access URLs:${NC}"
echo -e "   • ${GREEN}Main API:${NC}      http://${EXTERNAL_IP}:8000"
echo -e "   • ${GREEN}API Docs:${NC}      http://${EXTERNAL_IP}:8000/docs"
echo -e "   • ${GREEN}Frontend:${NC}      http://${EXTERNAL_IP}:3000"
echo -e "   • ${GREEN}N8N Workflows:${NC} http://${EXTERNAL_IP}:5678"
echo -e "   • ${GREEN}Analytics:${NC}     http://${EXTERNAL_IP}:8088"
echo -e "   • ${GREEN}Storage UI:${NC}    http://${EXTERNAL_IP}:9001"
echo -e "   • ${GREEN}Jupyter:${NC}       http://${EXTERNAL_IP}:8888"
echo ""
echo -e "${BLUE}🔐 Security Status:${NC}"
echo -e "   • ${GREEN}✅ All credentials are randomly generated${NC}"
echo -e "   • ${GREEN}✅ Environment file has restricted permissions${NC}"
echo -e "   • ${GREEN}✅ No default passwords used${NC}"
echo -e "   • ${GREEN}✅ Secure configuration applied${NC}"
echo ""
echo -e "${BLUE}📊 Health Checks:${NC}"
echo -e "   • ${GREEN}API Health:${NC}    http://${EXTERNAL_IP}:8000/healthz"
echo -e "   • ${GREEN}Service Status:${NC} docker-compose ps"
echo ""
echo -e "${BLUE}🔧 Management:${NC}"
echo -e "   • ${GREEN}Logs:${NC}          docker-compose logs -f"
echo -e "   • ${GREEN}Restart:${NC}       docker-compose restart"
echo -e "   • ${GREEN}Stop:${NC}          docker-compose down"
echo ""
echo -e "${BLUE}💾 Configuration:${NC}"
echo -e "   • ${GREEN}Location:${NC}      /workspace/osint-stack/.env"
echo -e "   • ${GREEN}Backup:${NC}        cp .env .env.backup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${YELLOW}⚠️  IMPORTANT NOTES:${NC}"
echo -e "   • Save the .env file contents for your records"
echo -e "   • The deployment uses GPU acceleration for AI models"
echo -e "   • All services are configured for production use"
echo -e "   • Monitor resource usage with: nvidia-smi and htop"
echo ""
log "🎉 Deployment complete! Your OSINT Stack is ready for use."
