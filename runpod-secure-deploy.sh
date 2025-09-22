#!/bin/bash
# Secure RunPod Deployment Script for OSINT Stack
# This script deploys the security-hardened OSINT stack with proper credentials

set -e

echo "🚀 Starting secure OSINT Stack deployment on RunPod..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get RunPod public IP for CORS configuration
PUBLIC_IP=$(curl -s ifconfig.me)
POD_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id || echo "unknown")

echo -e "${BLUE}📍 Pod Information:${NC}"
echo -e "   • Pod ID: ${POD_ID}"
echo -e "   • Public IP: ${PUBLIC_IP}"
echo -e "   • Deployment Mode: ${DEPLOYMENT_MODE:-secure}"

# Update system
echo -e "${YELLOW}📦 Updating system packages...${NC}"
apt-get update && apt-get upgrade -y

# Install required packages
echo -e "${YELLOW}📦 Installing dependencies...${NC}"
apt-get install -y curl wget git docker.io docker-compose-plugin openssl

# Start Docker service
systemctl start docker
systemctl enable docker

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}📦 Installing Docker Compose...${NC}"
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
fi

# Navigate to workspace
cd /workspace

# Clone or update repository
if [ -d "osint-stack" ]; then
    echo -e "${YELLOW}🔄 Updating existing repository...${NC}"
    cd osint-stack
    git fetch origin
    git reset --hard origin/main
else
    echo -e "${YELLOW}📥 Cloning repository...${NC}"
    git clone https://github.com/smiles9530/osint-stack.git
    cd osint-stack
fi

# Generate secure environment configuration
echo -e "${YELLOW}🔐 Generating secure environment configuration...${NC}"

# Use existing environment variables or generate new ones
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)}
SECRET_KEY=${SECRET_KEY:-$(openssl rand -base64 64 | tr -d "=+/" | cut -c1-64)}
TYPESENSE_API_KEY=${TYPESENSE_API_KEY:-$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)}
N8N_ENCRYPTION_KEY=${N8N_ENCRYPTION_KEY:-$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)}

# Generate MinIO credentials
MINIO_ACCESS_KEY=$(openssl rand -base64 16 | tr -d "=+/" | cut -c1-16)
MINIO_SECRET_KEY=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)
MINIO_ROOT_USER=$(openssl rand -base64 16 | tr -d "=+/" | cut -c1-16)
MINIO_ROOT_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)
SUPERSET_SECRET_KEY=$(openssl rand -base64 64 | tr -d "=+/" | cut -c1-64)

# Configure CORS for RunPod environment
CORS_ORIGINS="https://${PUBLIC_IP}:3000,https://${PUBLIC_IP}:8000,https://${PUBLIC_IP}:8080"

# Create secure .env file
cat > .env << EOF
# Database Configuration
POSTGRES_USER=osint
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
POSTGRES_DB=osint
POSTGRES_PORT=5432
POSTGRES_HOST=db

# API Configuration
API_PORT=8000
API_HOST=0.0.0.0
API_LOG_LEVEL=info
SECRET_KEY=${SECRET_KEY}
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Ollama Configuration
OLLAMA_HOST=http://ollama:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
EMBEDDINGS_BACKEND=ollama
LOCAL_EMBED_MODEL=BAAI/bge-m3

# Typesense Configuration
TYPESENSE_API_KEY=${TYPESENSE_API_KEY}

# MinIO Configuration
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY}
MINIO_SECRET_KEY=${MINIO_SECRET_KEY}
MINIO_SECURE=false
MINIO_ROOT_USER=${MINIO_ROOT_USER}
MINIO_ROOT_PASSWORD=${MINIO_ROOT_PASSWORD}

# N8N Configuration
N8N_HOST=0.0.0.0
N8N_PORT=5678
N8N_PROTOCOL=http
N8N_ENCRYPTION_KEY=${N8N_ENCRYPTION_KEY}

# Superset Configuration
SUPERSET_SECRET_KEY=${SUPERSET_SECRET_KEY}
SUPERSET_LOAD_EXAMPLES=no
SUPERSET_DATABASE_URI=postgresql+psycopg2://osint:${POSTGRES_PASSWORD}@db:5432/superset

# Timezone
GENERIC_TIMEZONE=UTC

# Parallel Processing Configuration
PARALLEL_WORKERS=4
MAX_BATCH_SIZE=50
ENABLE_PARALLEL_PROCESSING=true
API_CPU_LIMIT=4.0
API_MEMORY_LIMIT=4G
FORECAST_CPU_LIMIT=2.0
FORECAST_MEMORY_LIMIT=2G
ENABLE_MEMORY_MAPPING=true
ML_BATCH_LIMIT=100
FORECAST_BATCH_LIMIT=50
ENABLE_PERFORMANCE_LOGGING=true
PERFORMANCE_METRICS_INTERVAL=60
AUTO_SCALE_WORKERS=false
MIN_WORKERS=2
MAX_WORKERS=8
SCALE_UP_THRESHOLD=80
SCALE_DOWN_THRESHOLD=30

# CORS Configuration - RunPod Environment
CORS_ORIGINS=${CORS_ORIGINS}
EOF

# Set secure permissions
chmod 600 .env

echo -e "${GREEN}✅ Environment configuration created${NC}"

# Stop any existing containers
echo -e "${YELLOW}🛑 Stopping existing containers...${NC}"
docker-compose down --remove-orphans || true

# Build and start the stack
echo -e "${YELLOW}🏗️  Building and starting OSINT Stack...${NC}"
docker-compose up -d --build

echo -e "${YELLOW}⏳ Waiting for services to start...${NC}"
sleep 30

# Check service health
echo -e "${BLUE}🔍 Checking service status...${NC}"
docker-compose ps

# Display access information
echo -e "${GREEN}🎉 OSINT Stack deployed successfully!${NC}"
echo -e ""
echo -e "${BLUE}📊 Access URLs:${NC}"
echo -e "   • API Documentation: https://${PUBLIC_IP}:8000/docs"
echo -e "   • Frontend: https://${PUBLIC_IP}:3000"
echo -e "   • N8N Automation: https://${PUBLIC_IP}:5678"
echo -e "   • Superset Analytics: https://${PUBLIC_IP}:8088"
echo -e "   • MinIO Console: https://${PUBLIC_IP}:9001"
echo -e ""
echo -e "${YELLOW}🔐 Important Credentials (SAVE THESE SECURELY):${NC}"
echo -e "   • Database Password: ${POSTGRES_PASSWORD}"
echo -e "   • API Secret Key: ${SECRET_KEY}"
echo -e "   • MinIO Access Key: ${MINIO_ACCESS_KEY}"
echo -e "   • MinIO Secret Key: ${MINIO_SECRET_KEY}"
echo -e "   • MinIO Root User: ${MINIO_ROOT_USER}"
echo -e "   • MinIO Root Password: ${MINIO_ROOT_PASSWORD}"
echo -e ""
echo -e "${RED}⚠️  Security Notes:${NC}"
echo -e "   • No default admin users are created"
echo -e "   • Use the API or create_user.py script to create users"
echo -e "   • All credentials are randomly generated"
echo -e "   • CORS is configured for this RunPod instance"
echo -e ""
echo -e "${BLUE}📚 Next Steps:${NC}"
echo -e "   1. Create an admin user: cd services/api && python create_user.py"
echo -e "   2. Access the frontend at: https://${PUBLIC_IP}:3000"
echo -e "   3. Review security documentation: cat SECURITY.md"
echo -e ""
echo -e "${GREEN}🚀 Deployment complete!${NC}"
