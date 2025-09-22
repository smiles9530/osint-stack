#!/bin/bash
# Secure RunPod Deployment Script for OSINT Stack

set -e

echo "🚀 Deploying OSINT Stack on RunPod (Secure Version)..."

# Check if running on RunPod
if [ ! -f "/etc/runpod-release" ]; then
    echo "⚠️  This script is designed for RunPod environments"
fi

# Update system
apt-get update && apt-get upgrade -y

# Install required packages
apt-get install -y curl git wget gnupg lsb-release

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "📦 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    systemctl start docker
    systemctl enable docker
    rm get-docker.sh
fi

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null; then
    echo "📦 Installing Docker Compose..."
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
fi

# Install NVIDIA Container Toolkit if not present
if ! dpkg -l | grep -q nvidia-docker2; then
    echo "🔧 Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
    apt-get update && apt-get install -y nvidia-docker2
    systemctl restart docker
fi

# Create application directory
mkdir -p /workspace/osint-stack
cd /workspace/osint-stack

# Clone the clean repository
echo "📥 Cloning OSINT Stack repository..."
git clone https://github.com/smiles9530/osint-stack.git .

# Generate secure credentials
echo "🔐 Generating secure credentials..."
DB_PASSWORD=$(openssl rand -base64 32 | tr -d /=+ | cut -c1-25)
JWT_SECRET=$(openssl rand -base64 48 | tr -d /=+ | cut -c1-32)
TYPESENSE_KEY=$(openssl rand -base64 32 | tr -d /=+ | cut -c1-24)
N8N_ENCRYPTION_KEY=$(openssl rand -base64 48 | tr -d /=+ | cut -c1-32)
SUPERSET_SECRET=$(openssl rand -base64 64 | tr -d /=+ | cut -c1-50)
MINIO_PASSWORD=$(openssl rand -base64 32 | tr -d /=+ | cut -c1-20)

# Create secure environment file
echo "⚙️  Creating secure environment configuration..."
cat > .env << EOF
# Database Configuration
POSTGRES_USER=osint_admin
POSTGRES_PASSWORD=${DB_PASSWORD}
POSTGRES_DB=osint_production
POSTGRES_PORT=5432
POSTGRES_HOST=db
N8N_DB=n8n
SUPERSET_DB=superset

# API Configuration
API_PORT=8000
API_HOST=0.0.0.0
API_LOG_LEVEL=info
SECRET_KEY=${JWT_SECRET}
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
TYPESENSE_API_KEY=${TYPESENSE_KEY}

# Storage Configuration
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=osint_storage
MINIO_SECRET_KEY=${MINIO_PASSWORD}
MINIO_SECURE=false
MINIO_ROOT_USER=osint_storage
MINIO_ROOT_PASSWORD=${MINIO_PASSWORD}

# Workflow Configuration
N8N_HOST=0.0.0.0
N8N_PORT=5678
N8N_PROTOCOL=http
N8N_ENCRYPTION_KEY=${N8N_ENCRYPTION_KEY}

# Analytics Configuration
SUPERSET_SECRET_KEY=${SUPERSET_SECRET}
SUPERSET_LOAD_EXAMPLES=no
SUPERSET_DATABASE_URI=postgresql+psycopg2://osint_admin:${DB_PASSWORD}@db:5432/superset

# System Configuration
GENERIC_TIMEZONE=UTC
PARALLEL_WORKERS=4
MAX_BATCH_SIZE=50
ENABLE_PARALLEL_PROCESSING=true
API_CPU_LIMIT=4.0
API_MEMORY_LIMIT=4G
ENABLE_PERFORMANCE_LOGGING=true
EOF

# Set secure permissions on environment file
chmod 600 .env

# Build and start the stack
echo "🏗️  Building and starting OSINT Stack..."
docker-compose -f runpod-docker-compose.yml up -d --build

# Wait for services to be healthy
echo "⏳ Waiting for services to initialize..."
sleep 30

# Initialize Ollama models
echo "🤖 Initializing AI models..."
docker-compose -f runpod-docker-compose.yml --profile init up ollama-init

# Display deployment information
echo ""
echo "✅ OSINT Stack deployed successfully!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 Access URLs:"
echo "   • Main API:    http://$(curl -s ifconfig.me):8000"
echo "   • Frontend:    http://$(curl -s ifconfig.me):3000"
echo "   • N8N:         http://$(curl -s ifconfig.me):5678"
echo "   • Superset:    http://$(curl -s ifconfig.me):8088"
echo "   • MinIO:       http://$(curl -s ifconfig.me):9001"
echo ""
echo "🔐 Security Information:"
echo "   • All credentials are randomly generated and secure"
echo "   • Environment file (.env) has restricted permissions"
echo "   • No default passwords are used"
echo ""
echo "📊 Health Check:"
echo "   • API Health: http://$(curl -s ifconfig.me):8000/healthz"
echo ""
echo "🔧 Credentials saved in: /workspace/osint-stack/.env"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚠️  IMPORTANT: Save the .env file contents for your records!"
