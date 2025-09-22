#!/bin/bash
# RunPod Deployment Script for OSINT Stack

set -e

echo "🚀 Deploying OSINT Stack on RunPod..."

# Update system
apt-get update && apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
systemctl start docker
systemctl enable docker

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update && apt-get install -y nvidia-docker2
systemctl restart docker

# Clone repository
git clone https://github.com/your-username/osint-stack.git /app/osint-stack
cd /app/osint-stack

# Create environment file
cat > .env << EOF
# Database Configuration
POSTGRES_USER=osint
POSTGRES_PASSWORD=\$(openssl rand -base64 32)
POSTGRES_DB=osint
POSTGRES_PORT=5432
POSTGRES_HOST=db

# API Configuration
API_PORT=8000
API_HOST=0.0.0.0
API_LOG_LEVEL=info
SECRET_KEY=\$(openssl rand -base64 64)
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Ollama Configuration
OLLAMA_HOST=http://ollama:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
EMBEDDINGS_BACKEND=ollama
LOCAL_EMBED_MODEL=BAAI/bge-m3

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Typesense Configuration
TYPESENSE_API_KEY=\$(openssl rand -base64 32)

# MinIO Configuration
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=\$(openssl rand -base64 16)
MINIO_SECRET_KEY=\$(openssl rand -base64 32)
MINIO_SECURE=false
MINIO_ROOT_USER=\$(openssl rand -base64 16)
MINIO_ROOT_PASSWORD=\$(openssl rand -base64 32)

# N8N Configuration
N8N_HOST=0.0.0.0
N8N_PORT=5678
N8N_PROTOCOL=http
N8N_ENCRYPTION_KEY=\$(openssl rand -base64 32)

# Superset Configuration
SUPERSET_SECRET_KEY=\$(openssl rand -base64 64)
SUPERSET_LOAD_EXAMPLES=no
SUPERSET_DATABASE_URI=postgresql+psycopg2://osint:\${POSTGRES_PASSWORD}@db:5432/superset

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

# CORS Configuration - Production Environment
# Replace with your actual domain(s)
CORS_ORIGINS=https://yourdomain.com,https://api.yourdomain.com,https://www.yourdomain.com
EOF

# Start the stack
docker-compose up -d

echo "✅ OSINT Stack deployed successfully!"
echo "🌐 Access your stack at: http://$(curl -s ifconfig.me)"
echo "📊 API: http://$(curl -s ifconfig.me):8000"
echo "🔧 N8N: http://$(curl -s ifconfig.me):5678"
echo "📈 Superset: http://$(curl -s ifconfig.me):8088"
