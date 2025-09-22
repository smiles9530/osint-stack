#!/bin/bash
# Secure environment initialization script
# This script generates strong random credentials for all services

set -e

echo "🔐 Initializing secure environment variables..."

# Function to generate secure random password
generate_password() {
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-32
}

# Function to generate secure API key
generate_api_key() {
    openssl rand -base64 64 | tr -d "=+/" | cut -c1-64
}

# Generate secure credentials
POSTGRES_PASSWORD=$(generate_password)
SECRET_KEY=$(generate_api_key)
TYPESENSE_API_KEY=$(generate_password)
MINIO_ACCESS_KEY=$(openssl rand -base64 16 | tr -d "=+/" | cut -c1-16)
MINIO_SECRET_KEY=$(generate_password)
MINIO_ROOT_USER=$(openssl rand -base64 16 | tr -d "=+/" | cut -c1-16)
MINIO_ROOT_PASSWORD=$(generate_password)
N8N_ENCRYPTION_KEY=$(generate_password)
SUPERSET_SECRET_KEY=$(generate_api_key)

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
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Ollama Configuration
OLLAMA_HOST=http://ollama:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
EMBEDDINGS_BACKEND=ollama
LOCAL_EMBED_MODEL=BAAI/bge-m3

# Qdrant Configuration
QDRANT__SERVICE__API_KEY=

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
N8N_HOST=localhost
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

# CORS Configuration - Development Environment
# For production, replace with your actual domain(s): https://yourdomain.com,https://api.yourdomain.com
CORS_ORIGINS=http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000,http://127.0.0.1:8080
EOF

# Set secure permissions
chmod 600 .env

echo "✅ Secure environment file created with strong random credentials"
echo "⚠️  IMPORTANT: Save these credentials securely!"
echo ""
echo "Database Password: ${POSTGRES_PASSWORD}"
echo "API Secret Key: ${SECRET_KEY}"
echo "MinIO Access Key: ${MINIO_ACCESS_KEY}"
echo "MinIO Secret Key: ${MINIO_SECRET_KEY}"
echo ""
echo "🔒 .env file has been created with secure permissions (600)"
echo "📝 Consider backing up these credentials in a secure password manager"
