#!/bin/bash

# RunPod OSINT Stack Deployment Script
# This script deploys the OSINT Stack on your RunPod instance

set -e

echo "🚀 Starting OSINT Stack deployment on RunPod..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in a RunPod environment
if [ ! -d "/workspace" ]; then
    print_warning "Not in a RunPod environment. Creating /workspace directory..."
    mkdir -p /workspace
fi

cd /workspace

# Update system packages
print_status "Updating system packages..."
apt-get update -qq
apt-get install -y curl git docker-compose-plugin jq

# Install Docker Compose v2 if not available
if ! command -v docker-compose &> /dev/null; then
    print_status "Installing Docker Compose..."
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
fi

# Check if Docker is running and start if needed
if ! docker info &> /dev/null; then
    print_status "Starting Docker service..."
    systemctl start docker
    systemctl enable docker
fi

# Check NVIDIA Docker runtime
if ! docker info | grep -q "nvidia"; then
    print_warning "NVIDIA Docker runtime may not be properly configured"
    print_status "Installing/configuring NVIDIA container toolkit..."
    
    # Install NVIDIA container toolkit
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
    
    apt-get update
    apt-get install -y nvidia-docker2
    systemctl restart docker
fi

# Clone the repository
print_status "Cloning OSINT Stack repository..."
if [ -d "osint-stack" ]; then
    print_status "Repository already exists, updating..."
    cd osint-stack
    git pull origin main
else
    git clone https://github.com/smiles9530/osint-stack.git
    cd osint-stack
fi

# Copy RunPod-specific configuration files
print_status "Setting up RunPod-specific configurations..."
cp runpod-docker-compose.yml docker-compose.yml
cp runpod-ollama-config.json ollama-gpu-config.json

# Create environment file from template
print_status "Creating environment configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    
    # Generate secure passwords and keys
    POSTGRES_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
    SECRET_KEY=$(openssl rand -base64 64 | tr -d "=+/" | cut -c1-50)
    TYPESENSE_API_KEY=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
    MINIO_ACCESS_KEY=$(openssl rand -base64 16 | tr -d "=+/" | cut -c1-12)
    MINIO_SECRET_KEY=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
    MINIO_ROOT_USER=$(openssl rand -base64 16 | tr -d "=+/" | cut -c1-12)
    MINIO_ROOT_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
    N8N_ENCRYPTION_KEY=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
    SUPERSET_SECRET_KEY=$(openssl rand -base64 64 | tr -d "=+/" | cut -c1-50)
    
    # Update .env file with generated values
    sed -i "s/your_secure_password_here/$POSTGRES_PASSWORD/g" .env
    sed -i "s/your_secure_jwt_secret_key_here/$SECRET_KEY/g" .env
    sed -i "s/your_typesense_api_key_here/$TYPESENSE_API_KEY/g" .env
    sed -i "s/your_minio_access_key_here/$MINIO_ACCESS_KEY/g" .env
    sed -i "s/your_minio_secret_key_here/$MINIO_SECRET_KEY/g" .env
    sed -i "s/your_minio_root_user_here/$MINIO_ROOT_USER/g" .env
    sed -i "s/your_minio_root_password_here/$MINIO_ROOT_PASSWORD/g" .env
    sed -i "s/your_n8n_encryption_key_here/$N8N_ENCRYPTION_KEY/g" .env
    sed -i "s/your_superset_secret_key_here/$SUPERSET_SECRET_KEY/g" .env
    
    print_status "Generated secure credentials for all services"
else
    print_status "Environment file already exists, keeping current configuration"
fi

# Build and start services
print_status "Building and starting OSINT Stack services..."
docker-compose down --remove-orphans || true
docker-compose build --no-cache

print_status "Starting core services (database, cache, storage)..."
docker-compose up -d db redis minio typesense

print_status "Waiting for core services to be healthy..."
sleep 30

# Check if core services are healthy
print_status "Checking core services health..."
docker-compose ps

print_status "Starting AI services (Ollama)..."
docker-compose up -d ollama

print_status "Waiting for Ollama to start..."
sleep 60

# Download essential models
print_status "Downloading AI models..."
docker-compose exec -T ollama ollama pull nomic-embed-text || print_warning "Failed to download nomic-embed-text model"
docker-compose exec -T ollama ollama pull llama3.2:3b || print_warning "Failed to download llama3.2:3b model"

print_status "Starting application services..."
docker-compose up -d api frontend n8n superset nginx

print_status "Waiting for all services to be ready..."
sleep 60

# Final health check
print_status "Performing final health check..."
docker-compose ps

# Get RunPod public IP
PUBLIC_IP=$(curl -s https://ipv4.icanhazip.com/ || echo "Unable to determine public IP")

print_status "🎉 OSINT Stack deployment completed!"
echo ""
echo "📊 Service Access URLs:"
echo "🌐 Main Interface: http://$PUBLIC_IP"
echo "📊 API Documentation: http://$PUBLIC_IP:8000/docs"
echo "🔧 N8N Workflows: http://$PUBLIC_IP:5678"
echo "📈 Superset Dashboard: http://$PUBLIC_IP:8088"
echo "🤖 Ollama API: http://$PUBLIC_IP:11434"
echo "💾 MinIO Console: http://$PUBLIC_IP:9001"
echo ""
print_status "💡 Tips:"
echo "- Use 'docker-compose logs -f [service]' to view logs"
echo "- Use 'docker-compose ps' to check service status"
echo "- Use 'nvidia-smi' to monitor GPU usage"
echo ""
print_status "🔐 Security Note: Change default passwords in production!"
echo ""
