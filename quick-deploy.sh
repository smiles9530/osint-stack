#!/bin/bash
set -e
echo "🚀 Quick OSINT Stack Deployment"
cd /workspace
apt-get update
apt-get install -y git docker.io docker-compose-plugin openssl curl
systemctl start docker
if [ -d "osint-stack" ]; then
    cd osint-stack && git pull
else
    git clone https://github.com/smiles9530/osint-stack.git && cd osint-stack
fi
PUBLIC_IP=$(curl -s ifconfig.me)
POSTGRES_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)
SECRET_KEY=$(openssl rand -base64 64 | tr -d "=+/" | cut -c1-64)
cat > .env << EOF
POSTGRES_USER=osint
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
POSTGRES_DB=osint
SECRET_KEY=${SECRET_KEY}
CORS_ORIGINS=https://${PUBLIC_IP}:3000,https://${PUBLIC_IP}:8000
EOF
docker-compose down || true
docker-compose up -d --build
echo "✅ Deployed! Access: https://${PUBLIC_IP}:8000"
echo "Credentials: DB=${POSTGRES_PASSWORD}, KEY=${SECRET_KEY}"
