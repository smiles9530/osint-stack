# 🚀 RunPod OSINT Stack Deployment Instructions

## 📋 Your RunPod Details
- **Pod ID**: `c0ek6hldgjrxhz`
- **Pod Name**: `leading_magenta_barracuda`
- **Public IP**: `194.68.245.40`
- **SSH Port**: `22093`
- **GPU**: NVIDIA A40 (48GB VRAM)
- **Volume**: 50GB at `/workspace`

## 🔗 Step 1: Connect to Your RunPod

### Option A: SSH Connection (Recommended)
```bash
ssh root@194.68.245.40 -p 22093
```

### Option B: Web Terminal
- Go to your RunPod dashboard
- Click on your pod `leading_magenta_barracuda`
- Use the web terminal

## 🚀 Step 2: Deploy OSINT Stack

Once connected to your RunPod, run these commands:

```bash
# Download and run the deployment script
curl -fsSL https://raw.githubusercontent.com/smiles9530/osint-stack/main/runpod-deployment-script.sh -o deploy.sh
chmod +x deploy.sh
./deploy.sh
```

### Alternative Manual Deployment
If the automatic script doesn't work, you can deploy manually:

```bash
# 1. Navigate to workspace
cd /workspace

# 2. Clone repository
git clone https://github.com/smiles9530/osint-stack.git
cd osint-stack

# 3. Setup configuration
cp runpod-docker-compose.yml docker-compose.yml
cp runpod-ollama-config.json ollama-gpu-config.json
cp .env.example .env

# 4. Start services
docker-compose up -d --build

# 5. Download AI models
docker-compose exec ollama ollama pull nomic-embed-text
docker-compose exec ollama ollama pull llama3.2:3b
```

## 🌐 Step 3: Access Your Services

After deployment, access your services at:

| Service | URL | Description |
|---------|-----|-------------|
| 🌐 **Main Interface** | `http://194.68.245.40` | Web interface |
| 📊 **API Docs** | `http://194.68.245.40:8000/docs` | FastAPI documentation |
| 🔧 **N8N** | `http://194.68.245.40:5678` | Workflow automation |
| 📈 **Superset** | `http://194.68.245.40:8088` | Analytics dashboard |
| 🤖 **Ollama** | `http://194.68.245.40:11434/api/tags` | AI model API |
| 💾 **MinIO** | `http://194.68.245.40:9001` | Object storage console |

## 🔧 Step 4: Monitoring & Management

### Check Service Status
```bash
cd /workspace/osint-stack
docker-compose ps
```

### View Logs
```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f api
docker-compose logs -f ollama
```

### Monitor GPU Usage
```bash
nvidia-smi
watch -n 1 nvidia-smi
```

### Restart Services
```bash
# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart api
```

## 🔐 Security & Access

### Default Credentials
The deployment script generates secure random passwords. Check your `.env` file:
```bash
cat .env | grep PASSWORD
cat .env | grep KEY
```

### Firewall Configuration (Optional)
```bash
# Configure basic firewall
ufw allow 22     # SSH
ufw allow 80     # HTTP
ufw allow 443    # HTTPS  
ufw allow 8000   # API
ufw allow 5678   # N8N
ufw allow 8088   # Superset
ufw enable
```

## 🚨 Troubleshooting

### Common Issues

**1. GPU Not Detected**
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
```

**2. Out of Memory**
```bash
# Check GPU memory
nvidia-smi
# Restart Docker
systemctl restart docker
```

**3. Services Not Starting**
```bash
# Check logs
docker-compose logs
# Rebuild and restart
docker-compose down
docker-compose up -d --build
```

**4. Port Access Issues**
```bash
# Check if ports are open
netstat -tlnp | grep -E ":(80|8000|5678|8088|11434)"
```

## 💰 Cost Management

- **Current Cost**: ~$0.40/hour (~$288/month for 24/7)
- **Stop Pod**: When not in use to save costs
- **Monitor Usage**: Check RunPod dashboard regularly

### Auto-Stop Script (Optional)
```bash
# Create auto-stop after 2 hours of inactivity
echo '#!/bin/bash
sleep 7200
docker-compose down
poweroff' > /workspace/auto-stop.sh
chmod +x /workspace/auto-stop.sh
nohup /workspace/auto-stop.sh &
```

## 📞 Support

- **RunPod Issues**: RunPod support or Discord
- **OSINT Stack Issues**: GitHub repository issues
- **GPU Issues**: NVIDIA documentation

## 🎯 Next Steps

1. **Test API**: Visit `http://194.68.245.40:8000/docs`
2. **Configure N8N**: Set up automated workflows
3. **Import Data**: Start ingesting news articles
4. **Set Monitoring**: Configure alerts and dashboards
5. **Security**: Change default passwords and configure SSL
