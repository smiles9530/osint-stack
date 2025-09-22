# RunPod Deployment Guide for OSINT Stack

## 🚀 Quick Start

### 1. Create RunPod Account
- Go to [RunPod.io](https://runpod.io)
- Sign up and verify your account
- Add payment method (credit card required)

### 2. Create Pod
1. **Login to RunPod Console**
2. **Click "New Pod"**
3. **Select Template**: `runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04`
4. **GPU Selection**: Choose RTX 4090 (24GB VRAM) or RTX 3090 (24GB VRAM)
5. **Storage**: 50GB container + 100GB volume
6. **Network**: Public IP enabled
7. **Ports**: Configure the following ports:
   - Port 8000: API Service
   - Port 3000: Frontend
   - Port 5678: N8N Workflows
   - Port 8088: Superset Dashboard
   - Port 80: Web Interface

### 3. Deploy OSINT Stack

#### Option A: Quick Deploy (Recommended)
```bash
# SSH into your RunPod instance
ssh root@your-pod-ip

# Download and run deployment script
curl -fsSL https://raw.githubusercontent.com/your-username/osint-stack/main/runpod-deploy.sh | bash
```

#### Option B: Manual Deploy
```bash
# SSH into your RunPod instance
ssh root@your-pod-ip

# Clone repository
git clone https://github.com/your-username/osint-stack.git /app/osint-stack
cd /app/osint-stack

# Copy RunPod-specific files
cp runpod-docker-compose.yml docker-compose.yml
cp runpod-ollama-config.json ollama-gpu-config.json

# Make deployment script executable
chmod +x runpod-deploy.sh

# Run deployment
./runpod-deploy.sh
```

### 4. Access Your Stack

After deployment, access your services at:
- **🌐 Web Interface**: `http://your-pod-ip`
- **📊 API**: `http://your-pod-ip:8000`
- **🔧 N8N**: `http://your-pod-ip:5678`
- **📈 Superset**: `http://your-pod-ip:8088`
- **🤖 Ollama**: `http://your-pod-ip:11434`

## 🔧 Configuration

### GPU Settings
The deployment automatically configures:
- **CUDA Support**: Full GPU acceleration
- **Memory Management**: Optimized for large models
- **Model Loading**: GPU-optimized Ollama configuration

### Environment Variables
Key configurations in `.env`:
```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Ollama Configuration
OLLAMA_HOST=http://ollama:11434
OLLAMA_EMBED_MODEL=nomic-embed-text

# Database
POSTGRES_USER=osint
POSTGRES_PASSWORD=change_this_super_strong_password
POSTGRES_DB=osint
```

## 📊 Monitoring

### Check GPU Usage
```bash
# Monitor GPU usage
nvidia-smi

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Check Services
```bash
# Check all services
docker-compose ps

# Check logs
docker-compose logs -f api
docker-compose logs -f ollama
```

## 💰 Cost Management

### RunPod Pricing
- **RTX 4090**: ~$0.79/hour (~$570/month for 24/7)
- **RTX 3090**: ~$0.34/hour (~$245/month for 24/7)
- **RTX 3080**: ~$0.20/hour (~$145/month for 24/7)

### Cost Optimization
1. **Stop Pod When Not Using**: RunPod charges per hour
2. **Use Spot Instances**: Can save 30-50% on costs
3. **Auto-shutdown**: Set up scripts to stop pod after inactivity
4. **Monitor Usage**: Use RunPod's usage dashboard

## 🔒 Security

### Firewall Configuration
```bash
# Allow only necessary ports
ufw allow 22    # SSH
ufw allow 80    # HTTP
ufw allow 443   # HTTPS
ufw allow 8000  # API
ufw allow 5678  # N8N
ufw allow 8088  # Superset
ufw enable
```

### SSL Certificate (Optional)
```bash
# Install Certbot
apt-get install certbot

# Get SSL certificate
certbot certonly --standalone -d your-domain.com
```

## 🚨 Troubleshooting

### Common Issues

1. **GPU Not Detected**
   ```bash
   # Check NVIDIA driver
   nvidia-smi
   
   # Restart Docker with GPU support
   systemctl restart docker
   ```

2. **Out of Memory**
   ```bash
   # Check GPU memory
   nvidia-smi
   
   # Reduce model size in Ollama config
   # Edit ollama-gpu-config.json
   ```

3. **Port Not Accessible**
   ```bash
   # Check if service is running
   docker-compose ps
   
   # Check port binding
   netstat -tlnp
   ```

### Support
- **RunPod Documentation**: [docs.runpod.io](https://docs.runpod.io)
- **RunPod Discord**: [discord.gg/runpod](https://discord.gg/runpod)
- **GitHub Issues**: [github.com/your-username/osint-stack/issues](https://github.com/your-username/osint-stack/issues)

## 📈 Scaling

### Vertical Scaling
- Upgrade to larger GPU (RTX 4090 → A100)
- Increase RAM and storage
- Use RunPod's resize feature

### Horizontal Scaling
- Deploy multiple pods for different services
- Use load balancer for API services
- Implement auto-scaling based on demand

## 🎯 Best Practices

1. **Regular Backups**: Backup your data regularly
2. **Monitor Costs**: Set up billing alerts
3. **Update Security**: Keep system and dependencies updated
4. **Optimize Models**: Use appropriate model sizes for your use case
5. **Resource Monitoring**: Monitor GPU and memory usage

## 📞 Support

For issues specific to:
- **RunPod Platform**: Contact RunPod support
- **OSINT Stack**: Check GitHub issues or create new one
- **GPU Issues**: Check NVIDIA documentation
- **Docker Issues**: Check Docker documentation
