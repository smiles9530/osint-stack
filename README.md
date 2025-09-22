# OSINT News Analysis Stack

A comprehensive Open Source Intelligence platform for automated news collection, processing, and analysis using AI and modern data technologies.

## 🚀 Features

- **Automated News Collection**: RSS monitoring and web scraping
- **AI-Powered Analysis**: Text embeddings and semantic search
- **Real-time Processing**: Continuous data ingestion
- **Rich Analytics**: Trend detection and visualization
- **Secure API**: JWT authentication and validation
- **Scalable Architecture**: Microservices with Docker

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python 3.11)
- **Database**: PostgreSQL + TimescaleDB + PostGIS
- **Vector DB**: Qdrant
- **Search**: Meilisearch
- **AI/ML**: Ollama + Sentence Transformers
- **Workflows**: N8N
- **Analytics**: Apache Superset

## 📋 Prerequisites

- Docker and Docker Compose
- 8GB+ RAM (for AI models)
- 20GB+ disk space

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd osint-stack
```

### 2. Configure Environment

Create `.env` file with your configuration:

```env
# Database Configuration
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=your_database_name

# Security
SECRET_KEY=your_secure_jwt_secret_key
ACCESS_TOKEN_EXPIRE_MINUTES=30

# API Configuration
API_LOG_LEVEL=INFO

# AI Configuration
EMBEDDINGS_BACKEND=ollama
OLLAMA_EMBED_MODEL=nomic-embed-text
```

### 3. Start Services

#### Option A: Automated Deployment with Model Initialization (Recommended)

Use the provided deployment script that automatically handles model initialization:

**Linux/Mac:**
```bash
chmod +x scripts/deploy-with-models.sh
./scripts/deploy-with-models.sh
```

**Windows:**
```cmd
scripts\deploy-with-models.bat
```

#### Option B: Manual Deployment

Start the core services:
```bash
docker compose up -d --build
```

Then initialize Ollama models:
```bash
docker compose --profile init up ollama-init
```

### 4. Verify Installation

Check service status:

```bash
docker compose ps
```

Check Ollama models are available:
```bash
curl http://localhost:11434/api/tags
```

All services should be running and healthy, and the embedding model should be downloaded.

## 🔐 Authentication

The API requires JWT authentication for all endpoints except `/healthz` and `/auth/login`.

⚠️ **Set up secure credentials before deployment!**

### Getting Access Token

```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}'
```

### Using the Token

```bash
curl -X GET "http://localhost:8000/articles" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## 📚 API Documentation

### Key Endpoints

- `POST /auth/login` - Authentication
- `GET /healthz` - Health check
- `POST /ingest/fetch_extract` - Extract article content
- `POST /embed` - Generate embeddings
- `GET /articles` - List articles (paginated)
- `GET /articles/{id}` - Get specific article

Interactive API documentation available at: `http://localhost:8000/docs`

## 🔧 Configuration

### Service Ports

| Service | Port | Description |
|---------|------|-------------|
| API | 8000 | Main API server |
| Database | 5432 | PostgreSQL |
| Qdrant | 6333 | Vector database |
| Meilisearch | 7700 | Search engine |
| Ollama | 11434 | AI model server |
| N8N | 5678 | Workflow automation |
| Superset | 8088 | Analytics dashboard |

## 📊 Monitoring

### Health Checks

```bash
# API health
curl http://localhost:8000/healthz

# View logs
docker compose logs -f
```

## 🔒 Security Features

- **JWT Authentication**: Secure token-based authentication
- **Input Validation**: Comprehensive request validation
- **SQL Injection Protection**: Parameterized queries
- **CORS Configuration**: Configurable cross-origin policies
- **Request Logging**: Audit trail for all requests
- **Error Handling**: Secure error responses without information leakage

## 🚨 Troubleshooting

### Common Issues

1. **Services not starting**: Check logs with `docker compose logs`
2. **Database issues**: Restart with `docker compose down && docker compose up -d`
3. **Authentication errors**: Verify credentials and token format
4. **Memory issues**: Ensure 8GB+ RAM available
5. **Model issues**: Check Ollama status at `http://localhost:11434/api/tags`

## 🔄 Workflow Automation

**N8N**: Automated workflows for RSS monitoring and content processing
- Access at: http://localhost:5678

## 📈 Analytics

**Superset Dashboard**: Article trends and source analysis
- Access at: http://localhost:8088

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues:
1. Check troubleshooting section
2. Review logs with `docker compose logs`
3. Create an issue with details