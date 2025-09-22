from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Optional, List, Union

class Settings(BaseSettings):
    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    
    # Vector Database (using PGVector)
    # qdrant_url: str = Field("http://qdrant:6333", env="QDRANT_URL")  # Removed
    # qdrant_api_key: Optional[str] = Field(None, env="QDRANT_API_KEY")  # Removed
    
    # Search Engine
    typesense_url: str = Field("http://typesense:8108", env="TYPESENSE_URL")
    typesense_api_key: Optional[str] = Field(None, env="TYPESENSE_API_KEY")
    
    # AI/ML
    ollama_host: str = Field("http://ollama:11434", env="OLLAMA_HOST")
    ollama_embed_model: str = Field("nomic-embed-text", env="OLLAMA_EMBED_MODEL")
    embeddings_backend: str = Field("simple", env="EMBEDDINGS_BACKEND")
    local_embed_model: str = Field("BAAI/bge-m3", env="LOCAL_EMBED_MODEL")
    
    # API Configuration
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_log_level: str = Field("INFO", env="API_LOG_LEVEL")
    
    # Security
    secret_key: str = Field(..., env="SECRET_KEY", description="JWT secret key - MUST be set in production")
    access_token_expire_minutes: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # CORS - restrictive configuration for security
    cors_origins: List[str] = Field(default=["http://localhost:3000", "http://localhost:8080"], env="CORS_ORIGINS")
    
    @field_validator('cors_origins', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',') if origin.strip()]
        return v

    # Redis Configuration
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")

    # MinIO Configuration
    minio_endpoint: str = Field("localhost:9000", env="MINIO_ENDPOINT")
    minio_access_key: str = Field(..., env="MINIO_ACCESS_KEY", description="MinIO access key - MUST be set")
    minio_secret_key: str = Field(..., env="MINIO_SECRET_KEY", description="MinIO secret key - MUST be set")
    minio_secure: bool = Field(False, env="MINIO_SECURE")

    # Performance Configuration
    max_connections: int = Field(20, env="MAX_CONNECTIONS")
    connection_timeout: int = Field(30, env="CONNECTION_TIMEOUT")
    cache_ttl: int = Field(3600, env="CACHE_TTL")

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
