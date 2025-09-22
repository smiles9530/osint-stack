"""
Refactored main.py - Clean, organized FastAPI application
Uses routers for better organization and maintainability
"""

from fastapi import FastAPI, HTTPException, Depends, Request, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field
import datetime as dt
import uuid
import time
import json
from contextlib import asynccontextmanager
import asyncio

# Import configuration and core modules
from .config import settings
from . import db, embedding
from .auth import authenticate_user, create_access_token, get_current_active_user, create_user, update_user_last_login, Token, User
from .validators import (
    PuppeteerScrapeRequestValidator, PuppeteerScreenshotRequestValidator,
    PuppeteerPdfRequestValidator, PuppeteerMetadataRequestValidator,
    PuppeteerScrapeMultipleRequestValidator
)
from .enhanced_error_handling import APIError, ErrorHandler
from .logging_config import setup_logging, logger
from .cache import cache, article_cache_key, articles_list_cache_key, embedding_cache_key, source_cache_key
import redis.asyncio as redis
from .db_pool import db_pool
from .monitoring import monitoring_service, initialize_monitoring
from .async_tasks import task_manager, periodic_scheduler, schedule_embedding_task, schedule_processing_task
from .websocket_manager import websocket_endpoint, send_periodic_updates, NotificationService
from .ml_pipeline import ml_pipeline
from .parallel_ml import get_parallel_processing_stats

# Import all routers
from .routers import (
    auth, system, articles, search, ml, 
    scraping, topics, sources, intelligence, analysis, 
    gpu, bandit, feedback
)

# Import schemas
from .schemas import (
    LoginRequest, UserCreate, Article, ArticleList,
    EmbedRequest, EmbedResponse, SearchRequest, SearchResponse, SourceList,
    IntelligenceSourceList, HealthCheck, ErrorResponse,
    BiasAnalysis, StanceSentimentAnalysisRequest,
    StanceSentimentAnalysisResponse, ForecastRequest, ForecastResponse,
    BanditSelection, BanditUpdate, BanditStats, FeedbackSubmission, FeedbackResponse,
    Digest, DigestList
)

# Initialize logging
setup_logging()
logger.info("Starting OSINT Stack API")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up OSINT Stack API...")
    
    # Initialize monitoring
    redis_client = redis.from_url(settings.redis_url)
    await initialize_monitoring(redis_client)
    
    # Initialize database pool
    await db_pool.create_pool()
    
    # Start background tasks
    await task_manager.start()
    
    # Schedule periodic cleanup task (daily)
    async def cleanup_task():
        logger.info("Running daily cleanup task")
        # Clean up expired cache entries
        await cache.delete_pattern("expired:*")
        # Add other cleanup tasks here
    
    await periodic_scheduler.schedule_periodic(
        "daily_cleanup",
        cleanup_task,
        86400  # 24 hours
    )
    
    # Schedule performance monitoring task (every 5 minutes)
    async def performance_monitoring_task():
        try:
            # Log performance metrics
            cache_stats = await cache.get_stats()
            logger.info(f"Performance metrics - Cache: {cache_stats}")
        except Exception as e:
            logger.warning(f"Performance monitoring error: {e}")
    
    await periodic_scheduler.schedule_periodic(
        "performance_monitoring",
        performance_monitoring_task,
        300  # 5 minutes
    )
    
    # Start WebSocket updates
    periodic_task = asyncio.create_task(send_periodic_updates())
    # Store the task for proper cleanup
    app.state.periodic_task = periodic_task
    
    logger.info("OSINT Stack API startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down OSINT Stack API...")
    
    # Stop background tasks
    await task_manager.stop()
    await periodic_scheduler.cancel_all()
    
    # Cancel periodic updates task
    if hasattr(app.state, 'periodic_task'):
        app.state.periodic_task.cancel()
        try:
            await app.state.periodic_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error cancelling periodic task: {e}")
    
    # Shutdown WebSocket manager
    try:
        from .websocket_manager import manager
        await manager.shutdown()
    except Exception as e:
        logger.error(f"Error shutting down WebSocket manager: {e}")
    
    # Close database pool
    await db_pool.close_pool()
    
    logger.info("OSINT Stack API shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="OSINT Stack API",
    description="Advanced Open Source Intelligence platform with enhanced ML-powered analysis, real-time processing, and comprehensive monitoring",
    version="2.1.0",
    docs_url=None,  # Disable default docs
    redoc_url=None,  # Disable default redoc
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(auth.router)
app.include_router(system.router)
app.include_router(articles.router)
app.include_router(search.router)
app.include_router(ml.router)
app.include_router(scraping.router)
app.include_router(topics.router)
app.include_router(sources.router)
app.include_router(intelligence.router)
app.include_router(analysis.router)
app.include_router(gpu.router)
app.include_router(bandit.router)
app.include_router(feedback.router)

# Include existing routers

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="OSINT Stack API",
        version="2.1.0",
        description="Advanced Open Source Intelligence platform with enhanced ML-powered analysis, real-time processing, and comprehensive monitoring",
        routes=app.routes,
    )
    
    # Add enhanced OpenAPI metadata
    openapi_schema["info"]["contact"] = {
        "name": "OSINT Stack Team",
        "email": "support@osint-stack.com",
        "url": "https://github.com/osint-stack"
    }
    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
    openapi_schema["info"]["x-logo"] = {
        "url": "https://osint-stack.com/logo.png",
        "altText": "OSINT Stack Logo"
    }
    
    # Add custom tags with enhanced descriptions
    openapi_schema["tags"] = [
        {"name": "Authentication", "description": "User authentication, management, and security"},
        {"name": "System", "description": "System health monitoring, performance metrics, and operational status"},
        {"name": "Articles", "description": "Article ingestion, management, and content processing"},
        {"name": "Search", "description": "Advanced search functionality with hybrid vector/BM25 search and entity extraction"},
        {"name": "ML Processing", "description": "Machine learning operations, batch processing, and model management"},
        {"name": "Web Scraping", "description": "Web scraping with Puppeteer for content extraction"},
        {"name": "Topics", "description": "Topic discovery, trend analysis, and content clustering"},
        {"name": "Sources", "description": "Source management, quality scoring, and automated monitoring"},
        {"name": "Intelligence", "description": "Intelligence processing, threat detection, and real-time analytics"},
        {"name": "Analysis", "description": "Advanced text analysis with ML-powered sentiment, bias, and stance detection"},
        {"name": "GPU", "description": "GPU monitoring, utilization metrics, and performance optimization"},
        {"name": "Bandit", "description": "Contextual bandit algorithms for intelligent content selection and optimization"},
        {"name": "Feedback", "description": "User feedback collection, ML-powered digest generation, and recommendation systems"},
    ]
    
    # Add server information
    openapi_schema["servers"] = [
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        },
        {
            "url": "https://api.osint-stack.com",
            "description": "Production server"
        }
    ]
    
    # Add enhanced error responses
    openapi_schema["components"]["responses"] = {
        "ValidationError": {
            "description": "Validation Error",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "error": {"type": "string"},
                            "message": {"type": "string"},
                            "details": {"type": "object"}
                        }
                    }
                }
            }
        },
        "InternalServerError": {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "error": {"type": "string"},
                            "message": {"type": "string"},
                            "detail": {"type": "string"}
                        }
                    }
                }
            }
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Custom documentation endpoints
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="OSINT Stack API v2.1.0 - Enhanced Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_ui_parameters={
            "defaultModelsExpandDepth": 2,
            "defaultModelExpandDepth": 2,
            "displayRequestDuration": True,
            "docExpansion": "list",
            "filter": True,
            "showExtensions": True,
            "showCommonExtensions": True,
            "tryItOutEnabled": True
        }
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title="OSINT Stack API v2.1.0 - Enhanced Documentation",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@2.0.0/bundles/redoc.standalone.js",
    )

# Middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests"""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# Import enhanced error handling
from .enhanced_error_handling import enhanced_exception_handler, ErrorHandler, APIError

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Enhanced global exception handler"""
    return await enhanced_exception_handler(request, exc)

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint_handler(websocket: WebSocket, token: str = None):
    """WebSocket endpoint for real-time updates"""
    await websocket_endpoint(websocket, token)

# Legacy endpoints for backward compatibility
# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "OSINT Stack API",
        "version": "2.1.0",
        "description": "Advanced Open Source Intelligence platform with enhanced ML-powered analysis, real-time processing, and comprehensive monitoring",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_url": "/healthz",
        "status": "operational",
        "features": [
            "Enhanced ML-powered analysis with caching and batch processing",
            "Real-time WebSocket communication with message queuing",
            "Advanced contextual bandit algorithms",
            "Comprehensive source quality scoring and monitoring",
            "Intelligence processing with threat detection",
            "ML-powered digest generation and recommendations",
            "Circuit breaker and retry mechanisms for reliability",
            "Hybrid search with vector similarity and BM25"
        ]
    }

# Direct health check endpoint for Docker health checks
@app.get("/healthz", tags=["System"])
async def healthz():
    """Direct health check endpoint for Docker health checks"""
    try:
        # Import here to avoid circular imports
        from .routers.system import healthz as system_healthz
        return await system_healthz()
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": dt.datetime.utcnow(),
                "message": f"Health check failed: {str(e)}"
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.api_log_level.lower(),
        reload=True
    )
