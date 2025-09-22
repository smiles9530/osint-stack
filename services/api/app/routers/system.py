"""
System router
Handles health checks, metrics, and system monitoring
"""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any
import datetime as dt
from ..auth import get_current_active_user, User
from ..schemas import HealthCheck, ErrorResponse
from ..monitoring import monitoring_service

router = APIRouter(prefix="/system", tags=["System"])

@router.get(
    "/health",
    response_model=HealthCheck,
    summary="System Health Check",
    description="Comprehensive system health check including all services and dependencies",
    responses={
        200: {"description": "System health status retrieved successfully"}
    }
)
async def get_system_health():
    """Get comprehensive system health status"""
    try:
        # Get health status from monitoring service
        health_data = await monitoring_service.get_system_health()
        
        return HealthCheck(
            status="healthy" if health_data.get("overall_health", False) else "unhealthy",
            timestamp=health_data.get("timestamp"),
            message=health_data.get("message", "System operational")
        )
    except Exception as e:
        return HealthCheck(
            status="unhealthy",
            timestamp=dt.datetime.now().isoformat(),
            message=f"Health check failed: {str(e)}"
        )

@router.get(
    "/healthz",
    response_model=HealthCheck,
    summary="Health Check",
    description="Check API service health and status with detailed service information",
    responses={
        200: {
            "description": "Service is healthy", 
            "model": HealthCheck,
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "timestamp": "2024-01-15T10:30:00Z",
                        "version": "1.0.0",
                        "services": {
                            "database": {"status": "healthy", "response_time_ms": 5},
                            "cache": {"status": "healthy", "response_time_ms": 2},
                            "search": {"status": "healthy", "response_time_ms": 8}
                        },
                        "uptime_seconds": 86400,
                        "memory_usage_mb": 256.5,
                        "cpu_usage_percent": 15.2
                    }
                }
            }
        },
        503: {
            "description": "Service is unhealthy", 
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "example": {
                        "error": "Service unavailable",
                        "detail": "Database connection failed",
                        "request_id": "req_123456789",
                        "timestamp": "2024-01-15T10:30:00Z"
                    }
                }
            }
        }
    }
)
async def healthz():
    """Basic health check endpoint"""
    try:
        # Basic health check
        return HealthCheck(
            status="healthy",
            timestamp=dt.datetime.utcnow(),
            message="API service is operational"
        )
    except Exception as e:
        return HealthCheck(
            status="unhealthy",
            timestamp=dt.datetime.utcnow(),
            message=f"Service error: {str(e)}"
        )

@router.get(
    "/metrics",
    tags=["System"],
    summary="Prometheus Metrics",
    description="Get Prometheus metrics for monitoring",
    responses={
        200: {"description": "Metrics data", "content": {"text/plain": {"example": "# HELP http_requests_total Total HTTP requests\n# TYPE http_requests_total counter\nhttp_requests_total{method=\"GET\",endpoint=\"/healthz\"} 42"}}}
    }
)
async def metrics_endpoint():
    """Get Prometheus metrics"""
    try:
        metrics = await monitoring_service.get_metrics()
        return JSONResponse(content=metrics)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get metrics: {str(e)}"}
        )

@router.get(
    "/performance",
    response_model=Dict[str, Any],
    summary="Get Real-time Performance Metrics",
    description="Retrieve current system and processing performance metrics",
    responses={
        200: {"description": "Performance metrics retrieved successfully"}
    }
)
async def get_performance_metrics_endpoint(
    current_user: User = Depends(get_current_active_user)
):
    """Get real-time performance metrics"""
    try:
        metrics = await monitoring_service.get_performance_metrics()
        return metrics
    except Exception as e:
        return {"error": f"Failed to get performance metrics: {str(e)}"}

@router.get(
    "/performance/summary",
    response_model=Dict[str, Any],
    summary="Get Performance Summary",
    description="Get performance summary for specified time period",
    responses={
        200: {"description": "Performance summary retrieved successfully"}
    }
)
async def get_performance_summary_endpoint(
    hours: int = 1,
    current_user: User = Depends(get_current_active_user)
):
    """Get performance summary"""
    try:
        summary = await monitoring_service.get_performance_summary(hours)
        return summary
    except Exception as e:
        return {"error": f"Failed to get performance summary: {str(e)}"}

@router.get(
    "/performance/alerts",
    response_model=Dict[str, Any],
    summary="Get Performance Alerts",
    description="Get current performance alerts and warnings",
    responses={
        200: {"description": "Performance alerts retrieved successfully"}
    }
)
async def get_performance_alerts_endpoint(
    current_user: User = Depends(get_current_active_user)
):
    """Get performance alerts"""
    try:
        alerts = await monitoring_service.get_performance_alerts()
        return alerts
    except Exception as e:
        return {"error": f"Failed to get performance alerts: {str(e)}"}

@router.post(
    "/performance/export",
    response_model=Dict[str, Any],
    summary="Export Performance Metrics",
    description="Export performance metrics to JSON file",
    responses={
        200: {"description": "Performance metrics exported successfully"}
    }
)
async def export_performance_metrics(
    hours: int = 24,
    current_user: User = Depends(get_current_active_user)
):
    """Export performance metrics"""
    try:
        export_data = await monitoring_service.export_metrics(hours)
        return export_data
    except Exception as e:
        return {"error": f"Failed to export metrics: {str(e)}"}

@router.get("/queue/stats", response_model=dict)
async def get_queue_stats(current_user: User = Depends(get_current_active_user)):
    """Get queue statistics (admin only)"""
    try:
        # Check if current user is superuser
        if not hasattr(current_user, 'is_superuser') or not current_user.is_superuser:
            raise HTTPException(status_code=403, detail="Not enough permissions")
        
        stats = await task_manager.get_queue_stats()
        return {"queue_stats": stats}
    except Exception as e:
        logger.error(f"Error getting queue stats: {str(e)}")
        raise

@router.get(
    "/performance",
    response_model=Dict[str, Any],
    tags=["System"],
    summary="Get System Performance Metrics",
    description="Retrieve comprehensive system performance metrics including cache, database, and API statistics",
    responses={
        200: {"description": "System performance metrics retrieved successfully"}
    }
)
async def get_system_performance():
    """
    Get comprehensive system performance metrics
    
    Returns detailed performance metrics including:
    - Cache statistics (Redis)
    - Database connection pool status
    - API response times and throughput
    - Memory and CPU usage
    - Error rates and success rates
    """
    try:
        import psutil
        import asyncio
        from datetime import datetime
        
        # Get cache statistics
        cache_stats = await cache.get_stats()
        
        # Get database pool status
        db_pool_stats = {
            'pool_size': db_pool.pool.get_size() if db_pool.pool else 0,
            'pool_min_size': db_pool.pool.get_min_size() if db_pool.pool else 0,
            'pool_max_size': db_pool.pool.get_max_size() if db_pool.pool else 0,
            'pool_idle_size': db_pool.pool.get_idle_size() if db_pool.pool else 0,
        }
        
        # Get system metrics
        system_metrics = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        }
        
        
        # Calculate API performance metrics
        api_metrics = {
            'total_requests': getattr(get_system_performance, '_request_count', 0),
            'avg_response_time_ms': getattr(get_system_performance, '_avg_response_time', 0),
            'error_rate': getattr(get_system_performance, '_error_rate', 0),
            'throughput_rps': getattr(get_system_performance, '_throughput', 0)
        }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cache': cache_stats,
            'database': db_pool_stats,
            'system': system_metrics,
            'api': api_metrics,
            'status': 'healthy' if system_metrics['cpu_percent'] < 80 and system_metrics['memory_percent'] < 80 else 'warning'
        }
        
    except Exception as e:
        logger.error(f"Error retrieving system performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system performance metrics")
