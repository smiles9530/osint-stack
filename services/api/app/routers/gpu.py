"""
GPU router
Handles GPU monitoring, metrics, and alerts
"""

from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
from ..auth import get_current_active_user, User
from ..gpu_monitoring import get_gpu_status, get_gpu_metrics, get_gpu_alerts

router = APIRouter(prefix="/gpu", tags=["GPU"])

@router.get(
    "/status",
    response_class=JSONResponse,
    summary="GPU Status",
    description="Get current GPU status and performance metrics",
    responses={
        200: {"description": "GPU status retrieved successfully"}
    }
)
async def get_gpu_status_endpoint():
    """Get GPU status and performance metrics"""
    try:
        status = get_gpu_status()
        return status
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get GPU status: {str(e)}"
        )

@router.get(
    "/metrics",
    response_class=JSONResponse,
    summary="GPU Metrics",
    description="Get detailed GPU performance metrics and history",
    responses={
        200: {"description": "GPU metrics retrieved successfully"}
    }
)
async def get_gpu_metrics_endpoint(
    hours: int = Query(1, ge=1, le=24, description="Number of hours to include in metrics")
):
    """Get GPU performance metrics"""
    try:
        metrics = await get_gpu_metrics(hours)
        return metrics
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get GPU metrics: {str(e)}"
        )

@router.get(
    "/alerts",
    response_class=JSONResponse,
    summary="GPU Alerts",
    description="Get current GPU performance alerts and warnings",
    responses={
        200: {"description": "GPU alerts retrieved successfully"}
    }
)
async def get_gpu_alerts_endpoint():
    """Get GPU performance alerts"""
    try:
        alerts = await get_gpu_alerts()
        return alerts
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get GPU alerts: {str(e)}"
        )
