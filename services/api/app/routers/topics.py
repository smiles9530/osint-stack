"""
Topics router
Handles topic discovery, analysis, and trend detection
"""

from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional, List, Dict, Any
from ..auth import get_current_active_user, User
from ..topic_discovery import topic_discovery_service
from ..topic_batch_processor import topic_batch_processor

router = APIRouter(prefix="/topics", tags=["Topics"])

@router.post("/discover")
async def discover_topics_endpoint(
    hours_back: int = Query(24, description="Number of hours to look back for articles", ge=1, le=168),
    min_documents: int = Query(10, description="Minimum documents per topic", ge=3, le=100),
    similarity_threshold: float = Query(0.8, description="Similarity threshold for topic assignment", ge=0.1, le=1.0),
    current_user: User = Depends(get_current_active_user)
):
    """Discover topics from recent articles"""
    try:
        result = await topic_discovery_service.discover_topics(
            hours_back=hours_back,
            min_documents=min_documents,
            similarity_threshold=similarity_threshold
        )
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Topic discovery failed: {str(e)}"
        )

@router.get("/trends")
async def get_topic_trends_endpoint(
    days_back: int = Query(7, description="Number of days to analyze", ge=1, le=30),
    topic_id: Optional[int] = Query(None, description="Specific topic ID to analyze"),
    current_user: User = Depends(get_current_active_user)
):
    """Get topic trends over time"""
    try:
        trends = await topic_discovery_service.get_topic_trends(
            days_back=days_back,
            topic_id=topic_id
        )
        return trends
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get topic trends: {str(e)}"
        )

@router.get("/campaigns")
async def detect_campaigns_endpoint(
    min_volume_increase: float = Query(2.0, description="Minimum volume increase multiplier", ge=1.0, le=10.0),
    time_window_hours: int = Query(6, description="Time window for volume comparison", ge=1, le=24),
    current_user: User = Depends(get_current_active_user)
):
    """Detect potential information campaigns"""
    try:
        campaigns = await topic_discovery_service.detect_campaigns(
            min_volume_increase=min_volume_increase,
            time_window_hours=time_window_hours
        )
        return campaigns
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Campaign detection failed: {str(e)}"
        )

@router.get("/statistics")
async def get_topic_statistics_endpoint(
    current_user: User = Depends(get_current_active_user)
):
    """Get topic analysis statistics"""
    try:
        stats = await topic_discovery_service.get_topic_statistics()
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get topic statistics: {str(e)}"
        )

@router.post("/process/batch")
async def process_topic_batch_endpoint(
    hours_back: int = Query(24, description="Number of hours to process", ge=1, le=168),
    current_user: User = Depends(get_current_active_user)
):
    """Process topics in batch for specified time period"""
    try:
        result = await topic_batch_processor.process_immediate(hours_back=hours_back)
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch topic processing failed: {str(e)}"
        )

@router.post("/process/start")
async def start_topic_batch_processing_endpoint(
    current_user: User = Depends(get_current_active_user)
):
    """Start continuous topic batch processing"""
    try:
        if topic_batch_processor.is_running:
            return {"message": "Batch processing already running", "status": "running"}
        
        # Start batch processing in background
        import asyncio
        asyncio.create_task(topic_batch_processor.start_batch_processing())
        
        result = {
            "message": "Topic discovery batch processing started",
            "status": "started",
            "interval_seconds": topic_batch_processor.processing_interval
        }
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start topic batch processing: {str(e)}"
        )

@router.post("/process/stop")
async def stop_topic_batch_processing_endpoint(
    current_user: User = Depends(get_current_active_user)
):
    """Stop continuous topic batch processing"""
    try:
        await topic_batch_processor.stop_batch_processing()
        result = {
            "message": "Topic discovery batch processing stopped",
            "status": "stopped"
        }
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stop topic batch processing: {str(e)}"
        )

@router.get("/process/status")
async def get_topic_batch_status_endpoint(
    current_user: User = Depends(get_current_active_user)
):
    """Get status of topic batch processing"""
    try:
        status = topic_batch_processor.get_processing_status()
        return status
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get topic batch status: {str(e)}"
        )
