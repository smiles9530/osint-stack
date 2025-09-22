"""
Enhanced Intelligence router
Handles intelligence processing, dashboard updates, threat detection, and advanced analytics
"""

import asyncio
import logging
import time
import uuid
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from fastapi import APIRouter, Depends, HTTPException, Request, Query, BackgroundTasks
from pydantic import BaseModel, Field

from ..auth import get_current_active_user, User
from ..schemas import ErrorResponse
from ..enhanced_error_handling import ErrorHandler, APIError
from ..cache import cache
from ..monitoring import monitoring_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/intelligence", tags=["Intelligence"])

# Request/Response models
class IntelligenceDataPoint(BaseModel):
    id: Optional[str] = None
    type: str = Field(..., description="Type of intelligence data")
    content: Optional[str] = Field(None, description="Text content for analysis")
    title: Optional[str] = Field(None, description="Title of the data point")
    description: Optional[str] = Field(None, description="Description of the data point")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    priority: Optional[str] = Field("medium", description="Priority level: low, medium, high, critical")
    source: Optional[str] = Field(None, description="Source of the intelligence data")

class IntelligenceProcessingRequest(BaseModel):
    data: List[IntelligenceDataPoint] = Field(..., description="List of intelligence data points to process")
    processing_options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Processing configuration options")

class ThreatDetectionRequest(BaseModel):
    text: str = Field(..., description="Text to analyze for threats")
    threat_types: Optional[List[str]] = Field(default=["all"], description="Types of threats to detect")
    sensitivity: Optional[str] = Field("medium", description="Detection sensitivity: low, medium, high")

class IntelligenceAlert(BaseModel):
    id: str
    type: str
    severity: str
    message: str
    data_point_id: Optional[str] = None
    created_at: str
    metadata: Optional[Dict[str, Any]] = None

# Advanced Intelligence Processing Service
class AdvancedIntelligenceService:
    def __init__(self):
        self.threat_patterns = {
            "cyber_threats": [
                "malware", "virus", "trojan", "ransomware", "phishing", "hack", "breach",
                "exploit", "vulnerability", "attack", "intrusion", "ddos", "botnet"
            ],
            "security_incidents": [
                "security breach", "data leak", "unauthorized access", "compromise",
                "incident", "alert", "threat", "suspicious activity"
            ],
            "financial_crimes": [
                "fraud", "money laundering", "embezzlement", "scam", "ponzi",
                "insider trading", "market manipulation"
            ],
            "terrorism": [
                "terrorist", "bomb", "explosive", "attack", "threat", "violence",
                "extremist", "radical", "plot"
            ]
        }
        self.processing_queue = asyncio.Queue(maxsize=1000)
        self.alert_queue = asyncio.Queue(maxsize=500)
    
    async def process_intelligence_batch(self, data_points: List[IntelligenceDataPoint], 
                                       user_id: int, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process multiple intelligence data points with advanced analytics"""
        try:
            start_time = time.time()
            processed_points = []
            alerts = []
            
            # Process each data point
            for point in data_points:
                try:
                    # Extract text content
                    text_content = self._extract_text_content(point)
                    
                    # Perform comprehensive analysis
                    analysis_result = await self._analyze_intelligence_data(text_content, point.type)
                    
                    # Store processed data
                    point_id = await self._store_intelligence_data(point, analysis_result, user_id)
                    
                    # Generate alerts if needed
                    point_alerts = await self._generate_alerts(analysis_result, point_id, point)
                    alerts.extend(point_alerts)
                    
                    processed_points.append({
                        "id": point_id,
                        "original_id": point.id,
                        "analysis": analysis_result,
                        "status": "processed",
                        "processing_time": time.time() - start_time
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to process intelligence point {point.id}: {e}")
                    processed_points.append({
                        "id": point.id or str(uuid.uuid4()),
                        "status": "failed",
                        "error": str(e)
                    })
            
            # Store alerts
            if alerts:
                await self._store_alerts(alerts, user_id)
            
            processing_time = time.time() - start_time
            
            return {
                "status": "completed",
                "data_points_processed": len(processed_points),
                "successful_points": len([p for p in processed_points if p["status"] == "processed"]),
                "failed_points": len([p for p in processed_points if p["status"] == "failed"]),
                "alerts_generated": len(alerts),
                "processing_time": processing_time,
                "processed_data": processed_points,
                "alerts": alerts
            }
            
        except Exception as e:
            logger.error(f"Intelligence batch processing failed: {e}")
            raise APIError(
                status_code=500,
                error_code="INTELLIGENCE_PROCESSING_FAILED",
                message="Intelligence batch processing failed",
                detail=str(e)
            )
    
    def _extract_text_content(self, point: IntelligenceDataPoint) -> str:
        """Extract text content from intelligence data point"""
        if point.content:
            return point.content
        elif point.title and point.description:
            return f"{point.title} {point.description}"
        elif point.title:
            return point.title
        elif point.description:
            return point.description
        else:
            return ""
    
    async def _analyze_intelligence_data(self, text: str, data_type: str) -> Dict[str, Any]:
        """Perform comprehensive analysis on intelligence data"""
        try:
            if not text or len(text.strip()) < 5:
                return {"confidence": 0.0, "analysis_type": "insufficient_data"}
            
            # Import ML pipeline
            from ..ml_pipeline import ml_pipeline
            
            # Perform parallel analysis
            analysis_tasks = [
                self._analyze_sentiment(text, ml_pipeline),
                self._extract_entities(text, ml_pipeline),
                self._classify_topics(text, ml_pipeline),
                self._detect_threats(text),
                self._analyze_urgency(text, data_type)
            ]
            
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Combine results
            analysis_result = {
                "sentiment": results[0] if not isinstance(results[0], Exception) else None,
                "entities": results[1] if not isinstance(results[1], Exception) else [],
                "topics": results[2] if not isinstance(results[2], Exception) else [],
                "threats": results[3] if not isinstance(results[3], Exception) else [],
                "urgency": results[4] if not isinstance(results[4], Exception) else "medium",
                "confidence": self._calculate_confidence(results),
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "data_type": data_type
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Intelligence analysis failed: {e}")
            return {
                "confidence": 0.0,
                "analysis_type": "failed",
                "error": str(e)
            }
    
    async def _analyze_sentiment(self, text: str, ml_pipeline) -> Dict[str, Any]:
        """Analyze sentiment of intelligence data"""
        try:
            sentiment = await ml_pipeline.analyze_sentiment(text)
            return sentiment
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return {"sentiment": "neutral", "confidence": 0.0, "score": 0.0}
    
    async def _extract_entities(self, text: str, ml_pipeline) -> List[Dict[str, Any]]:
        """Extract entities from intelligence data"""
        try:
            entities = await ml_pipeline.extract_entities(text)
            return entities
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []
    
    async def _classify_topics(self, text: str, ml_pipeline) -> List[Dict[str, Any]]:
        """Classify topics in intelligence data"""
        try:
            topics = await ml_pipeline.classify_topics(text)
            return topics
        except Exception as e:
            logger.warning(f"Topic classification failed: {e}")
            return []
    
    async def _detect_threats(self, text: str) -> List[Dict[str, Any]]:
        """Detect potential threats in intelligence data"""
        try:
            threats = []
            text_lower = text.lower()
            
            for threat_type, patterns in self.threat_patterns.items():
                matches = [pattern for pattern in patterns if pattern in text_lower]
                if matches:
                    threats.append({
                        "type": threat_type,
                        "patterns_matched": matches,
                        "confidence": len(matches) / len(patterns),
                        "severity": "high" if len(matches) > 3 else "medium" if len(matches) > 1 else "low"
                    })
            
            return threats
            
        except Exception as e:
            logger.warning(f"Threat detection failed: {e}")
            return []
    
    async def _analyze_urgency(self, text: str, data_type: str) -> str:
        """Analyze urgency level of intelligence data"""
        try:
            urgency_indicators = {
                "critical": ["urgent", "immediate", "critical", "emergency", "asap", "now"],
                "high": ["important", "priority", "alert", "warning", "threat"],
                "medium": ["update", "report", "information", "data"],
                "low": ["routine", "regular", "scheduled", "background"]
            }
            
            text_lower = text.lower()
            
            for level, indicators in urgency_indicators.items():
                if any(indicator in text_lower for indicator in indicators):
                    return level
            
            # Default based on data type
            if data_type in ["threat_intelligence", "security_alert", "incident_report"]:
                return "high"
            elif data_type in ["routine_report", "analysis", "summary"]:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            logger.warning(f"Urgency analysis failed: {e}")
            return "medium"
    
    def _calculate_confidence(self, results: List[Any]) -> float:
        """Calculate overall confidence score for analysis"""
        try:
            valid_results = [r for r in results if not isinstance(r, Exception) and r is not None]
            if not valid_results:
                return 0.0
            
            # Simple confidence calculation based on successful analyses
            return min(1.0, len(valid_results) / len(results))
        except Exception:
            return 0.5
    
    async def _store_intelligence_data(self, point: IntelligenceDataPoint, 
                                     analysis_result: Dict[str, Any], user_id: int) -> str:
        """Store intelligence data in database"""
        try:
            from .. import db
            
            point_id = str(uuid.uuid4())
            
            with db.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO intelligence_data 
                        (id, user_id, data_type, raw_data, analysis_result, 
                         confidence_score, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        point_id,
                        user_id,
                        point.type,
                        json.dumps(point.dict()),
                        json.dumps(analysis_result),
                        analysis_result.get("confidence", 0.0),
                        datetime.utcnow()
                    ))
            
            return point_id
            
        except Exception as e:
            logger.error(f"Failed to store intelligence data: {e}")
            raise APIError(
                status_code=500,
                error_code="DATA_STORAGE_FAILED",
                message="Failed to store intelligence data",
                detail=str(e)
            )
    
    async def _generate_alerts(self, analysis_result: Dict[str, Any], 
                              point_id: str, point: IntelligenceDataPoint) -> List[Dict[str, Any]]:
        """Generate alerts based on analysis results"""
        try:
            alerts = []
            
            # High threat alert
            threats = analysis_result.get("threats", [])
            high_threats = [t for t in threats if t.get("severity") == "high"]
            if high_threats:
                alerts.append({
                    "id": str(uuid.uuid4()),
                    "type": "high_threat_detected",
                    "severity": "high",
                    "message": f"High severity threat detected: {', '.join([t['type'] for t in high_threats])}",
                    "data_point_id": point_id,
                    "metadata": {"threats": high_threats}
                })
            
            # Negative sentiment alert
            sentiment = analysis_result.get("sentiment", {})
            if sentiment.get("score", 0) < -0.7:
                alerts.append({
                    "id": str(uuid.uuid4()),
                    "type": "negative_sentiment",
                    "severity": "medium",
                    "message": f"Highly negative sentiment detected (score: {sentiment.get('score', 0):.2f})",
                    "data_point_id": point_id,
                    "metadata": {"sentiment": sentiment}
                })
            
            # High urgency alert
            if analysis_result.get("urgency") == "critical":
                alerts.append({
                    "id": str(uuid.uuid4()),
                    "type": "critical_urgency",
                    "severity": "critical",
                    "message": "Critical urgency intelligence data requires immediate attention",
                    "data_point_id": point_id,
                    "metadata": {"urgency": analysis_result.get("urgency")}
                })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Alert generation failed: {e}")
            return []
    
    async def _store_alerts(self, alerts: List[Dict[str, Any]], user_id: int):
        """Store alerts in database"""
        try:
            from .. import db
            
            with db.get_conn() as conn:
                with conn.cursor() as cur:
                    for alert in alerts:
                        cur.execute("""
                            INSERT INTO intelligence_insights 
                            (id, user_id, insight_type, message, severity, 
                             data_point_id, insight_data, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            alert["id"],
                            user_id,
                            alert["type"],
                            alert["message"],
                            alert["severity"],
                            alert["data_point_id"],
                            json.dumps(alert.get("metadata", {})),
                            datetime.utcnow()
                        ))
            
        except Exception as e:
            logger.error(f"Failed to store alerts: {e}")

# Initialize service
intelligence_service = AdvancedIntelligenceService()

@router.post(
    "/process",
    summary="Advanced Intelligence Processing",
    description="Process intelligence data with comprehensive analysis, threat detection, and real-time alerts",
    responses={
        200: {"description": "Intelligence data processed successfully"},
        400: {"description": "Invalid input", "model": ErrorResponse},
        500: {"description": "Processing failed", "model": ErrorResponse}
    }
)
async def process_intelligence_data_advanced(
    request: IntelligenceProcessingRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Advanced intelligence data processing with comprehensive analysis"""
    try:
        # Validate input
        if not request.data:
            ErrorHandler.raise_bad_request("No intelligence data points provided")
        
        if len(request.data) > 100:
            ErrorHandler.raise_bad_request("Too many data points (max 100 per request)")
        
        # Process intelligence data
        result = await intelligence_service.process_intelligence_batch(
            data_points=request.data,
            user_id=current_user.id,
            options=request.processing_options
        )
        
        return result
        
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Intelligence processing failed: {e}")
        ErrorHandler.raise_internal_server_error("Intelligence processing failed", str(e))

@router.post(
    "/process/batch",
    summary="Batch Intelligence Processing",
    description="Process large batches of intelligence data with parallel processing",
    responses={
        200: {"description": "Batch processing completed successfully"},
        400: {"description": "Invalid input", "model": ErrorResponse},
        500: {"description": "Batch processing failed", "model": ErrorResponse}
    }
)
async def process_intelligence_batch(
    request: IntelligenceProcessingRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    """Process large batches of intelligence data asynchronously"""
    try:
        # Validate input
        if not request.data:
            ErrorHandler.raise_bad_request("No intelligence data points provided")
        
        if len(request.data) > 1000:
            ErrorHandler.raise_bad_request("Too many data points for batch processing (max 1000)")
        
        # Process in background for large batches
        if len(request.data) > 50:
            background_tasks.add_task(
                intelligence_service.process_intelligence_batch,
                request.data,
                current_user.id,
                request.processing_options
            )
            
            return {
                "status": "queued",
                "message": f"Processing {len(request.data)} data points in background",
                "estimated_completion_time": f"{len(request.data) * 2} seconds"
            }
        else:
            # Process immediately for small batches
            result = await intelligence_service.process_intelligence_batch(
                data_points=request.data,
                user_id=current_user.id,
                options=request.processing_options
            )
            return result
        
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Batch intelligence processing failed: {e}")
        ErrorHandler.raise_internal_server_error("Batch intelligence processing failed", str(e))

@router.post(
    "/threats/detect",
    summary="Threat Detection",
    description="Detect potential threats in text using advanced pattern matching and ML analysis",
    responses={
        200: {"description": "Threat detection completed successfully"},
        400: {"description": "Invalid input", "model": ErrorResponse},
        500: {"description": "Threat detection failed", "model": ErrorResponse}
    }
)
async def detect_threats(
    request: ThreatDetectionRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Detect potential threats in intelligence data"""
    try:
        # Validate input
        if not request.text or len(request.text.strip()) < 5:
            ErrorHandler.raise_bad_request("Text must be at least 5 characters long")
        
        if len(request.text) > 10000:
            ErrorHandler.raise_bad_request("Text too long (max 10,000 characters)")
        
        # Detect threats
        threats = await intelligence_service._detect_threats(request.text)
        
        # Filter by threat types if specified
        if "all" not in request.threat_types:
            threats = [t for t in threats if t["type"] in request.threat_types]
        
        # Apply sensitivity filter
        if request.sensitivity == "high":
            threats = [t for t in threats if t["confidence"] > 0.3]
        elif request.sensitivity == "low":
            threats = [t for t in threats if t["confidence"] > 0.7]
        else:  # medium
            threats = [t for t in threats if t["confidence"] > 0.5]
        
        return {
            "threats_detected": len(threats),
            "threats": threats,
            "sensitivity": request.sensitivity,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Threat detection failed: {e}")
        ErrorHandler.raise_internal_server_error("Threat detection failed", str(e))

@router.post(
    "/dashboard/update",
    summary="Enhanced Intelligence Dashboard Update",
    description="Update intelligence dashboard with comprehensive metrics, trends, and real-time insights",
    responses={
        200: {"description": "Dashboard updated successfully"},
        400: {"description": "Invalid input", "model": ErrorResponse},
        500: {"description": "Dashboard update failed", "model": ErrorResponse}
    }
)
async def update_intelligence_dashboard_enhanced(
    dashboard_id: str = Query(..., description="Dashboard ID to update"),
    hours_back: int = Query(24, description="Hours of data to include in update"),
    current_user: User = Depends(get_current_active_user)
):
    """Enhanced intelligence dashboard update with comprehensive analytics"""
    try:
        from .. import db
        
        # Validate input
        if not dashboard_id:
            ErrorHandler.raise_bad_request("Dashboard ID is required")
        
        if hours_back < 1 or hours_back > 168:  # Max 1 week
            ErrorHandler.raise_bad_request("Hours back must be between 1 and 168")
        
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)
        
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                # Get comprehensive intelligence metrics
                cur.execute("""
                    SELECT 
                        data_type,
                        COUNT(*) as count,
                        AVG(confidence_score) as avg_confidence,
                        MAX(created_at) as last_updated,
                        COUNT(CASE WHEN confidence_score > 0.8 THEN 1 END) as high_confidence_count
                    FROM intelligence_data
                    WHERE user_id = %s 
                    AND created_at >= %s
                    GROUP BY data_type
                    ORDER BY count DESC
                """, (current_user.id, start_time))
                
                data_type_stats = cur.fetchall()
                
                # Get threat analysis
                cur.execute("""
                    SELECT 
                        insight_type,
                        COUNT(*) as count,
                        MAX(created_at) as last_insight,
                        COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical_count,
                        COUNT(CASE WHEN severity = 'high' THEN 1 END) as high_count
                    FROM intelligence_insights
                    WHERE user_id = %s 
                    AND created_at >= %s
                    GROUP BY insight_type
                    ORDER BY count DESC
                """, (current_user.id, start_time))
                
                insight_stats = cur.fetchall()
                
                # Get top entities with importance scores
                cur.execute("""
                    SELECT 
                        jsonb_array_elements(analysis_result->'entities')->>'text' as entity,
                        COUNT(*) as frequency,
                        AVG((jsonb_array_elements(analysis_result->'entities')->>'importance')::float) as avg_importance
                    FROM intelligence_data
                    WHERE user_id = %s 
                    AND analysis_result IS NOT NULL
                    AND created_at >= %s
                    GROUP BY entity
                    HAVING COUNT(*) > 1
                    ORDER BY frequency DESC, avg_importance DESC
                    LIMIT 15
                """, (current_user.id, start_time))
                
                top_entities = cur.fetchall()
                
                # Get sentiment trends with confidence
                cur.execute("""
                    SELECT 
                        DATE(created_at) as date,
                        AVG((analysis_result->'sentiment'->>'score')::float) as avg_sentiment,
                        COUNT(*) as data_points,
                        AVG(confidence_score) as avg_confidence
                    FROM intelligence_data
                    WHERE user_id = %s 
                    AND analysis_result IS NOT NULL
                    AND created_at >= %s
                    GROUP BY DATE(created_at)
                    ORDER BY date DESC
                    LIMIT 14
                """, (current_user.id, start_time))
                
                sentiment_trends = cur.fetchall()
                
                # Get urgency distribution
                cur.execute("""
                    SELECT 
                        analysis_result->>'urgency' as urgency_level,
                        COUNT(*) as count
                    FROM intelligence_data
                    WHERE user_id = %s 
                    AND analysis_result IS NOT NULL
                    AND created_at >= %s
                    GROUP BY analysis_result->>'urgency'
                """, (current_user.id, start_time))
                
                urgency_distribution = cur.fetchall()
                
                # Get recent high-priority alerts
                cur.execute("""
                    SELECT 
                        insight_type,
                        message,
                        severity,
                        created_at
                    FROM intelligence_insights
                    WHERE user_id = %s 
                    AND created_at >= %s
                    AND severity IN ('high', 'critical')
                    ORDER BY created_at DESC
                    LIMIT 10
                """, (current_user.id, start_time))
                
                recent_alerts = cur.fetchall()
        
        # Prepare enhanced dashboard metrics
        metrics_updated = []
        
        # Data type metrics with confidence analysis
        for row in data_type_stats:
            metrics_updated.append({
                "metric": f"data_type_{row[0]}",
                "value": row[1],
                "avg_confidence": float(row[2]) if row[2] else 0.0,
                "high_confidence_ratio": float(row[4]) / row[1] if row[1] > 0 else 0.0,
                "last_updated": row[3].isoformat() if row[3] else None
            })
        
        # Insight metrics with severity breakdown
        for row in insight_stats:
            metrics_updated.append({
                "metric": f"insight_{row[0]}",
                "value": row[1],
                "critical_count": row[3],
                "high_count": row[4],
                "last_insight": row[2].isoformat() if row[2] else None
            })
        
        # Enhanced top entities with importance
        top_entities_list = [
            {
                "entity": row[0], 
                "frequency": row[1],
                "avg_importance": float(row[2]) if row[2] else 0.0
            } 
            for row in top_entities
        ]
        
        # Enhanced sentiment trends with confidence
        sentiment_data = [
            {
                "date": row[0].isoformat(), 
                "avg_sentiment": float(row[1]) if row[1] else 0.0,
                "data_points": row[2],
                "avg_confidence": float(row[3]) if row[3] else 0.0
            } 
            for row in sentiment_trends
        ]
        
        # Urgency distribution
        urgency_data = [
            {
                "urgency_level": row[0] or "unknown",
                "count": row[1]
            }
            for row in urgency_distribution
        ]
        
        # Recent alerts
        alerts_list = [
            {
                "type": row[0],
                "message": row[1],
                "severity": row[2],
                "created_at": row[3].isoformat() if row[3] else None
            }
            for row in recent_alerts
        ]
        
        # Store enhanced dashboard update
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO dashboard_updates 
                    (dashboard_id, user_id, update_type, metrics_data, 
                     top_entities, sentiment_trends, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    dashboard_id,
                    current_user.id,
                    "intelligence_update_enhanced",
                    json.dumps({
                        "metrics": metrics_updated,
                        "urgency_distribution": urgency_data,
                        "recent_alerts": alerts_list,
                        "analysis_period_hours": hours_back
                    }),
                    json.dumps(top_entities_list),
                    json.dumps(sentiment_data),
                    datetime.utcnow()
                ))
        
        result = {
            "status": "updated",
            "dashboard_id": dashboard_id,
            "updated_at": datetime.utcnow().isoformat(),
            "analysis_period_hours": hours_back,
            "metrics_updated": metrics_updated,
            "top_entities": top_entities_list,
            "sentiment_trends": sentiment_data,
            "urgency_distribution": urgency_data,
            "recent_alerts": alerts_list,
            "data_type_stats": [
                {
                    "type": row[0], 
                    "count": row[1], 
                    "avg_confidence": float(row[2]) if row[2] else 0.0,
                    "high_confidence_ratio": float(row[4]) / row[1] if row[1] > 0 else 0.0
                } 
                for row in data_type_stats
            ],
            "insight_stats": [
                {
                    "type": row[0], 
                    "count": row[1],
                    "critical_count": row[3],
                    "high_count": row[4]
                } 
                for row in insight_stats
            ]
        }
        
        return result
        
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Enhanced dashboard update failed: {e}")
        ErrorHandler.raise_internal_server_error("Enhanced dashboard update failed", str(e))

@router.get(
    "/alerts",
    summary="Get Intelligence Alerts",
    description="Retrieve intelligence alerts with filtering and pagination",
    responses={
        200: {"description": "Alerts retrieved successfully"},
        500: {"description": "Failed to retrieve alerts", "model": ErrorResponse}
    }
)
async def get_intelligence_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity: low, medium, high, critical"),
    insight_type: Optional[str] = Query(None, description="Filter by insight type"),
    limit: int = Query(50, description="Maximum number of alerts to return"),
    offset: int = Query(0, description="Number of alerts to skip"),
    current_user: User = Depends(get_current_active_user)
):
    """Get intelligence alerts with filtering and pagination"""
    try:
        from .. import db
        
        # Build query with filters
        query = """
            SELECT 
                id, insight_type, message, severity, 
                data_point_id, created_at, insight_data
            FROM intelligence_insights
            WHERE user_id = %s
        """
        params = [current_user.id]
        
        if severity:
            query += " AND severity = %s"
            params.append(severity)
        
        if insight_type:
            query += " AND insight_type = %s"
            params.append(insight_type)
        
        query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                alerts = cur.fetchall()
                
                # Get total count for pagination
                count_query = """
                    SELECT COUNT(*) FROM intelligence_insights
                    WHERE user_id = %s
                """
                count_params = [current_user.id]
                
                if severity:
                    count_query += " AND severity = %s"
                    count_params.append(severity)
                
                if insight_type:
                    count_query += " AND insight_type = %s"
                    count_params.append(insight_type)
                
                cur.execute(count_query, count_params)
                total_count = cur.fetchone()[0]
        
        alerts_list = [
            {
                "id": row[0],
                "type": row[1],
                "message": row[2],
                "severity": row[3],
                "data_point_id": row[4],
                "created_at": row[5].isoformat() if row[5] else None,
                "metadata": json.loads(row[6]) if row[6] else {}
            }
            for row in alerts
        ]
        
        return {
            "alerts": alerts_list,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(alerts) < total_count
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve intelligence alerts: {e}")
        ErrorHandler.raise_internal_server_error("Failed to retrieve intelligence alerts", str(e))

@router.get(
    "/analytics/summary",
    summary="Intelligence Analytics Summary",
    description="Get comprehensive intelligence analytics summary",
    responses={
        200: {"description": "Analytics summary retrieved successfully"},
        500: {"description": "Failed to retrieve analytics", "model": ErrorResponse}
    }
)
async def get_intelligence_analytics(
    days_back: int = Query(7, description="Number of days to include in analysis"),
    current_user: User = Depends(get_current_active_user)
):
    """Get comprehensive intelligence analytics summary"""
    try:
        from .. import db
        
        if days_back < 1 or days_back > 30:
            ErrorHandler.raise_bad_request("Days back must be between 1 and 30")
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days_back)
        
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                # Overall statistics
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_data_points,
                        AVG(confidence_score) as avg_confidence,
                        COUNT(CASE WHEN confidence_score > 0.8 THEN 1 END) as high_confidence_count,
                        COUNT(DISTINCT data_type) as unique_data_types
                    FROM intelligence_data
                    WHERE user_id = %s AND created_at >= %s
                """, (current_user.id, start_time))
                
                overall_stats = cur.fetchone()
                
                # Threat analysis summary
                cur.execute("""
                    SELECT 
                        severity,
                        COUNT(*) as count
                    FROM intelligence_insights
                    WHERE user_id = %s AND created_at >= %s
                    GROUP BY severity
                """, (current_user.id, start_time))
                
                threat_summary = cur.fetchall()
                
                # Data processing trends
                cur.execute("""
                    SELECT 
                        DATE(created_at) as date,
                        COUNT(*) as data_points,
                        AVG(confidence_score) as avg_confidence
                    FROM intelligence_data
                    WHERE user_id = %s AND created_at >= %s
                    GROUP BY DATE(created_at)
                    ORDER BY date DESC
                """, (current_user.id, start_time))
                
                processing_trends = cur.fetchall()
                
                # Top data sources
                cur.execute("""
                    SELECT 
                        data_type,
                        COUNT(*) as count,
                        AVG(confidence_score) as avg_confidence
                    FROM intelligence_data
                    WHERE user_id = %s AND created_at >= %s
                    GROUP BY data_type
                    ORDER BY count DESC
                    LIMIT 10
                """, (current_user.id, start_time))
                
                top_sources = cur.fetchall()
        
        return {
            "analysis_period_days": days_back,
            "overall_statistics": {
                "total_data_points": overall_stats[0] or 0,
                "average_confidence": float(overall_stats[1]) if overall_stats[1] else 0.0,
                "high_confidence_ratio": float(overall_stats[2]) / overall_stats[0] if overall_stats[0] > 0 else 0.0,
                "unique_data_types": overall_stats[3] or 0
            },
            "threat_summary": {
                row[0]: row[1] for row in threat_summary
            },
            "processing_trends": [
                {
                    "date": row[0].isoformat(),
                    "data_points": row[1],
                    "avg_confidence": float(row[2]) if row[2] else 0.0
                }
                for row in processing_trends
            ],
            "top_data_sources": [
                {
                    "data_type": row[0],
                    "count": row[1],
                    "avg_confidence": float(row[2]) if row[2] else 0.0
                }
                for row in top_sources
            ],
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve intelligence analytics: {e}")
        ErrorHandler.raise_internal_server_error("Failed to retrieve intelligence analytics", str(e))
