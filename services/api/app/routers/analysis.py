"""
Enhanced Analysis router
Handles stance-sentiment-bias analysis, alerts, and dashboard with advanced features
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from fastapi import APIRouter, Depends, Query, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import numpy as np
import pandas as pd

from ..auth import get_current_active_user, User
from ..schemas import (
    StanceSentimentAnalysisRequest, StanceSentimentAnalysisResponse,
    ErrorResponse, BatchAnalysisRequest, AnalysisAlert
)
from ..enhanced_error_handling import ErrorHandler, APIError
from ..cache import cache
from ..monitoring import monitoring_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analysis", tags=["Analysis"])

# Performance monitoring
class AnalysisPerformanceMonitor:
    def __init__(self):
        self.metrics = {
            "total_analyses": 0,
            "avg_processing_time": 0.0,
            "cache_hit_rate": 0.0,
            "error_rate": 0.0,
            "concurrent_analyses": 0
        }
        self.active_analyses = 0
    
    def start_analysis(self):
        self.active_analyses += 1
        self.metrics["concurrent_analyses"] = self.active_analyses
    
    def end_analysis(self, processing_time: float, from_cache: bool = False):
        self.active_analyses -= 1
        self.metrics["concurrent_analyses"] = self.active_analyses
        self.metrics["total_analyses"] += 1
        
        # Update average processing time
        total = self.metrics["total_analyses"]
        current_avg = self.metrics["avg_processing_time"]
        self.metrics["avg_processing_time"] = (current_avg * (total - 1) + processing_time) / total
        
        # Update cache hit rate
        if from_cache:
            cache_hits = self.metrics["cache_hit_rate"] * (total - 1) + 1
            self.metrics["cache_hit_rate"] = cache_hits / total
        else:
            cache_hits = self.metrics["cache_hit_rate"] * (total - 1)
            self.metrics["cache_hit_rate"] = cache_hits / total

performance_monitor = AnalysisPerformanceMonitor()

# Advanced analysis service
class AdvancedAnalysisService:
    def __init__(self):
        self.batch_queue = asyncio.Queue(maxsize=1000)
        self.processing_batch = False
    
    async def analyze_text_advanced(
        self, 
        text: str, 
        article_id: Optional[str] = None,
        source_id: Optional[str] = None,
        topic: Optional[str] = None,
        analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Advanced text analysis with multiple ML models and caching"""
        start_time = time.time()
        performance_monitor.start_analysis()
        
        try:
            # Check cache first
            cache_key = f"analysis:{hash(text)}:{analysis_type}"
            cached_result = await cache.get(cache_key)
            if cached_result:
                performance_monitor.end_analysis(time.time() - start_time, from_cache=True)
                return cached_result
            
            # Import analysis modules
            from ..stance_sentiment_bias_analyzer import stance_sentiment_bias_analyzer
            from ..ml_pipeline import ml_pipeline
            
            # Perform comprehensive analysis
            analysis_tasks = []
            
            if analysis_type in ["comprehensive", "sentiment"]:
                analysis_tasks.append(self._analyze_sentiment(text, ml_pipeline))
            
            if analysis_type in ["comprehensive", "stance"]:
                analysis_tasks.append(self._analyze_stance(text, stance_sentiment_bias_analyzer))
            
            if analysis_type in ["comprehensive", "bias"]:
                analysis_tasks.append(self._analyze_bias(text, stance_sentiment_bias_analyzer))
            
            if analysis_type in ["comprehensive", "entities"]:
                analysis_tasks.append(self._extract_entities(text, ml_pipeline))
            
            if analysis_type in ["comprehensive", "topics"]:
                analysis_tasks.append(self._classify_topics(text, ml_pipeline))
            
            # Execute all analyses in parallel
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Combine results
            combined_result = self._combine_analysis_results(results, text, article_id, source_id, topic)
            
            # Cache result for 1 hour
            await cache.set(cache_key, combined_result, ttl=3600)
            
            performance_monitor.end_analysis(time.time() - start_time, from_cache=False)
            return combined_result
            
        except Exception as e:
            performance_monitor.end_analysis(time.time() - start_time, from_cache=False)
            logger.error(f"Advanced analysis failed: {e}")
            raise APIError(
                status_code=500,
                error_code="ANALYSIS_FAILED",
                message="Advanced analysis failed",
                detail=str(e)
            )
    
    async def _analyze_sentiment(self, text: str, ml_pipeline) -> Dict[str, Any]:
        """Analyze sentiment with confidence scoring"""
        try:
            sentiment_result = await ml_pipeline.analyze_sentiment(text)
            return {
                "type": "sentiment",
                "scores": sentiment_result.get("scores", {}),
                "confidence": sentiment_result.get("confidence", 0.0),
                "model": "twitter-roberta-base-sentiment"
            }
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return {"type": "sentiment", "error": str(e)}
    
    async def _analyze_stance(self, text: str, analyzer) -> Dict[str, Any]:
        """Analyze stance with political orientation detection"""
        try:
            stance_result = await analyzer.analyze_stance(text)
            return {
                "type": "stance",
                "scores": stance_result.get("stance_scores", {}),
                "confidence": stance_result.get("confidence", 0.0),
                "model": "stance-detection"
            }
        except Exception as e:
            logger.warning(f"Stance analysis failed: {e}")
            return {"type": "stance", "error": str(e)}
    
    async def _analyze_bias(self, text: str, analyzer) -> Dict[str, Any]:
        """Analyze political bias with detailed scoring"""
        try:
            bias_result = await analyzer.analyze_bias(text)
            return {
                "type": "bias",
                "scores": bias_result.get("bias_scores", {}),
                "confidence": bias_result.get("confidence", 0.0),
                "model": "bias-detection"
            }
        except Exception as e:
            logger.warning(f"Bias analysis failed: {e}")
            return {"type": "bias", "error": str(e)}
    
    async def _extract_entities(self, text: str, ml_pipeline) -> Dict[str, Any]:
        """Extract named entities with importance scoring"""
        try:
            entities_result = await ml_pipeline.extract_entities(text)
            return {
                "type": "entities",
                "entities": entities_result.get("entities", []),
                "confidence": entities_result.get("confidence", 0.0),
                "model": "entity-extraction"
            }
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return {"type": "entities", "error": str(e)}
    
    async def _classify_topics(self, text: str, ml_pipeline) -> Dict[str, Any]:
        """Classify topics with confidence scoring"""
        try:
            topics_result = await ml_pipeline.classify_topics(text)
            return {
                "type": "topics",
                "topics": topics_result.get("topics", []),
                "confidence": topics_result.get("confidence", 0.0),
                "model": "topic-classification"
            }
        except Exception as e:
            logger.warning(f"Topic classification failed: {e}")
            return {"type": "topics", "error": str(e)}
    
    def _combine_analysis_results(
        self, 
        results: List[Any], 
        text: str, 
        article_id: Optional[str],
        source_id: Optional[str],
        topic: Optional[str]
    ) -> Dict[str, Any]:
        """Combine multiple analysis results into comprehensive output"""
        combined = {
            "text": text,
            "article_id": article_id,
            "source_id": source_id,
            "topic": topic,
            "analysis_type": "comprehensive",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {},
            "summary": {},
            "confidence": 0.0,
            "risk_score": 0.0
        }
        
        # Process each analysis result
        for result in results:
            if isinstance(result, dict) and "type" in result:
                analysis_type = result["type"]
                combined["components"][analysis_type] = result
        
        # Calculate overall confidence and risk score
        confidences = [r.get("confidence", 0.0) for r in results if isinstance(r, dict) and "confidence" in r]
        combined["confidence"] = np.mean(confidences) if confidences else 0.0
        
        # Calculate risk score based on bias and sentiment
        risk_factors = []
        if "bias" in combined["components"]:
            bias_scores = combined["components"]["bias"].get("scores", {})
            if bias_scores:
                max_bias = max(bias_scores.values()) if bias_scores else 0
                risk_factors.append(max_bias)
        
        if "sentiment" in combined["components"]:
            sentiment_scores = combined["components"]["sentiment"].get("scores", {})
            if sentiment_scores:
                negative_score = sentiment_scores.get("negative", 0)
                risk_factors.append(negative_score)
        
        combined["risk_score"] = np.mean(risk_factors) if risk_factors else 0.0
        
        # Generate summary
        combined["summary"] = self._generate_analysis_summary(combined["components"])
        
        return combined
    
    def _generate_analysis_summary(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Generate human-readable summary of analysis results"""
        summary = {
            "overall_sentiment": "neutral",
            "political_bias": "center",
            "key_entities": [],
            "main_topics": [],
            "risk_level": "low"
        }
        
        # Sentiment summary
        if "sentiment" in components:
            sentiment_scores = components["sentiment"].get("scores", {})
            if sentiment_scores:
                max_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])
                summary["overall_sentiment"] = max_sentiment[0]
        
        # Bias summary
        if "bias" in components:
            bias_scores = components["bias"].get("scores", {})
            if bias_scores:
                max_bias = max(bias_scores.items(), key=lambda x: x[1])
                summary["political_bias"] = max_bias[0]
        
        # Entities summary
        if "entities" in components:
            entities = components["entities"].get("entities", [])
            summary["key_entities"] = [e.get("text", "") for e in entities[:5]]
        
        # Topics summary
        if "topics" in components:
            topics = components["topics"].get("topics", [])
            summary["main_topics"] = [t.get("name", "") for t in topics[:3]]
        
        # Risk level
        if "bias" in components or "sentiment" in components:
            risk_score = 0
            if "bias" in components:
                bias_scores = components["bias"].get("scores", {})
                if bias_scores:
                    risk_score += max(bias_scores.values())
            if "sentiment" in components:
                sentiment_scores = components["sentiment"].get("scores", {})
                if sentiment_scores:
                    risk_score += sentiment_scores.get("negative", 0)
            
            if risk_score > 0.7:
                summary["risk_level"] = "high"
            elif risk_score > 0.4:
                summary["risk_level"] = "medium"
            else:
                summary["risk_level"] = "low"
        
        return summary

analysis_service = AdvancedAnalysisService()

# Real-time alert system
class AnalysisAlertSystem:
    def __init__(self):
        self.alert_thresholds = {
            "high_bias": 0.8,
            "negative_sentiment": 0.7,
            "low_confidence": 0.3,
            "high_risk": 0.8
        }
    
    async def check_alerts(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check analysis result for alert conditions"""
        alerts = []
        
        # Check bias alerts
        if "bias" in analysis_result.get("components", {}):
            bias_scores = analysis_result["components"]["bias"].get("scores", {})
            for bias_type, score in bias_scores.items():
                if score > self.alert_thresholds["high_bias"]:
                    alerts.append({
                        "type": "high_bias",
                        "severity": "high",
                        "message": f"High {bias_type} bias detected: {score:.2f}",
                        "data": {"bias_type": bias_type, "score": score}
                    })
        
        # Check sentiment alerts
        if "sentiment" in analysis_result.get("components", {}):
            sentiment_scores = analysis_result["components"]["sentiment"].get("scores", {})
            if sentiment_scores.get("negative", 0) > self.alert_thresholds["negative_sentiment"]:
                alerts.append({
                    "type": "negative_sentiment",
                    "severity": "medium",
                    "message": f"High negative sentiment detected: {sentiment_scores['negative']:.2f}",
                    "data": {"sentiment_scores": sentiment_scores}
                })
        
        # Check confidence alerts
        if analysis_result.get("confidence", 0) < self.alert_thresholds["low_confidence"]:
            alerts.append({
                "type": "low_confidence",
                "severity": "low",
                "message": f"Low analysis confidence: {analysis_result['confidence']:.2f}",
                "data": {"confidence": analysis_result["confidence"]}
            })
        
        # Check risk alerts
        if analysis_result.get("risk_score", 0) > self.alert_thresholds["high_risk"]:
            alerts.append({
                "type": "high_risk",
                "severity": "high",
                "message": f"High risk content detected: {analysis_result['risk_score']:.2f}",
                "data": {"risk_score": analysis_result["risk_score"]}
            })
        
        return alerts

alert_system = AnalysisAlertSystem()

@router.post(
    "/analyze",
    response_model=StanceSentimentAnalysisResponse,
    summary="Advanced Text Analysis",
    description="Perform comprehensive analysis with caching, performance monitoring, and real-time alerts",
    responses={
        200: {"description": "Analysis completed successfully"},
        400: {"description": "Invalid input", "model": ErrorResponse},
        500: {"description": "Analysis failed", "model": ErrorResponse}
    }
)
async def analyze_text_advanced(
    request: StanceSentimentAnalysisRequest,
    analysis_type: str = Query(default="comprehensive", description="Type of analysis to perform"),
    current_user: User = Depends(get_current_active_user)
):
    """Advanced text analysis with multiple ML models"""
    try:
        # Validate input
        if not request.text or len(request.text.strip()) < 10:
            ErrorHandler.raise_bad_request("Text must be at least 10 characters long")
        
        if len(request.text) > 50000:
            ErrorHandler.raise_bad_request("Text too long (max 50,000 characters)")
        
        # Perform analysis
        result = await analysis_service.analyze_text_advanced(
            text=request.text,
            article_id=request.article_id,
            source_id=request.source_id,
            topic=request.topic,
            analysis_type=analysis_type
        )
        
        # Check for alerts
        alerts = await alert_system.check_alerts(result)
        if alerts:
            result["alerts"] = alerts
            # Store alerts in database
            await _store_alerts(alerts, request.article_id, request.source_id, current_user.id)
        
        # Store analysis in database
        if result and request.article_id:
            await _store_analysis_result(result, request.article_id, request.source_id, request.topic)
        
        return result
        
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        ErrorHandler.raise_internal_server_error("Analysis failed", str(e))

@router.post(
    "/analyze/batch",
    summary="Batch Text Analysis",
    description="Analyze multiple texts in parallel with performance optimization",
    responses={
        200: {"description": "Batch analysis completed successfully"},
        400: {"description": "Invalid input", "model": ErrorResponse},
        500: {"description": "Batch analysis failed", "model": ErrorResponse}
    }
)
async def analyze_batch(
    request: BatchAnalysisRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Batch analysis for multiple texts"""
    try:
        if not request.texts or len(request.texts) == 0:
            ErrorHandler.raise_bad_request("No texts provided")
        
        if len(request.texts) > 100:
            ErrorHandler.raise_bad_request("Too many texts (max 100)")
        
        # Process texts in parallel
        tasks = []
        for i, text in enumerate(request.texts):
            task = analysis_service.analyze_text_advanced(
                text=text,
                article_id=request.article_ids[i] if request.article_ids and i < len(request.article_ids) else None,
                source_id=request.source_id,
                topic=request.topic,
                analysis_type=request.analysis_type or "comprehensive"
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_results = []
        failed_results = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_results.append({
                    "index": i,
                    "error": str(result)
                })
            else:
                successful_results.append(result)
        
        return {
            "total_texts": len(request.texts),
            "successful_analyses": len(successful_results),
            "failed_analyses": len(failed_results),
            "results": successful_results,
            "failures": failed_results,
            "processing_time": time.time() - time.time()
        }
        
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        ErrorHandler.raise_internal_server_error("Batch analysis failed", str(e))

@router.get(
    "/source/{source_id}",
    summary="Enhanced Source Analysis",
    description="Get comprehensive source analysis with advanced metrics and trends",
    responses={
        200: {"description": "Source analysis retrieved successfully"},
        404: {"description": "Source not found", "model": ErrorResponse},
        500: {"description": "Analysis retrieval failed", "model": ErrorResponse}
    }
)
async def get_source_analysis_enhanced(
    source_id: str,
    days: int = Query(default=7, description="Number of days to analyze", ge=1, le=90),
    include_trends: bool = Query(default=True, description="Include trend analysis"),
    include_alerts: bool = Query(default=True, description="Include recent alerts"),
    current_user: User = Depends(get_current_active_user)
):
    """Enhanced source analysis with advanced metrics"""
    try:
        from .. import db
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                # Get comprehensive source statistics
                cur.execute("""
                    SELECT 
                        COUNT(DISTINCT aa.article_id) as total_articles,
                        AVG(aa.confidence_avg) as avg_confidence,
                        AVG((aa.sentiment_distribution->>'positive')::float) as avg_sentiment_positive,
                        AVG((aa.sentiment_distribution->>'negative')::float) as avg_sentiment_negative,
                        AVG((aa.sentiment_distribution->>'neutral')::float) as avg_sentiment_neutral,
                        AVG((aa.bias_scores->>'left')::float) as avg_bias_left,
                        AVG((aa.bias_scores->>'right')::float) as avg_bias_right,
                        AVG((aa.bias_scores->>'center')::float) as avg_bias_center,
                        AVG(aa.risk_score) as avg_risk_score,
                        STDDEV(aa.confidence_avg) as confidence_stddev,
                        STDDEV((aa.sentiment_distribution->>'positive')::float) as sentiment_stddev
                    FROM aggregated_analysis aa
                    JOIN articles a ON aa.article_id = a.id
                    WHERE aa.source_id = %s 
                    AND aa.created_at >= %s 
                    AND aa.created_at <= %s
                """, (source_id, start_date, end_date))
                
                summary_row = cur.fetchone()
                
                # Get time series data with more granular metrics
                cur.execute("""
                    SELECT 
                        DATE(aa.created_at) as date,
                        COUNT(DISTINCT aa.article_id) as articles_count,
                        AVG(aa.confidence_avg) as avg_confidence,
                        AVG((aa.sentiment_distribution->>'positive')::float) as avg_sentiment_positive,
                        AVG((aa.sentiment_distribution->>'negative')::float) as avg_sentiment_negative,
                        AVG((aa.bias_scores->>'center')::float) as avg_bias_center,
                        AVG(aa.risk_score) as avg_risk_score,
                        COUNT(CASE WHEN aa.risk_score > 0.7 THEN 1 END) as high_risk_articles
                    FROM aggregated_analysis aa
                    JOIN articles a ON aa.article_id = a.id
                    WHERE aa.source_id = %s 
                    AND aa.created_at >= %s 
                    AND aa.created_at <= %s
                    GROUP BY DATE(aa.created_at)
                    ORDER BY date
                """, (source_id, start_date, end_date))
                
                time_series = []
                for row in cur.fetchall():
                    time_series.append({
                        "date": row[0].isoformat(),
                        "articles_count": row[1],
                        "avg_confidence": float(row[2]) if row[2] else 0.0,
                        "avg_sentiment_positive": float(row[3]) if row[3] else 0.0,
                        "avg_sentiment_negative": float(row[4]) if row[4] else 0.0,
                        "avg_bias_center": float(row[5]) if row[5] else 0.0,
                        "avg_risk_score": float(row[6]) if row[6] else 0.0,
                        "high_risk_articles": row[7]
                    })
                
                # Get recent alerts if requested
                alerts = []
                if include_alerts:
                    cur.execute("""
                        SELECT id, alert_type, severity, title, message, created_at, data
                        FROM alerts
                        WHERE source_id = %s 
                        AND created_at >= %s
                        ORDER BY created_at DESC
                        LIMIT 20
                    """, (source_id, start_date))
                    
                    for row in cur.fetchall():
                        alerts.append({
                            "id": str(row[0]),
                            "alert_type": row[1],
                            "severity": row[2],
                            "title": row[3],
                            "message": row[4],
                            "created_at": row[5].isoformat(),
                            "data": json.loads(row[6]) if row[6] else {}
                        })
                
                # Calculate advanced trends
                trends = {}
                if include_trends and len(time_series) >= 2:
                    trends = _calculate_trends(time_series)
                
                # Calculate source health score
                health_score = _calculate_source_health_score(summary_row, time_series)
                
                result = {
                    "source_id": source_id,
                    "days_analyzed": days,
                    "analysis_summary": {
                        "total_articles": summary_row[0] if summary_row[0] else 0,
                        "avg_confidence": float(summary_row[1]) if summary_row[1] else 0.0,
                        "avg_sentiment_positive": float(summary_row[2]) if summary_row[2] else 0.0,
                        "avg_sentiment_negative": float(summary_row[3]) if summary_row[3] else 0.0,
                        "avg_sentiment_neutral": float(summary_row[4]) if summary_row[4] else 0.0,
                        "avg_bias_left": float(summary_row[5]) if summary_row[5] else 0.0,
                        "avg_bias_right": float(summary_row[6]) if summary_row[6] else 0.0,
                        "avg_bias_center": float(summary_row[7]) if summary_row[7] else 0.0,
                        "avg_risk_score": float(summary_row[8]) if summary_row[8] else 0.0,
                        "confidence_stddev": float(summary_row[9]) if summary_row[9] else 0.0,
                        "sentiment_stddev": float(summary_row[10]) if summary_row[10] else 0.0,
                        "health_score": health_score
                    },
                    "time_series": time_series,
                    "trends": trends,
                    "alerts": alerts,
                    "generated_at": datetime.utcnow().isoformat()
                }
        
        return result
        
    except Exception as e:
        logger.error(f"Source analysis failed: {e}")
        ErrorHandler.raise_internal_server_error("Source analysis failed", str(e))

@router.get(
    "/performance",
    summary="Analysis Performance Metrics",
    description="Get real-time performance metrics for the analysis system",
    responses={
        200: {"description": "Performance metrics retrieved successfully"},
        500: {"description": "Failed to retrieve metrics", "model": ErrorResponse}
    }
)
async def get_analysis_performance(
    current_user: User = Depends(get_current_active_user)
):
    """Get analysis performance metrics"""
    try:
        # Get cache statistics
        cache_stats = await cache.get_stats()
        
        # Get system health
        system_health = await monitoring_service.get_system_health()
        
        return {
            "performance_metrics": performance_monitor.metrics,
            "cache_statistics": cache_stats,
            "system_health": system_health.dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        ErrorHandler.raise_internal_server_error("Failed to retrieve performance metrics", str(e))

# Helper functions
async def _store_alerts(alerts: List[Dict[str, Any]], article_id: Optional[str], source_id: Optional[str], user_id: int):
    """Store alerts in database"""
    try:
        from .. import db
        
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                for alert in alerts:
                    cur.execute("""
                        INSERT INTO alerts 
                        (alert_type, severity, title, message, article_id, source_id, user_id, data, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        alert["type"],
                        alert["severity"],
                        f"Analysis Alert: {alert['type']}",
                        alert["message"],
                        article_id,
                        source_id,
                        user_id,
                        json.dumps(alert.get("data", {})),
                        datetime.utcnow()
                    ))
    except Exception as e:
        logger.error(f"Failed to store alerts: {e}")

async def _store_analysis_result(result: Dict[str, Any], article_id: str, source_id: Optional[str], topic: Optional[str]):
    """Store analysis result in database"""
    try:
        from .. import db
        
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                # Store aggregated analysis
                cur.execute("""
                    INSERT INTO aggregated_analysis 
                    (article_id, source_id, topic, total_chunks, sentiment_distribution,
                     stance_distribution, toxicity_levels, bias_scores, confidence_avg,
                     risk_flags, trend_direction, created_at, analysis_data)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (article_id) 
                    DO UPDATE SET
                        source_id = EXCLUDED.source_id,
                        topic = EXCLUDED.topic,
                        total_chunks = EXCLUDED.total_chunks,
                        sentiment_distribution = EXCLUDED.sentiment_distribution,
                        stance_distribution = EXCLUDED.stance_distribution,
                        toxicity_levels = EXCLUDED.toxicity_levels,
                        bias_scores = EXCLUDED.bias_scores,
                        confidence_avg = EXCLUDED.confidence_avg,
                        risk_flags = EXCLUDED.risk_flags,
                        trend_direction = EXCLUDED.trend_direction,
                        analysis_data = EXCLUDED.analysis_data,
                        created_at = EXCLUDED.created_at
                """, (
                    article_id,
                    source_id,
                    topic,
                    1,  # Single text analysis
                    json.dumps(result.get("components", {}).get("sentiment", {}).get("scores", {})),
                    json.dumps(result.get("components", {}).get("stance", {}).get("scores", {})),
                    json.dumps({}),  # toxicity_levels
                    json.dumps(result.get("components", {}).get("bias", {}).get("scores", {})),
                    result.get("confidence", 0.0),
                    json.dumps(result.get("alerts", [])),
                    result.get("summary", {}).get("overall_sentiment", "neutral"),
                    datetime.utcnow(),
                    json.dumps(result)
                ))
    except Exception as e:
        logger.error(f"Failed to store analysis result: {e}")

def _calculate_trends(time_series: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate trends from time series data"""
    if len(time_series) < 2:
        return {}
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(time_series)
    
    trends = {}
    
    # Calculate sentiment trend
    if 'avg_sentiment_positive' in df.columns:
        sentiment_trend = np.polyfit(range(len(df)), df['avg_sentiment_positive'], 1)[0]
        trends['sentiment_trend'] = 'increasing' if sentiment_trend > 0.01 else 'decreasing' if sentiment_trend < -0.01 else 'stable'
        trends['sentiment_slope'] = float(sentiment_trend)
    
    # Calculate confidence trend
    if 'avg_confidence' in df.columns:
        confidence_trend = np.polyfit(range(len(df)), df['avg_confidence'], 1)[0]
        trends['confidence_trend'] = 'increasing' if confidence_trend > 0.01 else 'decreasing' if confidence_trend < -0.01 else 'stable'
        trends['confidence_slope'] = float(confidence_trend)
    
    # Calculate risk trend
    if 'avg_risk_score' in df.columns:
        risk_trend = np.polyfit(range(len(df)), df['avg_risk_score'], 1)[0]
        trends['risk_trend'] = 'increasing' if risk_trend > 0.01 else 'decreasing' if risk_trend < -0.01 else 'stable'
        trends['risk_slope'] = float(risk_trend)
    
    return trends

def _calculate_source_health_score(summary_row: Tuple, time_series: List[Dict[str, Any]]) -> float:
    """Calculate overall source health score (0-100)"""
    if not summary_row or not time_series:
        return 50.0  # Default neutral score
    
    score = 100.0
    
    # Confidence penalty
    avg_confidence = float(summary_row[1]) if summary_row[1] else 0.0
    score -= (1.0 - avg_confidence) * 30  # Up to 30 point penalty for low confidence
    
    # Risk penalty
    avg_risk = float(summary_row[8]) if summary_row[8] else 0.0
    score -= avg_risk * 40  # Up to 40 point penalty for high risk
    
    # Consistency bonus
    if len(time_series) > 1:
        confidences = [row['avg_confidence'] for row in time_series]
        confidence_std = np.std(confidences)
        score += (1.0 - min(confidence_std, 0.5)) * 20  # Up to 20 point bonus for consistency
    
    return max(0.0, min(100.0, score))

# Legacy endpoints for backward compatibility
@router.post(
    "/stance-sentiment-bias",
    response_model=StanceSentimentAnalysisResponse,
    summary="Legacy Stance-Sentiment-Bias Analysis",
    description="Legacy endpoint for backward compatibility",
    deprecated=True
)
async def analyze_stance_sentiment_bias_legacy(
    request: StanceSentimentAnalysisRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Legacy stance-sentiment-bias analysis endpoint"""
    return await analyze_text_advanced(request, "comprehensive", current_user)
