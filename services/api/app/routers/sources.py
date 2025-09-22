"""
Enhanced Sources router
Handles source management, intelligence sources, quality scoring, automated monitoring, and performance analytics
"""

import asyncio
import logging
import time
import uuid
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from fastapi import APIRouter, Depends, Query, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from ..auth import get_current_active_user, User
from ..schemas import SourceList, IntelligenceSourceList, ErrorResponse
from ..enhanced_error_handling import ErrorHandler, APIError
from ..cache import cache
from ..monitoring import monitoring_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sources", tags=["Sources"])

# Request/Response models
class SourceQualityScore(BaseModel):
    source_id: str
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score (0-1)")
    reliability_score: float = Field(..., ge=0.0, le=1.0, description="Reliability score (0-1)")
    performance_score: float = Field(..., ge=0.0, le=1.0, description="Performance score (0-1)")
    content_quality_score: float = Field(..., ge=0.0, le=1.0, description="Content quality score (0-1)")
    uptime_score: float = Field(..., ge=0.0, le=1.0, description="Uptime score (0-1)")
    last_updated: str = Field(..., description="Last update timestamp")

class SourceMonitoringConfig(BaseModel):
    source_id: str
    check_interval_minutes: int = Field(60, ge=5, le=1440, description="Check interval in minutes")
    timeout_seconds: int = Field(30, ge=5, le=300, description="Request timeout in seconds")
    retry_attempts: int = Field(3, ge=1, le=10, description="Number of retry attempts")
    alert_thresholds: Dict[str, float] = Field(default_factory=dict, description="Alert thresholds")

class SourcePerformanceMetrics(BaseModel):
    source_id: str
    response_time_avg: float = Field(..., description="Average response time in seconds")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate (0-1)")
    uptime_percentage: float = Field(..., ge=0.0, le=100.0, description="Uptime percentage")
    articles_per_day: float = Field(..., description="Average articles per day")
    error_rate: float = Field(..., ge=0.0, le=1.0, description="Error rate (0-1)")
    last_checked: str = Field(..., description="Last check timestamp")

# Advanced Source Management Service
class AdvancedSourceService:
    def __init__(self):
        self.quality_scores = {}
        self.monitoring_configs = {}
        self.performance_metrics = {}
        self.health_checks = {}
        self.alert_thresholds = {
            "response_time": 5.0,  # seconds
            "success_rate": 0.7,   # 70%
            "uptime": 0.95,        # 95%
            "error_rate": 0.1      # 10%
        }
    
    async def calculate_source_quality_score(self, source_id: str) -> SourceQualityScore:
        """Calculate comprehensive quality score for a source"""
        try:
            from .. import db
            
            with db.get_conn() as conn:
                with conn.cursor() as cur:
                    # Get source data
                    cur.execute("""
                        SELECT 
                            id, name, url, success_rate, article_count, last_checked,
                            metadata, created_at, updated_at
                        FROM sources
                        WHERE id = %s
                    """, (source_id,))
                    
                    source_data = cur.fetchone()
                    if not source_data:
                        raise APIError(
                            status_code=404,
                            error_code="SOURCE_NOT_FOUND",
                            message="Source not found"
                        )
                    
                    # Get recent performance data
                    cur.execute("""
                        SELECT 
                            AVG(response_time) as avg_response_time,
                            COUNT(*) as total_checks,
                            SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_checks,
                            SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_checks
                        FROM source_health_checks
                        WHERE source_id = %s
                        AND created_at >= %s
                    """, (source_id, datetime.utcnow() - timedelta(days=7)))
                    
                    performance_data = cur.fetchone()
                    
                    # Calculate individual scores
                    reliability_score = self._calculate_reliability_score(source_data, performance_data)
                    performance_score = self._calculate_performance_score(performance_data)
                    content_quality_score = self._calculate_content_quality_score(source_data)
                    uptime_score = self._calculate_uptime_score(performance_data)
                    
                    # Calculate overall score (weighted average)
                    overall_score = (
                        reliability_score * 0.3 +
                        performance_score * 0.25 +
                        content_quality_score * 0.25 +
                        uptime_score * 0.2
                    )
                    
                    quality_score = SourceQualityScore(
                        source_id=source_id,
                        overall_score=overall_score,
                        reliability_score=reliability_score,
                        performance_score=performance_score,
                        content_quality_score=content_quality_score,
                        uptime_score=uptime_score,
                        last_updated=datetime.utcnow().isoformat()
                    )
                    
                    # Cache the score
                    self.quality_scores[source_id] = quality_score
                    await cache.set(f"source_quality:{source_id}", quality_score.dict(), ttl=3600)
                    
                    return quality_score
                    
        except APIError:
            raise
        except Exception as e:
            logger.error(f"Failed to calculate quality score for source {source_id}: {e}")
            raise APIError(
                status_code=500,
                error_code="QUALITY_SCORE_CALCULATION_FAILED",
                message="Failed to calculate quality score",
                detail=str(e)
            )
    
    def _calculate_reliability_score(self, source_data: Tuple, performance_data: Tuple) -> float:
        """Calculate reliability score based on historical performance"""
        try:
            success_rate = float(source_data[3]) if source_data[3] else 0.0
            article_count = source_data[4] or 0
            
            # Base score from success rate
            reliability = success_rate
            
            # Bonus for high article count (indicates consistent operation)
            if article_count > 1000:
                reliability += 0.1
            elif article_count > 100:
                reliability += 0.05
            
            # Penalty for very low article count
            if article_count < 10:
                reliability -= 0.2
            
            return max(0.0, min(1.0, reliability))
            
        except Exception:
            return 0.5  # Default score
    
    def _calculate_performance_score(self, performance_data: Tuple) -> float:
        """Calculate performance score based on response times and success rates"""
        try:
            if not performance_data or not performance_data[0]:
                return 0.5  # Default score
            
            avg_response_time = float(performance_data[0])
            total_checks = performance_data[1] or 0
            successful_checks = performance_data[2] or 0
            
            if total_checks == 0:
                return 0.5
            
            # Response time score (better = faster)
            response_score = max(0.0, 1.0 - (avg_response_time / 10.0))  # 10s = 0 score
            
            # Success rate score
            success_rate = successful_checks / total_checks
            success_score = success_rate
            
            # Combined performance score
            performance = (response_score * 0.4 + success_score * 0.6)
            
            return max(0.0, min(1.0, performance))
            
        except Exception:
            return 0.5  # Default score
    
    def _calculate_content_quality_score(self, source_data: Tuple) -> float:
        """Calculate content quality score based on metadata and patterns"""
        try:
            metadata = source_data[6] or {}
            
            # Base score
            quality = 0.5
            
            # Check for quality indicators in metadata
            if metadata.get("content_quality_score"):
                quality = float(metadata["content_quality_score"])
            elif metadata.get("reliability_score"):
                quality = float(metadata["reliability_score"])
            else:
                # Estimate based on source characteristics
                article_count = source_data[4] or 0
                if article_count > 500:
                    quality = 0.8
                elif article_count > 100:
                    quality = 0.6
                else:
                    quality = 0.4
            
            return max(0.0, min(1.0, quality))
            
        except Exception:
            return 0.5  # Default score
    
    def _calculate_uptime_score(self, performance_data: Tuple) -> float:
        """Calculate uptime score based on recent health checks"""
        try:
            if not performance_data:
                return 0.5  # Default score
            
            total_checks = performance_data[1] or 0
            successful_checks = performance_data[2] or 0
            
            if total_checks == 0:
                return 0.5
            
            uptime = successful_checks / total_checks
            return max(0.0, min(1.0, uptime))
            
        except Exception:
            return 0.5  # Default score
    
    async def perform_health_check(self, source_id: str) -> Dict[str, Any]:
        """Perform comprehensive health check on a source"""
        try:
            from .. import db
            import httpx
            
            # Get source URL
            with db.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT url, metadata FROM sources WHERE id = %s", (source_id,))
                    source_info = cur.fetchone()
                    
                    if not source_info:
                        raise APIError(
                            status_code=404,
                            error_code="SOURCE_NOT_FOUND",
                            message="Source not found"
                        )
                    
                    url = source_info[0]
                    metadata = source_info[1] or {}
            
            # Perform health check
            start_time = time.time()
            status = "success"
            error_message = None
            response_time = 0.0
            
            try:
                timeout = metadata.get("timeout_seconds", 30)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(url)
                    response_time = time.time() - start_time
                    
                    if response.status_code >= 400:
                        status = "error"
                        error_message = f"HTTP {response.status_code}"
                    elif response_time > self.alert_thresholds["response_time"]:
                        status = "warning"
                        error_message = f"Slow response: {response_time:.2f}s"
                    
            except Exception as e:
                status = "error"
                error_message = str(e)
                response_time = time.time() - start_time
            
            # Store health check result
            check_id = str(uuid.uuid4())
            with db.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO source_health_checks 
                        (id, source_id, status, response_time, error_message, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        check_id,
                        source_id,
                        status,
                        response_time,
                        error_message,
                        datetime.utcnow()
                    ))
            
            # Update source last_checked
            with db.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE sources 
                        SET last_checked = %s
                        WHERE id = %s
                    """, (datetime.utcnow(), source_id))
            
            return {
                "check_id": check_id,
                "source_id": source_id,
                "status": status,
                "response_time": response_time,
                "error_message": error_message,
                "checked_at": datetime.utcnow().isoformat()
            }
            
        except APIError:
            raise
        except Exception as e:
            logger.error(f"Health check failed for source {source_id}: {e}")
            raise APIError(
                status_code=500,
                error_code="HEALTH_CHECK_FAILED",
                message="Health check failed",
                detail=str(e)
            )
    
    async def get_source_performance_metrics(self, source_id: str, days_back: int = 7) -> SourcePerformanceMetrics:
        """Get comprehensive performance metrics for a source"""
        try:
            from .. import db
            
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days_back)
            
            with db.get_conn() as conn:
                with conn.cursor() as cur:
                    # Get performance data
                    cur.execute("""
                        SELECT 
                            AVG(response_time) as avg_response_time,
                            COUNT(*) as total_checks,
                            SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_checks,
                            SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_checks,
                            MAX(created_at) as last_checked
                        FROM source_health_checks
                        WHERE source_id = %s
                        AND created_at >= %s
                    """, (source_id, start_time))
                    
                    performance_data = cur.fetchone()
                    
                    # Get article count for the period
                    cur.execute("""
                        SELECT COUNT(*) as article_count
                        FROM articles
                        WHERE source_id = %s
                        AND created_at >= %s
                    """, (source_id, start_time))
                    
                    article_data = cur.fetchone()
                    article_count = article_data[0] if article_data else 0
                    
                    # Calculate metrics
                    if performance_data and performance_data[1] > 0:
                        avg_response_time = float(performance_data[0]) if performance_data[0] else 0.0
                        total_checks = performance_data[1]
                        successful_checks = performance_data[2] or 0
                        error_checks = performance_data[3] or 0
                        last_checked = performance_data[4]
                        
                        success_rate = successful_checks / total_checks
                        uptime_percentage = (successful_checks / total_checks) * 100
                        error_rate = error_checks / total_checks
                        articles_per_day = article_count / days_back
                    else:
                        avg_response_time = 0.0
                        success_rate = 0.0
                        uptime_percentage = 0.0
                        error_rate = 0.0
                        articles_per_day = 0.0
                        last_checked = None
                    
                    metrics = SourcePerformanceMetrics(
                        source_id=source_id,
                        response_time_avg=avg_response_time,
                        success_rate=success_rate,
                        uptime_percentage=uptime_percentage,
                        articles_per_day=articles_per_day,
                        error_rate=error_rate,
                        last_checked=last_checked.isoformat() if last_checked else datetime.utcnow().isoformat()
                    )
                    
                    return metrics
                    
        except Exception as e:
            logger.error(f"Failed to get performance metrics for source {source_id}: {e}")
            raise APIError(
                status_code=500,
                error_code="METRICS_RETRIEVAL_FAILED",
                message="Failed to retrieve performance metrics",
                detail=str(e)
            )

# Initialize service
source_service = AdvancedSourceService()

@router.get(
    "/enabled",
    response_model=SourceList,
    summary="Get Enabled Sources",
    description="Retrieve enabled sources with optional filtering",
    responses={
        200: {"description": "Sources retrieved successfully", "model": SourceList}
    }
)
async def get_enabled_sources(
    category: Optional[str] = Query(None, description="Filter by category name"),
    limit: Optional[int] = Query(None, description="Limit number of results", ge=1, le=1000)
):
    """Get enabled sources with optional filtering"""
    try:
        from .. import db
        
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                # Build query with optional category filter
                base_query = """
                    SELECT s.id, s.name, s.url, s.category, s.language, s.country,
                           s.is_enabled, s.last_checked, s.success_rate, s.article_count,
                           s.metadata, s.created_at, s.updated_at
                    FROM sources s
                    WHERE s.is_enabled = TRUE
                """
                params = []
                
                if category:
                    base_query += " AND s.category = %s"
                    params.append(category)
                
                # Get total counts
                count_query = "SELECT COUNT(*) FROM sources WHERE is_enabled = TRUE"
                count_params = []
                if category:
                    count_query += " AND category = %s"
                    count_params.append(category)
                
                cur.execute(count_query, count_params)
                filtered_count = cur.fetchone()[0]
                
                # Get total sources count
                with conn.cursor() as cur2:
                    cur2.execute("SELECT COUNT(*) FROM sources")
                    total_sources = cur2.fetchone()[0]
                
                # Get enabled sources count
                with conn.cursor() as cur3:
                    cur3.execute("SELECT COUNT(*) FROM sources WHERE is_enabled = TRUE")
                    total_enabled = cur3.fetchone()[0]
                
                # Add ordering and limit
                base_query += " ORDER BY s.success_rate DESC, s.article_count DESC"
                if limit:
                    base_query += " LIMIT %s"
                    params.append(limit)
                
                # Execute main query
                with conn.cursor() as cur4:
                    cur4.execute(base_query, params)
                    source_rows = cur4.fetchall()
                
        sources = []
        for row in source_rows:
            sources.append({
                "id": row[0],
                "name": row[1],
                "url": row[2],
                "category": row[3],
                "language": row[4],
                "country": row[5],
                "is_enabled": row[6],
                "last_checked": row[7].isoformat() if row[7] else None,
                "success_rate": float(row[8]) if row[8] else 0.0,
                "article_count": row[9] if row[9] else 0,
                "metadata": row[10] if row[10] else {},
                "created_at": row[11].isoformat() if row[11] else None,
                "updated_at": row[12].isoformat() if row[12] else None
            })
        
        return SourceList(
            sources=sources,
            total_enabled=total_enabled,
            total_sources=total_sources,
            filtered_count=filtered_count
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get enabled sources: {str(e)}"
        )

@router.get(
    "/intelligence",
    response_model=IntelligenceSourceList,
    summary="Get Intelligence Sources",
    description="Retrieve sources optimized for intelligence collection with AI-enhanced filtering",
    responses={
        200: {"description": "Intelligence sources retrieved successfully", "model": IntelligenceSourceList}
    }
)
async def get_intelligence_sources(
    domain: Optional[str] = Query(None, description="Filter by intelligence domain"),
    priority: Optional[str] = Query(None, description="Filter by priority level (high/medium/low)"),
    limit: Optional[int] = Query(500, description="Maximum number of sources to return", ge=1, le=1000)
):
    """Get intelligence-optimized sources"""
    try:
        from ..db_pool import db_pool
        from ..ml_pipeline import ml_pipeline
        
        async with db_pool.get_connection() as conn:
            # Get available domains
            domain_rows = await conn.fetch("""
                SELECT DISTINCT s.metadata->>'intelligence_domain' as domain
                FROM sources s
                WHERE s.is_enabled = TRUE 
                AND s.metadata->>'intelligence_domain' IS NOT NULL
                ORDER BY domain
            """)
            available_domains = [row['domain'] for row in domain_rows if row['domain']]
            
            # Get intelligence summary
            summary_rows = await conn.fetch("""
                SELECT 
                    s.metadata->>'intelligence_domain' as domain,
                    COUNT(*) as source_count,
                    AVG(COALESCE(s.metadata->>'reliability_score', '0.5')::float) as avg_reliability,
                    AVG(s.success_rate) as avg_success_rate
                FROM sources s
                WHERE s.is_enabled = TRUE
                AND s.metadata->>'intelligence_domain' IS NOT NULL
                GROUP BY s.metadata->>'intelligence_domain'
                ORDER BY source_count DESC
            """)
            
            intelligence_summary = {}
            for row in summary_rows:
                intelligence_summary[row['domain']] = {
                    "source_count": row['source_count'],
                    "avg_reliability": float(row['avg_reliability']) if row['avg_reliability'] else 0.0,
                    "avg_success_rate": float(row['avg_success_rate']) if row['avg_success_rate'] else 0.0
                }
            
            # Build query for intelligence sources
            base_query = """
                SELECT s.id, s.name, s.url, s.category, s.language, s.country,
                       s.is_enabled, s.last_checked, s.success_rate, s.article_count,
                       s.metadata, s.created_at, s.updated_at,
                       COALESCE(s.metadata->>'intelligence_domain', 'general') as intelligence_domain,
                       COALESCE(s.metadata->>'priority', 'medium') as priority,
                       COALESCE(s.metadata->>'reliability_score', '0.5')::float as reliability_score
                FROM sources s
                WHERE s.is_enabled = TRUE
            """
            params = []
            
            param_count = 0
            if domain:
                param_count += 1
                base_query += f" AND s.metadata->>'intelligence_domain' = ${param_count}"
                params.append(domain)
            
            if priority:
                param_count += 1
                base_query += f" AND s.metadata->>'priority' = ${param_count}"
                params.append(priority)
            
            # Add ordering and limit
            base_query += " ORDER BY reliability_score DESC, s.success_rate DESC, s.article_count DESC"
            if limit:
                param_count += 1
                base_query += f" LIMIT ${param_count}"
                params.append(limit)
            
            # Execute main query
            source_rows = await conn.fetch(base_query, *params)
            
            # Process source data
            sources = []
            sources_by_domain = {}
            
            for row in source_rows:
                source_data = {
                    "id": row['id'],
                    "title": row['name'],  # Map name to title
                    "url": row['url'],
                    "category_name": row['category'] or 'general',  # Map category to category_name
                    "language": row['language'] or 'en',
                    "enabled": row['is_enabled'],  # Map is_enabled to enabled
                    "intelligence_domains": [row['intelligence_domain']] if row['intelligence_domain'] else ['general'],
                    "priority_score": float(row['reliability_score']) if row['reliability_score'] else 0.5,
                    "intelligence_priority": row['priority'] or 'medium',
                    "subcategory_name": None,
                    "fetch_period": 3600,
                    # Additional fields for context
                    "country": row['country'],
                    "last_checked": row['last_checked'].isoformat() if row['last_checked'] else None,
                    "success_rate": float(row['success_rate']) if row['success_rate'] else 0.0,
                    "article_count": row['article_count'] if row['article_count'] else 0,
                    "metadata": row['metadata'] if row['metadata'] else {},
                    "created_at": row['created_at'].isoformat() if row['created_at'] else None,
                    "updated_at": row['updated_at'].isoformat() if row['updated_at'] else None,
                    "reliability_score": float(row['reliability_score']) if row['reliability_score'] else 0.0
                }
                
                sources.append(source_data)
                
                # Group by domain
                domain = row['intelligence_domain'] or 'general'
                if domain not in sources_by_domain:
                    sources_by_domain[domain] = []
                sources_by_domain[domain].append(source_data)
            
            # Get counts
            total_enabled_row = await conn.fetchrow("SELECT COUNT(*) FROM sources WHERE is_enabled = TRUE")
            total_enabled = total_enabled_row['count']
            
            filtered_count = len(sources)
            
            return IntelligenceSourceList(
                sources=sources,
                sources_by_domain=sources_by_domain,
                total_enabled=total_enabled,
                filtered_count=filtered_count,
                domain_filter=domain,
                priority_filter=priority,
                available_domains=available_domains,
                intelligence_summary=intelligence_summary
            )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get intelligence sources: {str(e)}"
        )

@router.get(
    "/stats",
    summary="Enhanced Source Statistics",
    description="Get comprehensive source statistics with performance analytics and quality metrics",
    responses={
        200: {"description": "Statistics retrieved successfully"},
        500: {"description": "Failed to retrieve statistics", "model": ErrorResponse}
    }
)
async def get_sources_stats_enhanced():
    """Enhanced source statistics with performance analytics"""
    try:
        from .. import db
        
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                # Get basic counts
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_sources,
                        SUM(CASE WHEN is_enabled = TRUE THEN 1 ELSE 0 END) as enabled_sources,
                        SUM(CASE WHEN is_enabled = FALSE THEN 1 ELSE 0 END) as disabled_sources,
                        AVG(success_rate) as avg_success_rate,
                        SUM(article_count) as total_articles
                    FROM sources
                """)
                
                basic_stats = cur.fetchone()
                
                # Get sources by category with enhanced metrics
                cur.execute("""
                    SELECT 
                        category, 
                        COUNT(*) as count, 
                        AVG(success_rate) as avg_success,
                        AVG(article_count) as avg_articles,
                        MAX(last_checked) as last_checked
                    FROM sources
                    WHERE category IS NOT NULL
                    GROUP BY category
                    ORDER BY count DESC
                """)
                
                sources_by_category = {}
                for row in cur.fetchall():
                    sources_by_category[row[0]] = {
                        "count": row[1],
                        "avg_success_rate": float(row[2]) if row[2] else 0.0,
                        "avg_articles": float(row[3]) if row[3] else 0.0,
                        "last_checked": row[4].isoformat() if row[4] else None
                    }
                
                # Get sources by language with enhanced metrics
                cur.execute("""
                    SELECT 
                        language, 
                        COUNT(*) as count, 
                        AVG(success_rate) as avg_success,
                        AVG(article_count) as avg_articles
                    FROM sources
                    WHERE language IS NOT NULL
                    GROUP BY language
                    ORDER BY count DESC
                """)
                
                sources_by_language = {}
                for row in cur.fetchall():
                    sources_by_language[row[0]] = {
                        "count": row[1],
                        "avg_success_rate": float(row[2]) if row[2] else 0.0,
                        "avg_articles": float(row[3]) if row[3] else 0.0
                    }
                
                # Get sources by country with enhanced metrics
                cur.execute("""
                    SELECT 
                        country, 
                        COUNT(*) as count, 
                        AVG(success_rate) as avg_success,
                        AVG(article_count) as avg_articles
                    FROM sources
                    WHERE country IS NOT NULL
                    GROUP BY country
                    ORDER BY count DESC
                    LIMIT 10
                """)
                
                sources_by_country = {}
                for row in cur.fetchall():
                    sources_by_country[row[0]] = {
                        "count": row[1],
                        "avg_success_rate": float(row[2]) if row[2] else 0.0,
                        "avg_articles": float(row[3]) if row[3] else 0.0
                    }
                
                # Get recent activity with more details
                cur.execute("""
                    SELECT 
                        DATE(last_checked) as date,
                        COUNT(*) as sources_checked,
                        AVG(success_rate) as avg_success_rate
                    FROM sources
                    WHERE last_checked >= %s
                    GROUP BY DATE(last_checked)
                    ORDER BY date DESC
                    LIMIT 7
                """, (datetime.utcnow() - timedelta(days=7),))
                
                recent_activity = []
                for row in cur.fetchall():
                    recent_activity.append({
                        "date": row[0].isoformat(),
                        "sources_checked": row[1],
                        "avg_success_rate": float(row[2]) if row[2] else 0.0
                    })
                
                # Get top performing sources with quality indicators
                cur.execute("""
                    SELECT 
                        s.name, s.url, s.success_rate, s.article_count, s.last_checked,
                        s.metadata->>'reliability_score' as reliability_score,
                        s.metadata->>'content_quality_score' as content_quality_score
                    FROM sources s
                    WHERE s.is_enabled = TRUE
                    ORDER BY s.success_rate DESC, s.article_count DESC
                    LIMIT 10
                """)
                
                top_sources = []
                for row in cur.fetchall():
                    top_sources.append({
                        "name": row[0],
                        "url": row[1],
                        "success_rate": float(row[2]) if row[2] else 0.0,
                        "article_count": row[3] if row[3] else 0,
                        "last_checked": row[4].isoformat() if row[4] else None,
                        "reliability_score": float(row[5]) if row[5] else None,
                        "content_quality_score": float(row[6]) if row[6] else None
                    })
                
                # Get health check statistics
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_checks,
                        SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_checks,
                        SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_checks,
                        AVG(response_time) as avg_response_time,
                        MAX(created_at) as last_check
                    FROM source_health_checks
                    WHERE created_at >= %s
                """, (datetime.utcnow() - timedelta(days=7),))
                
                health_stats = cur.fetchone()
                
                # Get quality score distribution
                cur.execute("""
                    SELECT 
                        CASE 
                            WHEN success_rate >= 0.9 THEN 'excellent'
                            WHEN success_rate >= 0.7 THEN 'good'
                            WHEN success_rate >= 0.5 THEN 'fair'
                            ELSE 'poor'
                        END as quality_tier,
                        COUNT(*) as count
                    FROM sources
                    WHERE is_enabled = TRUE
                    GROUP BY quality_tier
                    ORDER BY 
                        CASE quality_tier
                            WHEN 'excellent' THEN 1
                            WHEN 'good' THEN 2
                            WHEN 'fair' THEN 3
                            WHEN 'poor' THEN 4
                        END
                """)
                
                quality_distribution = {}
                for row in cur.fetchall():
                    quality_distribution[row[0]] = row[1]
        
        return {
            "total_sources": basic_stats[0] if basic_stats[0] else 0,
            "enabled_sources": basic_stats[1] if basic_stats[1] else 0,
            "disabled_sources": basic_stats[2] if basic_stats[2] else 0,
            "avg_success_rate": float(basic_stats[3]) if basic_stats[3] else 0.0,
            "total_articles": basic_stats[4] if basic_stats[4] else 0,
            "sources_by_category": sources_by_category,
            "sources_by_language": sources_by_language,
            "sources_by_country": sources_by_country,
            "recent_activity": recent_activity,
            "top_sources": top_sources,
            "quality_distribution": quality_distribution,
            "health_check_stats": {
                "total_checks": health_stats[0] if health_stats[0] else 0,
                "successful_checks": health_stats[1] if health_stats[1] else 0,
                "error_checks": health_stats[2] if health_stats[2] else 0,
                "avg_response_time": float(health_stats[3]) if health_stats[3] else 0.0,
                "last_check": health_stats[4].isoformat() if health_stats[4] else None
            },
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get enhanced source statistics: {e}")
        ErrorHandler.raise_internal_server_error("Failed to get enhanced source statistics", str(e))

@router.get(
    "/quality/{source_id}",
    summary="Get Source Quality Score",
    description="Calculate and retrieve comprehensive quality score for a specific source",
    responses={
        200: {"description": "Quality score retrieved successfully", "model": SourceQualityScore},
        404: {"description": "Source not found", "model": ErrorResponse},
        500: {"description": "Failed to calculate quality score", "model": ErrorResponse}
    }
)
async def get_source_quality_score(
    source_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get comprehensive quality score for a source"""
    try:
        # Check cache first
        cached_score = await cache.get(f"source_quality:{source_id}")
        if cached_score:
            return cached_score
        
        # Calculate quality score
        quality_score = await source_service.calculate_source_quality_score(source_id)
        
        return quality_score
        
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Failed to get quality score for source {source_id}: {e}")
        ErrorHandler.raise_internal_server_error("Failed to get quality score", str(e))

@router.post(
    "/health-check/{source_id}",
    summary="Perform Source Health Check",
    description="Perform comprehensive health check on a specific source",
    responses={
        200: {"description": "Health check completed successfully"},
        404: {"description": "Source not found", "model": ErrorResponse},
        500: {"description": "Health check failed", "model": ErrorResponse}
    }
)
async def perform_source_health_check(
    source_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Perform health check on a source"""
    try:
        result = await source_service.perform_health_check(source_id)
        return result
        
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Health check failed for source {source_id}: {e}")
        ErrorHandler.raise_internal_server_error("Health check failed", str(e))

@router.get(
    "/performance/{source_id}",
    summary="Get Source Performance Metrics",
    description="Get comprehensive performance metrics for a specific source",
    responses={
        200: {"description": "Performance metrics retrieved successfully", "model": SourcePerformanceMetrics},
        404: {"description": "Source not found", "model": ErrorResponse},
        500: {"description": "Failed to retrieve metrics", "model": ErrorResponse}
    }
)
async def get_source_performance_metrics(
    source_id: str,
    days_back: int = Query(7, description="Number of days to include in analysis", ge=1, le=30),
    current_user: User = Depends(get_current_active_user)
):
    """Get performance metrics for a source"""
    try:
        metrics = await source_service.get_source_performance_metrics(source_id, days_back)
        return metrics
        
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Failed to get performance metrics for source {source_id}: {e}")
        ErrorHandler.raise_internal_server_error("Failed to retrieve performance metrics", str(e))

@router.post(
    "/health-check/batch",
    summary="Batch Health Check",
    description="Perform health checks on multiple sources in parallel",
    responses={
        200: {"description": "Batch health check completed successfully"},
        500: {"description": "Batch health check failed", "model": ErrorResponse}
    }
)
async def perform_batch_health_check(
    source_ids: List[str],
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    """Perform health checks on multiple sources"""
    try:
        if not source_ids:
            ErrorHandler.raise_bad_request("No source IDs provided")
        
        if len(source_ids) > 50:
            ErrorHandler.raise_bad_request("Too many sources for batch processing (max 50)")
        
        # Process in background for large batches
        if len(source_ids) > 10:
            background_tasks.add_task(
                _process_batch_health_checks,
                source_ids,
                current_user.id
            )
            
            return {
                "status": "queued",
                "message": f"Health checks queued for {len(source_ids)} sources",
                "estimated_completion_time": f"{len(source_ids) * 2} seconds"
            }
        else:
            # Process immediately for small batches
            results = []
            for source_id in source_ids:
                try:
                    result = await source_service.perform_health_check(source_id)
                    results.append(result)
                except Exception as e:
                    results.append({
                        "source_id": source_id,
                        "status": "error",
                        "error": str(e)
                    })
            
            return {
                "status": "completed",
                "results": results,
                "total_sources": len(source_ids),
                "successful_checks": len([r for r in results if r.get("status") == "success"]),
                "failed_checks": len([r for r in results if r.get("status") == "error"])
            }
        
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Batch health check failed: {e}")
        ErrorHandler.raise_internal_server_error("Batch health check failed", str(e))

async def _process_batch_health_checks(source_ids: List[str], user_id: int):
    """Background task for processing batch health checks"""
    try:
        results = []
        for source_id in source_ids:
            try:
                result = await source_service.perform_health_check(source_id)
                results.append(result)
            except Exception as e:
                results.append({
                    "source_id": source_id,
                    "status": "error",
                    "error": str(e)
                })
        
        logger.info(f"Batch health check completed for {len(source_ids)} sources")
        
    except Exception as e:
        logger.error(f"Background batch health check failed: {e}")
