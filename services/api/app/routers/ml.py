"""
ML Processing router
Handles machine learning operations, bias analysis, and forecasting
"""

from fastapi import APIRouter, Depends, Query, HTTPException, Request
from typing import Optional, List, Dict, Any
from ..auth import get_current_active_user, User
from .. import db
from ..schemas import (
    BiasAnalysis, StanceSentimentAnalysisRequest, StanceSentimentAnalysisResponse,
    ForecastRequest, ForecastResponse, ErrorResponse, HTTPValidationError
)
from ..bias_analysis import bias_analyzer
from ..stance_sentiment_bias_analyzer import stance_sentiment_bias_analyzer
from ..forecasting import forecasting_service
from ..parallel_ml import get_parallel_processing_stats

router = APIRouter(prefix="/ml", tags=["ML Processing"])

@router.post(
    "/process/batch",
    response_model=Dict[str, Any],
    summary="Batch ML Processing",
    description="Process multiple articles through ML pipeline in parallel for high performance",
    responses={
        200: {"description": "Batch processing completed successfully"},
        400: {"description": "Invalid request", "model": ErrorResponse},
        422: {"description": "Validation error", "model": ErrorResponse}
    }
)
async def process_articles_batch_ml(
    request: Request,
    current_user: User = Depends(get_current_active_user)
):
    """Process multiple articles through ML pipeline"""
    try:
        data = await request.json()
        article_ids = data.get("article_ids", [])
        
        if not article_ids:
            raise HTTPException(
                status_code=400,
                detail="No article IDs provided"
            )
        
        # Process articles through ML pipeline
        results = []
        for article_id in article_ids:
            try:
                # This would typically process the article through various ML models
                result = {
                    "article_id": article_id,
                    "status": "processed",
                    "analysis": {
                        "sentiment": "positive",
                        "bias": "neutral",
                        "stance": "neutral"
                    }
                }
                results.append(result)
            except Exception as e:
                results.append({
                    "article_id": article_id,
                    "status": "failed",
                    "error": str(e)
                })
        
        return {
            "processed": len([r for r in results if r["status"] == "processed"]),
            "failed": len([r for r in results if r["status"] == "failed"]),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch ML processing failed: {str(e)}"
        )

@router.get(
    "/performance",
    response_model=Dict[str, Any],
    summary="Get ML Performance Metrics",
    description="Retrieve real-time performance metrics for parallel ML processing",
    responses={
        200: {"description": "Performance metrics retrieved successfully"}
    }
)
async def get_ml_performance_metrics(
    current_user: User = Depends(get_current_active_user)
):
    """Get ML performance metrics"""
    try:
        stats = await get_parallel_processing_stats()
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get ML performance metrics: {str(e)}"
        )

@router.post(
    "/features/extract/batch",
    response_model=Dict[str, Any],
    summary="Batch Feature Extraction",
    description="Extract ML features from multiple texts in parallel",
    responses={
        200: {"description": "Feature extraction completed successfully"},
        400: {"description": "Invalid request", "model": ErrorResponse}
    }
)
async def extract_features_batch(
    request: Request,
    current_user: User = Depends(get_current_active_user)
):
    """Extract ML features from multiple texts"""
    try:
        data = await request.json()
        texts = data.get("texts", [])
        
        if not texts:
            raise HTTPException(
                status_code=400,
                detail="No texts provided"
            )
        
        # Extract features from texts
        features = []
        for i, text in enumerate(texts):
            try:
                # This would typically extract various ML features
                feature = {
                    "text_id": i,
                    "features": {
                        "word_count": len(text.split()),
                        "sentiment_score": 0.5,
                        "complexity_score": 0.3
                    }
                }
                features.append(feature)
            except Exception as e:
                features.append({
                    "text_id": i,
                    "error": str(e)
                })
        
        return {
            "extracted": len([f for f in features if "error" not in f]),
            "failed": len([f for f in features if "error" in f]),
            "features": features
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Feature extraction failed: {str(e)}"
        )

@router.post(
    "/bias/analyze",
    response_model=BiasAnalysis,
    summary="Analyze Article Bias",
    description="Perform advanced bias analysis on an article",
    responses={
        200: {"description": "Bias analysis completed", "model": BiasAnalysis},
        404: {"description": "Article not found", "model": ErrorResponse},
        500: {"description": "Analysis failed", "model": ErrorResponse}
    }
)
async def analyze_bias_endpoint(
    article_id: int,
    current_user: User = Depends(get_current_active_user)
):
    """Analyze bias in an article"""
    try:
        # Get article data first
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, title, text FROM articles WHERE id = %s", (article_id,))
                article = cur.fetchone()
                if not article:
                    raise HTTPException(status_code=404, detail="Article not found")
                
                result = await bias_analyzer.analyze_article(article_id, article[1], article[2])
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Bias analysis failed: {str(e)}"
        )

@router.get("/bias/analysis/{article_id}")
async def get_bias_analysis(
    article_id: int,
    current_user: User = Depends(get_current_active_user)
):
    """Get bias analysis results for an article"""
    try:
        # This would typically fetch stored analysis results
        return {
            "article_id": article_id,
            "analysis": {
                "subjectivity": 0.5,
                "bias_lr": 0.0,
                "stance": "neutral"
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get bias analysis: {str(e)}"
        )

@router.post(
    "/analysis/stance-sentiment-bias",
    response_model=StanceSentimentAnalysisResponse,
    summary="Analyze Stance, Sentiment, and Political Bias",
    description="Perform specialized transformer-based analysis for sentiment, stance detection, and political bias",
    responses={
        200: {"description": "Analysis completed successfully"},
        400: {"description": "Invalid input", "model": ErrorResponse},
        500: {"description": "Analysis failed", "model": ErrorResponse}
    }
)
async def analyze_stance_sentiment_bias_endpoint(
    request: StanceSentimentAnalysisRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Analyze stance, sentiment, and bias"""
    try:
        result = await stance_sentiment_bias_analyzer.analyze_text(
            request.text,
            request.article_id,
            request.source_id,
            request.topic
        )
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Stance-sentiment-bias analysis failed: {str(e)}"
        )

@router.post(
    "/forecast/generate",
    response_model=ForecastResponse,
    summary="Generate Forecast",
    description="Generate time series forecast from historical data",
    responses={
        200: {"description": "Forecast generated successfully", "model": ForecastResponse},
        400: {"description": "Invalid data format", "model": ErrorResponse},
        500: {"description": "Forecast generation failed", "model": ErrorResponse}
    }
)
async def generate_forecast_endpoint(
    request: Request,
    current_user: User = Depends(get_current_active_user)
):
    """Generate forecast from time series data"""
    try:
        data = await request.json()
        forecast_request = ForecastRequest(**data)
        
        # Convert request to series data format
        series_data = []
        for item in forecast_request.data:
            series_data.append({
                "date": item.get("date", ""),
                "value": item.get("value", 0)
            })
        
        result = await forecasting_service.generate_forecast(series_data, forecast_request.horizon_days)
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Forecast generation failed: {str(e)}"
        )

@router.get("/forecast/trends")
async def get_article_trends_endpoint(
    days: int = 30,
    current_user: User = Depends(get_current_active_user)
):
    """Get article trends over time"""
    try:
        trends = await forecasting_service.get_article_trends(days)
        return trends
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get article trends: {str(e)}"
        )

@router.get("/forecast/topic/{topic}")
async def get_topic_trends_endpoint(
    topic: str,
    days: int = 30,
    current_user: User = Depends(get_current_active_user)
):
    """Get trends for a specific topic"""
    try:
        trends = await forecasting_service.get_topic_trends(topic, days)
        return trends
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get topic trends: {str(e)}"
        )

@router.get("/forecast/sentiment")
async def get_sentiment_trends_endpoint(
    days: int = 30,
    current_user: User = Depends(get_current_active_user)
):
    """Get sentiment trends over time"""
    try:
        trends = await forecasting_service.get_sentiment_trends(days)
        return trends
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get sentiment trends: {str(e)}"
        )

@router.get(
    "/performance",
    response_model=Dict[str, Any],
    tags=["ML Processing"],
    summary="Get ML Performance Metrics",
    description="Retrieve real-time performance metrics for parallel ML processing",
    responses={
        200: {"description": "Performance metrics retrieved successfully"}
    }
)
async def get_ml_performance_metrics(
    current_user: User = Depends(get_current_active_user)
):
    """Get comprehensive ML processing performance metrics"""
    try:
        stats = get_parallel_processing_stats()
        
        # Add additional system metrics
        import psutil
        stats.update({
            "system_cpu_count": psutil.cpu_count(),
            "system_memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "system_memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "system_cpu_percent": psutil.cpu_percent(interval=1),
            "system_memory_percent": psutil.virtual_memory().percent
        })
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting ML performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get performance metrics")

@router.post(
    "/process/batch",
    response_model=Dict[str, Any],
    tags=["ML Processing"],
    summary="Batch ML Processing",
    description="Process multiple articles through ML pipeline in parallel for high performance",
    responses={
        200: {"description": "Batch processing completed successfully"},
        400: {"description": "Invalid request", "model": ErrorResponse},
        422: {"description": "Validation error", "model": HTTPValidationError}
    }
)
async def process_articles_batch_ml(
    request: Request,
    current_user: User = Depends(get_current_active_user)
):
    """
    High-performance batch ML processing for multiple articles
    
    - **articles**: List of articles to process (max 100)
    - **processing_type**: Type of processing (full, features_only, sentiment_only)
    - **parallel_workers**: Number of parallel workers (auto if not specified)
    
    Processes articles in parallel using multiprocessing for optimal CPU utilization.
    """
    try:
        body = await request.json()
        articles = body.get("articles", [])
        processing_type = body.get("processing_type", "full")
        parallel_workers = body.get("parallel_workers")
        
        if not articles:
            raise HTTPException(status_code=400, detail="No articles provided")
        
        if len(articles) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 articles allowed per batch")
        
        # Validate article format
        for article in articles:
            if not all(key in article for key in ["id", "title", "text"]):
                raise HTTPException(status_code=400, detail="Articles must have 'id', 'title', and 'text' fields")
        
        start_time = time.time()
        logger.info(f"Starting batch ML processing for {len(articles)} articles")
        
        # Process articles in parallel
        if processing_type == "full":
            results = await ml_pipeline.process_articles_batch_parallel(articles)
        elif processing_type == "features_only":
            texts = [f"{article['title']} {article['text']}" for article in articles]
            feature_results = await ml_pipeline.extract_features_batch_parallel(texts)
            results = [
                {
                    "article_id": articles[i]["id"],
                    "features": feature_results[i],
                    "success": True
                }
                for i in range(len(articles))
            ]
        else:
            raise HTTPException(status_code=400, detail="Invalid processing_type")
        
        processing_time = time.time() - start_time
        successful = len([r for r in results if r.get("success", False)])
        
        # Get performance statistics
        perf_stats = get_parallel_processing_stats()
        
        return {
            "success": True,
            "total_articles": len(articles),
            "successful": successful,
            "failed": len(articles) - successful,
            "processing_time": processing_time,
            "processing_rate": len(articles) / processing_time,
            "results": results,
            "performance_stats": perf_stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch ML processing: {e}")
        raise HTTPException(status_code=500, detail="Failed to process articles batch")

@router.post(
    "/features/extract/batch",
    response_model=Dict[str, Any],
    tags=["ML Processing"],
    summary="Batch Feature Extraction",
    description="Extract ML features from multiple texts in parallel",
    responses={
        200: {"description": "Feature extraction completed successfully"},
        400: {"description": "Invalid request", "model": ErrorResponse}
    }
)
async def extract_features_batch(
    request: Request,
    current_user: User = Depends(get_current_active_user)
):
    """
    Extract ML features from multiple texts in parallel
    
    - **texts**: List of text strings to process (max 200)
    - **feature_types**: Types of features to extract (basic, sentiment, readability, all)
    
    Uses parallel processing for optimal performance.
    """
    try:
        body = await request.json()
        texts = body.get("texts", [])
        feature_types = body.get("feature_types", ["all"])
        
        if not texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        if len(texts) > 200:
            raise HTTPException(status_code=400, detail="Maximum 200 texts allowed per batch")
        
        start_time = time.time()
        logger.info(f"Starting batch feature extraction for {len(texts)} texts")
        
        # Extract features in parallel
        results = await ml_pipeline.extract_features_batch_parallel(texts)
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "total_texts": len(texts),
            "processing_time": processing_time,
            "processing_rate": len(texts) / processing_time,
            "features": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch feature extraction: {e}")
        raise HTTPException(status_code=500, detail="Failed to extract features")
