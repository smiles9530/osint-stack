"""
Forecasting service for time series analysis and predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from .db_pool import db_pool
from .enhanced_error_handling import APIError, ErrorHandler
import logging

logger = logging.getLogger(__name__)

class ForecastingService:
    """Time series forecasting service"""
    
    async def generate_forecast(self, series_data: List[Dict[str, Any]], horizon_days: int = 7) -> Dict[str, Any]:
        """
        Generate forecast for a time series
        
        Args:
            series_data: List of data points with 'date' and 'value' keys
            horizon_days: Number of days to forecast ahead
            
        Returns:
            Dictionary containing forecast results
        """
        try:
            if not series_data:
                return {"ok": False, "error": "No data provided"}
            
            # Convert to DataFrame
            df = pd.DataFrame(series_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Calculate rolling mean for baseline forecast
            window_size = min(7, max(3, len(df) // 3))
            if window_size < 3:
                window_size = len(df)
            
            mean_value = float(df['value'].tail(window_size).mean())
            
            # Generate forecast points
            last_date = df['date'].max()
            forecast_points = []
            
            for i in range(1, horizon_days + 1):
                forecast_date = last_date + timedelta(days=i)
                forecast_points.append({
                    "date": forecast_date.date().isoformat(),
                    "point": mean_value,
                    "confidence_lower": mean_value * 0.8,
                    "confidence_upper": mean_value * 1.2
                })
            
            # Generate scenarios
            scenarios = self._generate_scenarios(mean_value)
            
            return {
                "ok": True,
                "mean": mean_value,
                "horizon": forecast_points,
                "scenarios": scenarios,
                "data_points": len(df),
                "window_size": window_size
            }
            
        except Exception as e:
            logger.error(f"Forecast generation failed: {str(e)}")
            return {"ok": False, "error": str(e)}
    
    def _generate_scenarios(self, base_value: float) -> List[Dict[str, Any]]:
        """Generate forecast scenarios"""
        return [
            {
                "name": "base",
                "probability": 0.6,
                "delta": 0.0,
                "description": "Most likely scenario based on recent trends"
            },
            {
                "name": "upside",
                "probability": 0.2,
                "delta": 0.1 * base_value,
                "description": "Optimistic scenario with 10% increase"
            },
            {
                "name": "downside",
                "probability": 0.2,
                "delta": -0.1 * base_value,
                "description": "Pessimistic scenario with 10% decrease"
            }
        ]
    
    async def get_article_trends(self, days: int = 30) -> Dict[str, Any]:
        """
        Get article publication trends for forecasting
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dictionary containing trend data
        """
        try:
            async with db_pool.get_connection() as conn:
                # Get daily article counts
                rows = await conn.fetch("""
                    SELECT 
                        DATE(published_at) as date,
                        COUNT(*) as count
                    FROM articles 
                    WHERE published_at >= NOW() - INTERVAL '%s days'
                    GROUP BY DATE(published_at)
                    ORDER BY date
                """, days)
                
                if not rows:
                    return {"ok": False, "error": "No data available"}
                
                # Convert to series format
                series_data = [
                    {"date": row['date'].isoformat(), "value": row['count']}
                    for row in rows
                ]
                
                # Generate forecast
                forecast = await self.generate_forecast(series_data, 7)
                
                return {
                    "ok": True,
                    "historical_data": series_data,
                    "forecast": forecast,
                    "total_articles": sum(row['count'] for row in rows)
                }
                
        except Exception as e:
            logger.error(f"Failed to get article trends: {str(e)}")
            return {"ok": False, "error": str(e)}
    
    async def get_topic_trends(self, topic: str, days: int = 30) -> Dict[str, Any]:
        """
        Get trends for a specific topic
        
        Args:
            topic: Topic to analyze
            days: Number of days to look back
            
        Returns:
            Dictionary containing topic trend data
        """
        try:
            async with db_pool.get_connection() as conn:
                # Get daily counts for topic
                rows = await conn.fetch("""
                    SELECT 
                        DATE(a.published_at) as date,
                        COUNT(*) as count
                    FROM articles a
                    JOIN article_analysis aa ON a.id = aa.article_id
                    WHERE a.published_at >= NOW() - INTERVAL '%s days'
                    AND (aa.tags @> $1 OR a.title ILIKE $2 OR a.text ILIKE $2)
                    GROUP BY DATE(a.published_at)
                    ORDER BY date
                """, days, json.dumps([topic]), f"%{topic}%")
                
                if not rows:
                    return {"ok": False, "error": f"No data found for topic: {topic}"}
                
                # Convert to series format
                series_data = [
                    {"date": row['date'].isoformat(), "value": row['count']}
                    for row in rows
                ]
                
                # Generate forecast
                forecast = await self.generate_forecast(series_data, 7)
                
                return {
                    "ok": True,
                    "topic": topic,
                    "historical_data": series_data,
                    "forecast": forecast,
                    "total_articles": sum(row['count'] for row in rows)
                }
                
        except Exception as e:
            logger.error(f"Failed to get topic trends for {topic}: {str(e)}")
            return {"ok": False, "error": str(e)}
    
    async def get_sentiment_trends(self, days: int = 30) -> Dict[str, Any]:
        """
        Get sentiment trends over time
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dictionary containing sentiment trend data
        """
        try:
            async with db_pool.get_connection() as conn:
                # Get daily sentiment averages
                rows = await conn.fetch("""
                    SELECT 
                        DATE(a.published_at) as date,
                        AVG(aa.subjectivity) as avg_subjectivity,
                        AVG(aa.sensationalism) as avg_sensationalism,
                        AVG(aa.bias_lr) as avg_bias_lr,
                        COUNT(*) as article_count
                    FROM articles a
                    JOIN article_analysis aa ON a.id = aa.article_id
                    WHERE a.published_at >= NOW() - INTERVAL '%s days'
                    GROUP BY DATE(a.published_at)
                    ORDER BY date
                """, days)
                
                if not rows:
                    return {"ok": False, "error": "No sentiment data available"}
                
                # Convert to series format for each metric
                subjectivity_data = [
                    {"date": row['date'].isoformat(), "value": float(row['avg_subjectivity'] or 0)}
                    for row in rows
                ]
                
                sensationalism_data = [
                    {"date": row['date'].isoformat(), "value": float(row['avg_sensationalism'] or 0)}
                    for row in rows
                ]
                
                bias_data = [
                    {"date": row['date'].isoformat(), "value": float(row['avg_bias_lr'] or 0)}
                    for row in rows
                ]
                
                # Generate forecasts for each metric
                subjectivity_forecast = await self.generate_forecast(subjectivity_data, 7)
                sensationalism_forecast = await self.generate_forecast(sensationalism_data, 7)
                bias_forecast = await self.generate_forecast(bias_data, 7)
                
                return {
                    "ok": True,
                    "subjectivity": {
                        "historical": subjectivity_data,
                        "forecast": subjectivity_forecast
                    },
                    "sensationalism": {
                        "historical": sensationalism_data,
                        "forecast": sensationalism_forecast
                    },
                    "bias_lr": {
                        "historical": bias_data,
                        "forecast": bias_forecast
                    },
                    "total_articles": sum(row['article_count'] for row in rows)
                }
                
        except Exception as e:
            logger.error(f"Failed to get sentiment trends: {str(e)}")
            return {"ok": False, "error": str(e)}

# Global instance
forecasting_service = ForecastingService()
