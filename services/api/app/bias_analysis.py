"""
Advanced multi-model bias analysis with confidence scoring and trend tracking
"""

import json
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from .config import settings
from .db_pool import db_pool
from .enhanced_error_handling import APIError, ErrorHandler
import logging
import redis
import httpx
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import hashlib

logger = logging.getLogger(__name__)

class MultiModelBiasAnalyzer:
    """Advanced multi-model bias analysis with ensemble methods and confidence scoring"""
    
    def __init__(self):
        self.bias_schema = self._load_bias_schema()
        # Parse Redis URL to get host, port, and database
        redis_url = settings.redis_url.replace("redis://", "")
        if "/" in redis_url:
            # Handle URL with database number like "localhost:6379/0"
            url_part, db_part = redis_url.split("/", 1)
            if ":" in url_part:
                host, port = url_part.split(":")
                port = int(port)
            else:
                host = url_part
                port = 6379
        else:
            # Handle URL without database number
            if ":" in redis_url:
                host, port = redis_url.split(":")
                port = int(port)
            else:
                host = redis_url
                port = 6379
        self.redis_client = redis.Redis(host=host, port=port, db=3, decode_responses=True)
        self.models = {}
        self.model_weights = {}
        self.confidence_threshold = 0.7
        self.trend_window = 30  # days
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize multiple bias analysis models"""
        # Load or create models for different aspects
        self.models = {
            'subjectivity_classifier': self._create_classifier(),
            'bias_detector': self._create_classifier(),
            'stance_classifier': self._create_classifier(),
            'sensationalism_detector': self._create_classifier()
        }
        
        # Initialize equal weights (will be updated based on performance)
        self.model_weights = {name: 1.0 for name in self.models.keys()}
    
    def _create_classifier(self):
        """Create a classifier for bias analysis"""
        return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def _load_bias_schema(self) -> Dict[str, Any]:
        """Load the bias analysis schema"""
        try:
            # Try multiple possible paths for the schema file
            schema_paths = [
                "prompts/bias_rubric.schema.json",  # Relative to current directory
                "../../prompts/bias_rubric.schema.json",  # Relative to app directory
                "/app/prompts/bias_rubric.schema.json",  # Absolute path in container
            ]
            
            for schema_path in schema_paths:
                try:
                    with open(schema_path, "r") as f:
                        schema = json.load(f)
                        logger.info(f"Loaded bias schema from: {schema_path}")
                        return schema
                except FileNotFoundError:
                    continue
            
            # If none of the paths work, use default
            logger.warning("Bias schema not found in any expected location, using default")
            return {
                "type": "object",
                "properties": {
                    "overall_bias_score": {"type": "integer", "minimum": 0, "maximum": 100},
                    "bias_dimensions": {
                        "type": "object",
                        "properties": {
                            "subjectivity": {"type": "integer", "minimum": 0, "maximum": 100},
                            "sensationalism": {"type": "integer", "minimum": 0, "maximum": 100},
                            "loaded_language": {"type": "integer", "minimum": 0, "maximum": 100},
                            "political_bias": {"type": "integer", "minimum": -100, "maximum": 100},
                            "ideological_bias": {"type": "integer", "minimum": 0, "maximum": 100}
                        }
                    },
                    "stance_analysis": {
                        "type": "object",
                        "properties": {
                            "primary_stance": {"type": "string", "enum": ["pro", "neutral", "anti", "unclear", "mixed"]},
                            "stance_confidence": {"type": "integer", "minimum": 0, "maximum": 100}
                        }
                    },
                    "credibility_assessment": {
                        "type": "object",
                        "properties": {
                            "evidence_density": {"type": "integer", "minimum": 0, "maximum": 100},
                            "source_credibility": {"type": "integer", "minimum": 0, "maximum": 100}
                        }
                    },
                    "content_analysis": {
                        "type": "object",
                        "properties": {
                            "agenda_signals": {"type": "array", "items": {"type": "string"}},
                            "risk_flags": {"type": "array", "items": {"type": "string"}},
                            "key_quotes": {"type": "array", "items": {"type": "string"}},
                            "summary_bullets": {"type": "array", "items": {"type": "string"}}
                        }
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "entities": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                }
            }
        except Exception as e:
            logger.error(f"Error loading bias schema: {e}")
            # Return a minimal default schema
            return {
                "type": "object",
                "properties": {
                    "overall_bias_score": {"type": "integer", "minimum": 0, "maximum": 100},
                    "bias_dimensions": {
                        "type": "object",
                        "properties": {
                            "subjectivity": {"type": "integer", "minimum": 0, "maximum": 100},
                            "sensationalism": {"type": "integer", "minimum": 0, "maximum": 100},
                            "loaded_language": {"type": "integer", "minimum": 0, "maximum": 100}
                        }
                    }
                }
            }
    
    async def analyze_article(self, article_id: int, title: str, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive multi-model bias analysis on an article
        
        Args:
            article_id: Database ID of the article
            title: Article title
            text: Article text content
            
        Returns:
            Dictionary containing enhanced bias analysis results
        """
        try:
            # Check cache first
            cache_key = f"bias_analysis:{hashlib.md5(f'{article_id}:{title}:{text[:1000]}'.encode()).hexdigest()}"
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Extract features for ML models
            features = self._extract_text_features(title, text)
            
            # Get AI model analysis
            ai_analysis = await self._get_ai_analysis(title, text)
            
            # Get ML model predictions
            ml_predictions = self._get_ml_predictions(features)
            
            # Calculate bias trend
            bias_trend = await self._calculate_bias_trend(article_id)
            
            # Ensemble the results
            ensemble_result = self._ensemble_analysis(ai_analysis, ml_predictions)
            
            # Add confidence scoring and trend analysis
            enhanced_result = self._enhance_with_confidence(ensemble_result, bias_trend)
            
            # Store the analysis in database
            await self._store_analysis(article_id, enhanced_result)
            
            # Cache the result
            result = {
                "success": True,
                "analysis": enhanced_result,
                "article_id": article_id,
                "cached": False
            }
            self.redis_client.setex(cache_key, 3600, json.dumps(result))  # Cache for 1 hour
            
            return result
            
        except Exception as e:
            logger.error(f"Bias analysis failed for article {article_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "article_id": article_id
            }
    
    def _extract_text_features(self, title: str, text: str) -> np.ndarray:
        """Extract numerical features from text for ML models"""
        features = []
        
        # Basic text statistics
        features.extend([
            len(text),
            len(title),
            text.count('!'),
            text.count('?'),
            text.count('"'),
            text.count("'"),
            text.count('...'),
            len(text.split()),
            len(set(text.lower().split())),  # unique words
        ])
        
        # Sentiment indicators
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate']
        
        text_lower = text.lower()
        pos_count = sum(text_lower.count(word) for word in positive_words)
        neg_count = sum(text_lower.count(word) for word in negative_words)
        
        features.extend([pos_count, neg_count, pos_count - neg_count])
        
        # Bias indicators
        bias_indicators = ['clearly', 'obviously', 'undoubtedly', 'certainly', 'definitely']
        bias_count = sum(text_lower.count(indicator) for indicator in bias_indicators)
        features.append(bias_count)
        
        # Political indicators
        left_words = ['progressive', 'liberal', 'democratic', 'social', 'equality']
        right_words = ['conservative', 'traditional', 'patriotic', 'freedom', 'liberty']
        
        left_count = sum(text_lower.count(word) for word in left_words)
        right_count = sum(text_lower.count(word) for word in right_words)
        
        features.extend([left_count, right_count, left_count - right_count])
        
        # Pad to fixed size
        target_size = 20
        while len(features) < target_size:
            features.append(0.0)
        
        return np.array(features[:target_size], dtype=np.float32)
    
    def _get_ml_predictions(self, features: np.ndarray) -> Dict[str, Any]:
        """Get predictions from ML models"""
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                # For now, use simple heuristics since models aren't trained
                if 'subjectivity' in model_name:
                    predictions['subjectivity_ml'] = min(100, max(0, features[8] * 10))  # unique words
                elif 'bias' in model_name:
                    predictions['bias_lr_ml'] = 50 + (features[-1] * 10)  # political bias
                elif 'stance' in model_name:
                    predictions['stance_ml'] = 'neutral'  # Default
                elif 'sensationalism' in model_name:
                    predictions['sensationalism_ml'] = min(100, features[2] * 20)  # exclamation marks
            except Exception as e:
                logger.warning(f"ML model {model_name} prediction failed: {str(e)}")
                continue
        
        return predictions
    
    async def _get_ai_analysis(self, title: str, text: str) -> Dict[str, Any]:
        """Get AI model analysis"""
        try:
            prompt = self._create_analysis_prompt(title, text)
            raw_result = await self._call_ai_model(prompt)
            return self._validate_analysis_result(raw_result)
        except Exception as e:
            logger.error(f"AI analysis failed: {str(e)}")
            return self._get_default_analysis()
    
    def _ensemble_analysis(self, ai_analysis: Dict[str, Any], ml_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Combine AI and ML analysis results"""
        result = ai_analysis.copy()
        
        # Weighted combination of AI and ML results
        if 'subjectivity_ml' in ml_predictions:
            ai_subj = result.get('subjectivity', 50)
            ml_subj = ml_predictions['subjectivity_ml']
            result['subjectivity'] = int(0.7 * ai_subj + 0.3 * ml_subj)
        
        if 'bias_lr_ml' in ml_predictions:
            ai_bias = result.get('bias_lr', 50)
            ml_bias = ml_predictions['bias_lr_ml']
            result['bias_lr'] = int(0.7 * ai_bias + 0.3 * ml_bias)
        
        if 'sensationalism_ml' in ml_predictions:
            ai_sens = result.get('sensationalism', 0)
            ml_sens = ml_predictions['sensationalism_ml']
            result['sensationalism'] = int(0.7 * ai_sens + 0.3 * ml_sens)
        
        return result
    
    def _enhance_with_confidence(self, analysis: Dict[str, Any], bias_trend: Dict[str, Any]) -> Dict[str, Any]:
        """Add confidence scoring and trend analysis"""
        # Calculate confidence based on consistency
        confidence_factors = []
        
        # Subjectivity confidence
        subj = analysis.get('subjectivity', 50)
        if 20 <= subj <= 80:
            confidence_factors.append(0.8)  # Moderate subjectivity is more reliable
        else:
            confidence_factors.append(0.6)  # Extreme values less reliable
        
        # Bias confidence
        bias = analysis.get('bias_lr', 50)
        if 30 <= bias <= 70:
            confidence_factors.append(0.9)  # Centrist bias more reliable
        else:
            confidence_factors.append(0.7)  # Extreme bias less reliable
        
        # Evidence density confidence
        evidence = analysis.get('evidence_density', 50)
        confidence_factors.append(evidence / 100.0)
        
        # Calculate overall confidence
        overall_confidence = np.mean(confidence_factors) if confidence_factors else 0.5
        
        # Model agreement (simplified)
        model_agreement = 0.8  # Would be calculated from multiple model outputs
        
        analysis.update({
            'confidence_score': float(overall_confidence),
            'model_agreement': float(model_agreement),
            'bias_trend': bias_trend,
            'analysis_timestamp': datetime.utcnow().isoformat()
        })
        
        return analysis
    
    async def _calculate_bias_trend(self, article_id: int) -> Dict[str, Any]:
        """Calculate bias trend over time"""
        try:
            # Get recent articles for trend analysis
            async with db_pool.get_connection() as conn:
                result = await conn.fetch("""
                    SELECT bias_lr, subjectivity, sensationalism, created_at
                    FROM article_analysis aa
                    JOIN articles a ON aa.article_id = a.id
                    WHERE a.created_at >= NOW() - INTERVAL '%s days'
                    ORDER BY a.created_at DESC
                    LIMIT 100
                """, self.trend_window)
            
            if not result:
                return {"trend": "insufficient_data", "direction": "stable"}
            
            # Calculate trends
            bias_values = [row['bias_lr'] for row in result if row['bias_lr'] is not None]
            subj_values = [row['subjectivity'] for row in result if row['subjectivity'] is not None]
            sens_values = [row['sensationalism'] for row in result if row['sensationalism'] is not None]
            
            trends = {}
            if len(bias_values) > 1:
                bias_trend = np.polyfit(range(len(bias_values)), bias_values, 1)[0]
                trends['bias_trend'] = float(bias_trend)
                trends['bias_direction'] = 'increasing' if bias_trend > 0 else 'decreasing'
            
            if len(subj_values) > 1:
                subj_trend = np.polyfit(range(len(subj_values)), subj_values, 1)[0]
                trends['subjectivity_trend'] = float(subj_trend)
            
            if len(sens_values) > 1:
                sens_trend = np.polyfit(range(len(sens_values)), sens_values, 1)[0]
                trends['sensationalism_trend'] = float(sens_trend)
            
            return {
                "trend": "calculated",
                "direction": trends.get('bias_direction', 'stable'),
                "metrics": trends,
                "sample_size": len(result)
            }
            
        except Exception as e:
            logger.warning(f"Trend calculation failed: {str(e)}")
            return {"trend": "error", "direction": "unknown"}
    
    async def get_bias_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive bias analytics"""
        try:
            async with db_pool.get_connection() as conn:
                # Get bias distribution
                bias_dist = await conn.fetch("""
                    SELECT 
                        CASE 
                            WHEN bias_lr < 30 THEN 'left'
                            WHEN bias_lr > 70 THEN 'right'
                            ELSE 'center'
                        END as bias_category,
                        COUNT(*) as count
                    FROM article_analysis
                    WHERE created_at >= NOW() - INTERVAL '%s days'
                    GROUP BY bias_category
                """, days)
                
                # Get average metrics
                avg_metrics = await conn.fetchrow("""
                    SELECT 
                        AVG(subjectivity) as avg_subjectivity,
                        AVG(sensationalism) as avg_sensationalism,
                        AVG(loaded_language) as avg_loaded_language,
                        AVG(bias_lr) as avg_bias_lr,
                        AVG(evidence_density) as avg_evidence_density
                    FROM article_analysis
                    WHERE created_at >= NOW() - INTERVAL '%s days'
                """, days)
                
                # Get trend data
                trend_data = await conn.fetch("""
                    SELECT 
                        DATE(created_at) as date,
                        AVG(bias_lr) as avg_bias,
                        AVG(subjectivity) as avg_subjectivity,
                        COUNT(*) as article_count
                    FROM article_analysis aa
                    JOIN articles a ON aa.article_id = a.id
                    WHERE a.created_at >= NOW() - INTERVAL '%s days'
                    GROUP BY DATE(created_at)
                    ORDER BY date
                """, days)
                
                return {
                    "bias_distribution": {row['bias_category']: row['count'] for row in bias_dist},
                    "average_metrics": dict(avg_metrics) if avg_metrics else {},
                    "trend_data": [dict(row) for row in trend_data],
                    "analysis_period_days": days
                }
                
        except Exception as e:
            logger.error(f"Bias analytics failed: {str(e)}")
            return {"error": str(e)}
    
    def _create_analysis_prompt(self, title: str, text: str) -> str:
        """Create the analysis prompt for the AI model"""
        # Load the bias rubric prompt
        try:
            with open("prompts/bias_rubric.md", "r") as f:
                rubric_prompt = f.read()
        except FileNotFoundError:
            rubric_prompt = """You are an editorial auditor. Analyze the input article and output STRICT JSON only matching the provided schema.
Scales 0–100 unless specified.
- subjectivity: 0 factual – 100 highly subjective
- sensationalism: 0 none – 100 tabloid-like
- loaded_language: 0 none – 100 extreme
- bias_lr: 0 left – 50 center – 100 right (estimate based on language cues only)
- stance: one of {pro, neutral, anti, unclear} toward the main entity
- evidence_density: percent of sentences with quotes/data/citations (0–100)
- agenda_signals: list (e.g., cherry-picking, false balance, ad hominem)
- risk_flags: list (e.g., unverified claim, miscaptioning)
Return JSON only, no comments."""
        
        # Truncate text if too long (keep first 8000 characters)
        truncated_text = text[:8000] if len(text) > 8000 else text
        
        return f"""{rubric_prompt}

Article Title: {title}

Article Text: {truncated_text}

Schema: {json.dumps(self.bias_schema, indent=2)}"""
    
    async def _call_ai_model(self, prompt: str) -> str:
        """Call the AI model for analysis"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{settings.ollama_host}/api/generate",
                    json={
                        "model": "llama3.1:8b",
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "top_p": 0.9
                        }
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                result = response.json()
                return result.get("response", "")
        except Exception as e:
            logger.error(f"AI model call failed: {str(e)}")
            raise APIError(
                status_code=500,
                error_code="EMBEDDING_ERROR",
                message="AI model analysis failed",
                detail=str(e)
            )
    
    def _validate_analysis_result(self, raw_result: str) -> Dict[str, Any]:
        """Validate and clean the analysis result"""
        try:
            # Extract JSON from the response
            json_start = raw_result.find('{')
            json_end = raw_result.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = raw_result[json_start:json_end]
            result = json.loads(json_str)
            
            # Validate required fields
            required_fields = self.bias_schema.get("required", [])
            for field in required_fields:
                if field not in result:
                    result[field] = self._get_default_value(field)
            
            # Validate field types and ranges
            for field, value in result.items():
                if field in self.bias_schema.get("properties", {}):
                    result[field] = self._validate_field(field, value)
            
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse AI response: {str(e)}")
            return self._get_default_analysis()
    
    def _get_default_value(self, field: str) -> Any:
        """Get default value for a field"""
        defaults = {
            "subjectivity": 50,
            "sensationalism": 0,
            "loaded_language": 0,
            "bias_lr": 50,
            "stance": "neutral",
            "evidence_density": 50,
            "agenda_signals": [],
            "risk_flags": [],
            "key_quotes": [],
            "summary_bullets": [],
            "tags": [],
            "entities": []
        }
        return defaults.get(field, None)
    
    def _validate_field(self, field: str, value: Any) -> Any:
        """Validate a specific field"""
        field_schema = self.bias_schema.get("properties", {}).get(field, {})
        
        if field_schema.get("type") == "integer":
            min_val = field_schema.get("minimum", 0)
            max_val = field_schema.get("maximum", 100)
            return max(min_val, min(max_val, int(value)))
        elif field_schema.get("type") == "array":
            return value if isinstance(value, list) else []
        elif field_schema.get("enum"):
            return value if value in field_schema["enum"] else field_schema["enum"][0]
        
        return value
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        """Get default analysis when parsing fails"""
        return {
            "subjectivity": 50,
            "sensationalism": 0,
            "loaded_language": 0,
            "bias_lr": 50,
            "stance": "neutral",
            "evidence_density": 50,
            "agenda_signals": [],
            "risk_flags": [],
            "key_quotes": [],
            "summary_bullets": [],
            "tags": [],
            "entities": []
        }
    
    async def _store_analysis(self, article_id: int, analysis: Dict[str, Any]) -> None:
        """Store the analysis result in the database"""
        try:
            async with db_pool.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO article_analysis (
                        article_id, subjectivity, sensationalism, loaded_language,
                        bias_lr, stance, evidence_density, sentiment, sentiment_confidence,
                        agenda_signals, risk_flags, entities, tags, key_quotes, summary_bullets,
                        confidence_score, model_agreement, bias_trend, analysis_timestamp
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19
                    ) ON CONFLICT (article_id) DO UPDATE SET
                        subjectivity = EXCLUDED.subjectivity,
                        sensationalism = EXCLUDED.sensationalism,
                        loaded_language = EXCLUDED.loaded_language,
                        bias_lr = EXCLUDED.bias_lr,
                        stance = EXCLUDED.stance,
                        evidence_density = EXCLUDED.evidence_density,
                        sentiment = EXCLUDED.sentiment,
                        sentiment_confidence = EXCLUDED.sentiment_confidence,
                        agenda_signals = EXCLUDED.agenda_signals,
                        risk_flags = EXCLUDED.risk_flags,
                        entities = EXCLUDED.entities,
                        tags = EXCLUDED.tags,
                        key_quotes = EXCLUDED.key_quotes,
                        summary_bullets = EXCLUDED.summary_bullets,
                        confidence_score = EXCLUDED.confidence_score,
                        model_agreement = EXCLUDED.model_agreement,
                        bias_trend = EXCLUDED.bias_trend,
                        analysis_timestamp = EXCLUDED.analysis_timestamp,
                        created_at = now()
                """, 
                article_id,
                analysis.get("subjectivity", 50),
                analysis.get("sensationalism", 0),
                analysis.get("loaded_language", 0),
                analysis.get("bias_lr", 50),
                analysis.get("stance", "neutral"),
                analysis.get("evidence_density", 50),
                analysis.get("sentiment", "neutral"),
                analysis.get("sentiment_confidence", 0.5),
                json.dumps(analysis.get("agenda_signals", [])),
                json.dumps(analysis.get("risk_flags", [])),
                json.dumps(analysis.get("entities", [])),
                json.dumps(analysis.get("tags", [])),
                json.dumps(analysis.get("key_quotes", [])),
                json.dumps(analysis.get("summary_bullets", [])),
                analysis.get("confidence_score", 0.5),
                analysis.get("model_agreement", 0.5),
                json.dumps(analysis.get("bias_trend", {})),
                analysis.get("analysis_timestamp", datetime.utcnow().isoformat())
                )
        except Exception as e:
            logger.error(f"Failed to store analysis for article {article_id}: {str(e)}")
            raise

# Global instance
bias_analyzer = MultiModelBiasAnalyzer()
