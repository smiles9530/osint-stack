"""
Advanced Stance, Sentiment, and Bias Analysis Service
Implements specialized models for comprehensive content analysis
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
import json
import hashlib
import re
from dataclasses import dataclass
from collections import defaultdict, Counter

import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    pipeline, AutoModel
)
from .gpu_utils import gpu_manager, model_device_manager, log_gpu_status
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"sentence-transformers not available: {e}")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
import redis
import httpx

from .config import settings
from .db_pool import db_pool
from .enhanced_error_handling import APIError, ErrorHandler

logger = logging.getLogger("osint_api")

@dataclass
class AnalysisResult:
    """Comprehensive analysis result for a text chunk"""
    chunk_id: str
    text: str
    sentiment: Dict[str, float]
    stance: Dict[str, float]
    toxicity: Dict[str, float]
    bias: Dict[str, float]
    confidence: float
    timestamp: datetime

@dataclass
class AggregatedAnalysis:
    """Aggregated analysis results by source/topic"""
    source_id: Optional[str]
    topic: Optional[str]
    total_chunks: int
    sentiment_distribution: Dict[str, float]
    stance_distribution: Dict[str, float]
    toxicity_levels: Dict[str, float]
    bias_scores: Dict[str, float]
    confidence_avg: float
    risk_flags: List[str]
    trend_direction: str
    timestamp: datetime

class StanceSentimentBiasAnalyzer:
    """
    Specialized transformer-based analyzer for sentiment, stance, and political bias detection
    
    This analyzer focuses on:
    - Sentiment analysis using Twitter RoBERTa
    - Stance detection using NLI models (supports/refutes/neutral)
    - Toxicity detection using unbiased RoBERTa
    - Political bias using embedding similarity (complementary to bias_analysis.py)
    
    Note: Comprehensive bias analysis (subjectivity, sensationalism, loaded language, etc.)
    is handled by the existing bias_analysis.py module using AI + ML ensemble methods.
    """
    
    def __init__(self):
        # Model placeholders - will be loaded lazily
        self.sentiment_model = None
        self.sentiment_tokenizer = None
        self.sentiment_pipeline = None
        self.nli_model = None
        self.nli_tokenizer = None
        self.nli_pipeline = None
        self.toxicity_model = None
        self.toxicity_tokenizer = None
        self.toxicity_pipeline = None
        self.embedding_model = None
        self.embedding_tokenizer = None
        
        # Model loading flags
        self._models_loaded = {
            'sentiment': False,
            'nli': False,
            'toxicity': False,
            'embedding': False
        }
        
        # Redis for caching
        self.redis_client = self._init_redis()
        
        # Analysis configuration
        self.chunk_size = 512  # tokens
        self.chunk_overlap = 50  # tokens
        self.confidence_threshold = 0.7
        self.alert_thresholds = {
            'sentiment_shift': 0.3,  # 30% shift in sentiment
            'stance_change': 0.4,   # 40% change in stance distribution
            'toxicity_spike': 0.2,  # 20% increase in toxicity
            'bias_extreme': 0.8     # 80% extreme bias
        }
        
        # Don't initialize models at startup - use lazy loading
        logger.info("StanceSentimentBiasAnalyzer initialized with lazy loading")
    
    async def preload_models(self):
        """Preload all models (optional - for performance optimization)"""
        try:
            logger.info("Preloading all analysis models...")
            self._ensure_model_loaded('sentiment')
            self._ensure_model_loaded('nli')
            self._ensure_model_loaded('toxicity')
            self._ensure_model_loaded('embedding')
            logger.info("All models preloaded successfully")
        except Exception as e:
            logger.error(f"Model preloading failed: {e}")
            raise
    
    def _init_redis(self):
        """Initialize Redis client for caching"""
        try:
            redis_url = settings.redis_url.replace("redis://", "")
            if "/" in redis_url:
                url_part, db_part = redis_url.split("/", 1)
                if ":" in url_part:
                    host, port = url_part.split(":")
                    port = int(port)
                else:
                    host = url_part
                    port = 6379
            else:
                if ":" in redis_url:
                    host, port = redis_url.split(":")
                    port = int(port)
                else:
                    host = redis_url
                    port = 6379
            
            return redis.Redis(host=host, port=port, db=4, decode_responses=True)
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}")
            return None
    
    def _ensure_model_loaded(self, model_type: str):
        """Ensure a specific model is loaded (lazy loading)"""
        if not self._models_loaded.get(model_type, False):
            try:
                if model_type == 'sentiment':
                    self._load_sentiment_model()
                elif model_type == 'nli':
                    self._load_nli_model()
                elif model_type == 'toxicity':
                    self._load_toxicity_model()
                elif model_type == 'embedding':
                    self._load_embedding_model()
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                self._models_loaded[model_type] = True
                logger.info(f"Lazy loaded {model_type} model")
                
            except Exception as e:
                logger.error(f"Failed to lazy load {model_type} model: {e}")
                raise
    
    def _load_sentiment_model(self):
        """Load Twitter RoBERTa sentiment model with GPU acceleration"""
        try:
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            device = model_device_manager.get_model_device(model_name)
            
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Move model to appropriate device
            self.sentiment_model = model_device_manager.move_model_to_device(
                self.sentiment_model, model_name
            )
            
            # Create pipeline for easy inference with device specification
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.sentiment_model,
                tokenizer=self.sentiment_tokenizer,
                device=0 if gpu_manager.is_gpu_available() else -1,
                return_all_scores=True
            )
            
            logger.info(f"Loaded sentiment model: {model_name} on {device}")
            log_gpu_status()
            
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            raise
    
    def _load_nli_model(self):
        """Load RoBERTa NLI model for stance detection with GPU acceleration"""
        try:
            model_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
            device = model_device_manager.get_model_device(model_name)
            
            self.nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Move model to appropriate device
            self.nli_model = model_device_manager.move_model_to_device(
                self.nli_model, model_name
            )
            
            # Create pipeline for NLI with device specification
            self.nli_pipeline = pipeline(
                "text-classification",
                model=self.nli_model,
                tokenizer=self.nli_tokenizer,
                device=0 if gpu_manager.is_gpu_available() else -1,
                return_all_scores=True
            )
            
            logger.info(f"Loaded NLI model: {model_name} on {device}")
            log_gpu_status()
            
        except Exception as e:
            logger.error(f"Failed to load NLI model: {e}")
            raise
    
    def _load_toxicity_model(self):
        """Load toxicity detection model with GPU acceleration"""
        try:
            model_name = "unitary/unbiased-toxic-roberta"
            device = model_device_manager.get_model_device(model_name)
            
            self.toxicity_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.toxicity_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Move model to appropriate device
            self.toxicity_model = model_device_manager.move_model_to_device(
                self.toxicity_model, model_name
            )
            
            # Create pipeline for toxicity detection with device specification
            self.toxicity_pipeline = pipeline(
                "text-classification",
                model=self.toxicity_model,
                tokenizer=self.toxicity_tokenizer,
                device=0 if gpu_manager.is_gpu_available() else -1,
                return_all_scores=True
            )
            
            logger.info(f"Loaded toxicity model: {model_name} on {device}")
            log_gpu_status()
            
        except Exception as e:
            logger.error(f"Failed to load toxicity model: {e}")
            raise
    
    def _load_embedding_model(self):
        """Load embedding model for bias analysis with GPU acceleration"""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                # Use sentence-transformers for better performance
                model_name = 'BAAI/bge-m3'
                device = model_device_manager.get_model_device(model_name)
                
                self.embedding_model = SentenceTransformer(model_name)
                
                # Move to appropriate device
                if gpu_manager.is_gpu_available():
                    self.embedding_model = self.embedding_model.to(device)
                
                logger.info(f"Loaded embedding model: {model_name} on {device} (sentence-transformers)")
            else:
                # Fallback to transformers directly
                from transformers import AutoTokenizer, AutoModel
                model_name = 'sentence-transformers/all-MiniLM-L6-v2'
                device = model_device_manager.get_model_device(model_name)
                
                self.embedding_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.embedding_model = AutoModel.from_pretrained(model_name)
                
                # Move to appropriate device
                self.embedding_model = model_device_manager.move_model_to_device(
                    self.embedding_model, model_name
                )
                
                logger.info(f"Loaded embedding model: {model_name} on {device} (transformers fallback)")
            
            log_gpu_status()
            
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            self.embedding_model = None
    
    async def analyze_text_batch(
        self, 
        texts: List[str], 
        article_ids: Optional[List[int]] = None,
        source_ids: Optional[List[str]] = None,
        topics: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple texts in batch for better GPU utilization
        
        Args:
            texts: List of texts to analyze
            article_ids: Optional list of article IDs
            source_ids: Optional list of source identifiers
            topics: Optional list of topic identifiers
            
        Returns:
            List of comprehensive analysis results
        """
        try:
            if not texts:
                return []
            
            # Use GPU memory management
            with gpu_manager.memory_management():
                # Process in optimal batch sizes
                optimal_batch_size = gpu_manager.get_optimal_batch_size(500)  # ~500MB model
                batch_size = min(optimal_batch_size, len(texts))
                
                results = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_article_ids = article_ids[i:i + batch_size] if article_ids else None
                    batch_source_ids = source_ids[i:i + batch_size] if source_ids else None
                    batch_topics = topics[i:i + batch_size] if topics else None
                    
                    # Process batch
                    batch_results = await self._analyze_text_batch(
                        batch_texts, batch_article_ids, batch_source_ids, batch_topics
                    )
                    results.extend(batch_results)
                
                logger.info(f"Processed {len(texts)} texts in {len(results)} batches")
                return results
                
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            # Fallback to individual processing
            return [await self.analyze_text(text, aid, sid, topic) 
                   for text, aid, sid, topic in zip(texts, 
                   article_ids or [None] * len(texts),
                   source_ids or [None] * len(texts),
                   topics or [None] * len(texts))]

    async def analyze_text(
        self, 
        text: str, 
        article_id: Optional[int] = None,
        source_id: Optional[str] = None,
        topic: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on text
        
        Args:
            text: Text to analyze
            article_id: Optional article ID
            source_id: Optional source identifier
            topic: Optional topic identifier
            
        Returns:
            Comprehensive analysis results
        """
        try:
            # Check cache first
            cache_key = self._get_cache_key(text, article_id)
            if self.redis_client:
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    return json.loads(cached_result)
            
            # Use GPU memory management
            with gpu_manager.memory_management():
                # Split text into chunks
                chunks = self._split_into_chunks(text)
                
                # Analyze each chunk
                chunk_results = []
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{article_id}_{i}" if article_id else f"chunk_{i}"
                    result = await self._analyze_chunk(chunk_id, chunk)
                    chunk_results.append(result)
                
                # Aggregate results
                aggregated = self._aggregate_results(chunk_results, source_id, topic)
            
            # Generate alerts
            alerts = self._generate_alerts(aggregated, source_id, topic)
            
            # Store results
            if article_id:
                await self._store_analysis_results(article_id, chunk_results, aggregated)
            
            # Cache results
            final_result = {
                "chunks": [self._result_to_dict(r) for r in chunk_results],
                "aggregated": self._aggregated_to_dict(aggregated),
                "alerts": alerts,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
            if self.redis_client:
                self.redis_client.setex(cache_key, 3600, json.dumps(final_result))
            
            return final_result
            
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            raise
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks for analysis"""
        # Simple sentence-based chunking
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk.split()) + len(sentence.split()) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def _analyze_chunk(self, chunk_id: str, text: str) -> AnalysisResult:
        """Analyze a single text chunk"""
        try:
            # Run all analyses in parallel
            sentiment_task = self._analyze_sentiment(text)
            stance_task = self._analyze_stance(text)
            toxicity_task = self._analyze_toxicity(text)
            bias_task = self._analyze_bias(text)
            
            # Wait for all analyses to complete
            sentiment, stance, toxicity, bias = await asyncio.gather(
                sentiment_task, stance_task, toxicity_task, bias_task
            )
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(sentiment, stance, toxicity, bias)
            
            return AnalysisResult(
                chunk_id=chunk_id,
                text=text,
                sentiment=sentiment,
                stance=stance,
                toxicity=toxicity,
                bias=bias,
                confidence=confidence,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Chunk analysis failed: {e}")
            # Return default values on error
            return AnalysisResult(
                chunk_id=chunk_id,
                text=text,
                sentiment={"positive": 0.33, "negative": 0.33, "neutral": 0.34},
                stance={"supports": 0.33, "refutes": 0.33, "neutral": 0.34},
                toxicity={"toxic": 0.0, "non-toxic": 1.0},
                bias={"left": 0.33, "center": 0.34, "right": 0.33},
                confidence=0.0,
                timestamp=datetime.utcnow()
            )
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using Twitter RoBERTa"""
        try:
            # Ensure sentiment model is loaded
            self._ensure_model_loaded('sentiment')
            
            # Truncate text if too long
            if len(text) > 512:
                text = text[:512]
            
            results = self.sentiment_pipeline(text)
            
            # Convert to our format
            sentiment = {}
            for result in results[0]:
                label = result['label'].lower()
                score = result['score']
                
                if 'positive' in label:
                    sentiment['positive'] = score
                elif 'negative' in label:
                    sentiment['negative'] = score
                elif 'neutral' in label:
                    sentiment['neutral'] = score
            
            # Ensure all keys exist
            sentiment.setdefault('positive', 0.0)
            sentiment.setdefault('negative', 0.0)
            sentiment.setdefault('neutral', 0.0)
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
    
    async def _analyze_stance(self, text: str) -> Dict[str, float]:
        """Analyze stance using NLI model"""
        try:
            # Ensure NLI model is loaded
            self._ensure_model_loaded('nli')
            
            # For stance detection, we need to define what we're checking stance against
            # This is a simplified version - in practice, you'd have specific claims/topics
            claims = [
                "This statement supports the main argument",
                "This statement refutes the main argument", 
                "This statement is neutral regarding the main argument"
            ]
            
            stance_scores = {"supports": 0.0, "refutes": 0.0, "neutral": 0.0}
            
            for claim in claims:
                # Create premise-hypothesis pairs
                premise = text[:256]  # Truncate for NLI
                hypothesis = claim
                
                # Run NLI
                result = self.nli_pipeline(f"{premise} [SEP] {hypothesis}")
                
                # Map NLI labels to stance
                for r in result[0]:
                    label = r['label'].lower()
                    score = r['score']
                    
                    if 'entailment' in label or 'support' in label:
                        stance_scores['supports'] += score
                    elif 'contradiction' in label or 'refute' in label:
                        stance_scores['refutes'] += score
                    elif 'neutral' in label:
                        stance_scores['neutral'] += score
            
            # Normalize scores
            total = sum(stance_scores.values())
            if total > 0:
                stance_scores = {k: v/total for k, v in stance_scores.items()}
            else:
                stance_scores = {"supports": 0.33, "refutes": 0.33, "neutral": 0.34}
            
            return stance_scores
            
        except Exception as e:
            logger.error(f"Stance analysis failed: {e}")
            return {"supports": 0.33, "refutes": 0.33, "neutral": 0.34}
    
    async def _analyze_toxicity(self, text: str) -> Dict[str, float]:
        """Analyze toxicity using unbiased RoBERTa"""
        try:
            # Ensure toxicity model is loaded
            self._ensure_model_loaded('toxicity')
            
            # Truncate text if too long
            if len(text) > 512:
                text = text[:512]
            
            results = self.toxicity_pipeline(text)
            
            # Convert to our format
            toxicity = {"toxic": 0.0, "non-toxic": 0.0}
            
            for result in results[0]:
                label = result['label'].lower()
                score = result['score']
                
                if 'toxic' in label or 'hate' in label:
                    toxicity['toxic'] = max(toxicity['toxic'], score)
                else:
                    toxicity['non-toxic'] = max(toxicity['non-toxic'], score)
            
            # Ensure non-toxic is at least 1 - toxic
            toxicity['non-toxic'] = max(toxicity['non-toxic'], 1.0 - toxicity['toxic'])
            
            return toxicity
            
        except Exception as e:
            logger.error(f"Toxicity analysis failed: {e}")
            return {"toxic": 0.0, "non-toxic": 1.0}
    
    async def _analyze_bias(self, text: str) -> Dict[str, float]:
        """Analyze political bias using embedding similarity (complementary to bias_analysis.py)"""
        try:
            # Ensure embedding model is loaded
            self._ensure_model_loaded('embedding')
            
            # This is a lightweight political bias indicator using embeddings
            # The comprehensive bias analysis is handled by bias_analysis.py
            
            if self.embedding_model is None:
                # Fallback to simple keyword-based analysis
                return self._simple_bias_analysis(text)
            
            # Define political bias indicators
            left_indicators = [
                "progressive", "liberal", "democratic", "social justice", "equality",
                "diversity", "inclusion", "environmental", "climate change"
            ]
            right_indicators = [
                "conservative", "traditional", "patriotic", "freedom", "liberty",
                "individual", "free market", "family values", "law and order"
            ]
            
            # Get text embedding
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                text_embedding = self.embedding_model.encode([text])
                left_embeddings = self.embedding_model.encode(left_indicators)
                right_embeddings = self.embedding_model.encode(right_indicators)
            else:
                # Use transformers directly
                inputs = self.embedding_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.embedding_model(**inputs)
                    text_embedding = outputs.last_hidden_state.mean(dim=1).numpy()
                
                # Get indicator embeddings
                left_inputs = self.embedding_tokenizer(left_indicators, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    left_outputs = self.embedding_model(**left_inputs)
                    left_embeddings = left_outputs.last_hidden_state.mean(dim=1).numpy()
                
                right_inputs = self.embedding_tokenizer(right_indicators, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    right_outputs = self.embedding_model(**right_inputs)
                    right_embeddings = right_outputs.last_hidden_state.mean(dim=1).numpy()
            
            # Calculate similarities
            left_similarity = np.max(np.dot(text_embedding, left_embeddings.T))
            right_similarity = np.max(np.dot(text_embedding, right_embeddings.T))
            
            # Normalize to probabilities
            total = left_similarity + right_similarity + 0.1  # Add small value to avoid division by zero
            left_score = left_similarity / total
            right_score = right_similarity / total
            center_score = 1.0 - left_score - right_score
            
            return {
                "left": float(left_score),
                "center": float(center_score),
                "right": float(right_score)
            }
            
        except Exception as e:
            logger.error(f"Political bias analysis failed: {e}")
            return self._simple_bias_analysis(text)
    
    def _simple_bias_analysis(self, text: str) -> Dict[str, float]:
        """Simple keyword-based bias analysis fallback"""
        text_lower = text.lower()
        
        # Define political bias keywords
        left_keywords = [
            "progressive", "liberal", "democratic", "social justice", "equality",
            "diversity", "inclusion", "environmental", "climate change", "reform"
        ]
        right_keywords = [
            "conservative", "traditional", "patriotic", "freedom", "liberty",
            "individual", "free market", "family values", "law and order", "reform"
        ]
        
        # Count keyword occurrences
        left_count = sum(1 for keyword in left_keywords if keyword in text_lower)
        right_count = sum(1 for keyword in right_keywords if keyword in text_lower)
        
        # Calculate scores
        total = left_count + right_count + 1  # Add 1 to avoid division by zero
        left_score = left_count / total
        right_score = right_count / total
        center_score = 1.0 - left_score - right_score
        
        return {
            "left": float(left_score),
            "center": float(center_score),
            "right": float(right_score)
        }
    
    def _calculate_confidence(self, sentiment: Dict, stance: Dict, toxicity: Dict, bias: Dict) -> float:
        """Calculate overall confidence score"""
        try:
            # Calculate confidence based on score distributions
            sentiment_conf = 1.0 - min(sentiment.values())  # Higher when one sentiment dominates
            stance_conf = 1.0 - min(stance.values())
            toxicity_conf = max(toxicity.values())  # Higher when clear toxic/non-toxic
            bias_conf = 1.0 - min(bias.values())
            
            # Average confidence
            return (sentiment_conf + stance_conf + toxicity_conf + bias_conf) / 4.0
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    def _aggregate_results(
        self, 
        chunk_results: List[AnalysisResult], 
        source_id: Optional[str] = None,
        topic: Optional[str] = None
    ) -> AggregatedAnalysis:
        """Aggregate chunk results into overall analysis"""
        try:
            # Calculate distributions
            sentiment_dist = self._calculate_distribution([r.sentiment for r in chunk_results])
            stance_dist = self._calculate_distribution([r.stance for r in chunk_results])
            toxicity_levels = self._calculate_distribution([r.toxicity for r in chunk_results])
            bias_scores = self._calculate_distribution([r.bias for r in chunk_results])
            
            # Calculate average confidence
            avg_confidence = np.mean([r.confidence for r in chunk_results])
            
            # Generate risk flags
            risk_flags = self._generate_risk_flags(sentiment_dist, stance_dist, toxicity_levels, bias_scores)
            
            # Determine trend direction
            trend_direction = self._determine_trend_direction(sentiment_dist, stance_dist)
            
            return AggregatedAnalysis(
                source_id=source_id,
                topic=topic,
                total_chunks=len(chunk_results),
                sentiment_distribution=sentiment_dist,
                stance_distribution=stance_dist,
                toxicity_levels=toxicity_levels,
                bias_scores=bias_scores,
                confidence_avg=float(avg_confidence),
                risk_flags=risk_flags,
                trend_direction=trend_direction,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return AggregatedAnalysis(
                source_id=source_id,
                topic=topic,
                total_chunks=len(chunk_results),
                sentiment_distribution={"positive": 0.33, "negative": 0.33, "neutral": 0.34},
                stance_distribution={"supports": 0.33, "refutes": 0.33, "neutral": 0.34},
                toxicity_levels={"toxic": 0.0, "non-toxic": 1.0},
                bias_scores={"left": 0.33, "center": 0.34, "right": 0.33},
                confidence_avg=0.5,
                risk_flags=[],
                trend_direction="stable",
                timestamp=datetime.utcnow()
            )
    
    def _calculate_distribution(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate average distribution across results"""
        if not results:
            return {}
        
        # Get all unique keys
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())
        
        # Calculate averages
        distribution = {}
        for key in all_keys:
            values = [result.get(key, 0.0) for result in results]
            distribution[key] = float(np.mean(values))
        
        return distribution
    
    def _generate_risk_flags(
        self, 
        sentiment: Dict[str, float], 
        stance: Dict[str, float], 
        toxicity: Dict[str, float], 
        bias: Dict[str, float]
    ) -> List[str]:
        """Generate risk flags based on analysis results"""
        flags = []
        
        # High toxicity
        if toxicity.get('toxic', 0) > 0.3:
            flags.append("high_toxicity")
        
        # Extreme bias
        if max(bias.values()) > 0.8:
            flags.append("extreme_bias")
        
        # Overwhelming negative sentiment
        if sentiment.get('negative', 0) > 0.7:
            flags.append("overwhelming_negative")
        
        # Strong refutation stance
        if stance.get('refutes', 0) > 0.7:
            flags.append("strong_refutation")
        
        # Low confidence
        if any(score < 0.3 for score in [sentiment.get('positive', 0), sentiment.get('negative', 0), sentiment.get('neutral', 0)]):
            flags.append("low_confidence")
        
        return flags
    
    def _determine_trend_direction(self, sentiment: Dict[str, float], stance: Dict[str, float]) -> str:
        """Determine overall trend direction"""
        # This is simplified - in practice, you'd compare with historical data
        if sentiment.get('positive', 0) > 0.6:
            return "positive"
        elif sentiment.get('negative', 0) > 0.6:
            return "negative"
        elif stance.get('supports', 0) > 0.6:
            return "supportive"
        elif stance.get('refutes', 0) > 0.6:
            return "refutational"
        else:
            return "neutral"
    
    def _generate_alerts(
        self, 
        aggregated: AggregatedAnalysis, 
        source_id: Optional[str] = None,
        topic: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate alerts based on analysis results"""
        alerts = []
        
        # Check for sudden sentiment shifts
        if aggregated.sentiment_distribution.get('negative', 0) > self.alert_thresholds['sentiment_shift']:
            alerts.append({
                "type": "sentiment_shift",
                "severity": "medium",
                "message": f"High negative sentiment detected: {aggregated.sentiment_distribution['negative']:.2f}",
                "source_id": source_id,
                "topic": topic
            })
        
        # Check for stance changes
        if aggregated.stance_distribution.get('refutes', 0) > self.alert_thresholds['stance_change']:
            alerts.append({
                "type": "stance_change",
                "severity": "high",
                "message": f"High refutation stance detected: {aggregated.stance_distribution['refutes']:.2f}",
                "source_id": source_id,
                "topic": topic
            })
        
        # Check for toxicity spikes
        if aggregated.toxicity_levels.get('toxic', 0) > self.alert_thresholds['toxicity_spike']:
            alerts.append({
                "type": "toxicity_spike",
                "severity": "high",
                "message": f"High toxicity detected: {aggregated.toxicity_levels['toxic']:.2f}",
                "source_id": source_id,
                "topic": topic
            })
        
        # Check for extreme bias
        if max(aggregated.bias_scores.values()) > self.alert_thresholds['bias_extreme']:
            alerts.append({
                "type": "extreme_bias",
                "severity": "medium",
                "message": f"Extreme bias detected: {max(aggregated.bias_scores.values()):.2f}",
                "source_id": source_id,
                "topic": topic
            })
        
        return alerts
    
    async def _store_analysis_results(
        self, 
        article_id: int, 
        chunk_results: List[AnalysisResult],
        aggregated: AggregatedAnalysis
    ):
        """Store analysis results in database"""
        try:
            async with db_pool.get_connection() as conn:
                # Store chunk results
                for result in chunk_results:
                    await conn.execute("""
                        INSERT INTO stance_sentiment_analysis (
                            article_id, chunk_id, chunk_text, sentiment_scores, stance_scores,
                            toxicity_scores, bias_scores, confidence, created_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        ON CONFLICT (article_id, chunk_id) DO UPDATE SET
                            sentiment_scores = EXCLUDED.sentiment_scores,
                            stance_scores = EXCLUDED.stance_scores,
                            toxicity_scores = EXCLUDED.toxicity_scores,
                            bias_scores = EXCLUDED.bias_scores,
                            confidence = EXCLUDED.confidence,
                            created_at = EXCLUDED.created_at
                    """, 
                    article_id,
                    result.chunk_id,
                    result.text,
                    json.dumps(result.sentiment),
                    json.dumps(result.stance),
                    json.dumps(result.toxicity),
                    json.dumps(result.bias),
                    result.confidence,
                    result.timestamp
                    )
                
                # Store aggregated results
                await conn.execute("""
                    INSERT INTO aggregated_analysis (
                        article_id, source_id, topic, total_chunks, sentiment_distribution,
                        stance_distribution, toxicity_levels, bias_scores, confidence_avg,
                        risk_flags, trend_direction, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT (article_id) DO UPDATE SET
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
                        created_at = EXCLUDED.created_at
                """,
                article_id,
                aggregated.source_id,
                aggregated.topic,
                aggregated.total_chunks,
                json.dumps(aggregated.sentiment_distribution),
                json.dumps(aggregated.stance_distribution),
                json.dumps(aggregated.toxicity_levels),
                json.dumps(aggregated.bias_scores),
                aggregated.confidence_avg,
                json.dumps(aggregated.risk_flags),
                aggregated.trend_direction,
                aggregated.timestamp
                )
                
        except Exception as e:
            logger.error(f"Failed to store analysis results: {e}")
            raise
    
    async def _analyze_text_batch(
        self, 
        texts: List[str], 
        article_ids: Optional[List[int]],
        source_ids: Optional[List[str]],
        topics: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Internal batch processing method"""
        try:
            results = []
            
            # Process each text in the batch
            for i, text in enumerate(texts):
                article_id = article_ids[i] if article_ids else None
                source_id = source_ids[i] if source_ids else None
                topic = topics[i] if topics else None
                
                # Use individual analysis for now (can be optimized further)
                result = await self.analyze_text(text, article_id, source_id, topic)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise
    
    def _get_cache_key(self, text: str, article_id: Optional[int]) -> str:
        """Generate cache key for analysis results"""
        content_hash = hashlib.md5(f"{text[:1000]}:{article_id}".encode()).hexdigest()
        return f"stance_sentiment_analysis:{content_hash}"
    
    def _result_to_dict(self, result: AnalysisResult) -> Dict[str, Any]:
        """Convert AnalysisResult to dictionary"""
        return {
            "chunk_id": result.chunk_id,
            "text": result.text,
            "sentiment": result.sentiment,
            "stance": result.stance,
            "toxicity": result.toxicity,
            "bias": result.bias,
            "confidence": result.confidence,
            "timestamp": result.timestamp.isoformat()
        }
    
    def _aggregated_to_dict(self, aggregated: AggregatedAnalysis) -> Dict[str, Any]:
        """Convert AggregatedAnalysis to dictionary"""
        return {
            "source_id": aggregated.source_id,
            "topic": aggregated.topic,
            "total_chunks": aggregated.total_chunks,
            "sentiment_distribution": aggregated.sentiment_distribution,
            "stance_distribution": aggregated.stance_distribution,
            "toxicity_levels": aggregated.toxicity_levels,
            "bias_scores": aggregated.bias_scores,
            "confidence_avg": aggregated.confidence_avg,
            "risk_flags": aggregated.risk_flags,
            "trend_direction": aggregated.trend_direction,
            "timestamp": aggregated.timestamp.isoformat()
        }
    
    async def get_analysis_by_source(
        self, 
        source_id: str, 
        days: int = 7
    ) -> Dict[str, Any]:
        """Get aggregated analysis for a specific source"""
        try:
            async with db_pool.get_connection() as conn:
                results = await conn.fetch("""
                    SELECT * FROM aggregated_analysis 
                    WHERE source_id = $1 
                    AND created_at >= NOW() - INTERVAL '%s days'
                    ORDER BY created_at DESC
                """, source_id, days)
                
                if not results:
                    return {"error": "No analysis found for source"}
                
                # Aggregate across all results
                all_sentiment = [json.loads(r['sentiment_distribution']) for r in results]
                all_stance = [json.loads(r['stance_distribution']) for r in results]
                all_toxicity = [json.loads(r['toxicity_levels']) for r in results]
                all_bias = [json.loads(r['bias_scores']) for r in results]
                
                return {
                    "source_id": source_id,
                    "analysis_count": len(results),
                    "sentiment_trend": self._calculate_distribution(all_sentiment),
                    "stance_trend": self._calculate_distribution(all_stance),
                    "toxicity_trend": self._calculate_distribution(all_toxicity),
                    "bias_trend": self._calculate_distribution(all_bias),
                    "avg_confidence": np.mean([r['confidence_avg'] for r in results]),
                    "recent_alerts": [r['risk_flags'] for r in results if r['risk_flags']]
                }
                
        except Exception as e:
            logger.error(f"Failed to get analysis by source: {e}")
            return {"error": str(e)}
    
    async def get_analysis_by_topic(
        self, 
        topic: str, 
        days: int = 7
    ) -> Dict[str, Any]:
        """Get aggregated analysis for a specific topic"""
        try:
            async with db_pool.get_connection() as conn:
                results = await conn.fetch("""
                    SELECT * FROM aggregated_analysis 
                    WHERE topic = $1 
                    AND created_at >= NOW() - INTERVAL '%s days'
                    ORDER BY created_at DESC
                """, topic, days)
                
                if not results:
                    return {"error": "No analysis found for topic"}
                
                # Similar aggregation as source analysis
                all_sentiment = [json.loads(r['sentiment_distribution']) for r in results]
                all_stance = [json.loads(r['stance_distribution']) for r in results]
                all_toxicity = [json.loads(r['toxicity_levels']) for r in results]
                all_bias = [json.loads(r['bias_scores']) for r in results]
                
                return {
                    "topic": topic,
                    "analysis_count": len(results),
                    "sentiment_trend": self._calculate_distribution(all_sentiment),
                    "stance_trend": self._calculate_distribution(all_stance),
                    "toxicity_trend": self._calculate_distribution(all_toxicity),
                    "bias_trend": self._calculate_distribution(all_bias),
                    "avg_confidence": np.mean([r['confidence_avg'] for r in results]),
                    "recent_alerts": [r['risk_flags'] for r in results if r['risk_flags']]
                }
                
        except Exception as e:
            logger.error(f"Failed to get analysis by topic: {e}")
            return {"error": str(e)}

# Global instance
stance_sentiment_bias_analyzer = StanceSentimentBiasAnalyzer()
