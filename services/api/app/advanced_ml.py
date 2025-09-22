"""
Advanced ML Models for OSINT Analysis
Includes topic modeling, entity extraction, and trend analysis
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass
from collections import Counter, defaultdict
import json

# Topic Modeling
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import gensim
from gensim import corpora, models
from gensim.models import CoherenceModel

# Entity Extraction
import spacy
from spacy import displacy
from spacy.matcher import Matcher
import re
from .gpu_utils import gpu_manager, model_device_manager

# Trend Analysis
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from scipy import stats
from scipy.signal import find_peaks
import plotly.graph_objects as go
import plotly.express as px

# Time Series
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

logger = logging.getLogger(__name__)

@dataclass
class TopicModel:
    """Topic modeling result"""
    topic_id: int
    keywords: List[str]
    weight: float
    coherence_score: float
    documents: List[int]

@dataclass
class EntityResult:
    """Entity extraction result"""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    description: str

@dataclass
class TrendAnalysis:
    """Trend analysis result"""
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_strength: float  # 0-1
    change_rate: float
    significance: float
    period: str
    forecast: List[float]

class AdvancedMLAnalyzer:
    """Advanced ML analysis for OSINT data"""
    
    def __init__(self):
        self.nlp = None
        self.topic_model = None
        self.vectorizer = None
        self.trend_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models"""
        try:
            # Load spaCy model for NER
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.warning("spaCy model not found, using basic NER")
            self.nlp = None
        
        # Initialize topic modeling components
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        logger.info("Advanced ML models initialized")
    
    async def analyze_topics(
        self, 
        texts: List[str], 
        num_topics: int = 10,
        method: str = 'lda'
    ) -> List[TopicModel]:
        """
        Perform topic modeling on texts
        
        Args:
            texts: List of text documents
            num_topics: Number of topics to extract
            method: 'lda' or 'bertopic'
        
        Returns:
            List of TopicModel objects
        """
        try:
            if method == 'lda':
                return await self._lda_topic_modeling(texts, num_topics)
            elif method == 'bertopic':
                return await self._bertopic_modeling(texts, num_topics)
            else:
                raise ValueError(f"Unknown topic modeling method: {method}")
        except Exception as e:
            logger.error(f"Topic modeling failed: {e}")
            return []
    
    async def _lda_topic_modeling(
        self, 
        texts: List[str], 
        num_topics: int
    ) -> List[TopicModel]:
        """LDA topic modeling implementation"""
        try:
            # Preprocess texts
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # Create TF-IDF matrix
            tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
            feature_names = self.vectorizer.get_feature_names_out()
            
            # LDA model
            lda = LatentDirichletAllocation(
                n_components=num_topics,
                random_state=42,
                max_iter=100,
                learning_decay=0.7,
                learning_offset=50.0
            )
            
            lda.fit(tfidf_matrix)
            
            # Extract topics
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                # Get top keywords
                top_indices = topic.argsort()[-10:][::-1]
                keywords = [feature_names[i] for i in top_indices]
                weights = [topic[i] for i in top_indices]
                
                # Calculate coherence score
                coherence_score = self._calculate_coherence(
                    processed_texts, keywords
                )
                
                # Find documents belonging to this topic
                doc_topic_probs = lda.transform(tfidf_matrix)
                topic_docs = np.where(doc_topic_probs[:, topic_idx] > 0.1)[0].tolist()
                
                topics.append(TopicModel(
                    topic_id=topic_idx,
                    keywords=keywords,
                    weight=float(np.mean(weights)),
                    coherence_score=coherence_score,
                    documents=topic_docs
                ))
            
            return topics
            
        except Exception as e:
            logger.error(f"LDA topic modeling failed: {e}")
            return []
    
    async def _bertopic_modeling(
        self, 
        texts: List[str], 
        num_topics: int
    ) -> List[TopicModel]:
        """BERTopic modeling implementation (simplified)"""
        try:
            # For now, use a simplified approach
            # In production, you'd use the actual BERTopic library
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # Use K-means clustering as a simplified BERTopic
            tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
            feature_names = self.vectorizer.get_feature_names_out()
            
            kmeans = KMeans(n_clusters=num_topics, random_state=42)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            topics = []
            for cluster_id in range(num_topics):
                cluster_docs = np.where(cluster_labels == cluster_id)[0]
                if len(cluster_docs) == 0:
                    continue
                
                # Get cluster center
                cluster_center = kmeans.cluster_centers_[cluster_id]
                top_indices = cluster_center.argsort()[-10:][::-1]
                keywords = [feature_names[i] for i in top_indices]
                
                # Calculate coherence
                cluster_texts = [processed_texts[i] for i in cluster_docs]
                coherence_score = self._calculate_coherence(
                    cluster_texts, keywords
                )
                
                topics.append(TopicModel(
                    topic_id=cluster_id,
                    keywords=keywords,
                    weight=float(np.mean(cluster_center[top_indices])),
                    coherence_score=coherence_score,
                    documents=cluster_docs.tolist()
                ))
            
            return topics
            
        except Exception as e:
            logger.error(f"BERTopic modeling failed: {e}")
            return []
    
    async def extract_entities(
        self, 
        text: str,
        include_custom: bool = True
    ) -> List[EntityResult]:
        """
        Extract entities from text using NER
        
        Args:
            text: Input text
            include_custom: Whether to include custom entity patterns
        
        Returns:
            List of EntityResult objects
        """
        try:
            entities = []
            
            if self.nlp:
                # Use spaCy NER
                doc = self.nlp(text)
                for ent in doc.ents:
                    entities.append(EntityResult(
                        text=ent.text,
                        label=ent.label_,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=ent._.prob if hasattr(ent._, 'prob') else 0.8,
                        description=self._get_entity_description(ent.label_)
                    ))
            
            # Add custom entity patterns
            if include_custom:
                custom_entities = await self._extract_custom_entities(text)
                entities.extend(custom_entities)
            
            # Remove duplicates and sort by confidence
            entities = self._deduplicate_entities(entities)
            entities.sort(key=lambda x: x.confidence, reverse=True)
            
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    async def _extract_custom_entities(self, text: str) -> List[EntityResult]:
        """Extract custom entities using regex patterns"""
        entities = []
        
        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            entities.append(EntityResult(
                text=match.group(),
                label='EMAIL',
                start=match.start(),
                end=match.end(),
                confidence=0.9,
                description='Email address'
            ))
        
        # URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        for match in re.finditer(url_pattern, text):
            entities.append(EntityResult(
                text=match.group(),
                label='URL',
                start=match.start(),
                end=match.end(),
                confidence=0.9,
                description='Web URL'
            ))
        
        # Phone numbers
        phone_pattern = r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'
        for match in re.finditer(phone_pattern, text):
            entities.append(EntityResult(
                text=match.group(),
                label='PHONE',
                start=match.start(),
                end=match.end(),
                confidence=0.8,
                description='Phone number'
            ))
        
        # Social media handles
        social_pattern = r'@[A-Za-z0-9_]+|#[A-Za-z0-9_]+'
        for match in re.finditer(social_pattern, text):
            entities.append(EntityResult(
                text=match.group(),
                label='SOCIAL_MEDIA',
                start=match.start(),
                end=match.end(),
                confidence=0.8,
                description='Social media handle or hashtag'
            ))
        
        return entities
    
    async def analyze_trends(
        self, 
        data: pd.DataFrame,
        value_column: str,
        time_column: str = 'created_at',
        period: str = 'daily'
    ) -> TrendAnalysis:
        """
        Analyze trends in time series data
        
        Args:
            data: DataFrame with time series data
            value_column: Column containing values to analyze
            time_column: Column containing timestamps
            period: Aggregation period ('daily', 'weekly', 'monthly')
        
        Returns:
            TrendAnalysis object
        """
        try:
            # Prepare data
            data[time_column] = pd.to_datetime(data[time_column])
            data = data.sort_values(time_column)
            
            # Aggregate by period
            if period == 'daily':
                freq = 'D'
            elif period == 'weekly':
                freq = 'W'
            elif period == 'monthly':
                freq = 'M'
            else:
                freq = 'D'
            
            aggregated = data.set_index(time_column).resample(freq)[value_column].sum()
            
            # Calculate trend
            x = np.arange(len(aggregated))
            y = aggregated.values
            
            # Linear regression for trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Determine trend direction and strength
            if p_value < 0.05:  # Significant trend
                if slope > 0:
                    trend_direction = 'increasing'
                else:
                    trend_direction = 'decreasing'
                trend_strength = abs(r_value)
            else:
                trend_direction = 'stable'
                trend_strength = 0.0
            
            # Calculate change rate
            if len(y) > 1:
                change_rate = (y[-1] - y[0]) / y[0] if y[0] != 0 else 0
            else:
                change_rate = 0.0
            
            # Generate forecast (simple linear projection)
            forecast_periods = 7
            forecast = []
            for i in range(1, forecast_periods + 1):
                forecast_value = slope * (len(x) + i) + intercept
                forecast.append(max(0, forecast_value))  # Ensure non-negative
            
            return TrendAnalysis(
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                change_rate=change_rate,
                significance=1 - p_value,
                period=period,
                forecast=forecast
            )
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return TrendAnalysis(
                trend_direction='unknown',
                trend_strength=0.0,
                change_rate=0.0,
                significance=0.0,
                period=period,
                forecast=[]
            )
    
    async def detect_anomalies(
        self, 
        data: pd.DataFrame,
        value_column: str,
        contamination: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies in data using Isolation Forest
        
        Args:
            data: DataFrame with data
            value_column: Column to analyze for anomalies
            contamination: Expected proportion of anomalies
        
        Returns:
            List of anomaly records
        """
        try:
            # Prepare data
            X = data[[value_column]].values
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42
            )
            anomaly_labels = iso_forest.fit_predict(X)
            
            # Get anomaly records
            anomalies = data[anomaly_labels == -1].copy()
            anomalies['anomaly_score'] = iso_forest.score_samples(X[anomaly_labels == -1])
            
            return anomalies.to_dict('records')
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        if not text:
            return ""
        
        # Basic cleaning
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _calculate_coherence(
        self, 
        texts: List[str], 
        keywords: List[str]
    ) -> float:
        """Calculate topic coherence score"""
        try:
            if not texts or not keywords:
                return 0.0
            
            # Simple coherence based on keyword co-occurrence
            word_counts = Counter()
            for text in texts:
                words = text.split()
                word_counts.update(words)
            
            # Calculate average frequency of keywords
            keyword_freqs = [word_counts.get(word, 0) for word in keywords]
            avg_freq = np.mean(keyword_freqs) if keyword_freqs else 0
            
            # Normalize to 0-1 scale
            coherence = min(1.0, avg_freq / 10.0)
            return coherence
            
        except Exception as e:
            logger.error(f"Coherence calculation failed: {e}")
            return 0.0
    
    def _get_entity_description(self, label: str) -> str:
        """Get description for entity label"""
        descriptions = {
            'PERSON': 'Person name',
            'ORG': 'Organization',
            'GPE': 'Geopolitical entity',
            'LOC': 'Location',
            'EVENT': 'Event',
            'WORK_OF_ART': 'Work of art',
            'LAW': 'Legal document',
            'LANGUAGE': 'Language',
            'DATE': 'Date',
            'TIME': 'Time',
            'MONEY': 'Monetary value',
            'PERCENT': 'Percentage',
            'CARDINAL': 'Cardinal number',
            'ORDINAL': 'Ordinal number',
            'QUANTITY': 'Quantity',
            'NORP': 'Nationality or religious group',
            'FAC': 'Facility',
            'PRODUCT': 'Product',
            'EMAIL': 'Email address',
            'URL': 'Web URL',
            'PHONE': 'Phone number',
            'SOCIAL_MEDIA': 'Social media handle'
        }
        return descriptions.get(label, 'Unknown entity')
    
    def _deduplicate_entities(self, entities: List[EntityResult]) -> List[EntityResult]:
        """Remove duplicate entities"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity.text.lower(), entity.start, entity.end)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities

# Global analyzer instance
advanced_ml = AdvancedMLAnalyzer()
