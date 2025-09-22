"""
Machine Learning Pipeline for content classification and analysis
Enhanced with parallel processing for high-performance ML operations
"""
import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import re
from collections import Counter
import json
import time

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import spacy

from .db import get_conn
from .config import settings
from .parallel_ml import parallel_ml_processor, process_articles_parallel, extract_features_parallel
from .gpu_utils import gpu_manager, model_device_manager

logger = logging.getLogger("osint_api")

class MLPipeline:
    """Machine Learning Pipeline for OSINT data analysis"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.lda_model = None
        self.sentiment_analyzer = None
        self.lemmatizer = None
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
        # Initialize NLTK data
        self._download_nltk_data()
        
        # Initialize sentiment analyzer after downloading data
        try:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            self.lemmatizer = WordNetLemmatizer()
        except Exception as e:
            logger.warning(f"Could not initialize NLTK components: {e}")
            self.sentiment_analyzer = None
            self.lemmatizer = None
        
        # Load spaCy model with GPU support
        try:
            model_name = "en_core_web_sm"
            device = model_device_manager.get_model_device(model_name)
            
            self.nlp = spacy.load(model_name)
            
            # Configure spaCy for GPU if available
            if gpu_manager.is_gpu_available():
                try:
                    spacy.prefer_gpu()
                    logger.info(f"Loaded spaCy model: {model_name} with GPU acceleration")
                except Exception as e:
                    logger.warning(f"GPU acceleration not available for spaCy: {e}")
                    logger.info(f"Loaded spaCy model: {model_name} on CPU")
            else:
                logger.info(f"Loaded spaCy model: {model_name} on CPU")
                
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)
        except Exception as e:
            logger.error(f"Error downloading NLTK data: {e}")
    
    async def process_article(self, article_id: int, text: str, title: str = "") -> Dict[str, Any]:
        """Process a single article through the ML pipeline"""
        try:
            # Combine title and text
            full_text = f"{title} {text}".strip()
            
            # Extract features
            features = await self._extract_features(full_text)
            
            # Perform analysis
            analysis = {
                "article_id": article_id,
                "timestamp": datetime.utcnow().isoformat(),
                "sentiment": await self._analyze_sentiment(full_text),
                "entities": await self._extract_entities(full_text),
                "topics": await self._extract_topics(full_text),
                "language": await self._detect_language(full_text),
                "keywords": await self._extract_keywords(full_text),
                "readability": await self._calculate_readability(full_text),
                "features": features
            }
            
            # Store results in database
            await self._store_analysis_results(article_id, analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error processing article {article_id}: {e}")
            return {"error": str(e)}
    
    async def process_articles_batch_parallel(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple articles in parallel for high performance
        
        Args:
            articles: List of article dictionaries with 'id', 'title', 'text' keys
        
        Returns:
            List of analysis results
        """
        start_time = time.time()
        logger.info(f"Starting parallel processing of {len(articles)} articles")
        
        # Prepare articles for parallel processing
        prepared_articles = []
        for article in articles:
            prepared_articles.append({
                'id': article.get('id'),
                'title': article.get('title', ''),
                'text': article.get('text', ''),
                'url': article.get('url', ''),
                'source': article.get('source', '')
            })
        
        # Process articles in parallel
        results = await process_articles_parallel(
            prepared_articles, 
            self._process_single_article_sync,
            batch_size=min(20, max(5, len(articles) // 4))  # Optimize batch size
        )
        
        processing_time = time.time() - start_time
        successful = len([r for r in results if r.get('success', False)])
        
        logger.info(f"Parallel processing completed: {successful}/{len(articles)} successful in {processing_time:.2f}s")
        logger.info(f"Processing rate: {len(articles)/processing_time:.1f} articles/sec")
        
        return results
    
    def _process_single_article_sync(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous version of article processing for parallel execution
        This runs in a separate process/thread
        """
        try:
            article_id = article.get('id')
            title = article.get('title', '')
            text = article.get('text', '')
            
            # Combine title and text
            full_text = f"{title} {text}".strip()
            
            # Extract features synchronously
            features = self._extract_features_sync(full_text)
            
            # Perform analysis synchronously
            analysis = {
                "article_id": article_id,
                "timestamp": datetime.utcnow().isoformat(),
                "sentiment": self._analyze_sentiment_sync(full_text),
                "entities": self._extract_entities_sync(full_text),
                "topics": self._extract_topics_sync(full_text),
                "language": self._detect_language_sync(full_text),
                "keywords": self._extract_keywords_sync(full_text),
                "readability": self._calculate_readability_sync(full_text),
                "features": features,
                "success": True
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error processing article {article.get('id')}: {e}")
            return {
                "article_id": article.get('id'),
                "error": str(e),
                "success": False
            }
    
    async def extract_features_batch_parallel(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Extract features from multiple texts in parallel
        
        Args:
            texts: List of text strings
        
        Returns:
            List of feature dictionaries
        """
        logger.info(f"Extracting features from {len(texts)} texts in parallel")
        
        # Define feature extractors
        feature_extractors = [
            self._extract_basic_features_sync,
            self._extract_text_stats_sync,
            self._extract_sentiment_features_sync,
            self._extract_readability_features_sync
        ]
        
        # Extract features in parallel
        results = await extract_features_parallel(texts, feature_extractors)
        
        logger.info(f"Feature extraction completed for {len(results)} texts")
        return results
    
    # Synchronous versions for parallel processing
    def _extract_features_sync(self, text: str) -> Dict[str, Any]:
        """Synchronous version of feature extraction for parallel processing"""
        try:
            # Basic text features
            features = {
                "char_count": len(text),
                "word_count": len(text.split()),
                "sentence_count": len(sent_tokenize(text)) if text else 0,
                "avg_word_length": np.mean([len(word) for word in text.split()]) if text.split() else 0,
                "avg_sentence_length": len(text.split()) / max(1, len(sent_tokenize(text))) if text else 0,
                "unique_words": len(set(text.lower().split())),
                "lexical_diversity": len(set(text.lower().split())) / max(1, len(text.split())),
                "uppercase_ratio": sum(1 for c in text if c.isupper()) / max(1, len(text)),
                "digit_ratio": sum(1 for c in text if c.isdigit()) / max(1, len(text)),
                "punctuation_ratio": sum(1 for c in text if c in ".,!?;:") / max(1, len(text))
            }
            
            # N-gram features
            words = text.lower().split()
            if len(words) >= 2:
                bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
                features["bigram_count"] = len(set(bigrams))
                features["bigram_diversity"] = len(set(bigrams)) / max(1, len(bigrams))
            else:
                features["bigram_count"] = 0
                features["bigram_diversity"] = 0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {}
    
    def _analyze_sentiment_sync(self, text: str) -> Dict[str, Any]:
        """Synchronous sentiment analysis for parallel processing"""
        try:
            if not self.sentiment_analyzer:
                return {"vader": {"compound": 0, "pos": 0, "neu": 0, "neg": 0}}
            
            # VADER sentiment
            vader_scores = self.sentiment_analyzer.polarity_scores(text)
            
            return {
                "vader": {
                    "compound": vader_scores["compound"],
                    "pos": vader_scores["pos"],
                    "neu": vader_scores["neu"],
                    "neg": vader_scores["neg"]
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {"vader": {"compound": 0, "pos": 0, "neu": 0, "neg": 0}}
    
    def _extract_entities_sync(self, text: str) -> Dict[str, List[str]]:
        """Synchronous entity extraction for parallel processing"""
        try:
            entities = {"people": [], "organizations": [], "locations": [], "misc": []}
            
            if not self.nlp:
                return entities
            
            doc = self.nlp(text)
            
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    entities["people"].append(ent.text.strip())
                elif ent.label_ == "ORG":
                    entities["organizations"].append(ent.text.strip())
                elif ent.label_ in ["GPE", "LOC"]:
                    entities["locations"].append(ent.text.strip())
                else:
                    entities["misc"].append(ent.text.strip())
            
            # Remove duplicates
            for key in entities:
                entities[key] = list(set([ent for ent in entities[key] if len(ent) > 1]))
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {"people": [], "organizations": [], "locations": [], "misc": []}
    
    def _extract_topics_sync(self, text: str) -> List[Dict[str, Any]]:
        """Synchronous topic extraction for parallel processing"""
        try:
            topics = []
            
            # Define topic keywords
            topic_keywords = {
                "politics": ["government", "election", "president", "congress", "policy", "political"],
                "technology": ["tech", "software", "computer", "internet", "digital", "ai", "technology"],
                "business": ["business", "company", "market", "economy", "financial", "investment"],
                "sports": ["sport", "game", "team", "player", "match", "championship"],
                "health": ["health", "medical", "doctor", "hospital", "disease", "treatment"],
                "environment": ["environment", "climate", "pollution", "green", "sustainable", "energy"]
            }
            
            text_lower = text.lower()
            for topic, keywords in topic_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                if score > 0:
                    topics.append({
                        "topic": topic,
                        "score": score / len(keywords),
                        "keywords": [kw for kw in keywords if kw in text_lower]
                    })
            
            # Sort by score
            topics.sort(key=lambda x: x["score"], reverse=True)
            
            return topics[:5]  # Return top 5 topics
            
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return []
    
    def _detect_language_sync(self, text: str) -> str:
        """Synchronous language detection for parallel processing"""
        try:
            # Simple language detection based on common words
            english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
            spanish_words = ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le']
            french_words = ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour']
            
            text_lower = text.lower()
            english_count = sum(1 for word in english_words if word in text_lower)
            spanish_count = sum(1 for word in spanish_words if word in text_lower)
            french_count = sum(1 for word in french_words if word in text_lower)
            
            if english_count > spanish_count and english_count > french_count:
                return "en"
            elif spanish_count > french_count:
                return "es"
            elif french_count > 0:
                return "fr"
            else:
                return "en"  # Default to English
                
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return "en"
    
    def _extract_keywords_sync(self, text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Synchronous keyword extraction for parallel processing"""
        try:
            words = text.lower().split()
            
            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
            
            # Count word frequencies
            word_freq = Counter(filtered_words)
            
            # Get top keywords
            keywords = []
            for word, freq in word_freq.most_common(top_k):
                keywords.append({
                    "word": word,
                    "frequency": freq,
                    "score": freq / len(filtered_words) if filtered_words else 0
                })
            
            return keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def _calculate_readability_sync(self, text: str) -> Dict[str, float]:
        """Synchronous readability calculation for parallel processing"""
        try:
            sentences = sent_tokenize(text)
            words = text.split()
            
            if not sentences or not words:
                return {"flesch_score": 0, "grade_level": 0}
            
            # Flesch Reading Ease Score
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables_per_word = sum(self._count_syllables(word) for word in words) / len(words)
            
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            
            # Grade level
            grade_level = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59
            
            return {
                "flesch_score": max(0, min(100, flesch_score)),
                "grade_level": max(0, grade_level),
                "avg_sentence_length": avg_sentence_length,
                "avg_syllables_per_word": avg_syllables_per_word
            }
            
        except Exception as e:
            logger.error(f"Error calculating readability: {e}")
            return {"flesch_score": 0, "grade_level": 0}
    
    # Feature extractors for parallel processing
    def _extract_basic_features_sync(self, text: str) -> Dict[str, Any]:
        """Extract basic text features synchronously"""
        return {
            "char_count": len(text),
            "word_count": len(text.split()),
            "sentence_count": len(sent_tokenize(text)) if text else 0,
            "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
            "has_numbers": any(c.isdigit() for c in text),
            "has_uppercase": any(c.isupper() for c in text),
            "has_punctuation": any(c in ".,!?;:" for c in text)
        }
    
    def _extract_text_stats_sync(self, text: str) -> Dict[str, Any]:
        """Extract text statistics synchronously"""
        words = text.split()
        if not words:
            return {"avg_word_length": 0, "lexical_diversity": 0, "unique_words": 0}
        
        return {
            "avg_word_length": np.mean([len(word) for word in words]),
            "lexical_diversity": len(set(word.lower() for word in words)) / len(words),
            "unique_words": len(set(word.lower() for word in words)),
            "longest_word": max(len(word) for word in words) if words else 0,
            "shortest_word": min(len(word) for word in words) if words else 0
        }
    
    def _extract_sentiment_features_sync(self, text: str) -> Dict[str, Any]:
        """Extract sentiment features synchronously"""
        if not self.sentiment_analyzer:
            return {"sentiment_compound": 0, "sentiment_positive": 0, "sentiment_negative": 0}
        
        scores = self.sentiment_analyzer.polarity_scores(text)
        return {
            "sentiment_compound": scores["compound"],
            "sentiment_positive": scores["pos"],
            "sentiment_negative": scores["neg"],
            "sentiment_neutral": scores["neu"]
        }
    
    def _extract_readability_features_sync(self, text: str) -> Dict[str, Any]:
        """Extract readability features synchronously"""
        try:
            sentences = sent_tokenize(text)
            words = text.split()
            
            if not sentences or not words:
                return {"readability_score": 0, "grade_level": 0}
            
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables = sum(self._count_syllables(word) for word in words) / len(words)
            
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
            grade_level = 0.39 * avg_sentence_length + 11.8 * avg_syllables - 15.59
            
            return {
                "readability_score": max(0, min(100, flesch_score)),
                "grade_level": max(0, grade_level),
                "avg_sentence_length": avg_sentence_length,
                "avg_syllables_per_word": avg_syllables
            }
        except Exception as e:
            logger.error(f"Error extracting readability features: {e}")
            return {"readability_score": 0, "grade_level": 0}
    
    async def _extract_features(self, text: str) -> Dict[str, Any]:
        """Extract numerical features from text"""
        try:
            # Basic text features
            features = {
                "word_count": len(text.split()),
                "char_count": len(text),
                "sentence_count": len(sent_tokenize(text)),
                "avg_word_length": np.mean([len(word) for word in text.split()]),
                "unique_words": len(set(text.lower().split())),
                "exclamation_count": text.count('!'),
                "question_count": text.count('?'),
                "uppercase_ratio": sum(1 for c in text if c.isupper()) / len(text) if text else 0,
                "digit_ratio": sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
                "url_count": len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
                "email_count": len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)),
                "phone_count": len(re.findall(r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}', text))
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {}
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using multiple models"""
        try:
            # VADER sentiment (if available)
            vader_scores = {"compound": 0, "pos": 0, "neu": 1, "neg": 0}
            if self.sentiment_analyzer:
                try:
                    vader_scores = self.sentiment_analyzer.polarity_scores(text)
                except Exception as e:
                    logger.warning(f"VADER sentiment analysis failed: {e}")
            
            # Custom sentiment based on keywords
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'positive', 'success', 'win', 'victory']
            negative_words = ['bad', 'terrible', 'awful', 'horrible', 'negative', 'failure', 'lose', 'defeat', 'crisis', 'problem']
            
            words = text.lower().split()
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            
            custom_sentiment = {
                "positive": positive_count / len(words) if words else 0,
                "negative": negative_count / len(words) if words else 0,
                "neutral": 1 - (positive_count + negative_count) / len(words) if words else 1
            }
            
            return {
                "vader": vader_scores,
                "custom": custom_sentiment,
                "overall": vader_scores['compound']
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {"error": str(e)}
    
    async def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities (people, organizations, locations)"""
        try:
            entities = {
                "people": [],
                "organizations": [],
                "locations": [],
                "misc": []
            }
            
            if self.nlp:
                # Use spaCy for entity extraction
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ in ["PERSON"]:
                        entities["people"].append(ent.text)
                    elif ent.label_ in ["ORG", "GPE"]:
                        entities["organizations"].append(ent.text)
                    elif ent.label_ in ["GPE", "LOC"]:
                        entities["locations"].append(ent.text)
                    else:
                        entities["misc"].append(ent.text)
            else:
                # Fallback to NLTK
                tokens = word_tokenize(text)
                pos_tags = pos_tag(tokens)
                chunks = ne_chunk(pos_tags)
                
                for chunk in chunks:
                    if hasattr(chunk, 'label'):
                        if chunk.label() == 'PERSON':
                            entities["people"].append(' '.join([token for token, pos in chunk.leaves()]))
                        elif chunk.label() == 'ORGANIZATION':
                            entities["organizations"].append(' '.join([token for token, pos in chunk.leaves()]))
                        elif chunk.label() == 'GPE':
                            entities["locations"].append(' '.join([token for token, pos in chunk.leaves()]))
            
            # Remove duplicates and filter
            for key in entities:
                entities[key] = list(set([ent.strip() for ent in entities[key] if len(ent.strip()) > 1]))
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {"people": [], "organizations": [], "locations": [], "misc": []}
    
    async def _extract_topics(self, text: str) -> List[Dict[str, Any]]:
        """Extract topics using LDA"""
        try:
            # Simple keyword-based topic extraction
            # In production, you'd use a trained LDA model
            topics = []
            
            # Define topic keywords
            topic_keywords = {
                "politics": ["government", "election", "president", "congress", "policy", "political"],
                "technology": ["tech", "software", "computer", "internet", "digital", "ai", "technology"],
                "business": ["business", "company", "market", "economy", "financial", "investment"],
                "sports": ["sport", "game", "team", "player", "match", "championship"],
                "health": ["health", "medical", "doctor", "hospital", "disease", "treatment"],
                "environment": ["environment", "climate", "pollution", "green", "sustainable", "energy"]
            }
            
            text_lower = text.lower()
            for topic, keywords in topic_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                if score > 0:
                    topics.append({
                        "topic": topic,
                        "score": score / len(keywords),
                        "keywords": [kw for kw in keywords if kw in text_lower]
                    })
            
            # Sort by score
            topics.sort(key=lambda x: x["score"], reverse=True)
            
            return topics[:5]  # Return top 5 topics
            
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return []
    
    async def _detect_language(self, text: str) -> str:
        """Detect text language"""
        try:
            # Simple language detection based on common words
            english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
            spanish_words = ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le']
            french_words = ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour']
            
            text_lower = text.lower()
            english_count = sum(1 for word in english_words if word in text_lower)
            spanish_count = sum(1 for word in spanish_words if word in text_lower)
            french_count = sum(1 for word in french_words if word in text_lower)
            
            if english_count > spanish_count and english_count > french_count:
                return "en"
            elif spanish_count > french_count:
                return "es"
            elif french_count > 0:
                return "fr"
            else:
                return "en"  # Default to English
                
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return "en"
    
    async def _extract_keywords(self, text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Extract keywords using TF-IDF"""
        try:
            # Simple keyword extraction
            words = text.lower().split()
            
            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
            
            # Count word frequencies
            word_freq = Counter(filtered_words)
            
            # Get top keywords
            keywords = []
            for word, freq in word_freq.most_common(top_k):
                keywords.append({
                    "word": word,
                    "frequency": freq,
                    "score": freq / len(filtered_words) if filtered_words else 0
                })
            
            return keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    async def _calculate_readability(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics"""
        try:
            sentences = sent_tokenize(text)
            words = text.split()
            
            if not sentences or not words:
                return {"flesch_score": 0, "grade_level": 0}
            
            # Flesch Reading Ease Score
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables_per_word = sum(self._count_syllables(word) for word in words) / len(words)
            
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            
            # Grade level
            grade_level = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59
            
            return {
                "flesch_score": max(0, min(100, flesch_score)),
                "grade_level": max(0, grade_level),
                "avg_sentence_length": avg_sentence_length,
                "avg_syllables_per_word": avg_syllables_per_word
            }
            
        except Exception as e:
            logger.error(f"Error calculating readability: {e}")
            return {"flesch_score": 0, "grade_level": 0}
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word"""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    async def _store_analysis_results(self, article_id: int, analysis: Dict[str, Any]):
        """Store ML analysis results in database"""
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    # Store sentiment analysis
                    sentiment_data = analysis.get("sentiment", {})
                    cur.execute("""
                        INSERT INTO article_sentiment (article_id, vader_compound, vader_pos, vader_neu, vader_neg, 
                                                     custom_pos, custom_neg, custom_neu, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (article_id) DO UPDATE SET
                            vader_compound = EXCLUDED.vader_compound,
                            vader_pos = EXCLUDED.vader_pos,
                            vader_neu = EXCLUDED.vader_neu,
                            vader_neg = EXCLUDED.vader_neg,
                            custom_pos = EXCLUDED.custom_pos,
                            custom_neg = EXCLUDED.custom_neg,
                            custom_neu = EXCLUDED.custom_neu,
                            updated_at = NOW()
                    """, (
                        article_id,
                        sentiment_data.get("vader", {}).get("compound", 0),
                        sentiment_data.get("vader", {}).get("pos", 0),
                        sentiment_data.get("vader", {}).get("neu", 0),
                        sentiment_data.get("vader", {}).get("neg", 0),
                        sentiment_data.get("custom", {}).get("positive", 0),
                        sentiment_data.get("custom", {}).get("negative", 0),
                        sentiment_data.get("custom", {}).get("neutral", 0)
                    ))
                    
                    # Store entities
                    entities = analysis.get("entities", {})
                    for entity_type, entity_list in entities.items():
                        for entity in entity_list:
                            cur.execute("""
                                INSERT INTO article_entities (article_id, entity_type, entity_name, created_at)
                                VALUES (%s, %s, %s, NOW())
                                ON CONFLICT (article_id, entity_type, entity_name) DO NOTHING
                            """, (article_id, entity_type, entity))
                    
                    # Store topics
                    topics = analysis.get("topics", [])
                    for topic in topics:
                        cur.execute("""
                            INSERT INTO article_topics_ml (article_id, topic_name, topic_score, keywords, created_at)
                            VALUES (%s, %s, %s, %s, NOW())
                            ON CONFLICT (article_id, topic_name) DO UPDATE SET
                                topic_score = EXCLUDED.topic_score,
                                keywords = EXCLUDED.keywords,
                                updated_at = NOW()
                        """, (article_id, topic["topic"], topic["score"], json.dumps(topic["keywords"])))
                    
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Error storing analysis results: {e}")
    
    async def detect_anomalies(self, article_ids: List[int]) -> List[Dict[str, Any]]:
        """Detect anomalous articles"""
        try:
            # Get article features
            features_data = []
            for article_id in article_ids:
                with get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT word_count, char_count, sentence_count, avg_word_length, 
                                   unique_words, exclamation_count, question_count
                            FROM article_features WHERE article_id = %s
                        """, (article_id,))
                        result = cur.fetchone()
                        if result:
                            features_data.append([article_id] + list(result))
            
            if len(features_data) < 2:
                return []
            
            # Prepare data for anomaly detection
            article_ids_list = [row[0] for row in features_data]
            features_matrix = np.array([row[1:] for row in features_data])
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features_matrix)
            
            # Detect anomalies
            anomaly_labels = self.anomaly_detector.fit_predict(features_scaled)
            
            # Return anomalous articles
            anomalies = []
            for i, label in enumerate(anomaly_labels):
                if label == -1:  # Anomaly
                    anomalies.append({
                        "article_id": article_ids_list[i],
                        "anomaly_score": self.anomaly_detector.decision_function([features_scaled[i]])[0],
                        "features": features_matrix[i].tolist()
                    })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []
    
    async def get_insights(self, time_period: str = "7d") -> Dict[str, Any]:
        """Get ML insights for dashboard"""
        try:
            # Calculate time range
            if time_period == "1d":
                days = 1
            elif time_period == "7d":
                days = 7
            elif time_period == "30d":
                days = 30
            else:
                days = 7
            
            start_date = datetime.utcnow() - timedelta(days=days)
            
            insights = {
                "sentiment_trends": await self._get_sentiment_trends(start_date),
                "topic_distribution": await self._get_topic_distribution(start_date),
                "entity_frequency": await self._get_entity_frequency(start_date),
                "anomaly_alerts": await self._get_anomaly_alerts(start_date),
                "language_distribution": await self._get_language_distribution(start_date)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting insights: {e}")
            return {}
    
    async def _get_sentiment_trends(self, start_date: datetime) -> Dict[str, Any]:
        """Get sentiment trends over time"""
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT DATE(created_at) as date, 
                               AVG(vader_compound) as avg_sentiment,
                               COUNT(*) as article_count
                        FROM article_sentiment 
                        WHERE created_at >= %s
                        GROUP BY DATE(created_at)
                        ORDER BY date
                    """, (start_date,))
                    
                    results = cur.fetchall()
                    return {
                        "dates": [str(row[0]) for row in results],
                        "sentiment_scores": [float(row[1]) for row in results],
                        "article_counts": [int(row[2]) for row in results]
                    }
        except Exception as e:
            logger.error(f"Error getting sentiment trends: {e}")
            return {"dates": [], "sentiment_scores": [], "article_counts": []}
    
    async def _get_topic_distribution(self, start_date: datetime) -> Dict[str, Any]:
        """Get topic distribution"""
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT topic_name, AVG(topic_score) as avg_score, COUNT(*) as count
                        FROM article_topics_ml 
                        WHERE created_at >= %s
                        GROUP BY topic_name
                        ORDER BY avg_score DESC
                        LIMIT 10
                    """, (start_date,))
                    
                    results = cur.fetchall()
                    return {
                        "topics": [row[0] for row in results],
                        "scores": [float(row[1]) for row in results],
                        "counts": [int(row[2]) for row in results]
                    }
        except Exception as e:
            logger.error(f"Error getting topic distribution: {e}")
            return {"topics": [], "scores": [], "counts": []}
    
    async def _get_entity_frequency(self, start_date: datetime) -> Dict[str, Any]:
        """Get entity frequency"""
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT entity_type, entity_name, COUNT(*) as frequency
                        FROM article_entities 
                        WHERE created_at >= %s
                        GROUP BY entity_type, entity_name
                        ORDER BY frequency DESC
                        LIMIT 20
                    """, (start_date,))
                    
                    results = cur.fetchall()
                    entities = {}
                    for row in results:
                        entity_type = row[0]
                        if entity_type not in entities:
                            entities[entity_type] = []
                        entities[entity_type].append({
                            "name": row[1],
                            "frequency": int(row[2])
                        })
                    
                    return entities
        except Exception as e:
            logger.error(f"Error getting entity frequency: {e}")
            return {}
    
    async def _get_anomaly_alerts(self, start_date: datetime) -> List[Dict[str, Any]]:
        """Get recent anomaly alerts"""
        try:
            # This would query a anomalies table
            # For now, return mock data
            return [
                {
                    "article_id": 123,
                    "anomaly_type": "unusual_sentiment",
                    "severity": "medium",
                    "description": "Article shows unusually negative sentiment",
                    "timestamp": datetime.utcnow().isoformat()
                }
            ]
        except Exception as e:
            logger.error(f"Error getting anomaly alerts: {e}")
            return []
    
    async def _get_language_distribution(self, start_date: datetime) -> Dict[str, int]:
        """Get language distribution"""
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT lang, COUNT(*) as count
                        FROM articles 
                        WHERE fetched_at >= %s AND lang IS NOT NULL
                        GROUP BY lang
                        ORDER BY count DESC
                    """, (start_date,))
                    
                    results = cur.fetchall()
                    return {row[0]: int(row[1]) for row in results}
        except Exception as e:
            logger.error(f"Error getting language distribution: {e}")
            return {}
    
    async def process_articles_gpu_batch(
        self, 
        articles: List[Dict[str, Any]], 
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Process articles in GPU-optimized batches
        
        Args:
            articles: List of articles to process
            batch_size: Optional batch size (auto-calculated if None)
            
        Returns:
            List of processed articles with ML features
        """
        try:
            if not articles:
                return []
            
            # Use GPU memory management
            with gpu_manager.memory_management():
                # Calculate optimal batch size
                if batch_size is None:
                    batch_size = gpu_manager.get_optimal_batch_size(200)  # ~200MB model
                
                results = []
                for i in range(0, len(articles), batch_size):
                    batch_articles = articles[i:i + batch_size]
                    
                    # Process batch
                    batch_results = await self._process_article_batch(batch_articles)
                    results.extend(batch_results)
                
                logger.info(f"Processed {len(articles)} articles in GPU-optimized batches")
                return results
                
        except Exception as e:
            logger.error(f"GPU batch processing failed: {e}")
            # Fallback to individual processing
            return [await self.process_article(article) for article in articles]
    
    async def _process_article_batch(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of articles"""
        try:
            results = []
            
            # Extract texts for batch processing
            texts = [article.get('content', '') for article in articles]
            
            # Batch process features that can benefit from parallelization
            if self.nlp:
                # Batch process with spaCy
                docs = list(self.nlp.pipe(texts))
                
                for i, (article, doc) in enumerate(zip(articles, docs)):
                    # Extract entities
                    entities = [(ent.text, ent.label_) for ent in doc.ents]
                    
                    # Extract features
                    features = self._extract_features_from_doc(doc)
                    
                    # Combine with original article
                    result = article.copy()
                    result.update({
                        'entities': entities,
                        'ml_features': features,
                        'processed_at': datetime.now().isoformat()
                    })
                    results.append(result)
            else:
                # Fallback to individual processing
                for article in articles:
                    result = await self.process_article(article)
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Fallback to individual processing
            return [await self.process_article(article) for article in articles]
    
    def _extract_features_from_doc(self, doc) -> Dict[str, Any]:
        """Extract features from a spaCy document"""
        try:
            features = {
                'sentences': len(list(doc.sents)),
                'tokens': len(doc),
                'entities_count': len(doc.ents),
                'pos_tags': [token.pos_ for token in doc],
                'lemmas': [token.lemma_ for token in doc if not token.is_stop],
                'sentiment_score': 0.0  # Placeholder
            }
            
            # Add sentiment if analyzer is available
            if self.sentiment_analyzer:
                text = doc.text
                sentiment = self.sentiment_analyzer.polarity_scores(text)
                features['sentiment_score'] = sentiment['compound']
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {}

# Global ML pipeline instance
ml_pipeline = MLPipeline()
