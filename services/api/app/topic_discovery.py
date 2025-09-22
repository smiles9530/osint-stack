"""
Topic Discovery & Clustering Service
Implements BERTopic, HDBSCAN/UMAP for campaign emergence detection
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
import psycopg2.extras
from contextlib import contextmanager
import json

from .config import settings
from .db import get_conn
from .vector_search import pgvector_search

logger = logging.getLogger("osint_api")

class TopicDiscoveryService:
    """Topic discovery and clustering service for campaign emergence detection"""
    
    def __init__(self):
        self.bertopic_model = None
        self.umap_model = None
        self.hdbscan_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize clustering models with GPU acceleration"""
        try:
            from bertopic import BERTopic
            from umap import UMAP
            from hdbscan import HDBSCAN
            from sentence_transformers import SentenceTransformer
            from .gpu_utils import gpu_manager, model_device_manager
            
            # Initialize embedding model with GPU support
            model_name = 'BAAI/bge-m3'
            device = model_device_manager.get_model_device(model_name)
            
            embedding_model = SentenceTransformer(model_name)
            
            # Move to appropriate device
            if gpu_manager.is_gpu_available():
                embedding_model = embedding_model.to(device)
                logger.info(f"Topic discovery embedding model loaded on {device}")
            else:
                logger.info("Topic discovery embedding model loaded on CPU")
            
            # Initialize UMAP for dimensionality reduction
            # UMAP can benefit from GPU acceleration in some cases
            self.umap_model = UMAP(
                n_neighbors=15,
                n_components=5,
                min_dist=0.0,
                metric='cosine',
                random_state=42,
                n_jobs=-1  # Use all available cores
            )
            
            # Initialize HDBSCAN for clustering
            # HDBSCAN is CPU-optimized but can use multiple cores
            self.hdbscan_model = HDBSCAN(
                min_cluster_size=10,
                min_samples=5,
                metric='euclidean',
                cluster_selection_epsilon=0.0,
                n_jobs=-1  # Use all available cores
            )
            
            # Initialize BERTopic with custom models
            self.bertopic_model = BERTopic(
                embedding_model=embedding_model,
                umap_model=self.umap_model,
                hdbscan_model=self.hdbscan_model,
                min_topic_size=10,
                calculate_probabilities=True,
                verbose=True
            )
            
            logger.info(f"Loaded topic discovery models: BERTopic + UMAP + HDBSCAN on {device}")
            
        except ImportError as e:
            logger.warning(f"Topic discovery models not available: {e}")
            self.bertopic_model = None
            self.umap_model = None
            self.hdbscan_model = None
    
    async def discover_topics(
        self, 
        hours_back: int = 24,
        min_documents: int = 10,
        similarity_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Discover topics from recent articles using BERTopic
        
        Args:
            hours_back: Number of hours to look back for articles
            min_documents: Minimum documents per topic
            similarity_threshold: Threshold for topic assignment
        
        Returns:
            Topic discovery results with clusters and assignments
        """
        try:
            # Get recent articles
            articles = await self._get_recent_articles(hours_back)
            
            if len(articles) < min_documents:
                return {
                    'topics': [],
                    'assignments': [],
                    'total_articles': len(articles),
                    'message': f'Not enough articles ({len(articles)} < {min_documents})'
                }
            
            # Extract texts for clustering
            texts = [article['text'] for article in articles]
            article_ids = [article['id'] for article in articles]
            
            if self.bertopic_model:
                # Use BERTopic for topic modeling
                topics, probs = self.bertopic_model.fit_transform(texts)
                
                # Get topic information
                topic_info = self.bertopic_model.get_topic_info()
                
                # Process results
                topic_results = []
                for _, row in topic_info.iterrows():
                    if row['Topic'] != -1:  # Skip outlier topic
                        topic_words = self.bertopic_model.get_topic(row['Topic'])
                        topic_results.append({
                            'topic_id': int(row['Topic']),
                            'name': f"Topic_{row['Topic']}",
                            'count': int(row['Count']),
                            'keywords': [word[0] for word in topic_words[:10]],
                            'keyword_scores': [word[1] for word in topic_words[:10]],
                            'coherence_score': self._calculate_coherence_score(topic_words),
                            'created_at': datetime.now().isoformat()
                        })
                
                # Create assignments
                assignments = []
                for i, (article_id, topic, prob) in enumerate(zip(article_ids, topics, probs)):
                    if topic != -1 and prob[0] > similarity_threshold:
                        assignments.append({
                            'article_id': article_id,
                            'topic_id': int(topic),
                            'similarity_score': float(prob[0]),
                            'assigned_at': datetime.now().isoformat()
                        })
                
                # Store topics and assignments in database
                await self._store_topics_and_assignments(topic_results, assignments)
                
                return {
                    'topics': topic_results,
                    'assignments': assignments,
                    'total_articles': len(articles),
                    'topics_discovered': len(topic_results),
                    'articles_assigned': len(assignments),
                    'method': 'BERTopic'
                }
            else:
                # Fallback to simple keyword-based clustering
                return await self._simple_topic_clustering(articles, similarity_threshold)
                
        except Exception as e:
            logger.error(f"Topic discovery failed: {e}")
            return {
                'topics': [],
                'assignments': [],
                'total_articles': 0,
                'error': str(e)
            }
    
    async def _get_recent_articles(self, hours_back: int) -> List[Dict[str, Any]]:
        """Get recent articles for topic discovery"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            with get_conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("""
                        SELECT id, title, text, published_at, lang, source_id
                        FROM articles
                        WHERE published_at >= %s
                          AND text IS NOT NULL
                          AND LENGTH(text) > 100
                        ORDER BY published_at DESC
                    """, (cutoff_time,))
                    
                    return [dict(row) for row in cur.fetchall()]
                    
        except Exception as e:
            logger.error(f"Failed to get recent articles: {e}")
            return []
    
    async def _simple_topic_clustering(
        self, 
        articles: List[Dict[str, Any]], 
        similarity_threshold: float
    ) -> Dict[str, Any]:
        """Simple keyword-based topic clustering fallback"""
        try:
            from collections import Counter
            import re
            
            # Extract keywords from articles
            all_keywords = []
            article_keywords = {}
            
            for article in articles:
                text = article['text'].lower()
                # Simple keyword extraction
                words = re.findall(r'\b[a-zA-Z]{4,}\b', text)
                # Filter common words
                stop_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'would', 'there', 'could', 'other', 'after', 'first', 'well', 'also', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us', 'is', 'are', 'was', 'be', 'or', 'an', 'a', 'and', 'to', 'of', 'in', 'for', 'on', 'with', 'by', 'at', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'under', 'over', 'around', 'near', 'far', 'here', 'there', 'where', 'when', 'why', 'how', 'what', 'who', 'which', 'whose', 'whom'}
                keywords = [word for word in words if word not in stop_words and len(word) > 3]
                
                article_keywords[article['id']] = keywords[:20]  # Top 20 keywords per article
                all_keywords.extend(keywords)
            
            # Find common keywords
            keyword_counts = Counter(all_keywords)
            common_keywords = [word for word, count in keyword_counts.most_common(50) if count > 1]
            
            # Create simple topics based on keyword overlap
            topics = []
            assignments = []
            topic_id = 0
            
            for keyword in common_keywords[:10]:  # Top 10 topics
                topic_articles = []
                for article_id, keywords in article_keywords.items():
                    if keyword in keywords:
                        topic_articles.append(article_id)
                
                if len(topic_articles) >= 3:  # Minimum 3 articles per topic
                    topics.append({
                        'topic_id': topic_id,
                        'name': f"Topic_{keyword.title()}",
                        'count': len(topic_articles),
                        'keywords': [keyword],
                        'keyword_scores': [1.0],
                        'coherence_score': 0.5,
                        'created_at': datetime.now().isoformat()
                    })
                    
                    # Create assignments
                    for article_id in topic_articles:
                        assignments.append({
                            'article_id': article_id,
                            'topic_id': topic_id,
                            'similarity_score': 0.8,
                            'assigned_at': datetime.now().isoformat()
                        })
                    
                    topic_id += 1
            
            # Store in database
            await self._store_topics_and_assignments(topics, assignments)
            
            return {
                'topics': topics,
                'assignments': assignments,
                'total_articles': len(articles),
                'topics_discovered': len(topics),
                'articles_assigned': len(assignments),
                'method': 'Simple_Keyword'
            }
            
        except Exception as e:
            logger.error(f"Simple topic clustering failed: {e}")
            return {
                'topics': [],
                'assignments': [],
                'total_articles': len(articles),
                'error': str(e)
            }
    
    async def _store_topics_and_assignments(
        self, 
        topics: List[Dict[str, Any]], 
        assignments: List[Dict[str, Any]]
    ):
        """Store topics and assignments in database"""
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    # Store topics
                    for topic in topics:
                        cur.execute("""
                            INSERT INTO topic_clusters 
                            (cluster_name, cluster_keywords, article_count, coherence_score, created_at)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (cluster_name) 
                            DO UPDATE SET
                                cluster_keywords = EXCLUDED.cluster_keywords,
                                article_count = EXCLUDED.article_count,
                                coherence_score = EXCLUDED.coherence_score,
                                updated_at = NOW()
                            RETURNING id
                        """, (
                            topic['name'],
                            json.dumps(topic['keywords']),
                            topic['count'],
                            topic['coherence_score'],
                            topic['created_at']
                        ))
                        
                        cluster_id = cur.fetchone()[0]
                        topic['cluster_id'] = cluster_id
                    
                    # Store assignments
                    for assignment in assignments:
                        # Find cluster_id for this topic
                        topic_id = assignment['topic_id']
                        cluster_id = next(
                            (t['cluster_id'] for t in topics if t['topic_id'] == topic_id), 
                            None
                        )
                        
                        if cluster_id:
                            cur.execute("""
                                INSERT INTO article_topic_clusters 
                                (article_id, cluster_id, similarity_score, created_at)
                                VALUES (%s, %s, %s, %s)
                                ON CONFLICT (article_id, cluster_id) 
                                DO UPDATE SET
                                    similarity_score = EXCLUDED.similarity_score,
                                    created_at = EXCLUDED.created_at
                            """, (
                                assignment['article_id'],
                                cluster_id,
                                assignment['similarity_score'],
                                assignment['assigned_at']
                            ))
                    
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Failed to store topics and assignments: {e}")
    
    def _calculate_coherence_score(self, topic_words: List[Tuple[str, float]]) -> float:
        """Calculate topic coherence score"""
        try:
            if not topic_words:
                return 0.0
            
            # Simple coherence based on word frequency distribution
            scores = [score for word, score in topic_words]
            if not scores:
                return 0.0
            
            # Calculate normalized entropy as coherence measure
            total_score = sum(scores)
            if total_score == 0:
                return 0.0
            
            normalized_scores = [s / total_score for s in scores]
            entropy = -sum(p * np.log(p) if p > 0 else 0 for p in normalized_scores)
            max_entropy = np.log(len(normalized_scores)) if len(normalized_scores) > 1 else 1
            
            return 1 - (entropy / max_entropy) if max_entropy > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate coherence score: {e}")
            return 0.0
    
    async def get_topic_trends(
        self, 
        days_back: int = 7,
        topic_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get topic trends over time
        
        Args:
            days_back: Number of days to analyze
            topic_id: Specific topic ID (optional)
        
        Returns:
            Topic trends with volume and sentiment over time
        """
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)
            
            with get_conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    # Get topic trends
                    if topic_id is not None:
                        cur.execute("""
                            SELECT 
                                tc.id as topic_id,
                                tc.cluster_name as topic_name,
                                DATE(atc.created_at) as date,
                                COUNT(atc.article_id) as article_count,
                                AVG(atc.similarity_score) as avg_similarity
                            FROM topic_clusters tc
                            JOIN article_topic_clusters atc ON tc.id = atc.cluster_id
                            WHERE atc.created_at >= %s AND tc.id = %s
                            GROUP BY tc.id, tc.cluster_name, DATE(atc.created_at)
                            ORDER BY date DESC
                        """, (cutoff_time, topic_id))
                    else:
                        cur.execute("""
                            SELECT 
                                tc.id as topic_id,
                                tc.cluster_name as topic_name,
                                DATE(atc.created_at) as date,
                                COUNT(atc.article_id) as article_count,
                                AVG(atc.similarity_score) as avg_similarity
                            FROM topic_clusters tc
                            JOIN article_topic_clusters atc ON tc.id = atc.cluster_id
                            WHERE atc.created_at >= %s
                            GROUP BY tc.id, tc.cluster_name, DATE(atc.created_at)
                            ORDER BY date DESC
                        """, (cutoff_time,))
                    
                    trends = cur.fetchall()
                    
                    # Get sentiment trends (if available)
                    cur.execute("""
                        SELECT 
                            tc.id as topic_id,
                            DATE(atc.created_at) as date,
                            AVG(COALESCE(af.sentiment_score, 0)) as avg_sentiment
                        FROM topic_clusters tc
                        JOIN article_topic_clusters atc ON tc.id = atc.cluster_id
                        LEFT JOIN article_features af ON atc.article_id = af.article_id
                        WHERE atc.created_at >= %s
                        GROUP BY tc.id, DATE(atc.created_at)
                        ORDER BY date DESC
                    """, (cutoff_time,))
                    
                    sentiment_trends = cur.fetchall()
                    
                    return {
                        'trends': [dict(row) for row in trends],
                        'sentiment_trends': [dict(row) for row in sentiment_trends],
                        'days_analyzed': days_back,
                        'generated_at': datetime.now().isoformat()
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get topic trends: {e}")
            return {
                'trends': [],
                'sentiment_trends': [],
                'error': str(e)
            }
    
    async def detect_campaigns(
        self, 
        min_volume_increase: float = 2.0,
        time_window_hours: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Detect emerging campaigns based on topic volume spikes
        
        Args:
            min_volume_increase: Minimum volume increase multiplier
            time_window_hours: Time window for volume comparison
        
        Returns:
            List of detected campaigns
        """
        try:
            current_time = datetime.now()
            recent_window = current_time - timedelta(hours=time_window_hours)
            baseline_window = recent_window - timedelta(hours=time_window_hours)
            
            with get_conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    # Get volume comparison
                    cur.execute("""
                        WITH recent_volume AS (
                            SELECT 
                                tc.id as topic_id,
                                tc.cluster_name as topic_name,
                                COUNT(atc.article_id) as recent_count
                            FROM topic_clusters tc
                            JOIN article_topic_clusters atc ON tc.id = atc.cluster_id
                            WHERE atc.created_at >= %s
                            GROUP BY tc.id, tc.cluster_name
                        ),
                        baseline_volume AS (
                            SELECT 
                                tc.id as topic_id,
                                COUNT(atc.article_id) as baseline_count
                            FROM topic_clusters tc
                            JOIN article_topic_clusters atc ON tc.id = atc.cluster_id
                            WHERE atc.created_at >= %s AND atc.created_at < %s
                            GROUP BY tc.id
                        )
                        SELECT 
                            rv.topic_id,
                            rv.topic_name,
                            rv.recent_count,
                            COALESCE(bv.baseline_count, 0) as baseline_count,
                            CASE 
                                WHEN COALESCE(bv.baseline_count, 0) = 0 THEN rv.recent_count
                                ELSE rv.recent_count::float / bv.baseline_count
                            END as volume_increase
                        FROM recent_volume rv
                        LEFT JOIN baseline_volume bv ON rv.topic_id = bv.topic_id
                        WHERE rv.recent_count >= 5
                        ORDER BY volume_increase DESC
                    """, (recent_window, baseline_window, recent_window))
                    
                    results = cur.fetchall()
                    
                    campaigns = []
                    for row in results:
                        if row['volume_increase'] >= min_volume_increase:
                            campaigns.append({
                                'topic_id': row['topic_id'],
                                'topic_name': row['topic_name'],
                                'recent_volume': row['recent_count'],
                                'baseline_volume': row['baseline_count'],
                                'volume_increase': row['volume_increase'],
                                'detected_at': current_time.isoformat(),
                                'severity': 'high' if row['volume_increase'] >= 5.0 else 'medium'
                            })
                    
                    return campaigns
                    
        except Exception as e:
            logger.error(f"Campaign detection failed: {e}")
            return []
    
    async def get_topic_statistics(self) -> Dict[str, Any]:
        """Get comprehensive topic statistics"""
        try:
            with get_conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    # Get basic statistics
                    cur.execute("""
                        SELECT 
                            COUNT(DISTINCT tc.id) as total_topics,
                            COUNT(atc.article_id) as total_assignments,
                            AVG(tc.coherence_score) as avg_coherence,
                            MAX(tc.created_at) as latest_topic_created
                        FROM topic_clusters tc
                        LEFT JOIN article_topic_clusters atc ON tc.id = atc.cluster_id
                    """)
                    stats = cur.fetchone()
                    
                    # Get top topics by volume
                    cur.execute("""
                        SELECT 
                            tc.id,
                            tc.cluster_name,
                            tc.article_count,
                            tc.coherence_score,
                            tc.created_at
                        FROM topic_clusters tc
                        ORDER BY tc.article_count DESC
                        LIMIT 10
                    """)
                    top_topics = cur.fetchall()
                    
                    return {
                        'total_topics': stats['total_topics'],
                        'total_assignments': stats['total_assignments'],
                        'avg_coherence': float(stats['avg_coherence']) if stats['avg_coherence'] else 0.0,
                        'latest_topic_created': stats['latest_topic_created'].isoformat() if stats['latest_topic_created'] else None,
                        'top_topics': [dict(row) for row in top_topics],
                        'generated_at': datetime.now().isoformat()
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get topic statistics: {e}")
            return {
                'total_topics': 0,
                'total_assignments': 0,
                'error': str(e)
            }

# Global instance
topic_discovery_service = TopicDiscoveryService()
