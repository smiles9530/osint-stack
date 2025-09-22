"""
PGVector search integration for OSINT stack
Provides vector similarity search using PostgreSQL's pgvector extension
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
import psycopg2.extras
from contextlib import contextmanager

from .config import settings
from .db import get_conn
from .embedding import embed_texts

logger = logging.getLogger("osint_api")

class PGVectorSearch:
    """PGVector search service for semantic similarity search"""
    
    def __init__(self):
        self.embedding_model = None
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize embedding model for query processing"""
        try:
            from sentence_transformers import SentenceTransformer
            from .gpu_utils import gpu_manager, model_device_manager
            
            # Use a multilingual model for better cross-lingual support with GPU
            model_name = 'BAAI/bge-m3'
            device = model_device_manager.get_model_device(model_name)
            
            self.embedding_model = SentenceTransformer(model_name)
            
            # Move to appropriate device
            if gpu_manager.is_gpu_available():
                self.embedding_model = self.embedding_model.to(device)
            
            logger.info(f"Loaded multilingual embedding model: {model_name} on {device}")
        except ImportError:
            logger.warning("sentence-transformers not available, using Ollama for embeddings")
            self.embedding_model = None
    
    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for search query"""
        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode([query], normalize_embeddings=True)
                return embedding[0].tolist()
            except Exception as e:
                logger.error(f"Local embedding failed: {e}")
                # Fallback to Ollama
                pass
        
        # Fallback to Ollama or simple hash
        embeddings = await embed_texts([query])
        return embeddings[0]
    
    async def vector_search(
        self, 
        query: str, 
        limit: int = 20, 
        similarity_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using PGVector
        
        Args:
            query: Search query text
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score (0-1)
            filters: Additional filters (date_range, source_id, etc.)
        
        Returns:
            List of search results with similarity scores
        """
        try:
            # Generate query embedding
            query_vector = await self.embed_query(query)
            
            # Build SQL query with filters
            base_query = """
                SELECT 
                    a.id,
                    a.title,
                    a.text,
                    a.url,
                    a.published_at,
                    a.lang,
                    s.name as source_name,
                    1 - (e.vec <=> %s) as similarity_score,
                    e.model as embedding_model
                FROM articles a
                JOIN embeddings e ON a.id = e.article_id
                LEFT JOIN sources s ON a.source_id = s.id
                WHERE e.vec <=> %s < %s
            """
            
            params = [query_vector, query_vector, 1 - similarity_threshold]
            
            # Add filters
            if filters:
                if filters.get('date_from'):
                    base_query += " AND a.published_at >= %s"
                    params.append(filters['date_from'])
                
                if filters.get('date_to'):
                    base_query += " AND a.published_at <= %s"
                    params.append(filters['date_to'])
                
                if filters.get('source_id'):
                    base_query += " AND a.source_id = %s"
                    params.append(filters['source_id'])
                
                if filters.get('lang'):
                    base_query += " AND a.lang = %s"
                    params.append(filters['lang'])
            
            base_query += " ORDER BY e.vec <=> %s LIMIT %s"
            params.extend([query_vector, limit])
            
            # Execute query
            with get_conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(base_query, params)
                    results = cur.fetchall()
                    
                    # Convert to list of dicts
                    return [dict(row) for row in results]
                    
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def find_similar_articles(
        self, 
        article_id: int, 
        limit: int = 10,
        similarity_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Find articles similar to a given article using vector similarity
        
        Args:
            article_id: ID of the reference article
            limit: Maximum number of similar articles
            similarity_threshold: Minimum similarity score
        
        Returns:
            List of similar articles
        """
        try:
            with get_conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("""
                        WITH reference_vector AS (
                            SELECT vec FROM embeddings WHERE article_id = %s
                        )
                        SELECT 
                            a.id,
                            a.title,
                            a.text,
                            a.url,
                            a.published_at,
                            a.lang,
                            s.name as source_name,
                            1 - (e.vec <=> rv.vec) as similarity_score
                        FROM articles a
                        JOIN embeddings e ON a.id = e.article_id
                        LEFT JOIN sources s ON a.source_id = s.id
                        CROSS JOIN reference_vector rv
                        WHERE a.id != %s
                          AND e.vec <=> rv.vec < %s
                        ORDER BY e.vec <=> rv.vec
                        LIMIT %s
                    """, (article_id, article_id, 1 - similarity_threshold, limit))
                    
                    results = cur.fetchall()
                    return [dict(row) for row in results]
                    
        except Exception as e:
            logger.error(f"Similar articles search failed: {e}")
            return []
    
    async def detect_duplicates(
        self, 
        article_id: int,
        similarity_threshold: float = 0.9
    ) -> List[Dict[str, Any]]:
        """
        Detect potential duplicate articles using vector similarity
        
        Args:
            article_id: ID of the article to check for duplicates
            similarity_threshold: Similarity threshold for duplicate detection
        
        Returns:
            List of potential duplicate articles
        """
        try:
            with get_conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("""
                        WITH reference_vector AS (
                            SELECT vec FROM embeddings WHERE article_id = %s
                        )
                        SELECT 
                            a.id,
                            a.title,
                            a.url,
                            a.published_at,
                            s.name as source_name,
                            1 - (e.vec <=> rv.vec) as similarity_score
                        FROM articles a
                        JOIN embeddings e ON a.id = e.article_id
                        LEFT JOIN sources s ON a.source_id = s.id
                        CROSS JOIN reference_vector rv
                        WHERE a.id != %s
                          AND e.vec <=> rv.vec < %s
                        ORDER BY e.vec <=> rv.vec
                    """, (article_id, article_id, 1 - similarity_threshold))
                    
                    results = cur.fetchall()
                    return [dict(row) for row in results]
                    
        except Exception as e:
            logger.error(f"Duplicate detection failed: {e}")
            return []
    
    async def get_article_embedding(self, article_id: int) -> Optional[List[float]]:
        """Get embedding vector for a specific article"""
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT vec FROM embeddings WHERE article_id = %s", (article_id,))
                    result = cur.fetchone()
                    return result[0] if result else None
        except Exception as e:
            logger.error(f"Failed to get article embedding: {e}")
            return None
    
    async def batch_similarity_search(
        self, 
        queries: List[str], 
        limit_per_query: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform batch similarity search for multiple queries
        
        Args:
            queries: List of search queries
            limit_per_query: Maximum results per query
        
        Returns:
            Dictionary mapping queries to their results
        """
        results = {}
        
        for query in queries:
            try:
                query_results = await self.vector_search(query, limit=limit_per_query)
                results[query] = query_results
            except Exception as e:
                logger.error(f"Batch search failed for query '{query}': {e}")
                results[query] = []
        
        return results

# Global instance
pgvector_search = PGVectorSearch()
