"""
Hybrid search service combining PGVector and Typesense
Provides unified search interface with vector similarity and BM25
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import time

from .vector_search import pgvector_search
from .search import search_client
from .reranking import reranking_service
from .config import settings

logger = logging.getLogger("osint_api")

class HybridSearchService:
    """Hybrid search service combining PGVector and Typesense"""
    
    def __init__(self):
        self.vector_search = pgvector_search
        self.bm25_search = search_client
        self.reranking = reranking_service
    
    async def search(
        self, 
        query: str, 
        limit: int = 20, 
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        enable_reranking: bool = True,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4
    ) -> Dict[str, Any]:
        """
        Perform hybrid search combining vector and BM25 search
        
        Args:
            query: Search query
            limit: Maximum number of results
            offset: Number of results to skip
            filters: Additional filters (date_range, source_id, etc.)
            enable_reranking: Whether to use reranking
            vector_weight: Weight for vector search results
            bm25_weight: Weight for BM25 search results
        
        Returns:
            Search results with metadata
        """
        start_time = time.time()
        
        try:
            # Perform parallel searches
            vector_task = self._vector_search(query, limit * 2, filters)
            bm25_task = self._bm25_search(query, limit * 2, offset, filters)
            
            # Wait for both searches to complete
            vector_results, bm25_results = await asyncio.gather(
                vector_task, bm25_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(vector_results, Exception):
                logger.error(f"Vector search failed: {vector_results}")
                vector_results = []
            
            if isinstance(bm25_results, Exception):
                logger.error(f"BM25 search failed: {bm25_results}")
                bm25_results = {'hits': [], 'found': 0}
            
            # Extract results
            vector_hits = vector_results if isinstance(vector_results, list) else []
            bm25_hits = bm25_results.get('hits', []) if isinstance(bm25_results, dict) else []
            
            # Combine and rerank results
            if enable_reranking:
                combined_results = await self.reranking.rerank_hybrid_results(
                    query, vector_hits, bm25_hits, limit
                )
            else:
                combined_results = self._combine_results(
                    vector_hits, bm25_hits, vector_weight, bm25_weight, limit
                )
            
            # Calculate metadata
            query_time = time.time() - start_time
            total_found = len(vector_hits) + len(bm25_hits)
            
            return {
                'hits': combined_results,
                'total': total_found,
                'page': (offset // limit) + 1,
                'per_page': limit,
                'query_time': query_time,
                'search_type': 'hybrid',
                'vector_count': len(vector_hits),
                'bm25_count': len(bm25_hits),
                'reranked': enable_reranking
            }
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return {
                'hits': [],
                'total': 0,
                'page': 1,
                'per_page': limit,
                'query_time': time.time() - start_time,
                'search_type': 'hybrid',
                'error': str(e)
            }
    
    async def _vector_search(
        self, 
        query: str, 
        limit: int, 
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Perform vector search using PGVector"""
        try:
            return await self.vector_search.vector_search(
                query=query,
                limit=limit,
                similarity_threshold=0.7,
                filters=filters
            )
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    async def _bm25_search(
        self, 
        query: str, 
        limit: int, 
        offset: int, 
        filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform BM25 search using Typesense"""
        try:
            return self.bm25_search.search_articles(
                query=query,
                limit=limit,
                offset=offset,
                filters=filters
            )
        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            return {'hits': [], 'found': 0}
    
    def _combine_results(
        self, 
        vector_results: List[Dict[str, Any]], 
        bm25_results: List[Dict[str, Any]], 
        vector_weight: float, 
        bm25_weight: float, 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Combine vector and BM25 results without reranking"""
        # Create a dictionary to store unique results
        unique_results = {}
        
        # Add vector results
        for result in vector_results:
            article_id = result['id']
            if article_id not in unique_results:
                result['vector_score'] = result.get('similarity_score', 0)
                result['bm25_score'] = 0
                unique_results[article_id] = result
        
        # Add BM25 results
        for result in bm25_results:
            article_id = result['id']
            if article_id in unique_results:
                # Update existing result with BM25 score
                unique_results[article_id]['bm25_score'] = result.get('score', 0)
            else:
                # Add new result
                result['vector_score'] = 0
                result['bm25_score'] = result.get('score', 0)
                unique_results[article_id] = result
        
        # Calculate combined score
        for result in unique_results.values():
            vector_score = result.get('vector_score', 0)
            bm25_score = result.get('bm25_score', 0)
            
            # Normalize scores to 0-1 range
            vector_score = min(1.0, max(0.0, vector_score))
            bm25_score = min(1.0, max(0.0, bm25_score))
            
            # Weighted combination
            result['combined_score'] = (vector_score * vector_weight) + (bm25_score * bm25_weight)
        
        # Convert to list and sort by combined score
        combined_results = list(unique_results.values())
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return combined_results[:limit]
    
    async def search_by_type(
        self, 
        query: str, 
        search_type: str = 'hybrid',
        limit: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search using specific method (vector, bm25, or hybrid)
        
        Args:
            query: Search query
            search_type: 'vector', 'bm25', or 'hybrid'
            limit: Maximum number of results
            filters: Additional filters
        
        Returns:
            Search results
        """
        start_time = time.time()
        
        try:
            if search_type == 'vector':
                results = await self.vector_search.vector_search(
                    query=query,
                    limit=limit,
                    filters=filters
                )
                return {
                    'hits': results,
                    'total': len(results),
                    'query_time': time.time() - start_time,
                    'search_type': 'vector'
                }
            
            elif search_type == 'bm25':
                results = self.bm25_search.search_articles(
                    query=query,
                    limit=limit,
                    filters=filters
                )
                return {
                    'hits': results.get('hits', []),
                    'total': results.get('found', 0),
                    'query_time': time.time() - start_time,
                    'search_type': 'bm25'
                }
            
            else:  # hybrid
                return await self.search(query, limit, 0, filters)
                
        except Exception as e:
            logger.error(f"Search by type failed: {e}")
            return {
                'hits': [],
                'total': 0,
                'query_time': time.time() - start_time,
                'search_type': search_type,
                'error': str(e)
            }
    
    async def find_similar_articles(
        self, 
        article_id: int, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find articles similar to a given article"""
        try:
            return await self.vector_search.find_similar_articles(
                article_id=article_id,
                limit=limit
            )
        except Exception as e:
            logger.error(f"Similar articles search failed: {e}")
            return []
    
    async def detect_duplicates(
        self, 
        article_id: int,
        similarity_threshold: float = 0.9
    ) -> List[Dict[str, Any]]:
        """Detect potential duplicate articles"""
        try:
            return await self.vector_search.detect_duplicates(
                article_id=article_id,
                similarity_threshold=similarity_threshold
            )
        except Exception as e:
            logger.error(f"Duplicate detection failed: {e}")
            return []
    
    async def batch_search(
        self, 
        queries: List[str], 
        limit_per_query: int = 10
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Perform batch search for multiple queries"""
        results = {}
        
        for query in queries:
            try:
                search_result = await self.search(query, limit_per_query)
                results[query] = search_result.get('hits', [])
            except Exception as e:
                logger.error(f"Batch search failed for query '{query}': {e}")
                results[query] = []
        
        return results

# Global instance
hybrid_search_service = HybridSearchService()
