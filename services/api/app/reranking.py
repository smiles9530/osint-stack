"""
Reranking service for improving search result relevance
Uses cross-encoder models to rerank search results
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

logger = logging.getLogger("osint_api")

class RerankingService:
    """Service for reranking search results using cross-encoder models"""
    
    def __init__(self):
        self.rerank_model = None
        self._initialize_rerank_model()
    
    def _initialize_rerank_model(self):
        """Initialize reranking model"""
        try:
            from sentence_transformers import SentenceTransformer
            from .gpu_utils import gpu_manager, model_device_manager
            
            # Use a sentence transformer model for reranking with GPU support
            model_name = 'BAAI/bge-reranker-base'
            device = model_device_manager.get_model_device(model_name)
            
            self.rerank_model = SentenceTransformer(model_name)
            
            # Move to appropriate device
            if gpu_manager.is_gpu_available():
                self.rerank_model = self.rerank_model.to(device)
            
            logger.info(f"Loaded reranking model: {model_name} on {device}")
        except ImportError:
            logger.warning("sentence-transformers not available, using simple reranking")
            self.rerank_model = None
    
    async def rerank_results(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]], 
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results using cross-encoder model
        
        Args:
            query: Original search query
            candidates: List of candidate results to rerank
            top_k: Number of top results to return
        
        Returns:
            Reranked list of results
        """
        if not candidates:
            return []
        
        try:
            if self.rerank_model:
                # Use cross-encoder for reranking
                return await self._cross_encoder_rerank(query, candidates, top_k)
            else:
                # Fallback to simple reranking
                return await self._simple_rerank(query, candidates, top_k)
                
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return candidates[:top_k]
    
    async def _cross_encoder_rerank(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Rerank using sentence transformer model"""
        try:
            # Prepare documents
            documents = []
            for candidate in candidates:
                # Combine title and text for better context
                doc_text = f"{candidate.get('title', '')} {candidate.get('text', '')[:500]}"
                documents.append(doc_text)
            
            # Get relevance scores using sentence transformer
            query_embedding = self.rerank_model.encode([query])
            doc_embeddings = self.rerank_model.encode(documents)
            
            # Calculate cosine similarity
            import numpy as np
            similarities = np.dot(query_embedding, doc_embeddings.T)[0]
            
            # Combine with original candidates and sort
            reranked_candidates = []
            for i, candidate in enumerate(candidates):
                candidate['rerank_score'] = float(similarities[i])
                reranked_candidates.append(candidate)
            
            # Sort by rerank score (descending)
            reranked_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            return reranked_candidates[:top_k]
            
        except Exception as e:
            logger.error(f"Sentence transformer reranking failed: {e}")
            return candidates[:top_k]
    
    async def _simple_rerank(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Simple reranking based on existing scores and text matching"""
        try:
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            for candidate in candidates:
                # Calculate text matching score
                title = candidate.get('title', '').lower()
                text = candidate.get('text', '').lower()
                
                # Count query word matches in title (higher weight)
                title_matches = sum(1 for word in query_words if word in title)
                text_matches = sum(1 for word in query_words if word in text)
                
                # Calculate combined score
                text_score = (title_matches * 2 + text_matches) / len(query_words)
                
                # Combine with existing similarity score
                similarity_score = candidate.get('similarity_score', 0)
                candidate['rerank_score'] = (similarity_score * 0.7) + (text_score * 0.3)
            
            # Sort by rerank score
            reranked_candidates = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
            
            return reranked_candidates[:top_k]
            
        except Exception as e:
            logger.error(f"Simple reranking failed: {e}")
            return candidates[:top_k]
    
    async def rerank_hybrid_results(
        self, 
        query: str, 
        vector_results: List[Dict[str, Any]], 
        bm25_results: List[Dict[str, Any]], 
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Rerank combined vector and BM25 results
        
        Args:
            query: Search query
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            top_k: Number of top results to return
        
        Returns:
            Reranked combined results
        """
        try:
            # Combine results and deduplicate
            combined_results = self._combine_and_deduplicate(vector_results, bm25_results)
            
            # Rerank combined results
            reranked = await self.rerank_results(query, combined_results, top_k)
            
            return reranked
            
        except Exception as e:
            logger.error(f"Hybrid reranking failed: {e}")
            return vector_results[:top_k]
    
    def _combine_and_deduplicate(
        self, 
        vector_results: List[Dict[str, Any]], 
        bm25_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Combine and deduplicate vector and BM25 results"""
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
            
            # Weighted combination (can be tuned)
            result['combined_score'] = (vector_score * 0.6) + (bm25_score * 0.4)
        
        # Convert to list and sort by combined score
        combined_results = list(unique_results.values())
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return combined_results
    
    async def batch_rerank(
        self, 
        queries_and_candidates: List[Tuple[str, List[Dict[str, Any]]]], 
        top_k: int = 20
    ) -> List[List[Dict[str, Any]]]:
        """
        Batch rerank multiple queries and their candidates
        
        Args:
            queries_and_candidates: List of (query, candidates) tuples
            top_k: Number of top results per query
        
        Returns:
            List of reranked results for each query
        """
        results = []
        
        for query, candidates in queries_and_candidates:
            try:
                reranked = await self.rerank_results(query, candidates, top_k)
                results.append(reranked)
            except Exception as e:
                logger.error(f"Batch reranking failed for query '{query}': {e}")
                results.append(candidates[:top_k])
        
        return results

# Global instance
reranking_service = RerankingService()
