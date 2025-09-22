"""
High-performance parallel ML processing module for OSINT stack
Optimizes CPU utilization through multiprocessing and async operations
"""
import asyncio
import multiprocessing as mp
import logging
import time
import psutil
from typing import Dict, List, Any, Optional, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import numpy as np
import pandas as pd
from functools import partial
import json
import hashlib

logger = logging.getLogger("osint_api")

@dataclass
class ProcessingStats:
    """Statistics for parallel processing operations"""
    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    processing_time: float = 0.0
    cpu_utilization: float = 0.0
    memory_usage: float = 0.0
    parallel_workers: int = 0

class ParallelMLProcessor:
    """High-performance parallel ML processor with CPU optimization"""
    
    def __init__(self, max_workers: Optional[int] = None, use_threading: bool = False):
        """
        Initialize parallel ML processor
        
        Args:
            max_workers: Maximum number of parallel workers (default: CPU count)
            use_threading: Use threading instead of multiprocessing for I/O bound tasks
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.use_threading = use_threading
        self.executor_class = ThreadPoolExecutor if use_threading else ProcessPoolExecutor
        self.stats = ProcessingStats()
        
        # CPU and memory monitoring
        self.cpu_count = psutil.cpu_count()
        self.memory_total = psutil.virtual_memory().total
        
        logger.info(f"Initialized ParallelMLProcessor with {self.max_workers} workers")
        logger.info(f"CPU cores: {self.cpu_count}, Total memory: {self.memory_total / (1024**3):.1f}GB")
    
    async def process_articles_batch(
        self, 
        articles: List[Dict[str, Any]], 
        processing_func: Callable,
        batch_size: Optional[int] = None,
        chunk_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple articles in parallel with optimal batching
        
        Args:
            articles: List of article data dictionaries
            processing_func: Function to process each article
            batch_size: Size of each batch (default: auto-calculate)
            chunk_size: Size of chunks for parallel processing (default: auto-calculate)
        
        Returns:
            List of processed results
        """
        start_time = time.time()
        total_articles = len(articles)
        
        if not articles:
            return []
        
        # Auto-calculate optimal batch and chunk sizes
        if batch_size is None:
            batch_size = min(50, max(10, total_articles // self.max_workers))
        
        if chunk_size is None:
            chunk_size = min(batch_size, max(5, total_articles // (self.max_workers * 2)))
        
        logger.info(f"Processing {total_articles} articles in parallel")
        logger.info(f"Batch size: {batch_size}, Chunk size: {chunk_size}, Workers: {self.max_workers}")
        
        # Split articles into chunks for parallel processing
        article_chunks = self._chunk_list(articles, chunk_size)
        
        # Process chunks in parallel
        results = []
        failed_count = 0
        
        with self.executor_class(max_workers=self.max_workers) as executor:
            # Submit all chunks for processing
            future_to_chunk = {
                executor.submit(self._process_chunk, chunk, processing_func): chunk 
                for chunk in article_chunks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                    self.stats.successful += len(chunk_results)
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
                    failed_count += len(chunk)
                    self.stats.failed += len(chunk)
        
        # Update statistics
        processing_time = time.time() - start_time
        self.stats.total_processed += total_articles
        self.stats.processing_time += processing_time
        self.stats.parallel_workers = self.max_workers
        
        # Update system metrics
        self._update_system_metrics()
        
        logger.info(f"Batch processing completed: {len(results)} successful, {failed_count} failed")
        logger.info(f"Processing time: {processing_time:.2f}s, Rate: {total_articles/processing_time:.1f} articles/sec")
        
        return results
    
    def _chunk_list(self, items: List[Any], chunk_size: int) -> List[List[Any]]:
        """Split list into chunks of specified size"""
        return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    
    def _process_chunk(self, chunk: List[Dict[str, Any]], processing_func: Callable) -> List[Dict[str, Any]]:
        """Process a chunk of articles using the specified function"""
        results = []
        
        for article in chunk:
            try:
                result = processing_func(article)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing article {article.get('id', 'unknown')}: {e}")
                # Add error result to maintain order
                results.append({
                    'article_id': article.get('id'),
                    'error': str(e),
                    'success': False
                })
        
        return results
    
    def _update_system_metrics(self):
        """Update CPU and memory usage metrics"""
        try:
            self.stats.cpu_utilization = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            self.stats.memory_usage = memory_info.percent
        except Exception as e:
            logger.warning(f"Could not update system metrics: {e}")
    
    async def process_ml_features_batch(
        self, 
        texts: List[str], 
        feature_extractors: List[Callable],
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract ML features from multiple texts in parallel
        
        Args:
            texts: List of text strings to process
            feature_extractors: List of feature extraction functions
            batch_size: Size of each batch (default: auto-calculate)
        
        Returns:
            List of feature dictionaries
        """
        if not texts:
            return []
        
        # Auto-calculate batch size based on text length and CPU cores
        if batch_size is None:
            avg_text_length = sum(len(text) for text in texts) / len(texts)
            if avg_text_length > 10000:  # Long texts
                batch_size = max(5, self.max_workers)
            else:  # Short texts
                batch_size = max(10, self.max_workers * 2)
        
        logger.info(f"Extracting features from {len(texts)} texts with {len(feature_extractors)} extractors")
        
        # Create text chunks
        text_chunks = self._chunk_list(texts, batch_size)
        
        # Process each chunk with all feature extractors
        all_features = []
        
        with self.executor_class(max_workers=self.max_workers) as executor:
            # Submit feature extraction tasks
            future_to_chunk = {}
            
            for i, chunk in enumerate(text_chunks):
                for j, extractor in enumerate(feature_extractors):
                    future = executor.submit(self._extract_features_chunk, chunk, extractor)
                    future_to_chunk[future] = (i, j, chunk)
            
            # Collect results
            chunk_results = {}
            for future in as_completed(future_to_chunk):
                chunk_idx, extractor_idx, chunk = future_to_chunk[future]
                try:
                    features = future.result()
                    if chunk_idx not in chunk_results:
                        chunk_results[chunk_idx] = {}
                    chunk_results[chunk_idx][extractor_idx] = features
                except Exception as e:
                    logger.error(f"Error extracting features: {e}")
            
            # Combine features for each text
            for chunk_idx, chunk in enumerate(text_chunks):
                for text_idx, text in enumerate(chunk):
                    combined_features = {}
                    for extractor_idx in range(len(feature_extractors)):
                        if (chunk_idx in chunk_results and 
                            extractor_idx in chunk_results[chunk_idx]):
                            text_features = chunk_results[chunk_idx][extractor_idx][text_idx]
                            combined_features.update(text_features)
                    all_features.append(combined_features)
        
        return all_features
    
    def _extract_features_chunk(self, texts: List[str], extractor: Callable) -> List[Dict[str, Any]]:
        """Extract features from a chunk of texts using a single extractor"""
        features = []
        for text in texts:
            try:
                feature_dict = extractor(text)
                features.append(feature_dict)
            except Exception as e:
                logger.error(f"Error extracting features: {e}")
                features.append({})
        return features
    
    async def process_embeddings_batch(
        self, 
        texts: List[str], 
        embedding_func: Callable,
        batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in parallel
        
        Args:
            texts: List of text strings
            embedding_func: Function to generate embeddings
            batch_size: Size of each batch (default: auto-calculate)
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # For embeddings, use smaller batches to manage memory
        if batch_size is None:
            batch_size = min(20, max(5, len(texts) // self.max_workers))
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Create text chunks
        text_chunks = self._chunk_list(texts, batch_size)
        
        # Process chunks in parallel
        all_embeddings = []
        
        with self.executor_class(max_workers=self.max_workers) as executor:
            future_to_chunk = {
                executor.submit(self._generate_embeddings_chunk, chunk, embedding_func): chunk 
                for chunk in text_chunks
            }
            
            for future in as_completed(future_to_chunk):
                try:
                    chunk_embeddings = future.result()
                    all_embeddings.extend(chunk_embeddings)
                except Exception as e:
                    logger.error(f"Error generating embeddings: {e}")
                    # Add empty embeddings for failed chunk
                    chunk = future_to_chunk[future]
                    all_embeddings.extend([[] for _ in chunk])
        
        return all_embeddings
    
    def _generate_embeddings_chunk(self, texts: List[str], embedding_func: Callable) -> List[List[float]]:
        """Generate embeddings for a chunk of texts"""
        embeddings = []
        for text in texts:
            try:
                embedding = embedding_func(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                embeddings.append([])
        return embeddings
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return {
            "total_processed": self.stats.total_processed,
            "successful": self.stats.successful,
            "failed": self.stats.failed,
            "success_rate": self.stats.successful / max(1, self.stats.total_processed),
            "avg_processing_time": self.stats.processing_time / max(1, self.stats.total_processed),
            "parallel_workers": self.stats.parallel_workers,
            "cpu_utilization": self.stats.cpu_utilization,
            "memory_usage": self.stats.memory_usage,
            "cpu_count": self.cpu_count,
            "memory_total_gb": self.memory_total / (1024**3)
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.stats = ProcessingStats()

# Global instance
parallel_ml_processor = ParallelMLProcessor()

# Convenience functions for common operations
async def process_articles_parallel(
    articles: List[Dict[str, Any]], 
    processing_func: Callable,
    batch_size: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Process articles in parallel using the global processor"""
    return await parallel_ml_processor.process_articles_batch(
        articles, processing_func, batch_size
    )

async def extract_features_parallel(
    texts: List[str], 
    feature_extractors: List[Callable],
    batch_size: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Extract features from texts in parallel"""
    return await parallel_ml_processor.process_ml_features_batch(
        texts, feature_extractors, batch_size
    )

async def generate_embeddings_parallel(
    texts: List[str], 
    embedding_func: Callable,
    batch_size: Optional[int] = None
) -> List[List[float]]:
    """Generate embeddings for texts in parallel"""
    return await parallel_ml_processor.process_embeddings_batch(
        texts, embedding_func, batch_size
    )

def get_parallel_processing_stats() -> Dict[str, Any]:
    """Get parallel processing performance statistics"""
    return parallel_ml_processor.get_performance_stats()
