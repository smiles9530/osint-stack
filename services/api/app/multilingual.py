"""
Multilingual support service for cross-lingual search and analysis
Provides language detection, translation, and multilingual embeddings
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import re
from datetime import datetime

logger = logging.getLogger("osint_api")

class MultilingualService:
    """Service for multilingual text processing and analysis"""
    
    def __init__(self):
        self.language_detector = None
        self.translator = None
        self.multilingual_embedder = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize multilingual models"""
        try:
            # Language detection
            from langdetect import detect, detect_langs
            self.language_detector = detect
            self.language_detector_langs = detect_langs
            logger.info("Loaded language detection model")
        except ImportError:
            logger.warning("langdetect not available, using simple language detection")
            self.language_detector = None
        
        try:
            # Multilingual embeddings with GPU support
            from sentence_transformers import SentenceTransformer
            from .gpu_utils import gpu_manager, model_device_manager
            
            model_name = 'BAAI/bge-m3'
            device = model_device_manager.get_model_device(model_name)
            
            self.multilingual_embedder = SentenceTransformer(model_name)
            
            # Move to appropriate device
            if gpu_manager.is_gpu_available():
                self.multilingual_embedder = self.multilingual_embedder.to(device)
            
            logger.info(f"Loaded multilingual embedding model: {model_name} on {device}")
        except ImportError:
            logger.warning("sentence-transformers not available, using Ollama for embeddings")
            self.multilingual_embedder = None
        
        # Translation (using Ollama for now, can be enhanced with dedicated translation services)
        self.translator = None  # Will use Ollama for translation
    
    async def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect language of input text
        
        Args:
            text: Input text to analyze
        
        Returns:
            Language detection results
        """
        try:
            if self.language_detector:
                # Use langdetect for accurate detection
                detected_lang = self.language_detector(text)
                confidence_scores = self.language_detector_langs(text)
                
                return {
                    'language': detected_lang,
                    'confidence': max([score.prob for score in confidence_scores]),
                    'all_scores': [{'lang': score.lang, 'prob': score.prob} for score in confidence_scores],
                    'method': 'langdetect'
                }
            else:
                # Fallback to simple detection
                return await self._simple_language_detection(text)
                
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'all_scores': [],
                'method': 'fallback',
                'error': str(e)
            }
    
    async def _simple_language_detection(self, text: str) -> Dict[str, Any]:
        """Simple language detection based on character patterns"""
        try:
            # Count different character types
            latin_chars = len(re.findall(r'[a-zA-Z]', text))
            cyrillic_chars = len(re.findall(r'[а-яА-Я]', text))
            arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            total_chars = len(text)
            
            if total_chars == 0:
                return {'language': 'unknown', 'confidence': 0.0, 'method': 'simple'}
            
            # Determine language based on character distribution
            latin_ratio = latin_chars / total_chars
            cyrillic_ratio = cyrillic_chars / total_chars
            arabic_ratio = arabic_chars / total_chars
            chinese_ratio = chinese_chars / total_chars
            
            if latin_ratio > 0.7:
                return {'language': 'en', 'confidence': latin_ratio, 'method': 'simple'}
            elif cyrillic_ratio > 0.5:
                return {'language': 'ru', 'confidence': cyrillic_ratio, 'method': 'simple'}
            elif arabic_ratio > 0.5:
                return {'language': 'ar', 'confidence': arabic_ratio, 'method': 'simple'}
            elif chinese_ratio > 0.5:
                return {'language': 'zh', 'confidence': chinese_ratio, 'method': 'simple'}
            else:
                return {'language': 'unknown', 'confidence': 0.0, 'method': 'simple'}
                
        except Exception as e:
            logger.error(f"Simple language detection failed: {e}")
            return {'language': 'unknown', 'confidence': 0.0, 'method': 'simple', 'error': str(e)}
    
    async def translate_text(
        self, 
        text: str, 
        target_language: str = 'en',
        source_language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Translate text to target language
        
        Args:
            text: Text to translate
            target_language: Target language code
            source_language: Source language code (auto-detect if None)
        
        Returns:
            Translation results
        """
        try:
            # Detect source language if not provided
            if not source_language:
                lang_result = await self.detect_language(text)
                source_language = lang_result['language']
            
            # Skip translation if source and target are the same
            if source_language == target_language:
                return {
                    'original_text': text,
                    'translated_text': text,
                    'source_language': source_language,
                    'target_language': target_language,
                    'confidence': 1.0,
                    'method': 'no_translation_needed'
                }
            
            # Use Ollama for translation (simplified implementation)
            translated_text = await self._translate_with_ollama(text, source_language, target_language)
            
            return {
                'original_text': text,
                'translated_text': translated_text,
                'source_language': source_language,
                'target_language': target_language,
                'confidence': 0.8,  # Placeholder confidence
                'method': 'ollama'
            }
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return {
                'original_text': text,
                'translated_text': text,
                'source_language': source_language or 'unknown',
                'target_language': target_language,
                'confidence': 0.0,
                'method': 'failed',
                'error': str(e)
            }
    
    async def _translate_with_ollama(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str
    ) -> str:
        """Translate text using Ollama (simplified implementation)"""
        try:
            import httpx
            
            # Create translation prompt
            prompt = f"Translate the following text from {source_lang} to {target_lang}. Only return the translation, no explanations:\n\n{text}"
            
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{settings.ollama_host}/api/generate",
                    json={
                        "model": "llama3",  # Use available model
                        "prompt": prompt,
                        "stream": False
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result.get('response', text).strip()
                
        except Exception as e:
            logger.error(f"Ollama translation failed: {e}")
            return text
    
    async def get_multilingual_embedding(
        self, 
        text: str, 
        language: Optional[str] = None
    ) -> List[float]:
        """
        Get multilingual embedding for text
        
        Args:
            text: Input text
            language: Language code (optional, for optimization)
        
        Returns:
            Embedding vector
        """
        try:
            if self.multilingual_embedder:
                # Use multilingual model
                embedding = self.multilingual_embedder.encode([text], normalize_embeddings=True)
                return embedding[0].tolist()
            else:
                # Fallback to existing embedding service
                from .embedding import embed_texts
                embeddings = await embed_texts([text])
                return embeddings[0]
                
        except Exception as e:
            logger.error(f"Multilingual embedding failed: {e}")
            # Fallback to simple hash embedding
            from .embedding import _simple_hash_embedding
            return _simple_hash_embedding(text)
    
    async def search_multilingual(
        self, 
        query: str, 
        target_languages: List[str] = None,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Search across multiple languages
        
        Args:
            query: Search query
            target_languages: List of target languages to search
            limit: Maximum results per language
        
        Returns:
            Multilingual search results
        """
        try:
            if not target_languages:
                target_languages = ['en', 'es', 'fr', 'de', 'ar', 'zh', 'ru']
            
            # Detect query language
            lang_result = await self.detect_language(query)
            query_language = lang_result['language']
            
            # Translate query to target languages if needed
            translated_queries = {}
            for lang in target_languages:
                if lang == query_language:
                    translated_queries[lang] = query
                else:
                    translation = await self.translate_text(query, lang, query_language)
                    translated_queries[lang] = translation['translated_text']
            
            # Search in each language
            from .hybrid_search import hybrid_search_service
            results = {}
            
            for lang, translated_query in translated_queries.items():
                try:
                    lang_results = await hybrid_search_service.search_by_type(
                        query=translated_query,
                        search_type='hybrid',
                        limit=limit,
                        filters={'lang': lang}
                    )
                    results[lang] = {
                        'query': translated_query,
                        'results': lang_results.get('hits', []),
                        'total': lang_results.get('total', 0),
                        'query_time': lang_results.get('query_time', 0)
                    }
                except Exception as e:
                    logger.error(f"Search failed for language {lang}: {e}")
                    results[lang] = {
                        'query': translated_query,
                        'results': [],
                        'total': 0,
                        'error': str(e)
                    }
            
            return {
                'original_query': query,
                'query_language': query_language,
                'target_languages': target_languages,
                'results': results,
                'total_languages': len(target_languages)
            }
            
        except Exception as e:
            logger.error(f"Multilingual search failed: {e}")
            return {
                'original_query': query,
                'query_language': 'unknown',
                'target_languages': target_languages or [],
                'results': {},
                'error': str(e)
            }
    
    async def analyze_language_distribution(self) -> Dict[str, Any]:
        """Analyze language distribution in the corpus"""
        try:
            from .db import get_conn
            import psycopg2.extras
            
            with get_conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    # Get language distribution from articles
                    cur.execute("""
                        SELECT 
                            COALESCE(lang, 'unknown') as language,
                            COUNT(*) as article_count,
                            COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as percentage
                        FROM articles
                        WHERE created_at >= NOW() - INTERVAL '30 days'
                        GROUP BY lang
                        ORDER BY article_count DESC
                    """)
                    lang_distribution = cur.fetchall()
                    
                    # Get total articles
                    cur.execute("SELECT COUNT(*) as total FROM articles")
                    total_articles = cur.fetchone()['total']
                    
                    return {
                        'total_articles': total_articles,
                        'language_distribution': [dict(row) for row in lang_distribution],
                        'analyzed_at': datetime.now().isoformat()
                    }
                    
        except Exception as e:
            logger.error(f"Language distribution analysis failed: {e}")
            return {
                'total_articles': 0,
                'language_distribution': [],
                'error': str(e)
            }
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages"""
        return [
            {'code': 'en', 'name': 'English'},
            {'code': 'es', 'name': 'Spanish'},
            {'code': 'fr', 'name': 'French'},
            {'code': 'de', 'name': 'German'},
            {'code': 'ar', 'name': 'Arabic'},
            {'code': 'zh', 'name': 'Chinese'},
            {'code': 'ru', 'name': 'Russian'},
            {'code': 'ja', 'name': 'Japanese'},
            {'code': 'ko', 'name': 'Korean'},
            {'code': 'pt', 'name': 'Portuguese'},
            {'code': 'it', 'name': 'Italian'},
            {'code': 'nl', 'name': 'Dutch'},
            {'code': 'sv', 'name': 'Swedish'},
            {'code': 'no', 'name': 'Norwegian'},
            {'code': 'da', 'name': 'Danish'},
            {'code': 'fi', 'name': 'Finnish'},
            {'code': 'pl', 'name': 'Polish'},
            {'code': 'tr', 'name': 'Turkish'},
            {'code': 'hi', 'name': 'Hindi'},
            {'code': 'th', 'name': 'Thai'}
        ]

# Global instance
multilingual_service = MultilingualService()
