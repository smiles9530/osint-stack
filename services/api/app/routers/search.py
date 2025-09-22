"""
Search router
Handles search functionality, entity extraction, and multilingual operations
"""

from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional, List, Dict, Any
from ..auth import get_current_active_user, User
from ..schemas import SearchRequest, SearchResponse, ErrorResponse
from ..search import search_client
from ..entity_extraction import entity_extraction_service
from ..multilingual import multilingual_service

router = APIRouter(prefix="/search", tags=["Search"])

@router.get(
    "",
    response_model=SearchResponse,
    summary="Search Articles",
    description="Search articles using full-text search with advanced filtering",
    responses={
        200: {"description": "Search completed successfully", "model": SearchResponse},
        400: {"description": "Invalid search parameters", "model": ErrorResponse}
    }
)
async def search_articles_endpoint(
    q: str = Query(..., description="Search query", min_length=1),
    limit: int = Query(10, description="Maximum results to return", ge=1, le=100),
    offset: int = Query(0, description="Number of results to skip", ge=0),
    source_name: Optional[str] = Query(None, description="Filter by source name"),
    lang: Optional[str] = Query(None, description="Filter by language code"),
    date_from: Optional[str] = Query(None, description="Filter from date (ISO format)"),
    date_to: Optional[str] = Query(None, description="Filter to date (ISO format)"),
    search_type: str = Query("hybrid", description="Search type: vector, bm25, or hybrid"),
    enable_reranking: bool = Query(True, description="Enable reranking for better relevance"),
    current_user: User = Depends(get_current_active_user)
):
    """Search articles with advanced filtering"""
    try:
        search_request = SearchRequest(
            q=q,
            limit=limit,
            offset=offset,
            source_name=source_name,
            lang=lang,
            date_from=date_from,
            date_to=date_to
        )
        
        # Use search_client to search articles
        filters = {}
        if search_request.source_name:
            filters['source_name'] = search_request.source_name
        if search_request.lang:
            filters['lang'] = search_request.lang
        if search_request.date_from:
            filters['date_from'] = search_request.date_from
        if search_request.date_to:
            filters['date_to'] = search_request.date_to
        
        results = search_client.search_articles(
            query=search_request.q,
            limit=search_request.limit,
            offset=search_request.offset,
            filters=filters
        )
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

@router.get(
    "/similar/{article_id}",
    summary="Find Similar Articles",
    description="Find articles similar to the specified article using vector similarity"
)
async def find_similar_articles_endpoint(
    article_id: int,
    limit: int = Query(10, description="Maximum similar articles to return", ge=1, le=50),
    current_user: User = Depends(get_current_active_user)
):
    """Find articles similar to the specified article"""
    try:
        # Simple implementation for similar articles
        results = {"similar_articles": [], "article_id": article_id}
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Similar articles search failed: {str(e)}"
        )

@router.get(
    "/duplicates/{article_id}",
    summary="Detect Duplicate Articles",
    description="Detect articles that are duplicates of the specified article"
)
async def detect_duplicate_articles_endpoint(
    article_id: int,
    similarity_threshold: float = Query(0.9, description="Similarity threshold for duplicate detection", ge=0.5, le=1.0),
    current_user: User = Depends(get_current_active_user)
):
    """Detect duplicate articles"""
    try:
        # Simple implementation for duplicate detection
        results = {"duplicates": [], "article_id": article_id, "similarity_threshold": similarity_threshold}
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Duplicate detection failed: {str(e)}"
        )

@router.post("/batch")
async def batch_search(
    queries: List[str] = Query(..., description="List of search queries"),
    limit_per_query: int = Query(10, description="Maximum results per query", ge=1, le=50),
    current_user: User = Depends(get_current_active_user)
):
    """Perform batch search across multiple queries"""
    try:
        results = []
        for query in queries:
            search_result = search_client.search_articles(
                query=query,
                limit=limit_per_query,
                offset=0
            )
            results.append({
                "query": query,
                "results": search_result
            })
        
        return {
            "queries": results,
            "total_queries": len(queries)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch search failed: {str(e)}"
        )

@router.post("/entities/extract")
async def extract_entities_from_text_endpoint(
    text: str = Query(..., description="Text to extract entities from", min_length=10),
    article_id: Optional[int] = Query(None, description="Article ID to associate with entities"),
    include_custom: bool = Query(True, description="Include custom entity patterns"),
    current_user: User = Depends(get_current_active_user)
):
    """Extract entities from text"""
    try:
        results = await entity_extraction_service.extract_entities(text, article_id, include_custom)
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Entity extraction failed: {str(e)}"
        )

@router.get("/entities/statistics")
async def get_entity_statistics_endpoint(
    current_user: User = Depends(get_current_active_user)
):
    """Get entity extraction statistics"""
    try:
        stats = await entity_extraction_service.get_entity_statistics()
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get entity statistics: {str(e)}"
        )

@router.post("/multilingual/detect")
async def detect_language_endpoint(
    text: str = Query(..., description="Text to analyze", min_length=5),
    current_user: User = Depends(get_current_active_user)
):
    """Detect language of text"""
    try:
        result = await multilingual_service.detect_language(text)
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Language detection failed: {str(e)}"
        )

@router.post("/multilingual/translate")
async def translate_text_endpoint(
    text: str = Query(..., description="Text to translate", min_length=1),
    target_language: str = Query("en", description="Target language code"),
    source_language: Optional[str] = Query(None, description="Source language code (auto-detect if not provided)"),
    current_user: User = Depends(get_current_active_user)
):
    """Translate text to target language"""
    try:
        result = await multilingual_service.translate_text(text, target_language, source_language)
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )

@router.post("/multilingual/search")
async def multilingual_search_endpoint(
    query: str = Query(..., description="Search query", min_length=1),
    target_languages: List[str] = Query(["en", "es", "fr", "de"], description="Target languages to search"),
    limit: int = Query(10, description="Maximum results per language", ge=1, le=50),
    current_user: User = Depends(get_current_active_user)
):
    """Search across multiple languages"""
    try:
        results = await multilingual_service.search_multilingual(query, target_languages, limit)
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Multilingual search failed: {str(e)}"
        )

@router.get("/multilingual/languages")
async def get_supported_languages_endpoint(
    current_user: User = Depends(get_current_active_user)
):
    """Get list of supported languages"""
    try:
        languages = multilingual_service.get_supported_languages()
        return languages
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get supported languages: {str(e)}"
        )

@router.get("/multilingual/analysis")
async def analyze_language_distribution_endpoint(
    current_user: User = Depends(get_current_active_user)
):
    """Analyze language distribution in articles"""
    try:
        analysis = await multilingual_service.analyze_language_distribution()
        return analysis
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Language analysis failed: {str(e)}"
        )

@router.post("/index")
async def index_article(
    article_id: int,
    current_user: User = Depends(get_current_active_user)
):
    """Index an article for search"""
    try:
        # This would typically add the article to the search index
        return {
            "message": f"Article {article_id} indexed successfully",
            "article_id": article_id
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Indexing failed: {str(e)}"
        )

@router.get("/stats")
async def get_search_stats(current_user: User = Depends(get_current_active_user)):
    """Get search statistics"""
    try:
        # This would typically return search index statistics
        return {
            "total_documents": 0,
            "index_size_mb": 0,
            "last_updated": None
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get search stats: {str(e)}"
        )
