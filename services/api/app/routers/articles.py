"""
Articles router
Handles article CRUD operations, ingestion, and management
"""

from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional, List, Dict, Any
from ..auth import get_current_active_user, User
from ..schemas import (
    Article, ArticleList, ArticleCreate, 
    EmbedRequest, EmbedResponse, SearchRequest, SearchResponse,
    ErrorResponse, HTTPValidationError
)
from .. import db, embedding
from ..cache import cache, article_cache_key, articles_list_cache_key, embedding_cache_key

router = APIRouter(prefix="/articles", tags=["Articles"])


@router.post(
    "/embed",
    response_model=EmbedResponse,
    summary="Generate Embeddings",
    description="Generate vector embeddings for an article",
    responses={
        200: {"description": "Embeddings generated successfully", "model": EmbedResponse},
        404: {"description": "Article not found", "model": ErrorResponse},
        422: {"description": "Validation error", "model": HTTPValidationError}
    }
)
async def embed(
    payload: EmbedRequest, 
    current_user: User = Depends(get_current_active_user)
):
    """Generate embeddings for an article"""
    try:
        # Check if article exists
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, text FROM articles WHERE id = %s", (payload.article_id,))
                article = cur.fetchone()
                
                if not article:
                    raise HTTPException(
                        status_code=404,
                        detail="Article not found"
                    )
                
                article_id, text = article
                
                # Use provided text or fetch from article
                text_to_embed = payload.text or text
                
                # Generate embedding
                embedding_vector = embedding.generate_embedding(text_to_embed)
                
                # Store embedding
                db.insert_embedding(article_id, embedding_vector, "bge-m3")
                
                return EmbedResponse(
                    article_id=article_id,
                    dim=len(embedding_vector),
                    model="bge-m3"
                )
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding generation failed: {str(e)}"
        )

@router.get(
    "/{article_id}",
    response_model=Article,
    summary="Get Article",
    description="Retrieve a specific article by ID",
    responses={
        200: {"description": "Article retrieved successfully", "model": Article},
        404: {"description": "Article not found", "model": ErrorResponse}
    }
)
async def get_article(
    article_id: int, 
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific article by ID"""
    try:
        # Check cache first
        cache_key = article_cache_key(article_id)
        cached_article = await cache.get(cache_key)
        if cached_article:
            return Article(**cached_article)
        
        # Fetch from database
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT a.id, a.url, a.title, a.text, a.lang, a.published_at, 
                           a.fetched_at, s.name as source_name
                    FROM articles a
                    LEFT JOIN sources s ON a.source_id = s.id
                    WHERE a.id = %s
                """, (article_id,))
                
                article_data = cur.fetchone()
                if not article_data:
                    raise HTTPException(
                        status_code=404,
                        detail="Article not found"
                    )
                
                article = Article(
                    id=article_data[0],
                    url=article_data[1],
                    title=article_data[2],
                    text=article_data[3],
                    lang=article_data[4],
                    published_at=article_data[5],
                    fetched_at=article_data[6],
                    source_name=article_data[7]
                )
                
                # Cache the result
                await cache.set(cache_key, article.dict())
                
                return article
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve article: {str(e)}"
        )

@router.get(
    "",
    response_model=ArticleList,
    summary="List Articles",
    description="List articles with pagination and filtering",
    responses={
        200: {"description": "Articles retrieved successfully", "model": ArticleList}
    }
)
async def list_articles(
    limit: int = Query(10, description="Number of articles to return", ge=1, le=100), 
    offset: int = Query(0, description="Number of articles to skip", ge=0),
    current_user: User = Depends(get_current_active_user)
):
    """List articles with pagination"""
    try:
        # Check cache first
        cache_key = articles_list_cache_key(limit, offset)
        cached_result = await cache.get(cache_key)
        if cached_result:
            return ArticleList(**cached_result)
        
        # Fetch from database
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                # Get total count
                cur.execute("SELECT COUNT(*) FROM articles")
                total = cur.fetchone()[0]
                
                # Get articles
                cur.execute("""
                    SELECT a.id, a.url, a.title, a.text, a.lang, a.published_at, 
                           a.fetched_at, s.name as source_name
                    FROM articles a
                    LEFT JOIN sources s ON a.source_id = s.id
                    ORDER BY a.published_at DESC
                    LIMIT %s OFFSET %s
                """, (limit, offset))
                
                articles_data = cur.fetchall()
                articles = []
                
                for article_data in articles_data:
                    article = Article(
                        id=article_data[0],
                        url=article_data[1],
                        title=article_data[2],
                        text=article_data[3],
                        lang=article_data[4],
                        published_at=article_data[5],
                        fetched_at=article_data[6],
                        source_name=article_data[7]
                    )
                    articles.append(article)
                
                result = ArticleList(
                    articles=articles,
                    limit=limit,
                    offset=offset,
                    total=total
                )
                
                # Cache the result
                await cache.set(cache_key, result.dict())
                
                return result
                
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list articles: {str(e)}"
        )
