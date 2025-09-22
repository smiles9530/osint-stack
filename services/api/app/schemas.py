"""
Pydantic schemas for API documentation and validation
"""

from pydantic import BaseModel, Field, ConfigDict, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum


# ===== ENUMS =====

class IntelligencePriority(str, Enum):
    """Intelligence priority levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    
    class Config:
        json_schema_extra = {
            "examples": ["high", "medium", "low"]
        }


class ProcessingType(str, Enum):
    """Intelligence processing types"""
    COLLECTION = "collection"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    REPORTING = "reporting"
    
    class Config:
        json_schema_extra = {
            "examples": ["collection", "analysis", "synthesis", "reporting"]
        }


# ===== AUTHENTICATION SCHEMAS =====

class Token(BaseModel):
    """Authentication token response"""
    access_token: str = Field(..., description="JWT access token for API authentication")
    token_type: str = Field(default="bearer", description="Token type (always 'bearer')")
    
    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer"
            }
        }


class LoginRequest(BaseModel):
    """Login request schema"""
    username: str = Field(..., description="Username (3-50 characters)", min_length=3, max_length=50)
    password: str = Field(..., description="Password (6-100 characters)", min_length=6, max_length=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "username": "admin",
                "password": "your_secure_password"
            }
        }


class UserCreate(BaseModel):
    """User creation request schema"""
    username: str = Field(..., description="Username (3-50 characters)", min_length=3, max_length=50)
    email: str = Field(..., description="Valid email address", pattern=r'^[^@]+@[^@]+\.[^@]+$')
    password: str = Field(..., description="Password (6-100 characters)", min_length=6, max_length=100)
    is_superuser: bool = Field(default=False, description="Grant admin privileges")
    
    class Config:
        json_schema_extra = {
            "example": {
                "username": "newuser",
                "email": "user@example.com",
                "password": "your_secure_password",
                "is_superuser": False
            }
        }


class User(BaseModel):
    """User information schema"""
    id: int = Field(..., description="Unique user identifier")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    is_superuser: bool = Field(..., description="Admin privileges status")
    is_active: bool = Field(..., description="Account active status")
    created_at: datetime = Field(..., description="Account creation timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "username": "admin",
                "email": "admin@example.com",
                "is_superuser": True,
                "is_active": True,
                "created_at": "2024-01-01T00:00:00Z",
                "last_login": "2024-01-15T10:30:00Z"
            }
        }


# ===== ARTICLE SCHEMAS =====

class ArticleBase(BaseModel):
    """Base article schema"""
    url: str = Field(..., description="Article URL", example="https://example.com/article")
    title: str = Field(..., description="Article title", example="Breaking News: Important Update")
    text: str = Field(..., description="Article content text", example="This is the main content of the article...")
    lang: str = Field(default="en", description="Article language code (ISO 639-1)", example="en")
    published_at: Optional[datetime] = Field(None, description="Publication date", example="2024-01-15T10:30:00Z")
    source_name: Optional[str] = Field(None, description="Source publication name", example="Example News")


class ArticleCreate(ArticleBase):
    """Article creation schema"""
    pass


class Article(ArticleBase):
    """Article response schema"""
    id: int = Field(..., description="Unique article identifier")
    fetched_at: datetime = Field(..., description="Timestamp when article was fetched")
    source_name: Optional[str] = Field(None, description="Source publication name")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": 123,
                "url": "https://example.com/article",
                "title": "Breaking News: Important Update",
                "text": "This is the main content of the article...",
                "lang": "en",
                "published_at": "2024-01-15T10:30:00Z",
                "source_name": "Example News",
                "fetched_at": "2024-01-15T11:00:00Z"
            }
        }


class ArticleList(BaseModel):
    """Article list response schema"""
    articles: List[Article] = Field(..., description="List of articles")
    limit: int = Field(..., description="Pagination limit")
    offset: int = Field(..., description="Pagination offset")
    total: Optional[int] = Field(None, description="Total article count")
    
    class Config:
        json_schema_extra = {
            "example": {
                "articles": [
                    {
                        "id": 123,
                        "url": "https://example.com/article",
                        "title": "Breaking News: Important Update",
                        "text": "This is the main content...",
                        "lang": "en",
                        "published_at": "2024-01-15T10:30:00Z",
                        "source_name": "Example News",
                        "fetched_at": "2024-01-15T11:00:00Z"
                    }
                ],
                "limit": 10,
                "offset": 0,
                "total": 1
            }
        }




class EmbedRequest(BaseModel):
    """Embedding generation request schema"""
    article_id: int = Field(..., description="Article ID to embed")
    text: Optional[str] = Field(None, description="Text to embed (if not provided, will fetch from article)")


class EmbedResponse(BaseModel):
    """Embedding generation response schema"""
    article_id: int = Field(..., description="Article ID")
    dim: int = Field(..., description="Embedding dimension")
    model: str = Field(..., description="Model used for embedding")


# ===== SOURCE SCHEMAS =====

class Source(BaseModel):
    """Source configuration schema"""
    id: Optional[int] = Field(None, description="Source ID")
    title: str = Field(..., description="Source title")
    url: str = Field(..., description="Source URL")
    category_name: str = Field(..., description="Category name")
    subcategory_name: Optional[str] = Field(None, description="Subcategory name")
    language: str = Field(default="en", description="Source language")
    enabled: bool = Field(default=True, description="Source enabled status")
    fetch_period: int = Field(default=3600, description="Fetch period in seconds")
    intelligence_priority: Optional[IntelligencePriority] = Field(None, description="Intelligence priority")
    priority_score: Optional[float] = Field(None, description="Priority score (0.0-1.0)")


class SourceList(BaseModel):
    """Source list response schema"""
    sources: List[Source] = Field(..., description="List of sources")
    total_enabled: int = Field(..., description="Total enabled sources")
    total_sources: int = Field(..., description="Total sources")
    filtered_count: int = Field(..., description="Filtered count")


class IntelligenceSource(Source):
    """Intelligence-optimized source schema"""
    intelligence_domains: List[str] = Field(..., description="Intelligence domains")
    priority_score: float = Field(..., description="Priority score")
    intelligence_priority: IntelligencePriority = Field(..., description="Intelligence priority")


class IntelligenceSourceList(BaseModel):
    """Intelligence source list response schema"""
    sources: List[IntelligenceSource] = Field(..., description="List of intelligence sources")
    sources_by_domain: Dict[str, List[IntelligenceSource]] = Field(..., description="Sources grouped by domain")
    total_enabled: int = Field(..., description="Total enabled sources")
    filtered_count: int = Field(..., description="Filtered count")
    domain_filter: Optional[str] = Field(None, description="Applied domain filter")
    priority_filter: Optional[str] = Field(None, description="Applied priority filter")
    available_domains: List[str] = Field(..., description="Available domains")
    intelligence_summary: Dict[str, Any] = Field(..., description="Intelligence summary statistics")




# ===== PUPPETEER SCHEMAS =====

class PuppeteerScrapeRequest(BaseModel):
    """Puppeteer scraping request schema"""
    url: str = Field(..., min_length=1, max_length=2048, description="URL to scrape")
    wait_for: Optional[str] = Field(default="networkidle", description="Wait condition: networkidle, domcontentloaded, or CSS selector")
    timeout: int = Field(default=30000, ge=1000, le=300000, description="Timeout in milliseconds")
    user_agent: Optional[str] = Field(default="OSINT-Stack/1.0", description="User agent string")
    retries: int = Field(default=3, ge=0, le=10, description="Number of retry attempts")


class PuppeteerScreenshotRequest(BaseModel):
    """Puppeteer screenshot request schema"""
    url: str = Field(..., min_length=1, max_length=2048, description="URL to screenshot")
    full_page: bool = Field(default=True, description="Capture full page")
    format: str = Field(default="png", pattern="^(png|jpeg|webp)$", description="Image format")
    width: int = Field(default=1920, ge=320, le=4096, description="Viewport width")
    height: int = Field(default=1080, ge=240, le=4096, description="Viewport height")


class PuppeteerPdfRequest(BaseModel):
    """Puppeteer PDF generation request schema"""
    url: str = Field(..., min_length=1, max_length=2048, description="URL to convert to PDF")
    options: Optional[Dict[str, Any]] = Field(None, description="PDF generation options")


class PuppeteerMetadataRequest(BaseModel):
    """Puppeteer metadata request schema"""
    url: str = Field(..., min_length=1, max_length=2048, description="URL to get metadata from")


class PuppeteerScrapeMultipleRequest(BaseModel):
    """Puppeteer multiple URL scraping request schema"""
    urls: List[str] = Field(..., min_items=1, max_items=200, description="List of URLs to scrape")
    max_concurrent: int = Field(default=3, ge=1, le=10, description="Maximum concurrent requests")


class PuppeteerScrapeFlexibleRequest(BaseModel):
    """Flexible Puppeteer scraping request schema that accepts both single URLs and arrays"""
    urls: Union[str, List[str]] = Field(..., description="Single URL string or list of URLs to scrape")
    max_concurrent: int = Field(default=3, ge=1, le=10, description="Maximum concurrent requests")
    
    @validator('urls')
    def validate_urls(cls, v):
        if isinstance(v, str):
            # Convert single URL to list
            return [v]
        elif isinstance(v, list):
            # Validate list
            if len(v) < 1 or len(v) > 50:
                raise ValueError('URLs list must contain between 1 and 50 URLs')
            return v
        else:
            raise ValueError('URLs must be either a string or a list of strings')




class PuppeteerResponse(BaseModel):
    """Puppeteer operation response schema"""
    success: bool = Field(..., description="Operation success status")
    url: Optional[str] = Field(None, description="Processed URL")
    data: Optional[Dict[str, Any]] = Field(None, description="Extracted data")
    content: Optional[str] = Field(None, description="Extracted content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Page metadata")
    screenshot: Optional[str] = Field(None, description="Screenshot data (base64)")
    pdf: Optional[str] = Field(None, description="PDF data (base64)")
    performance: Optional[Dict[str, Any]] = Field(None, description="Performance metrics")
    error: Optional[str] = Field(None, description="Error message if failed")
    scraped_at: Optional[float] = Field(None, description="Timestamp when scraped")
    cached: Optional[bool] = Field(None, description="Whether content was cached")
    attempts: Optional[int] = Field(None, description="Number of attempts made")
    can_scrape: Optional[bool] = Field(None, description="Whether the URL can be scraped (compliance check result)")


# ===== BIAS ANALYSIS SCHEMAS =====

class BiasAnalysis(BaseModel):
    """Bias analysis result schema"""
    article_id: int = Field(..., description="Analyzed article ID")
    subjectivity: float = Field(..., description="Subjectivity score (0.0-1.0)")
    sensationalism: float = Field(..., description="Sensationalism score (0.0-1.0)")
    loaded_language: float = Field(..., description="Loaded language score (0.0-1.0)")
    bias_lr: float = Field(..., description="Left-right bias score (-1.0 to 1.0)")
    stance: str = Field(..., description="Political stance")
    evidence_density: float = Field(..., description="Evidence density score (0.0-1.0)")
    sentiment: str = Field(..., description="Sentiment classification")
    sentiment_confidence: float = Field(..., description="Sentiment confidence (0.0-1.0)")
    agenda_signals: List[str] = Field(..., description="Detected agenda signals")
    risk_flags: List[str] = Field(..., description="Risk flags")
    entities: List[Dict[str, Any]] = Field(..., description="Extracted entities")
    tags: List[str] = Field(..., description="Content tags")
    key_quotes: List[str] = Field(..., description="Key quotes")
    summary_bullets: List[str] = Field(..., description="Summary bullet points")
    created_at: datetime = Field(..., description="Analysis timestamp")


# ===== STANCE SENTIMENT BIAS ANALYSIS SCHEMAS =====

class StanceSentimentAnalysisRequest(BaseModel):
    """Request schema for stance/sentiment/bias analysis"""
    text: str = Field(..., description="Text to analyze", min_length=10)
    article_id: Optional[int] = Field(None, description="Article ID to associate with analysis")
    source_id: Optional[str] = Field(None, description="Source identifier")
    topic: Optional[str] = Field(None, description="Topic identifier")

class ChunkAnalysisResult(BaseModel):
    """Individual chunk analysis result"""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., description="Chunk text")
    sentiment: Dict[str, float] = Field(..., description="Sentiment scores")
    stance: Dict[str, float] = Field(..., description="Stance scores")
    toxicity: Dict[str, float] = Field(..., description="Toxicity scores")
    bias: Dict[str, float] = Field(..., description="Bias scores")
    confidence: float = Field(..., description="Analysis confidence")
    timestamp: str = Field(..., description="Analysis timestamp")

class AggregatedAnalysisResult(BaseModel):
    """Aggregated analysis result"""
    source_id: Optional[str] = Field(None, description="Source identifier")
    topic: Optional[str] = Field(None, description="Topic identifier")
    total_chunks: int = Field(..., description="Total number of chunks analyzed")
    sentiment_distribution: Dict[str, float] = Field(..., description="Average sentiment distribution")
    stance_distribution: Dict[str, float] = Field(..., description="Average stance distribution")
    toxicity_levels: Dict[str, float] = Field(..., description="Average toxicity levels")
    bias_scores: Dict[str, float] = Field(..., description="Average bias scores")
    confidence_avg: float = Field(..., description="Average confidence score")
    risk_flags: List[str] = Field(..., description="Detected risk flags")
    trend_direction: str = Field(..., description="Overall trend direction")
    timestamp: str = Field(..., description="Analysis timestamp")

class AnalysisAlert(BaseModel):
    """Analysis alert schema"""
    type: str = Field(..., description="Alert type")
    severity: str = Field(..., description="Alert severity")
    message: str = Field(..., description="Alert message")
    source_id: Optional[str] = Field(None, description="Source identifier")
    topic: Optional[str] = Field(None, description="Topic identifier")

class StanceSentimentAnalysisResponse(BaseModel):
    """Response schema for stance/sentiment/bias analysis"""
    chunks: List[ChunkAnalysisResult] = Field(..., description="Individual chunk results")
    aggregated: AggregatedAnalysisResult = Field(..., description="Aggregated analysis")
    alerts: List[AnalysisAlert] = Field(..., description="Generated alerts")
    analysis_timestamp: str = Field(..., description="Overall analysis timestamp")

class BatchAnalysisRequest(BaseModel):
    """Request schema for batch analysis"""
    texts: List[str] = Field(..., description="List of texts to analyze", min_items=1, max_items=100)
    analysis_types: List[str] = Field(default=["sentiment", "stance", "bias"], description="Types of analysis to perform")
    include_metadata: bool = Field(default=True, description="Whether to include metadata in results")

class SourceAnalysisRequest(BaseModel):
    """Request schema for source analysis"""
    source_id: str = Field(..., description="Source identifier")
    days: int = Field(default=7, description="Number of days to analyze", ge=1, le=30)

class TopicAnalysisRequest(BaseModel):
    """Request schema for topic analysis"""
    topic: str = Field(..., description="Topic identifier")
    days: int = Field(default=7, description="Number of days to analyze", ge=1, le=30)

class AnalysisThresholdUpdate(BaseModel):
    """Request schema for updating analysis thresholds"""
    threshold_name: str = Field(..., description="Threshold name")
    threshold_value: float = Field(..., description="New threshold value", ge=0.0, le=1.0)
    is_active: bool = Field(default=True, description="Whether threshold is active")

# ===== BANDIT SCHEMAS =====

class BanditSelection(BaseModel):
    """Bandit selection request schema"""
    keys: List[str] = Field(..., description="Available options")


class BanditUpdate(BaseModel):
    """Bandit reward update schema"""
    key: str = Field(..., description="Option key")
    reward: float = Field(..., description="Reward value (0.0-1.0)", ge=0.0, le=1.0)


class BanditStats(BaseModel):
    """Bandit statistics schema"""
    total_selections: int = Field(..., description="Total selections made")
    total_rewards: float = Field(..., description="Total rewards received")
    average_reward: float = Field(..., description="Average reward")
    option_stats: Dict[str, Dict[str, Any]] = Field(..., description="Statistics per option")


# ===== FORECASTING SCHEMAS =====

class ForecastRequest(BaseModel):
    """Forecast generation request schema"""
    series_data: List[Dict[str, Any]] = Field(..., description="Time series data")
    horizon_days: int = Field(default=7, description="Forecast horizon in days", ge=1, le=365)


class ForecastResponse(BaseModel):
    """Forecast response schema"""
    model_config = {"protected_namespaces": ()}
    
    forecast: List[Dict[str, Any]] = Field(..., description="Forecasted values")
    confidence_intervals: List[Dict[str, Any]] = Field(..., description="Confidence intervals")
    model_info: Dict[str, Any] = Field(..., description="Model information")
    accuracy_metrics: Dict[str, float] = Field(..., description="Accuracy metrics")


class TrendResponse(BaseModel):
    """Trend analysis response schema"""
    trends: List[Dict[str, Any]] = Field(..., description="Trend data")
    forecast: List[Dict[str, Any]] = Field(..., description="Forecast data")
    summary: Dict[str, Any] = Field(..., description="Trend summary")


# ===== INTELLIGENCE SCHEMAS =====

class IntelligenceProcessing(BaseModel):
    """Intelligence processing request schema"""
    domain: str = Field(..., description="Intelligence domain")
    sources: List[Source] = Field(..., description="Sources to process")
    analysis: Dict[str, Any] = Field(default_factory=dict, description="Analysis parameters")
    processing_type: ProcessingType = Field(..., description="Processing type")
    timestamp: Optional[datetime] = Field(None, description="Processing timestamp")


class IntelligenceProcessingResponse(BaseModel):
    """Intelligence processing response schema"""
    processing_id: int = Field(..., description="Processing ID")
    domain: str = Field(..., description="Processed domain")
    processing_type: ProcessingType = Field(..., description="Processing type")
    timestamp: datetime = Field(..., description="Processing timestamp")
    source_count: int = Field(..., description="Number of sources processed")
    summary: str = Field(..., description="Processing summary")
    status: str = Field(..., description="Processing status")
    insights: Optional[Dict[str, Any]] = Field(None, description="Domain-specific insights")


class IntelligenceSynthesis(BaseModel):
    """Intelligence synthesis request schema"""
    synthesis: Dict[str, Any] = Field(..., description="Synthesis data")
    processing_results: List[Dict[str, Any]] = Field(..., description="Processing results")
    timestamp: datetime = Field(..., description="Synthesis timestamp")
    collection_cycle: str = Field(..., description="Collection cycle identifier")


class IntelligenceSynthesisResponse(BaseModel):
    """Intelligence synthesis response schema"""
    synthesis_id: int = Field(..., description="Synthesis ID")
    status: str = Field(..., description="Synthesis status")
    timestamp: datetime = Field(..., description="Synthesis timestamp")
    collection_cycle: str = Field(..., description="Collection cycle")
    processing_count: int = Field(..., description="Number of processing results")


# ===== SEARCH SCHEMAS =====

class SearchRequest(BaseModel):
    """Search request schema"""
    q: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Maximum results", ge=1, le=100)
    offset: int = Field(default=0, description="Result offset", ge=0)
    source_name: Optional[str] = Field(None, description="Filter by source name")
    lang: Optional[str] = Field(None, description="Filter by language")
    date_from: Optional[str] = Field(None, description="Filter from date (ISO format)")
    date_to: Optional[str] = Field(None, description="Filter to date (ISO format)")


class SearchResponse(BaseModel):
    """Search response schema"""
    hits: List[Dict[str, Any]] = Field(..., description="Search results")
    total: int = Field(..., description="Total matching results")
    page: int = Field(..., description="Current page")
    per_page: int = Field(..., description="Results per page")
    query_time: float = Field(..., description="Query execution time")


# ===== FEEDBACK SCHEMAS =====

class FeedbackSubmission(BaseModel):
    """Feedback submission schema"""
    article_id: int = Field(..., description="Article ID")
    clicked: Optional[bool] = Field(None, description="Whether user clicked")
    upvote: Optional[bool] = Field(None, description="Whether user upvoted")
    correct_after_days: Optional[bool] = Field(None, description="Whether content was correct after days")
    feedback_score: Optional[float] = Field(None, description="Feedback score (0.0-1.0)", ge=0.0, le=1.0)


class FeedbackResponse(BaseModel):
    """Feedback response schema"""
    ok: bool = Field(..., description="Success status")
    message: str = Field(..., description="Response message")


# ===== DIGEST SCHEMAS =====

class Digest(BaseModel):
    """Digest schema"""
    id: str = Field(..., description="Digest ID")
    date: datetime = Field(..., description="Digest date")
    topic: str = Field(..., description="Digest topic")
    content_md: str = Field(..., description="Digest content (Markdown)")
    created_at: datetime = Field(..., description="Creation timestamp")


class DigestList(BaseModel):
    """Digest list response schema"""
    ok: bool = Field(..., description="Success status")
    digests: List[Digest] = Field(..., description="List of digests")
    count: int = Field(..., description="Number of digests")


# ===== HEALTH CHECK SCHEMAS =====

class HealthCheck(BaseModel):
    """Health check response schema"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Check timestamp")
    message: str = Field(..., description="Status message")


class QueueStats(BaseModel):
    """Queue statistics schema"""
    queue_stats: Dict[str, Any] = Field(..., description="Queue statistics")


# ===== ERROR SCHEMAS =====

class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    timestamp: Optional[datetime] = Field(None, description="Error timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Validation failed",
                "detail": "The provided URL is not valid",
                "request_id": "req_123456789",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class ValidationError(BaseModel):
    """Validation error schema"""
    loc: List[Union[str, int]] = Field(..., description="Error location in request")
    msg: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")
    
    class Config:
        json_schema_extra = {
            "example": {
                "loc": ["body", "url"],
                "msg": "field required",
                "type": "missing"
            }
        }


class HTTPValidationError(BaseModel):
    """HTTP validation error schema"""
    detail: List[ValidationError] = Field(..., description="List of validation errors")
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": [
                    {
                        "loc": ["body", "url"],
                        "msg": "field required",
                        "type": "missing"
                    }
                ]
            }
        }
