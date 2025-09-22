import re
import urllib.parse
from typing import Optional, Dict, List
from pydantic import BaseModel, validator, Field
from datetime import datetime
import logging

logger = logging.getLogger("osint_api")

class URLValidator:
    """URL validation and sanitization"""
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Check if URL is valid"""
        try:
            result = urllib.parse.urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    @staticmethod
    def sanitize_url(url: str) -> str:
        """Sanitize URL by removing dangerous characters"""
        # Remove any whitespace
        url = url.strip()
        
        # Ensure it has a scheme
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        return url

class TextSanitizer:
    """Text sanitization utilities"""
    
    @staticmethod
    def sanitize_text(text: str, max_length: Optional[int] = None) -> str:
        """Sanitize text input"""
        if not text:
            return ""
            
        # Remove null bytes and control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        # Truncate if too long
        if max_length and len(text) > max_length:
            text = text[:max_length]
            logger.warning(f"Text truncated to {max_length} characters")
            
        return text
    
    @staticmethod
    def sanitize_title(title: str) -> str:
        """Sanitize article title"""
        if not title:
            return ""
            
        # Remove HTML tags
        title = re.sub(r'<[^>]+>', '', title)
        
        # Sanitize and limit length
        return TextSanitizer.sanitize_text(title, max_length=500)

class LanguageValidator:
    """Language code validation"""
    
    SUPPORTED_LANGUAGES = {
        'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko', 'ar'
    }
    
    @staticmethod
    def is_valid_language(lang: str) -> bool:
        """Check if language code is supported"""
        if not lang:
            return True  # Optional field
        return lang.lower() in LanguageValidator.SUPPORTED_LANGUAGES

class IngestRequestValidator(BaseModel):
    """Validated model for ingest requests"""
    url: str = Field(..., min_length=1, max_length=2048)
    title: Optional[str] = Field(None, max_length=500)
    published_at: Optional[str] = None
    source: Optional[str] = Field(None, max_length=100)
    lang: Optional[str] = Field(None, max_length=5)
    
    @validator('url')
    def validate_url(cls, v):
        if not URLValidator.is_valid_url(v):
            raise ValueError('Invalid URL format')
        return URLValidator.sanitize_url(v)
    
    @validator('title')
    def validate_title(cls, v):
        if v is not None:
            return TextSanitizer.sanitize_title(v)
        return v
    
    @validator('published_at')
    def validate_published_at(cls, v):
        if v is not None:
            try:
                # Try to parse ISO format
                datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError('Invalid date format. Use ISO 8601 format.')
        return v
    
    @validator('source')
    def validate_source(cls, v):
        if v is not None:
            # Remove potentially dangerous characters
            v = re.sub(r'[<>"\']', '', v)
            return v.strip()[:100]
        return v
    
    @validator('lang')
    def validate_lang(cls, v):
        if v is not None and not LanguageValidator.is_valid_language(v):
            raise ValueError(f'Unsupported language code. Supported: {", ".join(LanguageValidator.SUPPORTED_LANGUAGES)}')
        return v.lower() if v else v

class EmbedRequestValidator(BaseModel):
    """Validated model for embed requests"""
    article_id: int = Field(..., gt=0)
    text: Optional[str] = Field(None, max_length=100000)  # 100k chars max
    
    @validator('text')
    def validate_text(cls, v):
        if v is not None:
            return TextSanitizer.sanitize_text(v, max_length=100000)
        return v

class LoginRequestValidator(BaseModel):
    """Validated model for login requests"""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=100)
    
    @validator('username')
    def validate_username(cls, v):
        # Only allow alphanumeric and underscore
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError('Username can only contain letters, numbers, and underscores')
        return v.lower()
    
    @validator('password')
    def validate_password(cls, v):
        # Strong password validation
        if len(v) < 12:
            raise ValueError('Password must be at least 12 characters long')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'[0-9]', v):
            raise ValueError('Password must contain at least one digit')
        if not re.search(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\?]', v):
            raise ValueError('Password must contain at least one special character')
        return v

class PuppeteerScrapeRequestValidator(BaseModel):
    """Validated model for puppeteer scrape requests"""
    url: str = Field(..., min_length=1, max_length=2048)
    wait_for: str = Field("networkidle2", max_length=50)
    timeout: int = Field(30000, ge=1000, le=300000)  # 1s to 5min
    
    @validator('url')
    def validate_url(cls, v):
        if not URLValidator.is_valid_url(v):
            raise ValueError('Invalid URL format')
        return URLValidator.sanitize_url(v)
    
    @validator('wait_for')
    def validate_wait_for(cls, v):
        valid_options = ['networkidle0', 'networkidle2', 'load', 'domcontentloaded']
        if v not in valid_options:
            raise ValueError(f'Invalid wait_for option. Must be one of: {", ".join(valid_options)}')
        return v

class PuppeteerScreenshotRequestValidator(BaseModel):
    """Validated model for puppeteer screenshot requests"""
    url: str = Field(..., min_length=1, max_length=2048)
    full_page: bool = Field(True)
    format: str = Field("png", max_length=10)
    
    @validator('url')
    def validate_url(cls, v):
        if not URLValidator.is_valid_url(v):
            raise ValueError('Invalid URL format')
        return URLValidator.sanitize_url(v)
    
    @validator('format')
    def validate_format(cls, v):
        valid_formats = ['png', 'jpeg', 'webp']
        if v.lower() not in valid_formats:
            raise ValueError(f'Invalid format. Must be one of: {", ".join(valid_formats)}')
        return v.lower()

class PuppeteerPdfRequestValidator(BaseModel):
    """Validated model for puppeteer PDF generation requests"""
    url: str = Field(..., min_length=1, max_length=2048)
    options: Optional[Dict] = Field(None)
    
    @validator('url')
    def validate_url(cls, v):
        if not URLValidator.is_valid_url(v):
            raise ValueError('Invalid URL format')
        return URLValidator.sanitize_url(v)

class PuppeteerMetadataRequestValidator(BaseModel):
    """Validated model for puppeteer metadata requests"""
    url: str = Field(..., min_length=1, max_length=2048)
    
    @validator('url')
    def validate_url(cls, v):
        if not URLValidator.is_valid_url(v):
            raise ValueError('Invalid URL format')
        return URLValidator.sanitize_url(v)

class PuppeteerScrapeMultipleRequestValidator(BaseModel):
    """Validated model for puppeteer multiple scrape requests"""
    urls: List[str] = Field(..., min_items=1, max_items=200)
    max_concurrent: int = Field(3, ge=1, le=10)
    
    @validator('urls')
    def validate_urls(cls, v):
        if not v:
            raise ValueError('At least one URL is required')
        
        validated_urls = []
        for url in v:
            if not URLValidator.is_valid_url(url):
                raise ValueError(f'Invalid URL format: {url}')
            validated_urls.append(URLValidator.sanitize_url(url))
        
        return validated_urls