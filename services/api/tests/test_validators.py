import pytest
from pydantic import ValidationError

from app.validators import (
    IngestRequestValidator, 
    EmbedRequestValidator, 
    LoginRequestValidator,
    URLValidator,
    TextSanitizer,
    LanguageValidator
)

class TestURLValidator:
    def test_valid_urls(self):
        """Test valid URL detection"""
        valid_urls = [
            "https://example.com",
            "http://example.com",
            "https://subdomain.example.com/path?query=value#fragment",
            "https://example.com:8080/path"
        ]
        
        for url in valid_urls:
            assert URLValidator.is_valid_url(url) == True

    def test_invalid_urls(self):
        """Test invalid URL detection"""
        invalid_urls = [
            "not-a-url",
            "ftp://example.com",  # Unsupported scheme
            "https://",  # Missing domain
            "example.com",  # Missing scheme
            ""
        ]
        
        for url in invalid_urls:
            assert URLValidator.is_valid_url(url) == False

    def test_url_sanitization(self):
        """Test URL sanitization"""
        # Test adding scheme
        assert URLValidator.sanitize_url("example.com") == "https://example.com"
        
        # Test whitespace removal
        assert URLValidator.sanitize_url("  https://example.com  ") == "https://example.com"
        
        # Test already valid URL
        assert URLValidator.sanitize_url("https://example.com") == "https://example.com"

class TestTextSanitizer:
    def test_sanitize_text(self):
        """Test text sanitization"""
        # Test null byte removal
        text_with_nulls = "Hello\x00World"
        assert TextSanitizer.sanitize_text(text_with_nulls) == "HelloWorld"
        
        # Test control character removal
        text_with_control = "Hello\x01\x02World"
        assert TextSanitizer.sanitize_text(text_with_control) == "HelloWorld"
        
        # Test whitespace normalization
        text_with_whitespace = "Hello   \n\n  World"
        assert TextSanitizer.sanitize_text(text_with_whitespace) == "Hello World"
        
        # Test length truncation
        long_text = "A" * 1000
        result = TextSanitizer.sanitize_text(long_text, max_length=100)
        assert len(result) == 100

    def test_sanitize_title(self):
        """Test title sanitization"""
        # Test HTML tag removal
        title_with_html = "<h1>Test Title</h1>"
        assert TextSanitizer.sanitize_title(title_with_html) == "Test Title"
        
        # Test length limit
        long_title = "A" * 1000
        result = TextSanitizer.sanitize_title(long_title)
        assert len(result) <= 500

class TestLanguageValidator:
    def test_valid_languages(self):
        """Test valid language detection"""
        valid_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko', 'ar']
        
        for lang in valid_languages:
            assert LanguageValidator.is_valid_language(lang) == True
            assert LanguageValidator.is_valid_language(lang.upper()) == True

    def test_invalid_languages(self):
        """Test invalid language detection"""
        invalid_languages = ['xx', 'invalid', 'english', '123', '']
        
        for lang in invalid_languages:
            assert LanguageValidator.is_valid_language(lang) == False

    def test_none_language(self):
        """Test None language handling"""
        assert LanguageValidator.is_valid_language(None) == True

class TestIngestRequestValidator:
    def test_valid_request(self):
        """Test valid ingest request"""
        data = {
            "url": "https://example.com/article",
            "title": "Test Article",
            "published_at": "2023-01-01T00:00:00Z",
            "source": "example.com",
            "lang": "en"
        }
        
        request = IngestRequestValidator(**data)
        assert request.url == "https://example.com/article"
        assert request.title == "Test Article"
        assert request.lang == "en"

    def test_url_validation(self):
        """Test URL validation in request"""
        # Valid URL
        data = {"url": "https://example.com"}
        request = IngestRequestValidator(**data)
        assert request.url == "https://example.com"
        
        # Invalid URL
        with pytest.raises(ValidationError):
            IngestRequestValidator(url="not-a-url")

    def test_url_sanitization(self):
        """Test URL sanitization in request"""
        data = {"url": "example.com"}  # Missing scheme
        request = IngestRequestValidator(**data)
        assert request.url == "https://example.com"

    def test_title_sanitization(self):
        """Test title sanitization in request"""
        data = {
            "url": "https://example.com",
            "title": "<h1>Test Title</h1>   "  # HTML tags and whitespace
        }
        request = IngestRequestValidator(**data)
        assert request.title == "Test Title"

    def test_language_validation(self):
        """Test language validation in request"""
        # Valid language
        data = {"url": "https://example.com", "lang": "en"}
        request = IngestRequestValidator(**data)
        assert request.lang == "en"
        
        # Invalid language
        with pytest.raises(ValidationError):
            IngestRequestValidator(url="https://example.com", lang="xx")

    def test_published_at_validation(self):
        """Test published_at validation in request"""
        # Valid ISO format
        data = {
            "url": "https://example.com",
            "published_at": "2023-01-01T00:00:00Z"
        }
        request = IngestRequestValidator(**data)
        assert request.published_at == "2023-01-01T00:00:00Z"
        
        # Invalid format
        with pytest.raises(ValidationError):
            IngestRequestValidator(url="https://example.com", published_at="invalid-date")

    def test_source_sanitization(self):
        """Test source sanitization in request"""
        data = {
            "url": "https://example.com",
            "source": "example.com<script>alert('xss')</script>"
        }
        request = IngestRequestValidator(**data)
        assert "<script>" not in request.source

class TestEmbedRequestValidator:
    def test_valid_request(self):
        """Test valid embed request"""
        data = {"article_id": 1, "text": "Test text"}
        request = EmbedRequestValidator(**data)
        assert request.article_id == 1
        assert request.text == "Test text"

    def test_article_id_validation(self):
        """Test article_id validation"""
        # Valid ID
        data = {"article_id": 1}
        request = EmbedRequestValidator(**data)
        assert request.article_id == 1
        
        # Invalid ID (must be positive)
        with pytest.raises(ValidationError):
            EmbedRequestValidator(article_id=0)
        
        with pytest.raises(ValidationError):
            EmbedRequestValidator(article_id=-1)

    def test_text_sanitization(self):
        """Test text sanitization in request"""
        data = {
            "article_id": 1,
            "text": "Test\x00text with control characters"
        }
        request = EmbedRequestValidator(**data)
        assert request.text == "Testtext with control characters"

    def test_text_length_limit(self):
        """Test text length limit"""
        long_text = "A" * 100001  # Over 100k characters
        data = {"article_id": 1, "text": long_text}
        request = EmbedRequestValidator(**data)
        assert len(request.text) == 100000  # Should be truncated

class TestLoginRequestValidator:
    def test_valid_request(self):
        """Test valid login request"""
        data = {"username": "testuser", "password": "password123"}
        request = LoginRequestValidator(**data)
        assert request.username == "testuser"
        assert request.password == "password123"

    def test_username_validation(self):
        """Test username validation"""
        # Valid username
        data = {"username": "testuser123", "password": "password123"}
        request = LoginRequestValidator(**data)
        assert request.username == "testuser123"
        
        # Invalid username (contains special characters)
        with pytest.raises(ValidationError):
            LoginRequestValidator(username="test@user", password="password123")
        
        # Username too short
        with pytest.raises(ValidationError):
            LoginRequestValidator(username="ab", password="password123")

    def test_password_validation(self):
        """Test password validation"""
        # Valid password
        data = {"username": "testuser", "password": "password123"}
        request = LoginRequestValidator(**data)
        assert request.password == "password123"
        
        # Password too short
        with pytest.raises(ValidationError):
            LoginRequestValidator(username="testuser", password="12345")

    def test_username_case_handling(self):
        """Test username case handling"""
        data = {"username": "TestUser", "password": "password123"}
        request = LoginRequestValidator(**data)
        assert request.username == "testuser"  # Should be lowercased
