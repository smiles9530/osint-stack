import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json

from app.main import app
from app.auth import create_access_token

client = TestClient(app)

class TestHealthEndpoint:
    def test_healthz_success(self):
        """Test health check endpoint returns success"""
        with patch('app.main.db.get_conn') as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = (1,)
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cursor
            
            response = client.get("/healthz")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert "timestamp" in data

    def test_healthz_database_error(self):
        """Test health check endpoint handles database errors"""
        with patch('app.main.db.get_conn') as mock_conn:
            mock_conn.side_effect = Exception("Database connection failed")
            
            response = client.get("/healthz")
            assert response.status_code == 500
            data = response.json()
            assert data["error"] == "Database Error"

class TestAuthentication:
    def test_login_success(self):
        """Test successful login"""
        login_data = {
            "username": "admin",
            "password": "test_password"
        }
        
        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_invalid_credentials(self):
        """Test login with invalid credentials"""
        login_data = {
            "username": "admin",
            "password": "wrongpassword"
        }
        
        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 422
        data = response.json()
        assert "Validation Error" in data["error"]

    def test_login_invalid_username_format(self):
        """Test login with invalid username format"""
        login_data = {
            "username": "admin@invalid",
            "password": "test_password"
        }
        
        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 422

class TestArticleEndpoints:
    def setup_method(self):
        """Setup test data"""
        # Get auth token
        login_response = client.post("/auth/login", json={
            "username": "admin",
            "password": "test_password"
        })
        self.token = login_response.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def test_fetch_extract_success(self):
        """Test successful article extraction"""
        with patch('app.main.trafilatura.fetch_url') as mock_fetch, \
             patch('app.main.trafilatura.extract') as mock_extract, \
             patch('app.main.db.upsert_article') as mock_upsert:
            
            mock_fetch.return_value = "<html>Test content</html>"
            mock_extract.return_value = "Extracted article text"
            mock_upsert.return_value = 1
            
            payload = {
                "url": "https://example.com/article",
                "title": "Test Article",
                "source": "example.com",
                "lang": "en"
            }
            
            response = client.post("/ingest/fetch_extract", 
                                 json=payload, 
                                 headers=self.headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["article_id"] == 1
            assert data["chars"] == len("Extracted article text")

    def test_fetch_extract_invalid_url(self):
        """Test article extraction with invalid URL"""
        payload = {
            "url": "not-a-valid-url",
            "title": "Test Article"
        }
        
        response = client.post("/ingest/fetch_extract", 
                             json=payload, 
                             headers=self.headers)
        
        assert response.status_code == 422
        data = response.json()
        assert "Validation Error" in data["error"]

    def test_fetch_extract_unauthorized(self):
        """Test article extraction without authentication"""
        payload = {
            "url": "https://example.com/article",
            "title": "Test Article"
        }
        
        response = client.post("/ingest/fetch_extract", json=payload)
        assert response.status_code == 401

    def test_embed_success(self):
        """Test successful embedding generation"""
        with patch('app.main.db.get_conn') as mock_conn, \
             patch('app.main.embedding.embed_texts') as mock_embed, \
             patch('app.main.db.insert_embedding') as mock_insert:
            
            # Mock database response
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = ("Test article text",)
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cursor
            
            # Mock embedding response
            mock_embed.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
            
            payload = {"article_id": 1}
            
            response = client.post("/embed", 
                                json=payload, 
                                headers=self.headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["article_id"] == 1
            assert data["dim"] == 5

    def test_embed_article_not_found(self):
        """Test embedding generation for non-existent article"""
        with patch('app.main.db.get_conn') as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = None
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cursor
            
            payload = {"article_id": 999}
            
            response = client.post("/embed", 
                                json=payload, 
                                headers=self.headers)
            
            assert response.status_code == 422
            data = response.json()
            assert "Validation Error" in data["error"]

    def test_get_article_success(self):
        """Test successful article retrieval"""
        with patch('app.main.db.get_conn') as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = (
                1, "https://example.com", "Test Article", 
                "Article content", "en", None, None, "example.com"
            )
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cursor
            
            response = client.get("/articles/1", headers=self.headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == 1
            assert data["url"] == "https://example.com"
            assert data["title"] == "Test Article"

    def test_get_article_not_found(self):
        """Test article retrieval for non-existent article"""
        with patch('app.main.db.get_conn') as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = None
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cursor
            
            response = client.get("/articles/999", headers=self.headers)
            
            assert response.status_code == 422
            data = response.json()
            assert "Validation Error" in data["error"]

    def test_list_articles_success(self):
        """Test successful article listing"""
        with patch('app.main.db.get_conn') as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [
                (1, "https://example.com", "Test Article", "en", None, None, "example.com")
            ]
            mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cursor
            
            response = client.get("/articles?limit=10&offset=0", headers=self.headers)
            
            assert response.status_code == 200
            data = response.json()
            assert "articles" in data
            assert len(data["articles"]) == 1
            assert data["limit"] == 10
            assert data["offset"] == 0

class TestValidation:
    def test_url_validation(self):
        """Test URL validation"""
        # Valid URLs
        valid_urls = [
            "https://example.com",
            "http://example.com",
            "https://subdomain.example.com/path?query=value"
        ]
        
        for url in valid_urls:
            payload = {"url": url, "title": "Test"}
            response = client.post("/ingest/fetch_extract", 
                                 json=payload, 
                                 headers={"Authorization": f"Bearer {self.get_token()}"})
            # Should not fail on URL validation (might fail on other parts)
            assert response.status_code != 422 or "Invalid URL format" not in str(response.json())

    def test_text_sanitization(self):
        """Test text sanitization"""
        malicious_text = "<script>alert('xss')</script>Normal text"
        payload = {
            "url": "https://example.com",
            "title": malicious_text
        }
        
        # The title should be sanitized
        response = client.post("/ingest/fetch_extract", 
                             json=payload, 
                             headers={"Authorization": f"Bearer {self.get_token()}"})
        
        # Should not contain script tags in the processed data
        if response.status_code == 200:
            data = response.json()
            assert "<script>" not in data.get("title", "")

    def get_token(self):
        """Helper method to get auth token"""
        login_response = client.post("/auth/login", json={
            "username": "admin",
            "password": "test_password"
        })
        return login_response.json()["access_token"]
