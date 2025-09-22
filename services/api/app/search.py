"""
Typesense search integration for OSINT stack
"""
import typesense
from typing import List, Dict, Any, Optional
from .config import settings
from .logging_config import logger

class TypesenseClient:
    def __init__(self):
        # Parse the URL properly
        url_parts = settings.typesense_url.replace('http://', '').replace('https://', '')
        if ':' in url_parts:
            host, port = url_parts.split(':')
        else:
            host = url_parts
            port = '8108'
            
        self.client = typesense.Client({
            'nodes': [{
                'host': host,
                'port': port,
                'protocol': 'http'
            }],
            'api_key': settings.typesense_api_key or 'osint-search-key',
            'connection_timeout_seconds': 2
        })
        self.collection_name = 'articles'
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Create the articles collection if it doesn't exist"""
        try:
            # Check if collection exists
            self.client.collections[self.collection_name].retrieve()
        except typesense.exceptions.ObjectNotFound:
            # Create collection
            schema = {
                'name': self.collection_name,
                'fields': [
                    {'name': 'id', 'type': 'int32'},
                    {'name': 'title', 'type': 'string'},
                    {'name': 'content', 'type': 'string'},
                    {'name': 'url', 'type': 'string'},
                    {'name': 'source_name', 'type': 'string'},
                    {'name': 'lang', 'type': 'string'},
                    {'name': 'published_at', 'type': 'int64'},
                    {'name': 'fetched_at', 'type': 'int64'},
                    {'name': 'tags', 'type': 'string[]', 'optional': True},
                    {'name': 'entities', 'type': 'string[]', 'optional': True}
                ],
                'default_sorting_field': 'fetched_at'
            }
            self.client.collections.create(schema)
            logger.info(f"Created Typesense collection: {self.collection_name}")
    
    def index_article(self, article: Dict[str, Any]) -> bool:
        """Index an article in Typesense"""
        try:
            # Convert datetime to timestamp
            document = {
                'id': article['id'],
                'title': article.get('title', ''),
                'content': article.get('text', ''),
                'url': article.get('url', ''),
                'source_name': article.get('source_name', ''),
                'lang': article.get('lang', 'en'),
                'published_at': int(article.get('published_at', 0)) if article.get('published_at') else 0,
                'fetched_at': int(article.get('fetched_at', 0)) if article.get('fetched_at') else 0,
                'tags': article.get('tags', []),
                'entities': article.get('entities', [])
            }
            
            self.client.collections[self.collection_name].documents.create(document)
            return True
        except Exception as e:
            logger.error(f"Failed to index article {article.get('id')}: {e}")
            return False
    
    def search_articles(self, query: str, limit: int = 10, offset: int = 0, 
                       filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Search articles using Typesense"""
        try:
            search_params = {
                'q': query,
                'query_by': 'title,content',
                'sort_by': 'fetched_at:desc',
                'per_page': limit,
                'page': (offset // limit) + 1,
                'highlight_full_fields': 'title,content',
                'snippet_threshold': 30,
                'num_typos': 2
            }
            
            if filters:
                filter_conditions = []
                if filters.get('source_name'):
                    filter_conditions.append(f"source_name:={filters['source_name']}")
                if filters.get('lang'):
                    filter_conditions.append(f"lang:={filters['lang']}")
                if filters.get('date_from'):
                    filter_conditions.append(f"published_at:>={filters['date_from']}")
                if filters.get('date_to'):
                    filter_conditions.append(f"published_at:<={filters['date_to']}")
                
                if filter_conditions:
                    search_params['filter_by'] = ' && '.join(filter_conditions)
            
            result = self.client.collections[self.collection_name].documents.search(search_params)
            return result
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {'hits': [], 'found': 0, 'page': 1}
    
    def delete_article(self, article_id: int) -> bool:
        """Delete an article from Typesense"""
        try:
            self.client.collections[self.collection_name].documents[str(article_id)].delete()
            return True
        except Exception as e:
            logger.error(f"Failed to delete article {article_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            collection = self.client.collections[self.collection_name].retrieve()
            return {
                'total_documents': collection.get('num_documents', 0),
                'collection_name': collection.get('name', ''),
                'created_at': collection.get('created_at', 0)
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'total_documents': 0, 'collection_name': '', 'created_at': 0}

# Global instance
search_client = TypesenseClient()
