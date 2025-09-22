"""
Enhanced Entity Extraction and Linking Service
Provides NER, entity linking, and knowledge graph capabilities
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import re
import json
from datetime import datetime
import psycopg2.extras
from contextlib import contextmanager

from .config import settings
from .db import get_conn

logger = logging.getLogger("osint_api")

class EntityExtractionService:
    """Enhanced entity extraction and linking service"""
    
    def __init__(self):
        self.nlp = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NER models with GPU acceleration"""
        try:
            import spacy
            from .gpu_utils import gpu_manager, model_device_manager
            
            # Load multilingual model for better entity recognition
            model_name = "xx_ent_wiki_sm"
            device = model_device_manager.get_model_device(model_name)
            
            self.nlp = spacy.load(model_name)
            
            # Configure spaCy for GPU if available
            if gpu_manager.is_gpu_available():
                try:
                    # Try to use GPU for spaCy processing
                    spacy.prefer_gpu()
                    logger.info(f"Loaded multilingual NER model: {model_name} with GPU acceleration")
                except Exception as e:
                    logger.warning(f"GPU acceleration not available for spaCy: {e}")
                    logger.info(f"Loaded multilingual NER model: {model_name} on CPU")
            else:
                logger.info(f"Loaded multilingual NER model: {model_name} on CPU")
                
        except OSError:
            try:
                import spacy
                from .gpu_utils import gpu_manager, model_device_manager
                
                # English fallback
                model_name = "en_core_web_sm"
                device = model_device_manager.get_model_device(model_name)
                
                self.nlp = spacy.load(model_name)
                
                # Configure spaCy for GPU if available
                if gpu_manager.is_gpu_available():
                    try:
                        spacy.prefer_gpu()
                        logger.info(f"Loaded English NER model: {model_name} with GPU acceleration")
                    except Exception as e:
                        logger.warning(f"GPU acceleration not available for spaCy: {e}")
                        logger.info(f"Loaded English NER model: {model_name} on CPU")
                else:
                    logger.info(f"Loaded English NER model: {model_name} on CPU")
                    
            except OSError:
                logger.warning("spaCy models not available, using basic NER")
                self.nlp = None
    
    async def extract_entities(
        self, 
        text: str, 
        article_id: Optional[int] = None,
        include_custom: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from text using NER and custom patterns
        
        Args:
            text: Input text to analyze
            article_id: Optional article ID for storing results
            include_custom: Whether to include custom entity patterns
        
        Returns:
            List of extracted entities with metadata
        """
        try:
            entities = []
            
            # Use spaCy NER if available
            if self.nlp:
                doc = self.nlp(text)
                for ent in doc.ents:
                    entity = {
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': getattr(ent, 'confidence', 0.8),
                        'description': self._get_entity_description(ent.label_),
                        'source': 'spacy'
                    }
                    entities.append(entity)
            
            # Add custom entity patterns
            if include_custom:
                custom_entities = await self._extract_custom_entities(text)
                entities.extend(custom_entities)
            
            # Remove duplicates and sort by confidence
            entities = self._deduplicate_entities(entities)
            entities.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Store entities in database if article_id provided
            if article_id:
                await self._store_entities(article_id, entities)
            
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    async def _extract_custom_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract custom entities using regex patterns"""
        entities = []
        
        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            entities.append({
                'text': match.group(),
                'label': 'EMAIL',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.9,
                'description': 'Email address',
                'source': 'regex'
            })
        
        # URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        for match in re.finditer(url_pattern, text):
            entities.append({
                'text': match.group(),
                'label': 'URL',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.9,
                'description': 'Web URL',
                'source': 'regex'
            })
        
        # Phone numbers
        phone_pattern = r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'
        for match in re.finditer(phone_pattern, text):
            entities.append({
                'text': match.group(),
                'label': 'PHONE',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.8,
                'description': 'Phone number',
                'source': 'regex'
            })
        
        # Social media handles
        social_pattern = r'@[A-Za-z0-9_]+|#[A-Za-z0-9_]+'
        for match in re.finditer(social_pattern, text):
            entities.append({
                'text': match.group(),
                'label': 'SOCIAL_MEDIA',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.8,
                'description': 'Social media handle or hashtag',
                'source': 'regex'
            })
        
        # IP addresses
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        for match in re.finditer(ip_pattern, text):
            entities.append({
                'text': match.group(),
                'label': 'IP_ADDRESS',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.7,
                'description': 'IP address',
                'source': 'regex'
            })
        
        return entities
    
    async def link_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Link entities to external knowledge bases (Wikidata, etc.)
        
        Args:
            entities: List of extracted entities
        
        Returns:
            Entities with linking information
        """
        try:
            linked_entities = []
            
            for entity in entities:
                # Try to link to Wikidata (simplified approach)
                wikidata_info = await self._link_to_wikidata(entity)
                if wikidata_info:
                    entity.update(wikidata_info)
                
                linked_entities.append(entity)
            
            return linked_entities
            
        except Exception as e:
            logger.error(f"Entity linking failed: {e}")
            return entities
    
    async def _link_to_wikidata(self, entity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Link entity to Wikidata (simplified implementation)"""
        try:
            # This is a simplified implementation
            # In production, you'd use proper Wikidata API or entity linking services
            
            entity_text = entity['text']
            entity_type = entity['label']
            
            # Simple mapping for common entities
            if entity_type == 'PERSON':
                # Look for person in local knowledge base
                person_info = await self._get_person_info(entity_text)
                if person_info:
                    return {
                        'wikidata_id': person_info.get('wikidata_id'),
                        'description': person_info.get('description'),
                        'aliases': person_info.get('aliases', []),
                        'linked': True
                    }
            
            elif entity_type == 'ORG':
                # Look for organization in local knowledge base
                org_info = await self._get_organization_info(entity_text)
                if org_info:
                    return {
                        'wikidata_id': org_info.get('wikidata_id'),
                        'description': org_info.get('description'),
                        'aliases': org_info.get('aliases', []),
                        'linked': True
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Wikidata linking failed for {entity['text']}: {e}")
            return None
    
    async def _get_person_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get person information from local knowledge base"""
        try:
            with get_conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("""
                        SELECT wikidata_id, description, aliases
                        FROM entity_nodes
                        WHERE name ILIKE %s AND type = 'PERSON'
                        LIMIT 1
                    """, (f"%{name}%",))
                    result = cur.fetchone()
                    return dict(result) if result else None
        except Exception as e:
            logger.error(f"Failed to get person info: {e}")
            return None
    
    async def _get_organization_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get organization information from local knowledge base"""
        try:
            with get_conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("""
                        SELECT wikidata_id, description, aliases
                        FROM entity_nodes
                        WHERE name ILIKE %s AND type = 'ORG'
                        LIMIT 1
                    """, (f"%{name}%",))
                    result = cur.fetchone()
                    return dict(result) if result else None
        except Exception as e:
            logger.error(f"Failed to get organization info: {e}")
            return None
    
    async def _store_entities(self, article_id: int, entities: List[Dict[str, Any]]):
        """Store extracted entities in database"""
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    # Clear existing entities for this article
                    cur.execute("DELETE FROM article_entities WHERE article_id = %s", (article_id,))
                    
                    # Insert new entities
                    for entity in entities:
                        cur.execute("""
                            INSERT INTO article_entities 
                            (article_id, entity_type, entity_name, confidence, start_pos, end_pos, context)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (article_id, entity_type, entity_name, start_pos) 
                            DO UPDATE SET confidence = EXCLUDED.confidence
                        """, (
                            article_id,
                            entity['label'],
                            entity['text'],
                            entity['confidence'],
                            entity['start'],
                            entity['end'],
                            json.dumps(entity)
                        ))
                    
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Failed to store entities: {e}")
    
    def _get_entity_description(self, label: str) -> str:
        """Get description for entity label"""
        descriptions = {
            'PERSON': 'Person name',
            'ORG': 'Organization',
            'GPE': 'Geopolitical entity',
            'LOC': 'Location',
            'EVENT': 'Event',
            'WORK_OF_ART': 'Work of art',
            'LAW': 'Legal document',
            'LANGUAGE': 'Language',
            'DATE': 'Date',
            'TIME': 'Time',
            'MONEY': 'Monetary value',
            'PERCENT': 'Percentage',
            'CARDINAL': 'Cardinal number',
            'ORDINAL': 'Ordinal number',
            'QUANTITY': 'Quantity',
            'NORP': 'Nationality or religious group',
            'FAC': 'Facility',
            'PRODUCT': 'Product',
            'EMAIL': 'Email address',
            'URL': 'Web URL',
            'PHONE': 'Phone number',
            'SOCIAL_MEDIA': 'Social media handle',
            'IP_ADDRESS': 'IP address'
        }
        return descriptions.get(label, 'Unknown entity')
    
    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entities"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity['text'].lower(), entity['start'], entity['end'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    async def get_entity_statistics(self) -> Dict[str, Any]:
        """Get statistics about extracted entities"""
        try:
            with get_conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    # Get entity type counts
                    cur.execute("""
                        SELECT entity_type, COUNT(*) as count
                        FROM article_entities
                        GROUP BY entity_type
                        ORDER BY count DESC
                    """)
                    type_counts = cur.fetchall()
                    
                    # Get total entities
                    cur.execute("SELECT COUNT(*) as total FROM article_entities")
                    total_entities = cur.fetchone()['total']
                    
                    # Get articles with entities
                    cur.execute("""
                        SELECT COUNT(DISTINCT article_id) as articles_with_entities
                        FROM article_entities
                    """)
                    articles_with_entities = cur.fetchone()['articles_with_entities']
                    
                    return {
                        'total_entities': total_entities,
                        'articles_with_entities': articles_with_entities,
                        'entity_types': [dict(row) for row in type_counts],
                        'generated_at': datetime.now().isoformat()
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get entity statistics: {e}")
            return {
                'total_entities': 0,
                'articles_with_entities': 0,
                'entity_types': [],
                'error': str(e)
            }

# Global instance
entity_extraction_service = EntityExtractionService()
