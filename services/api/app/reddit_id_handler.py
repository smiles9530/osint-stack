"""
Reddit ID Handler for OSINT Stack
Handles Reddit IDs (t3_*, t1_*, t2_*) and converts them to valid UUIDs
"""

import re
import uuid
import logging
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

def is_reddit_id(value: str) -> bool:
    """Check if a value is a Reddit ID"""
    if not isinstance(value, str):
        return False
    return bool(re.match(r'^t[123]_[a-zA-Z0-9]+$', value))

def reddit_id_to_uuid(reddit_id: str) -> str:
    """Convert Reddit ID to deterministic UUID"""
    if not is_reddit_id(reddit_id):
        raise ValueError(f"Invalid Reddit ID: {reddit_id}")
    
    # Use DNS namespace for deterministic UUID generation
    namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
    return str(uuid.uuid5(namespace, reddit_id))

def safe_uuid_conversion(value: Any) -> str:
    """Safely convert any value to a UUID string"""
    if value is None:
        return str(uuid.uuid4())
    
    if isinstance(value, uuid.UUID):
        return str(value)
    
    if isinstance(value, str):
        # Check if it's already a valid UUID
        try:
            uuid.UUID(value)
            return value
        except (ValueError, TypeError):
            pass
        
        # Check if it's a Reddit ID
        if is_reddit_id(value):
            logger.info(f"Converting Reddit ID {value} to UUID")
            return reddit_id_to_uuid(value)
        
        # Generate new UUID for other values
        logger.warning(f"Converting unknown ID format to UUID: {value}")
        return str(uuid.uuid4())
    
    # For any other type, generate a new UUID
    logger.warning(f"Converting {type(value)} to UUID: {value}")
    return str(uuid.uuid4())

def process_article_data_for_reddit(article_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process article data to handle Reddit IDs"""
    processed_data = article_data.copy()
    
    # Handle main ID field
    if 'id' in processed_data:
        original_id = processed_data['id']
        processed_data['id'] = safe_uuid_conversion(original_id)
        
        # If it was a Reddit ID, add mapping to metadata
        if is_reddit_id(original_id):
            reddit_mapping = {
                'original_reddit_id': original_id,
                'reddit_type': 'post' if original_id.startswith('t3_') else 
                              'comment' if original_id.startswith('t1_') else 'user',
                'mapped_at': str(uuid.uuid4())  # Use UUID as timestamp placeholder
            }
            
            if 'metadata' not in processed_data:
                processed_data['metadata'] = {}
            elif not isinstance(processed_data['metadata'], dict):
                processed_data['metadata'] = {}
            
            processed_data['metadata']['reddit_id_mapping'] = reddit_mapping
    
    return processed_data

def create_reddit_id_mapping(reddit_id: str) -> Dict[str, Any]:
    """Create a mapping for Reddit IDs"""
    if not is_reddit_id(reddit_id):
        return {}
    
    return {
        'original_reddit_id': reddit_id,
        'reddit_type': 'post' if reddit_id.startswith('t3_') else 
                      'comment' if reddit_id.startswith('t1_') else 'user',
        'generated_uuid': reddit_id_to_uuid(reddit_id)
    }

# Test function
def test_reddit_id_handling():
    """Test Reddit ID handling functions"""
    test_cases = [
        "t3_1nm0pvw",  # Reddit post ID from the error
        "t1_abc123",    # Reddit comment ID
        "t2_def456",    # Reddit user ID
        "550e8400-e29b-41d4-a716-446655440000",  # Valid UUID
        "invalid-id",   # Invalid ID
        None,           # None value
        12345,          # Integer
    ]
    
    print("Reddit ID Handling Test Results:")
    print("=" * 50)
    
    for test_value in test_cases:
        try:
            result = safe_uuid_conversion(test_value)
            print(f"Input: {test_value} -> Output: {result}")
        except Exception as e:
            print(f"Input: {test_value} -> Error: {e}")
    
    print("\nReddit ID Mapping Test:")
    print("=" * 30)
    
    reddit_ids = ["t3_1nm0pvw", "t1_abc123", "t2_def456"]
    for reddit_id in reddit_ids:
        mapping = create_reddit_id_mapping(reddit_id)
        print(f"Reddit ID: {reddit_id}")
        print(f"Mapping: {mapping}")
        print()

if __name__ == "__main__":
    test_reddit_id_handling()
