"""
UUID Validation Fix for OSINT Stack
Handles cases where non-UUID values (like Reddit post IDs) are passed to UUID fields
"""

import re
import uuid
from typing import Any, Union, Optional
import logging

logger = logging.getLogger(__name__)

def is_valid_uuid(value: Any) -> bool:
    """Check if a value is a valid UUID"""
    if not isinstance(value, str):
        return False
    
    try:
        uuid.UUID(value)
        return True
    except (ValueError, TypeError):
        return False

def is_reddit_post_id(value: str) -> bool:
    """Check if a value is a Reddit post ID (e.g., t3_1nm0pvw)"""
    if not isinstance(value, str):
        return False
    
    # Reddit post IDs start with t3_ followed by alphanumeric characters
    return bool(re.match(r'^t3_[a-zA-Z0-9]+$', value))

def is_reddit_comment_id(value: str) -> bool:
    """Check if a value is a Reddit comment ID (e.g., t1_1nm0pvw)"""
    if not isinstance(value, str):
        return False
    
    # Reddit comment IDs start with t1_ followed by alphanumeric characters
    return bool(re.match(r'^t1_[a-zA-Z0-9]+$', value))

def is_reddit_user_id(value: str) -> bool:
    """Check if a value is a Reddit user ID (e.g., t2_1nm0pvw)"""
    if not isinstance(value, str):
        return False
    
    # Reddit user IDs start with t2_ followed by alphanumeric characters
    return bool(re.match(r'^t2_[a-zA-Z0-9]+$', value))

def is_reddit_id(value: str) -> bool:
    """Check if a value is any Reddit ID"""
    return is_reddit_post_id(value) or is_reddit_comment_id(value) or is_reddit_user_id(value)

def generate_uuid_from_reddit_id(reddit_id: str) -> str:
    """Generate a deterministic UUID from a Reddit ID"""
    if not is_reddit_id(reddit_id):
        raise ValueError(f"Invalid Reddit ID: {reddit_id}")
    
    # Create a deterministic UUID using the Reddit ID as a seed
    # This ensures the same Reddit ID always generates the same UUID
    namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # DNS namespace
    return str(uuid.uuid5(namespace, reddit_id))

def safe_uuid_conversion(value: Any) -> str:
    """
    Safely convert any value to a UUID string.
    If it's already a valid UUID, return it.
    If it's a Reddit ID, generate a deterministic UUID.
    Otherwise, generate a new UUID.
    """
    if value is None:
        return str(uuid.uuid4())
    
    if isinstance(value, uuid.UUID):
        return str(value)
    
    if isinstance(value, str):
        if is_valid_uuid(value):
            return value
        elif is_reddit_id(value):
            logger.info(f"Converting Reddit ID {value} to UUID")
            return generate_uuid_from_reddit_id(value)
        else:
            logger.warning(f"Unknown ID format: {value}, generating new UUID")
            return str(uuid.uuid4())
    
    # For any other type, generate a new UUID
    logger.warning(f"Converting {type(value)} to UUID: {value}")
    return str(uuid.uuid4())

def validate_and_fix_uuid_field(data: dict, field_name: str) -> dict:
    """
    Validate and fix a UUID field in a dictionary.
    Returns the data with the field converted to a valid UUID if needed.
    """
    if field_name not in data:
        return data
    
    original_value = data[field_name]
    fixed_value = safe_uuid_conversion(original_value)
    
    if original_value != fixed_value:
        logger.info(f"Fixed UUID field {field_name}: {original_value} -> {fixed_value}")
        data[field_name] = fixed_value
    
    return data

def process_article_data(article_data: dict) -> dict:
    """
    Process article data to ensure all UUID fields are valid.
    This is specifically for handling Reddit RSS feeds and other sources
    that might have non-UUID identifiers.
    """
    # Create a copy to avoid modifying the original
    processed_data = article_data.copy()
    
    # Common UUID fields that might need fixing
    uuid_fields = ['id', 'article_id', 'parent_id', 'source_id']
    
    for field in uuid_fields:
        if field in processed_data:
            processed_data = validate_and_fix_uuid_field(processed_data, field)
    
    # Handle nested data structures
    if 'metadata' in processed_data and isinstance(processed_data['metadata'], dict):
        for key, value in processed_data['metadata'].items():
            if 'id' in key.lower() and isinstance(value, str):
                if not is_valid_uuid(value) and is_reddit_id(value):
                    processed_data['metadata'][key] = safe_uuid_conversion(value)
    
    return processed_data

def create_reddit_id_mapping(reddit_id: str) -> dict:
    """
    Create a mapping for Reddit IDs to help with debugging and tracking.
    This can be stored in metadata to maintain the original Reddit ID.
    """
    if not is_reddit_id(reddit_id):
        return {}
    
    return {
        'original_reddit_id': reddit_id,
        'reddit_type': 'post' if is_reddit_post_id(reddit_id) else 
                      'comment' if is_reddit_comment_id(reddit_id) else 'user',
        'generated_uuid': generate_uuid_from_reddit_id(reddit_id)
    }

# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_cases = [
        "t3_1nm0pvw",  # Reddit post ID from the error
        "t1_abc123",    # Reddit comment ID
        "t2_def456",    # Reddit user ID
        "550e8400-e29b-41d4-a716-446655440000",  # Valid UUID
        "invalid-id",   # Invalid ID
        None,           # None value
        12345,          # Integer
    ]
    
    print("UUID Validation Fix Test Results:")
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
