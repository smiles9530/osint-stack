import httpx
from typing import List
import logging
from .config import settings

logger = logging.getLogger("osint_api")

_local_model = None

def _load_local_model():
    global _local_model
    if _local_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            from .gpu_utils import gpu_manager, model_device_manager
            
            model_name = settings.local_embed_model
            device = model_device_manager.get_model_device(model_name)
            
            _local_model = SentenceTransformer(model_name)
            
            # Move to appropriate device
            if gpu_manager.is_gpu_available():
                _local_model = _local_model.to(device)
                logger.info(f"Loaded local embedding model: {model_name} on {device}")
            else:
                logger.info(f"Loaded local embedding model: {model_name} on CPU")
                
        except ImportError:
            logger.warning("sentence-transformers not available, using simple hash-based embeddings")
            _local_model = "simple_hash"
    return _local_model

def _simple_hash_embedding(text: str) -> List[float]:
    """Simple hash-based embedding for fallback when sentence-transformers is not available"""
    import hashlib
    import math
    
    # Create a hash of the text
    text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    # Convert hash to a 1024-dimensional vector
    vector = []
    for i in range(0, len(text_hash), 2):
        # Take pairs of hex characters and convert to float
        hex_pair = text_hash[i:i+2]
        value = int(hex_pair, 16) / 255.0  # Normalize to 0-1
        vector.append(value)
    
    # Pad or truncate to exactly 1024 dimensions
    while len(vector) < 1024:
        vector.append(0.0)
    
    return vector[:1024]

async def embed_texts(texts: list[str]) -> List[List[float]]:
    backend = settings.embeddings_backend.lower()
    
    if backend == "ollama":
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{settings.ollama_host}/api/embeddings",
                    json={"model": settings.ollama_embed_model, "input": texts},
                )
                resp.raise_for_status()
                data = resp.json()
                return data["embeddings"]
        except Exception as e:
            logger.error(f"Ollama embedding failed: {e}")
            # Fallback to simple hash embeddings
            return [_simple_hash_embedding(text) for text in texts]
    elif backend == "simple":
        # Use simple hash-based embeddings
        return [_simple_hash_embedding(text) for text in texts]
    else:
        model = _load_local_model()
        if model == "simple_hash":
            # Use simple hash-based embeddings
            return [_simple_hash_embedding(text) for text in texts]
        else:
            try:
                return model.encode(texts, normalize_embeddings=True).tolist()
            except Exception as e:
                logger.error(f"Local model embedding failed: {e}")
                # Fallback to simple hash embeddings
                return [_simple_hash_embedding(text) for text in texts]
