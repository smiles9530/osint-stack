"""
Redis caching module for the OSINT API
"""
import json
import logging
import asyncio
from typing import Any, Optional, Union, List, Dict
import redis.asyncio as redis
from .config import settings

logger = logging.getLogger("osint_api")

class CacheManager:
    """Redis cache manager with connection pooling"""
    
    def __init__(self):
        self.redis: Optional[redis.Redis] = None
        self._connection_pool: Optional[redis.ConnectionPool] = None
    
    async def connect(self):
        """Initialize Redis connection with optimized connection pooling for high load"""
        try:
            self._connection_pool = redis.ConnectionPool.from_url(
                settings.redis_url,
                max_connections=100,  # Increased for high load
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={
                    1: 1,  # TCP_KEEPIDLE
                    2: 3,  # TCP_KEEPINTVL
                    3: 5,  # TCP_KEEPCNT
                },
                socket_connect_timeout=5,
                socket_timeout=5,
                health_check_interval=30,
                # Connection pool optimizations
                decode_responses=True,
                encoding='utf-8',
                # High-load optimizations
                retry_on_error=[redis.ConnectionError, redis.TimeoutError],
                # Connection management
                single_connection_client=False,
                # Performance tuning
                socket_keepalive_idle=600,
                socket_keepalive_interval=30,
                socket_keepalive_count=3
            )
            self.redis = redis.Redis(connection_pool=self._connection_pool)
            
            # Test connection and configure Redis for high load
            await self.redis.ping()
            
            # Configure Redis for optimal performance
            await self._configure_redis_performance()
            
            logger.info("Connected to Redis with high-load optimizations")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis = None
    
    async def _configure_redis_performance(self):
        """Configure Redis for high-load performance"""
        try:
            # Set Redis configuration for high load
            config_commands = [
                # Memory optimization
                ("CONFIG", "SET", "maxmemory-policy", "allkeys-lru"),
                ("CONFIG", "SET", "maxmemory-samples", "10"),
                
                # Network optimization
                ("CONFIG", "SET", "tcp-keepalive", "60"),
                ("CONFIG", "SET", "timeout", "300"),
                
                # Performance tuning
                ("CONFIG", "SET", "hz", "10"),  # Lower frequency for better performance
                ("CONFIG", "SET", "dynamic-hz", "yes"),
                
                # Persistence optimization for cache
                ("CONFIG", "SET", "save", ""),  # Disable RDB saves for cache
                ("CONFIG", "SET", "appendonly", "no"),  # Disable AOF for cache
                
                # Client optimization
                ("CONFIG", "SET", "client-output-buffer-limit", "normal 0 0 0"),
                ("CONFIG", "SET", "client-output-buffer-limit", "replica 256mb 64mb 60"),
                ("CONFIG", "SET", "client-output-buffer-limit", "pubsub 32mb 8mb 60"),
            ]
            
            for command in config_commands:
                try:
                    await self.redis.execute_command(*command)
                except Exception as e:
                    logger.warning(f"Failed to set Redis config {command[1]}: {e}")
            
            logger.info("Redis configured for high-load performance")
        except Exception as e:
            logger.warning(f"Failed to configure Redis performance settings: {e}")
    
    async def disconnect(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
        if self._connection_pool:
            await self._connection_pool.disconnect()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.redis:
            return None
        
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, expire: int = 3600) -> bool:
        """Set value in cache with expiration"""
        if not self.redis:
            return False
        
        try:
            serialized = json.dumps(value, default=str)
            await self.redis.setex(key, expire, serialized)
            return True
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.redis:
            return False
        
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern"""
        if not self.redis:
            return 0
        
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                return await self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache delete pattern error for {pattern}: {e}")
            return 0
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.redis:
            return False
        
        try:
            return await self.redis.exists(key) > 0
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def get_or_set(self, key: str, func, expire: int = 3600) -> Any:
        """Get value from cache or compute and cache it"""
        cached_value = await self.get(key)
        if cached_value is not None:
            return cached_value
        
        # Compute value
        value = await func() if callable(func) else func
        
        # Cache it
        await self.set(key, value, expire)
        return value
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache in a single operation"""
        if not self.redis or not keys:
            return {}
        
        try:
            values = await self.redis.mget(keys)
            return {key: json.loads(value) for key, value in zip(keys, values) if value}
        except Exception as e:
            logger.error(f"Cache get_many error for keys {keys}: {e}")
            return {}
    
    async def set_many(self, mapping: Dict[str, Any], expire: int = 3600) -> bool:
        """Set multiple values in cache in a single operation"""
        if not self.redis or not mapping:
            return False
        
        try:
            # Serialize all values
            serialized = {key: json.dumps(value, default=str) for key, value in mapping.items()}
            
            # Use pipeline for atomic operation
            pipe = self.redis.pipeline()
            for key, value in serialized.items():
                pipe.setex(key, expire, value)
            await pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Cache set_many error: {e}")
            return False
    
    async def delete_many(self, keys: List[str]) -> int:
        """Delete multiple keys in a single operation"""
        if not self.redis or not keys:
            return 0
        
        try:
            return await self.redis.delete(*keys)
        except Exception as e:
            logger.error(f"Cache delete_many error for keys {keys}: {e}")
            return 0
    
    async def increment(self, key: str, amount: int = 1, expire: int = 3600) -> int:
        """Increment a counter in cache"""
        if not self.redis:
            return 0
        
        try:
            result = await self.redis.incrby(key, amount)
            if result == amount:  # First time setting this key
                await self.redis.expire(key, expire)
            return result
        except Exception as e:
            logger.error(f"Cache increment error for key {key}: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.redis:
            return {}
        
        try:
            info = await self.redis.info()
            return {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'hit_rate': (
                    info.get('keyspace_hits', 0) / 
                    (info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0))
                    if (info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0)) > 0 else 0
                ),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'instantaneous_ops_per_sec': info.get('instantaneous_ops_per_sec', 0)
            }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {}

# Global cache instance
cache = CacheManager()

# Cache key generators
def article_cache_key(article_id: int) -> str:
    """Generate cache key for article"""
    return f"article:{article_id}"

def articles_list_cache_key(limit: int, offset: int) -> str:
    """Generate cache key for articles list"""
    return f"articles:list:{limit}:{offset}"

def embedding_cache_key(article_id: int, model: str) -> str:
    """Generate cache key for embedding"""
    return f"embedding:{article_id}:{model}"

def source_cache_key(source_name: str) -> str:
    """Generate cache key for source"""
    return f"source:{source_name}"

# Cache decorators
def cache_result(expire: int = 3600, key_func=None):
    """Decorator to cache function results"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, expire)
            return result
        
        return wrapper
    return decorator
