"""
Redis Streams-based message queue implementation
"""
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
import redis.asyncio as redis
from .config import settings

logger = logging.getLogger("osint_api")

class RedisStreamsQueue:
    """Redis Streams-based message queue with consumer groups"""
    
    def __init__(self):
        self.redis: Optional[redis.Redis] = None
        self._connection_pool: Optional[redis.ConnectionPool] = None
        self.consumer_groups: Dict[str, str] = {}
        self.consumers: Dict[str, asyncio.Task] = {}
        self.running = False
    
    async def connect(self):
        """Initialize Redis connection"""
        try:
            self._connection_pool = redis.ConnectionPool.from_url(
                settings.redis_url,
                max_connections=20,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={}
            )
            self.redis = redis.Redis(connection_pool=self._connection_pool)
            
            # Test connection
            await self.redis.ping()
            logger.info("Connected to Redis Streams successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis Streams: {e}")
            self.redis = None
    
    async def disconnect(self):
        """Close Redis connection"""
        self.running = False
        for consumer in self.consumers.values():
            consumer.cancel()
        await asyncio.gather(*self.consumers.values(), return_exceptions=True)
        
        if self.redis:
            await self.redis.close()
        if self._connection_pool:
            await self._connection_pool.disconnect()
        logger.info("Redis Streams disconnected")
    
    async def create_stream(self, stream_name: str, consumer_group: str = None):
        """Create a stream and consumer group"""
        try:
            if not self.redis:
                return False
            
            # Create stream if it doesn't exist
            await self.redis.xgroup_create(stream_name, consumer_group, id='0', mkstream=True)
            self.consumer_groups[stream_name] = consumer_group
            logger.info(f"Created stream {stream_name} with consumer group {consumer_group}")
            return True
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                # Consumer group already exists
                logger.info(f"Consumer group {consumer_group} already exists for stream {stream_name}")
                return True
            else:
                logger.error(f"Error creating stream {stream_name}: {e}")
                return False
        except Exception as e:
            logger.error(f"Error creating stream {stream_name}: {e}")
            return False
    
    async def send_message(self, stream_name: str, message: Dict[str, Any], 
                          message_id: str = "*") -> str:
        """Send a message to a stream"""
        try:
            if not self.redis:
                return None
            
            # Convert message to Redis stream format
            stream_data = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) 
                          for k, v in message.items()}
            
            message_id = await self.redis.xadd(stream_name, stream_data, id=message_id)
            logger.debug(f"Sent message {message_id} to stream {stream_name}")
            return message_id
        except Exception as e:
            logger.error(f"Error sending message to stream {stream_name}: {e}")
            return None
    
    async def read_messages(self, stream_name: str, consumer_name: str, 
                           count: int = 10, block: int = 1000) -> List[Dict[str, Any]]:
        """Read messages from a stream"""
        try:
            if not self.redis or stream_name not in self.consumer_groups:
                return []
            
            consumer_group = self.consumer_groups[stream_name]
            
            # Read messages from consumer group
            messages = await self.redis.xreadgroup(
                consumer_group, consumer_name, 
                {stream_name: '>'}, count=count, block=block
            )
            
            result = []
            for stream, msgs in messages:
                for msg_id, fields in msgs:
                    # Parse message data
                    message_data = {}
                    for key, value in fields.items():
                        try:
                            message_data[key.decode()] = json.loads(value.decode())
                        except (json.JSONDecodeError, AttributeError):
                            message_data[key.decode()] = value.decode()
                    
                    result.append({
                        'id': msg_id.decode(),
                        'data': message_data
                    })
            
            return result
        except Exception as e:
            logger.error(f"Error reading messages from stream {stream_name}: {e}")
            return []
    
    async def acknowledge_message(self, stream_name: str, message_id: str) -> bool:
        """Acknowledge a message as processed"""
        try:
            if not self.redis or stream_name not in self.consumer_groups:
                return False
            
            consumer_group = self.consumer_groups[stream_name]
            await self.redis.xack(stream_name, consumer_group, message_id)
            logger.debug(f"Acknowledged message {message_id} from stream {stream_name}")
            return True
        except Exception as e:
            logger.error(f"Error acknowledging message {message_id}: {e}")
            return False
    
    async def get_pending_messages(self, stream_name: str, consumer_name: str) -> List[Dict[str, Any]]:
        """Get pending messages for a consumer"""
        try:
            if not self.redis or stream_name not in self.consumer_groups:
                return []
            
            consumer_group = self.consumer_groups[stream_name]
            pending = await self.redis.xpending_range(
                stream_name, consumer_group, '-', '+', 100, consumer_name
            )
            
            result = []
            for msg in pending:
                result.append({
                    'id': msg['message_id'].decode(),
                    'consumer': msg['consumer'].decode(),
                    'time_since_delivered': msg['time_since_delivered'],
                    'times_delivered': msg['times_delivered']
                })
            
            return result
        except Exception as e:
            logger.error(f"Error getting pending messages: {e}")
            return []
    
    async def start_consumer(self, stream_name: str, consumer_name: str, 
                           handler: Callable, auto_ack: bool = True):
        """Start a consumer for a stream"""
        try:
            if not self.redis:
                return False
            
            # Create stream and consumer group if they don't exist
            await self.create_stream(stream_name, f"{stream_name}_group")
            
            async def consumer_loop():
                logger.info(f"Starting consumer {consumer_name} for stream {stream_name}")
                while self.running:
                    try:
                        messages = await self.read_messages(stream_name, consumer_name)
                        
                        for message in messages:
                            try:
                                # Process message
                                await handler(message['data'])
                                
                                # Acknowledge message if auto_ack is enabled
                                if auto_ack:
                                    await self.acknowledge_message(stream_name, message['id'])
                                    
                            except Exception as e:
                                logger.error(f"Error processing message {message['id']}: {e}")
                                # Don't acknowledge failed messages
                        
                        # Small delay to prevent busy waiting
                        if not messages:
                            await asyncio.sleep(0.1)
                            
                    except Exception as e:
                        logger.error(f"Error in consumer loop: {e}")
                        await asyncio.sleep(1)
                
                logger.info(f"Consumer {consumer_name} stopped")
            
            # Start consumer task
            self.consumers[f"{stream_name}_{consumer_name}"] = asyncio.create_task(consumer_loop())
            return True
            
        except Exception as e:
            logger.error(f"Error starting consumer {consumer_name}: {e}")
            return False
    
    async def stop_consumer(self, stream_name: str, consumer_name: str):
        """Stop a consumer"""
        consumer_key = f"{stream_name}_{consumer_name}"
        if consumer_key in self.consumers:
            self.consumers[consumer_key].cancel()
            del self.consumers[consumer_key]
            logger.info(f"Stopped consumer {consumer_name} for stream {stream_name}")
    
    async def get_stream_info(self, stream_name: str) -> Dict[str, Any]:
        """Get stream information"""
        try:
            if not self.redis:
                return {}
            
            info = await self.redis.xinfo_stream(stream_name)
            return {
                'length': info['length'],
                'radix_tree_keys': info['radix-tree-keys'],
                'radix_tree_nodes': info['radix-tree-nodes'],
                'groups': info['groups'],
                'last_generated_id': info['last-generated-id'],
                'first_entry': info.get('first-entry'),
                'last_entry': info.get('last-entry')
            }
        except Exception as e:
            logger.error(f"Error getting stream info for {stream_name}: {e}")
            return {}
    
    async def cleanup_old_messages(self, stream_name: str, max_length: int = 10000):
        """Clean up old messages from stream"""
        try:
            if not self.redis:
                return False
            
            # Trim stream to max_length
            await self.redis.xtrim(stream_name, maxlen=max_length, approximate=True)
            logger.info(f"Trimmed stream {stream_name} to max {max_length} messages")
            return True
        except Exception as e:
            logger.error(f"Error cleaning up stream {stream_name}: {e}")
            return False

# Global Redis Streams queue instance
redis_streams = RedisStreamsQueue()

# Stream names
EMBEDDING_STREAM = "embedding_tasks"
PROCESSING_STREAM = "processing_tasks"
CLEANUP_STREAM = "cleanup_tasks"
NOTIFICATION_STREAM = "notifications"
