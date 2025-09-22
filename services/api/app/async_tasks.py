"""
Redis Streams-based asynchronous task processing module
"""
import asyncio
import logging
from typing import Callable, Any, Dict, Optional
from .redis_streams import redis_streams, EMBEDDING_STREAM, PROCESSING_STREAM, CLEANUP_STREAM, NOTIFICATION_STREAM

logger = logging.getLogger("osint_api")

class AsyncTaskManager:
    """Redis Streams-based task manager"""
    
    def __init__(self):
        self.running = False
        self.consumers = {}
    
    async def start(self):
        """Start the task processing system"""
        try:
            # Connect to Redis Streams
            await redis_streams.connect()
            
            # Create streams
            await redis_streams.create_stream(EMBEDDING_STREAM, f"{EMBEDDING_STREAM}_group")
            await redis_streams.create_stream(PROCESSING_STREAM, f"{PROCESSING_STREAM}_group")
            await redis_streams.create_stream(CLEANUP_STREAM, f"{CLEANUP_STREAM}_group")
            await redis_streams.create_stream(NOTIFICATION_STREAM, f"{NOTIFICATION_STREAM}_group")
            
            # Start consumers
            await self._start_consumers()
            
            self.running = True
            logger.info("Redis Streams task processing started")
        except Exception as e:
            logger.error(f"Failed to start task processing: {e}")
            self.running = False
    
    async def stop(self):
        """Stop the task processing system"""
        self.running = False
        
        # Stop all consumers
        for stream_name, consumer_name in self.consumers.items():
            await redis_streams.stop_consumer(stream_name, consumer_name)
        
        # Disconnect from Redis
        await redis_streams.disconnect()
        logger.info("Redis Streams task processing stopped")
    
    async def _start_consumers(self):
        """Start all task consumers"""
        # Embedding task consumer
        await redis_streams.start_consumer(
            EMBEDDING_STREAM, 
            "embedding_worker_1",
            self._handle_embedding_task
        )
        
        # Processing task consumer
        await redis_streams.start_consumer(
            PROCESSING_STREAM,
            "processing_worker_1", 
            self._handle_processing_task
        )
        
        # Cleanup task consumer
        await redis_streams.start_consumer(
            CLEANUP_STREAM,
            "cleanup_worker_1",
            self._handle_cleanup_task
        )
        
        # Notification consumer
        await redis_streams.start_consumer(
            NOTIFICATION_STREAM,
            "notification_worker_1",
            self._handle_notification_task
        )
    
    async def _handle_embedding_task(self, task_data: Dict[str, Any]):
        """Handle embedding generation tasks"""
        try:
            article_id = task_data.get('article_id')
            text = task_data.get('text')
            model = task_data.get('model', 'simple')
            
            logger.info(f"Processing embedding task for article {article_id}")
            
            # Simulate embedding generation
            await asyncio.sleep(2)
            
            # Here you would call the actual embedding generation
            # from .embedding import generate_embedding
            # embedding = await generate_embedding(text, model)
            
            logger.info(f"Embedding task completed for article {article_id}")
            
            # Send notification
            await self.send_notification({
                'type': 'embedding_completed',
                'article_id': article_id,
                'status': 'success'
            })
            
        except Exception as e:
            logger.error(f"Error processing embedding task: {e}")
            await self.send_notification({
                'type': 'embedding_failed',
                'article_id': task_data.get('article_id'),
                'error': str(e)
            })
    
    async def _handle_processing_task(self, task_data: Dict[str, Any]):
        """Handle article processing tasks"""
        try:
            article_id = task_data.get('article_id')
            url = task_data.get('url')
            title = task_data.get('title')
            text = task_data.get('text')
            
            logger.info(f"Processing article task for {url}")
            
            # Simulate article processing
            await asyncio.sleep(5)
            
            # Here you would call the actual article processing
            # from .db import upsert_article
            # await upsert_article(url, title, text, ...)
            
            logger.info(f"Article processing completed for {url}")
            
            # Send notification
            await self.send_notification({
                'type': 'processing_completed',
                'article_id': article_id,
                'url': url,
                'status': 'success'
            })
            
        except Exception as e:
            logger.error(f"Error processing article task: {e}")
            await self.send_notification({
                'type': 'processing_failed',
                'article_id': task_data.get('article_id'),
                'url': task_data.get('url'),
                'error': str(e)
            })
    
    async def _handle_cleanup_task(self, task_data: Dict[str, Any]):
        """Handle cleanup tasks"""
        try:
            days_old = task_data.get('days_old', 30)
            
            logger.info(f"Running cleanup task for data older than {days_old} days")
            
            # Simulate cleanup
            await asyncio.sleep(3)
            
            # Here you would call the actual cleanup logic
            # from .db import cleanup_old_data
            # await cleanup_old_data(days_old)
            
            logger.info("Cleanup task completed")
            
            # Send notification
            await self.send_notification({
                'type': 'cleanup_completed',
                'days_old': days_old,
                'status': 'success'
            })
            
        except Exception as e:
            logger.error(f"Error processing cleanup task: {e}")
            await self.send_notification({
                'type': 'cleanup_failed',
                'error': str(e)
            })
    
    async def _handle_notification_task(self, task_data: Dict[str, Any]):
        """Handle notification tasks"""
        try:
            notification_type = task_data.get('type')
            logger.info(f"Processing notification: {notification_type}")
            
            # Here you would implement actual notification logic
            # - Send emails
            # - Send webhooks
            # - Update dashboards
            # - Log to external systems
            
            logger.info(f"Notification {notification_type} processed")
            
        except Exception as e:
            logger.error(f"Error processing notification: {e}")
    
    async def send_embedding_task(self, article_id: int, text: str, model: str = "simple"):
        """Send embedding task to queue"""
        task_data = {
            'article_id': article_id,
            'text': text,
            'model': model,
            'timestamp': asyncio.get_event_loop().time()
        }
        
        message_id = await redis_streams.send_message(EMBEDDING_STREAM, task_data)
        if message_id:
            logger.info(f"Embedding task sent for article {article_id}")
        return message_id
    
    async def send_processing_task(self, article_id: int, url: str, title: str, text: str):
        """Send processing task to queue"""
        task_data = {
            'article_id': article_id,
            'url': url,
            'title': title,
            'text': text,
            'timestamp': asyncio.get_event_loop().time()
        }
        
        message_id = await redis_streams.send_message(PROCESSING_STREAM, task_data)
        if message_id:
            logger.info(f"Processing task sent for article {article_id}")
        return message_id
    
    async def send_cleanup_task(self, days_old: int = 30):
        """Send cleanup task to queue"""
        task_data = {
            'days_old': days_old,
            'timestamp': asyncio.get_event_loop().time()
        }
        
        message_id = await redis_streams.send_message(CLEANUP_STREAM, task_data)
        if message_id:
            logger.info(f"Cleanup task sent for {days_old} days old data")
        return message_id
    
    async def send_notification(self, notification_data: Dict[str, Any]):
        """Send notification to queue"""
        notification_data['timestamp'] = asyncio.get_event_loop().time()
        
        message_id = await redis_streams.send_message(NOTIFICATION_STREAM, notification_data)
        if message_id:
            logger.debug(f"Notification sent: {notification_data.get('type')}")
        return message_id
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        try:
            stats = {}
            
            for stream_name in [EMBEDDING_STREAM, PROCESSING_STREAM, CLEANUP_STREAM, NOTIFICATION_STREAM]:
                stream_info = await redis_streams.get_stream_info(stream_name)
                stats[stream_name] = {
                    'length': stream_info.get('length', 0),
                    'groups': stream_info.get('groups', 0)
                }
            
            return stats
        except Exception as e:
            logger.error(f"Error getting queue stats: {e}")
            return {}

class PeriodicScheduler:
    """Periodic task scheduler using Redis Streams"""
    
    def __init__(self):
        self.tasks: Dict[str, asyncio.Task] = {}
        self.task_manager = AsyncTaskManager()
    
    async def schedule_periodic(self, name: str, func: Callable, interval_seconds: int, *args, **kwargs):
        """Schedule a function to run periodically"""
        async def periodic_task_wrapper():
            while True:
                try:
                    logger.info(f"Running periodic task: {name}")
                    await func(*args, **kwargs)
                except asyncio.CancelledError:
                    logger.info(f"Periodic task {name} cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in periodic task {name}: {e}")
                await asyncio.sleep(interval_seconds)
        
        if name in self.tasks:
            self.tasks[name].cancel()
        self.tasks[name] = asyncio.create_task(periodic_task_wrapper())
        logger.info(f"Periodic task '{name}' scheduled to run every {interval_seconds} seconds")
    
    async def cancel_all(self):
        """Cancel all scheduled periodic tasks"""
        for name, task in self.tasks.items():
            task.cancel()
            logger.info(f"Cancelled periodic task: {name}")
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        self.tasks.clear()

# Global instances
task_manager = AsyncTaskManager()
periodic_scheduler = PeriodicScheduler()

# Convenience functions for backward compatibility
async def schedule_embedding_task(article_id: int, text: str, model: str = "simple"):
    """Schedule an embedding task"""
    return await task_manager.send_embedding_task(article_id, text, model)

async def schedule_processing_task(article_id: int, url: str, title: str, text: str):
    """Schedule a processing task"""
    return await task_manager.send_processing_task(article_id, url, title, text)

async def schedule_cleanup_task(days_old: int = 30):
    """Schedule a cleanup task"""
    return await task_manager.send_cleanup_task(days_old)