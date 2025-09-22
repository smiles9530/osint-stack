"""
Enhanced WebSocket Manager for Real-time Updates
Handles WebSocket connections, broadcasting, message queuing, persistence, and advanced real-time features
"""

import asyncio
import json
import logging
import time
import heapq
from typing import Dict, List, Set, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict, deque
from fastapi import WebSocket, WebSocketDisconnect
from .auth import get_current_user
from jose import JWTError, jwt
from .config import settings
from .enhanced_error_handling import error_monitor, ServiceErrorHandler

logger = logging.getLogger(__name__)

# Simple token validation for WebSocket
async def validate_websocket_token(token: str):
    """Validate JWT token for WebSocket connection"""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        username: str = payload.get("sub")
        if username is None:
            return None
        return {"username": username, "user_id": payload.get("user_id")}
    except JWTError:
        return None

# Enhanced WebSocket Models
class MessagePriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class MessageType(Enum):
    NOTIFICATION = "notification"
    ALERT = "alert"
    UPDATE = "update"
    STATUS = "status"
    DATA = "data"
    CONTROL = "control"

@dataclass
class QueuedMessage:
    id: str
    user_id: int
    message_type: MessageType
    priority: MessagePriority
    content: Dict[str, Any]
    created_at: datetime
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        # Higher priority messages come first
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        # Earlier messages come first
        return self.created_at < other.created_at

@dataclass
class ConnectionInfo:
    websocket: WebSocket
    user_id: int
    connected_at: datetime
    last_activity: datetime
    client_info: Dict[str, Any]
    subscriptions: Set[str] = field(default_factory=set)
    message_queue: List[QueuedMessage] = field(default_factory=list)
    is_active: bool = True

class MessageQueue:
    """Priority-based message queue with persistence"""
    
    def __init__(self, max_size: int = 10000):
        self.queue: List[QueuedMessage] = []
        self.message_store: Dict[str, QueuedMessage] = {}
        self.max_size = max_size
        self._lock = asyncio.Lock()
    
    async def enqueue(self, message: QueuedMessage) -> bool:
        """Add message to queue"""
        async with self._lock:
            if len(self.queue) >= self.max_size:
                # Remove oldest low priority message
                self._remove_oldest_low_priority()
            
            heapq.heappush(self.queue, message)
            self.message_store[message.id] = message
            return True
    
    async def dequeue(self, user_id: int) -> Optional[QueuedMessage]:
        """Get next message for user"""
        async with self._lock:
            # Find next message for user
            for i, message in enumerate(self.queue):
                if message.user_id == user_id and self._is_message_valid(message):
                    # Remove from queue
                    self.queue.pop(i)
                    heapq.heapify(self.queue)  # Reheap
                    del self.message_store[message.id]
                    return message
            return None
    
    async def get_user_messages(self, user_id: int, limit: int = 10) -> List[QueuedMessage]:
        """Get all messages for user"""
        async with self._lock:
            messages = []
            for message in self.queue:
                if message.user_id == user_id and self._is_message_valid(message):
                    messages.append(message)
                    if len(messages) >= limit:
                        break
            return messages
    
    def _is_message_valid(self, message: QueuedMessage) -> bool:
        """Check if message is still valid"""
        if message.expires_at and datetime.utcnow() > message.expires_at:
            return False
        return True
    
    def _remove_oldest_low_priority(self):
        """Remove oldest low priority message"""
        for i, message in enumerate(self.queue):
            if message.priority == MessagePriority.LOW:
                self.queue.pop(i)
                heapq.heapify(self.queue)
                del self.message_store[message.id]
                break
    
    async def cleanup_expired(self):
        """Remove expired messages"""
        async with self._lock:
            expired_indices = []
            for i, message in enumerate(self.queue):
                if not self._is_message_valid(message):
                    expired_indices.append(i)
            
            # Remove expired messages (in reverse order to maintain indices)
            for i in reversed(expired_indices):
                message = self.queue.pop(i)
                del self.message_store[message.id]
            
            heapq.heapify(self.queue)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            "total_messages": len(self.queue),
            "max_size": self.max_size,
            "priority_distribution": {
                priority.name: sum(1 for m in self.queue if m.priority == priority)
                for priority in MessagePriority
            }
        }

class EnhancedConnectionManager:
    """Enhanced WebSocket connection manager with queuing, persistence, and advanced features"""
    
    def __init__(self):
        # Store active connections by user ID
        self.active_connections: Dict[int, List[ConnectionInfo]] = {}
        # Store connection metadata by WebSocket
        self.connection_metadata: Dict[WebSocket, ConnectionInfo] = {}
        # Store subscription topics
        self.topic_subscriptions: Dict[str, Set[WebSocket]] = {}
        # Message queue for offline users
        self.message_queue = MessageQueue()
        # Message handlers
        self.message_handlers: Dict[str, Callable] = {}
        # Connection statistics
        self.stats = {
            "total_connections": 0,
            "total_messages_sent": 0,
            "total_messages_queued": 0,
            "connection_errors": 0
        }
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._background_tasks_started = False
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        if not self._background_tasks_started:
            try:
                self._background_tasks.add(asyncio.create_task(self._message_processor()))
                self._background_tasks.add(asyncio.create_task(self._connection_cleanup()))
                self._background_tasks.add(asyncio.create_task(self._queue_cleanup()))
                self._background_tasks_started = True
            except RuntimeError:
                # No event loop running, will start later
                pass
    
    async def shutdown(self):
        """Shutdown the connection manager and cleanup all tasks"""
        try:
            # Cancel all background tasks
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"Error cancelling background task: {e}")
            
            self._background_tasks.clear()
            self._background_tasks_started = False
            
            # Close all active connections
            for user_id, connections in list(self.active_connections.items()):
                for connection in connections:
                    try:
                        await connection.websocket.close(code=1000, reason="Server shutdown")
                    except Exception as e:
                        logger.error(f"Error closing WebSocket connection: {e}")
            
            # Clear all data structures
            self.active_connections.clear()
            self.connection_metadata.clear()
            self.topic_subscriptions.clear()
            
            logger.info("WebSocket manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during WebSocket manager shutdown: {e}")
    
    async def _message_processor(self):
        """Process queued messages for active connections"""
        while True:
            try:
                await asyncio.sleep(1)  # Process every second
                
                # Process messages for all active connections
                for user_id, connections in self.active_connections.items():
                    for connection in connections:
                        if connection.is_active:
                            await self._process_user_messages(connection)
                            
            except Exception as e:
                logger.error(f"Error in message processor: {e}")
                await asyncio.sleep(5)
    
    async def _process_user_messages(self, connection: ConnectionInfo):
        """Process queued messages for a specific connection"""
        try:
            # Get next message from queue
            message = await self.message_queue.dequeue(connection.user_id)
            if message:
                await self._send_message_to_connection(connection, message)
                self.stats["total_messages_sent"] += 1
        except Exception as e:
            logger.error(f"Error processing messages for user {connection.user_id}: {e}")
    
    async def _connection_cleanup(self):
        """Clean up inactive connections"""
        while True:
            try:
                await asyncio.sleep(60)  # Cleanup every minute
                
                current_time = datetime.utcnow()
                inactive_connections = []
                
                for websocket, connection in self.connection_metadata.items():
                    # Mark as inactive if no activity for 5 minutes
                    if (current_time - connection.last_activity).total_seconds() > 300:
                        connection.is_active = False
                        inactive_connections.append(websocket)
                
                # Remove inactive connections
                for websocket in inactive_connections:
                    self.disconnect(websocket)
                    
            except Exception as e:
                logger.error(f"Error in connection cleanup: {e}")
                await asyncio.sleep(60)
    
    async def _queue_cleanup(self):
        """Clean up expired messages from queue"""
        while True:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                await self.message_queue.cleanup_expired()
            except Exception as e:
                logger.error(f"Error in queue cleanup: {e}")
                await asyncio.sleep(300)
    
    async def connect(self, websocket: WebSocket, user_id: int, client_info: Dict[str, Any] = None):
        """Accept a WebSocket connection with enhanced features"""
        try:
            # Start background tasks if not already started
            self._start_background_tasks()
            
            await websocket.accept()
            
            # Create connection info
            connection_info = ConnectionInfo(
                websocket=websocket,
                user_id=user_id,
                connected_at=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                client_info=client_info or {}
            )
            
            # Add to active connections
            if user_id not in self.active_connections:
                self.active_connections[user_id] = []
            
            self.active_connections[user_id].append(connection_info)
            self.connection_metadata[websocket] = connection_info
            
            self.stats["total_connections"] += 1
            
            logger.info(f"WebSocket connected for user {user_id}")
            
            # Send welcome message with queued messages
            await self._send_welcome_message(connection_info)
            
            # Process any queued messages
            await self._process_user_messages(connection_info)
            
        except Exception as e:
            logger.error(f"Error connecting WebSocket for user {user_id}: {e}")
            self.stats["connection_errors"] += 1
            await error_monitor.record_error("websocket", e, {"user_id": user_id})
    
    async def _send_welcome_message(self, connection: ConnectionInfo):
        """Send welcome message with connection info"""
        welcome_message = {
            "type": "connection_established",
            "message": "WebSocket connection established",
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": connection.user_id,
            "connection_id": str(uuid.uuid4()),
            "server_time": datetime.utcnow().isoformat()
        }
        
        await self._send_message_to_connection(connection, QueuedMessage(
            id=str(uuid.uuid4()),
            user_id=connection.user_id,
            message_type=MessageType.CONTROL,
            priority=MessagePriority.NORMAL,
            content=welcome_message,
            created_at=datetime.utcnow()
        ))
    
    async def _send_message_to_connection(self, connection: ConnectionInfo, message: QueuedMessage):
        """Send message to specific connection with error handling"""
        try:
            await connection.websocket.send_text(json.dumps({
                "id": message.id,
                "type": message.message_type.value,
                "priority": message.priority.name,
                "content": message.content,
                "timestamp": message.created_at.isoformat()
            }))
            
            connection.last_activity = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error sending message to user {connection.user_id}: {e}")
            connection.retry_count += 1
            
            if connection.retry_count < message.max_retries:
                # Re-queue message for retry
                await self.message_queue.enqueue(message)
            else:
                logger.warning(f"Max retries exceeded for message {message.id}")
                self.stats["connection_errors"] += 1
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection with cleanup"""
        try:
            if websocket in self.connection_metadata:
                connection = self.connection_metadata[websocket]
                user_id = connection.user_id
                
                # Mark connection as inactive
                connection.is_active = False
                
                # Remove from active connections
                if user_id in self.active_connections:
                    self.active_connections[user_id] = [
                        conn for conn in self.active_connections[user_id] 
                        if conn.websocket != websocket
                    ]
                    if not self.active_connections[user_id]:
                        del self.active_connections[user_id]
                
                # Remove from topic subscriptions
                for topic in connection.subscriptions:
                    if topic in self.topic_subscriptions:
                        self.topic_subscriptions[topic].discard(websocket)
                        if not self.topic_subscriptions[topic]:
                            del self.topic_subscriptions[topic]
                
                # Clean up connection metadata
                del self.connection_metadata[websocket]
                
                logger.info(f"WebSocket disconnected for user {user_id}")
                
        except Exception as e:
            logger.error(f"Error disconnecting WebSocket: {e}")
            # Note: error_monitor.record_error is async, but disconnect is not
            # This will be handled by the calling context
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket, 
                                   priority: MessagePriority = MessagePriority.NORMAL,
                                   message_type: MessageType = MessageType.NOTIFICATION):
        """Send a message to a specific WebSocket connection with enhanced features"""
        try:
            if websocket in self.connection_metadata:
                connection = self.connection_metadata[websocket]
                queued_message = QueuedMessage(
                    id=str(uuid.uuid4()),
                    user_id=connection.user_id,
                    message_type=message_type,
                    priority=priority,
                    content=message,
                    created_at=datetime.utcnow()
                )
                await self._send_message_to_connection(connection, queued_message)
            else:
                logger.warning(f"WebSocket not found in connection metadata")
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            await error_monitor.record_error("websocket", e, {"message": str(message)[:100]})
    
    async def send_to_user(self, user_id: int, message: Dict[str, Any], 
                          priority: MessagePriority = MessagePriority.NORMAL,
                          message_type: MessageType = MessageType.NOTIFICATION):
        """Send a message to all connections for a specific user with queuing"""
        try:
            if user_id in self.active_connections:
                # Send to active connections
                for connection in self.active_connections[user_id]:
                    if connection.is_active:
                        queued_message = QueuedMessage(
                            id=str(uuid.uuid4()),
                            user_id=user_id,
                            message_type=message_type,
                            priority=priority,
                            content=message,
                            created_at=datetime.utcnow()
                        )
                        await self._send_message_to_connection(connection, queued_message)
                    else:
                        # Queue message for inactive connections
                        queued_message = QueuedMessage(
                            id=str(uuid.uuid4()),
                            user_id=user_id,
                            message_type=message_type,
                            priority=priority,
                            content=message,
                            created_at=datetime.utcnow()
                        )
                        await self.message_queue.enqueue(queued_message)
                        self.stats["total_messages_queued"] += 1
            else:
                # User not connected, queue message
                queued_message = QueuedMessage(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    message_type=message_type,
                    priority=priority,
                    content=message,
                    created_at=datetime.utcnow()
                )
                await self.message_queue.enqueue(queued_message)
                self.stats["total_messages_queued"] += 1
                
        except Exception as e:
            logger.error(f"Error sending message to user {user_id}: {e}")
            await error_monitor.record_error("websocket", e, {"user_id": user_id})
    
    async def broadcast_to_all(self, message: Dict[str, Any], 
                              priority: MessagePriority = MessagePriority.NORMAL,
                              message_type: MessageType = MessageType.NOTIFICATION):
        """Broadcast a message to all connected users with enhanced features"""
        try:
            queued_message = QueuedMessage(
                id=str(uuid.uuid4()),
                user_id=0,  # Broadcast message
                message_type=message_type,
                priority=priority,
                content=message,
                created_at=datetime.utcnow()
            )
            
            for user_id, connections in self.active_connections.items():
                for connection in connections:
                    if connection.is_active:
                        await self._send_message_to_connection(connection, queued_message)
                    else:
                        # Queue for inactive connections
                        await self.message_queue.enqueue(queued_message)
                        self.stats["total_messages_queued"] += 1
                        
        except Exception as e:
            logger.error(f"Error broadcasting message: {e}")
            await error_monitor.record_error("websocket", e, {"message": str(message)[:100]})
    
    async def send_to_topic(self, topic: str, message: Dict[str, Any],
                           priority: MessagePriority = MessagePriority.NORMAL,
                           message_type: MessageType = MessageType.NOTIFICATION):
        """Send a message to all users subscribed to a specific topic with queuing"""
        try:
            if topic in self.topic_subscriptions:
                queued_message = QueuedMessage(
                    id=str(uuid.uuid4()),
                    user_id=0,  # Topic message
                    message_type=message_type,
                    priority=priority,
                    content=message,
                    created_at=datetime.utcnow()
                )
                
                for websocket in self.topic_subscriptions[topic]:
                    if websocket in self.connection_metadata:
                        connection = self.connection_metadata[websocket]
                        if connection.is_active:
                            await self._send_message_to_connection(connection, queued_message)
                        else:
                            # Queue for inactive connections
                            await self.message_queue.enqueue(queued_message)
                            self.stats["total_messages_queued"] += 1
                            
        except Exception as e:
            logger.error(f"Error sending topic message to {topic}: {e}")
            await error_monitor.record_error("websocket", e, {"topic": topic})
    
    async def subscribe_to_topic(self, websocket: WebSocket, topic: str):
        """Subscribe a WebSocket connection to a topic with enhanced tracking"""
        try:
            if websocket in self.connection_metadata:
                connection = self.connection_metadata[websocket]
                if topic not in self.topic_subscriptions:
                    self.topic_subscriptions[topic] = set()
                self.topic_subscriptions[topic].add(websocket)
                connection.subscriptions.add(topic)
                logger.info(f"WebSocket subscribed to topic: {topic}")
                
                # Send confirmation
                await self.send_personal_message({
                    "type": "subscription_confirmed",
                    "topic": topic,
                    "timestamp": datetime.utcnow().isoformat()
                }, websocket, MessagePriority.LOW, MessageType.CONTROL)
                
        except Exception as e:
            logger.error(f"Error subscribing to topic {topic}: {e}")
            await error_monitor.record_error("websocket", e, {"topic": topic})
    
    async def unsubscribe_from_topic(self, websocket: WebSocket, topic: str):
        """Unsubscribe a WebSocket connection from a topic with enhanced tracking"""
        try:
            if websocket in self.connection_metadata:
                connection = self.connection_metadata[websocket]
                if topic in self.topic_subscriptions:
                    self.topic_subscriptions[topic].discard(websocket)
                    if not self.topic_subscriptions[topic]:
                        del self.topic_subscriptions[topic]
                connection.subscriptions.discard(topic)
                logger.info(f"WebSocket unsubscribed from topic: {topic}")
                
                # Send confirmation
                await self.send_personal_message({
                    "type": "unsubscription_confirmed",
                    "topic": topic,
                    "timestamp": datetime.utcnow().isoformat()
                }, websocket, MessagePriority.LOW, MessageType.CONTROL)
                
        except Exception as e:
            logger.error(f"Error unsubscribing from topic {topic}: {e}")
            await error_monitor.record_error("websocket", e, {"topic": topic})
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get enhanced WebSocket connection statistics"""
        try:
            total_connections = sum(len(connections) for connections in self.active_connections.values())
            active_connections = sum(
                len([conn for conn in connections if conn.is_active]) 
                for connections in self.active_connections.values()
            )
            total_topics = len(self.topic_subscriptions)
            
            # Get queue statistics
            queue_stats = self.message_queue.get_stats()
            
            return {
                "total_connections": total_connections,
                "active_connections": active_connections,
                "active_users": len(self.active_connections),
                "total_topics": total_topics,
                "topics": list(self.topic_subscriptions.keys()),
                "message_queue": queue_stats,
                "websocket_stats": self.stats,
                "last_updated": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting connection stats: {e}")
            return {"error": str(e)}

# Global enhanced connection manager instance
manager = EnhancedConnectionManager()

class NotificationService:
    """Service for sending real-time notifications"""
    
    @staticmethod
    async def send_analysis_complete(user_id: int, article_id: str, analysis_type: str, results: Dict[str, Any]):
        """Send analysis completion notification with enhanced features"""
        message = {
            "type": "analysis_complete",
            "article_id": article_id,
            "analysis_type": analysis_type,
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        await manager.send_to_user(user_id, message, MessagePriority.NORMAL, MessageType.NOTIFICATION)
    
    @staticmethod
    async def send_new_article(user_id: int, article: Dict[str, Any]):
        """Send new article notification with enhanced features"""
        message = {
            "type": "new_article",
            "article": article,
            "timestamp": datetime.utcnow().isoformat()
        }
        await manager.send_to_user(user_id, message, MessagePriority.NORMAL, MessageType.NOTIFICATION)
    
    @staticmethod
    async def send_alert(user_id: int, alert: Dict[str, Any]):
        """Send high-priority alert notification"""
        message = {
            "type": "alert",
            "alert": alert,
            "timestamp": datetime.utcnow().isoformat()
        }
        await manager.send_to_user(user_id, message, MessagePriority.HIGH, MessageType.ALERT)
    
    @staticmethod
    async def send_digest_ready(user_id: int, digest: Dict[str, Any]):
        """Send digest ready notification with enhanced features"""
        message = {
            "type": "digest_ready",
            "digest": digest,
            "timestamp": datetime.utcnow().isoformat()
        }
        await manager.send_to_user(user_id, message, MessagePriority.NORMAL, MessageType.NOTIFICATION)
    
    @staticmethod
    async def send_system_status(status: Dict[str, Any]):
        """Broadcast system status to all users with enhanced features"""
        message = {
            "type": "system_status",
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
        await manager.broadcast_to_all(message, MessagePriority.LOW, MessageType.SYSTEM)
    
    @staticmethod
    async def send_topic_update(topic: str, update: Dict[str, Any]):
        """Send topic update to subscribers with enhanced features"""
        message = {
            "type": "topic_update",
            "topic": topic,
            "update": update,
            "timestamp": datetime.utcnow().isoformat()
        }
        await manager.send_to_topic(topic, message, MessagePriority.NORMAL, MessageType.NOTIFICATION)
    
    @staticmethod
    async def send_control_message(user_id: int, message: Dict[str, Any]):
        """Send control message to a specific user"""
        await manager.send_to_user(user_id, message, MessagePriority.LOW, MessageType.CONTROL)
    
    @staticmethod
    async def send_batch_notifications(notifications: List[Dict[str, Any]]):
        """Send multiple notifications efficiently"""
        for notification in notifications:
            user_id = notification.get("user_id")
            if user_id:
                await manager.send_to_user(user_id, notification, MessagePriority.NORMAL, MessageType.NOTIFICATION)
    

class WebSocketHandler:
    """Handles WebSocket message processing"""
    
    @staticmethod
    async def handle_message(websocket: WebSocket, message: str):
        """Handle incoming WebSocket message with enhanced features"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "subscribe":
                topic = data.get("topic")
                if topic:
                    await manager.subscribe_to_topic(websocket, topic)
            
            elif message_type == "unsubscribe":
                topic = data.get("topic")
                if topic:
                    await manager.unsubscribe_from_topic(websocket, topic)
            
            elif message_type == "ping":
                await manager.send_personal_message({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                }, websocket, MessagePriority.LOW, MessageType.CONTROL)
            
            
            elif message_type == "get_queue_status":
                if websocket in manager.connection_metadata:
                    connection = manager.connection_metadata[websocket]
                    queue_stats = manager.message_queue.get_stats()
                    await manager.send_personal_message({
                        "type": "queue_status",
                        "user_id": connection.user_id,
                        "queue_stats": queue_stats,
                        "timestamp": datetime.utcnow().isoformat()
                    }, websocket, MessagePriority.LOW, MessageType.CONTROL)
            
            else:
                # Unknown message type
                await manager.send_personal_message({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}",
                    "timestamp": datetime.utcnow().isoformat()
                }, websocket, MessagePriority.LOW, MessageType.CONTROL)
                
        except json.JSONDecodeError:
            await manager.send_personal_message({
                "type": "error",
                "message": "Invalid JSON format",
                "timestamp": datetime.utcnow().isoformat()
            }, websocket, MessagePriority.LOW, MessageType.CONTROL)
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await error_monitor.record_error("websocket", e, {"message": message[:100]})
            await manager.send_personal_message({
                "type": "error",
                "message": "Internal server error",
                "timestamp": datetime.utcnow().isoformat()
            }, websocket, MessagePriority.LOW, MessageType.CONTROL)

async def websocket_endpoint(websocket: WebSocket, token: str = None):
    """Enhanced WebSocket endpoint handler with advanced features"""
    user_id = None
    client_info = {}
    
    try:
        # Authenticate user if token provided
        if token:
            try:
                user_data = await validate_websocket_token(token)
                if user_data:
                    user_id = user_data.get("user_id")
                    username = user_data.get("username")
                    client_info = {
                        "user_id": user_id,
                        "username": username,
                        "authenticated": True
                    }
                else:
                    logger.warning("WebSocket authentication failed: Invalid token")
                    await websocket.close(code=1008, reason="Authentication failed")
                    return
            except Exception as e:
                logger.warning(f"WebSocket authentication failed: {e}")
                await websocket.close(code=1008, reason="Authentication failed")
                return
        else:
            user_id = None
            client_info = {
                "user_id": None,
                "username": None,
                "authenticated": False
            }
        
        # Connect to manager with client info
        await manager.connect(websocket, user_id or 0, client_info)
        
        # Handle messages with enhanced error handling
        while True:
            try:
                message = await websocket.receive_text()
                await WebSocketHandler.handle_message(websocket, message)
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for user {user_id}")
                break
            except Exception as e:
                logger.error(f"WebSocket error for user {user_id}: {e}")
                await error_monitor.record_error("websocket", e, {"user_id": user_id})
                break
    
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        await error_monitor.record_error("websocket", e, {"user_id": user_id})
    finally:
        manager.disconnect(websocket)

async def send_periodic_updates():
    """Send enhanced periodic updates to all connected users"""
    while True:
        try:
            # Send system health update every 30 seconds
            from .monitoring import monitoring_service
            
            # Check if monitoring service is available
            if monitoring_service is None:
                logger.debug("Monitoring service not available, sending basic status update")
                # Send basic system status without detailed health data
                await NotificationService.send_system_status({
                    "health": True,  # Assume healthy if monitoring not available
                    "services": {"monitoring": "unavailable"},
                    "timestamp": datetime.utcnow().isoformat(),
                    "note": "Monitoring service not initialized"
                })
            else:
                try:
                    health_data = await monitoring_service.get_system_health()
                    
                    # Send system status notification
                    await NotificationService.send_system_status({
                        "health": health_data.get("overall_health", False),
                        "services": health_data.get("services", {}),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                except Exception as health_error:
                    logger.warning(f"Error getting system health: {health_error}")
                    # Send fallback status
                    await NotificationService.send_system_status({
                        "health": True,
                        "services": {"monitoring": "error"},
                        "timestamp": datetime.utcnow().isoformat(),
                        "error": str(health_error)
                    })
            
            # Send connection stats update
            try:
                stats = manager.get_connection_stats()
                await manager.broadcast_to_all({
                    "type": "connection_stats_update",
                    "stats": stats,
                    "timestamp": datetime.utcnow().isoformat()
                }, MessagePriority.LOW, MessageType.SYSTEM)
            except Exception as stats_error:
                logger.warning(f"Error getting connection stats: {stats_error}")
            
            await asyncio.sleep(30)
        
        except Exception as e:
            logger.error(f"Error in periodic updates: {type(e).__name__}: {str(e)}")
            logger.error(f"Error details: {repr(e)}")
            try:
                await error_monitor.record_error("websocket", e, {"function": "send_periodic_updates"})
            except Exception as record_error:
                logger.error(f"Failed to record error: {record_error}")
            await asyncio.sleep(30)
