"""
Comprehensive monitoring and metrics system
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict, deque
import redis
import psutil
from prometheus_client import Counter, Histogram, Gauge, start_http_server, generate_latest

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and stores system metrics"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.metrics = defaultdict(lambda: deque(maxlen=1000))
        
        # Prometheus metrics
        self.request_count = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
        self.request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
        self.cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')
        self.memory_usage = Gauge('memory_usage_bytes', 'Memory usage in bytes')
        
    async def add_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Add a metric value"""
        timestamp = time.time()
        metric_data = {
            'timestamp': timestamp,
            'value': value,
            'tags': tags or {}
        }
        
        self.metrics[name].append(metric_data)
        
        # Store in Redis for persistence
        try:
            await self.redis_client.lpush(f"metrics:{name}", json.dumps(metric_data))
            await self.redis_client.ltrim(f"metrics:{name}", 0, 999)
        except Exception as e:
            logger.warning(f"Failed to store metric in Redis: {e}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values"""
        current = {}
        for name, values in self.metrics.items():
            if values:
                current[name] = values[-1]['value']
        return current

class APIMonitor:
    """Monitors API performance"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.request_times = defaultdict(list)
        
    async def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record API request metrics"""
        self.request_count.labels(method=method, endpoint=endpoint, status=status_code).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
        
        await self.metrics_collector.add_metric('api_request_duration', duration, {
            'method': method,
            'endpoint': endpoint,
            'status': str(status_code)
        })
        
        self.request_times[endpoint].append(duration)
        if len(self.request_times[endpoint]) > 100:
            self.request_times[endpoint] = self.request_times[endpoint][-100:]
    
    def get_endpoint_stats(self) -> Dict[str, Any]:
        """Get statistics for each endpoint"""
        stats = {}
        for endpoint, times in self.request_times.items():
            if times:
                stats[endpoint] = {
                    'avg_response_time': sum(times) / len(times),
                    'max_response_time': max(times),
                    'request_count': len(times)
                }
        return stats

class SystemMonitor:
    """Monitors system resources"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.running = False
        
    async def start(self):
        """Start system monitoring"""
        self.running = True
        asyncio.create_task(self._monitor_loop())
        logger.info("System monitoring started")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                await self.metrics_collector.add_metric('cpu_usage', cpu_percent)
                self.metrics_collector.cpu_usage.set(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                await self.metrics_collector.add_metric('memory_usage', memory.used)
                self.metrics_collector.memory_usage.set(memory.used)
                
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(30)

class MonitoringService:
    """Main monitoring service"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.metrics_collector = MetricsCollector(redis_client)
        self.api_monitor = APIMonitor(self.metrics_collector)
        self.system_monitor = SystemMonitor(self.metrics_collector)
        
        # Start Prometheus server
        try:
            start_http_server(9090)
            logger.info("Prometheus metrics server started on port 9090")
        except Exception as e:
            logger.warning(f"Failed to start Prometheus server: {str(e)}")
    
    async def start(self):
        """Start monitoring services"""
        await self.system_monitor.start()
        logger.info("Monitoring service started")
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics"""
        return generate_latest()
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data"""
        return {
            'current_metrics': self.metrics_collector.get_current_metrics(),
            'endpoint_stats': self.api_monitor.get_endpoint_stats()
        }
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        try:
            # Get current metrics
            current_metrics = self.metrics_collector.get_current_metrics()
            
            # Check database connectivity
            db_health = await self._check_database_health()
            
            # Check Redis connectivity
            redis_health = await self._check_redis_health()
            
            # Check system resources
            system_health = self._check_system_resources()
            
            # Determine overall health
            overall_health = (
                db_health.get('status') == 'healthy' and
                redis_health.get('status') == 'healthy' and
                system_health.get('status') == 'healthy'
            )
            
            return {
                'overall_health': overall_health,
                'timestamp': datetime.utcnow().isoformat(),
                'message': 'System operational' if overall_health else 'System issues detected',
                'services': {
                    'database': db_health,
                    'cache': redis_health,
                    'system': system_health
                },
                'metrics': current_metrics
            }
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {
                'overall_health': False,
                'timestamp': datetime.utcnow().isoformat(),
                'message': f'Health check failed: {str(e)}',
                'services': {},
                'metrics': {}
            }
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        try:
            from . import db
            start_time = time.time()
            
            with db.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    result = cur.fetchone()
                    
            response_time = (time.time() - start_time) * 1000  # ms
            
            return {
                'status': 'healthy' if result and result[0] == 1 else 'unhealthy',
                'response_time_ms': round(response_time, 2),
                'last_checked': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'last_checked': datetime.utcnow().isoformat()
            }
    
    async def _check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connectivity and performance"""
        try:
            start_time = time.time()
            await self.redis_client.ping()
            response_time = (time.time() - start_time) * 1000  # ms
            
            return {
                'status': 'healthy',
                'response_time_ms': round(response_time, 2),
                'last_checked': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'last_checked': datetime.utcnow().isoformat()
            }
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine health based on thresholds
            cpu_healthy = cpu_percent < 90
            memory_healthy = memory.percent < 90
            disk_healthy = disk.percent < 90
            
            overall_healthy = cpu_healthy and memory_healthy and disk_healthy
            
            return {
                'status': 'healthy' if overall_healthy else 'degraded',
                'cpu_usage_percent': round(cpu_percent, 2),
                'memory_usage_percent': round(memory.percent, 2),
                'disk_usage_percent': round(disk.percent, 2),
                'memory_usage_mb': round(memory.used / 1024 / 1024, 2),
                'last_checked': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'last_checked': datetime.utcnow().isoformat()
            }

# Global instance
monitoring_service = None

async def initialize_monitoring(redis_client: redis.Redis):
    """Initialize monitoring service"""
    global monitoring_service
    monitoring_service = MonitoringService(redis_client)
    await monitoring_service.start()
    logger.info("Monitoring system initialized")