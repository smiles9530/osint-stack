"""
GPU Performance Monitoring Service
Provides real-time GPU metrics and performance monitoring for the OSINT stack
"""

import asyncio
import logging
import time
import psutil
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .gpu_utils import gpu_manager

logger = logging.getLogger(__name__)

@dataclass
class GPUMetrics:
    """GPU performance metrics"""
    timestamp: datetime
    gpu_available: bool
    gpu_name: Optional[str]
    total_memory_gb: float
    allocated_memory_gb: float
    cached_memory_gb: float
    free_memory_gb: float
    memory_utilization: float
    gpu_utilization: float
    temperature: Optional[float]
    power_usage: Optional[float]
    cpu_usage: float
    system_memory_usage: float
    active_models: List[str]

class GPUMonitor:
    """GPU performance monitoring service"""
    
    def __init__(self):
        self.metrics_history = []
        self.max_history = 1000  # Keep last 1000 measurements
        self.monitoring_active = False
        self.monitoring_interval = 30  # seconds
        self.active_models = set()
        
    def start_monitoring(self, interval: int = 30):
        """Start GPU monitoring"""
        self.monitoring_interval = interval
        self.monitoring_active = True
        logger.info(f"Started GPU monitoring with {interval}s interval")
        
    def stop_monitoring(self):
        """Stop GPU monitoring"""
        self.monitoring_active = False
        logger.info("Stopped GPU monitoring")
    
    def add_active_model(self, model_name: str):
        """Track active model"""
        self.active_models.add(model_name)
        logger.debug(f"Added active model: {model_name}")
    
    def remove_active_model(self, model_name: str):
        """Remove model from active tracking"""
        self.active_models.discard(model_name)
        logger.debug(f"Removed active model: {model_name}")
    
    async def collect_metrics(self) -> GPUMetrics:
        """Collect current GPU and system metrics"""
        try:
            timestamp = datetime.now()
            
            # Get GPU metrics
            gpu_info = gpu_manager.get_memory_info()
            gpu_available = gpu_info.get("gpu_available", False)
            
            if gpu_available and TORCH_AVAILABLE:
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_utilization = self._get_gpu_utilization()
                    temperature = self._get_gpu_temperature()
                    power_usage = self._get_gpu_power_usage()
                except Exception as e:
                    logger.warning(f"Failed to get GPU details: {e}")
                    gpu_name = "Unknown GPU"
                    gpu_utilization = 0.0
                    temperature = None
                    power_usage = None
            else:
                gpu_name = None
                gpu_utilization = 0.0
                temperature = None
                power_usage = None
            
            # Get system metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            system_memory = psutil.virtual_memory()
            
            metrics = GPUMetrics(
                timestamp=timestamp,
                gpu_available=gpu_available,
                gpu_name=gpu_name,
                total_memory_gb=gpu_info.get("total_memory_gb", 0.0),
                allocated_memory_gb=gpu_info.get("allocated_memory_gb", 0.0),
                cached_memory_gb=gpu_info.get("cached_memory_gb", 0.0),
                free_memory_gb=gpu_info.get("free_memory_gb", 0.0),
                memory_utilization=gpu_info.get("memory_utilization", 0.0),
                gpu_utilization=gpu_utilization,
                temperature=temperature,
                power_usage=power_usage,
                cpu_usage=cpu_usage,
                system_memory_usage=system_memory.percent,
                active_models=list(self.active_models)
            )
            
            # Store in history
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history:
                self.metrics_history.pop(0)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect GPU metrics: {e}")
            # Return minimal metrics on error
            return GPUMetrics(
                timestamp=datetime.now(),
                gpu_available=False,
                gpu_name=None,
                total_memory_gb=0.0,
                allocated_memory_gb=0.0,
                cached_memory_gb=0.0,
                free_memory_gb=0.0,
                memory_utilization=0.0,
                gpu_utilization=0.0,
                temperature=None,
                power_usage=None,
                cpu_usage=0.0,
                system_memory_usage=0.0,
                active_models=[]
            )
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                # This is a simplified approach - in production you might want to use nvidia-ml-py
                return 0.0  # Placeholder - would need nvidia-ml-py for real utilization
            return 0.0
        except Exception:
            return 0.0
    
    def _get_gpu_temperature(self) -> Optional[float]:
        """Get GPU temperature in Celsius"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                # This is a simplified approach - in production you might want to use nvidia-ml-py
                return None  # Placeholder - would need nvidia-ml-py for real temperature
            return None
        except Exception:
            return None
    
    def _get_gpu_power_usage(self) -> Optional[float]:
        """Get GPU power usage in watts"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                # This is a simplified approach - in production you might want to use nvidia-ml-py
                return None  # Placeholder - would need nvidia-ml-py for real power usage
            return None
        except Exception:
            return None
    
    def get_current_metrics(self) -> Optional[GPUMetrics]:
        """Get the most recent metrics"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics summary for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"error": "No metrics available for the specified time period"}
        
        # Calculate averages
        avg_gpu_utilization = sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics)
        avg_memory_utilization = sum(m.memory_utilization for m in recent_metrics) / len(recent_metrics)
        avg_cpu_usage = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_system_memory = sum(m.system_memory_usage for m in recent_metrics) / len(recent_metrics)
        
        # Get peak values
        peak_gpu_utilization = max(m.gpu_utilization for m in recent_metrics)
        peak_memory_utilization = max(m.memory_utilization for m in recent_metrics)
        peak_cpu_usage = max(m.cpu_usage for m in recent_metrics)
        
        # Get current values
        current = recent_metrics[-1]
        
        return {
            "time_period_hours": hours,
            "total_measurements": len(recent_metrics),
            "gpu_available": current.gpu_available,
            "gpu_name": current.gpu_name,
            "current_metrics": {
                "gpu_utilization": current.gpu_utilization,
                "memory_utilization": current.memory_utilization,
                "cpu_usage": current.cpu_usage,
                "system_memory_usage": current.system_memory_usage,
                "active_models": current.active_models
            },
            "average_metrics": {
                "gpu_utilization": avg_gpu_utilization,
                "memory_utilization": avg_memory_utilization,
                "cpu_usage": avg_cpu_usage,
                "system_memory_usage": avg_system_memory
            },
            "peak_metrics": {
                "gpu_utilization": peak_gpu_utilization,
                "memory_utilization": peak_memory_utilization,
                "cpu_usage": peak_cpu_usage
            },
            "memory_info": {
                "total_gb": current.total_memory_gb,
                "allocated_gb": current.allocated_memory_gb,
                "cached_gb": current.cached_memory_gb,
                "free_gb": current.free_memory_gb
            }
        }
    
    def get_performance_alerts(self) -> List[Dict[str, Any]]:
        """Get performance alerts based on current metrics"""
        alerts = []
        current = self.get_current_metrics()
        
        if not current:
            return alerts
        
        # Memory utilization alert
        if current.memory_utilization > 90:
            alerts.append({
                "type": "high_memory_usage",
                "severity": "warning",
                "message": f"GPU memory utilization is {current.memory_utilization:.1f}%",
                "value": current.memory_utilization,
                "threshold": 90
            })
        elif current.memory_utilization > 95:
            alerts.append({
                "type": "critical_memory_usage",
                "severity": "critical",
                "message": f"GPU memory utilization is critically high: {current.memory_utilization:.1f}%",
                "value": current.memory_utilization,
                "threshold": 95
            })
        
        # CPU usage alert
        if current.cpu_usage > 90:
            alerts.append({
                "type": "high_cpu_usage",
                "severity": "warning",
                "message": f"CPU usage is {current.cpu_usage:.1f}%",
                "value": current.cpu_usage,
                "threshold": 90
            })
        
        # System memory alert
        if current.system_memory_usage > 90:
            alerts.append({
                "type": "high_system_memory",
                "severity": "warning",
                "message": f"System memory usage is {current.system_memory_usage:.1f}%",
                "value": current.system_memory_usage,
                "threshold": 90
            })
        
        return alerts
    
    async def monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                await self.collect_metrics()
                await asyncio.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in GPU monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)

# Global instance
gpu_monitor = GPUMonitor()

async def start_gpu_monitoring(interval: int = 30):
    """Start GPU monitoring"""
    gpu_monitor.start_monitoring(interval)
    asyncio.create_task(gpu_monitor.monitoring_loop())

def stop_gpu_monitoring():
    """Stop GPU monitoring"""
    gpu_monitor.stop_monitoring()

def get_gpu_status() -> Dict[str, Any]:
    """Get current GPU status"""
    current = gpu_monitor.get_current_metrics()
    if not current:
        return {"error": "No GPU metrics available"}
    
    return {
        "gpu_available": current.gpu_available,
        "gpu_name": current.gpu_name,
        "memory_utilization": current.memory_utilization,
        "gpu_utilization": current.gpu_utilization,
        "active_models": current.active_models,
        "alerts": gpu_monitor.get_performance_alerts()
    }

# Global instance
gpu_monitor = GPUMonitor()

async def get_gpu_metrics(hours: int = 1) -> Dict[str, Any]:
    """Get GPU performance metrics for specified time period"""
    try:
        metrics = gpu_monitor.get_metrics_summary(hours)
        return metrics
    except Exception as e:
        logger.error(f"Failed to get GPU metrics: {e}")
        return {"error": str(e)}

async def get_gpu_alerts() -> List[Dict[str, Any]]:
    """Get current GPU performance alerts"""
    try:
        alerts = gpu_monitor.get_performance_alerts()
        return alerts
    except Exception as e:
        logger.error(f"Failed to get GPU alerts: {e}")
        return [{"error": str(e)}]