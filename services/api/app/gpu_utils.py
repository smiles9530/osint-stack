"""
GPU Utilities for OSINT Stack
Provides device detection, memory management, and GPU-optimized operations
"""

import os
import logging
import torch
from typing import Optional, Dict, Any, List, Tuple
from contextlib import contextmanager
import psutil
import time

logger = logging.getLogger(__name__)

class GPUManager:
    """Manages GPU resources and device placement for ML models"""
    
    def __init__(self):
        self.device = None
        self.gpu_available = False
        self.gpu_memory_total = 0
        self.gpu_memory_used = 0
        self.gpu_memory_free = 0
        self._initialize_gpu()
    
    def _initialize_gpu(self):
        """Initialize GPU detection and configuration"""
        try:
            # Check if CUDA is available
            if torch.cuda.is_available():
                self.gpu_available = True
                self.device = torch.device("cuda:0")
                
                # Get GPU memory info
                self.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                self.gpu_memory_used = torch.cuda.memory_allocated(0)
                self.gpu_memory_free = self.gpu_memory_total - self.gpu_memory_used
                
                # Set memory management
                torch.cuda.empty_cache()
                
                # Configure CUDA settings
                os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
                os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
                
                logger.info(f"GPU initialized: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU Memory: {self.gpu_memory_total / 1024**3:.1f}GB total, "
                          f"{self.gpu_memory_free / 1024**3:.1f}GB free")
            else:
                self.device = torch.device("cpu")
                logger.info("CUDA not available, using CPU")
                
        except Exception as e:
            logger.warning(f"GPU initialization failed: {e}")
            self.device = torch.device("cpu")
            self.gpu_available = False
    
    def get_device(self) -> torch.device:
        """Get the appropriate device for model placement"""
        return self.device
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available and ready"""
        return self.gpu_available and torch.cuda.is_available()
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get current GPU memory information"""
        if not self.is_gpu_available():
            return {"gpu_available": False}
        
        try:
            allocated = torch.cuda.memory_allocated(0)
            cached = torch.cuda.memory_reserved(0)
            total = torch.cuda.get_device_properties(0).total_memory
            
            return {
                "gpu_available": True,
                "total_memory_gb": total / 1024**3,
                "allocated_memory_gb": allocated / 1024**3,
                "cached_memory_gb": cached / 1024**3,
                "free_memory_gb": (total - allocated) / 1024**3,
                "memory_utilization": (allocated / total) * 100
            }
        except Exception as e:
            logger.error(f"Failed to get GPU memory info: {e}")
            return {"gpu_available": False, "error": str(e)}
    
    def clear_cache(self):
        """Clear GPU memory cache"""
        if self.is_gpu_available():
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")
    
    def get_optimal_batch_size(self, model_size_mb: float, max_batch_size: int = 32) -> int:
        """Calculate optimal batch size based on available GPU memory"""
        if not self.is_gpu_available():
            return min(max_batch_size, 8)  # Conservative for CPU
        
        try:
            memory_info = self.get_memory_info()
            free_memory_gb = memory_info.get("free_memory_gb", 0)
            
            # Estimate memory per sample (rough heuristic)
            memory_per_sample_mb = model_size_mb * 0.1  # 10% of model size per sample
            
            # Calculate safe batch size (leave 20% memory free)
            safe_memory_gb = free_memory_gb * 0.8
            optimal_batch_size = int((safe_memory_gb * 1024) / memory_per_sample_mb)
            
            return min(max(optimal_batch_size, 1), max_batch_size)
            
        except Exception as e:
            logger.warning(f"Failed to calculate optimal batch size: {e}")
            return min(max_batch_size, 4)
    
    @contextmanager
    def memory_management(self):
        """Context manager for GPU memory management"""
        if self.is_gpu_available():
            # Clear cache before operation
            self.clear_cache()
            try:
                yield
            finally:
                # Clear cache after operation
                self.clear_cache()
        else:
            yield

class ModelDeviceManager:
    """Manages device placement for specific models"""
    
    def __init__(self, gpu_manager: GPUManager):
        self.gpu_manager = gpu_manager
        self.model_devices = {}
    
    def get_model_device(self, model_name: str) -> torch.device:
        """Get device for a specific model"""
        if model_name in self.model_devices:
            return self.model_devices[model_name]
        
        # Determine best device for this model
        device = self.gpu_manager.get_device()
        self.model_devices[model_name] = device
        
        logger.info(f"Model {model_name} assigned to device: {device}")
        return device
    
    def move_model_to_device(self, model, model_name: str):
        """Move model to appropriate device"""
        device = self.get_model_device(model_name)
        
        try:
            if hasattr(model, 'to'):
                model = model.to(device)
                logger.debug(f"Moved {model_name} to {device}")
            return model
        except Exception as e:
            logger.error(f"Failed to move {model_name} to device: {e}")
            return model
    
    def get_batch_size_for_model(self, model_name: str, model_size_mb: float) -> int:
        """Get optimal batch size for a specific model"""
        return self.gpu_manager.get_optimal_batch_size(model_size_mb)

def get_gpu_manager() -> GPUManager:
    """Get global GPU manager instance"""
    if not hasattr(get_gpu_manager, '_instance'):
        get_gpu_manager._instance = GPUManager()
    return get_gpu_manager._instance

def get_model_device_manager() -> ModelDeviceManager:
    """Get global model device manager instance"""
    if not hasattr(get_model_device_manager, '_instance'):
        get_model_device_manager._instance = ModelDeviceManager(get_gpu_manager())
    return get_model_device_manager._instance

# Global instances
gpu_manager = get_gpu_manager()
model_device_manager = get_model_device_manager()

def log_gpu_status():
    """Log current GPU status for monitoring"""
    memory_info = gpu_manager.get_memory_info()
    
    if memory_info.get("gpu_available"):
        logger.info(f"GPU Status: {memory_info['memory_utilization']:.1f}% utilized, "
                   f"{memory_info['free_memory_gb']:.1f}GB free")
    else:
        logger.info("GPU Status: Not available, using CPU")

def optimize_model_loading(model_name: str, model_size_mb: float = 500) -> Dict[str, Any]:
    """Get optimization recommendations for model loading"""
    recommendations = {
        "model_name": model_name,
        "device": str(gpu_manager.get_device()),
        "gpu_available": gpu_manager.is_gpu_available(),
        "optimal_batch_size": gpu_manager.get_optimal_batch_size(model_size_mb),
        "memory_info": gpu_manager.get_memory_info()
    }
    
    if gpu_manager.is_gpu_available():
        recommendations["optimization_tips"] = [
            "Use mixed precision training if supported",
            "Enable gradient checkpointing for large models",
            "Consider model quantization for inference",
            "Use batch processing for multiple texts"
        ]
    else:
        recommendations["optimization_tips"] = [
            "Consider using smaller models for CPU inference",
            "Use CPU-optimized model variants",
            "Implement text chunking for large documents",
            "Use caching for repeated operations"
        ]
    
    return recommendations
