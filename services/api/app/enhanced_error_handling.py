"""
Enhanced error handling and validation utilities
Provides comprehensive error handling, logging, validation, circuit breakers, retry mechanisms, and advanced monitoring
"""

import asyncio
import logging
import traceback
import time
import random
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
import uuid
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import psycopg2
from psycopg2 import OperationalError, IntegrityError
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)

# Enhanced Error Classes
class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: type = Exception
    success_threshold: int = 3

@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_factor: float = 1.0

@dataclass
class ErrorMetrics:
    total_errors: int = 0
    error_rate: float = 0.0
    last_error_time: Optional[datetime] = None
    error_types: Dict[str, int] = field(default_factory=dict)
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=100))

class APIError(Exception):
    """Base API error class"""
    def __init__(self, message: str, status_code: int = 500, error_code: str = None, details: Dict[str, Any] = None):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or "API_ERROR"
        self.details = details or {}
        super().__init__(self.message)

class ValidationError(APIError):
    """Validation error"""
    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(
            message=message,
            status_code=400,
            error_code="VALIDATION_ERROR",
            details={"field": field, "value": str(value) if value is not None else None}
        )

class DatabaseError(APIError):
    """Database operation error"""
    def __init__(self, message: str, operation: str = None, table: str = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="DATABASE_ERROR",
            details={"operation": operation, "table": table}
        )

class MLProcessingError(APIError):
    """ML processing error"""
    def __init__(self, message: str, model: str = None, input_data: Any = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="ML_PROCESSING_ERROR",
            details={"model": model, "input_type": type(input_data).__name__ if input_data else None}
        )

class RateLimitError(APIError):
    """Rate limiting error"""
    def __init__(self, message: str, retry_after: int = None):
        super().__init__(
            message=message,
            status_code=429,
            error_code="RATE_LIMIT_ERROR",
            details={"retry_after": retry_after}
        )

class CircuitBreakerError(APIError):
    """Circuit breaker error"""
    def __init__(self, message: str, service: str = None):
        super().__init__(
            message=message,
            status_code=503,
            error_code="CIRCUIT_BREAKER_OPEN",
            details={"service": service}
        )

class RetryExhaustedError(APIError):
    """Retry exhausted error"""
    def __init__(self, message: str, attempts: int = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="RETRY_EXHAUSTED",
            details={"attempts": attempts}
        )

# Circuit Breaker Implementation
class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        async with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is OPEN",
                        service=self.name
                    )
            
            try:
                result = await func(*args, **kwargs)
                await self._on_success()
                return result
            except self.config.expected_exception as e:
                await self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.config.recovery_timeout
    
    async def _on_success(self):
        """Handle successful call"""
        self.last_success_time = datetime.utcnow()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit breaker '{self.name}' reset to CLOSED")
        else:
            self.failure_count = 0
    
    async def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker '{self.name}' opened due to {self.failure_count} failures")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None
        }

# Retry Mechanism Implementation
class RetryMechanism:
    """Retry mechanism with exponential backoff and jitter"""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.config.max_attempts:
                    break
                
                delay = self._calculate_delay(attempt)
                logger.warning(f"Attempt {attempt} failed for {func.__name__}: {e}. Retrying in {delay:.2f}s")
                await asyncio.sleep(delay)
        
        raise RetryExhaustedError(
            f"Function {func.__name__} failed after {self.config.max_attempts} attempts",
            attempts=self.config.max_attempts
        ) from last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        delay = self.config.base_delay * (self.config.exponential_base ** (attempt - 1))
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            jitter = random.uniform(0.1, 0.3) * delay
            delay += jitter
        
        return delay * self.config.backoff_factor

# Error Monitoring and Metrics
class ErrorMonitor:
    """Advanced error monitoring and metrics collection"""
    
    def __init__(self):
        self.metrics: Dict[str, ErrorMetrics] = defaultdict(ErrorMetrics)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_mechanisms: Dict[str, RetryMechanism] = {}
        self._lock = asyncio.Lock()
    
    async def record_error(self, service: str, error: Exception, context: Dict[str, Any] = None):
        """Record error occurrence"""
        async with self._lock:
            metrics = self.metrics[service]
            metrics.total_errors += 1
            metrics.last_error_time = datetime.utcnow()
            
            error_type = type(error).__name__
            metrics.error_types[error_type] = metrics.error_types.get(error_type, 0) + 1
            
            error_info = {
                "timestamp": datetime.utcnow().isoformat(),
                "error_type": error_type,
                "message": str(error),
                "context": context or {}
            }
            metrics.recent_errors.append(error_info)
            
            # Update error rate (last 100 errors)
            if len(metrics.recent_errors) > 0:
                recent_errors = list(metrics.recent_errors)
                time_window = 300  # 5 minutes
                now = datetime.utcnow()
                recent_errors_in_window = [
                    e for e in recent_errors 
                    if (now - datetime.fromisoformat(e["timestamp"])).total_seconds() <= time_window
                ]
                metrics.error_rate = len(recent_errors_in_window) / (time_window / 60)  # errors per minute
    
    async def get_metrics(self, service: str = None) -> Dict[str, Any]:
        """Get error metrics for service or all services"""
        async with self._lock:
            if service:
                return {
                    "service": service,
                    "metrics": self.metrics[service].__dict__,
                    "circuit_breaker": self.circuit_breakers.get(service).get_state() if service in self.circuit_breakers else None
                }
            else:
                return {
                    "services": {
                        svc: {
                            "metrics": metrics.__dict__,
                            "circuit_breaker": self.circuit_breakers.get(svc).get_state() if svc in self.circuit_breakers else None
                        }
                        for svc, metrics in self.metrics.items()
                    },
                    "total_services": len(self.metrics)
                }
    
    def get_circuit_breaker(self, service: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get or create circuit breaker for service"""
        if service not in self.circuit_breakers:
            self.circuit_breakers[service] = CircuitBreaker(service, config)
        return self.circuit_breakers[service]
    
    def get_retry_mechanism(self, service: str, config: RetryConfig = None) -> RetryMechanism:
        """Get or create retry mechanism for service"""
        if service not in self.retry_mechanisms:
            self.retry_mechanisms[service] = RetryMechanism(config)
        return self.retry_mechanisms[service]

# Global error monitor instance
error_monitor = ErrorMonitor()

class ErrorHandler:
    """Enhanced centralized error handling with circuit breakers and retry mechanisms"""
    
    @staticmethod
    async def with_circuit_breaker(service: str, config: CircuitBreakerConfig = None):
        """Decorator for circuit breaker protection"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                circuit_breaker = error_monitor.get_circuit_breaker(service, config)
                return await circuit_breaker.call(func, *args, **kwargs)
            return wrapper
        return decorator
    
    @staticmethod
    async def with_retry(service: str, config: RetryConfig = None):
        """Decorator for retry mechanism"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                retry_mechanism = error_monitor.get_retry_mechanism(service, config)
                return await retry_mechanism.execute(func, *args, **kwargs)
            return wrapper
        return decorator
    
    @staticmethod
    async def with_monitoring(service: str):
        """Decorator for error monitoring"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    await error_monitor.record_error(service, e, {
                        "function": func.__name__,
                        "args": str(args)[:200],  # Truncate for logging
                        "kwargs": str(kwargs)[:200]
                    })
                    raise
            return wrapper
        return decorator
    
    @staticmethod
    async def safe_execute(service: str, func: Callable, *args, **kwargs) -> Any:
        """Safely execute function with full error handling stack"""
        try:
            # Get circuit breaker and retry mechanism
            circuit_breaker = error_monitor.get_circuit_breaker(service)
            retry_mechanism = error_monitor.get_retry_mechanism(service)
            
            # Execute with circuit breaker protection
            async def protected_func():
                return await retry_mechanism.execute(func, *args, **kwargs)
            
            return await circuit_breaker.call(protected_func)
            
        except Exception as e:
            await error_monitor.record_error(service, e, {
                "function": func.__name__,
                "args": str(args)[:200],
                "kwargs": str(kwargs)[:200]
            })
            raise
    
    @staticmethod
    def handle_database_error(error: Exception, operation: str = None, table: str = None) -> DatabaseError:
        """Handle database errors with appropriate error codes"""
        if isinstance(error, IntegrityError):
            if "duplicate key" in str(error).lower():
                return DatabaseError(
                    message="Duplicate entry detected",
                    operation=operation,
                    table=table
                )
            elif "foreign key" in str(error).lower():
                return DatabaseError(
                    message="Referenced record not found",
                    operation=operation,
                    table=table
                )
            else:
                return DatabaseError(
                    message="Data integrity constraint violation",
                    operation=operation,
                    table=table
                )
        elif isinstance(error, OperationalError):
            return DatabaseError(
                message="Database connection error",
                operation=operation,
                table=table
            )
        else:
            return DatabaseError(
                message=f"Database error: {str(error)}",
                operation=operation,
                table=table
            )
    
    @staticmethod
    def handle_validation_error(error: ValidationError, field: str = None) -> ValidationError:
        """Handle Pydantic validation errors"""
        error_messages = []
        for err in error.errors():
            field_path = " -> ".join(str(loc) for loc in err["loc"])
            error_messages.append(f"{field_path}: {err['msg']}")
        
        return ValidationError(
            message="Validation failed: " + "; ".join(error_messages),
            field=field
        )
    
    @staticmethod
    def handle_ml_error(error: Exception, model: str = None) -> MLProcessingError:
        """Handle ML processing errors"""
        error_message = str(error)
        if "CUDA" in error_message:
            return MLProcessingError(
                message="GPU processing error - falling back to CPU",
                model=model
            )
        elif "memory" in error_message.lower():
            return MLProcessingError(
                message="Insufficient memory for ML processing",
                model=model
            )
        else:
            return MLProcessingError(
                message=f"ML processing failed: {error_message}",
                model=model
            )

def create_error_response(
    error: APIError,
    request_id: str = None,
    include_traceback: bool = False
) -> JSONResponse:
    """Create standardized error response"""
    
    response_data = {
        "error": error.error_code,
        "message": error.message,
        "status_code": error.status_code,
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": request_id or str(uuid.uuid4()),
        "details": error.details
    }
    
    if include_traceback:
        response_data["traceback"] = traceback.format_exc()
    
    return JSONResponse(
        status_code=error.status_code,
        content=response_data
    )

async def enhanced_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Enhanced global exception handler with monitoring and circuit breaker support"""
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    service_name = getattr(request.state, 'service_name', 'unknown')
    
    # Record error in monitoring system
    await error_monitor.record_error(service_name, exc, {
        "request_id": request_id,
        "method": request.method,
        "url": str(request.url),
        "user_agent": request.headers.get("user-agent", "unknown"),
        "client_ip": request.client.host if request.client else "unknown"
    })
    
    # Log the error with enhanced context
    logger.error(
        f"Unhandled exception in {request.method} {request.url}: {str(exc)}",
        extra={
            "request_id": request_id,
            "service": service_name,
            "method": request.method,
            "url": str(request.url),
            "traceback": traceback.format_exc(),
            "error_type": type(exc).__name__
        }
    )
    
    # Handle different types of exceptions
    if isinstance(exc, APIError):
        return create_error_response(exc, request_id)
    elif isinstance(exc, ValidationError):
        api_error = ErrorHandler.handle_validation_error(exc)
        return create_error_response(api_error, request_id)
    elif isinstance(exc, (psycopg2.Error, OperationalError, IntegrityError)):
        api_error = ErrorHandler.handle_database_error(exc)
        return create_error_response(api_error, request_id)
    elif isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP_ERROR",
                "message": exc.detail,
                "status_code": exc.status_code,
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": request_id,
                "service": service_name
            }
        )
    else:
        # Generic error handling
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "status_code": 500,
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": request_id,
                "service": service_name
            }
        )

# Enhanced utility functions
async def get_error_metrics(service: str = None) -> Dict[str, Any]:
    """Get error metrics for monitoring dashboard"""
    return await error_monitor.get_metrics(service)

def create_circuit_breaker_config(
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    expected_exception: type = Exception,
    success_threshold: int = 3
) -> CircuitBreakerConfig:
    """Create circuit breaker configuration"""
    return CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=expected_exception,
        success_threshold=success_threshold
    )

def create_retry_config(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    backoff_factor: float = 1.0
) -> RetryConfig:
    """Create retry configuration"""
    return RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        backoff_factor=backoff_factor
    )

# Service-specific error handling
class ServiceErrorHandler:
    """Service-specific error handling with predefined configurations"""
    
    @staticmethod
    async def database_operation(operation: str, func: Callable, *args, **kwargs) -> Any:
        """Execute database operation with error handling"""
        return await ErrorHandler.safe_execute(
            "database",
            func,
            *args,
            **kwargs
        )
    
    @staticmethod
    async def ml_processing(model: str, func: Callable, *args, **kwargs) -> Any:
        """Execute ML processing with error handling"""
        return await ErrorHandler.safe_execute(
            "ml_processing",
            func,
            *args,
            **kwargs
        )
    
    @staticmethod
    async def external_api(api_name: str, func: Callable, *args, **kwargs) -> Any:
        """Execute external API call with error handling"""
        return await ErrorHandler.safe_execute(
            f"external_api_{api_name}",
            func,
            *args,
            **kwargs
        )
    
    @staticmethod
    async def file_operation(operation: str, func: Callable, *args, **kwargs) -> Any:
        """Execute file operation with error handling"""
        return await ErrorHandler.safe_execute(
            "file_operation",
            func,
            *args,
            **kwargs
        )

def validate_request_data(data: Dict[str, Any], required_fields: List[str]) -> None:
    """Validate request data has required fields"""
    missing_fields = [field for field in required_fields if field not in data or data[field] is None]
    if missing_fields:
        raise ValidationError(
            message=f"Missing required fields: {', '.join(missing_fields)}",
            field="request_data"
        )

def validate_numeric_range(value: Any, min_val: float = None, max_val: float = None, field_name: str = "value") -> None:
    """Validate numeric value is within range"""
    try:
        num_value = float(value)
    except (ValueError, TypeError):
        raise ValidationError(
            message=f"{field_name} must be a valid number",
            field=field_name,
            value=value
        )
    
    if min_val is not None and num_value < min_val:
        raise ValidationError(
            message=f"{field_name} must be >= {min_val}",
            field=field_name,
            value=value
        )
    
    if max_val is not None and num_value > max_val:
        raise ValidationError(
            message=f"{field_name} must be <= {max_val}",
            field=field_name,
            value=value
        )

def validate_string_length(value: str, min_length: int = None, max_length: int = None, field_name: str = "value") -> None:
    """Validate string length"""
    if not isinstance(value, str):
        raise ValidationError(
            message=f"{field_name} must be a string",
            field=field_name,
            value=value
        )
    
    if min_length is not None and len(value) < min_length:
        raise ValidationError(
            message=f"{field_name} must be at least {min_length} characters",
            field=field_name,
            value=value
        )
    
    if max_length is not None and len(value) > max_length:
        raise ValidationError(
            message=f"{field_name} must be no more than {max_length} characters",
            field=field_name,
            value=value
        )

def validate_uuid(value: str, field_name: str = "id") -> None:
    """Validate UUID format"""
    try:
        uuid.UUID(value)
    except ValueError:
        raise ValidationError(
            message=f"{field_name} must be a valid UUID",
            field=field_name,
            value=value
        )

def validate_json_data(data: Any, field_name: str = "data") -> None:
    """Validate JSON data structure"""
    if not isinstance(data, (dict, list)):
        raise ValidationError(
            message=f"{field_name} must be valid JSON (dict or list)",
            field=field_name,
            value=type(data).__name__
        )

class RateLimiter:
    """Simple rate limiter for API endpoints"""
    
    def __init__(self):
        self.requests = {}  # In production, use Redis
    
    def is_allowed(self, key: str, limit: int = 100, window: int = 3600) -> bool:
        """Check if request is allowed based on rate limit"""
        now = datetime.utcnow().timestamp()
        window_start = now - window
        
        # Clean old entries
        if key in self.requests:
            self.requests[key] = [req_time for req_time in self.requests[key] if req_time > window_start]
        else:
            self.requests[key] = []
        
        # Check limit
        if len(self.requests[key]) >= limit:
            return False
        
        # Add current request
        self.requests[key].append(now)
        return True

# Global rate limiter instance
rate_limiter = RateLimiter()

def rate_limit(limit: int = 100, window: int = 3600):
    """Decorator for rate limiting endpoints"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract user ID or IP for rate limiting
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if request:
                user_id = getattr(request.state, 'user_id', None)
                client_ip = request.client.host if request.client else "unknown"
                rate_key = f"{user_id or client_ip}"
                
                if not rate_limiter.is_allowed(rate_key, limit, window):
                    raise RateLimitError(
                        message=f"Rate limit exceeded. Maximum {limit} requests per {window} seconds",
                        retry_after=window
                    )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator
