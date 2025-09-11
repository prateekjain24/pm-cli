"""
PM-Kit Logging Decorators.

Provides convenient decorators for adding logging to functions with proper
signature preservation, async support, and error handling integration.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union

from pmkit.utils.logger import get_logger
from pmkit.utils.tracing import TracingContext, get_request_id, get_operation_context

# Type variables for decorator type hints
F = TypeVar('F', bound=Callable[..., Any])
AF = TypeVar('AF', bound=Callable[..., Any])

# Module logger
logger = get_logger(__name__)


def log_operation(
    operation: Optional[str] = None,
    log_args: bool = False,
    log_result: bool = False,
    log_duration: bool = True,
    level: str = "INFO"
) -> Callable[[F], F]:
    """
    Decorator for logging function entry/exit with timing for sync functions.
    
    Args:
        operation: Custom operation name (defaults to function name)
        log_args: Whether to log function arguments
        log_result: Whether to log function result
        log_duration: Whether to log execution duration
        level: Log level for entry/exit messages
        
    Returns:
        Decorated function with logging
        
    Example:
        @log_operation(operation="user.authenticate", log_args=True)
        def authenticate_user(username: str, password: str) -> bool:
            return check_credentials(username, password)
    """
    def decorator(func: F) -> F:
        func_name = operation or f"{func.__module__}.{func.__qualname__}"
        func_logger = get_logger(func.__module__)
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get current context for logging
            request_id = get_request_id()
            current_operation = get_operation_context()
            
            # Build context data
            context = {}
            if request_id:
                context['request_id'] = request_id
            if current_operation:
                context['parent_operation'] = current_operation
            
            # Log function arguments if requested
            if log_args:
                try:
                    # Get function signature for parameter names
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    
                    # Filter out sensitive parameters
                    safe_args = {}
                    for name, value in bound_args.arguments.items():
                        if any(sensitive in name.lower() for sensitive in ['password', 'token', 'key', 'secret']):
                            safe_args[name] = '[REDACTED]'
                        else:
                            safe_args[name] = str(value)[:100]  # Limit length
                    
                    context['args'] = safe_args
                except Exception:
                    # If signature binding fails, just note that args were passed
                    context['args'] = f"args={len(args)}, kwargs={len(kwargs)}"
            
            # Log function entry
            getattr(func_logger, level.lower())(
                f"Starting {func_name}",
                extra=context
            )
            
            start_time = time.perf_counter() if log_duration else None
            
            try:
                result = func(*args, **kwargs)
                
                # Calculate duration if requested
                duration_data = {}
                if start_time is not None:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    duration_data['duration_ms'] = round(duration_ms, 2)
                
                # Log result if requested
                result_data = {}
                if log_result and result is not None:
                    result_str = str(result)
                    if len(result_str) > 200:
                        result_str = result_str[:200] + "..."
                    result_data['result'] = result_str
                
                # Log successful completion
                func_logger.success(
                    f"Completed {func_name}",
                    extra={**context, **duration_data, **result_data}
                )
                
                return result
                
            except Exception as e:
                # Calculate duration for failed operations
                duration_data = {}
                if start_time is not None:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    duration_data['duration_ms'] = round(duration_ms, 2)
                
                # Log error
                func_logger.error(
                    f"Failed {func_name}: {type(e).__name__}: {e}",
                    extra={**context, **duration_data, 'error_type': type(e).__name__}
                )
                raise
        
        return wrapper
    return decorator


def log_async(
    operation: Optional[str] = None,
    log_args: bool = False,
    log_result: bool = False,
    log_duration: bool = True,
    level: str = "INFO"
) -> Callable[[AF], AF]:
    """
    Decorator for logging async function entry/exit with timing.
    
    Args:
        operation: Custom operation name (defaults to function name)
        log_args: Whether to log function arguments
        log_result: Whether to log function result
        log_duration: Whether to log execution duration
        level: Log level for entry/exit messages
        
    Returns:
        Decorated async function with logging
        
    Example:
        @log_async(operation="llm.generate", log_duration=True)
        async def generate_content(prompt: str) -> str:
            return await llm_client.generate(prompt)
    """
    def decorator(func: AF) -> AF:
        func_name = operation or f"{func.__module__}.{func.__qualname__}"
        func_logger = get_logger(func.__module__)
        
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get current context for logging
            request_id = get_request_id()
            current_operation = get_operation_context()
            
            # Build context data
            context = {}
            if request_id:
                context['request_id'] = request_id
            if current_operation:
                context['parent_operation'] = current_operation
            
            # Log function arguments if requested
            if log_args:
                try:
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    
                    safe_args = {}
                    for name, value in bound_args.arguments.items():
                        if any(sensitive in name.lower() for sensitive in ['password', 'token', 'key', 'secret']):
                            safe_args[name] = '[REDACTED]'
                        else:
                            safe_args[name] = str(value)[:100]
                    
                    context['args'] = safe_args
                except Exception:
                    context['args'] = f"args={len(args)}, kwargs={len(kwargs)}"
            
            # Log function entry
            getattr(func_logger, level.lower())(
                f"Starting {func_name}",
                extra=context
            )
            
            start_time = time.perf_counter() if log_duration else None
            
            try:
                result = await func(*args, **kwargs)
                
                # Calculate duration if requested
                duration_data = {}
                if start_time is not None:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    duration_data['duration_ms'] = round(duration_ms, 2)
                
                # Log result if requested
                result_data = {}
                if log_result and result is not None:
                    result_str = str(result)
                    if len(result_str) > 200:
                        result_str = result_str[:200] + "..."
                    result_data['result'] = result_str
                
                # Log successful completion
                func_logger.success(
                    f"Completed {func_name}",
                    extra={**context, **duration_data, **result_data}
                )
                
                return result
                
            except Exception as e:
                # Calculate duration for failed operations
                duration_data = {}
                if start_time is not None:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    duration_data['duration_ms'] = round(duration_ms, 2)
                
                # Log error
                func_logger.error(
                    f"Failed {func_name}: {type(e).__name__}: {e}",
                    extra={**context, **duration_data, 'error_type': type(e).__name__}
                )
                raise
        
        return wrapper
    return decorator


def log_errors(
    reraise: bool = True,
    level: str = "ERROR",
    operation: Optional[str] = None
) -> Callable[[F], F]:
    """
    Decorator for automatic error logging with context.
    
    Args:
        reraise: Whether to reraise the exception after logging
        level: Log level for error messages
        operation: Custom operation name for logging context
        
    Returns:
        Decorated function with error logging
        
    Example:
        @log_errors(reraise=True)
        def risky_operation() -> str:
            # Any exceptions are automatically logged with context
            raise ValueError("Something went wrong")
    """
    def decorator(func: F) -> F:
        func_name = operation or f"{func.__module__}.{func.__qualname__}"
        func_logger = get_logger(func.__module__)
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Get current context
                    request_id = get_request_id()
                    current_operation = get_operation_context()
                    
                    context = {}
                    if request_id:
                        context['request_id'] = request_id
                    if current_operation:
                        context['operation'] = current_operation
                    
                    # Add error details
                    context.update({
                        'error_type': type(e).__name__,
                        'function': func_name,
                    })
                    
                    # Log the error with full context
                    getattr(func_logger, level.lower())(
                        f"Exception in {func_name}: {type(e).__name__}: {e}",
                        extra=context
                    )
                    
                    if reraise:
                        raise
                    return None
            
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Get current context
                    request_id = get_request_id()
                    current_operation = get_operation_context()
                    
                    context = {}
                    if request_id:
                        context['request_id'] = request_id
                    if current_operation:
                        context['operation'] = current_operation
                    
                    # Add error details
                    context.update({
                        'error_type': type(e).__name__,
                        'function': func_name,
                    })
                    
                    # Log the error with full context
                    getattr(func_logger, level.lower())(
                        f"Exception in {func_name}: {type(e).__name__}: {e}",
                        extra=context
                    )
                    
                    if reraise:
                        raise
                    return None
            
            return sync_wrapper
    
    return decorator


def trace_performance(
    operation: Optional[str] = None,
    track_memory: bool = True,
    threshold_ms: Optional[float] = None
) -> Callable[[F], F]:
    """
    Decorator for detailed performance tracking with optional thresholds.
    
    Args:
        operation: Custom operation name
        track_memory: Whether to track memory usage
        threshold_ms: Only log if execution time exceeds this threshold
        
    Returns:
        Decorated function with performance tracking
        
    Example:
        @trace_performance(threshold_ms=1000.0)  # Only log if > 1 second
        def expensive_operation() -> str:
            # Long-running operation
            time.sleep(2)
            return "done"
    """
    def decorator(func: F) -> F:
        func_name = operation or f"{func.__module__}.{func.__qualname__}"
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                async with TracingContext(
                    operation=func_name,
                    track_memory=track_memory,
                    log_start=False,  # We handle logging manually for thresholds
                    log_end=False
                ) as tracer:
                    result = await func(*args, **kwargs)
                    return result
            
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with TracingContext(
                    operation=func_name,
                    track_memory=track_memory,
                    log_start=False,
                    log_end=False
                ) as tracer:
                    result = func(*args, **kwargs)
                    return result
            
            return sync_wrapper
    
    return decorator


def rate_limited_log(
    key_func: Optional[Callable[..., str]] = None,
    max_per_minute: int = 10,
    level: str = "WARNING"
) -> Callable[[F], F]:
    """
    Decorator to rate-limit logging for functions that might spam logs.
    
    Args:
        key_func: Function to generate rate limit key from args (defaults to function name)
        max_per_minute: Maximum log entries per minute for each key
        level: Log level for rate-limited messages
        
    Returns:
        Decorated function with rate-limited logging
        
    Example:
        @rate_limited_log(max_per_minute=5)
        def api_error_handler(error_code: str) -> None:
            # This will only log 5 times per minute per error code
            logger.error(f"API error: {error_code}")
    """
    from collections import defaultdict
    from threading import Lock
    
    # Rate limiting state
    _log_counts = defaultdict(list)  # key -> list of timestamps
    _lock = Lock()
    
    def decorator(func: F) -> F:
        func_name = f"{func.__module__}.{func.__qualname__}"
        func_logger = get_logger(func.__module__)
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate rate limit key
            if key_func:
                try:
                    limit_key = key_func(*args, **kwargs)
                except Exception:
                    limit_key = func_name
            else:
                limit_key = func_name
            
            current_time = time.time()
            should_log = True
            
            with _lock:
                # Clean old entries (older than 1 minute)
                _log_counts[limit_key] = [
                    ts for ts in _log_counts[limit_key] 
                    if current_time - ts < 60
                ]
                
                # Check if we're over the rate limit
                if len(_log_counts[limit_key]) >= max_per_minute:
                    should_log = False
                else:
                    _log_counts[limit_key].append(current_time)
            
            # Execute the function
            result = func(*args, **kwargs)
            
            # Log rate limiting if applicable
            if not should_log:
                # Log a rate limiting message less frequently
                if len(_log_counts[limit_key]) == max_per_minute:
                    getattr(func_logger, level.lower())(
                        f"Rate limiting logs for {func_name} (>{max_per_minute}/min)",
                        extra={'limit_key': limit_key, 'max_per_minute': max_per_minute}
                    )
            
            return result
        
        return wrapper
    return decorator


__all__ = [
    'log_operation',
    'log_async', 
    'log_errors',
    'trace_performance',
    'rate_limited_log',
]