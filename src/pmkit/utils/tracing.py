"""
PM-Kit Request Tracing and Context Injection Module.

Provides unique request IDs, context injection for operations, performance 
tracking, and async-safe context management for structured logging.
"""

from __future__ import annotations

import asyncio
import psutil
import time
import uuid
from contextlib import contextmanager, asynccontextmanager
from contextvars import ContextVar, copy_context
from functools import wraps
from typing import Any, AsyncIterator, Callable, Dict, Iterator, Optional, TypeVar, Union

from pmkit.utils.logger import get_logger

# Type variables for decorators
F = TypeVar('F', bound=Callable[..., Any])
AF = TypeVar('AF', bound=Callable[..., Any])

# Context variables for request tracing
_request_id: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
_operation_context: ContextVar[Optional[str]] = ContextVar('operation_context', default=None)
_trace_data: ContextVar[Dict[str, Any]] = ContextVar('trace_data', default_factory=dict)

# Module logger
logger = get_logger(__name__)


def generate_request_id() -> str:
    """
    Generate a unique request ID for tracing.
    
    Returns:
        8-character hex string for compact logging
    """
    return uuid.uuid4().hex[:8]


def get_request_id() -> Optional[str]:
    """
    Get the current request ID from context.
    
    Returns:
        Current request ID or None if not set
    """
    return _request_id.get()


def get_operation_context() -> Optional[str]:
    """
    Get the current operation context.
    
    Returns:
        Current operation context or None if not set
    """
    return _operation_context.get()


def get_trace_data() -> Dict[str, Any]:
    """
    Get the current trace data dictionary.
    
    Returns:
        Dictionary of trace data for the current context
    """
    return _trace_data.get().copy()


def set_trace_data(**data: Any) -> None:
    """
    Set trace data for the current context.
    
    Args:
        **data: Key-value pairs to add to trace data
    """
    current_data = _trace_data.get().copy()
    current_data.update(data)
    _trace_data.set(current_data)


@contextmanager
def request_context(
    request_id: Optional[str] = None, 
    operation: Optional[str] = None,
    **trace_data: Any
) -> Iterator[str]:
    """
    Context manager for request tracing with automatic cleanup.
    
    Args:
        request_id: Custom request ID (generates one if None)
        operation: Operation name for logging context
        **trace_data: Additional trace data to attach
        
    Yields:
        The request ID for the context
        
    Example:
        with request_context(operation="prd.generate") as req_id:
            logger.info("Starting PRD generation")
            # All logs will include request_id and operation context
    """
    req_id = request_id or generate_request_id()
    
    # Save current context
    old_request_id = _request_id.get()
    old_operation = _operation_context.get()
    old_trace_data = _trace_data.get()
    
    try:
        # Set new context
        _request_id.set(req_id)
        if operation:
            _operation_context.set(operation)
        
        # Initialize trace data
        new_trace_data = trace_data.copy()
        new_trace_data.update({
            'request_id': req_id,
            'operation': operation,
        })
        _trace_data.set(new_trace_data)
        
        logger.debug(
            f"Started request context: {operation or 'operation'}",
            extra=new_trace_data
        )
        
        yield req_id
        
    finally:
        # Restore previous context
        _request_id.set(old_request_id)
        _operation_context.set(old_operation)
        _trace_data.set(old_trace_data)


@asynccontextmanager
async def async_request_context(
    request_id: Optional[str] = None,
    operation: Optional[str] = None,
    **trace_data: Any
) -> AsyncIterator[str]:
    """
    Async context manager for request tracing.
    
    Args:
        request_id: Custom request ID (generates one if None)
        operation: Operation name for logging context
        **trace_data: Additional trace data to attach
        
    Yields:
        The request ID for the context
        
    Example:
        async with async_request_context(operation="llm.generate") as req_id:
            result = await llm_client.generate(prompt)
            # All async operations inherit the context
    """
    req_id = request_id or generate_request_id()
    
    # Use regular context manager within async context
    with request_context(req_id, operation, **trace_data):
        yield req_id


class TracingContext:
    """
    Advanced context manager for operation tracking with performance metrics.
    
    Tracks execution time, memory usage, and provides structured logging
    with automatic success/failure detection.
    """
    
    def __init__(
        self,
        operation: str,
        request_id: Optional[str] = None,
        track_memory: bool = True,
        log_start: bool = True,
        log_end: bool = True,
        **context_data: Any
    ):
        """
        Initialize tracing context.
        
        Args:
            operation: Name of the operation being tracked
            request_id: Custom request ID (generates if None)
            track_memory: Whether to track memory usage
            log_start: Whether to log operation start
            log_end: Whether to log operation completion
            **context_data: Additional context data
        """
        self.operation = operation
        self.request_id = request_id or generate_request_id()
        self.track_memory = track_memory
        self.log_start = log_start
        self.log_end = log_end
        self.context_data = context_data
        
        self.start_time: Optional[float] = None
        self.start_memory: Optional[float] = None
        self.logger = get_logger(__name__).bind(
            request_id=self.request_id,
            operation=self.operation
        )
    
    def __enter__(self) -> "TracingContext":
        """Enter the tracing context."""
        self.start_time = time.perf_counter()
        
        if self.track_memory:
            try:
                process = psutil.Process()
                self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
            except (psutil.Error, OSError):
                self.start_memory = None
        
        # Set context variables
        _request_id.set(self.request_id)
        _operation_context.set(self.operation)
        
        trace_data = {
            'request_id': self.request_id,
            'operation': self.operation,
            **self.context_data
        }
        _trace_data.set(trace_data)
        
        if self.log_start:
            self.logger.info(f"Starting {self.operation}", extra=trace_data)
        
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the tracing context with performance logging."""
        duration_ms = None
        memory_delta = None
        
        if self.start_time:
            duration_ms = (time.perf_counter() - self.start_time) * 1000
        
        if self.track_memory and self.start_memory:
            try:
                process = psutil.Process()
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_delta = current_memory - self.start_memory
            except (psutil.Error, OSError):
                pass
        
        # Prepare metrics
        metrics = {}
        if duration_ms is not None:
            metrics['duration_ms'] = round(duration_ms, 2)
        if memory_delta is not None:
            metrics['memory_delta_mb'] = round(memory_delta, 2)
        
        if self.log_end:
            if exc_type is None:
                self.logger.success(
                    f"Completed {self.operation}",
                    extra=metrics
                )
            else:
                self.logger.error(
                    f"Failed {self.operation}: {exc_val}",
                    extra=metrics
                )
    
    async def __aenter__(self) -> "TracingContext":
        """Async enter method."""
        return self.__enter__()
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async exit method."""
        self.__exit__(exc_type, exc_val, exc_tb)


def with_context(
    operation: Optional[str] = None,
    request_id: Optional[str] = None,
    track_performance: bool = True,
    **context_data: Any
) -> Callable[[F], F]:
    """
    Decorator to add tracing context to sync functions.
    
    Args:
        operation: Operation name (defaults to function name)
        request_id: Custom request ID (generates if None)
        track_performance: Whether to track execution time and memory
        **context_data: Additional context data
        
    Returns:
        Decorated function with tracing context
        
    Example:
        @with_context(operation="user.authenticate")
        def authenticate_user(username: str) -> bool:
            # Function automatically has tracing context
            logger.info(f"Authenticating {username}")
            return True
    """
    def decorator(func: F) -> F:
        op_name = operation or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with TracingContext(
                operation=op_name,
                request_id=request_id,
                track_memory=track_performance,
                **context_data
            ):
                return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def with_async_context(
    operation: Optional[str] = None,
    request_id: Optional[str] = None,
    track_performance: bool = True,
    **context_data: Any
) -> Callable[[AF], AF]:
    """
    Decorator to add tracing context to async functions.
    
    Args:
        operation: Operation name (defaults to function name)
        request_id: Custom request ID (generates if None)
        track_performance: Whether to track execution time and memory
        **context_data: Additional context data
        
    Returns:
        Decorated async function with tracing context
        
    Example:
        @with_async_context(operation="llm.generate")
        async def generate_prd(prompt: str) -> str:
            # Async function has tracing context
            logger.info("Generating PRD with LLM")
            return await llm_client.generate(prompt)
    """
    def decorator(func: AF) -> AF:
        op_name = operation or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with TracingContext(
                operation=op_name,
                request_id=request_id,
                track_memory=track_performance,
                **context_data
            ):
                return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def copy_context_to_thread() -> Dict[str, Any]:
    """
    Copy current context for thread propagation.
    
    Returns:
        Dictionary containing current context state
    """
    return {
        'request_id': get_request_id(),
        'operation_context': get_operation_context(),
        'trace_data': get_trace_data(),
    }


def restore_context_from_dict(context_dict: Dict[str, Any]) -> None:
    """
    Restore context from dictionary (for thread propagation).
    
    Args:
        context_dict: Context state dictionary from copy_context_to_thread()
    """
    if context_dict.get('request_id'):
        _request_id.set(context_dict['request_id'])
    if context_dict.get('operation_context'):
        _operation_context.set(context_dict['operation_context'])
    if context_dict.get('trace_data'):
        _trace_data.set(context_dict['trace_data'])


def get_logger_with_context(name: str) -> Any:
    """
    Get a logger that automatically includes current tracing context.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger with current context bound
    """
    base_logger = get_logger(name)
    
    # Get current context
    request_id = get_request_id()
    operation = get_operation_context()
    trace_data = get_trace_data()
    
    # Bind context to logger
    context = {}
    if request_id:
        context['request_id'] = request_id
    if operation:
        context['operation'] = operation
    
    # Add other trace data (excluding duplicates)
    for key, value in trace_data.items():
        if key not in ('request_id', 'operation'):
            context[key] = value
    
    return base_logger.bind(**context) if context else base_logger


__all__ = [
    'generate_request_id',
    'get_request_id', 
    'get_operation_context',
    'get_trace_data',
    'set_trace_data',
    'request_context',
    'async_request_context',
    'TracingContext',
    'with_context',
    'with_async_context',
    'copy_context_to_thread',
    'restore_context_from_dict',
    'get_logger_with_context',
]