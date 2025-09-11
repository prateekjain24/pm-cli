"""
PM-Kit Async Utilities.

Simple async/sync bridge utilities for MVP.
Focuses on the essentials: running async from sync, retries, and timeouts.
"""

import asyncio
import functools
from typing import TypeVar, Callable, Any, Optional, Coroutine
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from pmkit.utils.logger import get_logger

# Type variables
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

# Module logger
logger = get_logger(__name__)


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async function from sync context (e.g., CLI commands).
    
    This is the bridge between Typer CLI commands and async LLM operations.
    
    Args:
        coro: The coroutine to run
        
    Returns:
        The result of the coroutine
        
    Example:
        # In a CLI command
        result = run_async(llm_client.generate(prompt))
    """
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop running, create a new one
        return asyncio.run(coro)
    else:
        # Loop is already running (shouldn't happen in CLI, but safe fallback)
        import nest_asyncio
        nest_asyncio.apply()
        return asyncio.run(coro)


def retry_with_backoff(
    max_attempts: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 10.0,
    multiplier: float = 2.0,
    retry_on: Optional[tuple[type[Exception], ...]] = None,
) -> Callable[[F], F]:
    """
    Simple retry decorator with exponential backoff for LLM API calls.
    
    Args:
        max_attempts: Maximum number of retry attempts
        initial_wait: Initial wait time in seconds
        max_wait: Maximum wait time in seconds
        multiplier: Multiplier for exponential backoff
        retry_on: Tuple of exception types to retry on (defaults to LLMError)
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @retry_with_backoff(max_attempts=3)
        async def call_openai(prompt: str) -> str:
            return await openai_client.generate(prompt)
    """
    if retry_on is None:
        # Default to common network/API errors
        # LLMError will be imported at runtime to avoid circular import
        retry_on = (ConnectionError, TimeoutError)
    
    def decorator(func: F) -> F:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(
                multiplier=multiplier,
                min=initial_wait,
                max=max_wait
            ),
            retry=retry_if_exception_type(retry_on),
            before_sleep=before_sleep_log(logger, "INFO"),
            reraise=True
        )
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return await func(*args, **kwargs)
        
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(
                multiplier=multiplier,
                min=initial_wait,
                max=max_wait
            ),
            retry=retry_if_exception_type(retry_on),
            before_sleep=before_sleep_log(logger, "INFO"),
            reraise=True
        )
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def timeout(seconds: float) -> Callable[[F], F]:
    """
    Simple timeout decorator for both sync and async functions.
    
    Args:
        seconds: Timeout duration in seconds
        
    Returns:
        Decorated function with timeout
        
    Example:
        @timeout(30.0)
        async def slow_api_call() -> str:
            # This will timeout after 30 seconds
            return await external_api.fetch()
    """
    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=seconds
                    )
                except asyncio.TimeoutError:
                    func_name = f"{func.__module__}.{func.__qualname__}"
                    logger.error(f"Timeout after {seconds}s: {func_name}")
                    raise TimeoutError(f"Operation timed out after {seconds} seconds")
            return async_wrapper
        else:
            # For sync functions, we'll use asyncio with a thread
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(func, *args, **kwargs)
                    try:
                        return future.result(timeout=seconds)
                    except concurrent.futures.TimeoutError:
                        func_name = f"{func.__module__}.{func.__qualname__}"
                        logger.error(f"Timeout after {seconds}s: {func_name}")
                        raise TimeoutError(f"Operation timed out after {seconds} seconds")
            return sync_wrapper
    
    return decorator


def ensure_async(func: Callable[..., T]) -> Callable[..., Coroutine[Any, Any, T]]:
    """
    Convert a sync function to async if needed.
    
    Useful for APIs that might be sync or async depending on the provider.
    
    Args:
        func: Function that might be sync or async
        
    Returns:
        An async function
        
    Example:
        # Works with both sync and async functions
        async_func = ensure_async(some_function)
        result = await async_func()
    """
    if asyncio.iscoroutinefunction(func):
        return func
    
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        # Run sync function in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        # Use partial to properly handle kwargs
        from functools import partial
        return await loop.run_in_executor(None, partial(func, *args, **kwargs))
    
    return wrapper


__all__ = [
    'run_async',
    'retry_with_backoff',
    'timeout',
    'ensure_async',
]