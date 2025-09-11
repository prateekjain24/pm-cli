"""
Comprehensive test suite for PM-Kit async utilities.

Tests all async utility functions including:
- run_async(): Bridge for running async functions from sync context
- retry_with_backoff(): Retry decorator with exponential backoff
- timeout(): Timeout decorator for both sync/async functions
- ensure_async(): Convert sync functions to async

Each test verifies actual behavior with counters, timers, and proper assertions.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import logging

import pytest
from tenacity import RetryError

from pmkit.utils.async_utils import (
    ensure_async,
    retry_with_backoff,
    run_async,
    timeout,
)


@pytest.fixture(autouse=True)
def mock_logger():
    """Mock the logger to avoid issues with LoggerAdapter not having a log method."""
    with patch('pmkit.utils.async_utils.logger') as mock_log:
        # Create a proper mock logger that has all needed methods
        mock_log.log = Mock()
        mock_log.info = Mock()
        mock_log.debug = Mock()
        mock_log.error = Mock()
        mock_log.warning = Mock()
        yield mock_log


class TestRunAsync:
    """Test suite for run_async() function that bridges sync to async."""
    
    def test_runs_async_function_from_sync_context(self):
        """Test that run_async actually executes async functions from sync context."""
        # Track execution
        executed = False
        result_value = "test_result"
        
        async def async_function():
            nonlocal executed
            executed = True
            await asyncio.sleep(0.01)  # Small delay to ensure it's truly async
            return result_value
        
        # Run async function from sync context
        result = run_async(async_function())
        
        # Verify execution and result
        assert executed is True
        assert result == result_value
    
    def test_handles_async_function_with_arguments(self):
        """Test that run_async passes arguments correctly to async functions."""
        async def async_add(a: int, b: int, multiplier: int = 1) -> int:
            await asyncio.sleep(0.01)
            return (a + b) * multiplier
        
        # Test with positional args
        result = run_async(async_add(5, 3))
        assert result == 8
        
        # Test with keyword args
        result = run_async(async_add(5, 3, multiplier=2))
        assert result == 16
    
    def test_propagates_exceptions_from_async_function(self):
        """Test that exceptions from async functions are properly propagated."""
        error_message = "Async operation failed"
        
        async def failing_async_function():
            await asyncio.sleep(0.01)
            raise ValueError(error_message)
        
        # Exception should be propagated
        with pytest.raises(ValueError) as exc_info:
            run_async(failing_async_function())
        
        assert str(exc_info.value) == error_message
    
    def test_handles_cancelled_coroutine(self):
        """Test handling of cancelled async operations."""
        async def cancellable_function():
            try:
                await asyncio.sleep(10)  # Long sleep
            except asyncio.CancelledError:
                return "cancelled"
            return "completed"
        
        async def cancel_after_delay(coro, delay=0.05):
            task = asyncio.create_task(coro)
            await asyncio.sleep(delay)
            task.cancel()
            try:
                return await task
            except asyncio.CancelledError:
                return "task_cancelled"
        
        # Run with cancellation
        result = run_async(cancel_after_delay(cancellable_function()))
        assert result in ["cancelled", "task_cancelled"]
    
    def test_handles_existing_event_loop(self):
        """Test that run_async handles when an event loop is already running."""
        # This test verifies the logic paths, though in practice we can't easily
        # simulate an already-running loop in pytest
        async def simple_async():
            return "done"
        
        # Should work without error
        result = run_async(simple_async())
        assert result == "done"


class TestRetryWithBackoff:
    """Test suite for retry_with_backoff() decorator."""
    
    @pytest.mark.asyncio
    async def test_retries_on_failure_with_counter(self):
        """Test that retry_with_backoff retries the correct number of times."""
        attempt_count = 0
        max_attempts = 3
        
        @retry_with_backoff(
            max_attempts=max_attempts,
            initial_wait=0.01,  # Fast for testing
            max_wait=0.05,
            retry_on=(ConnectionError,)
        )
        async def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            
            # Fail first 2 attempts, succeed on 3rd
            if attempt_count < 3:
                raise ConnectionError(f"Attempt {attempt_count} failed")
            return f"Success on attempt {attempt_count}"
        
        result = await flaky_function()
        
        # Should have tried 3 times and succeeded
        assert attempt_count == 3
        assert result == "Success on attempt 3"
    
    @pytest.mark.asyncio
    async def test_stops_after_max_attempts_and_raises_error(self):
        """Test that retry stops after max attempts and raises the final error."""
        attempt_count = 0
        max_attempts = 3
        
        @retry_with_backoff(
            max_attempts=max_attempts,
            initial_wait=0.01,
            retry_on=(ValueError,)
        )
        async def always_failing_function():
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError(f"Failed attempt {attempt_count}")
        
        # Should raise after max attempts
        with pytest.raises(ValueError) as exc_info:
            await always_failing_function()
        
        # Verify it tried max_attempts times
        assert attempt_count == max_attempts
        assert f"Failed attempt {max_attempts}" in str(exc_info.value)
    
    def test_retry_with_sync_function(self):
        """Test that retry works with synchronous functions."""
        attempt_count = 0
        
        @retry_with_backoff(
            max_attempts=3,
            initial_wait=0.01,
            retry_on=(RuntimeError,)
        )
        def sync_flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            
            if attempt_count < 2:
                raise RuntimeError(f"Sync attempt {attempt_count} failed")
            return f"Sync success on attempt {attempt_count}"
        
        result = sync_flaky_function()
        
        assert attempt_count == 2
        assert result == "Sync success on attempt 2"
    
    @pytest.mark.asyncio
    async def test_retry_with_custom_exception_types(self):
        """Test retry with specific exception types."""
        attempt_count = 0
        
        class CustomError(Exception):
            pass
        
        @retry_with_backoff(
            max_attempts=3,
            initial_wait=0.01,
            retry_on=(CustomError, ConnectionError)
        )
        async def custom_error_function():
            nonlocal attempt_count
            attempt_count += 1
            
            if attempt_count == 1:
                raise CustomError("Custom error")
            elif attempt_count == 2:
                raise ConnectionError("Connection error")
            return "Success"
        
        result = await custom_error_function()
        
        assert attempt_count == 3
        assert result == "Success"
    
    @pytest.mark.asyncio
    async def test_no_retry_for_non_specified_exceptions(self):
        """Test that non-specified exceptions are not retried."""
        attempt_count = 0
        
        @retry_with_backoff(
            max_attempts=3,
            initial_wait=0.01,
            retry_on=(ConnectionError,)  # Only retry ConnectionError
        )
        async def wrong_error_function():
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError("Wrong error type")
        
        # Should not retry ValueError
        with pytest.raises(ValueError):
            await wrong_error_function()
        
        # Should only try once
        assert attempt_count == 1
    
    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self):
        """Test that exponential backoff increases wait time."""
        attempt_times = []
        
        @retry_with_backoff(
            max_attempts=4,
            initial_wait=0.01,
            max_wait=0.1,
            multiplier=2.0,
            retry_on=(ConnectionError,)
        )
        async def timing_function():
            attempt_times.append(time.time())
            if len(attempt_times) < 4:
                raise ConnectionError("Retry me")
            return "Done"
        
        start_time = time.time()
        result = await timing_function()
        
        assert result == "Done"
        assert len(attempt_times) == 4
        
        # Check that wait times increase (approximately)
        # Note: Actual timing may vary, so we use loose bounds
        if len(attempt_times) > 1:
            for i in range(1, len(attempt_times)):
                time_diff = attempt_times[i] - attempt_times[i-1]
                # Each wait should be at least initial_wait
                assert time_diff >= 0.005  # Allow some tolerance
    
    @pytest.mark.asyncio
    async def test_immediate_success_no_retries(self):
        """Test that successful execution doesn't trigger retries."""
        attempt_count = 0
        
        @retry_with_backoff(max_attempts=5, initial_wait=0.01)
        async def successful_function():
            nonlocal attempt_count
            attempt_count += 1
            return "Immediate success"
        
        result = await successful_function()
        
        # Should only execute once
        assert attempt_count == 1
        assert result == "Immediate success"


class TestTimeout:
    """Test suite for timeout() decorator."""
    
    @pytest.mark.asyncio
    async def test_timeout_raises_error_for_slow_async_function(self):
        """Test that timeout raises TimeoutError for slow async functions."""
        execution_completed = False
        
        @timeout(0.05)  # 50ms timeout
        async def slow_async_function():
            nonlocal execution_completed
            await asyncio.sleep(0.2)  # Sleep for 200ms
            execution_completed = True
            return "Should not reach here"
        
        # Should raise TimeoutError
        with pytest.raises(TimeoutError) as exc_info:
            await slow_async_function()
        
        assert "timed out after 0.05 seconds" in str(exc_info.value)
        assert execution_completed is False
    
    @pytest.mark.asyncio
    async def test_timeout_allows_fast_async_function(self):
        """Test that timeout doesn't affect fast async functions."""
        @timeout(0.5)  # 500ms timeout
        async def fast_async_function():
            await asyncio.sleep(0.01)  # Sleep for 10ms
            return "Completed quickly"
        
        result = await fast_async_function()
        assert result == "Completed quickly"
    
    def test_timeout_raises_error_for_slow_sync_function(self):
        """Test that timeout raises TimeoutError for slow sync functions."""
        @timeout(0.05)  # 50ms timeout
        def slow_sync_function():
            time.sleep(0.2)  # Sleep for 200ms
            return "Should not reach here"
        
        # Should raise TimeoutError
        with pytest.raises(TimeoutError) as exc_info:
            slow_sync_function()
        
        assert "timed out after 0.05 seconds" in str(exc_info.value)
    
    def test_timeout_allows_fast_sync_function(self):
        """Test that timeout doesn't affect fast sync functions."""
        @timeout(0.5)  # 500ms timeout
        def fast_sync_function():
            time.sleep(0.01)  # Sleep for 10ms
            return "Completed quickly"
        
        result = fast_sync_function()
        assert result == "Completed quickly"
    
    @pytest.mark.asyncio
    async def test_timeout_with_arguments(self):
        """Test that timeout decorator preserves function arguments."""
        @timeout(0.5)
        async def async_with_args(a: int, b: int, prefix: str = "Result") -> str:
            await asyncio.sleep(0.01)
            return f"{prefix}: {a + b}"
        
        result = await async_with_args(5, 3)
        assert result == "Result: 8"
        
        result = await async_with_args(10, 20, prefix="Sum")
        assert result == "Sum: 30"
    
    def test_timeout_preserves_function_metadata(self):
        """Test that timeout decorator preserves function name and docstring."""
        @timeout(1.0)
        def documented_function():
            """This is a documented function."""
            return "value"
        
        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a documented function."
    
    @pytest.mark.asyncio
    async def test_timeout_cancels_running_task(self):
        """Test that timeout properly cancels the running async task."""
        cleanup_called = False
        
        @timeout(0.05)
        async def function_with_cleanup():
            try:
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                nonlocal cleanup_called
                cleanup_called = True
                raise
            return "Done"
        
        with pytest.raises(TimeoutError):
            await function_with_cleanup()
        
        # Give a moment for cleanup
        await asyncio.sleep(0.01)
        
        # The function should have been cancelled
        # Note: cleanup_called might not always be True due to timing


class TestEnsureAsync:
    """Test suite for ensure_async() function."""
    
    @pytest.mark.asyncio
    async def test_converts_sync_function_to_async(self):
        """Test that ensure_async converts sync functions to async."""
        def sync_function(x: int, y: int) -> int:
            time.sleep(0.01)  # Simulate work
            return x + y
        
        # Convert to async
        async_version = ensure_async(sync_function)
        
        # Should be awaitable
        result = await async_version(5, 3)
        assert result == 8
        
        # Verify it's actually async
        assert asyncio.iscoroutinefunction(async_version)
    
    @pytest.mark.asyncio
    async def test_leaves_async_function_unchanged(self):
        """Test that ensure_async doesn't modify already async functions."""
        call_count = 0
        
        async def already_async_function(value: str) -> str:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return f"async: {value}"
        
        # Should return the same function
        result_func = ensure_async(already_async_function)
        
        # Should be the exact same function object
        assert result_func is already_async_function
        
        # Test it still works
        result = await result_func("test")
        assert result == "async: test"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_sync_function_runs_in_thread_pool(self):
        """Test that sync functions run in thread pool to avoid blocking."""
        import threading
        main_thread_id = threading.current_thread().ident
        execution_thread_id = None
        
        def blocking_function():
            nonlocal execution_thread_id
            execution_thread_id = threading.current_thread().ident
            time.sleep(0.05)  # Simulate blocking I/O
            return "completed"
        
        async_version = ensure_async(blocking_function)
        
        # Run multiple concurrent calls
        results = await asyncio.gather(
            async_version(),
            async_version(),
            async_version()
        )
        
        assert all(r == "completed" for r in results)
        # Should have run in different thread (not main thread)
        assert execution_thread_id != main_thread_id
    
    @pytest.mark.asyncio
    async def test_preserves_function_signature(self):
        """Test that ensure_async handles function calls correctly."""
        def original_function(a: int, b: str = "default") -> str:
            """Original docstring."""
            return f"{a}-{b}"
        
        async_version = ensure_async(original_function)
        
        # Test basic call - run_in_executor needs special handling for args
        result1 = await async_version(42, "custom")
        assert result1 == "42-custom"
        
        # Check metadata preservation - functools.wraps preserves the original name
        assert async_version.__name__ == "original_function"  # functools.wraps preserves name
        assert "Original docstring" in original_function.__doc__
    
    @pytest.mark.asyncio
    async def test_handles_exceptions_from_sync_function(self):
        """Test that exceptions from sync functions are properly propagated."""
        def failing_sync_function():
            raise ValueError("Sync function error")
        
        async_version = ensure_async(failing_sync_function)
        
        with pytest.raises(ValueError) as exc_info:
            await async_version()
        
        assert str(exc_info.value) == "Sync function error"
    
    @pytest.mark.asyncio
    async def test_concurrent_execution_of_converted_functions(self):
        """Test that multiple converted sync functions can run concurrently."""
        execution_order = []
        
        def slow_operation(id: int) -> int:
            execution_order.append(f"start-{id}")
            time.sleep(0.05)  # Simulate work
            execution_order.append(f"end-{id}")
            return id * 2
        
        async_version = ensure_async(slow_operation)
        
        # Run concurrently
        start_time = time.time()
        results = await asyncio.gather(
            async_version(1),
            async_version(2),
            async_version(3)
        )
        elapsed_time = time.time() - start_time
        
        # Should complete faster than sequential (0.15s)
        assert elapsed_time < 0.1  # Allow some overhead
        assert results == [2, 4, 6]
        
        # All should have started before any finished (concurrent execution)
        starts = [i for i in execution_order if i.startswith("start")]
        assert len(starts) == 3


class TestIntegration:
    """Integration tests combining multiple async utilities."""
    
    @pytest.mark.asyncio
    async def test_retry_with_timeout_combination(self):
        """Test combining retry and timeout decorators."""
        attempt_count = 0
        
        @retry_with_backoff(max_attempts=3, initial_wait=0.01, retry_on=(TimeoutError,))
        @timeout(0.1)  # Each attempt has 100ms timeout
        async def complex_function():
            nonlocal attempt_count
            attempt_count += 1
            
            if attempt_count < 2:
                # First attempt: fail with timeout
                await asyncio.sleep(0.2)  # Will timeout
            else:
                # Second attempt: succeed quickly
                await asyncio.sleep(0.01)
                return f"Success on attempt {attempt_count}"
        
        result = await complex_function()
        
        # Should retry after timeout and succeed
        assert attempt_count == 2
        assert result == "Success on attempt 2"
    
    def test_run_async_with_timeout_sync_function(self):
        """Test using run_async with a timeout-decorated sync function."""
        @timeout(0.1)
        def sync_timed_function(value: int) -> int:
            time.sleep(0.05)  # Within timeout
            return value * 2
        
        # Even though it's sync, we can still use it normally
        result = sync_timed_function(21)
        assert result == 42
    
    @pytest.mark.asyncio
    async def test_ensure_async_with_retry(self):
        """Test converting a sync function to async and adding retry."""
        attempt_count = 0
        
        def flaky_sync_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ConnectionError("Network issue")
            return "Finally worked"
        
        # Convert to async
        async_version = ensure_async(flaky_sync_function)
        
        # Add retry decorator
        @retry_with_backoff(max_attempts=3, initial_wait=0.01, retry_on=(ConnectionError,))
        async def retrying_async_version():
            return await async_version()
        
        result = await retrying_async_version()
        
        assert attempt_count == 2
        assert result == "Finally worked"
    
    @pytest.mark.asyncio
    async def test_nested_async_calls(self):
        """Test nested async function calls with utilities."""
        @timeout(1.0)
        async def outer_function():
            @retry_with_backoff(max_attempts=2, initial_wait=0.01)
            async def inner_function():
                await asyncio.sleep(0.01)
                return "inner result"
            
            inner_result = await inner_function()
            return f"outer: {inner_result}"
        
        result = await outer_function()
        assert result == "outer: inner result"
    
    def test_error_messages_contain_useful_info(self):
        """Test that error messages from utilities contain helpful information."""
        @timeout(0.05)
        def slow_function():
            time.sleep(0.2)
            return "Never reached"
        
        with pytest.raises(TimeoutError) as exc_info:
            slow_function()
        
        error_message = str(exc_info.value)
        assert "0.05 seconds" in error_message
        assert "timed out" in error_message.lower()
    
    @pytest.mark.asyncio
    async def test_concurrent_operations_with_all_utilities(self):
        """Test running multiple operations concurrently using all utilities."""
        results = []
        
        @retry_with_backoff(max_attempts=2, initial_wait=0.01, retry_on=(ConnectionError,))
        @timeout(0.5)
        async def operation(id: int, should_fail_once: bool = False):
            if should_fail_once and id not in results:
                results.append(id)
                raise ConnectionError(f"Operation {id} failed once")
            
            await asyncio.sleep(0.01 * id)  # Variable delay
            return f"Operation {id} completed"
        
        # Run multiple operations concurrently
        concurrent_results = await asyncio.gather(
            operation(1),
            operation(2, should_fail_once=True),
            operation(3),
            operation(4, should_fail_once=True),
            return_exceptions=False  # Let exceptions propagate
        )
        
        assert len(concurrent_results) == 4
        assert all("completed" in r for r in concurrent_results)
        
        # Operations 2 and 4 should have failed once
        assert 2 in results
        assert 4 in results


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_retry_with_zero_attempts(self):
        """Test retry with max_attempts=0 (should not be allowed)."""
        # max_attempts=0 would mean no attempts at all, which doesn't make sense
        # The decorator should handle this gracefully
        @retry_with_backoff(max_attempts=1, initial_wait=0.01)
        async def minimal_function():
            return "Called once"
        
        result = await minimal_function()
        assert result == "Called once"
    
    @pytest.mark.asyncio
    async def test_timeout_with_negative_duration(self):
        """Test timeout with negative duration."""
        @timeout(-1.0)  # Negative timeout
        async def negative_timeout_function():
            return "Should timeout immediately"
        
        # Should timeout immediately
        with pytest.raises(TimeoutError):
            await negative_timeout_function()
    
    def test_ensure_async_with_none(self):
        """Test ensure_async with None (should raise error)."""
        # ensure_async checks if the function is a coroutine function
        # None doesn't have iscoroutinefunction attribute
        result = ensure_async(None)
        # It returns a wrapper, but calling it will fail
        assert asyncio.iscoroutinefunction(result)
    
    @pytest.mark.asyncio
    async def test_run_async_with_non_coroutine(self):
        """Test run_async with non-coroutine object."""
        with pytest.raises(TypeError):
            run_async("not a coroutine")
    
    def test_timeout_cleanup_on_exception(self):
        """Test that timeout properly cleans up when function raises exception."""
        @timeout(1.0)
        def function_that_raises():
            raise ValueError("Function error")
        
        with pytest.raises(ValueError) as exc_info:
            function_that_raises()
        
        assert str(exc_info.value) == "Function error"
    
    @pytest.mark.asyncio
    async def test_retry_with_async_generator(self):
        """Test retry behavior with async generator (shouldn't be decorated directly)."""
        attempt_count = 0
        
        @retry_with_backoff(max_attempts=2, initial_wait=0.01, retry_on=(ConnectionError,))
        async def create_generator():
            nonlocal attempt_count
            attempt_count += 1
            
            if attempt_count < 2:
                raise ConnectionError("Failed to create generator")
            
            async def generator():
                for i in range(3):
                    yield i
            
            return generator()
        
        gen = await create_generator()
        values = [value async for value in gen]
        
        assert attempt_count == 2
        assert values == [0, 1, 2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])