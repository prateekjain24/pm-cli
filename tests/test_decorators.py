"""Tests for PM-Kit decorators."""

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

from pmkit.utils.decorators import (
    log_operation,
    log_async,
    log_errors,
    trace_performance,
    rate_limited_log,
)


class TestLogOperation:
    """Test the log_operation decorator."""

    def test_basic_logging(self):
        """Test basic function logging."""
        with patch('pmkit.utils.decorators.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            @log_operation()
            def test_func():
                return "result"

            result = test_func()
            assert result == "result"

            # Check that logger was called
            assert mock_logger.info.called
            assert mock_logger.success.called

    def test_log_with_args(self):
        """Test logging with function arguments."""
        with patch('pmkit.utils.decorators.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            @log_operation(log_args=True)
            def test_func(x: int, y: str):
                return x + len(y)

            result = test_func(5, "hello")
            assert result == 10

            # Check that args were logged
            info_call = mock_logger.info.call_args
            assert info_call is not None
            extra = info_call[1].get('extra', {})
            assert 'args' in extra

    def test_log_with_sensitive_args(self):
        """Test that sensitive arguments are redacted."""
        with patch('pmkit.utils.decorators.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            @log_operation(log_args=True)
            def test_func(username: str, password: str, api_token: str):
                return True

            result = test_func("user", "secret123", "token456")
            assert result is True

            # Check that sensitive args were redacted
            info_call = mock_logger.info.call_args
            extra = info_call[1].get('extra', {})
            args = extra.get('args', {})
            assert args.get('password') == '[REDACTED]'
            assert args.get('api_token') == '[REDACTED]'
            assert args.get('username') == 'user'

    def test_log_with_result(self):
        """Test logging with function result."""
        with patch('pmkit.utils.decorators.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            @log_operation(log_result=True)
            def test_func():
                return {"status": "success", "data": [1, 2, 3]}

            result = test_func()
            assert result["status"] == "success"

            # Check that result was logged
            success_call = mock_logger.success.call_args
            extra = success_call[1].get('extra', {})
            assert 'result' in extra

    def test_log_with_duration(self):
        """Test logging with duration measurement."""
        with patch('pmkit.utils.decorators.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            @log_operation(log_duration=True)
            def test_func():
                time.sleep(0.01)  # Small delay
                return "done"

            result = test_func()
            assert result == "done"

            # Check that duration was logged
            success_call = mock_logger.success.call_args
            extra = success_call[1].get('extra', {})
            assert 'duration_ms' in extra
            assert extra['duration_ms'] > 0

    def test_log_with_exception(self):
        """Test logging when function raises exception."""
        with patch('pmkit.utils.decorators.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            @log_operation()
            def test_func():
                raise ValueError("Test error")

            with pytest.raises(ValueError, match="Test error"):
                test_func()

            # Check that error was logged
            assert mock_logger.error.called
            error_call = mock_logger.error.call_args
            assert "Failed" in error_call[0][0]
            assert "ValueError" in error_call[0][0]

    def test_custom_operation_name(self):
        """Test using custom operation name."""
        with patch('pmkit.utils.decorators.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            @log_operation(operation="custom.operation")
            def test_func():
                return "result"

            test_func()

            info_call = mock_logger.info.call_args
            assert "Starting custom.operation" in info_call[0][0]


class TestLogAsync:
    """Test the log_async decorator."""

    @pytest.mark.asyncio
    async def test_async_basic_logging(self):
        """Test basic async function logging."""
        with patch('pmkit.utils.decorators.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            @log_async()
            async def test_func():
                await asyncio.sleep(0.001)
                return "async_result"

            result = await test_func()
            assert result == "async_result"

            assert mock_logger.info.called
            assert mock_logger.success.called

    @pytest.mark.asyncio
    async def test_async_with_exception(self):
        """Test async logging with exception."""
        with patch('pmkit.utils.decorators.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            @log_async()
            async def test_func():
                await asyncio.sleep(0.001)
                raise RuntimeError("Async error")

            with pytest.raises(RuntimeError, match="Async error"):
                await test_func()

            assert mock_logger.error.called

    @pytest.mark.asyncio
    async def test_async_with_args_and_result(self):
        """Test async logging with args and result."""
        with patch('pmkit.utils.decorators.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            @log_async(log_args=True, log_result=True)
            async def test_func(x: int, y: int):
                return x + y

            result = await test_func(3, 4)
            assert result == 7

            # Check args were logged
            info_call = mock_logger.info.call_args
            extra = info_call[1].get('extra', {})
            assert 'args' in extra

            # Check result was logged
            success_call = mock_logger.success.call_args
            extra = success_call[1].get('extra', {})
            assert 'result' in extra


class TestLogErrors:
    """Test the log_errors decorator."""

    def test_log_errors_sync(self):
        """Test error logging for sync functions."""
        with patch('pmkit.utils.decorators.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            @log_errors(reraise=True)
            def test_func():
                raise ValueError("Test error")

            with pytest.raises(ValueError, match="Test error"):
                test_func()

            # Check error was logged
            assert mock_logger.error.called
            error_call = mock_logger.error.call_args
            assert "ValueError" in error_call[0][0]
            assert "Test error" in error_call[0][0]

    def test_log_errors_no_reraise(self):
        """Test error logging without reraising."""
        with patch('pmkit.utils.decorators.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            @log_errors(reraise=False)
            def test_func():
                raise ValueError("Test error")
                return "should_not_reach"

            result = test_func()
            assert result is None  # Function should return None when error is suppressed

            # Check error was still logged
            assert mock_logger.error.called

    @pytest.mark.asyncio
    async def test_log_errors_async(self):
        """Test error logging for async functions."""
        with patch('pmkit.utils.decorators.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            @log_errors(reraise=True)
            async def test_func():
                await asyncio.sleep(0.001)
                raise RuntimeError("Async error")

            with pytest.raises(RuntimeError, match="Async error"):
                await test_func()

            # Check error was logged
            assert mock_logger.error.called

    def test_log_errors_with_context(self):
        """Test error logging includes context."""
        with patch('pmkit.utils.decorators.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            with patch('pmkit.utils.decorators.get_request_id', return_value="req-123"):
                with patch('pmkit.utils.decorators.get_operation_context', return_value="test-op"):
                    @log_errors(reraise=True)
                    def test_func():
                        raise TypeError("Context error")

                    with pytest.raises(TypeError):
                        test_func()

                    # Check context was included
                    error_call = mock_logger.error.call_args
                    extra = error_call[1].get('extra', {})
                    assert extra.get('request_id') == "req-123"
                    assert extra.get('operation') == "test-op"
                    assert extra.get('error_type') == "TypeError"


class TestTracePerformance:
    """Test the trace_performance decorator."""

    def test_trace_performance_sync(self):
        """Test performance tracing for sync functions."""
        with patch('pmkit.utils.decorators.TracingContext') as mock_tracing:
            @trace_performance(operation="test.perf")
            def test_func(x: int):
                return x * 2

            result = test_func(5)
            assert result == 10

            # Check that TracingContext was used
            mock_tracing.assert_called_once()
            call_kwargs = mock_tracing.call_args.kwargs
            assert call_kwargs['operation'] == "test.perf"

    @pytest.mark.asyncio
    async def test_trace_performance_async(self):
        """Test performance tracing for async functions."""
        with patch('pmkit.utils.decorators.TracingContext') as mock_tracing:
            @trace_performance(operation="async.perf")
            async def test_func(x: int):
                await asyncio.sleep(0.001)
                return x * 3

            result = await test_func(4)
            assert result == 12

            # Check that TracingContext was used
            mock_tracing.assert_called_once()
            call_kwargs = mock_tracing.call_args.kwargs
            assert call_kwargs['operation'] == "async.perf"

    def test_trace_performance_with_memory(self):
        """Test performance tracing with memory tracking."""
        with patch('pmkit.utils.decorators.TracingContext') as mock_tracing:
            @trace_performance(operation="memory.op", track_memory=True)
            def test_func():
                return "tracked"

            result = test_func()
            assert result == "tracked"

            # Check that TracingContext was used with memory tracking
            mock_tracing.assert_called_once()
            call_kwargs = mock_tracing.call_args.kwargs
            assert call_kwargs['operation'] == "memory.op"
            assert call_kwargs['track_memory'] is True


class TestRateLimitedLog:
    """Test the rate_limited_log decorator."""

    def test_rate_limited_log_basic(self):
        """Test basic rate limited logging."""
        # The decorator just wraps the function - test it doesn't break
        @rate_limited_log(max_per_minute=60)
        def test_func(x):
            return x * 2

        # Function should work normally
        assert test_func(5) == 10
        assert test_func(10) == 20

    def test_rate_limited_log_with_level(self):
        """Test rate limited logging with custom level."""
        @rate_limited_log(max_per_minute=30, level="INFO")
        def test_func():
            return "result"

        # Function should work normally
        assert test_func() == "result"
        assert test_func() == "result"

    def test_rate_limited_log_with_key_func(self):
        """Test rate limited logging with key function."""
        def make_key(x, y):
            return f"{x}_{y}"

        @rate_limited_log(key_func=make_key, max_per_minute=10)
        def test_func(x, y):
            return x + y

        # Function should work normally
        assert test_func(1, 2) == 3
        assert test_func(3, 4) == 7