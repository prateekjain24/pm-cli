"""
PM-Kit Logging Infrastructure using loguru with Rich integration.

Provides beautiful console output with Rich colors and structured JSON logs
for production use. Integrates with the config system and supports log
rotation, sensitive data filtering, and async-safe logging.
"""

from __future__ import annotations

import json
import os
import sys
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union, TYPE_CHECKING

from loguru import logger
from rich.console import Console
from rich.highlighter import Highlighter
from rich.text import Text

if TYPE_CHECKING:
    from pmkit.config.models import Config


def _get_log_level_colors() -> Dict[str, str]:
    """Get log level colors with lazy import to avoid circular dependency."""
    from pmkit.cli.theme import SUCCESS, WARNING, ERROR, INFO, MUTED
    
    return {
        "TRACE": MUTED,
        "DEBUG": MUTED, 
        "INFO": INFO,
        "SUCCESS": SUCCESS,
        "WARNING": WARNING,
        "ERROR": ERROR,
        "CRITICAL": ERROR,
    }

# Sensitive data patterns to filter from logs
SENSITIVE_PATTERNS = [
    re.compile(r'api[_-]?key[\'"\s]*[:=][\'"\s]*([^\s\'",}]+)', re.IGNORECASE),
    re.compile(r'token[\'"\s]*[:=][\'"\s]*([^\s\'",}]+)', re.IGNORECASE),
    re.compile(r'password[\'"\s]*[:=][\'"\s]*([^\s\'",}]+)', re.IGNORECASE),
    re.compile(r'secret[\'"\s]*[:=][\'"\s]*([^\s\'",}]+)', re.IGNORECASE),
    re.compile(r'bearer\s+([^\s\'",}]+)', re.IGNORECASE),
    re.compile(r'authorization[\'"\s]*[:=][\'"\s]*([^\s\'",}]+)', re.IGNORECASE),
]


class RichLogHighlighter(Highlighter):
    """Custom highlighter for Rich console logs with PM-Kit theme colors."""
    
    def highlight(self, text: Text) -> None:
        """Apply highlighting to log text based on level and content."""
        # Get the log level from the text (assuming format: "LEVEL | message")
        plain_text = text.plain
        
        # Get colors with lazy import
        log_level_colors = _get_log_level_colors()
        
        # Highlight log levels
        for level, color in log_level_colors.items():
            if plain_text.startswith(level):
                text.stylize(f"bold {color}", 0, len(level))
                break
        
        # Get individual colors for patterns
        from pmkit.cli.theme import SUCCESS, WARNING, ERROR, INFO, MUTED
        
        # Highlight common patterns
        text.highlight_regex(r'\b(ERROR|FAILED|EXCEPTION)\b', f"bold {ERROR}")
        text.highlight_regex(r'\b(SUCCESS|COMPLETED|DONE)\b', f"bold {SUCCESS}")
        text.highlight_regex(r'\b(WARNING|WARN)\b', f"bold {WARNING}")
        text.highlight_regex(r'\b(INFO|INFORMATION)\b', f"bold {INFO}")
        
        # Highlight file paths
        text.highlight_regex(r'[/\\][\w/\\.-]+\.[a-zA-Z]+', f"dim {MUTED}")
        
        # Highlight URLs
        text.highlight_regex(r'https?://[^\s]+', f"underline {INFO}")
        
        # Highlight durations/times
        text.highlight_regex(r'\d+\.?\d*\s?(ms|s|sec|seconds?|minutes?|hours?)', f"italic {SUCCESS}")


def filter_sensitive_data(message: str) -> str:
    """
    Filter sensitive data from log messages.
    
    Args:
        message: Log message to filter
        
    Returns:
        Filtered message with sensitive data replaced
    """
    filtered = message
    
    for pattern in SENSITIVE_PATTERNS:
        filtered = pattern.sub(lambda m: m.group(0).replace(m.group(1), '[REDACTED]'), filtered)
    
    return filtered


def json_serializer(record: Dict[str, Any]) -> str:
    """
    Custom JSON serializer for log records.
    
    Args:
        record: Log record dictionary
        
    Returns:
        JSON string representation of the record
    """
    # Create a clean record for JSON serialization
    clean_record = {
        'timestamp': record['time'].isoformat(),
        'level': record['level'].name,
        'logger': record['name'],
        'module': record['module'],
        'function': record['function'],
        'line': record['line'],
        'message': filter_sensitive_data(record['message']),
    }
    
    # Add extra fields if present
    extra = record.get('extra', {})
    if extra:
        clean_record['extra'] = extra
    
    # Add exception info if present
    if record.get('exception'):
        clean_record['exception'] = {
            'type': record['exception'].type.__name__,
            'value': str(record['exception'].value),
            'traceback': record['exception'].traceback,
        }
    
    return json.dumps(clean_record, default=str, ensure_ascii=False)


def console_formatter(record: Dict[str, Any]) -> str:
    """
    Format log records for Rich console output.
    
    Args:
        record: Log record dictionary
        
    Returns:
        Formatted string for console display
    """
    # Get colors with lazy import
    from pmkit.cli.theme import INFO
    log_level_colors = _get_log_level_colors()
    level_color = log_level_colors.get(record['level'].name, INFO)
    
    # Format timestamp
    time_str = record['time'].strftime('%H:%M:%S')
    
    # Format level with fixed width and color
    level_str = f"{record['level'].name:<8}"
    
    # Format logger name (shortened for console)
    logger_name = record['name']
    if logger_name.startswith('pmkit.'):
        logger_name = logger_name[6:]  # Remove 'pmkit.' prefix
    logger_str = f"{logger_name:<15}"
    
    # Filter sensitive data from message
    message = filter_sensitive_data(record['message'])
    
    # Build the formatted message
    parts = [
        f"[dim]{time_str}[/dim]",
        f"[bold {level_color}]{level_str}[/bold {level_color}]",
        f"[dim]{logger_str}[/dim]",
        message
    ]
    
    # Add extra context if present
    extra = record.get('extra', {})
    if extra:
        context_parts = []
        for key, value in extra.items():
            if key not in ('request_id', 'operation', 'duration_ms'):
                context_parts.append(f"{key}={value}")
        
        if context_parts:
            parts.append(f"[dim]({', '.join(context_parts)})[/dim]")
    
    return " | ".join(parts)


def setup_logging(config: Optional[Config] = None) -> None:
    """
    Setup loguru with Rich console output and JSON file logging.
    
    Args:
        config: PM-Kit configuration object. If None, uses environment variables
    """
    # Remove default loguru handler
    logger.remove()
    
    # Determine log level and debug mode
    if config:
        log_level = config.app.log_level
        debug_mode = config.app.debug
        no_color = config.app.no_color
        project_root = config.project_root
    else:
        log_level = os.getenv('PMKIT_LOG_LEVEL', 'INFO').upper()
        debug_mode = os.getenv('PMKIT_DEBUG', '').lower() in ('1', 'true', 'yes', 'on')
        no_color = bool(os.getenv('NO_COLOR'))
        project_root = Path.cwd()
    
    # Override log level to DEBUG if debug mode is enabled
    if debug_mode:
        log_level = 'DEBUG'
    
    # Setup console handler with Rich formatting
    console = Console(
        force_terminal=not no_color,
        highlighter=RichLogHighlighter() if not no_color else None,
        width=None,  # Auto-detect
    )
    
    def console_sink(message: str) -> None:
        """Rich console sink function."""
        console.print(message, highlight=False, markup=True)
    
    # Add Rich console handler
    logger.add(
        console_sink,
        format=console_formatter,
        level=log_level,
        colorize=False,  # We handle colors in the formatter
        backtrace=debug_mode,
        diagnose=debug_mode,
        enqueue=True,  # Thread-safe
    )
    
    # Setup file handler with JSON formatting and rotation
    log_dir = project_root / '.pmkit' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / 'pmkit.log'
    
    logger.add(
        log_file,
        format="{message}",  # We'll serialize to JSON in the sink
        serialize=True,  # Serialize the record to JSON
        level='DEBUG',  # Always log everything to file
        rotation='10 MB',
        retention='7 days',
        compression='gz',
        backtrace=True,
        diagnose=True,
        enqueue=True,  # Thread-safe
    )
    
    # Log startup message
    logger.info(
        "PM-Kit logging initialized",
        console_level=log_level,
        debug_mode=debug_mode,
        log_file=str(log_file),
    )


def get_logger(name: str) -> "LoggerAdapter":
    """
    Get a logger instance for the given module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        LoggerAdapter instance with context support
    """
    return LoggerAdapter(logger.bind(name=name), name)


class LoggerAdapter:
    """
    Adapter that provides additional context and convenience methods for logging.
    
    Wraps loguru logger to provide request context, timing, and structured logging.
    """
    
    def __init__(self, logger_instance: Any, name: str):
        self._logger = logger_instance
        self.name = name
    
    def bind(self, **kwargs: Any) -> "LoggerAdapter":
        """Bind additional context to the logger."""
        return LoggerAdapter(self._logger.bind(**kwargs), self.name)
    
    def with_context(self, **context: Any) -> "LoggerAdapter":
        """Add context to log messages."""
        return self.bind(**context)
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._logger.info(message, **kwargs)
    
    def success(self, message: str, **kwargs: Any) -> None:
        """Log success message."""
        self._logger.success(message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self._logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self._logger.critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        self._logger.exception(message, **kwargs)
    
    def time_operation(self, operation: str) -> "TimedOperation":
        """Create a context manager that times an operation."""
        return TimedOperation(self, operation)


class TimedOperation:
    """Context manager for timing operations and logging the results."""
    
    def __init__(self, logger_adapter: LoggerAdapter, operation: str):
        self.logger = logger_adapter
        self.operation = operation
        self.start_time = None
    
    def __enter__(self) -> "TimedOperation":
        """Start timing the operation."""
        import time
        self.start_time = time.perf_counter()
        self.logger.debug(f"Starting {self.operation}")
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """End timing and log the result."""
        import time
        if self.start_time is not None:
            duration_ms = (time.perf_counter() - self.start_time) * 1000
            
            if exc_type is None:
                self.logger.success(
                    f"Completed {self.operation}",
                    extra={'duration_ms': round(duration_ms, 2)}
                )
            else:
                self.logger.error(
                    f"Failed {self.operation}: {exc_val}",
                    extra={'duration_ms': round(duration_ms, 2)}
                )


# Module-level logger for this file
_module_logger = get_logger(__name__)


__all__ = [
    'setup_logging',
    'get_logger', 
    'LoggerAdapter',
    'TimedOperation',
    'filter_sensitive_data',
    'RichLogHighlighter',
]