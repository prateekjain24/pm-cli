# PM-Kit Logging Infrastructure Guide

PM-Kit provides a comprehensive logging infrastructure using [loguru](https://github.com/Delgan/loguru) with Rich console output and structured JSON logs. This guide covers how to use the logging system effectively.

## Quick Start

```python
from pmkit.utils import setup_logging, get_logger

# Initialize logging (done automatically in CLI)
setup_logging(config)  # Pass config object, or None for env-based setup

# Get a logger for your module
logger = get_logger(__name__)

# Basic logging
logger.info("Processing user request")
logger.success("Operation completed")
logger.warning("Rate limit approaching")
logger.error("Authentication failed")
```

## Features

### 1. Beautiful Console Output with Rich

- **Colored output** using PM-Kit theme colors
- **Emoji support** for visual hierarchy
- **Structured formatting** with timestamps and context
- **Syntax highlighting** for URLs, file paths, and keywords
- **Progress indicators** and status panels

### 2. Structured JSON File Logs

- **Automatic log rotation** (10MB max size, 7 days retention)
- **Compressed archives** for space efficiency
- **Machine-readable format** for analysis and monitoring
- **Full stack traces** and error context
- **Request correlation** via unique IDs

### 3. Request Tracing & Context

```python
from pmkit.utils import request_context, get_logger

logger = get_logger(__name__)

# Automatic request tracing
with request_context(operation="prd.generate") as req_id:
    logger.info("Starting PRD generation")  # Auto includes request_id
    # ... all nested operations inherit context
    logger.success("PRD generation completed")
```

### 4. Performance Monitoring

```python
from pmkit.utils import TracingContext

# Track execution time and memory usage
with TracingContext("expensive_operation", track_memory=True):
    # Your code here
    process_large_dataset()
    # Automatically logs duration and memory delta
```

### 5. Decorators for Easy Integration

```python
from pmkit.utils import log_operation, log_async, log_errors

@log_operation(operation="user.authenticate", log_args=True, log_duration=True)
def authenticate_user(username: str, password: str) -> bool:
    # Function automatically logged with timing
    return check_credentials(username, password)

@log_async(operation="llm.generate")
async def generate_content(prompt: str) -> str:
    # Async functions supported
    return await llm_client.generate(prompt)

@log_errors(reraise=True)
def risky_operation() -> str:
    # Automatic error logging with context
    raise ValueError("Something went wrong")
```

## Configuration

### Environment Variables

- `PMKIT_DEBUG=1` - Enable debug logging
- `PMKIT_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR` - Set log level
- `NO_COLOR=1` - Disable colored output

### Config File (.pmrc.yaml)

```yaml
app:
  debug: false
  log_level: INFO  # DEBUG, INFO, WARNING, ERROR
  no_color: false
```

## Log Levels and Colors

| Level    | Color      | Use Case                    |
|----------|------------|----------------------------|
| DEBUG    | Gray       | Development debugging      |
| INFO     | Blue       | General information        |
| SUCCESS  | Green      | Successful operations      |
| WARNING  | Orange     | Non-critical issues        |
| ERROR    | Red        | Errors and failures        |
| CRITICAL | Red        | System-level failures      |

## Security Features

### Sensitive Data Filtering

The logging system automatically redacts sensitive information:

```python
logger.info("User authenticated", extra={
    'username': 'john@example.com',
    'api_key': 'sk-1234567890abcdef',  # Will be logged as [REDACTED]
    'password': 'secret123',           # Will be logged as [REDACTED]
})
```

Patterns automatically filtered:
- API keys (`api_key`, `apiKey`, etc.)
- Tokens (`token`, `bearer_token`, etc.)  
- Passwords (`password`, `pwd`, etc.)
- Secrets (`secret`, `client_secret`, etc.)
- Authorization headers

## Advanced Usage

### Custom Context Data

```python
from pmkit.utils import set_trace_data, get_logger_with_context

# Add context data for all subsequent logs
set_trace_data(
    user_id="12345",
    feature="prd_generation", 
    version="2.1.0"
)

# Get logger that automatically includes current context
logger = get_logger_with_context(__name__)
logger.info("This log includes all context data")
```

### Async Context Propagation

```python
from pmkit.utils import async_request_context

async def process_request():
    async with async_request_context(operation="api.process") as req_id:
        # All async operations inherit context
        result = await fetch_data()
        await process_data(result)
        await save_result(result)
        # All logs correlated by req_id
```

### Rate-Limited Logging

```python
from pmkit.utils import rate_limited_log

@rate_limited_log(max_per_minute=5)
def api_error_handler(error_code: str):
    # Only logs 5 times per minute to prevent spam
    logger.error(f"API error: {error_code}")
```

## File Locations

### Console Logs
Displayed in terminal with Rich formatting.

### File Logs
- **Location**: `.pmkit/logs/pmkit.log`
- **Format**: JSON (one object per line)
- **Rotation**: 10MB files, 7 days retention
- **Compression**: Older files compressed with gzip

### Example JSON Log Entry

```json
{
  "timestamp": "2025-01-15T10:30:45.123456",
  "level": "INFO",
  "logger": "pmkit.agents.prd",
  "module": "generator", 
  "function": "generate_prd",
  "line": 145,
  "message": "PRD generation completed",
  "extra": {
    "request_id": "abc12345",
    "operation": "prd.generate",
    "duration_ms": 2450.75,
    "prd_title": "Mobile App Authentication",
    "sections_generated": 8
  }
}
```

## Integration with CLI Commands

The CLI automatically initializes logging based on configuration:

```python
from pmkit.utils import setup_logging, get_logger
from pmkit.config import get_config_safe

# In CLI command
def my_command():
    logger = get_logger(__name__)
    logger.info("Command started", extra={'command': 'prd-generate'})
    
    try:
        result = perform_operation()
        logger.success("Command completed", extra={'result': result})
    except Exception as e:
        logger.exception("Command failed")
        raise
```

## Best Practices

### 1. Use Structured Logging

```python
# Good - structured with context
logger.info("User authenticated", extra={
    'user_id': user.id,
    'auth_method': 'oauth',
    'duration_ms': 150
})

# Avoid - unstructured string formatting
logger.info(f"User {user.id} authenticated via oauth in 150ms")
```

### 2. Appropriate Log Levels

```python
logger.debug("Cache hit for key: user_123")        # Development info
logger.info("Processing PRD generation request")    # Normal operations
logger.success("PRD generated successfully")       # Positive outcomes
logger.warning("Rate limit at 80% capacity")       # Concerning but not critical
logger.error("Failed to authenticate user")        # Errors requiring attention
logger.critical("Database connection lost")        # System-level failures
```

### 3. Context Management

```python
# Use request contexts for user operations
with request_context(operation="user.onboard", user_id="123"):
    create_user_profile()
    send_welcome_email()
    setup_initial_workspace()
    # All logs automatically correlated
```

### 4. Performance Monitoring

```python
# Monitor expensive operations
with TracingContext("ai.generate_content", track_memory=True):
    content = await llm_client.generate(prompt)
    # Automatic timing and memory tracking
```

### 5. Error Handling

```python
@log_errors(reraise=True)  # Log but still raise
async def critical_operation():
    # Errors automatically logged with full context
    await perform_critical_task()
```

## Troubleshooting

### Common Issues

1. **Logs not appearing**: Check `PMKIT_LOG_LEVEL` and `NO_COLOR` environment variables
2. **Performance impact**: Use appropriate log levels (`DEBUG` only in development)
3. **File permissions**: Ensure `.pmkit/logs/` directory is writable
4. **Memory usage**: Loguru uses background threads and queues - normal for async apps

### Debug Mode

Enable comprehensive logging:

```bash
PMKIT_DEBUG=1 pm new prd "My Product"
```

This enables:
- `DEBUG` level logging
- Full stack traces in console  
- Extended error context
- Performance timing for all operations

## Example Usage

See `examples/logging_demo.py` for a complete demonstration of all logging features.