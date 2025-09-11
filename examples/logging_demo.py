#!/usr/bin/env python3
"""
Demo script showing PM-Kit's comprehensive logging infrastructure.

This script demonstrates:
1. Logger setup with Rich console and JSON file output
2. Request tracing and context injection
3. Performance monitoring with decorators
4. Async/sync logging patterns
5. Error handling with structured logging
"""

import asyncio
import time
from pathlib import Path

# Add src to path for demo
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pmkit.utils.logger import setup_logging, get_logger
from pmkit.utils.tracing import request_context, async_request_context, TracingContext
from pmkit.utils.decorators import log_operation, log_async, log_errors


# Initialize logging (normally done in main CLI)
setup_logging(None)

# Get module logger
logger = get_logger(__name__)


@log_operation(operation="demo.sync_operation", log_args=True, log_duration=True)
def sync_operation_example(name: str, delay: float = 1.0) -> str:
    """Example sync function with logging decorator."""
    logger.info(f"Processing {name}")
    time.sleep(delay)
    return f"Processed {name}"


@log_async(operation="demo.async_operation", log_result=True)
async def async_operation_example(name: str, delay: float = 0.5) -> dict:
    """Example async function with logging decorator."""
    logger.info(f"Async processing {name}")
    await asyncio.sleep(delay)
    return {"name": name, "status": "completed", "timestamp": time.time()}


@log_errors(reraise=False)
def error_example() -> None:
    """Example function that demonstrates error logging."""
    logger.warning("About to raise an error for demo purposes")
    raise ValueError("This is a demo error for logging testing")


async def tracing_context_demo() -> None:
    """Demo of request tracing with context propagation."""
    logger.info("=== Tracing Context Demo ===")
    
    # Manual context management
    with request_context(operation="prd.generate") as req_id:
        logger.info(f"Inside request context {req_id}")
        logger.debug("This log has request context", extra={"step": "initial"})
        
        # Nested operation
        with TracingContext("prd.analyze_requirements", track_memory=True):
            logger.info("Analyzing requirements")
            time.sleep(0.1)  # Simulate work
            logger.success("Requirements analyzed")
    
    # Async context
    async with async_request_context(operation="llm.generate") as req_id:
        logger.info(f"Inside async request context {req_id}")
        result = await async_operation_example("AI Generated PRD")
        logger.info("LLM generation completed", extra={"result_size": len(str(result))})


def performance_demo() -> None:
    """Demo of performance tracking."""
    logger.info("=== Performance Demo ===")
    
    with TracingContext("expensive_operation", track_memory=True):
        logger.info("Starting expensive operation")
        
        # Simulate CPU work
        data = []
        for i in range(100000):
            data.append(i ** 2)
        
        time.sleep(0.5)  # Simulate I/O
        logger.info(f"Generated {len(data)} items")


def main() -> None:
    """Main demo function."""
    logger.success("=== PM-Kit Logging Infrastructure Demo ===")
    
    try:
        # Sync operations with decorators
        logger.info("=== Decorated Operations Demo ===")
        result1 = sync_operation_example("Document Analysis", 0.3)
        logger.info(f"Sync result: {result1}")
        
        # Error handling demo
        logger.info("=== Error Handling Demo ===")
        error_example()  # Won't crash due to reraise=False
        
        # Performance tracking
        performance_demo()
        
        # Async tracing demo
        asyncio.run(tracing_context_demo())
        
        # Async decorated operation
        async def async_demo():
            result2 = await async_operation_example("User Authentication")
            logger.info(f"Async result: {result2}")
        
        asyncio.run(async_demo())
        
        logger.success("Demo completed successfully!")
        
    except Exception:
        logger.exception("Demo failed with exception")
        raise


if __name__ == "__main__":
    main()