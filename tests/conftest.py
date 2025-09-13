"""
Pytest configuration and fixtures for PM-Kit testing.

This module provides comprehensive fixtures for testing PM-Kit functionality including:
- Temporary project directories with proper .pmkit structure
- Mock configurations for testing without real API keys
- CLI test runners for Typer commands
- Rich console output capturing
- Async test support
- Environment variable management
"""

from __future__ import annotations

import asyncio
import os
import sys
from io import StringIO
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator, Optional
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
import yaml
from faker import Faker
from pydantic import SecretStr
from rich.console import Console
from typer.testing import CliRunner

# Add src to path for imports during testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pmkit.cli.main import app
from pmkit.config.models import (
    ApplicationConfig,
    CacheConfig,
    Config,
    ContextConfig,
    IntegrationConfig,
    LLMProviderConfig,
)
from pmkit.exceptions import PMKitError
from pmkit.utils.console import PMKitConsole

# Initialize faker for test data generation
fake = Faker()


@pytest.fixture(scope="session")
def event_loop():
    """
    Create an event loop for the entire test session.
    
    This fixture ensures that all async tests share the same event loop,
    which is important for testing async functions that might interact.
    
    Returns:
        asyncio.AbstractEventLoop: The event loop for async tests
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def tmp_project(tmp_path: Path) -> Generator[Path, None, None]:
    """
    Create a temporary project directory with PM-Kit structure.
    
    This fixture creates a complete .pmkit directory structure that mimics
    a real PM-Kit project, including context files, cache directories, and
    configuration files.
    
    Args:
        tmp_path: Pytest's tmp_path fixture for temporary directories
        
    Yields:
        Path: Path to the temporary project directory
        
    Example:
        def test_project_structure(tmp_project):
            assert (tmp_project / ".pmkit").exists()
            assert (tmp_project / ".pmkit" / "context").is_dir()
    """
    # Create .pmkit directory structure
    pmkit_dir = tmp_path / ".pmkit"
    pmkit_dir.mkdir()
    
    # Create subdirectories
    (pmkit_dir / "context").mkdir()
    (pmkit_dir / "context" / "history").mkdir()
    (pmkit_dir / ".cache").mkdir()
    (pmkit_dir / "templates").mkdir()
    
    # Create sample context files with test data
    context_data = {
        "company.yaml": {
            "name": fake.company(),
            "domain": fake.domain_name(),
            "industry": fake.bs(),
            "size": fake.random_element(["startup", "medium", "enterprise"]),
            "type": fake.random_element(["b2b", "b2c", "b2b2c"]),
        },
        "product.yaml": {
            "name": fake.catch_phrase(),
            "description": fake.text(max_nb_chars=200),
            "version": "1.0.0",
            "stage": fake.random_element(["mvp", "growth", "mature"]),
        },
        "market.yaml": {
            "total_addressable_market": f"${fake.random_number(digits=9)}",
            "competitors": [fake.company() for _ in range(3)],
            "target_segment": fake.job(),
        },
        "team.yaml": {
            "size": fake.random_int(min=5, max=50),
            "roles": {
                "pm": fake.name(),
                "engineering_lead": fake.name(),
                "design_lead": fake.name(),
            },
        },
        "okrs.yaml": {
            "quarter": "Q1 2025",
            "objectives": [
                {
                    "title": fake.sentence(),
                    "key_results": [fake.sentence() for _ in range(3)],
                }
            ],
        },
    }
    
    # Write context files
    for filename, data in context_data.items():
        file_path = pmkit_dir / "context" / filename
        with open(file_path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)
    
    # Create a sample .pmrc.yaml configuration file
    config_data = {
        "project_name": fake.word(),
        "llm": {
            "provider": "openai",
            "model": "gpt-4-turbo-preview",
        },
        "cache": {
            "enabled": True,
            "directory": ".pmkit/.cache",
        },
    }
    
    config_path = tmp_path / ".pmrc.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config_data, f, default_flow_style=False)
    
    # Change to the temporary directory for the test
    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    
    yield tmp_path
    
    # Restore original working directory
    os.chdir(original_cwd)


@pytest.fixture
def mock_config() -> Config:
    """
    Create a mock PM-Kit configuration for testing.
    
    This fixture provides a complete configuration object with all sections
    populated with test data. API keys are mocked to avoid requiring real
    credentials during testing.
    
    Returns:
        Config: A complete PM-Kit configuration object
        
    Example:
        def test_config_validation(mock_config):
            assert mock_config.llm.provider == "openai"
            assert mock_config.cache.enabled is True
    """
    return Config(
        project_name="test-project",
        project_root=Path("/tmp/test-project"),
        llm=LLMProviderConfig(
            provider="openai",
            api_key=SecretStr("test-api-key-123"),
            model="gpt-4-turbo-preview",
            timeout=30,
            max_retries=3,
        ),
        cache=CacheConfig(
            enabled=True,
            directory=Path("/tmp/test-project/.pmkit/.cache"),
            ttl_seconds=3600,
            max_size_mb=50,
        ),
        integrations=IntegrationConfig(
            confluence_url="https://test.atlassian.net",
            confluence_username="test@example.com",
            confluence_api_token=SecretStr("test-confluence-token"),
            confluence_space_key="TEST",
            github_token=SecretStr("test-github-token"),
            github_owner="test-owner",
            github_repo="test-repo",
        ),
        app=ApplicationConfig(
            debug=False,
            log_level="INFO",
            pretty_json=True,
            enable_telemetry=False,
            no_color=False,
        ),
        context=ContextConfig(
            auto_enrich=True,
            validation_mode="strict",
            auto_backup=True,
            max_history_items=10,
        ),
    )


@pytest.fixture
def cli_runner() -> CliRunner:
    """
    Create a Typer CLI test runner.
    
    This fixture provides a CliRunner instance for testing CLI commands
    in isolation without actually running the full application.
    
    Returns:
        CliRunner: Typer test runner for CLI testing
        
    Example:
        def test_cli_command(cli_runner):
            result = cli_runner.invoke(app, ["init", "--force"])
            assert result.exit_code == 0
            assert "Initialized" in result.stdout
    """
    return CliRunner()


@pytest.fixture
def mock_console() -> Generator[tuple[Any, StringIO], None, None]:
    """
    Create a mock Rich console for capturing output.
    
    This fixture creates a test console that captures output to a StringIO buffer.
    It patches the console's print method in the exceptions module and provides
    a mock PMKitConsole for tests.
    
    Yields:
        tuple[Any, StringIO]: Mock console and output buffer
        
    Example:
        def test_console_output(mock_console):
            console, output = mock_console
            console.success("Test passed!")
            assert "Test passed!" in output.getvalue()
    """
    from rich.theme import Theme
    from rich.console import Console
    from rich.text import Text
    
    # Create a StringIO buffer for capturing output
    output_buffer = StringIO()
    
    # Create a simple theme with required styles
    test_theme = Theme({
        "success.text": "green",
        "error.text": "red",
        "warning.text": "yellow",
        "info.text": "cyan",
        "panel.border": "blue",
        "help.option": "magenta",
        "help.example": "dim cyan",
    })
    
    # Create a test console with the buffer and theme
    test_console = Console(
        file=output_buffer,
        force_terminal=True,
        width=120,
        legacy_windows=False,
        theme=test_theme,
    )
    
    # Create a mock PMKitConsole-like object
    class MockConsole:
        def __init__(self, console):
            self._console = console
            
        def print(self, *args, **kwargs):
            self._console.print(*args, **kwargs)
            
        def success(self, message: str):
            self._console.print(f"✅ {message}", style="green")
            
        def error(self, message: str):
            self._console.print(f"❌ {message}", style="red")
            
        def warning(self, message: str):
            self._console.print(f"⚠️  {message}", style="yellow")
            
        def info(self, message: str):
            self._console.print(f"ℹ️  {message}", style="cyan")
            
        def status_panel(self, title: str, content: str, status: str = "info", emoji: str = ""):
            from rich.panel import Panel
            from rich.text import Text
            
            # Create panel content
            text = Text()
            if emoji:
                text.append(f"{emoji} ")
            text.append(content)
            
            # Determine border style based on status
            border_styles = {
                "success": "green",
                "error": "red",
                "warning": "yellow",
                "info": "cyan"
            }
            border_style = border_styles.get(status, "blue")
            
            # Create and print panel
            panel = Panel(
                text,
                title=title,
                title_align="left",
                border_style=border_style,
                padding=(1, 2)
            )
            self._console.print(panel)
            
        def __getattr__(self, name):
            # Forward any other attribute access to the underlying console
            return getattr(self._console, name)
    
    mock_console_obj = MockConsole(test_console)
    
    # Import the exceptions module to ensure console is imported
    import pmkit.exceptions
    
    # Store original print method
    original_print = pmkit.exceptions.console.print
    
    # Replace print method with our test console's print
    pmkit.exceptions.console.print = mock_console_obj.print
    
    try:
        yield mock_console_obj, output_buffer
    finally:
        # Restore original print method
        pmkit.exceptions.console.print = original_print


@pytest.fixture
def mock_env(monkeypatch) -> Generator[Dict[str, str], None, None]:
    """
    Set up test environment variables.
    
    This fixture provides a clean environment with test API keys and
    configuration values. It uses monkeypatch to safely set and restore
    environment variables.
    
    Args:
        monkeypatch: Pytest's monkeypatch fixture
        
    Yields:
        Dict[str, str]: Dictionary of environment variables set
        
    Example:
        def test_env_vars(mock_env):
            assert os.getenv("OPENAI_API_KEY") == mock_env["OPENAI_API_KEY"]
            assert "test-" in os.getenv("OPENAI_API_KEY")
    """
    env_vars = {
        "OPENAI_API_KEY": "test-openai-key-123",
        "ANTHROPIC_API_KEY": "test-anthropic-key-456",
        "GOOGLE_API_KEY": "test-google-key-789",
        "CONFLUENCE_API_TOKEN": "test-confluence-token",
        "JIRA_API_TOKEN": "test-jira-token",
        "GITHUB_TOKEN": "test-github-token",
        "PMKIT_DEBUG": "0",
        "PMKIT_LOG_LEVEL": "INFO",
        "NO_COLOR": "0",
    }
    
    # Set all environment variables
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    yield env_vars
    
    # Cleanup is automatic with monkeypatch


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[MagicMock, None]:
    """
    Create a mock async client for testing async functions.
    
    This fixture provides a mock client that can be used to test async
    functions that make API calls. It includes common async methods
    with mock implementations.
    
    Yields:
        MagicMock: Mock async client with async methods
        
    Example:
        async def test_async_api_call(async_client):
            async_client.generate.return_value = "Test response"
            result = await some_async_function(async_client)
            assert result == "Test response"
    """
    mock_client = MagicMock()
    
    # Add common async methods
    async def mock_generate(*args, **kwargs):
        return {"response": "Mock LLM response", "tokens": 100}
    
    async def mock_search(*args, **kwargs):
        return {"results": ["result1", "result2"], "total": 2}
    
    async def mock_fetch(*args, **kwargs):
        return {"status": 200, "data": {"key": "value"}}
    
    mock_client.generate = MagicMock(side_effect=mock_generate)
    mock_client.search = MagicMock(side_effect=mock_search)
    mock_client.fetch = MagicMock(side_effect=mock_fetch)
    
    yield mock_client


@pytest.fixture
def mock_llm_response() -> Dict[str, Any]:
    """
    Provide sample LLM response data for testing.
    
    This fixture returns a dictionary with typical LLM response structure
    that can be used in tests that mock LLM API calls.
    
    Returns:
        Dict[str, Any]: Sample LLM response data
        
    Example:
        def test_llm_parsing(mock_llm_response):
            assert "content" in mock_llm_response
            assert mock_llm_response["model"] == "gpt-4-turbo-preview"
    """
    return {
        "id": fake.uuid4(),
        "model": "gpt-4-turbo-preview",
        "content": fake.text(max_nb_chars=500),
        "role": "assistant",
        "tokens": {
            "prompt": fake.random_int(100, 1000),
            "completion": fake.random_int(100, 1000),
            "total": fake.random_int(200, 2000),
        },
        "finish_reason": "stop",
        "created_at": fake.iso8601(),
    }


@pytest.fixture
def mock_prd_data() -> Dict[str, Any]:
    """
    Generate sample PRD data for testing.
    
    This fixture creates a complete PRD structure with all required fields
    populated with test data.
    
    Returns:
        Dict[str, Any]: Sample PRD data structure
        
    Example:
        def test_prd_generation(mock_prd_data):
            assert "title" in mock_prd_data
            assert len(mock_prd_data["requirements"]) > 0
    """
    return {
        "title": fake.catch_phrase(),
        "slug": fake.slug(),
        "problem_statement": fake.text(max_nb_chars=300),
        "solution_overview": fake.text(max_nb_chars=300),
        "target_users": [fake.job() for _ in range(3)],
        "requirements": [
            {
                "id": f"REQ-{i:03d}",
                "description": fake.sentence(),
                "priority": fake.random_element(["must", "should", "could"]),
                "type": fake.random_element(["functional", "non-functional"]),
            }
            for i in range(1, 6)
        ],
        "success_metrics": [
            {
                "metric": fake.sentence(),
                "target": f"{fake.random_int(10, 100)}%",
                "timeframe": fake.random_element(["30 days", "Q1", "6 months"]),
            }
            for _ in range(3)
        ],
        "risks": [
            {
                "description": fake.sentence(),
                "likelihood": fake.random_element(["low", "medium", "high"]),
                "impact": fake.random_element(["low", "medium", "high"]),
                "mitigation": fake.sentence(),
            }
            for _ in range(2)
        ],
        "timeline": {
            "start_date": fake.future_date(),
            "phases": [
                {
                    "name": f"Phase {i}",
                    "duration": f"{fake.random_int(1, 4)} weeks",
                    "deliverables": [fake.sentence() for _ in range(2)],
                }
                for i in range(1, 4)
            ],
        },
    }


@pytest.fixture
def mock_context_data() -> Dict[str, Dict[str, Any]]:
    """
    Generate complete context data for testing.
    
    This fixture creates a full set of context files data that represents
    a complete PM-Kit context layer.
    
    Returns:
        Dict[str, Dict[str, Any]]: Context data organized by file
        
    Example:
        def test_context_loading(mock_context_data):
            assert "company.yaml" in mock_context_data
            assert mock_context_data["company.yaml"]["type"] in ["b2b", "b2c"]
    """
    return {
        "company.yaml": {
            "name": fake.company(),
            "domain": fake.domain_name(),
            "industry": fake.bs(),
            "size": fake.random_element(["startup", "medium", "enterprise"]),
            "type": fake.random_element(["b2b", "b2c"]),
            "founded": fake.year(),
            "headquarters": fake.city(),
            "mission": fake.catch_phrase(),
            "values": [fake.word() for _ in range(4)],
        },
        "product.yaml": {
            "name": fake.catch_phrase(),
            "description": fake.text(max_nb_chars=200),
            "version": f"{fake.random_int(1, 3)}.{fake.random_int(0, 9)}.{fake.random_int(0, 9)}",
            "stage": fake.random_element(["mvp", "growth", "mature"]),
            "platform": fake.random_element(["web", "mobile", "desktop", "api"]),
            "tech_stack": [fake.word() for _ in range(5)],
            "integrations": [fake.company() for _ in range(3)],
        },
        "market.yaml": {
            "total_addressable_market": f"${fake.random_number(digits=9)}",
            "serviceable_addressable_market": f"${fake.random_number(digits=8)}",
            "serviceable_obtainable_market": f"${fake.random_number(digits=7)}",
            "growth_rate": f"{fake.random_int(10, 100)}%",
            "competitors": [
                {
                    "name": fake.company(),
                    "market_share": f"{fake.random_int(5, 30)}%",
                    "strengths": [fake.word() for _ in range(3)],
                    "weaknesses": [fake.word() for _ in range(3)],
                }
                for _ in range(3)
            ],
            "target_segments": [
                {
                    "name": fake.job(),
                    "size": fake.random_number(digits=6),
                    "characteristics": [fake.sentence() for _ in range(3)],
                }
                for _ in range(2)
            ],
        },
        "team.yaml": {
            "size": fake.random_int(min=10, max=100),
            "structure": fake.random_element(["flat", "hierarchical", "matrix"]),
            "roles": {
                "product_manager": fake.name(),
                "engineering_lead": fake.name(),
                "design_lead": fake.name(),
                "qa_lead": fake.name(),
                "data_analyst": fake.name(),
            },
            "departments": [
                {
                    "name": dept,
                    "size": fake.random_int(5, 20),
                    "lead": fake.name(),
                }
                for dept in ["Engineering", "Design", "QA", "Data", "Marketing"]
            ],
        },
        "okrs.yaml": {
            "quarter": f"Q{fake.random_int(1, 4)} {fake.year()}",
            "objectives": [
                {
                    "id": f"O{i}",
                    "title": fake.sentence(),
                    "description": fake.text(max_nb_chars=200),
                    "owner": fake.name(),
                    "key_results": [
                        {
                            "id": f"KR{i}.{j}",
                            "description": fake.sentence(),
                            "target": f"{fake.random_int(10, 100)}%",
                            "current": f"{fake.random_int(0, 50)}%",
                            "status": fake.random_element(["on-track", "at-risk", "behind"]),
                        }
                        for j in range(1, 4)
                    ],
                }
                for i in range(1, 4)
            ],
        },
    }


@pytest.fixture
def mock_cache_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """
    Create a temporary cache directory for testing.
    
    This fixture creates a cache directory structure and provides
    utilities for testing cache operations.
    
    Args:
        tmp_path: Pytest's tmp_path fixture
        
    Yields:
        Path: Path to the cache directory
        
    Example:
        def test_cache_operations(mock_cache_dir):
            cache_file = mock_cache_dir / "test.cache"
            cache_file.write_text("cached data")
            assert cache_file.exists()
    """
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir()
    
    # Create subdirectories for different cache types
    (cache_dir / "llm").mkdir()
    (cache_dir / "context").mkdir()
    (cache_dir / "search").mkdir()
    
    yield cache_dir


@pytest.fixture
def capture_logs(caplog):
    """
    Fixture for capturing and asserting on log messages.
    
    This fixture configures the logging capture and provides it
    in a convenient form for testing.
    
    Args:
        caplog: Pytest's caplog fixture
        
    Returns:
        LogCaptureFixture: Configured log capture
        
    Example:
        def test_logging(capture_logs):
            some_function_that_logs()
            assert "Expected message" in capture_logs.text
            assert capture_logs.records[0].levelname == "INFO"
    """
    # Set capture level to DEBUG to catch all logs
    caplog.set_level("DEBUG")
    return caplog


@pytest.fixture
def mock_datetime(freezegun):
    """
    Fixture for controlling time in tests.
    
    This fixture uses freezegun to allow tests to control the current
    time, which is useful for testing time-dependent functionality.
    
    Args:
        freezegun: The freezegun fixture (requires freezegun package)
        
    Returns:
        FrozenDateTimeFactory: Time control interface
        
    Example:
        def test_with_time(mock_datetime):
            mock_datetime.freeze_time("2025-01-15 10:30:00")
            # Test code that depends on current time
    """
    return freezegun


@pytest.fixture(autouse=True)
def reset_singletons():
    """
    Reset singleton instances between tests.
    
    This fixture automatically runs before each test to ensure that
    singleton instances are reset, preventing test interference.
    """
    # Reset PMKitConsole singleton
    PMKitConsole._instance = None
    
    # Reset any other singletons here as needed
    
    yield
    
    # Cleanup after test if needed
    PMKitConsole._instance = None


@pytest.fixture
def mock_api_responses():
    """
    Provide mock responses for various API calls.
    
    This fixture returns a dictionary of mock responses that can be used
    to simulate different API endpoints.
    
    Returns:
        Dict[str, Any]: Dictionary of mock API responses
        
    Example:
        def test_api_integration(mock_api_responses):
            response = mock_api_responses["openai_completion"]
            assert response["choices"][0]["text"] != ""
    """
    return {
        "openai_completion": {
            "id": f"cmpl-{fake.uuid4()}",
            "object": "text_completion",
            "created": fake.unix_time(),
            "model": "gpt-4-turbo-preview",
            "choices": [
                {
                    "text": fake.text(max_nb_chars=500),
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": fake.random_int(100, 500),
                "completion_tokens": fake.random_int(100, 500),
                "total_tokens": fake.random_int(200, 1000),
            },
        },
        "anthropic_message": {
            "id": f"msg_{fake.uuid4()}",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": fake.text(max_nb_chars=500),
                }
            ],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": fake.random_int(100, 500),
                "output_tokens": fake.random_int(100, 500),
            },
        },
        "confluence_page": {
            "id": fake.uuid4(),
            "type": "page",
            "status": "current",
            "title": fake.sentence(),
            "space": {"key": "TEST", "name": "Test Space"},
            "version": {"number": 1},
            "body": {
                "storage": {
                    "value": f"<p>{fake.text()}</p>",
                    "representation": "storage",
                }
            },
        },
        "github_issue": {
            "id": fake.random_int(1000000, 9999999),
            "number": fake.random_int(1, 1000),
            "title": fake.sentence(),
            "body": fake.text(),
            "state": "open",
            "user": {"login": fake.user_name()},
            "labels": [{"name": fake.word()} for _ in range(3)],
            "created_at": fake.iso8601(),
            "updated_at": fake.iso8601(),
        },
    }


# Markers for test categorization
pytest.mark.unit = pytest.mark.mark(name="unit")
pytest.mark.integration = pytest.mark.mark(name="integration")
pytest.mark.slow = pytest.mark.mark(name="slow")
pytest.mark.asyncio = pytest.mark.asyncio