# PM-Kit Test Fixtures Guide

This guide explains all the pytest fixtures available in `tests/conftest.py` for testing PM-Kit functionality.

## Quick Reference

| Fixture | Purpose | Returns |
|---------|---------|---------|
| `tmp_project` | Creates temporary PM-Kit project structure | `Path` to project directory |
| `mock_config` | Provides complete test configuration | `Config` object |
| `cli_runner` | Tests CLI commands | `CliRunner` instance |
| `mock_console` | Captures Rich console output | `(PMKitConsole, StringIO)` tuple |
| `mock_env` | Sets test environment variables | `Dict[str, str]` of env vars |
| `async_client` | Mock async API client | `MagicMock` with async methods |
| `mock_llm_response` | Sample LLM response data | `Dict` with response structure |
| `mock_prd_data` | Sample PRD data | `Dict` with PRD fields |
| `mock_context_data` | Complete context files data | `Dict` of context files |
| `mock_cache_dir` | Temporary cache directory | `Path` to cache directory |
| `capture_logs` | Captures log messages | `LogCaptureFixture` |
| `mock_api_responses` | Various API mock responses | `Dict` of API responses |

## Detailed Fixture Documentation

### Project Structure Fixtures

#### `tmp_project`
Creates a complete temporary PM-Kit project with:
- `.pmkit/` directory structure
- All context files (company.yaml, product.yaml, etc.)
- `.pmrc.yaml` configuration file
- Proper directory hierarchy

```python
def test_project_setup(tmp_project):
    assert (tmp_project / ".pmkit").exists()
    assert (tmp_project / ".pmkit/context/company.yaml").exists()
```

#### `mock_cache_dir`
Creates a temporary cache directory with subdirectories for different cache types:
- `llm/` - LLM response cache
- `context/` - Context versioning cache
- `search/` - Search results cache

```python
def test_cache_operations(mock_cache_dir):
    cache_file = mock_cache_dir / "llm" / "response.json"
    cache_file.write_text('{"data": "cached"}')
    assert cache_file.exists()
```

### Configuration Fixtures

#### `mock_config`
Provides a complete `Config` object with all sections populated:
- LLM provider settings with mock API keys
- Cache configuration
- Integration settings (Confluence, GitHub, etc.)
- Application settings
- Context management settings

```python
def test_config_usage(mock_config):
    assert mock_config.llm.provider == "openai"
    assert mock_config.cache.enabled is True
    # Secrets are properly wrapped
    api_key = mock_config.llm.api_key.get_secret_value()
```

#### `mock_env`
Sets up test environment variables for:
- API keys (OpenAI, Anthropic, Google)
- Integration tokens (Confluence, GitHub, Jira)
- PM-Kit settings (debug mode, log level)

```python
def test_with_env(mock_env):
    assert os.getenv("OPENAI_API_KEY") == "test-openai-key-123"
    assert mock_env["PMKIT_DEBUG"] == "0"
```

### CLI Testing Fixtures

#### `cli_runner`
Provides a Typer `CliRunner` for testing CLI commands:

```python
def test_cli_command(cli_runner):
    from pmkit.cli.main import app
    
    result = cli_runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout
    
    result = cli_runner.invoke(app, ["init", "--force"])
    assert "Initialized" in result.stdout
```

#### `mock_console`
Captures Rich console output for testing formatted output:

```python
def test_console_output(mock_console):
    console, output = mock_console
    
    console.success("Operation completed")
    console.error("Something failed")
    
    output_text = output.getvalue()
    assert "Operation completed" in output_text
    assert "Something failed" in output_text
```

### Async Testing Fixtures

#### `async_client`
Mock client for testing async operations:

```python
@pytest.mark.asyncio
async def test_async_operation(async_client):
    async_client.generate.return_value = {"response": "Test"}
    
    result = await async_client.generate("prompt")
    assert result["response"] == "Test"
```

### Test Data Fixtures

#### `mock_llm_response`
Provides sample LLM response structure:

```python
def test_llm_handling(mock_llm_response):
    assert "content" in mock_llm_response
    assert "tokens" in mock_llm_response
    assert mock_llm_response["model"] == "gpt-4-turbo-preview"
```

#### `mock_prd_data`
Complete PRD data structure with all fields:

```python
def test_prd_processing(mock_prd_data):
    assert "requirements" in mock_prd_data
    assert len(mock_prd_data["requirements"]) > 0
    for req in mock_prd_data["requirements"]:
        assert req["priority"] in ["must", "should", "could"]
```

#### `mock_context_data`
Complete context layer data:

```python
def test_context_loading(mock_context_data):
    company = mock_context_data["company.yaml"]
    assert company["type"] in ["b2b", "b2c"]
    
    product = mock_context_data["product.yaml"]
    assert "version" in product
```

#### `mock_api_responses`
Various API response mocks:

```python
def test_api_integration(mock_api_responses):
    openai_resp = mock_api_responses["openai_completion"]
    assert "choices" in openai_resp
    
    confluence_resp = mock_api_responses["confluence_page"]
    assert confluence_resp["space"]["key"] == "TEST"
```

### Utility Fixtures

#### `capture_logs`
Captures log messages for testing:

```python
def test_logging(capture_logs):
    import logging
    logger = logging.getLogger("test")
    
    logger.info("Test message")
    logger.error("Error occurred")
    
    assert "Test message" in capture_logs.text
    assert any(r.levelname == "ERROR" for r in capture_logs.records)
```

#### `reset_singletons` (autouse)
Automatically resets singleton instances between tests to prevent test interference.

## Usage Examples

### Testing a Complete PRD Generation Flow

```python
def test_prd_generation_flow(
    tmp_project,
    mock_config,
    mock_console,
    mock_llm_response,
    async_client
):
    """Test complete PRD generation with all fixtures."""
    console, output = mock_console
    
    # Setup async mock
    async_client.generate.return_value = mock_llm_response
    
    # Run PRD generation (pseudo-code)
    from pmkit.prd import PRDGenerator
    
    generator = PRDGenerator(config=mock_config)
    result = asyncio.run(generator.create_prd("Test Feature"))
    
    # Verify output
    assert "Test Feature" in output.getvalue()
    async_client.generate.assert_called()
```

### Testing CLI Commands with Context

```python
def test_cli_with_context(cli_runner, tmp_project, mock_env):
    """Test CLI command in project context."""
    from pmkit.cli.main import app
    
    # tmp_project changes cwd to temp directory
    result = cli_runner.invoke(app, ["status"])
    
    assert result.exit_code == 0
    assert ".pmkit" in result.stdout
```

### Testing Configuration Loading

```python
def test_config_loading(tmp_project, mock_env):
    """Test configuration loading from files and env."""
    from pmkit.config import load_config
    
    config = load_config()
    
    # Should load from tmp_project's .pmrc.yaml
    assert config.project_name is not None
    
    # Should use env variables for API keys
    assert config.llm.api_key is not None
```

## Best Practices

1. **Use appropriate fixtures**: Choose fixtures that match your test needs
2. **Combine fixtures**: Many fixtures work well together
3. **Leverage tmp_project**: For integration tests that need full project structure
4. **Mock external calls**: Use `async_client` and `mock_api_responses` for external APIs
5. **Capture output**: Use `mock_console` to verify user-facing output
6. **Test with different configs**: Use `mock_config` variations for different scenarios

## Adding New Fixtures

When adding new fixtures to `conftest.py`:

1. Follow the existing naming convention (`mock_*` for mocks, `tmp_*` for temporary resources)
2. Add comprehensive docstrings with examples
3. Use type hints for clarity
4. Clean up resources properly (use `yield` for cleanup)
5. Update this guide with the new fixture

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_context.py

# Run with coverage
pytest --cov=pmkit

# Run only unit tests
pytest -m unit

# Run without coverage (faster)
pytest --no-cov
```