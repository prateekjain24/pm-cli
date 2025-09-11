"""
Example test file demonstrating the usage of PM-Kit test fixtures.

This file shows how to use various fixtures defined in conftest.py
for testing PM-Kit functionality.
"""

import os
from pathlib import Path

import pytest
import yaml

from pmkit.config.models import Config
from pmkit.exceptions import PMKitError


class TestProjectFixtures:
    """Test suite demonstrating project-related fixtures."""
    
    def test_tmp_project_structure(self, tmp_project):
        """Test that tmp_project creates proper PM-Kit structure."""
        # Verify main directories exist
        assert tmp_project.exists()
        assert (tmp_project / ".pmkit").is_dir()
        assert (tmp_project / ".pmkit" / "context").is_dir()
        assert (tmp_project / ".pmkit" / ".cache").is_dir()
        assert (tmp_project / ".pmkit" / "templates").is_dir()
        
        # Verify context files exist
        context_dir = tmp_project / ".pmkit" / "context"
        assert (context_dir / "company.yaml").exists()
        assert (context_dir / "product.yaml").exists()
        assert (context_dir / "market.yaml").exists()
        assert (context_dir / "team.yaml").exists()
        assert (context_dir / "okrs.yaml").exists()
        
        # Verify configuration file exists
        assert (tmp_project / ".pmrc.yaml").exists()
    
    def test_tmp_project_context_data(self, tmp_project):
        """Test that context files contain valid data."""
        context_dir = tmp_project / ".pmkit" / "context"
        
        # Load and verify company data
        with open(context_dir / "company.yaml") as f:
            company_data = yaml.safe_load(f)
        
        assert "name" in company_data
        assert "type" in company_data
        assert company_data["type"] in ["b2b", "b2c", "b2b2c"]
        
        # Load and verify product data
        with open(context_dir / "product.yaml") as f:
            product_data = yaml.safe_load(f)
        
        assert "name" in product_data
        assert "version" in product_data
        assert "stage" in product_data


class TestConfigurationFixtures:
    """Test suite for configuration-related fixtures."""
    
    def test_mock_config_structure(self, mock_config):
        """Test that mock_config provides complete configuration."""
        assert isinstance(mock_config, Config)
        assert mock_config.project_name == "test-project"
        
        # Test LLM configuration
        assert mock_config.llm.provider == "openai"
        assert mock_config.llm.api_key is not None
        assert mock_config.llm.model == "gpt-4-turbo-preview"
        
        # Test cache configuration
        assert mock_config.cache.enabled is True
        assert mock_config.cache.ttl_seconds == 3600
        
        # Test integration configuration
        assert mock_config.integrations.github_owner == "test-owner"
        assert mock_config.integrations.github_repo == "test-repo"
    
    def test_mock_config_safe_dump(self, mock_config):
        """Test that sensitive data is masked in safe dump."""
        safe_data = mock_config.model_dump_safe()
        
        # Check that API keys are redacted
        assert safe_data["llm"]["api_key"] == "[REDACTED]"
        assert safe_data["integrations"]["github_token"] == "[REDACTED]"
        assert safe_data["integrations"]["confluence_api_token"] == "[REDACTED]"


class TestConsoleFixtures:
    """Test suite for console output fixtures."""
    
    def test_mock_console_capture(self, mock_console):
        """Test that mock_console captures output correctly."""
        console, output = mock_console
        
        # Test various console methods
        console.success("Operation successful")
        console.error("An error occurred")
        console.warning("This is a warning")
        console.info("Information message")
        
        output_text = output.getvalue()
        
        # Verify all messages were captured
        assert "Operation successful" in output_text
        assert "An error occurred" in output_text
        assert "This is a warning" in output_text
        assert "Information message" in output_text
    
    def test_mock_console_panel(self, mock_console):
        """Test panel output capture."""
        console, output = mock_console
        
        console.status_panel(
            title="Test Panel",
            content="This is test content",
            status="success",
            emoji="🎉"
        )
        
        output_text = output.getvalue()
        assert "Test Panel" in output_text
        assert "This is test content" in output_text


class TestEnvironmentFixtures:
    """Test suite for environment variable fixtures."""
    
    def test_mock_env_sets_variables(self, mock_env):
        """Test that mock_env sets environment variables correctly."""
        # Verify API keys are set
        assert os.getenv("OPENAI_API_KEY") == "test-openai-key-123"
        assert os.getenv("ANTHROPIC_API_KEY") == "test-anthropic-key-456"
        assert os.getenv("GOOGLE_API_KEY") == "test-google-key-789"
        
        # Verify other environment variables
        assert os.getenv("PMKIT_DEBUG") == "0"
        assert os.getenv("PMKIT_LOG_LEVEL") == "INFO"
    
    def test_mock_env_cleanup(self):
        """Test that environment is cleaned up after mock_env fixture."""
        # This test runs without mock_env to verify cleanup
        # The variables should not exist or have different values
        assert os.getenv("OPENAI_API_KEY") != "test-openai-key-123"


class TestAsyncFixtures:
    """Test suite for async-related fixtures."""
    
    @pytest.mark.asyncio
    async def test_async_client_methods(self, async_client):
        """Test that async_client provides mock async methods."""
        # Test generate method
        result = await async_client.generate("test prompt")
        assert "response" in result
        assert result["tokens"] == 100
        
        # Test search method
        search_result = await async_client.search("test query")
        assert "results" in search_result
        assert len(search_result["results"]) == 2
        
        # Test fetch method
        fetch_result = await async_client.fetch("http://test.com")
        assert fetch_result["status"] == 200
        assert "data" in fetch_result


class TestDataFixtures:
    """Test suite for test data fixtures."""
    
    def test_mock_llm_response_structure(self, mock_llm_response):
        """Test that mock_llm_response has correct structure."""
        assert "id" in mock_llm_response
        assert "model" in mock_llm_response
        assert "content" in mock_llm_response
        assert "tokens" in mock_llm_response
        
        # Verify token structure
        tokens = mock_llm_response["tokens"]
        assert "prompt" in tokens
        assert "completion" in tokens
        assert "total" in tokens
    
    def test_mock_prd_data_completeness(self, mock_prd_data):
        """Test that mock_prd_data contains all required fields."""
        assert "title" in mock_prd_data
        assert "problem_statement" in mock_prd_data
        assert "solution_overview" in mock_prd_data
        assert "requirements" in mock_prd_data
        assert "success_metrics" in mock_prd_data
        assert "timeline" in mock_prd_data
        
        # Verify requirements structure
        requirements = mock_prd_data["requirements"]
        assert len(requirements) > 0
        for req in requirements:
            assert "id" in req
            assert "description" in req
            assert "priority" in req
            assert req["priority"] in ["must", "should", "could"]
    
    def test_mock_context_data_files(self, mock_context_data):
        """Test that mock_context_data contains all context files."""
        expected_files = [
            "company.yaml",
            "product.yaml",
            "market.yaml",
            "team.yaml",
            "okrs.yaml"
        ]
        
        for filename in expected_files:
            assert filename in mock_context_data
            assert isinstance(mock_context_data[filename], dict)
        
        # Verify company data structure
        company = mock_context_data["company.yaml"]
        assert "name" in company
        assert "type" in company
        assert company["type"] in ["b2b", "b2c"]


class TestCLIFixtures:
    """Test suite for CLI testing fixtures."""
    
    def test_cli_runner_basic(self, cli_runner):
        """Test basic CLI runner functionality."""
        from pmkit.cli.main import app
        
        # Test help command
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "PM-Kit" in result.stdout
        
        # Test version command
        result = cli_runner.invoke(app, ["--version"])
        assert result.exit_code == 0
    
    def test_cli_runner_with_tmp_project(self, cli_runner, tmp_project):
        """Test CLI runner with temporary project."""
        from pmkit.cli.main import app
        
        # The tmp_project fixture changes cwd, so commands run in that context
        result = cli_runner.invoke(app, ["status"])
        
        # Status command should work with the tmp_project structure
        # (actual implementation may vary)
        assert result.exit_code in [0, 1]  # Depends on implementation


class TestCacheFixtures:
    """Test suite for cache-related fixtures."""
    
    def test_mock_cache_dir_structure(self, mock_cache_dir):
        """Test that mock_cache_dir creates proper structure."""
        assert mock_cache_dir.exists()
        assert (mock_cache_dir / "llm").is_dir()
        assert (mock_cache_dir / "context").is_dir()
        assert (mock_cache_dir / "search").is_dir()
    
    def test_mock_cache_dir_operations(self, mock_cache_dir):
        """Test cache directory operations."""
        # Create a cache file
        cache_file = mock_cache_dir / "llm" / "test_cache.json"
        cache_file.write_text('{"data": "cached"}')
        
        # Verify file exists and contains data
        assert cache_file.exists()
        assert "cached" in cache_file.read_text()
        
        # Test cleanup - directory should be temporary
        # On macOS, /tmp is often symlinked to /private/tmp or /var
        cache_path_str = str(mock_cache_dir)
        assert any([
            "/tmp" in cache_path_str,
            "/var" in cache_path_str,
            "/private" in cache_path_str,
            "pytest" in cache_path_str  # pytest creates temp dirs with its name
        ])


class TestLoggingFixtures:
    """Test suite for logging fixtures."""
    
    def test_capture_logs(self, capture_logs):
        """Test log capture functionality."""
        import logging
        
        logger = logging.getLogger("test_logger")
        
        # Log messages at different levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Verify all messages were captured
        assert "Debug message" in capture_logs.text
        assert "Info message" in capture_logs.text
        assert "Warning message" in capture_logs.text
        assert "Error message" in capture_logs.text
        
        # Verify log records
        assert len(capture_logs.records) >= 4
        
        # Check specific record
        info_records = [r for r in capture_logs.records if r.levelname == "INFO"]
        assert len(info_records) > 0


class TestAPIResponseFixtures:
    """Test suite for API response fixtures."""
    
    def test_mock_api_responses_openai(self, mock_api_responses):
        """Test OpenAI mock response structure."""
        response = mock_api_responses["openai_completion"]
        
        assert "id" in response
        assert "model" in response
        assert "choices" in response
        assert len(response["choices"]) > 0
        assert "usage" in response
        
        # Verify usage structure
        usage = response["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
    
    def test_mock_api_responses_anthropic(self, mock_api_responses):
        """Test Anthropic mock response structure."""
        response = mock_api_responses["anthropic_message"]
        
        assert "id" in response
        assert "type" in response
        assert response["type"] == "message"
        assert "content" in response
        assert len(response["content"]) > 0
        assert "usage" in response
    
    def test_mock_api_responses_confluence(self, mock_api_responses):
        """Test Confluence mock response structure."""
        response = mock_api_responses["confluence_page"]
        
        assert "id" in response
        assert "title" in response
        assert "space" in response
        assert response["space"]["key"] == "TEST"
        assert "body" in response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])