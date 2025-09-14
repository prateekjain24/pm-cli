"""Tests for CLI commands."""

from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import pytest
from typer.testing import CliRunner


class TestInitCommand:
    """Test the init command."""

    def test_init_new_project(self, tmp_path):
        """Test initializing a new project."""
        from pmkit.cli.commands.init import init_pmkit

        with patch('pmkit.cli.commands.init.Path.cwd', return_value=tmp_path):
            with patch('pmkit.cli.commands.init.initialize_context_structure') as mock_init:
                mock_init.return_value = True

                with patch('pmkit.cli.commands.init.console'):
                    with patch('pmkit.cli.commands.init.run_onboarding') as mock_onboard:
                        mock_onboard.return_value = (True, MagicMock())

                        # Use non_interactive=True instead of skip_onboarding
                        result = init_pmkit(non_interactive=True)

                        # Should initialize structure
                        mock_init.assert_called_once()
                        # Should return True for success
                        assert result is True

    def test_init_with_force(self, tmp_path):
        """Test force initializing existing project."""
        from pmkit.cli.commands.init import init_pmkit

        # Create existing .pmkit directory
        pmkit_dir = tmp_path / ".pmkit"
        pmkit_dir.mkdir()

        with patch('pmkit.cli.commands.init.Path.cwd', return_value=tmp_path):
            with patch('pmkit.cli.commands.init.initialize_context_structure') as mock_init:
                mock_init.return_value = True

                with patch('pmkit.cli.commands.init.console'):
                    with patch('pmkit.cli.commands.init.run_onboarding') as mock_onboard:
                        mock_onboard.return_value = (True, MagicMock())

                        result = init_pmkit(force=True, non_interactive=True)

                        # Should initialize despite existing
                        mock_init.assert_called_once()
                        assert result is True

    def test_init_with_onboarding(self, tmp_path):
        """Test init with onboarding flow."""
        from pmkit.cli.commands.init import init_pmkit
        from pmkit.context.models import Context, CompanyContext, ProductContext

        with patch('pmkit.cli.commands.init.Path.cwd', return_value=tmp_path):
            with patch('pmkit.cli.commands.init.initialize_context_structure') as mock_init:
                mock_init.return_value = True

                with patch('pmkit.cli.commands.init.run_onboarding') as mock_onboard:
                    mock_context = Context(
                        company=CompanyContext(name="Test", type="b2b", stage="growth"),
                        product=ProductContext(name="Product", description="Test product")
                    )
                    mock_onboard.return_value = (True, mock_context)

                    with patch('pmkit.cli.commands.init.console'):
                        # Use non_interactive=False to trigger onboarding
                        result = init_pmkit(non_interactive=False)

                        # Should run onboarding
                        mock_onboard.assert_called_once()
                        assert result is True


class TestStatusCommand:
    """Test the status command."""

    def test_check_status_simple(self):
        """Test basic status check functionality."""
        from pmkit.cli.commands.status import check_status

        with patch('pmkit.cli.commands.status.Path.cwd') as mock_cwd:
            # Set up mock directory structure
            mock_cwd.return_value = Path("/test/project")

            with patch('pmkit.cli.commands.status.console'):
                # Should not raise any errors
                check_status()


class TestConfigCommand:
    """Test config management commands."""

    def test_show_config_with_cli_runner(self):
        """Test showing configuration using CLI runner."""
        from pmkit.cli.commands.config import app as config_app

        runner = CliRunner()

        with patch('pmkit.cli.commands.config.get_config_safe') as mock_get_config:
            mock_config = MagicMock()
            mock_config.model_dump_safe.return_value = {
                "llm": {"provider": "openai"},
                "app": {"debug": False}
            }
            mock_get_config.return_value = mock_config

            result = runner.invoke(config_app, ["show"])
            # Command should run successfully
            assert result.exit_code == 0

    def test_validate_config_with_cli_runner(self):
        """Test validating configuration using CLI runner."""
        from pmkit.cli.commands.config import app as config_app

        runner = CliRunner()

        with patch('pmkit.cli.commands.config.load_config') as mock_load:
            mock_config = MagicMock()
            mock_load.return_value = mock_config

            result = runner.invoke(config_app, ["validate"])
            # Command should run successfully
            assert result.exit_code == 0