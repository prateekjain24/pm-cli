"""Tests for the OnboardingAgent."""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import yaml
from prompt_toolkit import PromptSession

from pmkit.agents.onboarding import (
    OnboardingAgent,
    OnboardingPhase,
    OnboardingState,
    run_onboarding,
)
from pmkit.config.models import Config, LLMProviderConfig
from pmkit.context.models import CompanyContext, Context, ProductContext
from pmkit.llm.grounding import GroundingAdapter
from pmkit.llm.models import SearchResult


@pytest.fixture
def temp_context_dir(tmp_path):
    """Create a temporary context directory."""
    context_dir = tmp_path / ".pmkit" / "context"
    context_dir.mkdir(parents=True)
    return context_dir


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = Config(
        llm=LLMProviderConfig(
            provider="openai",
            api_key="test-key",
            model="gpt-5"
        )
    )
    return config


@pytest.fixture
def mock_grounding():
    """Create a mock grounding adapter."""
    grounding = Mock(spec=GroundingAdapter)
    grounding.search = AsyncMock(return_value=SearchResult(
        content="Test company is a leading provider of test services.",
        citations=[],
        query="test query",
        cached=False
    ))
    return grounding


@pytest.fixture
def sample_state():
    """Create a sample onboarding state."""
    return OnboardingState(
        current_phase=OnboardingPhase.ESSENTIALS,
        company_data={
            "name": "TestCorp",
            "type": "b2b",
            "stage": "growth",
            "domain": "testcorp.com"
        },
        product_data={
            "name": "TestProduct",
            "description": "A test product for testing",
            "stage": "pmf"
        }
    )


class TestOnboardingState:
    """Test OnboardingState model."""

    def test_initial_state(self):
        """Test initial state creation."""
        state = OnboardingState()

        assert state.current_phase == OnboardingPhase.ESSENTIALS
        assert state.completed_steps == []
        assert state.skipped_steps == []
        assert not state.is_complete
        assert state.completion_percentage == 0

    def test_mark_step_complete(self):
        """Test marking steps as complete."""
        state = OnboardingState()

        state.mark_step_complete("company_name")
        assert "company_name" in state.completed_steps

        # Should not duplicate
        state.mark_step_complete("company_name")
        assert state.completed_steps.count("company_name") == 1

    def test_mark_step_skipped(self):
        """Test marking steps as skipped."""
        state = OnboardingState()

        state.mark_step_skipped("team_setup")
        assert "team_setup" in state.skipped_steps

        # Should not duplicate
        state.mark_step_skipped("team_setup")
        assert state.skipped_steps.count("team_setup") == 1

    def test_completion_percentage(self):
        """Test completion percentage calculation."""
        state = OnboardingState()

        # Add some completed steps
        for i in range(5):
            state.mark_step_complete(f"step_{i}")

        # Should be approximately 33% (5/15 steps)
        assert 30 <= state.completion_percentage <= 35

    def test_is_complete(self):
        """Test completion check."""
        state = OnboardingState()
        assert not state.is_complete

        state.current_phase = OnboardingPhase.COMPLETE
        assert state.is_complete


class TestOnboardingAgent:
    """Test OnboardingAgent class."""

    @pytest.mark.asyncio
    async def test_agent_initialization(self, mock_config, temp_context_dir):
        """Test agent initialization."""
        agent = OnboardingAgent(
            config=mock_config,
            context_dir=temp_context_dir,
            resume=False
        )

        assert agent.config == mock_config
        assert agent.context_dir == temp_context_dir
        assert agent.state.current_phase == OnboardingPhase.ESSENTIALS
        assert not agent.cancelled

    @pytest.mark.asyncio
    async def test_load_existing_state(self, mock_config, temp_context_dir, sample_state):
        """Test loading existing state from file."""
        # Save state to file
        state_file = temp_context_dir / "onboarding_state.yaml"
        with open(state_file, 'w') as f:
            yaml.dump(sample_state.model_dump(mode='json'), f)

        # Create agent with resume=True
        agent = OnboardingAgent(
            config=mock_config,
            context_dir=temp_context_dir,
            resume=True
        )

        # Should load the saved state
        assert agent.state.company_data["name"] == "TestCorp"
        assert agent.state.product_data["name"] == "TestProduct"

    @pytest.mark.asyncio
    async def test_save_state(self, mock_config, temp_context_dir):
        """Test saving state to file."""
        agent = OnboardingAgent(
            config=mock_config,
            context_dir=temp_context_dir,
            resume=False
        )

        # Modify state
        agent.state.company_data["name"] = "SavedCorp"
        agent._save_state()

        # Check file was created
        state_file = temp_context_dir / "onboarding_state.yaml"
        assert state_file.exists()

        # Load and verify
        with open(state_file, 'r') as f:
            data = yaml.safe_load(f)
        assert data["company_data"]["name"] == "SavedCorp"

    @pytest.mark.asyncio
    async def test_build_context(self, mock_config, temp_context_dir):
        """Test building Context from collected data."""
        agent = OnboardingAgent(
            config=mock_config,
            context_dir=temp_context_dir,
            resume=False
        )

        # Set up state with data
        agent.state.company_data = {
            "name": "BuildCorp",
            "type": "b2b",
            "stage": "growth",
            "domain": "buildcorp.com",
            "description": "Building things"
        }
        agent.state.product_data = {
            "name": "Builder",
            "description": "A product that builds",
            "stage": "pmf"
        }
        agent.state.market_data = {
            "competitors": ["CompA", "CompB"],
            "differentiator": "We build better"
        }
        agent.state.team_data = {
            "size": 10,
            "roles": {"engineers": 6, "designers": 2}
        }

        # Build context
        context = agent._build_context()

        assert isinstance(context, Context)
        assert context.company.name == "BuildCorp"
        assert context.product.name == "Builder"
        assert context.market.competitors == ["CompA", "CompB"]
        assert context.team.size == 10

    @pytest.mark.asyncio
    async def test_enrich_company_data(self, mock_config, temp_context_dir, mock_grounding):
        """Test company data enrichment."""
        agent = OnboardingAgent(
            config=mock_config,
            context_dir=temp_context_dir,
            grounding=mock_grounding,
            resume=False
        )

        agent.state.company_data = {
            "name": "EnrichCorp",
            "domain": "enrichcorp.com"
        }

        # Run enrichment
        result = await agent._enrich_company_data()

        assert result is True
        assert mock_grounding.search.called
        # Should have extracted description from search result
        assert "description" in agent.state.company_data

    @pytest.mark.asyncio
    async def test_enrich_market_data(self, mock_config, temp_context_dir, mock_grounding):
        """Test market data enrichment."""
        agent = OnboardingAgent(
            config=mock_config,
            context_dir=temp_context_dir,
            grounding=mock_grounding,
            resume=False
        )

        agent.state.product_data = {
            "name": "TestProduct",
            "description": "A product for testing"
        }

        # Run enrichment
        result = await agent._enrich_market_data()

        assert result is True
        assert mock_grounding.search.called
        assert agent.state.market_data.get("has_research") is True

    @pytest.mark.asyncio
    async def test_prompt_yes_no(self, mock_config, temp_context_dir):
        """Test yes/no prompt helper."""
        agent = OnboardingAgent(
            config=mock_config,
            context_dir=temp_context_dir,
            resume=False
        )

        # Mock the prompt session
        with patch.object(agent.session, 'prompt_async') as mock_prompt:
            # Test yes response
            mock_prompt.return_value = "y"
            result = await agent._prompt_yes_no("Test question?", default=False)
            assert result is True

            # Test no response
            mock_prompt.return_value = "n"
            result = await agent._prompt_yes_no("Test question?", default=True)
            assert result is False

            # Test default (empty response)
            mock_prompt.return_value = ""
            result = await agent._prompt_yes_no("Test question?", default=True)
            assert result is True

    @pytest.mark.asyncio
    async def test_run_essentials_phase(self, mock_config, temp_context_dir):
        """Test running essentials phase."""
        agent = OnboardingAgent(
            config=mock_config,
            context_dir=temp_context_dir,
            resume=False
        )

        # Mock the prompt session
        with patch.object(agent.session, 'prompt_async') as mock_prompt:
            # Set up responses for company and product info
            mock_prompt.side_effect = [
                "TestCompany",  # Company name
                "b2b",          # Business type
                "growth",       # Company stage
                "",             # Domain (skip)
                "TestProduct",  # Product name
                "A test product", # Product description
                "pmf",          # Product stage
                "y"             # Want enrichment
            ]

            await agent._run_essentials_phase()

            # Check data was collected
            assert agent.state.company_data["name"] == "TestCompany"
            assert agent.state.product_data["name"] == "TestProduct"
            assert agent.state.current_phase == OnboardingPhase.ENRICHMENT
            assert "company_name" in agent.state.completed_steps
            assert "product_name" in agent.state.completed_steps

    @pytest.mark.asyncio
    async def test_run_advanced_phase(self, mock_config, temp_context_dir):
        """Test running advanced phase."""
        agent = OnboardingAgent(
            config=mock_config,
            context_dir=temp_context_dir,
            resume=False
        )

        # Mock the prompt session
        with patch.object(agent.session, 'prompt_async') as mock_prompt:
            # Set up responses for advanced setup
            mock_prompt.side_effect = [
                "y",    # Set up team?
                "10",   # Team size
                "6",    # Engineers
                "2",    # PMs
                "2",    # Designers
                "n",    # Add OKRs?
                "n",    # Add market details?
            ]

            await agent._run_advanced_phase()

            # Check team data was collected
            assert agent.state.team_data["size"] == 10
            assert agent.state.team_data["roles"]["engineers"] == 6
            assert "team_setup" in agent.state.completed_steps
            assert "okr_setup" in agent.state.skipped_steps
            assert "market_setup" in agent.state.skipped_steps
            assert agent.state.current_phase == OnboardingPhase.COMPLETE

    @pytest.mark.asyncio
    async def test_full_onboarding_flow(self, mock_config, temp_context_dir, mock_grounding):
        """Test complete onboarding flow."""
        agent = OnboardingAgent(
            config=mock_config,
            context_dir=temp_context_dir,
            grounding=mock_grounding,
            resume=False
        )

        # Mock all user inputs
        with patch.object(agent.session, 'prompt_async') as mock_prompt:
            mock_prompt.side_effect = [
                # Essentials
                "TestCorp",         # Company name
                "b2b",              # Business type
                "growth",           # Company stage
                "testcorp.com",     # Domain
                "TestProduct",      # Product name
                "Testing product",  # Product description
                "pmf",              # Product stage
                "n",                # Skip enrichment
                "n",                # Skip advanced
            ]

            # Mock the context save
            with patch.object(agent, '_save_context', return_value=True):
                success, context = await agent.run()

                assert success is True
                assert context is not None
                assert context.company.name == "TestCorp"
                assert context.product.name == "TestProduct"
                assert agent.state.is_complete

    @pytest.mark.asyncio
    async def test_cancellation_handling(self, mock_config, temp_context_dir):
        """Test handling of cancellation (Ctrl+C)."""
        agent = OnboardingAgent(
            config=mock_config,
            context_dir=temp_context_dir,
            resume=False
        )

        # Simulate cancellation
        agent.cancelled = True

        with patch.object(agent.session, 'prompt_async'):
            success, context = await agent.run()

            assert success is False
            assert context is None

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_config, temp_context_dir):
        """Test error handling during onboarding."""
        agent = OnboardingAgent(
            config=mock_config,
            context_dir=temp_context_dir,
            resume=False
        )

        # Mock an error during essentials phase
        with patch.object(agent, '_run_essentials_phase', side_effect=Exception("Test error")):
            success, context = await agent.run()

            assert success is False
            assert context is None


class TestRunOnboarding:
    """Test the synchronous wrapper function."""

    def test_run_onboarding_sync(self, mock_config, temp_context_dir):
        """Test synchronous onboarding wrapper."""
        # Mock the async agent
        with patch('pmkit.agents.onboarding.OnboardingAgent') as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.run = AsyncMock(return_value=(True, Mock(spec=Context)))

            # Run synchronous wrapper
            success, context = run_onboarding(
                config=mock_config,
                context_dir=temp_context_dir,
                resume=False,
                skip_enrichment=False
            )

            assert success is True
            assert context is not None

    def test_run_onboarding_with_grounding_error(self, mock_config, temp_context_dir):
        """Test onboarding when grounding initialization fails."""
        # Mock grounding initialization to fail
        with patch('pmkit.agents.onboarding.GroundingAdapter', side_effect=Exception("No API key")):
            with patch('pmkit.agents.onboarding.OnboardingAgent') as MockAgent:
                mock_agent = MockAgent.return_value
                mock_agent.run = AsyncMock(return_value=(True, Mock(spec=Context)))

                # Should still work without grounding
                success, context = run_onboarding(
                    config=mock_config,
                    context_dir=temp_context_dir,
                    resume=False,
                    skip_enrichment=False
                )

                # Agent should be created with grounding=None
                MockAgent.assert_called_with(
                    config=mock_config,
                    context_dir=temp_context_dir,
                    grounding=None,
                    resume=False
                )

    def test_run_onboarding_keyboard_interrupt(self, mock_config, temp_context_dir):
        """Test handling keyboard interrupt during onboarding."""
        with patch('pmkit.agents.onboarding.OnboardingAgent') as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.run = AsyncMock(side_effect=KeyboardInterrupt())

            # Run synchronous wrapper
            success, context = run_onboarding(
                config=mock_config,
                context_dir=temp_context_dir,
                resume=False,
                skip_enrichment=True
            )

            assert success is False
            assert context is None

    def test_run_onboarding_general_error(self, mock_config, temp_context_dir):
        """Test handling general errors during onboarding."""
        with patch('pmkit.agents.onboarding.OnboardingAgent') as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.run = AsyncMock(side_effect=Exception("Unexpected error"))

            # Run synchronous wrapper
            success, context = run_onboarding(
                config=mock_config,
                context_dir=temp_context_dir,
                resume=False,
                skip_enrichment=True
            )

            assert success is False
            assert context is None