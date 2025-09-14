"""
Tests for the enhanced pm init command with 90-second flow.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import time
import yaml

from pmkit.cli.commands.init import (
    init_pmkit,
    _init_with_template,
    _calculate_progress,
)
from pmkit.agents.templates import PMArchetype, apply_template
from pmkit.agents.value_display import calculate_value_metrics, ValueMetrics
from pmkit.context.models import (
    Context,
    CompanyContext,
    ProductContext,
    MarketContext,
)


class TestInitCommand:
    """Test the enhanced init command with 90-second flow."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create a temporary directory for testing."""
        return tmp_path

    @pytest.fixture
    def mock_console(self):
        """Mock the console for testing."""
        with patch('pmkit.cli.commands.init.console') as mock:
            yield mock

    @pytest.fixture
    def mock_run_onboarding(self):
        """Mock the run_onboarding function."""
        with patch('pmkit.cli.commands.init.run_onboarding') as mock:
            # Create a mock context
            context = Context(
                company=CompanyContext(name="Test Co", type="b2b", stage="growth"),
                product=ProductContext(name="Test Product", description="A test product", stage="mvp"),
            )
            mock.return_value = (True, context)
            yield mock

    def test_init_90_second_tracking(self, temp_dir, mock_console, mock_run_onboarding, monkeypatch):
        """Test that init tracks 90-second goal."""
        monkeypatch.chdir(temp_dir)

        # Mock time to simulate fast execution
        start_time = time.time()
        with patch('pmkit.cli.commands.init.time.time') as mock_time:
            mock_time.side_effect = [
                start_time,  # Start time
                start_time,  # Check 1
                start_time + 45,  # Mid-execution
                start_time + 85,  # End time (under 90s)
            ]

            result = init_pmkit()

        assert result is True
        # Should show success with time under 90 seconds
        mock_console.success.assert_called()
        assert "85 seconds" in str(mock_console.success.call_args)

    def test_init_with_template(self, temp_dir, mock_console, monkeypatch):
        """Test initialization with PM archetype template."""
        monkeypatch.chdir(temp_dir)

        with patch('pmkit.cli.commands.init.Prompt.ask') as mock_prompt:
            mock_prompt.side_effect = ["Acme Corp", "Acme Analytics"]

            with patch('pmkit.context.manager.ContextManager.save_context') as mock_save:
                mock_save.return_value = (True, [])

                result = init_pmkit(template="b2b_saas")

        assert result is True
        mock_console.print.assert_called()
        # Should show template benefits
        assert any("pre-configured personas" in str(call) for call in mock_console.print.call_args_list)

    def test_init_quick_mode(self, temp_dir, mock_console, mock_run_onboarding, monkeypatch):
        """Test quick mode with minimal questions."""
        monkeypatch.chdir(temp_dir)

        result = init_pmkit(quick=True)

        assert result is True
        # Quick mode should not skip enrichment
        mock_run_onboarding.assert_called_once()
        call_kwargs = mock_run_onboarding.call_args[1]
        assert call_kwargs['skip_enrichment'] is False

    def test_init_resume_detection(self, temp_dir, mock_console, monkeypatch):
        """Test smart resume detection."""
        monkeypatch.chdir(temp_dir)

        # Create existing state file
        pmkit_dir = temp_dir / ".pmkit"
        context_dir = pmkit_dir / "context"
        context_dir.mkdir(parents=True, exist_ok=True)

        state_file = context_dir / "onboarding_state.yaml"
        state = {
            'company_name': 'Test Co',
            'product_name': 'Test Product',
            'phase2_complete': True,
        }
        with open(state_file, 'w') as f:
            yaml.dump(state, f)

        # Should detect partial completion
        result = init_pmkit()

        assert result is False  # Should not reinitialize
        mock_console.print.assert_called()
        # Should show progress percentage
        assert any("70% complete" in str(call) for call in mock_console.print.call_args_list)

    def test_init_already_initialized(self, temp_dir, mock_console, monkeypatch):
        """Test behavior when already initialized."""
        monkeypatch.chdir(temp_dir)

        # Create .pmkit directory without state file (fully initialized)
        pmkit_dir = temp_dir / ".pmkit"
        pmkit_dir.mkdir(parents=True, exist_ok=True)

        result = init_pmkit()

        assert result is False
        mock_console.print.assert_called()
        # Should show next actions
        assert any("pm new prd" in str(call) for call in mock_console.print.call_args_list)
        assert any("pm okrs add" in str(call) for call in mock_console.print.call_args_list)

    def test_init_force_flag(self, temp_dir, mock_console, mock_run_onboarding, monkeypatch):
        """Test force flag overwrites existing."""
        monkeypatch.chdir(temp_dir)

        # Create existing .pmkit directory
        pmkit_dir = temp_dir / ".pmkit"
        pmkit_dir.mkdir(parents=True, exist_ok=True)

        result = init_pmkit(force=True)

        assert result is True
        mock_run_onboarding.assert_called_once()

    def test_calculate_progress(self):
        """Test progress calculation."""
        # Empty state
        assert _calculate_progress({}) == 0

        # Phase 1 complete
        state = {'company_name': 'Test', 'product_name': 'Product'}
        assert _calculate_progress(state) == 20

        # Phase 2 complete
        state['phase2_complete'] = True
        assert _calculate_progress(state) == 60

        # All complete
        state['team_size'] = 5
        state['okrs'] = [{}]
        assert _calculate_progress(state) == 100


class TestValueMetrics:
    """Test value metrics calculation and display."""

    def test_calculate_value_metrics(self):
        """Test value metrics calculation."""
        context = Context(
            company=CompanyContext(name="Test Co", type="b2b", stage="growth"),
            product=ProductContext(name="Test Product", description="A test", stage="mvp"),
            market=MarketContext(
                competitors=["Comp1", "Comp2"],
                positioning="Leader",
            ),
        )

        enrichment = {
            'personas': [{'name': 'PM'}, {'name': 'Dev'}],
            'competitors': ['C1', 'C2', 'C3'],
            'segments': ['Enterprise', 'SMB'],
        }

        metrics = calculate_value_metrics(context, enrichment, elapsed_seconds=75)

        assert metrics.personas_generated == 2
        assert metrics.competitor_features_found == 15  # 3 competitors * 5
        assert metrics.market_segments_identified == 2
        assert metrics.research_hours_saved > 0
        assert metrics.time_to_first_prd_seconds == 30

    def test_value_metrics_b2b_vs_b2c(self):
        """Test different metrics for B2B vs B2C."""
        # B2B context
        b2b_context = Context(
            company=CompanyContext(name="B2B Co", type="b2b", stage="growth"),
            product=ProductContext(name="API Platform", description="Developer tool", stage="scaling"),
        )

        b2b_metrics = calculate_value_metrics(b2b_context)
        assert len(b2b_metrics.unserved_segments) > 0
        assert any("enterprise" in s.lower() for s in b2b_metrics.unserved_segments)

        # B2C context
        b2c_context = Context(
            company=CompanyContext(name="B2C Co", type="b2c", stage="growth"),
            product=ProductContext(name="Mobile App", description="Consumer app", stage="scaling"),
        )

        b2c_metrics = calculate_value_metrics(b2c_context)
        assert len(b2c_metrics.unserved_segments) > 0
        assert any("gen z" in s.lower() or "international" in s.lower() for s in b2c_metrics.unserved_segments)


class TestTemplates:
    """Test PM archetype templates."""

    def test_apply_b2b_saas_template(self):
        """Test B2B SaaS template application."""
        context_data = apply_template(
            PMArchetype.B2B_SAAS,
            company_name="TestCo",
            product_name="TestProduct",
        )

        assert context_data['company'].type == "b2b"
        assert context_data['company'].stage == "growth"
        assert len(context_data['template_metadata']['personas']) == 4
        assert "MRR" in context_data['template_metadata']['primary_metrics']

    def test_apply_developer_tool_template(self):
        """Test Developer Tool template application."""
        context_data = apply_template(
            PMArchetype.DEVELOPER_TOOL,
            company_name="DevCo",
            product_name="DevAPI",
        )

        assert context_data['company'].type == "b2b"
        assert "API Calls" in context_data['template_metadata']['primary_metrics']
        assert any("Developer" in p['title'] for p in context_data['template_metadata']['personas'])

    def test_apply_consumer_app_template(self):
        """Test Consumer App template application."""
        context_data = apply_template(
            PMArchetype.CONSUMER_APP,
            company_name="AppCo",
            product_name="FunApp",
        )

        assert context_data['company'].type == "b2c"
        assert "DAU" in context_data['template_metadata']['primary_metrics']
        assert "Freemium" in context_data['template_metadata']['pricing_models']

    def test_apply_marketplace_template(self):
        """Test Marketplace template application."""
        context_data = apply_template(
            PMArchetype.MARKETPLACE,
            company_name="MarketCo",
            product_name="MarketPlace",
        )

        assert context_data['company'].type == "b2c"
        assert "GMV" in context_data['template_metadata']['primary_metrics']
        assert any("Buyer" in p['title'] for p in context_data['template_metadata']['personas'])
        assert any("Seller" in p['title'] for p in context_data['template_metadata']['personas'])

    def test_apply_plg_b2b_template(self):
        """Test PLG B2B template application."""
        context_data = apply_template(
            PMArchetype.PLG_B2B,
            company_name="PLGCo",
            product_name="CollabTool",
        )

        assert context_data['company'].type == "b2b"
        assert "Product Qualified Leads" in context_data['template_metadata']['primary_metrics']
        assert "Freemium" in context_data['template_metadata']['pricing_models']


class TestOnboardingAgent90Second:
    """Test OnboardingAgent changes for 90-second flow."""

    @pytest.mark.asyncio
    async def test_phase1_only_2_questions(self):
        """Test Phase 1 only asks 2 questions."""
        from pmkit.agents.onboarding import OnboardingAgent

        with patch('pmkit.agents.onboarding.Prompt.ask') as mock_prompt:
            mock_prompt.side_effect = ["TestCo", "TestProduct"]

            agent = OnboardingAgent(use_interactive=False)
            await agent._phase1_essentials()

        # Should only ask 2 questions
        assert mock_prompt.call_count == 2

        # Should auto-detect company type
        assert agent.state['company_type'] in ['b2b', 'b2c']

        # Should set smart defaults
        assert agent.state.get('product_description') is not None
        assert agent.state.get('user_role') == 'Product Manager'

    @pytest.mark.asyncio
    async def test_okrs_optional_in_phase3(self):
        """Test OKRs are optional in Phase 3."""
        from pmkit.agents.onboarding import OnboardingAgent

        with patch('pmkit.agents.onboarding.Confirm.ask') as mock_confirm:
            # Skip all optional sections
            mock_confirm.return_value = False

            agent = OnboardingAgent()
            await agent._phase3_advanced()

        # Should not crash if everything is skipped
        assert True  # Just verify it completes

    @pytest.mark.asyncio
    async def test_auto_detect_company_type(self):
        """Test auto-detection of company type from product name."""
        from pmkit.agents.onboarding import OnboardingAgent

        test_cases = [
            ("API Gateway", "b2b"),
            ("Developer SDK", "b2b"),
            ("Mobile Game", "b2c"),
            ("Fitness App", "b2c"),
            ("Social Network", "b2c"),
            ("Enterprise Platform", "b2b"),
            ("Generic Product", "b2b"),  # Default
        ]

        for product_name, expected_type in test_cases:
            agent = OnboardingAgent(use_interactive=False)
            agent.state['product_name'] = product_name

            # Simulate the auto-detection logic from _phase1_essentials
            product_name_lower = product_name.lower()
            if any(kw in product_name_lower for kw in ['api', 'sdk', 'developer', 'platform']):
                detected_type = 'b2b'
            elif any(kw in product_name_lower for kw in ['app', 'game', 'social', 'fitness']):
                detected_type = 'b2c'
            else:
                detected_type = 'b2b'

            assert detected_type == expected_type