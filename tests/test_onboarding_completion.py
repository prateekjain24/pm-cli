"""
Tests for the enhanced onboarding completion experience.

Tests value demonstration, metrics calculation, and delightful completion features.
"""

import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml
from rich.console import Console

from pmkit.agents.completion import (
    CompletionMetrics,
    calculate_completion_metrics,
    calculate_completeness_score,
    calculate_enrichment_coverage,
    calculate_okr_confidence,
    count_data_points,
    suggest_next_actions,
    time_saved_message,
)
from pmkit.agents.onboarding import OnboardingAgent
from pmkit.context.models import (
    CompanyContext,
    Context,
    KeyResult,
    MarketContext,
    Objective,
    OKRContext,
    ProductContext,
    TeamContext,
)


class TestCompletionMetrics:
    """Test the CompletionMetrics class and its methods."""

    def test_metrics_properties(self):
        """Test metrics property calculations."""
        metrics = CompletionMetrics(
            time_spent=120.0,
            data_points_collected=25,
            enrichment_coverage=0.75,
            okr_confidence=80.0,
            completeness_score=0.85
        )

        assert metrics.annual_hours_saved == 39  # 45 * 52 / 60
        assert metrics.work_weeks_saved == pytest.approx(0.975, rel=0.01)

    def test_format_time_spent(self):
        """Test time formatting for different durations."""
        # Seconds
        metrics = CompletionMetrics(
            time_spent=45.5,
            data_points_collected=10,
            enrichment_coverage=0.5,
            okr_confidence=60,
            completeness_score=0.7
        )
        assert metrics.format_time_spent() == "45.5s"

        # Minutes
        metrics.time_spent = 150
        assert metrics.format_time_spent() == "2.5m"

        # Hours
        metrics.time_spent = 3700
        assert metrics.format_time_spent() == "1.0h"

    def test_format_coverage_bar(self):
        """Test visual coverage bar formatting."""
        metrics = CompletionMetrics(
            time_spent=60,
            data_points_collected=10,
            enrichment_coverage=0.85,
            okr_confidence=70,
            completeness_score=0.8
        )

        bar = metrics.format_coverage_bar()
        assert "████████" in bar  # 8 filled blocks for 85%
        assert "85%" in bar
        assert "Excellent" in bar

        # Test partial coverage
        metrics.enrichment_coverage = 0.45
        bar = metrics.format_coverage_bar()
        assert "████" in bar  # 4 filled blocks for 45%
        assert "45%" in bar
        assert "Partial" in bar

    def test_completeness_emoji(self):
        """Test emoji selection based on completeness."""
        metrics = CompletionMetrics(
            time_spent=60,
            data_points_collected=10,
            enrichment_coverage=0.5,
            okr_confidence=70,
            completeness_score=0.95
        )
        assert metrics.get_completeness_emoji() == "🌟"

        metrics.completeness_score = 0.75
        assert metrics.get_completeness_emoji() == "✅"

        metrics.completeness_score = 0.55
        assert metrics.get_completeness_emoji() == "🟡"

        metrics.completeness_score = 0.3
        assert metrics.get_completeness_emoji() == "⚠️"


class TestMetricsCalculation:
    """Test metrics calculation functions."""

    def test_count_data_points(self):
        """Test counting non-empty data points in context."""
        context = Context(
            company=CompanyContext(
                name="TestCorp",
                type="b2b",
                stage="growth",
                domain="test.com",
                description="Test company",
                target_market="Enterprise"
            ),
            product=ProductContext(
                name="TestProduct",
                description="Product description",
                category="SaaS",
                main_metric="MRR",
                target_audience="Enterprises"
            ),
            market=MarketContext(
                competitors=["Comp1", "Comp2", "Comp3"],
                differentiator="Better UX",
                industry="Tech",
                market_size="$1B"
            ),
            team=TeamContext(
                size=10,
                roles={"PM": 2, "Engineer": 6, "Designer": 2}
            ),
            okrs=OKRContext(
                quarter="Q1 2025",
                objectives=[
                    Objective(
                        title="Increase revenue",
                        key_results=[
                            KeyResult(description="Hit $1M MRR", confidence=80),
                            KeyResult(description="Close 10 enterprise deals", confidence=70)
                        ]
                    )
                ]
            )
        )

        count = count_data_points(context)
        assert count >= 20  # Should count all the populated fields

    def test_calculate_enrichment_coverage(self):
        """Test enrichment coverage calculation."""
        context = Context(
            company=CompanyContext(
                name="TestCorp",
                type="b2b",
                stage="growth",
                domain="test.com",
                target_market="Enterprise",
                description="A long description that indicates enrichment"
            ),
            product=ProductContext(
                name="TestProduct",
                description="Product"
            ),
            market=MarketContext(
                competitors=["C1", "C2", "C3", "C4"],
                industry="Tech"
            )
        )

        # With enrichment
        coverage = calculate_enrichment_coverage(context, enrichment_used=True)
        assert coverage > 0.5  # Should have decent coverage

        # Without enrichment
        coverage = calculate_enrichment_coverage(context, enrichment_used=False)
        assert coverage == 0.0

    def test_calculate_okr_confidence(self):
        """Test OKR confidence calculation."""
        okrs = OKRContext(
            quarter="Q1 2025",
            objectives=[
                Objective(
                    title="Obj1",
                    key_results=[
                        KeyResult(description="KR1 description", confidence=80),
                        KeyResult(description="KR2 description", confidence=60),
                        KeyResult(description="KR3 description", confidence=70)
                    ]
                ),
                Objective(
                    title="Obj2",
                    key_results=[
                        KeyResult(description="KR4 description", confidence=90)
                    ]
                )
            ]
        )

        avg_confidence = calculate_okr_confidence(okrs)
        assert avg_confidence == pytest.approx(75.0, rel=0.1)  # (80+60+70+90)/4

        # Test with no OKRs
        assert calculate_okr_confidence(None) == 0.0

    def test_calculate_completeness_score(self):
        """Test overall completeness score calculation."""
        # Full context
        context = Context(
            company=CompanyContext(
                name="TestCorp",
                type="b2b",
                stage="growth"
            ),
            product=ProductContext(
                name="TestProduct",
                description="Description",
                main_metric="MRR"
            ),
            market=MarketContext(
                competitors=["C1", "C2"],
                differentiator="Better"
            ),
            team=TeamContext(
                size=10,
                roles={"PM": 1}
            ),
            okrs=OKRContext(
                quarter="Q1",
                objectives=[
                    Objective(title="Obj1", key_results=[]),
                    Objective(title="Obj2", key_results=[]),
                    Objective(title="Obj3", key_results=[])
                ]
            )
        )

        score = calculate_completeness_score(context)
        assert score > 0.7  # Should have high completeness

        # Minimal context
        minimal_context = Context(
            company=CompanyContext(name="Test", type="b2b", stage="seed"),
            product=ProductContext(name="Prod", description="Desc")
        )

        minimal_score = calculate_completeness_score(minimal_context)
        assert minimal_score < 0.5  # Should have low completeness


class TestNextActions:
    """Test next action suggestion logic."""

    def test_suggest_next_actions_b2b(self):
        """Test action suggestions for B2B companies."""
        context = Context(
            company=CompanyContext(
                name="B2BCorp",
                type="b2b",
                stage="growth"
            ),
            product=ProductContext(
                name="API Platform",
                description="Enterprise API management"
            )
        )

        metrics = CompletionMetrics(
            time_spent=60,
            data_points_collected=15,
            enrichment_coverage=0.5,
            okr_confidence=0,
            completeness_score=0.6
        )

        actions = suggest_next_actions(context, metrics)

        # Should suggest B2B-focused PRD
        assert any("API" in action[0] for action in actions)

        # Should suggest enrichment since coverage is low
        assert any("enrich" in action[0] for action in actions)

    def test_suggest_next_actions_b2c(self):
        """Test action suggestions for B2C companies."""
        context = Context(
            company=CompanyContext(
                name="B2CApp",
                type="b2c",
                stage="seed"
            ),
            product=ProductContext(
                name="Mobile App",
                description="Consumer mobile app"
            )
        )

        metrics = CompletionMetrics(
            time_spent=60,
            data_points_collected=20,
            enrichment_coverage=0.8,
            okr_confidence=50,
            completeness_score=0.7
        )

        actions = suggest_next_actions(context, metrics)

        # Should suggest B2C-focused PRD
        assert any("Onboarding" in action[0] for action in actions)

        # Should suggest MVP features for seed stage
        assert any("MVP" in action[0] for action in actions)

    def test_time_saved_message(self):
        """Test motivational time saved message generation."""
        metrics = CompletionMetrics(
            time_spent=45,
            data_points_collected=20,
            enrichment_coverage=0.7,
            okr_confidence=70,
            completeness_score=0.8
        )

        message = time_saved_message(metrics)
        assert "less than a minute" in message
        assert "39 hours" in message  # annual hours saved
        assert "work weeks" in message


class TestOnboardingCompletion:
    """Test the enhanced OnboardingAgent completion methods."""

    @pytest.fixture
    def mock_console(self):
        """Create a mock console."""
        console = MagicMock(spec=Console)
        console.print = Mock()
        return console

    @pytest.fixture
    def agent(self, tmp_path, mock_console):
        """Create an OnboardingAgent instance for testing."""
        context_dir = tmp_path / ".pmkit" / "context"
        context_dir.mkdir(parents=True)

        agent = OnboardingAgent(
            console=mock_console,
            context_dir=context_dir
        )
        agent.start_time = time.time() - 120  # 2 minutes ago
        agent.enrichment_used = True
        agent.searches_performed = 3

        return agent

    @pytest.fixture
    def sample_context(self):
        """Create a sample context for testing."""
        return Context(
            company=CompanyContext(
                name="TestCo",
                type="b2b",
                stage="growth",
                domain="test.com",
                description="Test company description"
            ),
            product=ProductContext(
                name="TestProduct",
                description="A test product for testing",
                main_metric="MRR",
                category="SaaS"
            ),
            market=MarketContext(
                competitors=["Competitor1", "Competitor2", "Competitor3"],
                differentiator="Better testing",
                industry="Technology"
            ),
            team=TeamContext(
                size=10,
                roles={"PM": 2, "Engineer": 6, "Designer": 2}
            ),
            okrs=OKRContext(
                quarter="Q1 2025",
                objectives=[
                    Objective(
                        title="Improve testing",
                        key_results=[
                            KeyResult(description="Achieve 100% coverage", confidence=80),
                            KeyResult(description="Reduce bugs by 50%", confidence=70)
                        ]
                    )
                ]
            )
        )

    def test_create_initialization_marker(self, agent, sample_context):
        """Test creation of initialization marker file."""
        metrics = CompletionMetrics(
            time_spent=120,
            data_points_collected=25,
            enrichment_coverage=0.75,
            okr_confidence=75,
            completeness_score=0.85
        )

        agent._create_initialization_marker(metrics)

        marker_path = agent.context_dir.parent / '.initialized'
        assert marker_path.exists()

        # Check marker content
        with open(marker_path) as f:
            marker_data = yaml.safe_load(f)

        assert 'initialized_at' in marker_data
        assert marker_data['version'] == '0.3.0'
        assert marker_data['setup_time_seconds'] == 120
        assert marker_data['completeness_score'] == 0.85

    def test_generate_context_card(self, agent, sample_context):
        """Test generation of shareable context card."""
        card_path = agent._generate_context_card(sample_context)

        assert card_path is not None
        assert card_path.exists()
        assert card_path.name == 'context-card.md'

        # Check card content
        content = card_path.read_text()
        assert "TestCo Product Context" in content
        assert "TestProduct" in content
        assert "B2B" in content
        assert "MRR" in content
        assert "Competitor1" in content
        assert "Improve testing" in content

    def test_display_value_panel(self, agent, mock_console):
        """Test value panel display."""
        metrics = CompletionMetrics(
            time_spent=120,
            data_points_collected=25,
            enrichment_coverage=0.75,
            okr_confidence=75,
            completeness_score=0.85
        )

        agent._display_value_panel(metrics)

        # Check that console.print was called
        assert mock_console.print.called

        # Check that value information was included
        call_args = str(mock_console.print.call_args_list)
        assert "Value Created" in call_args or "Time Savings" in call_args

    def test_display_context_highlights_b2b(self, agent, sample_context, mock_console):
        """Test context highlights display for B2B."""
        agent._display_context_highlights(sample_context)

        assert mock_console.print.called
        call_args = str(mock_console.print.call_args_list)

        # B2B specific content
        assert "ROI" in call_args or "Enterprise" in call_args or "Technical PRDs" in call_args

    def test_display_context_highlights_b2c(self, agent, mock_console):
        """Test context highlights display for B2C."""
        b2c_context = Context(
            company=CompanyContext(
                name="B2CApp",
                type="b2c",
                stage="growth"
            ),
            product=ProductContext(
                name="Consumer App",
                description="Mobile app for consumers",
                main_metric="MAU"
            )
        )

        agent._display_context_highlights(b2c_context)

        assert mock_console.print.called
        call_args = str(mock_console.print.call_args_list)

        # B2C specific content should be present
        assert "Engagement" in call_args or "Retention" in call_args or "Growth PRDs" in call_args

    def test_display_next_actions(self, agent, sample_context, mock_console):
        """Test next actions display."""
        metrics = CompletionMetrics(
            time_spent=120,
            data_points_collected=25,
            enrichment_coverage=0.75,
            okr_confidence=75,
            completeness_score=0.85
        )

        agent._display_next_actions(sample_context, metrics)

        assert mock_console.print.called
        call_args = str(mock_console.print.call_args_list)
        assert "Recommended Next Steps" in call_args

    def test_show_activation_prompt(self, agent, sample_context, mock_console):
        """Test activation prompt display."""
        agent._show_activation_prompt(sample_context)

        assert mock_console.print.called
        call_args = str(mock_console.print.call_args_list)
        assert "Ready to" in call_args or "pm new prd" in call_args

    @patch('pmkit.agents.onboarding.ContextValidator')
    def test_show_completion_full_flow(self, mock_validator_class, agent, sample_context, mock_console):
        """Test the complete _show_completion flow."""
        # Setup mock validator
        mock_validator = MagicMock()
        mock_validator.validate.return_value = (True, [])
        mock_validator_class.return_value = mock_validator

        # Run completion
        agent._show_completion(sample_context)

        # Check that all components were called
        assert mock_validator.validate.called

        # Check that initialization marker was created
        marker_path = agent.context_dir.parent / '.initialized'
        assert marker_path.exists()

        # Check that context card was created
        card_path = agent.context_dir.parent / 'exports' / 'context-card.md'
        assert card_path.exists()

        # Check that console.print was called multiple times for panels
        assert mock_console.print.call_count >= 3  # At least 3 panels + activation prompt