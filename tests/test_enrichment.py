"""
Tests for the EnrichmentService with agentic behavior.

Tests the 3-2 Rule, coverage calculation, confidence scoring,
and smart search decision logic.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pmkit.agents.enrichment import (
    EnrichmentParser,
    EnrichmentResult,
    EnrichmentService,
    ParsedField,
)
from pmkit.llm.grounding import GroundingAdapter
from pmkit.llm.models import SearchResult


class TestEnrichmentService:
    """Test the EnrichmentService with agentic behavior."""

    @pytest.fixture
    def mock_grounding(self):
        """Create a mock GroundingAdapter."""
        grounding = MagicMock(spec=GroundingAdapter)
        grounding.search = AsyncMock()
        return grounding

    @pytest.fixture
    def service(self, mock_grounding):
        """Create an EnrichmentService instance."""
        return EnrichmentService(mock_grounding)

    @pytest.fixture
    def company_info(self):
        """Sample company information."""
        return {
            'name': 'Linear',
            'domain': 'linear.app',
            'description': 'Issue tracking for software teams',
            'type': 'b2b',
            'product_name': 'Linear',
        }

    @pytest.mark.asyncio
    async def test_stops_at_70_percent_coverage(self, service, mock_grounding, company_info):
        """Test that enrichment stops at 70% coverage threshold."""
        # Mock primary search with high coverage data - make it more parseable
        mock_grounding.search.return_value = SearchResult(
            query="Linear linear.app company overview",
            content="""
            Linear operates in the project management software industry.
            Business model: B2B SaaS subscription.
            Linear competes with Jira, Asana, and Monday.com in the issue tracking space.
            Built for software development teams and startups.
            Pricing: starts at $8 per user per month with freemium tier available.
            Tech stack: Built with React, GraphQL, and Node.js.
            Founded in 2019, Series B funding of $50M.
            """,
            citations=[]
        )

        result = await service.enrich_company(company_info)

        # Should only use 1 search (primary) if coverage >= 70%
        assert result.searches_used == 1
        assert result.coverage >= 0.7
        assert mock_grounding.search.call_count == 1
        assert result.remaining_searches == 4  # 5 - 1 = 4

    @pytest.mark.asyncio
    async def test_max_3_searches(self, service, mock_grounding, company_info):
        """Test that enrichment never uses more than 3 searches."""
        # Mock searches with low coverage to trigger adaptive searches
        mock_grounding.search.return_value = SearchResult(
            query="Linear company search",
            content="Linear is a company.",  # Minimal data
            citations=[]
        )

        result = await service.enrich_company(company_info)

        # Should use max 3 searches even with low coverage
        assert result.searches_used <= 3
        assert mock_grounding.search.call_count <= 3
        assert result.remaining_searches >= 2  # At least 2 reserved

    @pytest.mark.asyncio
    async def test_adaptive_search_for_competitors(self, service, mock_grounding, company_info):
        """Test that adaptive search fetches competitors when missing."""
        # Primary search without competitors but with other good data
        primary_result = SearchResult(
            query="Linear linear.app company overview",
            content="""
            Linear is a B2B SaaS project management tool.
            Industry: Software development tools.
            Target customers: software teams and startups.
            Business model: subscription based.
            Founded in 2019.
            """,
            citations=[]
        )

        # Competitors search with results
        competitors_result = SearchResult(
            query="Linear vs alternatives",
            content="""
            Linear competes with Jira, Asana, and Trello.
            Alternatives to Linear include Monday.com and ClickUp.
            Linear vs Jira comparison shows Linear is faster.
            """,
            citations=[]
        )

        mock_grounding.search.side_effect = [primary_result, competitors_result]

        result = await service.enrich_company(company_info)

        # Should trigger competitors search for B2B company without competitors
        assert result.searches_used == 2
        assert mock_grounding.search.call_count == 2

        # Check that competitors were found
        assert 'competitors' in result.data

    @pytest.mark.asyncio
    async def test_coverage_calculation(self, service):
        """Test coverage calculation based on weighted fields."""
        # Test with no data
        assert service._calculate_coverage({}) == 0.0

        # Test with partial data (company basics only)
        data = {
            'company_name': 'Linear',
            'industry': 'Software',
            'business_model': 'B2B',
        }
        coverage = service._calculate_coverage(data)
        assert 0.2 < coverage < 0.4  # Should be around 30%

        # Test with more complete data
        data.update({
            'core_offering': 'Issue tracking',
            'target_customer': 'Dev teams',
            'competitors': ['Jira', 'Asana'],
            'funding': 'Series B',
        })
        coverage = service._calculate_coverage(data)
        assert coverage > 0.7  # Should be above threshold

    @pytest.mark.asyncio
    async def test_progress_callback(self, service, mock_grounding, company_info):
        """Test that progress callbacks are invoked correctly."""
        mock_grounding.search.return_value = SearchResult(
            query="Linear company search",
            content="Linear data with high coverage.",
            citations=[]
        )

        progress_updates = []

        async def track_progress(message: str, percent: int):
            progress_updates.append((message, percent))

        result = await service.enrich_company(
            company_info,
            progress_callback=track_progress
        )

        # Should have progress updates
        assert len(progress_updates) > 0
        assert any('primary' in msg.lower() for msg, _ in progress_updates)
        assert any(pct == 100 for _, pct in progress_updates)

    def test_decide_next_search_b2b_no_competitors(self, service):
        """Test decision logic prioritizes competitors for B2B without them."""
        current_data = {'industry': 'Software'}
        company_info = {'type': 'b2b', 'description': 'CRM software'}

        next_search = service._decide_next_search(current_data, company_info)

        assert next_search == "competitors"

    def test_decide_next_search_saas_no_pricing(self, service):
        """Test decision logic prioritizes pricing for SaaS without it."""
        current_data = {'competitors': ['A', 'B']}
        company_info = {'type': 'b2b', 'description': 'SaaS platform'}

        next_search = service._decide_next_search(current_data, company_info)

        assert next_search == "pricing"

    def test_decide_next_search_sufficient_data(self, service):
        """Test decision logic returns None when sufficient data exists."""
        current_data = {
            'company_name': 'Test',
            'industry': 'Tech',
            'business_model': 'B2B',
            'competitors': ['A', 'B'],
            'target_customer': 'Enterprises',
            'funding': 'Series C',
        }
        company_info = {'type': 'b2b'}

        # Calculate coverage to verify it's high
        coverage = service._calculate_coverage(current_data)
        assert coverage > 0.6

        next_search = service._decide_next_search(current_data, company_info)

        # Should not need more searches
        assert next_search is None

    def test_merge_results_prefers_high_confidence(self, service):
        """Test that merge prefers higher confidence data."""
        current = {
            'industry': {'value': 'Tech', 'confidence': 'LOW'}
        }
        new = {
            'industry': {'value': 'Software', 'confidence': 'HIGH'}
        }

        merged = service._merge_results(current, new)

        # Should prefer HIGH confidence
        assert merged['industry']['value'] == 'Software'
        assert merged['industry']['confidence'] == 'HIGH'


class TestEnrichmentParser:
    """Test the EnrichmentParser for data extraction."""

    @pytest.fixture
    def parser(self):
        """Create an EnrichmentParser instance."""
        return EnrichmentParser()

    def test_parse_competitors(self, parser):
        """Test extraction of competitors from content."""
        content = """
        Linear competes with Jira, Asana, and Monday.com in the project
        management space. It's also an alternative to Trello.
        """

        result = parser.parse_search_results(content, "primary")

        assert 'competitors' in result
        competitors = result['competitors']['value']
        assert isinstance(competitors, list)
        assert 'Jira' in competitors
        assert 'Asana' in competitors

    def test_parse_pricing(self, parser):
        """Test extraction of pricing information."""
        content = """
        Linear pricing starts at $8 per user per month for the team plan.
        Enterprise pricing available. They offer a free tier for small teams.
        """

        result = parser.parse_search_results(content, "pricing")

        assert 'pricing_tiers' in result
        assert 'pricing_model' in result

        # Check that price was extracted
        pricing = result['pricing_tiers']['value']
        assert '$8' in str(pricing) or '8' in str(pricing)

    def test_parse_funding(self, parser):
        """Test extraction of funding information."""
        content = """
        Linear raised $35M in Series B funding led by Sequoia Capital.
        The company was founded in 2019.
        """

        result = parser.parse_search_results(content, "primary")

        assert 'funding' in result
        funding = result['funding']['value']
        assert '35' in str(funding) or 'Series B' in str(funding)

    def test_confidence_scoring_high(self, parser):
        """Test HIGH confidence scoring with authoritative source."""
        content = """
        According to Crunchbase, Linear has raised $52M in total funding.
        TechCrunch reported in 2024 that Linear is valued at $400M.
        """

        result = parser.parse_search_results(content, "primary")

        if 'funding' in result:
            assert result['funding']['confidence'] == 'HIGH'

    def test_confidence_scoring_medium(self, parser):
        """Test MEDIUM confidence scoring."""
        content = """
        Linear appears to be a growing startup in the project management space.
        Some sources suggest they have significant funding.
        """

        result = parser.parse_search_results(content, "primary")

        # With vague language, confidence should not be HIGH
        for field, data in result.items():
            if isinstance(data, dict) and 'confidence' in data:
                assert data['confidence'] in ['MEDIUM', 'LOW']

    def test_clean_matches_deduplication(self, parser):
        """Test that matches are cleaned and deduplicated."""
        matches = ['Jira', 'jira', 'Asana', 'asana', 'Jira']

        cleaned = parser._clean_matches(matches)

        # Should deduplicate case-insensitively
        assert len(cleaned) == 2
        assert 'Jira' in cleaned or 'jira' in cleaned
        assert 'Asana' in cleaned or 'asana' in cleaned

    def test_tech_stack_extraction(self, parser):
        """Test extraction of technology stack."""
        content = """
        Linear is built with React for the frontend and uses GraphQL for APIs.
        The platform runs on AWS and integrates with GitHub, Slack, and Figma.
        """

        result = parser.parse_search_results(content, "tech_stack")

        assert 'tech_stack' in result
        tech = result['tech_stack']['value']
        assert 'React' in str(tech) or 'GraphQL' in str(tech)

        assert 'integrations' in result
        integrations = result['integrations']['value']
        assert 'GitHub' in str(integrations) or 'Slack' in str(integrations)


class TestEnrichmentResult:
    """Test the EnrichmentResult model."""

    def test_enrichment_result_defaults(self):
        """Test EnrichmentResult default values."""
        result = EnrichmentResult()

        assert result.data == {}
        assert result.coverage == 0.0
        assert result.searches_used == 0
        assert result.remaining_searches == 5
        assert result.confidence_scores == {}
        assert result.search_history == []

    def test_enrichment_result_validation(self):
        """Test EnrichmentResult field validation."""
        # Coverage must be between 0 and 1
        result = EnrichmentResult(coverage=0.75)
        assert result.coverage == 0.75

        # Searches must be between 0 and 5
        result = EnrichmentResult(searches_used=3, remaining_searches=2)
        assert result.searches_used == 3
        assert result.remaining_searches == 2

    def test_enrichment_result_with_data(self):
        """Test EnrichmentResult with actual data."""
        data = {
            'competitors': {'value': ['Jira', 'Asana'], 'confidence': 'HIGH'},
            'industry': {'value': 'Software', 'confidence': 'MEDIUM'},
        }
        confidence_scores = {
            'competitors': 'HIGH',
            'industry': 'MEDIUM',
        }

        result = EnrichmentResult(
            data=data,
            coverage=0.65,
            searches_used=2,
            remaining_searches=3,
            confidence_scores=confidence_scores,
            search_history=['primary', 'competitors']
        )

        assert result.data == data
        assert result.coverage == 0.65
        assert result.searches_used == 2
        assert result.search_history == ['primary', 'competitors']


@pytest.mark.asyncio
class TestEnrichmentIntegration:
    """Integration tests for enrichment with OnboardingAgent."""

    async def test_enrichment_with_onboarding_flow(self):
        """Test enrichment integration with onboarding flow."""
        # This would test the full integration with OnboardingAgent
        # Placeholder for integration test
        pass