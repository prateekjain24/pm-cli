"""
Enrichment service for company and product information.

This module provides automated web search and information extraction
capabilities to enrich company context with minimal user input.

The enrichment process uses progressive enhancement:
1. Primary search for comprehensive overview
2. Adaptive searches based on gaps
3. Structured extraction with confidence scoring
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

from pmkit.config.models import LLMProviderConfig
from pmkit.exceptions import PMKitError
from pmkit.llm.grounding import GroundingAdapter
from pmkit.llm.search.base import SearchOptions
from pmkit.utils.logger import get_logger
from openai import OpenAI

logger = get_logger(__name__)


class EnrichmentResult(BaseModel):
    """Result from enrichment process."""

    data: Dict[str, Any] = Field(default_factory=dict)
    searches_used: int = 0
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    coverage: float = 0.0
    cached: bool = False
    remaining_searches: int = 0


class ParsedField(BaseModel):
    """A parsed field with confidence score."""

    value: Any
    confidence: str = "LOW"  # HIGH, MEDIUM, LOW
    sources: int = 1


# Pydantic models for structured extraction
class Phase1Enrichment(BaseModel):
    """Phase 1: Core product description."""
    product_description: Optional[str] = None


class Phase2Enrichment(BaseModel):
    """Phase 2: Core business information."""
    company_stage: Optional[Literal["idea", "seed", "growth", "mature"]] = None
    target_market: Optional[str] = None
    competitors: List[str] = Field(default_factory=list)
    north_star_metric: Optional[str] = None


class Phase3Enrichment(BaseModel):
    """Phase 3: Additional details."""
    website: Optional[str] = None
    team_size: Optional[str] = None
    pricing_model: Optional[str] = None
    user_count: Optional[str] = None
    key_differentiator: Optional[str] = None


class FullEnrichment(BaseModel):
    """Full enrichment data model."""
    # Phase 1
    product_description: Optional[str] = None

    # Phase 2
    company_stage: Optional[Literal["idea", "seed", "growth", "mature"]] = None
    target_market: Optional[str] = None
    competitors: List[str] = Field(default_factory=list)
    north_star_metric: Optional[str] = None

    # Phase 3
    website: Optional[str] = None
    team_size: Optional[str] = None
    pricing_model: Optional[str] = None
    user_count: Optional[str] = None
    key_differentiator: Optional[str] = None


class EnrichmentService:
    """
    Service for enriching company information via web search.

    Implements PM expert's phased approach:
    1. Primary dense search for broad coverage
    2. Adaptive targeted searches for gaps
    3. Smart stopping based on coverage thresholds
    """

    # Search configuration from PM expert
    MAX_SEARCHES = 5
    STOP_THRESHOLD = 0.7  # 70% coverage threshold
    MAX_ACTIVE_SEARCHES = 3  # Primary + 2 adaptive

    # Search templates from prompt-engineer
    SEARCH_TEMPLATES = {
        "primary": (
            "{company_name} {company_domain} company overview competitors "
            "pricing model target customers technology stack {product_description} "
            "{business_model} market position valuation funding"
        ),
        "product": (
            '"{company_name}" what is {product_name} product description features '
            "capabilities use cases value proposition {product_description}"
        ),
        "business_model": (
            "{company_name} company stage startup growth scale funding series "
            "revenue model pricing plans subscription business model"
        ),
        "market": (
            "{company_name} target market customer segments {business_model} "
            "demographics industries verticals buyer persona"
        ),
        "competitors": (
            '{company_name} vs alternatives comparison "{product_category}" '
            "competing products similar to {company_name} {industry} landscape players"
        ),
        "metrics": (
            "{company_name} north star metric KPIs ARR revenue users customers "
            "growth rate retention churn MRR user count DAU MAU"
        ),
        "pricing": (
            '{company_name} pricing plans subscription tiers freemium enterprise sales '
            '"{product_name}" cost ROI case studies customer segments'
        ),
        "recent": (
            '{company_name} {current_year} announcement funding round product launch '
            'partnership acquisition "latest news" key differentiator unique value'
        ),
    }

    # Required fields for minimum viable context (MVC)
    REQUIRED_FIELDS = {
        "company_basics": ["company_name", "industry", "business_model"],
        "product": ["product_name", "product_description", "value_proposition"],
        "market": ["target_customer", "competitors"],
        "metrics": ["users", "revenue", "funding"],
    }

    # Field weights for coverage calculation
    FIELD_WEIGHTS = {
        "company_basics": 0.3,
        "product": 0.3,
        "market": 0.25,
        "metrics": 0.15,
    }

    def __init__(self, grounding: GroundingAdapter):
        """
        Initialize enrichment service.

        Args:
            grounding: Web search adapter
        """
        self.grounding = grounding
        self.cache_dir = Path.home() / ".pmkit" / "cache" / "enrichment"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.current_year = str(datetime.now().year)

    async def enrich_company(
        self,
        company_info: Dict[str, str],
        progress_callback: Optional[Callable] = None,
    ) -> EnrichmentResult:
        """
        Enrich company information with web search.

        Implements the PM expert's adaptive search strategy.

        Args:
            company_info: Basic company information
            progress_callback: Optional callback for progress updates

        Returns:
            EnrichmentResult with enriched data and metadata
        """
        logger.info(f"Starting enrichment for {company_info.get('name', 'Unknown')}")

        # Check cache first
        cache_key = self._generate_cache_key(company_info)
        cached_result = self._load_cache(cache_key)
        if cached_result:
            logger.info("Using cached enrichment result")
            return cached_result

        # Initialize result
        result = EnrichmentResult()

        # Phase 1: Primary dense search
        if progress_callback:
            await progress_callback("🔍 Running primary search...", 10)

        primary_data = await self._primary_search(company_info)
        result.data.update(primary_data)
        result.searches_used = 1

        # Calculate initial coverage
        coverage = self._calculate_coverage(result.data)
        result.coverage = coverage

        if progress_callback:
            await progress_callback(
                f"📊 Initial coverage: {coverage*100:.0f}%",
                30
            )

        # Check if we need adaptive searches
        if coverage >= self.STOP_THRESHOLD:
            logger.info(f"Stopping at {coverage*100:.0f}% coverage (threshold: {self.STOP_THRESHOLD*100}%)")
            if progress_callback:
                await progress_callback(
                    f"✅ Enrichment complete",
                    100
                )
            result.remaining_searches = self.MAX_SEARCHES - result.searches_used
            return result

        # Perform adaptive searches (max 2 more)
        result = await self._adaptive_search(
            result,
            company_info,
            progress_callback
        )

        # Final coverage calculation
        result.coverage = self._calculate_coverage(result.data)
        result.remaining_searches = self.MAX_SEARCHES - result.searches_used

        if progress_callback:
            await progress_callback(
                f"✅ Enrichment complete: {result.coverage*100:.0f}% coverage",
                100
            )

        logger.info(
            f"Enrichment complete: {result.coverage*100:.0f}% coverage, "
            f"{result.searches_used} searches used"
        )

        return result

    async def _primary_search(self, company_info: Dict[str, str]) -> Dict[str, Any]:
        """
        Execute primary search with dense keyword packing.

        Args:
            company_info: Company information

        Returns:
            Parsed data from primary search
        """
        # Build primary search query
        query = self.SEARCH_TEMPLATES["primary"].format(
            company_name=company_info.get("name", ""),
            company_domain=company_info.get("domain", ""),
            product_description=company_info.get("description", ""),
            business_model=company_info.get("type", "B2B").upper(),
            product_category="",  # Will be inferred
            industry="",  # Will be discovered
            product_name=company_info.get("product_name", company_info.get("name", "")),
            current_year=self.current_year,
        )

        # Execute search
        try:
            search_result = await self.grounding.search(
                query.strip(),
                SearchOptions(max_results=5)
            )

            if search_result and search_result.content:
                # Parse the results
                return self._parse_search_results(
                    search_result.content,
                    "primary"
                )

        except Exception as e:
            logger.warning(f"Primary search failed: {e}")

        return {}

    async def _adaptive_search(
        self,
        result: EnrichmentResult,
        company_info: Dict[str, str],
        progress_callback: Optional[Callable] = None,
    ) -> EnrichmentResult:
        """
        Perform adaptive searches based on gaps.

        Args:
            result: Current enrichment result
            company_info: Company information
            progress_callback: Optional progress callback

        Returns:
            Updated EnrichmentResult
        """
        max_adaptive = min(2, self.MAX_SEARCHES - result.searches_used)
        progress_step = 50

        for i in range(max_adaptive):
            # Decide what to search for next
            search_type = self._decide_next_search(
                result.data,
                company_info
            )

            if not search_type:
                logger.info("No more searches needed")
                break

            if progress_callback:
                await progress_callback(
                    f"🔎 Searching for {search_type}...",
                    progress_step + (i * 20)
                )

            # Execute targeted search
            search_data = await self._execute_targeted_search(
                search_type,
                company_info,
                result.data
            )

            # Merge results
            if search_data:
                result.data = self._merge_results(result.data, search_data)
                result.searches_used += 1

            # Recalculate coverage
            coverage = self._calculate_coverage(result.data)
            result.coverage = coverage

            # Check if we've reached sufficient coverage
            if coverage >= self.STOP_THRESHOLD:
                logger.info(f"Reached {coverage*100:.0f}% coverage, stopping")
                break

        return result

    def _decide_next_search(
        self,
        current_data: Dict[str, Any],
        company_info: Dict[str, str]
    ) -> Optional[str]:
        """
        Decide which search to perform next based on gaps.

        Implements phased approach priority.

        Args:
            current_data: Currently collected data
            company_info: Company information

        Returns:
            Next search type or None if sufficient data
        """
        # Phase 1: Product description (most critical)
        if not current_data.get("product_description"):
            return "product"

        # Phase 2: Core business info
        if not current_data.get("company_stage"):
            return "business_model"
        if not current_data.get("target_market"):
            return "market"
        if not current_data.get("competitors"):
            return "competitors"
        if not current_data.get("north_star_metric"):
            return "metrics"

        # Phase 3: Additional details
        if not current_data.get("pricing_model"):
            return "pricing"
        if not current_data.get("user_count"):
            return "metrics"
        if not current_data.get("key_differentiator"):
            return "competitors"

        # Check if we have minimum viable context
        coverage = self._calculate_coverage(current_data)
        if coverage < 0.4:
            # Too little data, try recent info
            return "recent"

        return None  # Have enough data

    async def _execute_targeted_search(
        self,
        search_type: str,
        company_info: Dict[str, str],
        current_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a targeted follow-up search.

        Args:
            search_type: Type of search to perform
            company_info: Company information
            current_data: Currently collected data

        Returns:
            Parsed search results
        """
        # Get the appropriate template
        template = self.SEARCH_TEMPLATES.get(search_type, "")
        if not template:
            return {}

        # Extract additional context from current data
        industry = current_data.get("industry", "")
        product_category = current_data.get("category", "")

        # Build search query
        query = template.format(
            company_name=company_info.get("name", ""),
            company_domain=company_info.get("domain", ""),
            product_description=company_info.get("description", ""),
            business_model=company_info.get("type", "B2B").upper(),
            product_category=product_category,
            industry=industry,
            product_name=company_info.get("product_name", company_info.get("name", "")),
            current_year=self.current_year,
        )

        # Execute search
        try:
            search_result = await self.grounding.search(
                query.strip(),
                SearchOptions(max_results=3)  # Fewer results for targeted searches
            )

            if search_result and search_result.content:
                return self._parse_search_results(
                    search_result.content,
                    search_type
                )

        except Exception as e:
            logger.warning(f"Targeted search '{search_type}' failed: {e}")

        return {}

    def _parse_search_results(
        self,
        content: str,
        search_type: str
    ) -> Dict[str, Any]:
        """
        Parse search results using GPT-5 structured output.

        Args:
            content: Search result content
            search_type: Type of search performed

        Returns:
            Parsed and structured data
        """
        client = OpenAI()

        # For primary search, get everything at once
        if search_type == "primary":
            try:
                response = client.responses.parse(
                    model="gpt-5",
                    input=[
                        {
                            "role": "system",
                            "content": """Extract company information from search results.
                            Focus on finding:
                            - What the product/company actually does (product_description)
                            - Company stage (funding, IPO, acquisition status)
                            - Target customers and market (target_market)
                            - Main competitors (competitors - company names only)
                            - Key metrics they track (north_star_metric)
                            - Website URL if mentioned (website)
                            - Number of employees or team size (team_size)
                            - Pricing structure or model (pricing_model)
                            - Number of users, customers, or MAU (user_count)
                            - Main differentiator or competitive advantage (key_differentiator)
                            Only include information explicitly mentioned in the text."""
                        },
                        {"role": "user", "content": f"Extract company info:\n{content[:8000]}"}
                    ],
                    text_format=FullEnrichment,
                    reasoning={"effort": "medium"}
                )

                result = response.output_parsed.model_dump(exclude_none=True)

                # Limit competitors to 5
                if 'competitors' in result and result['competitors']:
                    result['competitors'] = result['competitors'][:5]

                return result

            except Exception as e:
                logger.warning(f"GPT-5 extraction failed: {e}")
                return {}

        # For targeted searches, use simpler extraction or return empty
        else:
            # Could implement phase-specific extraction here if needed
            return {}

    def _calculate_coverage(self, data: Dict[str, Any]) -> float:
        """
        Calculate data coverage based on phased approach.

        Phase 1 (20%): Product description
        Phase 2 (40%): Core business info (stage, market, competitors, metric)
        Phase 3 (40%): Additional details (website, team, pricing, users, differentiator)

        Args:
            data: Current data

        Returns:
            Coverage score between 0 and 1
        """
        total_score = 0.0

        # Phase 1: Product description (20%)
        if data.get('product_description'):
            total_score += 0.20

        # Phase 2: Core business info (40% total, 10% each)
        phase2_fields = ['company_stage', 'target_market', 'competitors', 'north_star_metric']
        for field in phase2_fields:
            if data.get(field):
                total_score += 0.10

        # Phase 3: Additional details (40% total, 8% each)
        phase3_fields = ['website', 'team_size', 'pricing_model', 'user_count', 'key_differentiator']
        for field in phase3_fields:
            if data.get(field):
                total_score += 0.08

        return min(1.0, total_score)

    def _merge_results(
        self,
        current: Dict[str, Any],
        new: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge new search results with existing data.
        Prefers higher confidence data when conflicts exist.

        Args:
            current: Current data
            new: New data to merge

        Returns:
            Merged data dictionary
        """
        merged = current.copy()

        for key, value in new.items():
            if key not in merged:
                # New field, add it
                merged[key] = value
            elif isinstance(value, dict) and "confidence" in value:
                # Has confidence score, check if better
                current_conf = self._get_confidence_score(merged.get(key, {}))
                new_conf = self._get_confidence_score(value)

                if new_conf > current_conf:
                    merged[key] = value

        return merged

    def _generate_cache_key(self, company_info: Dict[str, str]) -> str:
        """Generate a cache key for company info."""
        import hashlib

        # Create a deterministic key from company info
        key_parts = [
            company_info.get("name", ""),
            company_info.get("domain", ""),
            company_info.get("type", ""),
        ]
        key_string = "|".join(key_parts).lower()
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    def _load_cache(self, cache_key: str) -> Optional[EnrichmentResult]:
        """Load cached result if available and fresh."""
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        # Check age (cache for 7 days)
        age_days = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
        if age_days > 7:
            return None

        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
                result = EnrichmentResult(**data)
                result.cached = True
                return result
        except Exception as e:
            logger.debug(f"Cache load failed: {e}")
            return None

    def _save_cache(self, cache_key: str, result: EnrichmentResult) -> None:
        """Save result to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            with open(cache_file, "w") as f:
                json.dump(result.model_dump(), f, indent=2)
        except Exception as e:
            logger.debug(f"Cache save failed: {e}")

    def _show_enrichment_summary(self, result: EnrichmentResult) -> None:
        """Display a summary of enrichment results."""
        from rich.table import Table
        from pmkit.utils.console import console

        table = Table(title="Enrichment Summary")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")
        table.add_column("Confidence", style="green")

        for field, value in result.data.items():
            if isinstance(value, dict) and "value" in value:
                table.add_row(
                    field,
                    str(value["value"])[:50],
                    value.get("confidence", "N/A")
                )
            else:
                table.add_row(field, str(value)[:50], "HIGH")

        console.print(table)

    def _get_confidence_score(self, value: Any) -> float:
        """
        Convert confidence level to numeric score.

        Args:
            value: Field value or dict with confidence

        Returns:
            Numeric confidence score
        """
        if isinstance(value, dict) and "confidence" in value:
            confidence_map = {"HIGH": 0.9, "MEDIUM": 0.5, "LOW": 0.2}
            return confidence_map.get(value["confidence"], 0.1)
        return 0.1