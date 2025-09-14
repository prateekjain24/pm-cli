"""
EnrichmentService - Smart, agentic company data enrichment for onboarding.

This module implements intelligent web search enrichment with the 3-2 Rule:
- Primary search (1): Dense keyword-packed comprehensive search
- Adaptive searches (2): Critical gaps only
- Reserve pool (2): User-triggered or high-value opportunities

Features:
- Smart decision tree for follow-up searches
- 70% coverage threshold for stopping
- Confidence scoring (HIGH/MEDIUM/LOW)
- Real-time progress updates
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from pmkit.llm.grounding import GroundingAdapter
from pmkit.llm.models import SearchResult
from pmkit.llm.search.base import SearchOptions
from pmkit.utils.logger import get_logger

logger = get_logger(__name__)


class EnrichmentResult(BaseModel):
    """Result of company enrichment with coverage metrics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: Dict[str, Any] = Field(default_factory=dict)
    coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    searches_used: int = Field(default=0, ge=0, le=5)
    remaining_searches: int = Field(default=5, ge=0, le=5)
    confidence_scores: Dict[str, str] = Field(default_factory=dict)  # field -> HIGH/MEDIUM/LOW
    search_history: List[str] = Field(default_factory=list)


class ParsedField(BaseModel):
    """A single parsed field with value and confidence."""

    value: Any
    confidence: str = "LOW"  # HIGH, MEDIUM, LOW
    sources: int = 1
    last_updated: Optional[str] = None


# Phased enrichment models for structured extraction
class Phase1Enrichment(BaseModel):
    """Phase 1 - Product description enrichment."""
    product_description: Optional[str] = Field(None, description="Clear description of what the product does")


class Phase2Enrichment(BaseModel):
    """Phase 2 - Core business information."""
    company_stage: Optional[str] = Field(None, description="Seed, Series A-E, IPO, Acquired")
    target_market: Optional[str] = Field(None, description="Who are their customers")
    competitors: List[str] = Field(default_factory=list, description="Main competitor names")
    north_star_metric: Optional[str] = Field(None, description="Their primary success metric")


class Phase3Enrichment(BaseModel):
    """Phase 3 - Additional details."""
    website: Optional[str] = Field(None, description="Company website URL")
    team_size: Optional[str] = Field(None, description="Number of employees")
    pricing_model: Optional[str] = Field(None, description="How they charge customers")
    user_count: Optional[str] = Field(None, description="Number of users/customers")
    key_differentiator: Optional[str] = Field(None, description="What makes them unique")


class FullEnrichment(BaseModel):
    """Combined model for single extraction of all phases."""
    # Phase 1
    product_description: Optional[str] = None
    # Phase 2
    company_stage: Optional[str] = None
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
    Smart enrichment service with agentic behavior.

    Implements the 3-2 Rule for optimal search allocation and
    stops at 70% coverage to save searches for later.
    """

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
        "product_understanding": ["core_offering", "target_customer"],
        "market_position": ["competitors"],  # OR market_size, but competitors preferred
        "unique_insights": ["funding"],  # OR team OR recent_news
    }

    # Field weights for coverage calculation
    FIELD_WEIGHTS = {
        "company_basics": 0.3,
        "product_understanding": 0.3,
        "market_position": 0.2,
        "unique_insights": 0.2,
    }

    def __init__(self, grounding: GroundingAdapter):
        """
        Initialize the enrichment service.

        Args:
            grounding: GroundingAdapter for web search
        """
        self.grounding = grounding
        self.current_year = str(datetime.now().year)

    async def enrich_company(
        self,
        company_info: Dict[str, str],
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> EnrichmentResult:
        """
        Enrich company data with intelligent search strategy.

        Args:
            company_info: Basic company information (name, domain, description, type)
            progress_callback: Optional callback for progress updates

        Returns:
            EnrichmentResult with data, coverage, and search metrics
        """
        logger.info(f"Starting enrichment for {company_info.get('name')}")

        # Initialize result
        result = EnrichmentResult()

        # Primary search with dense keywords
        if progress_callback:
            await progress_callback("🔍 Running primary search...", 10)

        primary_data = await self._primary_search(company_info)
        result.data.update(primary_data)
        result.searches_used = 1
        result.search_history.append("primary")

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
                    f"✅ Sufficient coverage ({coverage*100:.0f}%), saving searches",
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
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> EnrichmentResult:
        """
        Perform adaptive follow-up searches based on gaps.

        Args:
            result: Current enrichment result
            company_info: Company information
            progress_callback: Progress callback

        Returns:
            Updated enrichment result
        """
        searches_used = result.searches_used

        while searches_used < self.MAX_ACTIVE_SEARCHES:
            # Decide next search based on gaps
            next_search = self._decide_next_search(result.data, company_info)

            if not next_search:
                logger.info("No critical gaps found, stopping adaptive search")
                break

            # Update progress
            if progress_callback:
                progress_percent = 30 + (searches_used * 20)
                await progress_callback(
                    f"🔎 Searching for {next_search}...",
                    progress_percent
                )

            # Execute targeted search
            search_data = await self._execute_targeted_search(
                next_search,
                company_info,
                result.data
            )

            # Merge results
            if search_data:
                result.data = self._merge_results(result.data, search_data)
                result.search_history.append(next_search)

            searches_used += 1
            result.searches_used = searches_used

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

        Implements PM expert's priority matrix.

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
            Merged data
        """
        merged = current.copy()

        for key, value in new.items():
            if key not in merged:
                # New field
                merged[key] = value
            elif isinstance(value, dict) and "confidence" in value:
                # Compare confidence scores
                current_confidence = self._get_confidence_score(merged.get(key))
                new_confidence = self._get_confidence_score(value)

                if new_confidence > current_confidence:
                    merged[key] = value
            elif value and not merged[key]:
                # Fill empty field
                merged[key] = value

        return merged

    def _get_confidence_score(self, value: Any) -> float:
        """
        Get numeric confidence score for comparison.

        Args:
            value: Field value (may include confidence)

        Returns:
            Numeric confidence score
        """
        if isinstance(value, dict) and "confidence" in value:
            confidence_map = {"HIGH": 0.9, "MEDIUM": 0.5, "LOW": 0.2}
            return confidence_map.get(value["confidence"], 0.1)
        return 0.1


class EnrichmentParser:
    """
    Parser for extracting structured data from search results.

    Uses regex patterns and heuristics to extract company information
    with confidence scoring.
    """

    # Extraction patterns organized by search type
    EXTRACTION_PATTERNS = {
        "primary": {
            "industry": [
                r"(?:industry|sector|vertical)[:\s]+([^,\.\n]+)",
                r"operates in (?:the\s+)?([^,\.\n]+)(?:\s+industry)?",
                r"([A-Z][a-z]+(?:Tech|FinTech|EdTech|HealthTech|SaaS|Commerce))",
            ],
            "competitors": [
                r"competes with\s+([^,\.\n]+(?:,\s*[^,\.\n]+)*)",
                r"alternatives?\s+(?:to\s+\w+\s+)?(?:include|are)\s+([^,\.\n]+(?:,\s*[^,\.\n]+)*)",
                r"versus\s+([^,\.\n]+)",
                r"vs\.?\s+([^,\.\n]+)",
            ],
            "funding": [
                r"\$(\d+(?:\.\d+)?)\s*([MBK])\s*(?:in\s+)?(?:funding|raised|series)",
                r"raised\s+\$(\d+(?:\.\d+)?)\s*([MBK])",
                r"Series\s+([A-E])\s+(?:funding|round)",
            ],
            "business_model": [
                r"(B2B|B2C|B2B2C|SaaS|marketplace|platform|subscription)",
                r"business model[:\s]+([^,\.\n]+)",
                r"(enterprise|consumer|SMB|mid-market)\s+(?:focused|customers|clients)",
            ],
            "target_customer": [
                r"(?:serves|for|targets?)\s+(enterprises?|startups?|SMBs?|consumers?|developers?)",
                r"(?:built|designed|created)\s+for\s+([^,\.\n]+)",
                r"(small businesses?|large companies|fortune \d+|teams)",
            ],
        },
        "competitors": {
            "competitors": [
                r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+(?:is\s+)?(?:a\s+)?competitor",
                r"competes?\s+with\s+([^,\.\n]+(?:,\s*[^,\.\n]+)*)",
                r"similar\s+(?:to|companies|products)[:\s]+([^,\.\n]+(?:,\s*[^,\.\n]+)*)",
                r"alternatives?[:\s]+([^,\.\n]+(?:,\s*[^,\.\n]+)*)",
            ],
            "market_position": [
                r"(leader|challenger|niche player|innovator)\s+in",
                r"(#\d+|top \d+|leading|fastest growing)",
                r"market share[:\s]+(\d+(?:\.\d+)?%)",
            ],
        },
        "pricing": {
            "pricing_model": [
                r"(freemium|subscription|usage-based|enterprise|pay-per-use|one-time)",
                r"pricing\s+(?:model|strategy)[:\s]+([^,\.\n]+)",
                r"(free tier|free plan|trial|demo)\s+available",
            ],
            "pricing_tiers": [
                r"\$(\d+(?:\.\d+)?)\s*(?:per|/)\s*(?:month|year|user|seat)",
                r"starts?\s+at\s+\$(\d+(?:\.\d+)?)",
                r"(?:from|pricing\s+from)\s+\$(\d+(?:\.\d+)?)",
                r"(free|starter|pro|enterprise|team|business)\s+(?:plan|tier|pricing)",
            ],
        },
        "tech_stack": {
            "tech_stack": [
                r"built\s+(?:with|on|using)\s+([^,\.\n]+(?:,\s*[^,\.\n]+)*)",
                r"(?:uses|utilizes|leverages)\s+([^,\.\n]+(?:,\s*[^,\.\n]+)*)\s+for",
                r"(?:tech\s+)?stack[:\s]+([^,\.\n]+(?:,\s*[^,\.\n]+)*)",
                r"powered\s+by\s+([^,\.\n]+)",
            ],
            "platforms": [
                r"available\s+on\s+([^,\.\n]+(?:,\s*[^,\.\n]+)*)",
                r"(web|mobile|iOS|Android|desktop|API|SDK)",
                r"supports?\s+([^,\.\n]+(?:,\s*[^,\.\n]+)*)",
            ],
            "integrations": [
                r"integrates?\s+with\s+([^,\.\n]+(?:,\s*[^,\.\n]+)*)",
                r"connects?\s+to\s+([^,\.\n]+(?:,\s*[^,\.\n]+)*)",
                r"works?\s+with\s+([^,\.\n]+(?:,\s*[^,\.\n]+)*)",
            ],
        },
        "recent": {
            "recent_news": [
                r"announced?\s+([^,\.\n]+)",
                r"launched?\s+([^,\.\n]+)",
                r"(?:recently|just)\s+([^,\.\n]+)",
            ],
            "partnerships": [
                r"partner(?:ed|ship)?\s+with\s+([^,\.\n]+)",
                r"collaboration\s+with\s+([^,\.\n]+)",
                r"joined\s+forces\s+with\s+([^,\.\n]+)",
            ],
            "company_stage": [
                r"(seed|series [A-E]|growth|mature|public|IPO)",
                r"(\d+)\s+employees?",
                r"founded\s+(?:in\s+)?(\d{4})",
            ],
        },
    }

    def parse_search_results(
        self,
        content: str,
        search_type: str
    ) -> Dict[str, Any]:
        """
        Parse search results based on search type.

        Args:
            content: Search result content
            search_type: Type of search performed

        Returns:
            Parsed data with confidence scores
        """
        patterns = self.EXTRACTION_PATTERNS.get(search_type, {})
        extracted = {}

        for field, pattern_list in patterns.items():
            matches = []

            for pattern in pattern_list:
                found = re.findall(pattern, content, re.IGNORECASE)
                if found:
                    # Flatten tuples if regex has groups
                    for match in found:
                        if isinstance(match, tuple):
                            matches.append(" ".join(str(m) for m in match if m))
                        else:
                            matches.append(str(match))

            if matches:
                # Clean and deduplicate matches
                cleaned_matches = self._clean_matches(matches)

                if cleaned_matches:
                    confidence = self._calculate_confidence(
                        cleaned_matches,
                        content,
                        field
                    )

                    # Store as single value or list
                    value = cleaned_matches[0] if len(cleaned_matches) == 1 else cleaned_matches

                    extracted[field] = {
                        "value": value,
                        "confidence": confidence,
                        "sources": len(set(cleaned_matches)),
                    }

        return extracted

    def _clean_matches(self, matches: List[str]) -> List[str]:
        """
        Clean and deduplicate extracted matches.

        Args:
            matches: Raw matches from regex

        Returns:
            Cleaned and deduplicated matches
        """
        cleaned = []

        for match in matches:
            # Clean up the match
            match = match.strip()
            match = re.sub(r'\s+', ' ', match)  # Normalize whitespace

            # Split on common separators for list fields
            if ',' in match or ' and ' in match:
                parts = re.split(r',\s*|\s+and\s+', match)
                cleaned.extend([p.strip() for p in parts if p.strip()])
            else:
                if match and match not in cleaned:
                    cleaned.append(match)

        # Remove duplicates while preserving order
        seen = set()
        deduped = []
        for item in cleaned:
            if item.lower() not in seen:
                seen.add(item.lower())
                deduped.append(item)

        return deduped[:5]  # Limit to top 5 matches

    def _calculate_confidence(
        self,
        matches: List[str],
        content: str,
        field: str
    ) -> str:
        """
        Calculate confidence score for extracted data.

        Args:
            matches: Extracted matches
            content: Full content
            field: Field being extracted

        Returns:
            Confidence level (HIGH/MEDIUM/LOW)
        """
        score = 0.0

        # Check source authority
        authority_domains = [
            'crunchbase', 'pitchbook', 'techcrunch', 'forbes',
            'bloomberg', 'reuters', 'official', '.com'
        ]
        if any(domain in content.lower() for domain in authority_domains):
            score += 0.4

        # Multiple sources agreement
        if len(matches) == 1:
            # Single clear match
            score += 0.2
        elif len(set(matches)) == 1 and len(matches) > 1:
            # Same value from multiple extractions
            score += 0.3

        # Recency check
        current_year = str(datetime.now().year)
        last_year = str(datetime.now().year - 1)
        if current_year in content or last_year in content:
            score += 0.2

        # Field-specific confidence boosters
        if field == "competitors" and len(matches) >= 2:
            score += 0.1  # Multiple competitors found
        elif field == "funding" and "$" in str(matches):
            score += 0.1  # Monetary value found
        elif field == "business_model" and matches[0].upper() in ["B2B", "B2C", "SAAS"]:
            score += 0.1  # Clear business model

        # Determine confidence level
        if score >= 0.7:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"