"""
PM Archetype Templates for Quick Start.

Provides pre-configured templates for common PM scenarios to enable
instant value delivery without lengthy setup.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

from pmkit.context.models import (
    CompanyContext,
    ProductContext,
    MarketContext,
    TeamContext,
    OKRContext,
    Objective,
    KeyResult,
)


class PMArchetype(Enum):
    """Available PM archetype templates."""

    B2B_SAAS = "b2b_saas"
    DEVELOPER_TOOL = "developer_tool"
    CONSUMER_APP = "consumer_app"
    MARKETPLACE = "marketplace"
    PLG_B2B = "plg_b2b"


@dataclass
class ArchetypeTemplate:
    """Template configuration for a PM archetype."""

    name: str
    description: str
    example_companies: List[str]

    # Pre-configured context elements
    company_type: str
    company_stage: str
    product_stage: str

    # Suggested metrics
    primary_metrics: List[str]
    secondary_metrics: List[str]

    # Common personas
    personas: List[Dict[str, str]]

    # Competitive landscape hints
    competitor_categories: List[str]

    # Sample OKRs
    sample_objectives: List[str]
    sample_key_results: List[str]

    # Focus areas for PRDs
    prd_focus_areas: List[str]

    # Pricing model hints
    pricing_models: List[str]


# Template Definitions
TEMPLATES = {
    PMArchetype.B2B_SAAS: ArchetypeTemplate(
        name="B2B SaaS",
        description="Enterprise software with subscription model",
        example_companies=["Slack", "Salesforce", "HubSpot"],
        company_type="b2b",
        company_stage="growth",
        product_stage="pmf",
        primary_metrics=["MRR", "ARR", "Net Revenue Retention", "CAC Payback"],
        secondary_metrics=["Churn Rate", "LTV:CAC", "Expansion Revenue", "ARPU"],
        personas=[
            {"title": "IT Administrator", "pain": "Managing user access and security"},
            {"title": "Team Lead", "pain": "Coordinating team collaboration"},
            {"title": "Executive Sponsor", "pain": "Proving ROI to leadership"},
            {"title": "End User", "pain": "Learning new tools quickly"},
        ],
        competitor_categories=["Legacy Enterprise", "Modern SaaS", "Point Solutions"],
        sample_objectives=[
            "Increase enterprise adoption",
            "Improve product stickiness",
            "Expand into new verticals",
        ],
        sample_key_results=[
            "Achieve $10M ARR",
            "Reach 120% net revenue retention",
            "Sign 50 enterprise customers",
            "Reduce churn to <5% monthly",
        ],
        prd_focus_areas=[
            "Enterprise security features",
            "Admin controls and permissions",
            "Integration ecosystem",
            "Scalability and performance",
            "Compliance (SOC2, GDPR)",
        ],
        pricing_models=["Per seat", "Tiered plans", "Usage-based", "Platform fee"],
    ),

    PMArchetype.DEVELOPER_TOOL: ArchetypeTemplate(
        name="Developer Tool",
        description="API-first platform for developers",
        example_companies=["Stripe", "Twilio", "GitHub", "Vercel"],
        company_type="b2b",
        company_stage="growth",
        product_stage="scale",
        primary_metrics=["API Calls", "Active Developers", "SDK Adoption", "Time to First Call"],
        secondary_metrics=["Documentation Views", "Support Tickets", "Community Size", "PR Contributions"],
        personas=[
            {"title": "Individual Developer", "pain": "Quick integration and testing"},
            {"title": "Engineering Lead", "pain": "Team adoption and standardization"},
            {"title": "DevOps Engineer", "pain": "Monitoring and reliability"},
            {"title": "CTO", "pain": "Build vs buy decisions"},
        ],
        competitor_categories=["Open Source", "Enterprise Vendors", "Cloud Providers"],
        sample_objectives=[
            "Become the developer standard",
            "Reduce integration time",
            "Build vibrant community",
        ],
        sample_key_results=[
            "10,000 active developers monthly",
            "Average integration time <30 minutes",
            "1M+ API calls daily",
            "500+ community contributors",
        ],
        prd_focus_areas=[
            "Developer experience (DX)",
            "API design and versioning",
            "SDK language coverage",
            "Documentation and examples",
            "Rate limiting and pricing",
        ],
        pricing_models=["Usage-based", "Free tier + paid", "Enterprise contracts"],
    ),

    PMArchetype.CONSUMER_APP: ArchetypeTemplate(
        name="Consumer App",
        description="Mobile-first consumer application",
        example_companies=["Spotify", "Instagram", "Duolingo", "Headspace"],
        company_type="b2c",
        company_stage="growth",
        product_stage="scale",
        primary_metrics=["DAU", "MAU", "Retention D1/D7/D30", "Session Length"],
        secondary_metrics=["App Store Rating", "Viral Coefficient", "ARPDAU", "Churn"],
        personas=[
            {"title": "Power User", "pain": "Advanced features and customization"},
            {"title": "Casual User", "pain": "Simple, intuitive experience"},
            {"title": "New User", "pain": "Quick value realization"},
            {"title": "Churned User", "pain": "Missing features or poor experience"},
        ],
        competitor_categories=["Direct Competitors", "Attention Competitors", "Substitutes"],
        sample_objectives=[
            "Increase user engagement",
            "Improve retention metrics",
            "Drive viral growth",
        ],
        sample_key_results=[
            "Reach 1M DAU",
            "Achieve 40% D30 retention",
            "4.5+ App Store rating",
            "20% of users refer a friend",
        ],
        prd_focus_areas=[
            "Onboarding experience",
            "Core loop optimization",
            "Social features",
            "Personalization",
            "Push notification strategy",
        ],
        pricing_models=["Freemium", "Subscription", "In-app purchases", "Ad-supported"],
    ),

    PMArchetype.MARKETPLACE: ArchetypeTemplate(
        name="Marketplace",
        description="Two-sided marketplace platform",
        example_companies=["Airbnb", "Uber", "Etsy", "DoorDash"],
        company_type="b2c",
        company_stage="growth",
        product_stage="pmf",
        primary_metrics=["GMV", "Take Rate", "Supply/Demand Ratio", "Market Liquidity"],
        secondary_metrics=["CAC by Side", "Repeat Rate", "Cross-side Network Effects", "Match Rate"],
        personas=[
            {"title": "Buyer", "pain": "Finding quality options quickly"},
            {"title": "Seller", "pain": "Reaching customers profitably"},
            {"title": "Power Seller", "pain": "Scaling their business"},
            {"title": "First-time Buyer", "pain": "Trust and safety concerns"},
        ],
        competitor_categories=["Direct Marketplaces", "Vertical Solutions", "DIY Alternatives"],
        sample_objectives=[
            "Balance supply and demand",
            "Increase transaction frequency",
            "Improve match quality",
        ],
        sample_key_results=[
            "$100M GMV monthly",
            "15% take rate",
            "80% successful match rate",
            "3x annual purchase frequency",
        ],
        prd_focus_areas=[
            "Search and discovery",
            "Trust and safety",
            "Pricing and incentives",
            "Supply acquisition",
            "Transaction flow",
        ],
        pricing_models=["Commission", "Listing fees", "Subscription", "Featured placement"],
    ),

    PMArchetype.PLG_B2B: ArchetypeTemplate(
        name="PLG B2B",
        description="Product-led growth B2B with bottom-up adoption",
        example_companies=["Figma", "Notion", "Airtable", "Miro"],
        company_type="b2b",
        company_stage="growth",
        product_stage="scale",
        primary_metrics=["Product Qualified Leads", "Time to Value", "Expansion Revenue", "Team Adoption"],
        secondary_metrics=["Viral Actions", "Workspace Creation", "Collaboration Metrics", "Feature Adoption"],
        personas=[
            {"title": "Individual User", "pain": "Personal productivity"},
            {"title": "Team Champion", "pain": "Getting team buy-in"},
            {"title": "Team Admin", "pain": "Managing team workspace"},
            {"title": "Decision Maker", "pain": "Evaluating for org-wide rollout"},
        ],
        competitor_categories=["Legacy Tools", "Point Solutions", "Platform Plays"],
        sample_objectives=[
            "Drive viral team adoption",
            "Accelerate land and expand",
            "Reduce time to value",
        ],
        sample_key_results=[
            "50% of users invite teammates",
            "Average team size 10+ users",
            "TTV reduced to <10 minutes",
            "30% of free teams convert to paid",
        ],
        prd_focus_areas=[
            "Single-player to multi-player",
            "Collaboration features",
            "Templates and examples",
            "Sharing and permissions",
            "Team onboarding",
        ],
        pricing_models=["Freemium", "Per seat", "Team tiers", "Enterprise custom"],
    ),
}


def get_template(archetype: PMArchetype) -> ArchetypeTemplate:
    """
    Get a specific archetype template.

    Args:
        archetype: The PM archetype to retrieve

    Returns:
        ArchetypeTemplate configuration
    """
    return TEMPLATES[archetype]


def list_templates() -> Dict[PMArchetype, str]:
    """
    List all available templates with descriptions.

    Returns:
        Dictionary of archetype to description
    """
    return {
        archetype: template.description
        for archetype, template in TEMPLATES.items()
    }


def apply_template(
    archetype: PMArchetype,
    company_name: str,
    product_name: str,
    product_description: Optional[str] = None,
) -> Dict:
    """
    Apply an archetype template to create initial context.

    Args:
        archetype: The PM archetype to apply
        company_name: Name of the company
        product_name: Name of the product
        product_description: Optional product description

    Returns:
        Dictionary with pre-filled context based on template
    """
    template = TEMPLATES[archetype]

    # Build company context
    company = CompanyContext(
        name=company_name,
        type=template.company_type,
        stage=template.company_stage,
        description=f"{company_name} - {template.description}",
    )

    # Build product context
    product = ProductContext(
        name=product_name,
        description=product_description or f"{product_name} - {template.description}",
        stage=template.product_stage,
        target_audience=", ".join([p["title"] for p in template.personas[:2]]),
        key_features=[],  # To be filled by PRD generation
        tech_stack=[],  # To be discovered
    )

    # Build market context
    market = MarketContext(
        industry=template.name,
        market_size="",  # To be enriched
        growth_rate="",  # To be enriched
        competitors=[],  # To be enriched
        positioning=f"{template.description} similar to {', '.join(template.example_companies[:2])}",
    )

    # Build team context with typical roles
    team = TeamContext(
        size=5,  # Default small team
        roles={
            "PM": 1,
            "Engineering": 3,
            "Design": 1,
        },
    )

    # Build sample OKRs
    objectives = []
    for i, obj_text in enumerate(template.sample_objectives[:2]):
        key_results = []
        for j, kr_text in enumerate(template.sample_key_results[i*2:(i+1)*2]):
            key_results.append(
                KeyResult(
                    description=kr_text,
                    target_value="",  # To be specified
                    current_value="",  # To be tracked
                    confidence=70,  # Default medium confidence
                )
            )
        objectives.append(
            Objective(
                title=obj_text,
                key_results=key_results,
            )
        )

    okrs = OKRContext(
        quarter="",  # To be specified
        year="",  # To be specified
        objectives=objectives,
    )

    return {
        "company": company,
        "product": product,
        "market": market,
        "team": team,
        "okrs": okrs,
        "template_metadata": {
            "archetype": archetype.value,
            "primary_metrics": template.primary_metrics,
            "personas": template.personas,
            "prd_focus_areas": template.prd_focus_areas,
            "pricing_models": template.pricing_models,
        },
    }


def get_template_by_name(name: str) -> Optional[ArchetypeTemplate]:
    """
    Get template by string name (for CLI).

    Args:
        name: Template name (e.g., "b2b_saas", "developer_tool")

    Returns:
        ArchetypeTemplate or None if not found
    """
    try:
        archetype = PMArchetype(name.lower())
        return TEMPLATES[archetype]
    except (KeyError, ValueError):
        return None


def format_template_choices() -> List[str]:
    """
    Format templates for CLI selection.

    Returns:
        List of formatted template choices
    """
    choices = []
    for archetype, template in TEMPLATES.items():
        examples = ", ".join(template.example_companies[:2])
        choices.append(
            f"{archetype.value}: {template.name} (like {examples})"
        )
    return choices