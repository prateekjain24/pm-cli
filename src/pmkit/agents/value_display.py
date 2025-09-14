"""
Value Metrics Display for PM-Kit.

Shows concrete PM deliverables and value created during onboarding,
not just generic time saved. Focuses on metrics that matter to PMs.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import timedelta

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.text import Text

from pmkit.context.models import Context


@dataclass
class ValueMetrics:
    """Concrete value metrics that resonate with PMs."""

    # Discovery metrics
    personas_generated: int = 0
    competitor_features_found: int = 0
    market_segments_identified: int = 0
    market_size_discovered: bool = False

    # Time savings
    research_hours_saved: float = 0.0
    time_to_first_prd_seconds: int = 30

    # Context completeness
    context_coverage_percent: int = 0
    enrichment_confidence: str = "LOW"  # LOW, MEDIUM, HIGH

    # Opportunities found
    unserved_segments: List[str] = None
    differentiator_opportunities: List[str] = None

    def __post_init__(self):
        if self.unserved_segments is None:
            self.unserved_segments = []
        if self.differentiator_opportunities is None:
            self.differentiator_opportunities = []


def calculate_value_metrics(
    context: Context,
    enrichment_result: Optional[Dict] = None,
    elapsed_seconds: int = 0,
) -> ValueMetrics:
    """
    Calculate concrete value metrics from context and enrichment.

    Args:
        context: The generated context
        enrichment_result: Results from enrichment service
        elapsed_seconds: Time taken for setup

    Returns:
        ValueMetrics with calculated values
    """
    metrics = ValueMetrics()

    # Count personas generated
    if enrichment_result and 'personas' in enrichment_result:
        metrics.personas_generated = len(enrichment_result.get('personas', []))
    elif context.market and hasattr(context.market, 'personas'):
        metrics.personas_generated = len(getattr(context.market, 'personas', []))

    # Count competitor features
    if enrichment_result and 'competitors' in enrichment_result:
        competitors = enrichment_result.get('competitors', [])
        # Estimate 5 features per competitor discovered
        metrics.competitor_features_found = len(competitors) * 5
    elif context.market and context.market.competitors:
        metrics.competitor_features_found = len(context.market.competitors) * 5

    # Count market segments
    if enrichment_result and 'segments' in enrichment_result:
        metrics.market_segments_identified = len(enrichment_result.get('segments', []))
    else:
        # Default segments based on company type
        if context.company and context.company.type == 'b2b':
            metrics.market_segments_identified = 3  # Enterprise, Mid-market, SMB
        else:
            metrics.market_segments_identified = 2  # Free users, Premium users

    # Check market size discovery
    if context.market and context.market.market_size:
        metrics.market_size_discovered = True

    # Calculate research time saved
    # Rough estimates:
    # - Each persona: 30 minutes of research
    # - Competitor analysis: 2 hours
    # - Market sizing: 3 hours
    persona_hours = metrics.personas_generated * 0.5
    competitor_hours = 2.0 if metrics.competitor_features_found > 0 else 0
    market_hours = 3.0 if metrics.market_size_discovered else 0
    metrics.research_hours_saved = persona_hours + competitor_hours + market_hours

    # Calculate context coverage
    required_fields = 0
    filled_fields = 0

    # Check company context
    if context.company:
        required_fields += 3  # name, type, stage
        if context.company.name:
            filled_fields += 1
        if context.company.type:
            filled_fields += 1
        if context.company.stage:
            filled_fields += 1

    # Check product context
    if context.product:
        required_fields += 3  # name, description, stage
        if context.product.name:
            filled_fields += 1
        if context.product.description:
            filled_fields += 1
        if context.product.stage:
            filled_fields += 1

    # Check market context
    if context.market:
        required_fields += 2  # competitors, positioning
        if context.market.competitors:
            filled_fields += 1
        if context.market.positioning:
            filled_fields += 1

    metrics.context_coverage_percent = int((filled_fields / required_fields * 100) if required_fields > 0 else 0)

    # Determine enrichment confidence
    if metrics.context_coverage_percent >= 80:
        metrics.enrichment_confidence = "HIGH"
    elif metrics.context_coverage_percent >= 60:
        metrics.enrichment_confidence = "MEDIUM"
    else:
        metrics.enrichment_confidence = "LOW"

    # Identify opportunities (mock data for now, would be from enrichment)
    if context.company and context.company.type == 'b2b':
        metrics.unserved_segments = ["Enterprise Security", "Compliance Automation", "API Management"]
        metrics.differentiator_opportunities = ["Real-time collaboration", "AI-powered insights"]
    else:
        metrics.unserved_segments = ["Gen Z users", "International markets"]
        metrics.differentiator_opportunities = ["Social features", "Gamification"]

    return metrics


def display_value_metrics(
    metrics: ValueMetrics,
    console: Console,
    show_details: bool = True,
) -> None:
    """
    Display value metrics in a beautiful, PM-focused way.

    Args:
        metrics: The calculated value metrics
        console: Rich console for output
        show_details: Whether to show detailed breakdown
    """
    # Create main value panel
    value_text = Text()

    # Headline metrics
    if metrics.personas_generated > 0:
        value_text.append(f"📊 Generated {metrics.personas_generated} user personas from real market data\n", style="green")

    if metrics.competitor_features_found > 0:
        value_text.append(f"🔍 Discovered {metrics.competitor_features_found} competitor features you haven't tracked\n", style="green")

    if metrics.market_segments_identified > 0:
        value_text.append(f"💡 Identified {metrics.market_segments_identified} market segments\n", style="green")

    if metrics.research_hours_saved > 0:
        days = metrics.research_hours_saved / 8  # Convert to workdays
        if days >= 1:
            value_text.append(f"📝 Created context worth {days:.1f} days of market research\n", style="bold green")
        else:
            value_text.append(f"📝 Created context worth {metrics.research_hours_saved:.1f} hours of research\n", style="green")

    value_text.append(f"\n🚀 Ready to generate your first PRD in {metrics.time_to_first_prd_seconds} seconds!", style="bold cyan")

    panel = Panel(
        value_text,
        title="✨ Value Created",
        title_align="left",
        border_style="green",
        padding=(1, 2),
    )
    console.print(panel)

    if show_details and (metrics.unserved_segments or metrics.differentiator_opportunities):
        # Show opportunities discovered
        opp_table = Table(title="🎯 Opportunities Discovered", show_header=True, header_style="bold cyan")
        opp_table.add_column("Type", style="dim")
        opp_table.add_column("Opportunity", style="white")

        for segment in metrics.unserved_segments[:3]:
            opp_table.add_row("Unserved Segment", segment)

        for diff in metrics.differentiator_opportunities[:2]:
            opp_table.add_row("Differentiator", diff)

        console.print(opp_table)


def format_quick_stats(metrics: ValueMetrics) -> str:
    """
    Format quick stats for inline display.

    Args:
        metrics: The calculated value metrics

    Returns:
        Formatted string with key metrics
    """
    stats = []

    if metrics.personas_generated > 0:
        stats.append(f"{metrics.personas_generated} personas")

    if metrics.competitor_features_found > 0:
        stats.append(f"{metrics.competitor_features_found} competitor insights")

    if metrics.research_hours_saved >= 8:
        days = metrics.research_hours_saved / 8
        stats.append(f"{days:.1f} days saved")
    elif metrics.research_hours_saved > 0:
        stats.append(f"{metrics.research_hours_saved:.1f} hours saved")

    return " • ".join(stats) if stats else "Context initialized"


def display_90_second_progress(
    console: Console,
    phase: str,
    elapsed: int,
    details: Optional[str] = None,
) -> None:
    """
    Display progress toward 90-second goal.

    Args:
        console: Rich console for output
        phase: Current phase name
        elapsed: Elapsed seconds
        details: Optional details about what's happening
    """
    remaining = max(0, 90 - elapsed)

    if elapsed <= 30:
        status = "[green]⚡ On track[/]"
        emoji = "🚀"
    elif elapsed <= 60:
        status = "[yellow]📍 Good pace[/]"
        emoji = "⏱️"
    elif elapsed <= 90:
        status = "[yellow]🏃 Almost there[/]"
        emoji = "⚡"
    else:
        status = "[dim]Finalizing[/]"
        emoji = "✨"

    progress_text = f"{emoji} {phase} • {elapsed}s elapsed"
    if remaining > 0:
        progress_text += f" • {remaining}s to goal"

    if details:
        progress_text += f"\n[dim]{details}[/]"

    console.print(Panel(
        progress_text,
        title=f"90-Second Setup {status}",
        border_style="cyan" if elapsed <= 90 else "dim",
        padding=(0, 1),
    ))