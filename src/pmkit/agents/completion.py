"""
Completion metrics and helpers for PM-Kit onboarding.

This module provides PM-focused metrics calculation and value demonstration
for the onboarding completion experience.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from pmkit.context.models import Context, OKRContext


@dataclass
class CompletionMetrics:
    """
    Metrics calculated at onboarding completion.

    Focuses on demonstrating value to PMs through time savings
    and context completeness.
    """

    time_spent: float  # seconds
    data_points_collected: int
    enrichment_coverage: float  # 0.0 to 1.0
    okr_confidence: float  # average confidence across all KRs
    completeness_score: float  # 0.0 to 1.0
    time_saved_per_prd: int = 45  # minutes

    @property
    def annual_hours_saved(self) -> int:
        """Calculate annual time savings assuming weekly PRDs."""
        weekly_minutes = self.time_saved_per_prd
        annual_minutes = weekly_minutes * 52
        return annual_minutes // 60

    @property
    def work_weeks_saved(self) -> float:
        """Convert annual hours to work weeks (40-hour weeks)."""
        return self.annual_hours_saved / 40

    def format_time_spent(self) -> str:
        """Format time spent in human-readable format."""
        if self.time_spent < 60:
            return f"{self.time_spent:.1f}s"
        elif self.time_spent < 3600:
            minutes = self.time_spent / 60
            return f"{minutes:.1f}m"
        else:
            hours = self.time_spent / 3600
            return f"{hours:.1f}h"

    def format_coverage_bar(self) -> str:
        """Create visual coverage bar with color coding."""
        filled = int(self.enrichment_coverage * 10)
        empty = 10 - filled

        bar = "█" * filled + "░" * empty

        if self.enrichment_coverage >= 0.8:
            color = "green"
            label = "Excellent"
        elif self.enrichment_coverage >= 0.6:
            color = "yellow"
            label = "Good"
        else:
            color = "red"
            label = "Partial"

        return f"[{color}]{bar}[/{color}] {self.enrichment_coverage*100:.0f}% ({label})"

    def get_completeness_emoji(self) -> str:
        """Return emoji based on completeness score."""
        if self.completeness_score >= 0.9:
            return "🌟"
        elif self.completeness_score >= 0.7:
            return "✅"
        elif self.completeness_score >= 0.5:
            return "🟡"
        else:
            return "⚠️"


def calculate_completion_metrics(
    context: Context,
    time_spent: float,
    enrichment_used: bool = False,
    searches_performed: int = 0
) -> CompletionMetrics:
    """
    Calculate comprehensive metrics for onboarding completion.

    Args:
        context: The completed context object
        time_spent: Time taken for onboarding in seconds
        enrichment_used: Whether web enrichment was used
        searches_performed: Number of web searches performed

    Returns:
        CompletionMetrics object with calculated values
    """
    # Count data points collected
    data_points = count_data_points(context)

    # Calculate enrichment coverage
    coverage = calculate_enrichment_coverage(context, enrichment_used)

    # Calculate OKR confidence
    okr_confidence = calculate_okr_confidence(context.okrs) if context.okrs else 0.0

    # Calculate overall completeness
    completeness = calculate_completeness_score(context)

    return CompletionMetrics(
        time_spent=time_spent,
        data_points_collected=data_points,
        enrichment_coverage=coverage,
        okr_confidence=okr_confidence,
        completeness_score=completeness
    )


def count_data_points(context: Context) -> int:
    """
    Count the number of non-empty data points in the context.

    Args:
        context: The context to analyze

    Returns:
        Number of populated data points
    """
    count = 0

    # Company data points
    if context.company:
        if context.company.name:
            count += 1
        if context.company.type:
            count += 1
        if context.company.stage:
            count += 1
        if context.company.domain:
            count += 1
        if context.company.description:
            count += 1
        if context.company.target_market:
            count += 1

    # Product data points
    if context.product:
        if context.product.name:
            count += 1
        if context.product.description:
            count += 1
        if context.product.category:
            count += 1
        if context.product.main_metric:
            count += 1
        if context.product.target_audience:
            count += 1

    # Market data points
    if context.market:
        if context.market.competitors:
            count += len(context.market.competitors[:5])  # Cap at 5
        if context.market.differentiator:
            count += 1
        if context.market.industry:
            count += 1
        if context.market.market_size:
            count += 1

    # Team data points
    if context.team:
        if context.team.size:
            count += 1
        if context.team.roles:
            count += len(context.team.roles)

    # OKR data points
    if context.okrs and context.okrs.objectives:
        count += len(context.okrs.objectives)
        for obj in context.okrs.objectives:
            count += len(obj.key_results)

    return count


def calculate_enrichment_coverage(context: Context, enrichment_used: bool) -> float:
    """
    Calculate how much of the context was enriched vs manual.

    Args:
        context: The context to analyze
        enrichment_used: Whether enrichment was attempted

    Returns:
        Coverage score from 0.0 to 1.0
    """
    if not enrichment_used:
        return 0.0

    total_fields = 0
    enriched_fields = 0

    # Check which fields were likely enriched
    if context.company:
        total_fields += 3  # domain, target_market, description
        if context.company.domain:
            enriched_fields += 1
        if context.company.target_market:
            enriched_fields += 1
        if context.company.description and len(context.company.description) > 50:
            enriched_fields += 1

    if context.market:
        total_fields += 3  # competitors, industry, market_size
        if context.market.competitors and len(context.market.competitors) >= 3:
            enriched_fields += 2  # Good competitor coverage
        if context.market.industry:
            enriched_fields += 1

    if total_fields == 0:
        return 0.0

    return enriched_fields / total_fields


def calculate_okr_confidence(okrs: Optional[OKRContext]) -> float:
    """
    Calculate average confidence across all key results.

    Args:
        okrs: The OKR context to analyze

    Returns:
        Average confidence score (0-100)
    """
    if not okrs or not okrs.objectives:
        return 0.0

    total_confidence = 0
    total_krs = 0

    for objective in okrs.objectives:
        for kr in objective.key_results:
            if kr.confidence is not None:
                total_confidence += kr.confidence
                total_krs += 1

    if total_krs == 0:
        return 0.0

    return total_confidence / total_krs


def calculate_completeness_score(context: Context) -> float:
    """
    Calculate overall context completeness score.

    Weights different sections based on importance for PRD generation.

    Args:
        context: The context to analyze

    Returns:
        Completeness score from 0.0 to 1.0
    """
    scores = []
    weights = []

    # Company section (required, high weight)
    if context.company:
        company_score = 0
        if context.company.name:
            company_score += 0.4
        if context.company.type:
            company_score += 0.3
        if context.company.stage:
            company_score += 0.3
        scores.append(company_score)
        weights.append(0.25)

    # Product section (required, high weight)
    if context.product:
        product_score = 0
        if context.product.name:
            product_score += 0.3
        if context.product.description:
            product_score += 0.4
        if context.product.main_metric:
            product_score += 0.3
        scores.append(product_score)
        weights.append(0.25)

    # Market section (important, medium weight)
    if context.market:
        market_score = 0
        if context.market.competitors:
            market_score += 0.5
        if context.market.differentiator:
            market_score += 0.5
        scores.append(market_score)
        weights.append(0.2)
    else:
        scores.append(0)
        weights.append(0.2)

    # Team section (optional, low weight)
    if context.team:
        team_score = 0
        if context.team.size:
            team_score += 0.5
        if context.team.roles:
            team_score += 0.5
        scores.append(team_score)
        weights.append(0.1)
    else:
        scores.append(0)
        weights.append(0.1)

    # OKRs section (important, medium weight)
    if context.okrs and context.okrs.objectives:
        okr_score = min(1.0, len(context.okrs.objectives) / 3)  # 3+ objectives = full score
        scores.append(okr_score)
        weights.append(0.2)
    else:
        scores.append(0)
        weights.append(0.2)

    # Calculate weighted average
    if sum(weights) == 0:
        return 0.0

    return sum(s * w for s, w in zip(scores, weights)) / sum(weights)


def suggest_next_actions(context: Context, metrics: CompletionMetrics) -> List[Tuple[str, str]]:
    """
    Generate personalized next action suggestions based on context.

    Args:
        context: The completed context
        metrics: Calculated completion metrics

    Returns:
        List of (command, description) tuples for next actions
    """
    actions = []

    # Primary action based on B2B vs B2C
    if context.company and context.company.type == 'b2b':
        actions.append((
            'pm new prd "API Authentication Strategy"',
            'Create a technical PRD for enterprise integration needs'
        ))
    else:
        actions.append((
            'pm new prd "User Onboarding Optimization"',
            'Design a growth-focused PRD to improve activation'
        ))

    # Based on completeness gaps
    if metrics.enrichment_coverage < 0.7:
        actions.append((
            'pm enrich market',
            'Deepen market intelligence with additional research'
        ))

    if not context.team or context.team.size == 0:
        actions.append((
            'pm context edit --section team',
            'Add your team structure for better collaboration features'
        ))

    if context.okrs and metrics.okr_confidence < 60:
        actions.append((
            'pm refine okrs',
            'Improve OKR confidence with clearer metrics'
        ))

    # Based on company stage
    if context.company:
        if context.company.stage == 'seed':
            actions.append((
                'pm new prd "MVP Feature Prioritization"',
                'Define your core MVP features systematically'
            ))
        elif context.company.stage == 'growth':
            actions.append((
                'pm new prd "Scaling Infrastructure"',
                'Plan for 10x growth with proper infrastructure'
            ))
        elif context.company.stage == 'enterprise':
            actions.append((
                'pm new prd "Enterprise Security Compliance"',
                'Meet enterprise security requirements'
            ))

    # Always suggest viewing the context
    actions.append((
        'pm status',
        'View your complete context and available commands'
    ))

    # Return top 3 most relevant actions
    return actions[:3]


def time_saved_message(metrics: CompletionMetrics) -> str:
    """
    Generate a motivating message about time saved.

    Args:
        metrics: The completion metrics

    Returns:
        Formatted message about time savings
    """
    if metrics.time_spent < 60:
        setup_time = "less than a minute"
    elif metrics.time_spent < 180:
        setup_time = "just a few minutes"
    else:
        setup_time = metrics.format_time_spent()

    return (
        f"You invested {setup_time} to save [bold green]{metrics.annual_hours_saved} hours[/bold green] "
        f"per year (~{metrics.work_weeks_saved:.0f} work weeks). "
        f"That's a [cyan]{int(metrics.annual_hours_saved * 60 / max(1, metrics.time_spent))}x[/cyan] return on time invested!"
    )