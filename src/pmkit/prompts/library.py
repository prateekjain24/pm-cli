"""
Centralized prompt library for PM-Kit.

This module provides a single source of truth for all LLM prompts used
throughout the application, following the agent-architecture-guide patterns.
"""

from __future__ import annotations

from typing import Dict, Optional


class PromptLibrary:
    """
    Centralized storage for all LLM prompts used in PM-Kit.

    Organized by feature area to maintain clarity and enable
    easy updates without searching through multiple files.
    """

    # ============================================================
    # ONBOARDING PROMPTS
    # ============================================================

    COMPANY_ENRICHMENT = """
    Search for information about the company "{company_name}" and extract:

    1. Industry/vertical
    2. Business model (B2B, B2C, or B2B2C)
    3. Company stage (idea, seed, growth, or mature)
    4. Main competitors (top 3 companies in the same space)
    5. Target market (e.g., SMBs, Enterprise, Consumers)
    6. Company website domain (if available)

    Return the information in a structured format. If any information
    cannot be found, indicate that clearly rather than guessing.

    Focus on recent, factual information from 2024-2025.
    """

    COMPETITOR_SEARCH = """
    Find the top competitors for {company_name} in the {industry} space.

    Requirements:
    - List 3-5 direct competitors
    - Focus on companies with similar products/services
    - Include both established players and notable startups
    - Prefer companies from the same geographic market if relevant

    For each competitor, provide:
    - Company name
    - One-line description of what they do
    - Why they're a competitor

    Return as a structured list.
    """

    MARKET_ANALYSIS = """
    Analyze the market for {company_name} operating in {industry}:

    1. Market size (TAM if available)
    2. Growth rate/trends
    3. Key market drivers
    4. Main challenges in this market
    5. Unique differentiators for a company in this space

    Focus on 2024-2025 data and trends. Be specific with numbers
    where available, but indicate if data is estimated.
    """

    # ============================================================
    # PRD GENERATION PROMPTS
    # ============================================================

    PRD_PROBLEM_B2B = """
    You are writing the Problem Statement section of a PRD for a B2B product.

    Context:
    - Company: {company_name} ({company_type})
    - Product: {product_name}
    - Stage: {product_stage}
    - Target Market: {target_market}

    PRD Title: {prd_title}

    Write a compelling problem statement that:
    1. Clearly identifies the business problem
    2. Quantifies the impact (time lost, money wasted, opportunities missed)
    3. Explains why existing solutions fall short
    4. Emphasizes ROI and efficiency gains
    5. Addresses compliance/security concerns if relevant

    Format as markdown with clear sections. Focus on enterprise buyer priorities:
    ROI, integration capabilities, security, and scalability.
    """

    PRD_PROBLEM_B2C = """
    You are writing the Problem Statement section of a PRD for a B2C product.

    Context:
    - Company: {company_name} ({company_type})
    - Product: {product_name}
    - Stage: {product_stage}
    - Users: {user_count}

    PRD Title: {prd_title}

    Write an engaging problem statement that:
    1. Identifies the user pain point or unmet need
    2. Describes the emotional impact on users
    3. Explains frequency and severity of the problem
    4. Shows why current alternatives aren't good enough
    5. Highlights the opportunity for delight

    Format as markdown with clear sections. Focus on user experience,
    engagement, retention, and potential for viral growth.
    """

    PRD_SOLUTION = """
    Based on the problem statement provided, write the Solution Approach section.

    Problem Context:
    {problem_statement}

    Additional Context:
    - Competitors: {competitors}
    - Team composition: {team_composition}
    - Current OKRs: {okrs}

    Create a solution that:
    1. Directly addresses each problem point
    2. Explains the core approach and why it works
    3. Highlights key differentiators from competitors
    4. Outlines the user journey/workflow
    5. Identifies success metrics
    6. Considers technical feasibility given team composition

    Format as markdown with clear sections and visual hierarchy.
    """

    PRD_REQUIREMENTS = """
    Convert the solution approach into detailed requirements.

    Solution Context:
    {solution_statement}

    Generate:
    1. User Stories (5-8 stories with acceptance criteria)
    2. Functional Requirements (must-have features)
    3. Non-functional Requirements (performance, security, etc.)
    4. Out of Scope (what we're NOT building)
    5. Dependencies and Assumptions

    Format each user story as:
    - Story ID: [PREFIX-XXX]
    - As a [user type], I want [goal] so that [benefit]
    - Acceptance Criteria: (bulleted list)

    Be specific and measurable. Avoid ambiguous terms.
    """

    # ============================================================
    # CONTEXT VALIDATION PROMPTS
    # ============================================================

    VALIDATE_METRICS = """
    Review these success metrics for a {company_type} product:

    {metrics}

    Evaluate if they are:
    1. Specific and measurable
    2. Appropriate for the business model
    3. Realistic and achievable
    4. Aligned with typical industry KPIs

    Suggest improvements if needed. For {company_type}, typical metrics include:
    {typical_metrics}
    """

    # ============================================================
    # HELPER METHODS
    # ============================================================

    @classmethod
    def get_prompt(cls, prompt_name: str, **kwargs) -> str:
        """
        Retrieve and format a prompt with the given parameters.

        Args:
            prompt_name: Name of the prompt (e.g., 'COMPANY_ENRICHMENT')
            **kwargs: Variables to format into the prompt

        Returns:
            Formatted prompt string

        Raises:
            AttributeError: If prompt_name doesn't exist
            KeyError: If required format variables are missing
        """
        prompt_template = getattr(cls, prompt_name)
        return prompt_template.format(**kwargs)

    @classmethod
    def list_prompts(cls) -> list[str]:
        """
        List all available prompt names.

        Returns:
            List of prompt attribute names
        """
        return [
            attr for attr in dir(cls)
            if not attr.startswith('_')
            and isinstance(getattr(cls, attr), str)
            and attr.isupper()
        ]

    @classmethod
    def get_typical_metrics(cls, company_type: str) -> str:
        """
        Get typical metrics for a company type.

        Args:
            company_type: 'b2b', 'b2c', or 'b2b2c'

        Returns:
            Comma-separated list of typical metrics
        """
        metrics_map = {
            'b2b': 'MRR, ARR, CAC, LTV, Churn Rate, NPS, Sales Cycle Length',
            'b2c': 'MAU, DAU, Retention Rate, ARPU, Virality Coefficient, CAC, LTV',
            'b2b2c': 'MRR, MAU, Platform GMV, Take Rate, Network Effects, CAC by Channel',
        }
        return metrics_map.get(company_type, 'Revenue, Users, Retention, Growth Rate')