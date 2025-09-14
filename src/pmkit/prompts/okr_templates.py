"""
Context-aware OKR templates for B2B and B2C companies.

Provides smart suggestions based on company type and stage.
"""

from typing import Dict, List


class OKRTemplates:
    """
    OKR template library with PM-specific suggestions.
    """

    # B2B Objective Templates by Stage
    B2B_OBJECTIVES = {
        'seed': [
            "Achieve product-market fit in {target_segment}",
            "Establish repeatable sales process",
            "Build foundation for scalable growth",
            "Validate enterprise readiness"
        ],
        'growth': [
            "Expand into {new_market} segment",
            "Achieve efficient growth metrics",
            "Strengthen product differentiation",
            "Build world-class customer success",
            "Establish market leadership in {category}"
        ],
        'scale': [
            "Dominate {market} with {share}% market share",
            "Achieve operational excellence at scale",
            "Expand platform ecosystem",
            "Drive international expansion",
            "Optimize unit economics for profitability"
        ]
    }

    # B2C Objective Templates by Stage
    B2C_OBJECTIVES = {
        'seed': [
            "Find product-market fit with early adopters",
            "Build viral growth loop",
            "Establish brand identity and voice",
            "Create delightful user experience"
        ],
        'growth': [
            "Scale user acquisition efficiently",
            "Build engagement and retention engine",
            "Achieve viral coefficient > 1",
            "Establish monetization model",
            "Create network effects"
        ],
        'scale': [
            "Become category-defining brand",
            "Achieve market penetration of {x}%",
            "Build platform for ecosystem",
            "Expand internationally to {regions}",
            "Optimize LTV/CAC to {ratio}"
        ]
    }

    # B2B Key Result Templates
    B2B_KEY_RESULTS = {
        'revenue': [
            "Increase ARR from ${current}M to ${target}M",
            "Grow MRR from ${current}k to ${target}k",
            "Achieve net revenue retention of {x}%",
            "Reduce CAC payback period to {x} months",
            "Increase ACV from ${current}k to ${target}k"
        ],
        'sales': [
            "Close {x} enterprise deals (>$100k ACV)",
            "Increase win rate from {current}% to {target}%",
            "Reduce sales cycle from {current} to {target} days",
            "Build pipeline of ${x}M qualified opportunities",
            "Achieve {x}% quota attainment across team"
        ],
        'product': [
            "Launch {x} tier-1 features requested by enterprise",
            "Achieve {x}% feature adoption within 30 days",
            "Reduce time to value from {current} to {target} days",
            "Achieve platform uptime of {x}%",
            "Complete SOC2/ISO certification"
        ],
        'customer': [
            "Achieve NPS of {x} (from {current})",
            "Reduce churn from {current}% to {target}%",
            "Increase product qualified leads to {x}/month",
            "Achieve {x}% CSAT score",
            "Onboard {x} reference customers"
        ]
    }

    # B2C Key Result Templates
    B2C_KEY_RESULTS = {
        'growth': [
            "Grow MAU from {current}M to {target}M",
            "Increase DAU from {current}k to {target}k",
            "Achieve viral coefficient of {x}",
            "Reduce CPA from ${current} to ${target}",
            "Increase organic traffic to {x}% of total"
        ],
        'engagement': [
            "Increase D7 retention from {current}% to {target}%",
            "Grow daily sessions per user from {current} to {target}",
            "Increase time spent from {current} to {target} min/day",
            "Achieve {x}% WAU/MAU ratio",
            "Increase user-generated content by {x}%"
        ],
        'monetization': [
            "Increase ARPU from ${current} to ${target}",
            "Grow paid conversion from {current}% to {target}%",
            "Achieve LTV/CAC ratio of {x}",
            "Increase subscription revenue to ${x}M",
            "Launch and scale {x} revenue streams"
        ],
        'product': [
            "Launch {x} viral features",
            "Achieve {x}% feature adoption in first week",
            "Reduce app crash rate to <{x}%",
            "Improve app store rating to {x} stars",
            "Reduce core action friction by {x}%"
        ]
    }

    @classmethod
    def get_objective_suggestions(cls, company_type: str, stage: str) -> List[str]:
        """
        Get objective suggestions based on company context.

        Args:
            company_type: 'B2B' or 'B2C'
            stage: 'seed', 'growth', or 'scale'

        Returns:
            List of objective templates
        """
        if company_type.upper() == 'B2B':
            return cls.B2B_OBJECTIVES.get(stage.lower(), cls.B2B_OBJECTIVES['growth'])
        else:
            return cls.B2C_OBJECTIVES.get(stage.lower(), cls.B2C_OBJECTIVES['growth'])

    @classmethod
    def get_key_result_suggestions(cls, company_type: str, category: str = None) -> List[str]:
        """
        Get key result suggestions based on company type and category.

        Args:
            company_type: 'B2B' or 'B2C'
            category: Optional category filter (e.g., 'revenue', 'growth')

        Returns:
            List of key result templates
        """
        if company_type.upper() == 'B2B':
            if category and category in cls.B2B_KEY_RESULTS:
                return cls.B2B_KEY_RESULTS[category]
            # Return mix of all categories
            results = []
            for cat_results in cls.B2B_KEY_RESULTS.values():
                results.extend(cat_results[:2])  # Take first 2 from each
            return results
        else:
            if category and category in cls.B2C_KEY_RESULTS:
                return cls.B2C_KEY_RESULTS[category]
            # Return mix of all categories
            results = []
            for cat_results in cls.B2C_KEY_RESULTS.values():
                results.extend(cat_results[:2])  # Take first 2 from each
            return results

    @classmethod
    def get_quick_start_okrs(cls, company_type: str, stage: str) -> Dict:
        """
        Get complete quick-start OKR set for fast setup.

        Args:
            company_type: 'B2B' or 'B2C'
            stage: 'seed', 'growth', or 'scale'

        Returns:
            Dictionary with pre-filled OKRs
        """
        if company_type.upper() == 'B2B':
            if stage == 'seed':
                return {
                    'objectives': [
                        {
                            'title': 'Achieve product-market fit in enterprise segment',
                            'key_results': [
                                {'description': 'Close 10 paid enterprise pilots', 'confidence': 60},
                                {'description': 'Achieve 80% pilot conversion rate', 'confidence': 50},
                                {'description': 'Reach $1M ARR', 'confidence': 40}
                            ]
                        }
                    ]
                }
            elif stage == 'scale':
                return {
                    'objectives': [
                        {
                            'title': 'Accelerate revenue growth efficiently',
                            'key_results': [
                                {'description': 'Grow ARR from $10M to $25M', 'confidence': 60},
                                {'description': 'Achieve 130% net revenue retention', 'confidence': 70},
                                {'description': 'Reduce CAC payback to 12 months', 'confidence': 50}
                            ]
                        }
                    ]
                }
            else:  # growth
                return {
                    'objectives': [
                        {
                            'title': 'Build repeatable and scalable growth engine',
                            'key_results': [
                                {'description': 'Increase MRR from $50k to $150k', 'confidence': 55},
                                {'description': 'Achieve 25% win rate on qualified deals', 'confidence': 60},
                                {'description': 'Onboard 50 new customers', 'confidence': 65}
                            ]
                        }
                    ]
                }
        else:  # B2C
            if stage == 'seed':
                return {
                    'objectives': [
                        {
                            'title': 'Find product-market fit with early adopters',
                            'key_results': [
                                {'description': 'Reach 10k MAU', 'confidence': 60},
                                {'description': 'Achieve 40% D7 retention', 'confidence': 50},
                                {'description': 'Get to 1,000 daily active users', 'confidence': 55}
                            ]
                        }
                    ]
                }
            elif stage == 'scale':
                return {
                    'objectives': [
                        {
                            'title': 'Achieve market leadership position',
                            'key_results': [
                                {'description': 'Grow MAU from 5M to 15M', 'confidence': 60},
                                {'description': 'Increase ARPU from $2 to $5', 'confidence': 45},
                                {'description': 'Achieve 4.5+ app store rating', 'confidence': 70}
                            ]
                        }
                    ]
                }
            else:  # growth
                return {
                    'objectives': [
                        {
                            'title': 'Build sustainable growth and engagement',
                            'key_results': [
                                {'description': 'Grow MAU from 100k to 500k', 'confidence': 55},
                                {'description': 'Increase D7 retention from 30% to 50%', 'confidence': 50},
                                {'description': 'Achieve viral coefficient of 0.7', 'confidence': 45}
                            ]
                        }
                    ]
                }

    @classmethod
    def get_confidence_guidance(cls, objective_type: str) -> Dict[str, str]:
        """
        Get confidence level guidance for different objective types.

        Returns:
            Dictionary with confidence ranges and their meanings
        """
        return {
            'revenue': {
                '70-100%': 'Conservative growth - consider being more ambitious',
                '50-70%': 'Healthy stretch - challenging but achievable',
                '30-50%': 'Aggressive target - ensure you have clear tactics',
                '0-30%': 'Moonshot - have a backup plan'
            },
            'product': {
                '70-100%': 'Well-scoped - shipping is likely',
                '50-70%': 'Good balance - some uncertainty is healthy',
                '30-50%': 'Ambitious scope - consider MVPs',
                '0-30%': 'High risk - break into smaller milestones'
            },
            'user_growth': {
                '70-100%': 'Predictable growth - mostly organic',
                '50-70%': 'Requires execution - paid + organic mix',
                '30-50%': 'Needs breakthrough - new channels or viral',
                '0-30%': 'Hockey stick - depends on going viral'
            }
        }

    @classmethod
    def get_industry_specific_templates(cls, industry: str) -> Dict:
        """
        Get industry-specific OKR templates.

        Args:
            industry: Industry vertical (e.g., 'SaaS', 'Marketplace', 'FinTech')

        Returns:
            Dictionary with industry-specific suggestions
        """
        templates = {
            'SaaS': {
                'objectives': ['Achieve predictable revenue growth', 'Build sticky product'],
                'key_results': ['Reach 120% NRR', 'Reduce churn to <5% monthly']
            },
            'Marketplace': {
                'objectives': ['Balance supply and demand growth', 'Improve liquidity'],
                'key_results': ['Achieve 80% fill rate', 'Reduce time to first transaction']
            },
            'FinTech': {
                'objectives': ['Build trust and compliance', 'Scale transaction volume'],
                'key_results': ['Complete regulatory approval', 'Process $X in transactions']
            },
            'E-commerce': {
                'objectives': ['Optimize unit economics', 'Expand product catalog'],
                'key_results': ['Improve gross margin to X%', 'Launch X new categories']
            },
            'EdTech': {
                'objectives': ['Improve learning outcomes', 'Scale content library'],
                'key_results': ['Achieve X% course completion', 'Add X hours of content']
            }
        }
        return templates.get(industry, templates['SaaS'])  # Default to SaaS