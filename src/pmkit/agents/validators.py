"""
Smart validators for PM-Kit onboarding with three-level validation.

This module provides intelligent validation with error (blocking), warning (advisory),
and autocorrect (automatic fixes) levels to help PMs avoid common mistakes.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from prompt_toolkit.document import Document
from prompt_toolkit.validation import ValidationError, Validator

from pmkit.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# VALIDATION RULES
# ============================================================

class ValidationRule:
    """Represents a single validation rule."""

    def __init__(
        self,
        level: str,  # error, warning, autocorrect
        check: Callable,
        message: str,
        fix: Optional[Callable] = None,
    ):
        self.level = level
        self.check = check
        self.message = message
        self.fix = fix  # For autocorrect


# Common patterns and fixes
COMMON_TYPOS = {
    # Metric typos
    'MMR': 'MRR',
    'DAUs': 'DAU',
    'MAUs': 'MAU',
    'ARR ': 'ARR',
    'CAC ': 'CAC',
    'LVT': 'LTV',
    'NPS ': 'NPS',

    # Company stage typos
    'ideation': 'idea',
    'scaling': 'growth',
    'scale': 'growth',
}

TEST_DATA_PATTERNS = [
    'test', 'demo', 'asdf', 'qwerty', 'example',
    'my company', 'todo', 'tbd', 'xxx', 'foo', 'bar'
]

FAMOUS_COMPANIES = [
    'google', 'apple', 'meta', 'facebook', 'amazon',
    'microsoft', 'netflix', 'uber', 'airbnb', 'spotify'
]


# ============================================================
# SMART VALIDATORS
# ============================================================

class SmartValidator:
    """
    Base class for smart validators with three-level validation.
    """

    def __init__(self):
        self.rules: List[ValidationRule] = []

    def validate(self, value: Any, context: Optional[Dict] = None) -> Tuple[bool, List[str], Any]:
        """
        Validate a value and return status, messages, and corrected value.

        Args:
            value: Value to validate
            context: Additional context for cross-field validation

        Returns:
            Tuple of (is_valid, messages, corrected_value)
        """
        is_valid = True
        messages = []
        corrected_value = value

        for rule in self.rules:
            if rule.level == 'autocorrect' and rule.fix:
                # Apply autocorrection
                corrected_value = rule.fix(corrected_value)
            elif rule.level == 'error':
                # Check for blocking errors
                if not rule.check(corrected_value, context):
                    is_valid = False
                    messages.append(f"❌ {rule.message}")
            elif rule.level == 'warning':
                # Check for warnings
                if not rule.check(corrected_value, context):
                    messages.append(f"⚠️ {rule.message}")

        return is_valid, messages, corrected_value


class CompanyNameSmartValidator(SmartValidator):
    """Smart validator for company names."""

    def __init__(self):
        super().__init__()
        self.rules = [
            # Autocorrect common issues
            ValidationRule(
                level='autocorrect',
                check=None,
                message='',
                fix=lambda x: x.strip().replace('  ', ' ')  # Clean whitespace
            ),

            # Error validations
            ValidationRule(
                level='error',
                check=lambda x, ctx: len(x) >= 2,
                message='Company name must be at least 2 characters',
            ),
            ValidationRule(
                level='error',
                check=lambda x, ctx: len(x) <= 50,
                message='Company name must be less than 50 characters',
            ),
            ValidationRule(
                level='error',
                check=lambda x, ctx: not re.search(r'[<>:"/\\|?*]', x),
                message='Company name cannot contain special characters: < > : " / \\ | ? *',
            ),

            # Warning validations
            ValidationRule(
                level='warning',
                check=lambda x, ctx: x.lower() not in TEST_DATA_PATTERNS,
                message='This looks like test data. Use your real company name.',
            ),
            ValidationRule(
                level='warning',
                check=lambda x, ctx: x.lower() not in FAMOUS_COMPANIES,
                message='Using a famous company name? Consider using your actual company.',
            ),
        ]


class ProductDescriptionSmartValidator(SmartValidator):
    """Smart validator for product descriptions."""

    def __init__(self):
        super().__init__()
        self.rules = [
            # Autocorrect
            ValidationRule(
                level='autocorrect',
                check=None,
                message='',
                fix=lambda x: x.strip().replace('  ', ' ')
            ),

            # Errors
            ValidationRule(
                level='error',
                check=lambda x, ctx: len(x.split()) >= 5,
                message='Too brief. Include what it does and who it\'s for (e.g., "AI code review for enterprise teams")',
            ),
            ValidationRule(
                level='error',
                check=lambda x, ctx: any(f' {word} ' in f' {x.lower()} ' for word in ['for', 'helps', 'enables', 'that']),
                message='Include who this product is FOR (e.g., "for engineering teams")',
            ),

            # Warnings
            ValidationRule(
                level='warning',
                check=lambda x, ctx: len(x) <= 200,
                message='Description is quite long. Consider being more concise.',
            ),
        ]


class NorthStarMetricSmartValidator(SmartValidator):
    """Smart validator for north star metrics."""

    def __init__(self):
        super().__init__()
        self.rules = [
            # Autocorrect common typos
            ValidationRule(
                level='autocorrect',
                check=None,
                message='',
                fix=self._fix_metric_typos
            ),

            # Warnings for B2B/B2C alignment
            ValidationRule(
                level='warning',
                check=self._check_b2b_alignment,
                message='MAU is unusual for B2B. Consider MRR, ARR, or Logo Retention.',
            ),
            ValidationRule(
                level='warning',
                check=self._check_b2c_alignment,
                message='MRR is unusual for B2C. Consider MAU, DAU, or Retention Rate.',
            ),
        ]

    def _fix_metric_typos(self, value: str) -> str:
        """Fix common metric typos."""
        upper_value = value.upper()
        for typo, correct in COMMON_TYPOS.items():
            if typo in upper_value:
                return upper_value.replace(typo, correct)
        return value

    def _check_b2b_alignment(self, value: str, context: Optional[Dict]) -> bool:
        """Check if metric aligns with B2B model."""
        if not context or context.get('company_type', '').lower() != 'b2b':
            return True  # Not B2B, skip check

        b2c_metrics = ['MAU', 'DAU', 'viral coefficient', 'daily active']
        return not any(metric in value.upper() for metric in b2c_metrics)

    def _check_b2c_alignment(self, value: str, context: Optional[Dict]) -> bool:
        """Check if metric aligns with B2C model."""
        if not context or context.get('company_type', '').lower() != 'b2c':
            return True  # Not B2C, skip check

        b2b_metrics = ['MRR', 'ARR', 'logo retention', 'CAC payback']
        return not any(metric in value.upper() for metric in b2b_metrics)


class CompetitorSmartValidator(SmartValidator):
    """Smart validator for competitors."""

    def __init__(self):
        super().__init__()
        self.rules = [
            # Warning validations
            ValidationRule(
                level='warning',
                check=self._check_not_self,
                message='You listed your own company as a competitor',
            ),
            ValidationRule(
                level='warning',
                check=lambda x, ctx: len(x) <= 5 if isinstance(x, list) else True,
                message='Consider focusing on 3-5 main competitors for clarity',
            ),
        ]

    def _check_not_self(self, value: Any, context: Optional[Dict]) -> bool:
        """Check that company didn't list itself as competitor."""
        if not context or not isinstance(value, list):
            return True

        company_name = context.get('company_name', '').lower()
        if not company_name:
            return True

        competitor_names = [c.lower() for c in value if isinstance(c, str)]
        return company_name not in competitor_names


class CompanyStageSmartValidator(SmartValidator):
    """Smart validator for company stage with auto-correction."""

    def __init__(self):
        super().__init__()
        self.rules = [
            # Autocorrect common mistakes
            ValidationRule(
                level='autocorrect',
                check=None,
                message='',
                fix=self._autocorrect_stage,
            ),
            # Error validation
            ValidationRule(
                level='error',
                check=lambda x, ctx: x in ['idea', 'seed', 'growth', 'mature'],
                message="Stage must be one of: idea, seed, growth, mature",
            ),
        ]

    def _autocorrect_stage(self, value: str) -> str:
        """Auto-correct common stage mistakes."""
        if not isinstance(value, str):
            return value

        value_lower = value.lower().strip()

        # Map common mistakes to valid values
        corrections = {
            'scale': 'growth',
            'scaling': 'growth',
            'series a': 'seed',
            'series b': 'seed',
            'series c': 'growth',
            'series d': 'growth',
            'ipo': 'mature',
            'public': 'mature',
            'enterprise': 'mature',
            'early': 'seed',
            'startup': 'seed',
            'pre-seed': 'idea',
            'preseed': 'idea',
            'concept': 'idea',
            'mvp': 'seed',
            'pmf': 'growth',
            'product-market fit': 'growth',
            'established': 'mature',
        }

        # Check for exact matches first
        if value_lower in ['idea', 'seed', 'growth', 'mature']:
            return value_lower

        # Check for common mistakes
        for mistake, correct in corrections.items():
            if mistake in value_lower:
                return correct

        return value


class TeamSizeSmartValidator(SmartValidator):
    """Smart validator for team size with cross-field validation."""

    def __init__(self):
        super().__init__()
        self.rules = [
            # Error validations
            ValidationRule(
                level='error',
                check=lambda x, ctx: 0 < x < 100000 if isinstance(x, int) else True,
                message='Team size must be between 1 and 100,000',
            ),

            # Warning for misalignment with company stage
            ValidationRule(
                level='warning',
                check=self._check_stage_alignment,
                message='Team size seems unusual for company stage',
            ),
        ]

    def _check_stage_alignment(self, value: Any, context: Optional[Dict]) -> bool:
        """Check if team size aligns with company stage."""
        if not context or not isinstance(value, int):
            return True

        stage = context.get('company_stage', '').lower()

        # Check for obvious mismatches
        if stage == 'idea' and value > 10:
            return False  # Too many people for idea stage
        elif stage == 'seed' and value > 50:
            return False  # Quite large for seed
        elif stage == 'mature' and value < 10:
            return False  # Too small for mature

        return True


class URLSmartValidator(SmartValidator):
    """Smart validator for URLs."""

    def __init__(self):
        super().__init__()
        self.rules = [
            # Autocorrect - add protocol if missing
            ValidationRule(
                level='autocorrect',
                check=None,
                message='',
                fix=self._fix_url
            ),

            # Error validation
            ValidationRule(
                level='error',
                check=self._is_valid_url,
                message='Invalid URL format',
            ),
        ]

    def _fix_url(self, value: str) -> str:
        """Add https:// if protocol is missing."""
        if value and not value.startswith(('http://', 'https://')):
            return f'https://{value}'
        return value

    def _is_valid_url(self, value: str, context: Optional[Dict]) -> bool:
        """Check if URL is valid."""
        if not value:
            return True  # Empty is OK for optional field

        url_pattern = re.compile(
            r'^https?://'  # Protocol
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # Domain
            r'localhost|'  # Localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
            r'(?::\d+)?'  # Optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE
        )
        return bool(url_pattern.match(value))


# ============================================================
# PROMPT TOOLKIT VALIDATORS
# ============================================================

class SmartPromptValidator(Validator):
    """
    Adapter to use SmartValidator with prompt_toolkit.
    """

    def __init__(self, smart_validator: SmartValidator, context: Optional[Dict] = None):
        self.smart_validator = smart_validator
        self.context = context or {}

    def validate(self, document: Document) -> None:
        """Validate document for prompt_toolkit."""
        text = document.text.strip()

        is_valid, messages, corrected = self.smart_validator.validate(text, self.context)

        if not is_valid:
            # Show only error messages in prompt validation
            error_messages = [msg for msg in messages if msg.startswith('❌')]
            if error_messages:
                raise ValidationError(
                    message=error_messages[0].replace('❌ ', ''),
                    cursor_position=len(text)
                )


# ============================================================
# FACTORY FUNCTIONS
# ============================================================

def get_validator(field_name: str, context: Optional[Dict] = None) -> Optional[SmartValidator]:
    """
    Get the appropriate smart validator for a field.

    Args:
        field_name: Name of the field to validate
        context: Context for cross-field validation

    Returns:
        SmartValidator instance or None
    """
    validators = {
        'company_name': CompanyNameSmartValidator,
        'product_description': ProductDescriptionSmartValidator,
        'north_star_metric': NorthStarMetricSmartValidator,
        'competitors': CompetitorSmartValidator,
        'team_size': TeamSizeSmartValidator,
        'website': URLSmartValidator,
    }

    validator_class = validators.get(field_name)
    if validator_class:
        return validator_class()

    return None


def validate_and_fix(
    field_name: str,
    value: Any,
    context: Optional[Dict] = None
) -> Tuple[bool, List[str], Any]:
    """
    Validate and fix a field value.

    Args:
        field_name: Name of the field
        value: Value to validate
        context: Context for cross-field validation

    Returns:
        Tuple of (is_valid, messages, corrected_value)
    """
    validator = get_validator(field_name, context)

    if not validator:
        return True, [], value

    return validator.validate(value, context)


def apply_autocorrect(field_name: str, value: str) -> str:
    """
    Apply only autocorrections to a value without validation.

    Args:
        field_name: Name of the field
        value: Value to correct

    Returns:
        Corrected value
    """
    validator = get_validator(field_name)

    if not validator:
        return value

    _, _, corrected = validator.validate(value, {})
    return corrected