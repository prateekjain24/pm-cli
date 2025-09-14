"""
Smart validators for OKR collection that guide without annoying.

These validators provide helpful feedback to improve OKR quality
while maintaining a delightful user experience.
"""

import re
from typing import Optional

from prompt_toolkit.validation import ValidationError, Validator
from prompt_toolkit.document import Document


class ObjectiveValidator(Validator):
    """
    Validator for OKR objectives.

    Ensures objectives are outcome-focused, clear, and meaningful.
    """

    # Words that indicate output instead of outcome
    OUTPUT_INDICATORS = [
        'ship', 'launch', 'release', 'deploy', 'build', 'create',
        'implement', 'develop', 'code', 'design', 'write'
    ]

    # Jargon to avoid
    JARGON_TERMS = [
        'synergy', 'leverage', 'ideate', 'circle back', 'boil the ocean',
        'move the needle', 'paradigm shift', 'disrupt', 'pivot'
    ]

    # Good objective starters
    GOOD_STARTERS = [
        'achieve', 'reach', 'improve', 'increase', 'decrease',
        'expand', 'strengthen', 'establish', 'optimize', 'accelerate'
    ]

    def validate(self, document: Document) -> None:
        """Validate objective input."""
        text = document.text.strip()

        # Check minimum length
        if len(text.split()) < 5:
            raise ValidationError(
                message="📝 Objectives should be at least 5 words. Be specific about what you want to achieve!",
                cursor_position=len(document.text)
            )

        # Check maximum length
        if len(text) > 200:
            raise ValidationError(
                message="📏 That's quite long! Try to keep objectives under 200 characters for clarity.",
                cursor_position=len(document.text)
            )

        # Check for output vs outcome
        lower_text = text.lower()
        for output_word in self.OUTPUT_INDICATORS:
            if output_word in lower_text.split():
                # This is a warning, not a hard error - make it educational
                suggestion = self._get_outcome_suggestion(output_word)
                raise ValidationError(
                    message=f"💡 '{output_word}' sounds like an output. {suggestion}",
                    cursor_position=len(document.text)
                )

        # Check for jargon
        for jargon in self.JARGON_TERMS:
            if jargon in lower_text:
                raise ValidationError(
                    message=f"🎯 Let's avoid '{jargon}' - use plain language that everyone understands.",
                    cursor_position=len(document.text)
                )

        # Positive reinforcement for good patterns
        first_word = text.split()[0].lower()
        if first_word not in self.GOOD_STARTERS and not self._is_contextual_start(text):
            # Soft suggestion, not blocking
            pass  # Allow it through, they might have a good reason

    def _get_outcome_suggestion(self, output_word: str) -> str:
        """Get helpful suggestion for converting output to outcome."""
        suggestions = {
            'ship': "Try focusing on the impact: 'Achieve product-market fit' instead of 'Ship feature X'",
            'launch': "Consider the goal: 'Expand into new market' instead of 'Launch in region Y'",
            'build': "Think about why: 'Improve user retention' instead of 'Build feature Z'",
            'release': "Focus on value: 'Increase customer satisfaction' instead of 'Release version 2.0'",
        }
        return suggestions.get(output_word, "Try focusing on the business outcome or user impact instead.")

    def _is_contextual_start(self, text: str) -> bool:
        """Check if the objective starts with context-setting words."""
        contextual_starts = ['become', 'make', 'ensure', 'maintain', 'deliver']
        first_word = text.split()[0].lower()
        return first_word in contextual_starts


class KeyResultValidator(Validator):
    """
    Validator for key results.

    Ensures key results are measurable, specific, and time-bound.
    """

    # Words that indicate measurement
    MEASUREMENT_INDICATORS = [
        '%', 'percent', 'number', 'count', 'rate', 'ratio', 'score',
        'users', 'customers', 'revenue', 'arr', 'mrr', 'nps', 'csat',
        'retention', 'churn', 'conversion', 'engagement', 'adoption'
    ]

    # Vague terms to avoid
    VAGUE_TERMS = [
        'improve', 'better', 'more', 'less', 'increase', 'decrease',
        'enhance', 'optimize'  # When used without specifics
    ]

    def validate(self, document: Document) -> None:
        """Validate key result input."""
        text = document.text.strip()

        # Check minimum length
        if len(text.split()) < 3:
            raise ValidationError(
                message="📊 Key results should be at least 3 words. Include what you're measuring!",
                cursor_position=len(document.text)
            )

        # Check for measurability
        lower_text = text.lower()
        has_measurement = any(indicator in lower_text for indicator in self.MEASUREMENT_INDICATORS)
        has_number = bool(re.search(r'\d', text))

        if not (has_measurement or has_number):
            raise ValidationError(
                message="📈 Key results must be measurable. Include a specific metric or number.\n"
                        "Example: 'Increase DAU from 10k to 50k' or 'Achieve 80% customer satisfaction'",
                cursor_position=len(document.text)
            )

        # Check for vague language without specifics
        words = lower_text.split()
        for vague in self.VAGUE_TERMS:
            if vague in words and not has_number:
                raise ValidationError(
                    message=f"🎯 '{vague}' needs specifics. Add 'from X to Y' or a target number.\n"
                            f"Example: '{vague} from 20% to 35%'",
                    cursor_position=len(document.text)
                )

        # Check for binary outcomes (yes/no)
        if lower_text.startswith(('complete', 'finish', 'done', 'ship')):
            raise ValidationError(
                message="✨ This sounds binary (done/not done). Key results should be gradual.\n"
                        "Try: 'X% of users using feature' or 'X customers onboarded'",
                cursor_position=len(document.text)
            )


class ConfidenceValidator(Validator):
    """
    Validator for confidence scores with coaching feedback.
    """

    def validate(self, document: Document) -> None:
        """Validate confidence input."""
        text = document.text.strip()

        if not text:
            return  # Allow empty for default

        try:
            value = int(text)
        except ValueError:
            raise ValidationError(
                message="Please enter a number between 0 and 100",
                cursor_position=len(document.text)
            )

        if value < 0 or value > 100:
            raise ValidationError(
                message="Confidence must be between 0 and 100",
                cursor_position=len(document.text)
            )

        # Educational feedback (non-blocking)
        if value < 30:
            # Note: This is just validation, actual feedback is in the wizard
            pass  # Very ambitious
        elif value > 90:
            # Note: This is just validation, actual feedback is in the wizard
            pass  # Might be sandbagging


def validate_okr_completeness(objectives: list) -> tuple[bool, list[str]]:
    """
    Validate overall OKR completeness and quality.

    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []

    # Check number of objectives
    if len(objectives) == 0:
        warnings.append("No objectives set - you need at least one!")
        return False, warnings
    elif len(objectives) > 5:
        warnings.append("You have >5 objectives. Consider focusing on 3-5 for better execution.")
    elif len(objectives) == 1:
        warnings.append("Only 1 objective set. Consider adding 1-2 more for balanced growth.")

    # Check key results per objective
    for obj in objectives:
        if len(obj.key_results) == 0:
            warnings.append(f"Objective '{obj.title[:30]}...' has no key results!")
            return False, warnings
        elif len(obj.key_results) == 1:
            warnings.append(f"Objective '{obj.title[:30]}...' has only 1 key result. Add 2-3 more.")
        elif len(obj.key_results) > 5:
            warnings.append(f"Objective '{obj.title[:30]}...' has >5 key results. That's a lot to track!")

    # Check confidence distribution
    all_confidences = []
    for obj in objectives:
        for kr in obj.key_results:
            if kr.confidence:
                all_confidences.append(kr.confidence)

    if all_confidences:
        avg_confidence = sum(all_confidences) / len(all_confidences)

        if avg_confidence < 30:
            warnings.append("Overall confidence is very low (<30%). These OKRs might be too ambitious.")
        elif avg_confidence > 80:
            warnings.append("Overall confidence is very high (>80%). Consider more stretch goals.")

        # Check for variety
        if len(set(all_confidences)) == 1:
            warnings.append("All key results have the same confidence. Consider varying ambition levels.")

    return len(warnings) == 0, warnings


def get_okr_health_score(objectives: list) -> tuple[int, str]:
    """
    Calculate overall OKR health score.

    Returns:
        Tuple of (score 0-100, health_message)
    """
    if not objectives:
        return 0, "No OKRs set"

    score = 100

    # Objective count (optimal is 3)
    obj_count = len(objectives)
    if obj_count < 2:
        score -= 20
    elif obj_count > 4:
        score -= 10

    # Key results per objective (optimal is 3-4)
    for obj in objectives:
        kr_count = len(obj.key_results)
        if kr_count < 2:
            score -= 10
        elif kr_count > 5:
            score -= 5

    # Confidence distribution (optimal is 50-70%)
    confidences = []
    for obj in objectives:
        for kr in obj.key_results:
            if kr.confidence:
                confidences.append(kr.confidence)

    if confidences:
        avg_confidence = sum(confidences) / len(confidences)

        if 50 <= avg_confidence <= 70:
            pass  # Optimal
        elif 40 <= avg_confidence < 50 or 70 < avg_confidence <= 80:
            score -= 10
        else:
            score -= 20
    else:
        score -= 30  # No confidence scores

    # Generate message
    if score >= 90:
        message = "Excellent! Your OKRs are well-balanced and ambitious."
    elif score >= 70:
        message = "Good! Your OKRs are solid with room for minor improvements."
    elif score >= 50:
        message = "Fair. Consider refining your OKRs for better focus and ambition."
    else:
        message = "Needs work. Your OKRs could benefit from restructuring."

    return max(0, score), message