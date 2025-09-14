"""
PM-Kit Prompts module.

This module contains centralized prompt templates and UI text for all
PM-Kit features.
"""

from pmkit.prompts.library import PromptLibrary
from pmkit.prompts.onboarding_prompts import OnboardingPrompts

__all__ = [
    "PromptLibrary",
    "OnboardingPrompts",
]