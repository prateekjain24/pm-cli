"""
OKR Wizard - Delightful OKR collection experience for PMs.

This module provides a conversation-style OKR collection flow with
progressive disclosure, smart templates, and confidence coaching.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.validation import ValidationError, Validator
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
import yaml

from pmkit.context.models import KeyResult, Objective, OKRContext
from pmkit.utils.console import console as pmkit_console
from pmkit.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OKRWizardState:
    """State management for OKR wizard."""

    objectives: List[Objective]
    current_quarter: str
    company_type: str  # 'B2B' or 'B2C'
    company_stage: str  # 'seed', 'growth', 'scale'
    saved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for persistence."""
        return {
            'objectives': [
                {
                    'title': obj.title,
                    'key_results': [
                        {
                            'description': kr.description,
                            'target_value': kr.target_value,
                            'current_value': kr.current_value,
                            'confidence': kr.confidence
                        }
                        for kr in obj.key_results
                    ]
                }
                for obj in self.objectives
            ],
            'current_quarter': self.current_quarter,
            'company_type': self.company_type,
            'company_stage': self.company_stage,
            'saved': self.saved
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OKRWizardState:
        """Create state from dictionary."""
        objectives = []
        for obj_data in data.get('objectives', []):
            key_results = [
                KeyResult(**kr_data)
                for kr_data in obj_data.get('key_results', [])
            ]
            objectives.append(Objective(
                title=obj_data['title'],
                key_results=key_results
            ))

        return cls(
            objectives=objectives,
            current_quarter=data.get('current_quarter', 'Q1 2025'),
            company_type=data.get('company_type', 'B2B'),
            company_stage=data.get('company_stage', 'growth'),
            saved=data.get('saved', False)
        )


class OKRWizard:
    """
    Interactive OKR collection wizard with progressive disclosure.

    Features:
    - Three-phase flow: Quick Win, Expand, Polish
    - Smart templates based on B2B/B2C context
    - Confidence coaching with visual indicators
    - Auto-save and resume capability
    - Delightful celebration moments
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        state_file: Optional[Path] = None,
        company_type: str = 'B2B',
        company_stage: str = 'growth'
    ):
        """
        Initialize OKR wizard.

        Args:
            console: Rich console for output
            state_file: Path to save/load state
            company_type: 'B2B' or 'B2C' for context-aware templates
            company_stage: 'seed', 'growth', or 'scale'
        """
        self.console = console or pmkit_console.console
        self.state_file = state_file or Path.home() / '.pmkit' / 'okr_state.yaml'
        self.session = PromptSession()

        # Initialize or load state
        self.state = self._load_state() or OKRWizardState(
            objectives=[],
            current_quarter=self._get_current_quarter(),
            company_type=company_type,
            company_stage=company_stage
        )

    def _get_current_quarter(self) -> str:
        """Get current quarter string (e.g., 'Q1 2025')."""
        from datetime import datetime
        now = datetime.now()
        quarter = (now.month - 1) // 3 + 1
        return f"Q{quarter} {now.year}"

    def _load_state(self) -> Optional[OKRWizardState]:
        """Load saved state if exists."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = yaml.safe_load(f)
                    if data:
                        return OKRWizardState.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load OKR state: {e}")
        return None

    def _save_state(self):
        """Save current state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            yaml.dump(self.state.to_dict(), f, default_flow_style=False)
        self.state.saved = True

    async def run(self) -> OKRContext:
        """
        Run the OKR collection wizard.

        Returns:
            OKRContext with collected objectives and key results
        """
        # Welcome message
        self._show_welcome()

        # Check for existing state
        if self.state.objectives and not self.state.saved:
            if await self._confirm_resume():
                self._show_okr_preview()
            else:
                self.state.objectives = []

        # Phase 1: Quick Win (30 seconds)
        await self._phase_quick_win()

        # Phase 2: Expand (optional, 2 minutes)
        if await self._confirm_add_more():
            await self._phase_expand()

        # Phase 3: Polish (optional, 1 minute)
        if await self._confirm_polish():
            await self._phase_polish()

        # Final celebration
        self._show_completion()

        # Clean up state file
        if self.state_file.exists():
            self.state_file.unlink()

        return OKRContext(
            objectives=self.state.objectives,
            quarter=self.state.current_quarter
        )

    def _show_welcome(self):
        """Show welcome message with context."""
        welcome = Panel(
            Text.from_markup(
                f"[bold cyan]🎯 Let's Set Your OKRs for {self.state.current_quarter}[/]\n\n"
                f"I'll guide you through creating impactful Objectives and Key Results.\n"
                f"This should take about [yellow]3 minutes[/] total.\n\n"
                f"[dim]💡 Pro tip: Good OKRs are ambitious but achievable (50-70% confidence is perfect)[/]"
            ),
            title="[bold]OKR Setup Wizard[/]",
            border_style="cyan"
        )
        self.console.print(welcome)
        self.console.print()

    async def _confirm_resume(self) -> bool:
        """Ask if user wants to resume from saved state."""
        response = await self.session.prompt_async(
            HTML(
                f"<ansigreen>📝 Found {len(self.state.objectives)} saved objective(s).</ansigreen> "
                "Resume where you left off? (y/n): "
            )
        )
        return response.lower() in ['y', 'yes']

    async def _phase_quick_win(self):
        """Phase 1: Capture the most important objective quickly."""
        self.console.print("[bold cyan]Phase 1: Your Primary Goal[/] (30 seconds)\n")

        if not self.state.objectives:
            # Collect first objective
            objective = await self._collect_objective(is_first=True)
            self.state.objectives.append(objective)

            # Auto-save after first objective
            self._save_state()

            # Show celebration
            self.console.print("[bold green]🎯 Great first objective![/] This sets a clear direction.\n")

            # Preview what we have
            self._show_okr_preview()

    async def _phase_expand(self):
        """Phase 2: Add more objectives if desired."""
        self.console.print("[bold cyan]Phase 2: Additional Goals[/] (optional)\n")

        while len(self.state.objectives) < 3:  # Recommend max 3 objectives
            objective = await self._collect_objective(is_first=False)
            self.state.objectives.append(objective)

            # Auto-save after each objective
            self._save_state()

            # Show updated preview
            self._show_okr_preview()

            if len(self.state.objectives) < 3:
                if not await self._confirm_add_another():
                    break
            else:
                self.console.print("[dim]✓ You've set 3 objectives - a perfect focused set![/]\n")
                break

    async def _phase_polish(self):
        """Phase 3: Refine confidence levels and targets."""
        self.console.print("[bold cyan]Phase 3: Polish & Refine[/] (optional)\n")

        # Show current confidence levels
        self._show_confidence_summary()

        # Offer to adjust specific key results
        for obj_idx, objective in enumerate(self.state.objectives):
            for kr_idx, kr in enumerate(objective.key_results):
                if kr.confidence and (kr.confidence < 30 or kr.confidence > 90):
                    if await self._confirm_adjust_confidence(objective, kr):
                        new_confidence = await self._collect_confidence(kr.description)
                        kr.confidence = new_confidence
                        self._save_state()

    async def _collect_objective(self, is_first: bool = False) -> Objective:
        """Collect a single objective with its key results."""
        # Get objective title
        if is_first:
            prompt = f"What's your most important goal for {self.state.current_quarter}?"
        else:
            prompt = f"What's another key goal for {self.state.current_quarter}?"

        title = await self._prompt_with_validation(
            prompt,
            self._get_objective_validator(),
            self._get_objective_template()
        )

        # Collect key results
        self.console.print(f"\n[bold]Now let's define success metrics for:[/] {title}\n")

        key_results = []
        for i in range(3):  # Collect up to 3 KRs
            if i > 0 and not await self._confirm_add_kr(i):
                break

            kr = await self._collect_key_result(i + 1)
            key_results.append(kr)

            # Show progress
            self._show_kr_added(kr)

        return Objective(title=title, key_results=key_results)

    async def _collect_key_result(self, number: int) -> KeyResult:
        """Collect a single key result."""
        # Get KR description
        description = await self._prompt_with_validation(
            f"Key Result #{number}: What specific outcome will you measure?",
            self._get_kr_validator(),
            self._get_kr_template()
        )

        # Get target value
        target = await self._prompt_with_validation(
            "What's the target value?",
            None,
            "e.g., 100k, 80%, $1M"
        )

        # Get current value (optional)
        current = None
        # Use a simple prompt instead of confirm in async context
        know_current = await self._prompt_with_validation(
            "Do you know the current value? (yes/no)",
            None,
            "Type 'yes' or 'no'"
        )
        if know_current.lower() in ['yes', 'y']:
            current = await self._prompt_with_validation(
                "Current value:",
                None,
                "e.g., 50k, 40%, $500k"
            )

        # Get confidence level
        confidence = await self._collect_confidence(description)

        return KeyResult(
            description=description,
            target_value=target,
            current_value=current,
            confidence=confidence
        )

    async def _collect_confidence(self, kr_description: str) -> int:
        """Collect confidence level with coaching."""
        self.console.print("\n[bold]Confidence Assessment[/]")
        self.console.print(f"[dim]For: {kr_description}[/]")
        self.console.print(self._get_confidence_guidance())

        while True:
            try:
                value = await self.session.prompt_async(
                    HTML("Your confidence (0-100): "),
                    default="50"
                )
                confidence = int(value)
                if 0 <= confidence <= 100:
                    # Show feedback
                    self._show_confidence_feedback(confidence)
                    return confidence
                else:
                    self.console.print("[red]Please enter a value between 0 and 100[/]")
            except ValueError:
                self.console.print("[red]Please enter a number[/]")

    def _show_okr_preview(self):
        """Show current OKRs in a tree format."""
        tree = Tree(f"[bold cyan]📊 Your OKRs for {self.state.current_quarter}[/]")

        for objective in self.state.objectives:
            obj_branch = tree.add(f"[bold]{objective.title}[/]")

            for kr in objective.key_results:
                confidence_icon = self._get_confidence_icon(kr.confidence)
                kr_text = f"{confidence_icon} {kr.description}"
                if kr.target_value:
                    kr_text += f" → [yellow]{kr.target_value}[/]"
                if kr.confidence:
                    kr_text += f" ({kr.confidence}%)"
                obj_branch.add(kr_text)

        self.console.print()
        self.console.print(tree)
        self.console.print()

    def _show_confidence_summary(self):
        """Show summary of confidence levels across all OKRs."""
        table = Table(title="Confidence Summary", show_header=True)
        table.add_column("Objective", style="cyan")
        table.add_column("Avg Confidence", justify="center")
        table.add_column("Health", justify="center")

        for objective in self.state.objectives:
            confidences = [kr.confidence for kr in objective.key_results if kr.confidence]
            if confidences:
                avg = sum(confidences) // len(confidences)
                health = self._get_confidence_health(avg)
                table.add_row(
                    objective.title[:50] + "..." if len(objective.title) > 50 else objective.title,
                    f"{avg}%",
                    health
                )

        self.console.print(table)
        self.console.print()

    def _show_completion(self):
        """Show celebration message on completion."""
        completion = Panel(
            Text.from_markup(
                "[bold green]🚀 OKRs Complete![/]\n\n"
                f"You've set [cyan]{len(self.state.objectives)}[/] objective(s) with "
                f"[cyan]{sum(len(obj.key_results) for obj in self.state.objectives)}[/] key results.\n\n"
                "[dim]Your OKRs have been saved and you're ready to execute![/]"
            ),
            title="[bold]Success![/]",
            border_style="green"
        )
        self.console.print(completion)

    # Helper methods for templates, validators, and formatting

    def _get_objective_validator(self) -> Validator:
        """Get validator for objective input."""
        from pmkit.validators.okr_validators import ObjectiveValidator
        return ObjectiveValidator()

    def _get_kr_validator(self) -> Validator:
        """Get validator for key result input."""
        from pmkit.validators.okr_validators import KeyResultValidator
        return KeyResultValidator()

    def _get_objective_template(self) -> str:
        """Get template hint for objectives."""
        if self.state.company_type == 'B2B':
            return "e.g., 'Achieve product-market fit in enterprise segment'"
        else:
            return "e.g., 'Reach 1M monthly active users'"

    def _get_kr_template(self) -> str:
        """Get template hint for key results."""
        if self.state.company_type == 'B2B':
            return "e.g., 'Increase ARR from $2M to $5M'"
        else:
            return "e.g., 'Improve D7 retention from 40% to 60%'"

    def _get_confidence_icon(self, confidence: Optional[int]) -> str:
        """Get icon based on confidence level."""
        if not confidence:
            return "⭕"
        elif confidence >= 70:
            return "🟢"
        elif confidence >= 50:
            return "🟡"
        else:
            return "🔴"

    def _get_confidence_health(self, avg_confidence: int) -> str:
        """Get health indicator for average confidence."""
        if avg_confidence >= 70:
            return "[green]Strong ✓[/]"
        elif avg_confidence >= 50:
            return "[yellow]Good ✓[/]"
        else:
            return "[red]At Risk ⚠[/]"

    def _get_confidence_guidance(self) -> str:
        """Get guidance text for confidence scoring."""
        return (
            "[dim]"
            "🟢 70-100%: High confidence (might be too easy?)\n"
            "🟡 50-70%: Perfect OKR stretch goal\n"
            "🔴 0-50%: Very ambitious (ensure it's realistic)\n"
            "[/]"
        )

    def _show_confidence_feedback(self, confidence: int):
        """Show feedback based on confidence level."""
        if confidence >= 90:
            self.console.print("[yellow]⚡ Very high confidence! Consider a more ambitious target.[/]")
        elif confidence >= 70:
            self.console.print("[green]💪 Strong confidence! This feels achievable.[/]")
        elif confidence >= 50:
            self.console.print("[green]🎯 Perfect stretch goal! This is ideal OKR territory.[/]")
        elif confidence >= 30:
            self.console.print("[yellow]🚀 Ambitious target! Make sure you have a clear path.[/]")
        else:
            self.console.print("[red]⚠️ Very ambitious! Consider breaking this down or adjusting the target.[/]")

    def _show_kr_added(self, kr: KeyResult):
        """Show confirmation when KR is added."""
        icon = self._get_confidence_icon(kr.confidence)
        self.console.print(f"\n{icon} Added: {kr.description} → {kr.target_value}\n")

    async def _prompt_with_validation(
        self,
        prompt: str,
        validator: Optional[Validator],
        hint: Optional[str] = None
    ) -> str:
        """Prompt with optional validation and hint."""
        if hint:
            self.console.print(f"[dim]{hint}[/]")

        return await self.session.prompt_async(
            HTML(f"{prompt}\n> "),
            validator=validator
        )

    async def _confirm_add_more(self) -> bool:
        """Ask if user wants to add more objectives."""
        if len(self.state.objectives) >= 3:
            return False
        response = await self.session.prompt_async(
            HTML("<ansigreen>Would you like to add another objective?</ansigreen> (recommended: 2-3 total) (y/n): ")
        )
        return response.lower() in ['y', 'yes']

    async def _confirm_add_another(self) -> bool:
        """Ask if user wants to add another objective during expand phase."""
        response = await self.session.prompt_async(
            HTML("<dim>Add another objective? (y/n): </dim>")
        )
        return response.lower() in ['y', 'yes']

    async def _confirm_add_kr(self, count: int) -> bool:
        """Ask if user wants to add another key result."""
        if count >= 3:
            return False
        response = await self.session.prompt_async(
            HTML(f"<dim>Add key result #{count + 1}? (recommended: 3-5 per objective) (y/n): </dim>")
        )
        return response.lower() in ['y', 'yes']

    async def _confirm_polish(self) -> bool:
        """Ask if user wants to polish confidence levels."""
        response = await self.session.prompt_async(
            HTML("<ansigreen>Would you like to review and adjust confidence levels? (y/n): </ansigreen>")
        )
        return response.lower() in ['y', 'yes']

    async def _confirm_adjust_confidence(self, objective: Objective, kr: KeyResult) -> bool:
        """Ask if user wants to adjust a specific KR's confidence."""
        icon = self._get_confidence_icon(kr.confidence)
        response = await self.session.prompt_async(
            HTML(
                f"<dim>Adjust confidence for:</dim> {kr.description} "
                f"(currently {icon} {kr.confidence}%) (y/n): "
            )
        )
        return response.lower() in ['y', 'yes']