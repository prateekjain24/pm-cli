"""
Manual input form with review/edit capabilities for PM-Kit onboarding.

This module provides a delightful fallback experience when automated enrichment
fails or returns partial results. It focuses on editing what's wrong rather than
entering everything from scratch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.validation import ValidationError, Validator
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from pmkit.utils.console import console as pmkit_console
from pmkit.utils.logger import get_logger
from pmkit.utils.paths import safe_write

logger = get_logger(__name__)


# ============================================================
# ENUMS & CONSTANTS
# ============================================================

class FieldStatus(Enum):
    """Status of enriched fields."""
    CONFIRMED = "confirmed"     # ✅ Data looks good
    REVIEW = "review"           # ⚠️ Data present but needs review
    MISSING = "missing"         # ❌ Required field empty
    OPTIONAL = "optional"       # ➖ Optional field, can skip


class ValidationLevel(Enum):
    """Validation severity levels."""
    ERROR = "error"             # Blocking issue
    WARNING = "warning"         # Advisory, not blocking
    AUTOCORRECT = "autocorrect"  # Automatic fix applied


@dataclass
class FieldMetadata:
    """Metadata for a context field."""
    name: str
    display_name: str
    field_type: str  # text, choice, number, list
    required: bool
    default: Any = None
    help_text: str = ""
    validator: Optional[Validator] = None
    completer: Optional[WordCompleter] = None
    choices: Optional[List[str]] = None
    phase: int = 1  # 1=Essentials, 2=Enrichment, 3=Advanced
    b2b_specific: bool = False
    b2c_specific: bool = False


# Field definitions
FIELD_DEFINITIONS = {
    # Phase 1: Essentials (MUST_HAVE)
    'company_name': FieldMetadata(
        name='company_name',
        display_name='Company Name',
        field_type='text',
        required=True,
        help_text='Your company or project name',
        phase=1,
    ),
    'company_type': FieldMetadata(
        name='company_type',
        display_name='Business Model',
        field_type='choice',
        required=True,
        choices=['B2B', 'B2C', 'B2B2C'],
        default='B2B',
        help_text='B2B (enterprise), B2C (consumer), or B2B2C (both)',
        phase=1,
    ),
    'product_name': FieldMetadata(
        name='product_name',
        display_name='Product Name',
        field_type='text',
        required=True,
        help_text='Name of your product or service',
        phase=1,
    ),
    'product_description': FieldMetadata(
        name='product_description',
        display_name='Product Description',
        field_type='text',
        required=True,
        help_text='Brief description including target audience (e.g., "AI code review for enterprise teams")',
        phase=1,
    ),

    # Phase 2: Enrichment (SHOULD_HAVE)
    'company_stage': FieldMetadata(
        name='company_stage',
        display_name='Company Stage',
        field_type='choice',
        required=False,
        choices=['idea', 'seed', 'growth', 'mature'],
        default='growth',
        help_text='Current stage of your company',
        phase=2,
    ),
    'target_market': FieldMetadata(
        name='target_market',
        display_name='Target Market',
        field_type='text',
        required=False,
        help_text='Primary target audience (e.g., SMBs, Enterprise, Consumers)',
        phase=2,
    ),
    'competitors': FieldMetadata(
        name='competitors',
        display_name='Main Competitors',
        field_type='list',
        required=False,
        help_text='List 2-3 main competitors (comma-separated)',
        phase=2,
    ),
    'north_star_metric': FieldMetadata(
        name='north_star_metric',
        display_name='North Star Metric',
        field_type='text',
        required=False,
        help_text='Primary success metric (e.g., MRR for B2B, MAU for B2C)',
        phase=2,
    ),

    # Phase 3: Advanced (NICE_TO_HAVE)
    'website': FieldMetadata(
        name='website',
        display_name='Website',
        field_type='text',
        required=False,
        help_text='Company website URL',
        phase=3,
    ),
    'team_size': FieldMetadata(
        name='team_size',
        display_name='Team Size',
        field_type='number',
        required=False,
        help_text='Number of team members',
        phase=3,
    ),
    'pricing_model': FieldMetadata(
        name='pricing_model',
        display_name='Pricing Model',
        field_type='text',
        required=False,
        help_text='How you charge (e.g., subscription, freemium, one-time)',
        phase=3,
        b2b_specific=True,
    ),
    'user_count': FieldMetadata(
        name='user_count',
        display_name='User Count',
        field_type='number',
        required=False,
        help_text='Current number of users/customers',
        phase=3,
    ),
    'differentiator': FieldMetadata(
        name='differentiator',
        display_name='Key Differentiator',
        field_type='text',
        required=False,
        help_text='What makes you unique vs competitors',
        phase=3,
    ),
}


class ManualInputForm:
    """
    Handles manual input with review/edit capabilities.

    Features:
    - Review & Edit pattern for partial enrichment
    - Field-by-field validation with helpful feedback
    - Auto-save after each field
    - Smart defaults based on context
    - Visual status indicators
    """

    def __init__(self, console: Optional[Console] = None):
        """Initialize the manual input form."""
        self.console = console or pmkit_console.console
        self.session = PromptSession()
        self.state_file = Path.home() / ".pmkit" / "manual_input_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def review_and_edit(
        self,
        enriched_data: Dict[str, Any],
        company_type: str = 'b2b',
        required_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Review and edit enriched data with visual status indicators.

        Args:
            enriched_data: Data from automated enrichment (may be partial)
            company_type: B2B or B2C to customize fields
            required_only: Only show required fields

        Returns:
            Updated data after review/edit
        """
        # Analyze field status
        field_status = self._analyze_field_status(enriched_data, company_type)

        # Show review table
        self._show_review_table(field_status, enriched_data)

        # Check if any fields need attention
        needs_review = any(
            status in [FieldStatus.REVIEW, FieldStatus.MISSING]
            for status in field_status.values()
        )

        if not needs_review and not required_only:
            self.console.print("\n[green]✅ All fields look good![/green]")
            # Use prompt_toolkit for consistency (avoid library conflict with Rich)
            confirm = self.session.prompt(
                HTML("<ansigreen>Would you like to review/edit any fields? (y/n): </ansigreen>")
            ).strip().lower()
            if confirm not in ['y', 'yes']:
                return enriched_data

        # Enter edit mode
        updated_data = enriched_data.copy()

        while True:
            # Show current status
            self.console.print("\n[cyan]Navigation:[/cyan]")
            self.console.print("  • Enter field name to edit")
            self.console.print("  • Type 'next' to go to next missing field")
            self.console.print("  • Type 'done' when finished")
            self.console.print("  • Type 'status' to see current values")

            # Get user choice
            choice = self.session.prompt(
                HTML("<ansigreen>Edit field (or command): </ansigreen>"),
                completer=WordCompleter(
                    list(field_status.keys()) + ['next', 'done', 'status'],
                    ignore_case=True
                ),
            ).strip().lower()

            if choice == 'done':
                break
            elif choice == 'status':
                self._show_review_table(field_status, updated_data)
            elif choice == 'next':
                # Find next field needing attention
                next_field = self._find_next_field(field_status)
                if next_field:
                    updated_data[next_field] = self._edit_field(
                        next_field,
                        updated_data.get(next_field),
                        company_type
                    )
                    field_status[next_field] = FieldStatus.CONFIRMED
                    self._save_progress(updated_data)
                else:
                    self.console.print("[green]No more fields need attention[/green]")
            elif choice in field_status:
                # Edit specific field
                updated_data[choice] = self._edit_field(
                    choice,
                    updated_data.get(choice),
                    company_type
                )
                field_status[choice] = FieldStatus.CONFIRMED
                self._save_progress(updated_data)
            else:
                self.console.print(f"[red]Unknown field or command: {choice}[/red]")

        return updated_data

    def collect_missing_fields(
        self,
        existing_data: Dict[str, Any],
        company_type: str = 'b2b',
        phase: int = 2,
    ) -> Dict[str, Any]:
        """
        Collect only missing required fields.

        Args:
            existing_data: Current data
            company_type: B2B or B2C
            phase: Which phase to collect (1=Essentials, 2=Enrichment, 3=Advanced)

        Returns:
            Updated data with missing fields filled
        """
        updated_data = existing_data.copy()

        # Get fields for this phase
        phase_fields = [
            (name, meta) for name, meta in FIELD_DEFINITIONS.items()
            if meta.phase <= phase and self._should_show_field(meta, company_type)
        ]

        # Collect missing required fields
        for field_name, metadata in phase_fields:
            if metadata.required and not updated_data.get(field_name):
                self.console.print(f"\n[yellow]Missing required field: {metadata.display_name}[/yellow]")
                updated_data[field_name] = self._collect_field(metadata, company_type)
                self._save_progress(updated_data)

        return updated_data

    def _analyze_field_status(
        self,
        data: Dict[str, Any],
        company_type: str,
    ) -> Dict[str, FieldStatus]:
        """Analyze the status of each field."""
        status = {}

        for field_name, metadata in FIELD_DEFINITIONS.items():
            if not self._should_show_field(metadata, company_type):
                continue

            value = data.get(field_name)

            if metadata.required:
                if not value:
                    status[field_name] = FieldStatus.MISSING
                elif self._needs_review(field_name, value):
                    status[field_name] = FieldStatus.REVIEW
                else:
                    status[field_name] = FieldStatus.CONFIRMED
            else:
                if value:
                    status[field_name] = FieldStatus.CONFIRMED
                else:
                    status[field_name] = FieldStatus.OPTIONAL

        return status

    def _needs_review(self, field_name: str, value: Any) -> bool:
        """Check if a field value needs review."""
        # Generic/test data patterns
        suspicious_patterns = ['test', 'demo', 'asdf', 'todo', 'tbd']

        if isinstance(value, str):
            value_lower = value.lower()
            # Check for suspicious patterns
            if any(pattern in value_lower for pattern in suspicious_patterns):
                return True

            # Field-specific checks
            if field_name == 'product_description' and len(value.split()) < 5:
                return True  # Too brief
            elif field_name == 'website' and not value.startswith('http'):
                return True  # Missing protocol

        return False

    def _show_review_table(
        self,
        field_status: Dict[str, FieldStatus],
        data: Dict[str, Any],
    ) -> None:
        """Display review table with status indicators."""
        table = Table(title="Review Enrichment Results", box=box.ROUNDED)
        table.add_column("Status", style="cyan", width=8)
        table.add_column("Field", style="white", width=20)
        table.add_column("Value", style="white", width=100, overflow="fold")

        status_icons = {
            FieldStatus.CONFIRMED: "✅",
            FieldStatus.REVIEW: "⚠️",
            FieldStatus.MISSING: "❌",
            FieldStatus.OPTIONAL: "➖",
        }

        # Group by phase
        for phase in [1, 2, 3]:
            phase_fields = [
                (name, FIELD_DEFINITIONS[name])
                for name, meta in FIELD_DEFINITIONS.items()
                if name in field_status and meta.phase == phase
            ]

            if not phase_fields:
                continue

            # Add phase separator
            if phase > 1:
                table.add_row("", f"[dim]Phase {phase}[/dim]", "")

            for field_name, metadata in phase_fields:
                status = field_status.get(field_name, FieldStatus.OPTIONAL)
                icon = status_icons[status]
                value = data.get(field_name, "")

                # Format value display
                if isinstance(value, list):
                    value_str = ", ".join(str(v) for v in value[:5])  # Limit to 5 items
                    if len(value) > 5:
                        value_str += f" ... (+{len(value)-5} more)"
                elif not value:
                    value_str = "[dim]Not set[/dim]"
                    if status == FieldStatus.MISSING:
                        value_str = "[red]Missing - Required[/red]"
                else:
                    value_str = str(value)
                    # Truncate very long values
                    if len(value_str) > 200:
                        value_str = value_str[:197] + "..."
                    if status == FieldStatus.REVIEW:
                        value_str = f"{value_str} [yellow][Edit?][/yellow]"

                table.add_row(icon, metadata.display_name, value_str)

        # Add legend
        self.console.print("\n")
        self.console.print(table)
        self.console.print("\n[dim]Legend: ✅ Confirmed  ⚠️ Review needed  ❌ Missing  ➖ Optional[/dim]")

    def _find_next_field(
        self,
        field_status: Dict[str, FieldStatus],
    ) -> Optional[str]:
        """Find the next field needing attention."""
        # Priority: MISSING > REVIEW > None
        for field_name, status in field_status.items():
            if status == FieldStatus.MISSING:
                return field_name

        for field_name, status in field_status.items():
            if status == FieldStatus.REVIEW:
                return field_name

        return None

    def _edit_field(
        self,
        field_name: str,
        current_value: Any,
        company_type: str,
    ) -> Any:
        """Edit a single field."""
        # Special handling for OKRs - use the OKR wizard
        if field_name == 'okrs':
            return self._edit_okrs(current_value, company_type)

        metadata = FIELD_DEFINITIONS.get(field_name)
        if not metadata:
            return current_value

        self.console.print(f"\n[cyan]Editing: {metadata.display_name}[/cyan]")
        if metadata.help_text:
            self.console.print(f"[dim]{metadata.help_text}[/dim]")

        # Show current value
        if current_value:
            if isinstance(current_value, list):
                current_str = ", ".join(str(v) for v in current_value)
            else:
                current_str = str(current_value)
            self.console.print(f"Current value: [yellow]{current_str}[/yellow]")

        return self._collect_field(metadata, company_type, current_value)

    def _edit_okrs(self, current_okrs: Any, company_type: str) -> Any:
        """
        Edit OKRs using the delightful OKR wizard.

        Args:
            current_okrs: Current OKR data (if any)
            company_type: 'b2b' or 'b2c'

        Returns:
            Updated OKR data
        """
        from pmkit.agents.okr_wizard import OKRWizard
        from pmkit.utils.async_utils import run_async
        from pmkit.context.models import Objective, KeyResult

        # Map company type to expected format
        company_type_upper = 'B2B' if 'b2b' in company_type.lower() else 'B2C'

        # Create OKR wizard
        okr_wizard = OKRWizard(
            console=self.console,
            state_file=self.state_file.parent / 'okr_edit_state.yaml',
            company_type=company_type_upper,
            company_stage='growth'  # Default to growth for editing
        )

        # Pre-populate with existing OKRs if available
        if current_okrs and isinstance(current_okrs, list):
            objectives = []
            for obj_data in current_okrs:
                if isinstance(obj_data, dict):
                    key_results = []
                    for kr_data in obj_data.get('key_results', []):
                        key_results.append(KeyResult(
                            description=kr_data.get('description', ''),
                            target_value=kr_data.get('target_value'),
                            current_value=kr_data.get('current_value'),
                            confidence=kr_data.get('confidence')
                        ))
                    objectives.append(Objective(
                        title=obj_data.get('title', ''),
                        key_results=key_results
                    ))
            okr_wizard.state.objectives = objectives

        # Run the wizard
        try:
            from pmkit.utils.async_utils import run_async
            okr_context = run_async(okr_wizard.run())

            # Convert back to dictionary format
            okr_list = []
            for obj in okr_context.objectives:
                okr_dict = {
                    'title': obj.title,
                    'key_results': []
                }
                for kr in obj.key_results:
                    okr_dict['key_results'].append({
                        'description': kr.description,
                        'target_value': kr.target_value,
                        'current_value': kr.current_value,
                        'confidence': kr.confidence
                    })
                okr_list.append(okr_dict)

            return okr_list

        except Exception as e:
            self.console.print(f"[red]Error editing OKRs: {e}[/red]")
            return current_okrs or []

    def _collect_field(
        self,
        metadata: FieldMetadata,
        company_type: str,
        current_value: Any = None,
    ) -> Any:
        """Collect a single field value."""
        # Prepare default
        default = current_value or metadata.default
        if not default and metadata.name == 'north_star_metric':
            default = 'MRR' if company_type.lower() == 'b2b' else 'MAU'
        elif not default and metadata.name == 'target_market':
            default = 'SMBs' if company_type.lower() == 'b2b' else 'Consumers'

        # Handle different field types
        if metadata.field_type == 'choice' and metadata.choices:
            # Show choices
            for i, choice in enumerate(metadata.choices, 1):
                self.console.print(f"  {i}. {choice}")

            choice_input = self.session.prompt(
                HTML(f"<ansigreen>{metadata.display_name}: </ansigreen>"),
                default=str(default) if default else "",
            ).strip()

            # Map number to choice or use direct input
            if choice_input.isdigit():
                idx = int(choice_input) - 1
                if 0 <= idx < len(metadata.choices):
                    return metadata.choices[idx]
            return choice_input or default

        elif metadata.field_type == 'list':
            list_input = self.session.prompt(
                HTML(f"<ansigreen>{metadata.display_name} (comma-separated): </ansigreen>"),
                default=", ".join(default) if isinstance(default, list) else str(default) if default else "",
            ).strip()

            if list_input:
                return [item.strip() for item in list_input.split(',') if item.strip()]
            return default or []

        elif metadata.field_type == 'number':
            while True:
                num_input = self.session.prompt(
                    HTML(f"<ansigreen>{metadata.display_name}: </ansigreen>"),
                    default=str(default) if default else "",
                ).strip()

                if not num_input and not metadata.required:
                    return None

                try:
                    return int(num_input)
                except ValueError:
                    self.console.print("[red]Please enter a valid number[/red]")

        else:  # text
            text_input = self.session.prompt(
                HTML(f"<ansigreen>{metadata.display_name}: </ansigreen>"),
                default=str(default) if default else "",
                completer=metadata.completer,
                validator=metadata.validator,
            ).strip()

            return text_input or default

    def _should_show_field(
        self,
        metadata: FieldMetadata,
        company_type: str,
    ) -> bool:
        """Check if field should be shown based on company type."""
        company_type_lower = company_type.lower()

        if metadata.b2b_specific and 'b2b' not in company_type_lower:
            return False
        if metadata.b2c_specific and 'b2c' not in company_type_lower:
            return False

        return True

    def _save_progress(self, data: Dict[str, Any]) -> None:
        """Save progress after each field change."""
        try:
            save_data = {
                'data': data,
                'last_modified': datetime.now().isoformat(),
                'version': '1.0',
            }

            safe_write(
                self.state_file,
                json.dumps(save_data, indent=2),
                backup=False  # Don't need backup for state files
            )

            # Show subtle saved indicator
            self.console.print("[dim]✓ Saved[/dim]", end="\r")

        except Exception as e:
            logger.warning(f"Failed to save progress: {e}")

    def load_progress(self) -> Optional[Dict[str, Any]]:
        """Load saved progress if available."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    save_data = json.load(f)

                # Check staleness
                last_modified = datetime.fromisoformat(save_data['last_modified'])
                days_old = (datetime.now() - last_modified).days

                if days_old > 7:
                    self.console.print(
                        f"[yellow]⚠️ Your saved data is {days_old} days old. "
                        "Company info may be outdated.[/yellow]"
                    )

                return save_data['data']

        except Exception as e:
            logger.warning(f"Failed to load progress: {e}")

        return None

    def clear_progress(self) -> None:
        """Clear saved progress."""
        try:
            if self.state_file.exists():
                self.state_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to clear progress: {e}")