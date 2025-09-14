"""
OnboardingAgent - Handles the complete onboarding flow for PM-Kit.

This module implements a wizard-style onboarding experience with progressive
disclosure, auto-enrichment, and beautiful Rich-based UI.
"""

from __future__ import annotations

import asyncio
import json
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table

from pmkit.config.models import LLMProviderConfig
from pmkit.context.manager import ContextManager
from pmkit.context.models import (
    CompanyContext,
    Context,
    KeyResult,
    MarketContext,
    Objective,
    OKRContext,
    ProductContext,
    TeamContext,
)
from pmkit.context.validator import ContextValidator
from pmkit.exceptions import PMKitError
from pmkit.llm.grounding import GroundingAdapter
from pmkit.llm.search.base import SearchOptions
from pmkit.prompts.library import PromptLibrary
from pmkit.prompts.onboarding_prompts import OnboardingPrompts
from pmkit.utils.console import console as pmkit_console
from pmkit.utils.logger import get_logger

# Import the new interactive prompt flow
from pmkit.agents.interactive_prompt import (
    InteractivePromptFlow,
    WizardStep,
    create_quick_setup_wizard,
)

# Import the new manual input form for review/edit pattern
from pmkit.agents.manual_input import ManualInputForm

# Import the new enrichment service for agentic search
from pmkit.agents.enrichment import EnrichmentService, EnrichmentResult

# Import completion metrics for value demonstration
from pmkit.agents.completion import (
    CompletionMetrics,
    calculate_completion_metrics,
    suggest_next_actions,
    time_saved_message,
)

logger = get_logger(__name__)


class OnboardingAgent:
    """
    Manages the complete onboarding flow for PM-Kit.

    Features:
    - Progressive disclosure in 3 phases
    - Auto-enrichment with web search
    - State persistence for resume capability
    - Beautiful Rich UI with progress indicators
    - Graceful error handling
    """

    def __init__(
        self,
        config: Optional[LLMProviderConfig] = None,
        grounding: Optional[GroundingAdapter] = None,
        console: Optional[Console] = None,
        context_manager: Optional[ContextManager] = None,
        context_dir: Optional[Path] = None,
        use_interactive: bool = False,  # Default to non-interactive for compatibility
    ):
        """
        Initialize the OnboardingAgent.

        Args:
            config: LLM provider configuration
            grounding: GroundingAdapter for web search
            console: Rich console instance
            context_manager: Context manager for persistence
            context_dir: Directory for context files (for testing)
            use_interactive: Whether to use the new interactive prompt flow
        """
        self.config = config
        self.grounding = grounding
        self.console = console or pmkit_console.console

        # Use context_dir if provided (for testing), otherwise use default
        if context_dir:
            self.context_dir = Path(context_dir)
            self.state_file = Path(context_dir) / "onboarding_state.yaml"
        else:
            self.context_dir = Path.home() / ".pmkit" / "context"
            self.state_file = Path.home() / ".pmkit" / "onboarding_state.yaml"

        # Initialize context manager with the context directory
        self.context_manager = context_manager or ContextManager(self.context_dir)

        self.prompts = OnboardingPrompts
        self.prompt_library = PromptLibrary

        # Initialize interactive prompt flow if enabled (can be toggled via flag)
        self.use_interactive = use_interactive
        if use_interactive:
            self.interactive_flow = InteractivePromptFlow(console=self.console)

        # Initialize manual input form for review/edit pattern
        self.manual_form = ManualInputForm(console=self.console)

        # State management
        self.state: Dict[str, Any] = {}
        self.cancelled = False

        # Time tracking for metrics
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.enrichment_used = False
        self.searches_performed = 0

        # Setup signal handlers for graceful cancellation
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful cancellation."""
        def handle_interrupt(signum, frame):
            self.cancelled = True
            self.console.print("\n[yellow]⚠️ Onboarding cancelled. Progress saved.[/yellow]")
            self._save_state()
            raise KeyboardInterrupt()

        signal.signal(signal.SIGINT, handle_interrupt)

    async def run(self, force: bool = False) -> Context:
        """
        Run the complete onboarding flow.

        Args:
            force: Whether to overwrite existing context

        Returns:
            Completed Context object

        Raises:
            PMKitError: If onboarding fails
        """
        try:
            # Start time tracking
            self.start_time = time.time()

            # Check for existing context
            if not force and self.context_manager.exists():
                self.console.print(self.prompts.ERROR_MESSAGES['context_exists'])
                if not Confirm.ask("Do you want to overwrite the existing context?"):
                    raise PMKitError("Onboarding cancelled")

            # Check for saved state (both from OnboardingAgent and ManualInputForm)
            onboarding_state_exists = self._load_state()
            manual_form_state = self.manual_form.load_progress()

            if onboarding_state_exists or manual_form_state:
                if Confirm.ask("Found saved progress. Resume from where you left off?"):
                    self.console.print("[green]Resuming onboarding...[/green]")
                    # Merge saved states (OnboardingAgent state takes precedence)
                    if manual_form_state and not onboarding_state_exists:
                        self.state.update(manual_form_state)
                else:
                    self.state = {}
                    self.manual_form.clear_progress()

            # Show welcome message
            self._show_welcome()

            # Phase 1: Essentials (always required)
            if not self.state.get('phase1_complete'):
                await self._phase1_essentials()
                self.state['phase1_complete'] = True
                self._save_state()

            # Ask if they want to continue
            self.console.print(self.prompts.ESSENTIALS_COMPLETE)
            if not Confirm.ask("Add more context?", default=True):
                return await self._finalize_context()

            # Phase 2: Enrichment (optional but recommended)
            if not self.state.get('phase2_complete'):
                await self._phase2_enrichment()
                self.state['phase2_complete'] = True
                self._save_state()

            # Phase 3: Advanced (optional) - Following 90-second rule
            if not self.state.get('phase3_complete'):
                # Show value first, then ask about advanced setup
                self.console.print("\n[green]✨ Initial context ready![/]")
                self.console.print("[dim]You can generate your first PRD in 30 seconds![/]\n")

                if Confirm.ask("[yellow]Add OKRs and team info?[/] (optional, can be done later)",
                              console=self.console, default=False):
                    await self._phase3_advanced()
                    self.state['phase3_complete'] = True
                    self._save_state()
                else:
                    self.console.print("\n[dim]💡 Tip: Add OKRs anytime with [bold]pm okrs add[/][/]")
                    self.state['phase3_complete'] = True  # Mark as complete even if skipped

            # Finalize and save context
            context = await self._finalize_context()

            # Clean up state files (both OnboardingAgent and ManualInputForm)
            self._cleanup_state()
            self.manual_form.clear_progress()

            # Show completion message
            self._show_completion(context)

            return context

        except KeyboardInterrupt:
            if self.cancelled:
                # Already handled by signal handler
                raise PMKitError("Onboarding cancelled")
            raise
        except Exception as e:
            logger.error(f"Onboarding failed: {e}")
            self._save_state()
            raise PMKitError(f"Onboarding failed: {str(e)}")

    def _show_welcome(self) -> None:
        """Display the welcome message."""
        panel = Panel(
            self.prompts.WELCOME_MESSAGE,
            title="🚀 PM-Kit Setup",
            title_align="left",
            border_style="cyan",
            padding=(1, 2),
        )
        self.console.print(panel)

    async def _phase1_essentials(self) -> None:
        """
        Phase 1: Collect essential information (90-second quick start).

        Following the 90-second rule, we only ask 2 questions:
        1. Company name
        2. Product name

        Everything else is enriched or uses smart defaults.
        """
        self.console.print("\n[bold cyan]Let's set up your PM workspace in 90 seconds![/]")
        self.console.print("[dim]We'll only ask 2 questions, then enrich the rest automatically.[/]\n")

        if self.use_interactive:
            # Use the new interactive prompt flow for a delightful experience
            # Create wizard steps for Phase 1 (only 2 questions for 90-second flow)
            wizard_data = self.interactive_flow.multi_step_wizard(
                steps=create_quick_setup_wizard('b2b')[:2],  # Only company and product name
                allow_skip=False,  # These 2 are required
                show_progress=True,
            )

            # Update state with collected data
            self.state.update(wizard_data)

            # Auto-detect company type based on product name patterns
            product_name = self.state.get('product_name', '').lower()
            if any(kw in product_name for kw in ['api', 'sdk', 'developer', 'platform']):
                self.state['company_type'] = 'b2b'
            elif any(kw in product_name for kw in ['app', 'game', 'social', 'fitness']):
                self.state['company_type'] = 'b2c'
            else:
                # Default to b2b as it's most common for PM tools
                self.state['company_type'] = 'b2b'

            # Set smart defaults
            if not self.state.get('product_description'):
                self.state['product_description'] = f"{self.state['product_name']} - To be enriched"
            if not self.state.get('user_role'):
                self.state['user_role'] = 'Product Manager'

        else:
            # Fallback to original Rich-based prompts (but only 2 questions)
            # Company name
            if not self.state.get('company_name'):
                company_name = Prompt.ask(
                    "[bold]Company name[/]",
                    console=self.console
                )
                self.state['company_name'] = company_name

            # Product name
            if not self.state.get('product_name'):
                product_name = Prompt.ask(
                    "[bold]Product name[/]",
                    console=self.console
                )
                self.state['product_name'] = product_name

            # Auto-detect company type
            product_name_lower = self.state.get('product_name', '').lower()
            if any(kw in product_name_lower for kw in ['api', 'sdk', 'developer', 'platform']):
                self.state['company_type'] = 'b2b'
            elif any(kw in product_name_lower for kw in ['app', 'game', 'social', 'fitness']):
                self.state['company_type'] = 'b2c'
            else:
                self.state['company_type'] = 'b2b'

            # Set smart defaults for other fields
            if not self.state.get('product_description'):
                self.state['product_description'] = f"{self.state['product_name']} - To be enriched"
            if not self.state.get('user_role'):
                self.state['user_role'] = 'Product Manager'  # Most common

    async def _phase2_enrichment(self) -> None:
        """
        Phase 2: Auto-enrich with web search using agentic behavior.

        This phase uses the EnrichmentService with the 3-2 Rule:
        - Primary search + up to 2 adaptive searches
        - Stops at 70% coverage to save searches
        - Shows real-time progress with confidence indicators
        """
        self.console.print("\n" + self.prompts.PHASE_INTRO[2])

        # Handle enrichment based on grounding availability
        if not self.grounding:
            self.console.print(self.prompts.ENRICHMENT_NOT_FOUND)
            await self._manual_enrichment()
        else:
            # Prepare company info for enrichment
            company_info = {
                'name': self.state.get('company_name', ''),
                'domain': self.state.get('website', ''),
                'description': self.state.get('product_description', ''),
                'type': self.state.get('company_type', 'b2b'),
                'product_name': self.state.get('product_name', ''),
            }

            # Initialize enrichment service
            enricher = EnrichmentService(self.grounding)
            self.enrichment_used = True

            # Progress tracking with Rich
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                "[progress.percentage]{task.percentage:>3.0f}%",
                console=self.console
            ) as progress:
                task = progress.add_task(
                    "🔍 Starting enrichment...",
                    total=100
                )

                # Progress callback for real-time updates
                async def update_progress(message: str, percent: int):
                    progress.update(task, description=message, completed=percent)

                try:
                    # Run enrichment with agentic behavior
                    result: EnrichmentResult = await enricher.enrich_company(
                        company_info,
                        progress_callback=update_progress
                    )

                    # Show enrichment summary
                    self._show_enrichment_summary(result)

                    # Convert enriched data to state format
                    enriched_state = self._convert_enrichment_to_state(result)

                    if result.coverage >= 0.4:  # Minimum 40% coverage to proceed
                        # Merge with existing state
                        review_data = {**self.state, **enriched_state}

                        # Review and edit with confidence indicators
                        self.console.print(
                            f"\n[cyan]Found data with {result.coverage*100:.0f}% coverage. "
                            f"Let's review what we found:[/cyan]"
                        )

                        updated_data = self.manual_form.review_and_edit(
                            review_data,
                            company_type=self.state.get('company_type', 'b2b'),
                            required_only=False
                        )

                        # Update state immediately with enriched data
                        self.state.update(updated_data)

                        # Store remaining searches for potential later use
                        self.state['_remaining_searches'] = result.remaining_searches

                    else:
                        # Coverage too low, fall back to manual
                        self.console.print(
                            f"[yellow]Coverage only {result.coverage*100:.0f}%. "
                            f"Let's gather more information manually.[/yellow]"
                        )
                        await self._manual_enrichment(prefilled=enriched_state)

                except Exception as e:
                    logger.warning(f"Enrichment failed: {e}")
                    progress.stop()
                    self.console.print(self.prompts.ENRICHMENT_NOT_FOUND)
                    await self._manual_enrichment()

        # Only ask for manual input if not already enriched
        if not self.state.get('competitors'):
            await self._collect_competitors()

        # Suggest metric only if not already set
        if not self.state.get('north_star_metric'):
            await self._suggest_metric()

    async def _phase3_advanced(self) -> None:
        """
        Phase 3: Advanced configuration (OPTIONAL).

        This phase is completely optional and can be done later.
        Includes: team structure, OKRs, market differentiators.
        """
        self.console.print("\n[bold cyan]Advanced Configuration[/] (all optional)")
        self.console.print("[dim]Skip any section - you can add them later.[/]\n")

        # Team information (optional)
        if Confirm.ask("Set up team structure?", console=self.console, default=False):
            await self._collect_team_info()

        # OKRs (optional)
        if Confirm.ask("Define OKRs now?", console=self.console, default=False):
            await self._collect_okrs()
        else:
            self.console.print("[dim]→ Use [bold]pm okrs add[/] to add OKRs later[/]")

        # Market differentiator (optional)
        if Confirm.ask("Add market differentiators?", console=self.console, default=False):
            await self._collect_differentiator()

    async def _manual_enrichment(self, prefilled: Optional[Dict] = None) -> None:
        """
        Manually collect enrichment data using the new review/edit pattern.

        Args:
            prefilled: Pre-filled data from partial enrichment
        """
        # Use the new ManualInputForm for better UX
        company_type = self.state.get('company_type', 'b2b')

        if prefilled:
            # We have partial data - use review and edit mode
            self.console.print("\n[cyan]Let's review what we found and fill in the gaps:[/cyan]")

            # Merge current state with prefilled data
            review_data = {**self.state, **prefilled}

            # Review and edit with visual indicators
            updated_data = self.manual_form.review_and_edit(
                review_data,
                company_type=company_type,
                required_only=False  # Show all fields for phase 2
            )

            # Update state with reviewed data
            self.state.update(updated_data)
        else:
            # No enrichment data - collect missing required fields
            self.console.print("\n[yellow]Let's gather some additional information:[/yellow]")

            # Collect only missing fields for phase 2
            updated_data = self.manual_form.collect_missing_fields(
                self.state,
                company_type=company_type,
                phase=2  # Phase 2: Enrichment
            )

            # Update state
            self.state.update(updated_data)

    async def _collect_competitors(self) -> None:
        """Collect competitor information."""
        if self.state.get('competitors'):
            # Show found competitors
            found = ", ".join(self.state['competitors'][:3])
            self.console.print(
                self.prompts.COMPETITOR_HELP.format(found_competitors=found)
            )

        competitors_input = Prompt.ask(
            self.prompts.COMPETITOR_PROMPT,
            default=", ".join(self.state.get('competitors', [])),
            console=self.console
        )

        if competitors_input:
            competitors = [c.strip() for c in competitors_input.split(',') if c.strip()]
            self.state['competitors'] = competitors[:5]  # Limit to 5

    async def _suggest_metric(self) -> None:
        """Suggest and collect north star metric."""
        company_type = self.state.get('company_type', 'b2b')

        if company_type == 'b2b':
            self.console.print(self.prompts.METRIC_DEFAULT_B2B)
            default = "MRR"
        else:
            self.console.print(self.prompts.METRIC_DEFAULT_B2C)
            default = "MAU"

        metric = Prompt.ask(
            self.prompts.METRIC_PROMPT,
            default=self.state.get('north_star_metric', default),
            console=self.console
        )
        self.state['north_star_metric'] = metric

    async def _collect_team_info(self) -> None:
        """Collect team information."""
        # Team size
        team_size = Prompt.ask(
            self.prompts.TEAM_SIZE_PROMPT,
            default="",
            console=self.console
        )

        if team_size and team_size.isdigit():
            self.state['team_size'] = int(team_size)

            # Team composition
            composition = Prompt.ask(
                self.prompts.TEAM_COMPOSITION_PROMPT,
                default="",
                console=self.console
            )

            if composition:
                # Parse composition (e.g., "5 engineers, 2 designers")
                roles = {}
                for part in composition.split(','):
                    part = part.strip()
                    if ' ' in part:
                        count, role = part.split(' ', 1)
                        if count.isdigit():
                            roles[role.lower()] = int(count)

                if roles:
                    self.state['team_roles'] = roles

    async def _collect_okrs(self) -> None:
        """Collect OKR information."""
        # Use the new delightful OKR wizard in interactive mode
        if self.use_interactive:
            from pmkit.agents.okr_wizard import OKRWizard

            # Determine company type and stage from state
            company_type = 'B2B' if self.state.get('business_model', '').upper() == 'B2B' else 'B2C'

            # Map funding to stage
            funding = self.state.get('funding_stage', 'Series A').lower()
            if 'seed' in funding or 'pre' in funding:
                company_stage = 'seed'
            elif 'series c' in funding or 'series d' in funding or 'ipo' in funding:
                company_stage = 'scale'
            else:
                company_stage = 'growth'

            # Create and run OKR wizard
            okr_wizard = OKRWizard(
                console=self.console,
                state_file=self.state_file.parent / 'okr_wizard_state.yaml',
                company_type=company_type,
                company_stage=company_stage
            )

            try:
                okr_context = await okr_wizard.run()

                # Convert to state format
                objectives = []
                for obj in okr_context.objectives:
                    objective = {
                        'title': obj.title,
                        'key_results': []
                    }
                    for kr in obj.key_results:
                        objective['key_results'].append({
                            'description': kr.description,
                            'target_value': kr.target_value,
                            'current_value': kr.current_value,
                            'confidence': kr.confidence
                        })
                    objectives.append(objective)

                self.state['okrs'] = objectives
                self.state['okr_quarter'] = okr_context.quarter
                self._save_state()
                return
            except Exception as e:
                logger.warning(f"OKR wizard failed: {e}, falling back to simple flow")
                # Fall back to simple flow

        # Original simple flow for non-interactive mode
        self.console.print("\n" + self.prompts.OKR_INTRO)
        self.console.print(self.prompts.OKR_SKIP)

        # Quick entry option
        quick_goal = Prompt.ask(
            self.prompts.OKR_QUICK,
            default="",
            console=self.console
        )

        if not quick_goal:
            return  # Skip OKRs

        objectives = []

        # Convert quick goal to objective
        objective = {
            'title': quick_goal,
            'key_results': []
        }

        # Ask for key results
        kr_count_raw = Prompt.ask(
            self.prompts.KEY_RESULT_COUNT,
            choices=["1", "2", "3", "4", "5"],
            default="3",
            console=self.console
        )

        try:
            kr_count = int(kr_count_raw)
        except (TypeError, ValueError):
            kr_count = 3
        if kr_count < 1 or kr_count > 5:
            kr_count = 3

        for i in range(kr_count):
            kr_desc = Prompt.ask(
                self.prompts.KEY_RESULT_PROMPT.format(number=i+1),
                console=self.console
            )

            if kr_desc:
                confidence = Prompt.ask(
                    self.prompts.CONFIDENCE_PROMPT,
                    default="70",
                    console=self.console
                )

                objective['key_results'].append({
                    'description': kr_desc,
                    'confidence': int(confidence) if confidence.isdigit() else 70
                })

        objectives.append(objective)

        # Ask for more objectives
        if Confirm.ask(self.prompts.OKR_ADD_MORE, default=False):
            # Simplified: Just one more for MVP
            another_obj = Prompt.ask(
                self.prompts.OBJECTIVE_PROMPT,
                console=self.console
            )
            if another_obj:
                objectives.append({'title': another_obj, 'key_results': []})

        self.state['objectives'] = objectives

        # Auto-detect quarter
        from datetime import datetime
        now = datetime.now()
        quarter = f"Q{(now.month - 1) // 3 + 1} {now.year}"
        self.state['okr_quarter'] = quarter

    async def _collect_differentiator(self) -> None:
        """Collect market differentiator."""
        self.console.print(self.prompts.MARKET_DIFFERENTIATOR_HELP)

        differentiator = Prompt.ask(
            self.prompts.MARKET_DIFFERENTIATOR_PROMPT,
            default="",
            console=self.console
        )

        if differentiator:
            self.state['differentiator'] = differentiator

    async def _finalize_context(self) -> Context:
        """
        Finalize and save the context.

        Returns:
            Completed Context object
        """
        # Build context objects
        company = CompanyContext(
            name=self.state['company_name'],
            type=self.state['company_type'],
            stage=self.state.get('company_stage', 'seed'),
            domain=self.state.get('company_domain'),
            description=self.state.get('company_description'),
            target_market=self.state.get('target_market'),
        )

        product = ProductContext(
            name=self.state['product_name'],
            description=self.state['product_description'],
            stage=self.state.get('product_stage', 'mvp'),
            users=self.state.get('product_users'),
            pricing_model=self.state.get('pricing_model'),
            main_metric=self.state.get('north_star_metric'),
        )

        market = None
        if self.state.get('competitors') or self.state.get('differentiator'):
            market = MarketContext(
                competitors=self.state.get('competitors', []),
                market_size=self.state.get('market_size'),
                differentiator=self.state.get('differentiator'),
            )

        team = None
        if self.state.get('team_size') or self.state.get('team_roles'):
            team = TeamContext(
                size=self.state.get('team_size'),
                roles=self.state.get('team_roles', {}),
            )

        okrs = None
        if self.state.get('objectives'):
            objectives = []
            for obj_data in self.state['objectives']:
                key_results = [
                    KeyResult(
                        description=kr['description'],
                        confidence=kr.get('confidence', 70)
                    )
                    for kr in obj_data.get('key_results', [])
                ]
                objectives.append(
                    Objective(
                        title=obj_data['title'],
                        key_results=key_results
                    )
                )

            okrs = OKRContext(
                objectives=objectives,
                quarter=self.state.get('okr_quarter'),
            )

        # Create complete context
        context = Context(
            company=company,
            product=product,
            market=market,
            team=team,
            okrs=okrs,
        )

        # Save context (support both legacy and new API)
        # Prefer legacy 'save' if provided by injected mock/manager for compatibility with tests
        save_fn = getattr(self.context_manager, "save", None)
        if callable(save_fn):
            save_fn(context)
        else:
            # Fall back to new API that returns (success, errors)
            self.context_manager.save_context(context)

        return context

    def _parse_enrichment(self, content: str) -> Dict[str, Any]:
        """
        Parse enrichment results from search.

        Args:
            content: Search result content

        Returns:
            Parsed enrichment data
        """
        # Simple parsing - in production, use LLM to extract structured data
        enriched = {}

        content_lower = content.lower()

        # Try to detect company stage
        if any(word in content_lower for word in ['startup', 'seed', 'series a']):
            enriched['company_stage'] = 'seed'
        elif any(word in content_lower for word in ['growth', 'series b', 'series c']):
            enriched['company_stage'] = 'growth'
        elif any(word in content_lower for word in ['public', 'fortune', 'enterprise']):
            enriched['company_stage'] = 'mature'

        # Try to extract competitors (simplified)
        # In production, use proper NER or LLM extraction

        return enriched

    def _show_enrichment_summary(self, result: EnrichmentResult) -> None:
        """
        Display a beautiful summary of enrichment results.

        Args:
            result: EnrichmentResult from the EnrichmentService
        """
        # Build coverage bar
        coverage_bar = '█' * int(result.coverage * 10) + '░' * (10 - int(result.coverage * 10))

        # Format found data with confidence indicators
        found_items = []
        for field, data in result.data.items():
            if isinstance(data, dict) and 'value' in data:
                confidence = data.get('confidence', 'LOW')
                emoji = {'HIGH': '🟢', 'MEDIUM': '🟡', 'LOW': '🔴'}.get(confidence, '⚪')
                value = data['value']
                if isinstance(value, list):
                    value = ', '.join(str(v) for v in value[:3])  # Show first 3
                found_items.append(f"  {emoji} {field}: {value}")
            elif data:  # Simple value
                found_items.append(f"  ✓ {field}: {data}")

        found_text = '\n'.join(found_items) if found_items else "  No data found"

        # Create summary panel
        summary = Panel(
            f"""[bold green]✅ Enrichment Complete[/bold green]

Coverage: {coverage_bar} {result.coverage*100:.0f}%
Searches Used: {result.searches_used} of {EnrichmentService.MAX_SEARCHES}

[bold]Found Data:[/bold]
{found_text}

[dim]Remaining searches: {result.remaining_searches} (saved for later)[/dim]""",
            title="🎯 Company Context Enriched",
            title_align="left",
            border_style="green",
            padding=(1, 2),
        )
        self.console.print(summary)

    def _convert_enrichment_to_state(self, result: EnrichmentResult) -> Dict[str, Any]:
        """
        Convert EnrichmentResult to state format for the onboarding flow.
        Simple direct mapping since GPT-5 returns plain values.

        Args:
            result: EnrichmentResult from the EnrichmentService

        Returns:
            Dictionary suitable for state update
        """
        state_data = {}

        # Direct field mapping - GPT-5 returns plain values, not dicts
        field_mapping = {
            # Phase 1
            'product_description': 'product_description',
            # Phase 2
            'company_stage': 'company_stage',
            'target_market': 'target_market',
            'competitors': 'competitors',
            'north_star_metric': 'north_star_metric',
            # Phase 3
            'website': 'website',
            'team_size': 'team_size',
            'pricing_model': 'pricing_model',
            'user_count': 'user_count',
            'key_differentiator': 'key_differentiator',
        }

        # Simple direct assignment - no dict checking needed
        for enriched_field, state_field in field_mapping.items():
            if enriched_field in result.data:
                state_data[state_field] = result.data[enriched_field]

        return state_data

    def _display_enriched_data(self, data: Dict[str, Any]) -> None:
        """
        Display enriched data for confirmation.

        Args:
            data: Enriched data to display
        """
        if not data:
            return

        self.console.print("\n[bold]Found information:[/bold]")
        for key, value in data.items():
            display_key = key.replace('_', ' ').title()
            if isinstance(value, list):
                value = ', '.join(value)
            self.console.print(f"  • {display_key}: {value}")

    def _show_completion(self, context: Context) -> None:
        """
        Enhanced completion experience with value demonstration and celebration.

        Args:
            context: Completed context object
        """
        # Track end time
        self.end_time = time.time()
        time_spent = self.end_time - self.start_time if self.start_time else 60.0

        # Validate context with auto-repair
        validator = ContextValidator()
        is_valid, validation_errors = validator.validate(context, auto_repair=True)

        # Calculate completion metrics
        metrics = calculate_completion_metrics(
            context=context,
            time_spent=time_spent,
            enrichment_used=self.enrichment_used,
            searches_performed=self.searches_performed
        )

        # Create initialization marker
        self._create_initialization_marker(metrics)

        # Generate shareable context card
        card_path = self._generate_context_card(context)

        # Display multi-panel completion experience
        self._display_value_panel(metrics)
        self._display_context_highlights(context)
        self._display_next_actions(context, metrics)

        # Show shareable card notification
        if card_path:
            self.console.print(f"\n📄 Shareable context card saved to: [cyan]{card_path}[/cyan]")

        # Show final activation prompt
        self._show_activation_prompt(context)

    def _create_initialization_marker(self, metrics: CompletionMetrics) -> None:
        """
        Create .pmkit/.initialized with metadata.

        Args:
            metrics: Completion metrics to include in marker
        """
        try:
            marker_data = {
                'initialized_at': datetime.now().isoformat(),
                'version': '0.3.0',
                'setup_time_seconds': metrics.time_spent,
                'completeness_score': metrics.completeness_score,
                'data_points_collected': metrics.data_points_collected,
                'enrichment_coverage': metrics.enrichment_coverage,
                'okr_confidence': metrics.okr_confidence,
            }

            marker_path = self.context_dir.parent / '.initialized'
            marker_path.write_text(yaml.dump(marker_data, default_flow_style=False))
            logger.debug(f"Created initialization marker at {marker_path}")
        except Exception as e:
            logger.warning(f"Could not create initialization marker: {e}")

    def _display_value_panel(self, metrics: CompletionMetrics) -> None:
        """
        Show time savings and value created.

        Args:
            metrics: Calculated completion metrics
        """
        value_text = f"""⏱️  Setup Time: [cyan]{metrics.format_time_spent()}[/cyan]
📊 Data Points: [cyan]{metrics.data_points_collected}[/cyan]
📈 Coverage: {metrics.format_coverage_bar()}

[bold green]Time Savings Unlocked:[/bold green]
• Per PRD: ~45 minutes → 30 seconds
• Weekly: ~3 hours saved
• Annually: [bold]{metrics.annual_hours_saved} hours[/bold] (~{metrics.work_weeks_saved:.0f} work weeks)

{time_saved_message(metrics)}"""

        panel = Panel(
            value_text,
            title="✨ Value Created",
            border_style="green",
            padding=(1, 2)
        )
        self.console.print("\n")
        self.console.print(panel)

    def _display_context_highlights(self, context: Context) -> None:
        """
        Display key context highlights in a beautiful panel.

        Args:
            context: The completed context
        """
        # Build highlights based on B2B vs B2C
        if context.company.type == 'b2b':
            focus_areas = "ROI • Integrations • Enterprise Features • Compliance"
            templates = "Technical PRDs • API Docs • Security Reviews"
        else:
            focus_areas = "Engagement • Retention • Viral Loops • Mobile UX"
            templates = "Growth PRDs • User Stories • A/B Tests"

        highlights = f"""[bold cyan]{context.company.name}[/bold cyan] - {context.product.name}
{context.product.description[:100]}...

[bold]Business Model:[/bold] {context.company.type.upper()}
[bold]Company Stage:[/bold] {context.company.stage.title()}
[bold]North Star:[/bold] {context.product.main_metric or 'TBD'}

[bold]Focus Areas:[/bold] {focus_areas}
[bold]Templates:[/bold] {templates}"""

        if context.okrs and context.okrs.objectives:
            okr_count = len(context.okrs.objectives)
            kr_count = sum(len(obj.key_results) for obj in context.okrs.objectives)
            avg_confidence = sum(
                kr.confidence or 0
                for obj in context.okrs.objectives
                for kr in obj.key_results
            ) / max(kr_count, 1)

            confidence_emoji = "🟢" if avg_confidence >= 70 else "🟡" if avg_confidence >= 50 else "🔴"
            okr_line = f"\n[bold]OKRs:[/bold] {okr_count} objectives, {kr_count} key results {confidence_emoji} {avg_confidence:.0f}% confidence"
            highlights += okr_line

        panel = Panel(
            highlights,
            title=f"🎯 {context.company.name} Context Activated",
            border_style="cyan",
            padding=(1, 2)
        )
        self.console.print(panel)

    def _display_next_actions(self, context: Context, metrics: CompletionMetrics) -> None:
        """
        Suggest personalized next steps based on context.

        Args:
            context: The completed context
            metrics: Calculated completion metrics
        """
        actions = suggest_next_actions(context, metrics)

        # Format actions with descriptions
        action_lines = []
        for i, (cmd, desc) in enumerate(actions, 1):
            action_lines.append(f"{i}. [cyan]{cmd}[/cyan]")
            action_lines.append(f"   [dim]{desc}[/dim]")

        action_text = "\n".join(action_lines)

        panel = Panel(
            action_text,
            title="🚀 Recommended Next Steps",
            border_style="yellow",
            padding=(1, 2)
        )
        self.console.print(panel)

    def _generate_context_card(self, context: Context) -> Optional[Path]:
        """
        Generate markdown summary for team sharing.

        Args:
            context: The completed context

        Returns:
            Path to the generated context card, or None if failed
        """
        try:
            # Format OKRs for markdown
            okr_section = ""
            if context.okrs and context.okrs.objectives:
                okr_lines = []
                for obj in context.okrs.objectives:
                    okr_lines.append(f"\n### {obj.title}")
                    for kr in obj.key_results:
                        confidence_emoji = "🟢" if kr.confidence >= 70 else "🟡" if kr.confidence >= 50 else "🔴"
                        okr_lines.append(f"- {kr.description} ({confidence_emoji} {kr.confidence}% confidence)")
                okr_section = "\n".join(okr_lines)
            else:
                okr_section = "_No OKRs defined yet_"

            # Format competitors
            competitor_list = "Not specified"
            if context.market and context.market.competitors:
                competitor_list = ", ".join(context.market.competitors[:3])
                if len(context.market.competitors) > 3:
                    competitor_list += f" (+{len(context.market.competitors)-3} more)"

            card_content = f"""# {context.company.name} Product Context
Generated by PM-Kit on {datetime.now().strftime('%Y-%m-%d')}

## Quick Facts
- **Product**: {context.product.name}
- **Model**: {context.company.type.upper()}
- **Stage**: {context.company.stage.title()}
- **North Star**: {context.product.main_metric or 'TBD'}

## Product Description
{context.product.description}

## Current OKRs ({context.okrs.quarter if context.okrs else 'Not Set'})
{okr_section}

## Competitive Landscape
- **Primary Competitors**: {competitor_list}
- **Key Differentiator**: {context.market.differentiator if context.market and context.market.differentiator else 'TBD'}

## Team
- **Size**: {context.team.size if context.team else 'Not specified'} people
- **Roles**: {', '.join(f"{role}: {count}" for role, count in context.team.roles.items()) if context.team and context.team.roles else 'Not specified'}

---
*Share this with your team to align on product context*
*Update context anytime: `pm context edit`*
"""

            # Save to exports directory
            export_dir = self.context_dir.parent / 'exports'
            export_dir.mkdir(exist_ok=True)
            card_path = export_dir / 'context-card.md'
            card_path.write_text(card_content)

            logger.debug(f"Generated context card at {card_path}")
            return card_path

        except Exception as e:
            logger.error(f"Failed to generate context card: {e}")
            return None

    def _show_activation_prompt(self, context: Context) -> None:
        """
        Show final call-to-action to drive immediate value.

        Args:
            context: The completed context
        """
        # Get the top recommended action
        actions = suggest_next_actions(context, CompletionMetrics(
            time_spent=60,
            data_points_collected=20,
            enrichment_coverage=0.7,
            okr_confidence=70,
            completeness_score=0.8
        ))

        if actions:
            first_cmd = actions[0][0]
            # Extract just the command and title for cleaner display
            if 'pm new prd' in first_cmd:
                title = first_cmd.split('"')[1] if '"' in first_cmd else "Your First PRD"
                prompt = f"\n[bold]Ready to ship faster?[/bold] Try: [cyan]pm new prd \"{title}\"[/cyan]\n"
            else:
                prompt = f"\n[bold]Ready to continue?[/bold] Try: [cyan]{first_cmd}[/cyan]\n"

            self.console.print(prompt)

    def _save_state(self) -> None:
        """Save current state to disk."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                yaml.dump(self.state, f)
            logger.debug(f"State saved to {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _load_state(self) -> bool:
        """
        Load saved state from disk.

        Returns:
            True if state was loaded, False otherwise
        """
        if not self.state_file.exists():
            return False

        try:
            with open(self.state_file, 'r') as f:
                self.state = yaml.safe_load(f) or {}
            logger.debug(f"State loaded from {self.state_file}")
            return bool(self.state)
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False

    def _cleanup_state(self) -> None:
        """Remove the state file after successful completion."""
        try:
            if self.state_file.exists():
                self.state_file.unlink()
                logger.debug("State file cleaned up")
        except Exception as e:
            logger.error(f"Failed to cleanup state: {e}")


# Sync wrapper for CLI integration
def run_onboarding(
    config: Optional[Any] = None,  # Can be Config or LLMProviderConfig
    context_dir: Optional[Path] = None,
    resume: bool = True,
    skip_enrichment: bool = False,
) -> tuple[bool, Optional[Context]]:
    """
    Synchronous wrapper for running onboarding from CLI.

    Args:
        config: Configuration object (Config or LLMProviderConfig)
        context_dir: Directory for context files
        resume: Whether to resume from saved state
        skip_enrichment: Whether to skip web enrichment

    Returns:
        Tuple of (success, context) where success is True if completed
    """
    try:
        # Extract LLM config if it's a full Config object
        llm_config = None
        if config and hasattr(config, 'llm'):
            llm_config = config.llm
        elif isinstance(config, LLMProviderConfig):
            llm_config = config

        # Create grounding adapter if config provided and not skipping enrichment
        grounding = None
        if llm_config and not skip_enrichment:
            try:
                from pmkit.llm.grounding import GroundingAdapter
                grounding = GroundingAdapter(llm_config)
            except Exception as e:
                logger.warning(f"Could not initialize grounding: {e}")

        # Create context manager with the provided directory
        ctx_manager = None
        if context_dir:
            from pmkit.context.manager import ContextManager
            ctx_manager = ContextManager(context_dir)

        # Create and run agent
        agent = OnboardingAgent(
            config=llm_config,
            grounding=grounding,
            context_manager=ctx_manager,
        )

        # Bridge to async using proper utility that handles nested loops
        from pmkit.utils.async_utils import run_async
        context = run_async(agent.run(force=not resume))
        return (True, context)

    except KeyboardInterrupt:
        logger.info("Onboarding cancelled by user")
        return (False, None)
    except Exception as e:
        logger.error(f"Onboarding failed: {e}")
        return (False, None)
