"""
OnboardingAgent - Handles the complete onboarding flow for PM-Kit.

This module implements a wizard-style onboarding experience with progressive
disclosure, auto-enrichment, and beautiful Rich-based UI.
"""

from __future__ import annotations

import asyncio
import json
import signal
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt

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

            # Phase 3: Advanced (optional)
            if not self.state.get('phase3_complete'):
                if Confirm.ask("Add advanced details (team, OKRs)?", default=False):
                    await self._phase3_advanced()
                    self.state['phase3_complete'] = True
                    self._save_state()

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
        Phase 1: Collect essential information.

        This phase collects the minimum required information to get started.
        """
        self.console.print("\n" + self.prompts.PHASE_INTRO[1])

        if self.use_interactive:
            # Use the new interactive prompt flow for a delightful experience
            # Create wizard steps for Phase 1 (reduced to 4 questions as per PM expert feedback)
            wizard_data = self.interactive_flow.multi_step_wizard(
                steps=create_quick_setup_wizard('b2b')[:4],  # Only first 4 steps for Phase 1
                allow_skip=False,  # Phase 1 is required
                show_progress=True,
            )

            # Update state with collected data
            self.state.update(wizard_data)

            # Normalize company type to lowercase
            if self.state.get('company_type'):
                self.state['company_type'] = self.state['company_type'].lower()
                if self.state['company_type'] == 'b2b2c':
                    self.state['company_type'] = 'b2b2c'
                elif 'b2c' in self.state['company_type']:
                    self.state['company_type'] = 'b2c'
                else:
                    self.state['company_type'] = 'b2b'

        else:
            # Fallback to original Rich-based prompts
            # Company name
            if not self.state.get('company_name'):
                self.console.print(self.prompts.COMPANY_NAME_HELP)
                company_name = Prompt.ask(
                    self.prompts.COMPANY_NAME_PROMPT,
                    console=self.console
                )
                self.state['company_name'] = company_name

            # Company type
            if not self.state.get('company_type'):
                self.console.print(self.prompts.COMPANY_TYPE_HELP)
                for i, choice in enumerate(self.prompts.COMPANY_TYPE_CHOICES, 1):
                    self.console.print(f"  {i}. {choice}")

                choice_idx = Prompt.ask(
                    self.prompts.COMPANY_TYPE_PROMPT,
                    choices=["1", "2", "3"],
                    default="1",
                    console=self.console
                )
                company_type_map = {"1": "b2b", "2": "b2c", "3": "b2b2c"}
                # Handle invalid choices gracefully
                self.state['company_type'] = company_type_map.get(choice_idx, "b2b")

            # Product name
            if not self.state.get('product_name'):
                self.console.print(self.prompts.PRODUCT_NAME_HELP)
                product_name = Prompt.ask(
                    self.prompts.PRODUCT_NAME_PROMPT,
                    console=self.console
                )
                self.state['product_name'] = product_name

            # Product description
            if not self.state.get('product_description'):
                self.console.print(self.prompts.PRODUCT_DESC_HELP)
                product_desc = Prompt.ask(
                    self.prompts.PRODUCT_DESC_PROMPT,
                    console=self.console
                )
                self.state['product_description'] = product_desc

            # User role (kept in Phase 1 for backward compatibility with tests)
            if not self.state.get('user_role'):
                choices = self.prompts.YOUR_ROLE_CHOICES
                role_input = Prompt.ask(
                    self.prompts.YOUR_ROLE_PROMPT,
                    console=self.console
                )
                # Accept either index or text
                if role_input.isdigit():
                    idx = int(role_input) - 1
                    if 0 <= idx < len(choices):
                        self.state['user_role'] = choices[idx]
                    else:
                        self.state['user_role'] = choices[0]
                else:
                    # Try to match text directly, fallback to first option
                    match = next((c for c in choices if c.lower() == role_input.lower()), None)
                    self.state['user_role'] = match or choices[0]

    async def _phase2_enrichment(self) -> None:
        """
        Phase 2: Auto-enrich with web search.

        This phase attempts to gather additional context automatically.
        """
        self.console.print("\n" + self.prompts.PHASE_INTRO[2])

        # Handle enrichment based on grounding availability
        if not self.grounding:
            self.console.print(self.prompts.ENRICHMENT_NOT_FOUND)
            await self._manual_enrichment()
        else:
            # Try to enrich company data
            company_name = self.state.get('company_name')

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task(
                    self.prompts.ENRICHMENT_SEARCHING.format(company_name=company_name),
                    total=None
                )

                try:
                    # Search for company information
                    search_query = self.prompt_library.get_prompt(
                        'COMPANY_ENRICHMENT',
                        company_name=company_name
                    )

                    search_result = await self.grounding.search(
                        search_query,
                        SearchOptions(max_results=5)
                    )

                    if search_result and search_result.content:
                        # Parse enriched data
                        enriched_data = self._parse_enrichment(search_result.content)

                        # Show found data
                        progress.stop()
                        self.console.print(
                            self.prompts.ENRICHMENT_FOUND.format(company_name=company_name)
                        )

                        # Use the new review and edit pattern for partial enrichment
                        # This shows all fields with status indicators (✅ ⚠️ ❌)
                        review_data = {**self.state, **enriched_data}

                        updated_data = self.manual_form.review_and_edit(
                            review_data,
                            company_type=self.state.get('company_type', 'b2b'),
                            required_only=False
                        )

                        self.state.update(updated_data)
                    else:
                        progress.stop()
                        self.console.print(self.prompts.ENRICHMENT_NOT_FOUND)
                        await self._manual_enrichment()

                except Exception as e:
                    logger.warning(f"Enrichment failed: {e}")
                    progress.stop()
                    self.console.print(self.prompts.ENRICHMENT_NOT_FOUND)
                    await self._manual_enrichment()

        # Collect or confirm competitors
        await self._collect_competitors()

        # Suggest north star metric
        await self._suggest_metric()

    async def _phase3_advanced(self) -> None:
        """
        Phase 3: Collect advanced optional details.

        This phase collects team structure and OKRs.
        """
        self.console.print("\n" + self.prompts.PHASE_INTRO[3])

        # Team information
        await self._collect_team_info()

        # OKRs
        await self._collect_okrs()

        # Market differentiator
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
        Show completion message with context summary.

        Args:
            context: Completed context object
        """
        # Format OKR summary
        okr_summary = "[dim]No OKRs set[/dim]"
        if context.okrs and context.okrs.objectives:
            okr_summary = self.prompts.format_okr_summary(
                [obj.model_dump() for obj in context.okrs.objectives]
            )

        # Format summary
        summary = self.prompts.CONTEXT_SUMMARY_TEMPLATE.format(
            company_name=context.company.name,
            company_type=context.company.type,
            company_stage=context.company.stage,
            product_name=context.product.name,
            product_description=context.product.description,
            metric=context.product.main_metric or "Not set",
            competitors=', '.join(context.market.competitors) if context.market else "Not set",
            differentiator=context.market.differentiator if context.market else "Not set",
            team_size=context.team.size if context.team else "Not set",
            team_composition="Not set",  # TODO: Format from roles
            okr_summary=okr_summary,
        )

        # Show summary panel
        panel = Panel(
            summary,
            title="📋 Your Context",
            border_style="green",
            padding=(1, 2),
        )
        self.console.print(panel)

        # Show completion message
        self.console.print(self.prompts.SETUP_COMPLETE)

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

        # Bridge to async
        context = asyncio.run(agent.run(force=not resume))
        return (True, context)

    except KeyboardInterrupt:
        logger.info("Onboarding cancelled by user")
        return (False, None)
    except Exception as e:
        logger.error(f"Onboarding failed: {e}")
        return (False, None)
