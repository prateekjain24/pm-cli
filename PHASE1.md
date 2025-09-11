# PHASE1.md - Context-Aware PM Assistant with Native LLM Search - 11 September 2025

## Executive Summary

Phase 1 builds a context-aware PM assistant system that leverages native web search capabilities from OpenAI, Anthropic, and Gemini. The system establishes persistent context about the PM's company, product, and market through intelligent onboarding, then uses this context for all PM tasks (PRDs, roadmaps, OKRs, personas, release notes).

## Core Architecture: Two-Layer System

### 1. Context Layer (Persistent, Evolving)
- Company profile (enriched via native LLM search)
- Product details and positioning
- Market intelligence and competitors
- Team structure and OKRs
- Historical decisions and learnings

### 2. Task Layer (Stateless Agents)
- **PRDAgent** (Phase 1 implementation)
- RoadmapAgent (future)
- PersonaAgent (future)
- OKRAgent (future)
- ReleaseNotesAgent (future)

## Key Innovation: Native Search Integration

Instead of external dependencies like Perplexity, we leverage native search capabilities:
- **OpenAI**: Web search tools
- **Anthropic**: Native web search
- **Gemini**: Grounding feature
- **Ollama**: Fallback to Perplexity/SerpAPI

```python
class GroundingAdapter:
    """Unified interface for web search across LLM providers"""
    
    def __init__(self, provider: str, api_key: str):
        self.provider = provider
        self.api_key = api_key
        self.cache = SearchCache()
        
    async def search(self, query: str) -> SearchResult:
        # Check cache first
        if cached := self.cache.get(query):
            return cached
            
        # Provider-specific search
        if self.provider == "openai":
            result = await self._openai_search(query)
        elif self.provider == "anthropic":
            result = await self._anthropic_search(query)
        elif self.provider == "gemini":
            result = await self._gemini_grounding(query)
        elif self.provider == "ollama":
            result = await self._fallback_search(query)
        else:
            # No search - graceful degradation
            result = SearchResult(content="", citations=[])
            
        # Cache result
        self.cache.set(query, result, ttl=86400)
        return result
```

## Week 1: Context Foundation & Intelligent Onboarding

### Day 1-2: GroundingAdapter Implementation

#### Provider-Specific Search Methods

```python
async def _openai_search(self, query: str) -> SearchResult:
    """OpenAI web search via tools"""
    client = OpenAI(api_key=self.api_key)
    response = await client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": query}],
        tools=[{"type": "web_search"}],
        tool_choice="auto"
    )
    return self._parse_openai_response(response)

async def _anthropic_search(self, query: str) -> SearchResult:
    """Anthropic native web search"""
    client = Anthropic(api_key=self.api_key)
    response = await client.messages.create(
        model="claude-3-opus-20240229",
        messages=[{"role": "user", "content": query}],
        web_search=True
    )
    return self._parse_anthropic_response(response)

async def _gemini_grounding(self, query: str) -> SearchResult:
    """Gemini grounding feature"""
    genai.configure(api_key=self.api_key)
    model = genai.GenerativeModel('gemini-pro')
    response = await model.generate_content(
        query,
        grounding_config={"enable_grounding": True}
    )
    return self._parse_gemini_response(response)
```

### Day 3: Onboarding Flow (`pm init`)

#### Progressive Disclosure UX

```python
class OnboardingAgent:
    def __init__(self, config: Config):
        self.grounding = GroundingAdapter(
            provider=config.llm.provider,
            api_key=config.keys[f"{config.llm.provider}_api_key"]
        )
        
    async def interactive_onboard(self) -> Context:
        # Step 1: Minimal input
        company = prompt("Company name: ")
        
        # Step 2: Enrich via search
        with console.status(f"Researching {company}..."):
            enriched = await self.enrich_company(company)
        
        # Step 3: Confirm findings
        console.print(Panel(
            f"""
            [bold]{enriched.name}[/bold]
            
            Industry: {enriched.industry}
            Model: {enriched.model} ({enriched.b2b_or_b2c})
            Stage: {enriched.stage}
            Employees: {enriched.employees}
            
            Main Product: {enriched.product}
            Competitors: {', '.join(enriched.competitors[:3])}
            """,
            title="Company Profile Found"
        ))
        
        if not confirm("Is this correct?"):
            enriched = self.manual_correction(enriched)
        
        # Step 4: Collect PM-specific context
        role = prompt("Your role [Senior PM]: ") or "Senior PM"
        team_size = prompt("Team size [5-10]: ") or "5-10"
        
        # Step 5: Current OKRs
        console.print("[bold]Current Quarter OKRs[/bold]")
        okrs = self.collect_okrs()
        
        return Context(
            company=enriched,
            role=role,
            team_size=team_size,
            okrs=okrs
        )
    
    async def enrich_company(self, company_name: str) -> CompanyContext:
        """Smart enrichment with batched search"""
        
        search_query = f"""
        Find comprehensive information about {company_name}:
        
        1. Company Overview:
           - Official website and domain
           - Industry and market segment
           - Business model (B2B/B2C/B2B2C)
           - Company stage (seed/series A/B/C/public)
           - Employee count and headquarters
           - Year founded
        
        2. Product & Services:
           - Main products/services offered
           - Target customers
           - Pricing model
           - Key differentiators
        
        3. Market Position:
           - Main competitors (list top 5)
           - Market share if available
           - Recent funding or acquisitions
           - Key partnerships
        
        4. Recent Developments:
           - Latest news (last 6 months)
           - Product launches
           - Leadership changes
        """
        
        try:
            result = await self.grounding.search(search_query)
            return self.parse_company_data(result)
        except SearchUnavailable:
            console.print("[yellow]Search unavailable. Manual input required.[/yellow]")
            return self.manual_company_input(company_name)
```

### Day 4: Context Schema & Persistence

#### Context File Structure

```yaml
# .pmkit/context/company.yaml
name: Acme Corp
domain: acme.com
industry: CRM Software
model: B2B SaaS
stage: Series B
funding: $45M
employees: 150-200
founded: 2019
headquarters: San Francisco, CA
last_updated: 2025-01-11T10:00:00

# .pmkit/context/product.yaml
name: Acme CRM
type: Enterprise SaaS
pricing_model: Subscription
deployment: Cloud
key_features:
  - Sales automation
  - Lead scoring
  - Email integration
  - Analytics dashboard
target_users:
  primary: Sales teams (10-500 people)
  secondary: Marketing teams
  buyer: VP Sales, CRO
differentiators:
  - AI-powered insights
  - Mobile-first design
  - 500+ integrations

# .pmkit/context/market.yaml
tam: $50B
sam: $5B
som: $500M
growth_rate: 12% CAGR
competitors:
  - name: Salesforce
    strengths: [Market leader, ecosystem, enterprise features]
    weaknesses: [Complex, expensive, slow implementation]
    market_share: 23%
  - name: HubSpot
    strengths: [User friendly, marketing integration, free tier]
    weaknesses: [Limited enterprise features, expensive at scale]
    market_share: 8%
  - name: Pipedrive
    strengths: [Simple, affordable, good UX]
    weaknesses: [Limited features, few integrations]
    market_share: 3%
trends:
  - AI and automation becoming table stakes
  - Shift to product-led growth
  - Consolidation of point solutions
  - Privacy regulations impacting data handling

# .pmkit/context/team.yaml
pm:
  name: John Doe
  role: Senior PM
  email: john@acme.com
  experience_years: 5
  focus_areas: [Core CRM, Integrations, Mobile]
stakeholders:
  engineering:
    lead: Jane Smith
    team_size: 15
    methodology: Agile/Scrum
  design:
    lead: Bob Johnson
    team_size: 3
  marketing:
    lead: Sarah Wilson
  sales:
    lead: Mike Brown
    
# .pmkit/context/okrs.yaml
quarter: Q1 2025
objectives:
  - description: Achieve product-market fit in mid-market segment
    key_results:
      - metric: New mid-market customers
        target: 25
        current: 5
      - metric: Mid-market NPS
        target: 50
        current: 35
      - metric: Mid-market MRR
        target: $250K
        current: $50K
    confidence: 0.7
  - description: Improve platform reliability
    key_results:
      - metric: Uptime
        target: 99.9%
        current: 99.5%
      - metric: P1 incidents
        target: <2/month
        current: 5/month
```

### Day 5: Context Manager with Versioning

```python
class ContextManager:
    def __init__(self, project_root: Path):
        self.context_dir = project_root / ".pmkit" / "context"
        self.context_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir = self.context_dir / "history"
        self.history_dir.mkdir(exist_ok=True)
        
    def save_context(self, context: Context) -> None:
        """Save context with automatic versioning"""
        timestamp = datetime.now().isoformat()
        
        for section in ['company', 'product', 'market', 'team', 'okrs']:
            file_path = self.context_dir / f"{section}.yaml"
            
            # Backup existing if present
            if file_path.exists():
                backup_path = self.history_dir / f"{section}_{timestamp}.yaml"
                shutil.copy(file_path, backup_path)
            
            # Save new version
            data = getattr(context, section).dict()
            data['last_updated'] = timestamp
            
            with open(file_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        
        # Git commit if in repo
        if (self.context_dir.parent / ".git").exists():
            subprocess.run(
                ["git", "add", ".pmkit/context/*.yaml"],
                cwd=self.context_dir.parent,
                capture_output=True
            )
            subprocess.run(
                ["git", "commit", "-m", f"Update context: {timestamp}"],
                cwd=self.context_dir.parent,
                capture_output=True
            )
    
    def load_context(self) -> Optional[Context]:
        """Load context from disk"""
        if not self.context_exists():
            return None
            
        context_data = {}
        for section in ['company', 'product', 'market', 'team', 'okrs']:
            file_path = self.context_dir / f"{section}.yaml"
            if file_path.exists():
                with open(file_path) as f:
                    context_data[section] = yaml.safe_load(f)
        
        return Context(**context_data)
    
    def context_exists(self) -> bool:
        """Check if context has been initialized"""
        return (self.context_dir / "company.yaml").exists()
```

## Week 2: Context-Aware PRD Generation

### Day 6-7: PRD Agent Architecture

```python
class PRDAgent:
    """Context-aware PRD generation with 5 phases"""
    
    def __init__(self, context_manager: ContextManager, config: Config):
        self.context = context_manager.load_context()
        if not self.context:
            raise ValueError("No context found. Run 'pm init' first.")
            
        self.grounding = GroundingAdapter(
            provider=config.llm.provider,
            api_key=config.keys[f"{config.llm.provider}_api_key"]
        )
        
        # Initialize phase agents with context
        self.phases = [
            ProblemPhase(self.context, self.grounding),
            SolutionPhase(self.context, self.grounding),
            RequirementsPhase(self.context, self.grounding),
            PrototypePhase(self.context, self.grounding),
            FinalPRDPhase(self.context, self.grounding)
        ]
        
        self.cache = PRDCache()
    
    async def generate_prd(self, title: str, description: str = "") -> PRD:
        """Generate context-aware PRD"""
        
        console.print(f"\n[bold]Generating PRD: {title}[/bold]\n")
        
        results = {}
        for phase in self.phases:
            phase_name = phase.__class__.__name__.replace("Phase", "")
            
            with console.status(f"Phase: {phase_name}..."):
                # Prepare input with context
                phase_input = PhaseInput(
                    title=title,
                    description=description,
                    context=self.context,
                    prior_phases=results
                )
                
                # Check cache
                cache_key = self.cache.compute_key(phase_name, phase_input)
                if cached := self.cache.get(cache_key):
                    results[phase_name] = cached
                    console.print(f"✓ {phase_name} (cached)")
                else:
                    # Generate new
                    output = await phase.generate(phase_input)
                    results[phase_name] = output
                    self.cache.set(cache_key, output)
                    console.print(f"✓ {phase_name}")
        
        return PRD(
            title=title,
            phases=results,
            context=self.context,
            generated_at=datetime.now()
        )
```

### Day 8: Context-Aware Phase Implementations

```python
class ProblemPhase:
    """Context-aware problem definition"""
    
    def __init__(self, context: Context, grounding: GroundingAdapter):
        self.context = context
        self.grounding = grounding
        
    async def generate(self, input: PhaseInput) -> PhaseOutput:
        # Tailor approach based on B2B vs B2C
        if self.context.product.model == "B2B SaaS":
            template = self.B2B_PROBLEM_TEMPLATE
            search_focus = """
            Focus on:
            - Enterprise pain points and inefficiencies
            - ROI and business impact
            - Integration challenges
            - Compliance and security concerns
            - Decision maker priorities (VP/C-level)
            """
        else:  # B2C
            template = self.B2C_PROBLEM_TEMPLATE
            search_focus = """
            Focus on:
            - User frustrations and friction points
            - Engagement and retention issues
            - Onboarding challenges
            - Mobile experience problems
            - Social and viral growth blockers
            """
        
        # Research if search available
        research = None
        if self.grounding.available:
            search_query = f"""
            Research problem space for: {input.title}
            Company: {self.context.company.name}
            Industry: {self.context.company.industry}
            Current competitors: {', '.join([c.name for c in self.context.market.competitors[:3]])}
            {search_focus}
            """
            
            research = await self.grounding.search(search_query)
        
        # Generate problem statement
        llm_input = {
            "title": input.title,
            "description": input.description,
            "company": self.context.company,
            "product": self.context.product,
            "market": self.context.market,
            "okrs": self.context.okrs,
            "research": research.content if research else None
        }
        
        content = await self.llm.generate(template, llm_input)
        
        return PhaseOutput(
            content=content,
            citations=research.citations if research else [],
            confidence=0.85 if research else 0.70,
            metadata={
                "search_used": bool(research),
                "context_version": self.context.version
            }
        )
    
    B2B_PROBLEM_TEMPLATE = """
    # Problem Statement: {title}
    
    ## Executive Summary
    [Concise problem statement focused on business impact]
    
    ## Current State Analysis
    ### Business Impact
    - Operational inefficiencies
    - Revenue implications
    - Competitive disadvantages
    
    ### Stakeholder Pain Points
    - Economic Buyer ({context.product.target_users.buyer}):
    - End Users ({context.product.target_users.primary}):
    - IT/Admin:
    
    ## Market Context
    - How competitors address this: {research}
    - Industry trends: {context.market.trends}
    
    ## Alignment with OKRs
    {context.okrs}
    
    ## Success Metrics
    - Business metrics (ROI, efficiency gains)
    - User adoption metrics
    - Technical metrics
    """
    
    B2C_PROBLEM_TEMPLATE = """
    # Problem Statement: {title}
    
    ## User Problem
    [Clear articulation of user frustration/need]
    
    ## User Journey Pain Points
    - Discovery:
    - Onboarding:
    - Core experience:
    - Retention:
    
    ## Market Evidence
    {research}
    
    ## Competitive Landscape
    How others solve this: {context.market.competitors}
    
    ## Impact on Key Metrics
    - User acquisition:
    - Activation:
    - Retention:
    - Revenue:
    - Referral:
    
    ## Alignment with OKRs
    {context.okrs}
    """
```

### Day 9: Caching & Performance Optimization

```python
class PRDCache:
    """Multi-level caching for PRD generation"""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path.home() / ".pmkit" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory L1 cache
        self.memory_cache = {}
        
        # Disk-based L2 cache
        self.disk_cache = DiskCache(self.cache_dir)
        
    def compute_key(self, phase: str, input: PhaseInput) -> str:
        """Compute deterministic cache key"""
        # Include context version in cache key
        key_data = {
            "phase": phase,
            "title": input.title,
            "description": input.description,
            "context_version": input.context.version,
            "prior_phases": {
                k: v.content_hash for k, v in input.prior_phases.items()
            }
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[PhaseOutput]:
        """Get from cache (memory first, then disk)"""
        # L1: Memory
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # L2: Disk
        if cached := self.disk_cache.get(key):
            self.memory_cache[key] = cached  # Promote to L1
            return cached
            
        return None
    
    def set(self, key: str, value: PhaseOutput) -> None:
        """Set in both cache levels"""
        self.memory_cache[key] = value
        self.disk_cache.set(key, value, ttl=86400 * 7)  # 7 days
```

### Day 10: CLI Integration & Testing

```python
@app.command()
def init():
    """Initialize PM context with intelligent onboarding"""
    
    # Check if already initialized
    manager = ContextManager(Path.cwd())
    if manager.context_exists():
        if not confirm("Context exists. Reinitialize?"):
            return
    
    # Run onboarding
    config = load_config()
    agent = OnboardingAgent(config)
    
    try:
        context = asyncio.run(agent.interactive_onboard())
        manager.save_context(context)
        
        console.print("\n[green]✓ Context initialized successfully![/green]")
        console.print(f"Company: {context.company.name}")
        console.print(f"Industry: {context.company.industry}")
        console.print(f"Model: {context.product.model}")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def new(
    artifact: str = typer.Argument(..., help="Artifact type: prd, roadmap, persona"),
    title: str = typer.Option(None, "--title", "-t", help="Title for the artifact")
):
    """Create new PM artifact using context"""
    
    # Check context
    manager = ContextManager(Path.cwd())
    if not manager.context_exists():
        console.print("[red]No context found. Run 'pm init' first.[/red]")
        raise typer.Exit(1)
    
    # Get title if not provided
    if not title:
        title = typer.prompt(f"{artifact.upper()} title")
    
    # Load config
    config = load_config()
    
    if artifact == "prd":
        # Generate PRD
        agent = PRDAgent(manager, config)
        
        with console.status(f"Generating PRD: {title}"):
            prd = asyncio.run(agent.generate_prd(title))
        
        # Save PRD
        prd_dir = Path("product/prds") / slugify(title)
        prd_dir.mkdir(parents=True, exist_ok=True)
        
        for phase_name, phase_output in prd.phases.items():
            file_path = prd_dir / f"{phase_name.lower()}.md"
            file_path.write_text(phase_output.content)
        
        # Save manifest
        manifest = {
            "title": title,
            "created": prd.generated_at.isoformat(),
            "context_version": prd.context.version,
            "phases": list(prd.phases.keys())
        }
        
        with open(prd_dir / "manifest.yaml", 'w') as f:
            yaml.dump(manifest, f)
        
        console.print(f"\n[green]✓ PRD created: {prd_dir}[/green]")
        
    else:
        console.print(f"[yellow]{artifact} generation coming soon![/yellow]")

@app.command()
def status():
    """Show current context status"""
    manager = ContextManager(Path.cwd())
    
    if not manager.context_exists():
        console.print("[yellow]No context initialized. Run 'pm init' first.[/yellow]")
        return
    
    context = manager.load_context()
    
    console.print(Panel(
        f"""
        [bold]Company:[/bold] {context.company.name}
        [bold]Industry:[/bold] {context.company.industry}
        [bold]Model:[/bold] {context.product.model}
        [bold]Stage:[/bold] {context.company.stage}
        
        [bold]Current Quarter:[/bold] {context.okrs.quarter}
        [bold]Objectives:[/bold] {len(context.okrs.objectives)}
        
        [bold]Last Updated:[/bold] {context.company.last_updated}
        """,
        title="PM Context Status"
    ))
```

## Testing Strategy

```python
# tests/test_grounding.py
@pytest.mark.asyncio
async def test_grounding_all_providers():
    """Test search works with all providers"""
    providers = ["openai", "anthropic", "gemini", "ollama"]
    
    for provider in providers:
        adapter = GroundingAdapter(provider, "test_key")
        
        # Mock the provider-specific method
        with patch.object(adapter, f"_{provider}_search") as mock:
            mock.return_value = SearchResult(
                content="Test content",
                citations=["https://example.com"]
            )
            
            result = await adapter.search("test query")
            
            assert result.content == "Test content"
            assert len(result.citations) == 1

# tests/test_context.py
def test_context_b2b_vs_b2c():
    """Test context influences PRD generation"""
    
    # B2B context
    b2b_context = Context(
        company=CompanyContext(name="Enterprise Corp"),
        product=ProductContext(model="B2B SaaS")
    )
    
    # B2C context
    b2c_context = Context(
        company=CompanyContext(name="Consumer App"),
        product=ProductContext(model="B2C Mobile")
    )
    
    # Generate PRDs
    b2b_prd = PRDAgent(b2b_context).generate_prd("Feature X")
    b2c_prd = PRDAgent(b2c_context).generate_prd("Feature X")
    
    # Verify different focus
    assert "ROI" in b2b_prd.phases["problem"].content
    assert "enterprise" in b2b_prd.phases["problem"].content.lower()
    
    assert "user engagement" in b2c_prd.phases["problem"].content.lower()
    assert "retention" in b2c_prd.phases["problem"].content.lower()
```

## Success Metrics

### Week 1 Targets
- ✅ Onboarding completes in <3 minutes
- ✅ Native search works with 3+ LLM providers
- ✅ Graceful degradation when search unavailable
- ✅ Context persisted in version-controlled YAML

### Week 2 Targets
- ✅ PRD generation in <60 seconds
- ✅ B2B vs B2C differentiation works
- ✅ Caching reduces redundant API calls by 80%
- ✅ Context influences all 5 PRD phases

## Key Advantages

1. **No External Dependencies**: Uses native LLM search capabilities
2. **Provider Flexibility**: Works with OpenAI, Anthropic, Gemini, Ollama
3. **Context-Aware**: Every output tailored to specific company/product
4. **Cost Effective**: Smart caching minimizes API calls
5. **Graceful Degradation**: Works offline or without search
6. **Extensible**: Easy to add new artifacts (roadmap, personas, etc.)

## Next Steps

1. Complete Week 1 implementation (onboarding + context)
2. Test with real companies (B2B and B2C)
3. Implement PRD generation with context awareness
4. Add telemetry to measure cache hit rates
5. Prepare for Phase 2 (quality gates + CI integration)

---

# Implementation Clarifications & Technical Details

## 1. Context Versioning Strategy

### Problem
The original plan mentions `context.version` for cache invalidation but doesn't specify how this version is generated. Timestamps aren't sufficient for tracking meaningful changes.

### Solution: Content-Based Hashing

```python
class ContextVersion:
    """Generate deterministic version from context content"""
    
    @staticmethod
    def compute_version(context_dir: Path) -> str:
        """
        Create SHA256 hash of all context files combined.
        This ensures cache invalidation happens exactly when content changes.
        """
        hasher = hashlib.sha256()
        
        # Sort files for deterministic ordering
        context_files = ['company.yaml', 'product.yaml', 'market.yaml', 
                        'team.yaml', 'okrs.yaml']
        
        for filename in context_files:
            filepath = context_dir / filename
            if filepath.exists():
                # Hash file content, excluding timestamps
                content = filepath.read_bytes()
                hasher.update(content)
        
        # Return first 12 chars for readability (still unique enough)
        return hasher.hexdigest()[:12]
    
    @staticmethod
    def has_changed(context_dir: Path, cached_version: str) -> bool:
        """Check if context has changed since cache was created"""
        current_version = ContextVersion.compute_version(context_dir)
        return current_version != cached_version
```

### Benefits
- Deterministic: Same content always produces same version
- Automatic invalidation: Changes trigger new version automatically
- No manual version bumping required
- Works across machines and environments

## 2. Async/Sync Pattern Clarification

### Clear Architectural Rules

1. **Async Operations** (I/O bound):
   - All LLM API calls
   - Web search operations
   - Network requests
   
2. **Sync Operations** (CPU bound or fast I/O):
   - File reading/writing (YAML files are small)
   - Context hashing
   - Cache lookups (in-memory)

### Implementation Pattern

```python
# Async for network operations
class GroundingAdapter:
    async def search(self, query: str) -> SearchResult:
        """All network calls must be async"""
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                return await response.json()

class PRDAgent:
    async def generate_prd(self, title: str) -> PRD:
        """LLM generation is async"""
        phases = []
        for phase in self.phase_generators:
            result = await phase.generate()  # Async LLM call
            phases.append(result)
        return PRD(phases=phases)

# Sync for file operations
class ContextManager:
    def save_context(self, context: Context) -> None:
        """File I/O can be sync - YAML files are small"""
        for section in ['company', 'product', 'market']:
            path = self.context_dir / f"{section}.yaml"
            with open(path, 'w') as f:
                yaml.dump(getattr(context, section), f)
    
    def load_context(self) -> Context:
        """Reading YAML is fast enough to be sync"""
        data = {}
        for section in ['company', 'product', 'market']:
            path = self.context_dir / f"{section}.yaml"
            with open(path) as f:
                data[section] = yaml.safe_load(f)
        return Context(**data)

# CLI bridges async/sync worlds
@app.command()
def new(artifact: str, title: str):
    """CLI commands are sync but call async operations internally"""
    
    if artifact == "prd":
        # Load context synchronously
        context = ContextManager().load_context()
        
        # Create agent
        agent = PRDAgent(context)
        
        # Bridge to async world for generation
        prd = asyncio.run(agent.generate_prd(title))
        
        # Save results synchronously
        save_prd(prd)
```

## 3. MVP Scope Reduction Strategy

### Phase 1A: Core MVP (Week 1-2)

Focus on single provider initially:

```python
class SimpleOpenAIGrounding:
    """MVP: OpenAI-only implementation, no abstraction"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.cache = {}  # Simple in-memory cache for MVP
    
    async def search(self, query: str) -> SearchResult:
        # Check cache
        if query in self.cache:
            return self.cache[query]
        
        # OpenAI web search
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": query}],
                tools=[{"type": "web_search"}],
                tool_choice="auto"
            )
            
            result = SearchResult(
                content=response.choices[0].message.content,
                citations=self.extract_citations(response)
            )
            
            self.cache[query] = result
            return result
            
        except Exception as e:
            # Graceful degradation
            logger.warning(f"Search failed: {e}")
            return SearchResult(content="", citations=[])
```

### Phase 1B: Multi-Provider Support (Post-MVP)

Add abstraction only after core functionality works:

```python
# Future enhancement - not needed for MVP
class GroundingAdapter:
    """Multi-provider abstraction - implement after MVP ships"""
    
    @classmethod
    def create(cls, provider: str, api_key: str) -> 'GroundingProvider':
        """Factory method for provider selection"""
        if provider == "openai":
            return OpenAIGrounding(api_key)
        elif provider == "anthropic":
            return AnthropicGrounding(api_key)
        elif provider == "gemini":
            return GeminiGrounding(api_key)
        else:
            return NoOpGrounding()  # Graceful degradation
```

## 4. Production Code Requirements

### From Architectural Sketch to Production

The PHASE1.md code samples are **architectural sketches**. Production implementation requires:

#### Required for MVP
- [ ] Input validation on all public methods
- [ ] Basic error handling (try/except on API calls)
- [ ] Logging at key points (info level)
- [ ] Type hints on public interfaces
- [ ] Basic retry logic (3 attempts with exponential backoff)
- [ ] Config validation on startup

#### Nice to Have (Post-MVP)
- [ ] Comprehensive error messages
- [ ] Debug logging
- [ ] Metrics/telemetry
- [ ] Circuit breakers
- [ ] Rate limiting
- [ ] Complete type coverage

### Example: Production-Ready Method

```python
from typing import Optional, Dict, Any
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class OnboardingAgent:
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def enrich_company(
        self, 
        company_name: str,
        timeout: float = 30.0
    ) -> Optional[CompanyContext]:
        """
        Enrich company information using web search.
        
        Args:
            company_name: Name of company to research
            timeout: Maximum time to wait for search results
            
        Returns:
            Enriched company context or None if search fails
            
        Raises:
            ValueError: If company_name is invalid
        """
        # Input validation
        if not company_name:
            raise ValueError("Company name cannot be empty")
        
        if len(company_name) < 2:
            raise ValueError("Company name too short")
        
        # Sanitize input
        company_name = company_name.strip()
        
        logger.info(f"Starting enrichment for: {company_name}")
        
        try:
            # Build search query
            query = self._build_search_query(company_name)
            
            # Execute search with timeout
            result = await asyncio.wait_for(
                self.grounding.search(query),
                timeout=timeout
            )
            
            # Parse and validate results
            context = self._parse_company_data(result)
            
            if not self._validate_context(context):
                logger.warning(f"Invalid context for {company_name}")
                return None
            
            logger.info(f"Successfully enriched: {company_name}")
            return context
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout enriching {company_name}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to enrich {company_name}: {e}")
            raise  # Let retry decorator handle it
```

## 5. Revised Implementation Timeline

### Week 1: Foundation (Simplified)

**Day 1-2: Project Setup**
- Basic project structure
- Configuration loading (YAML only)
- Logging setup

**Day 3: OpenAI Integration**
- Simple OpenAI client wrapper
- Basic web search function (no abstraction)
- In-memory caching

**Day 4: Context System**
- Context schema definition (Pydantic models)
- YAML persistence
- Content-based versioning

**Day 5: Basic Onboarding**
- Interactive prompts (using Rich)
- Manual context input (no search initially)
- Context validation

### Week 2: PRD Generation

**Day 6-7: PRD Phases**
- 5-phase structure
- B2B vs B2C templates
- Simple template rendering

**Day 8: Integration**
- Wire up onboarding → context → PRD
- Basic CLI commands

**Day 9: Testing**
- Core functionality tests
- B2B vs B2C differentiation test

**Day 10: Polish**
- Error handling
- Documentation
- README update

## 6. Testing Strategy for MVP

### Essential Tests Only

```python
# tests/test_context_version.py
def test_version_changes_with_content():
    """Version should change when content changes"""
    version1 = ContextVersion.compute_version(context_dir)
    
    # Modify content
    (context_dir / "company.yaml").write_text("name: NewCorp")
    
    version2 = ContextVersion.compute_version(context_dir)
    assert version1 != version2

def test_version_stable_without_changes():
    """Version should be stable when content unchanged"""
    version1 = ContextVersion.compute_version(context_dir)
    version2 = ContextVersion.compute_version(context_dir)
    assert version1 == version2

# tests/test_prd_generation.py
def test_b2b_uses_correct_template():
    """B2B context should use B2B template"""
    context = create_b2b_context()
    agent = PRDAgent(context)
    prd = agent.generate_prd("Test Feature")
    
    assert "ROI" in prd.phases["problem"].content
    assert "enterprise" in prd.phases["problem"].content.lower()

def test_b2c_uses_correct_template():
    """B2C context should use B2C template"""
    context = create_b2c_context()
    agent = PRDAgent(context)
    prd = agent.generate_prd("Test Feature")
    
    assert "user engagement" in prd.phases["problem"].content.lower()
    assert "retention" in prd.phases["problem"].content.lower()
```

## Summary of Changes

1. **Context Versioning**: Use SHA256 content hashing instead of timestamps
2. **Async/Sync**: Clear rules - async for I/O, sync for files, asyncio.run() in CLI
3. **MVP Scope**: Start with OpenAI only, no multi-provider abstraction initially
4. **Code Clarity**: Architectural sketches vs production requirements clearly defined
5. **Timeline**: Simplified to focus on core value delivery
6. **Testing**: Minimal essential tests for MVP

This approach reduces implementation risk while maintaining architectural flexibility for future enhancements.