# PM-Kit Detailed Implementation Plan

## Executive Summary

PM-Kit is a CLI-first product management documentation system that treats PRDs, roadmaps, OKRs, personas, and release notes as code. Documents live in Git, flow through review gates like code, and auto-publish to Confluence/Notion without manual copy-paste or flag management.

### Core Value Proposition
- **Local-first, publish-later**: Git is the source of truth; wikis are mirrors
- **Zero-brain CLI**: Defaults over flags, infer everything from context
- **Deterministic phases**: Only re-run what changed, resume where comments start
- **Quality gates**: Block merges on ambiguity/TBD/missing metrics
- **Full traceability**: Every story maps to an issue, every issue maps back to PRD

## Technical Architecture

### Updated Technology Stack (2025)

#### Core Framework
- **CLI Framework**: Typer v0.12.0+ (FastAPI-style CLI development)
- **Console Output**: Rich v13.0.0+ (beautiful terminal formatting)
- **REPL Interface**: prompt-toolkit v3.0.52 (interactive shell)
- **Configuration**: PyYAML v6.0.0+ (YAML config management)

#### LLM Integrations
- **Anthropic SDK**: v0.67.0 (Claude integration with Documents support) (Sonnet-4)
- **OpenAI SDK**: v1.107.1 (GPT-5 and web search)
- **Google Generative AI**: google-genai ~1.35.0 (Gemini 2.5 Pro)
- **Ollama**: v0.2.0+ (local model support)

#### External Integrations
- **Confluence/Jira**: atlassian-python-api (latest)
- **GitHub Actions**: Reviewdog for PR annotations
- **WebSocket**: websockets v12.0+ (IDE companion)

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                       │
├──────────────┬────────────────┬──────────────┬─────────────┤
│   CLI (pm)   │  REPL (pmkit)  │  IDE Bridge  │   GitHub    │
│              │  /prd /review  │  (WebSocket) │   Actions   │
└──────┬───────┴────────┬───────┴──────┬───────┴──────┬──────┘
       │                │              │              │
┌──────▼────────────────▼──────────────▼──────────────▼──────┐
│                      CORE ENGINE                            │
├─────────────┬──────────────┬──────────────┬────────────────┤
│  Phase      │   Content    │   Quality    │   Publisher    │
│  Orchestrator│   Hashing   │   Gates      │   Adapters     │
└─────────────┴──────────────┴──────────────┴────────────────┘
       │                                              │
┌──────▼──────────────────────────────────────────────▼──────┐
│                      INTEGRATIONS                           │
├──────────────┬──────────────┬──────────────┬──────────────┤
│   LLM APIs   │  Confluence  │     Jira     │    GitHub    │
│  (Multi-provider) │   REST API   │   REST API   │     API      │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

## Implementation Roadmap

### Phase 0: Foundation & Scaffolding (Week 1)

#### Goals
- Establish project structure and core CLI framework
- Implement configuration system and LLM abstraction
- Set up basic REPL with slash commands

#### Deliverables

1. **Project Structure**
```
pmkit/
├── __init__.py
├── __main__.py              # Entry point
├── cli.py                   # Typer CLI + REPL
├── config/
│   ├── __init__.py
│   ├── loader.py           # Config management
│   └── schema.py           # Config validation
├── context/
│   ├── __init__.py
│   └── store.py            # Context persistence
├── llm/
│   ├── __init__.py
│   ├── backends.py         # Provider abstraction
│   ├── normalize.py        # Citation normalization
│   └── grounding.py        # Web search integration
├── slash/
│   ├── __init__.py
│   ├── parser.py           # Command parsing
│   ├── registry.py         # Handler registry
│   └── handlers.py         # Command implementations
└── utils/
    ├── __init__.py
    ├── hashing.py          # SHA256 content hashing
    ├── files.py            # File operations
    └── update_check.py     # PyPI version checking
```

2. **Configuration System**
```yaml
# ~/.pmkit/config.yaml
llm:
  provider: openai           # openai | anthropic | gemini | ollama
  model: gpt-5
  temperature: 0.0           # Deterministic by default
  max_tokens: 8000

keys:
  openai_api_key: ${OPENAI_API_KEY}
  anthropic_api_key: ${ANTHROPIC_API_KEY}
  gemini_api_key: ${GEMINI_API_KEY}

grounding:
  enabled: true
  strategy: native           # native | perplexity | custom
  max_searches: 3
  allowed_domains: []
  blocked_domains: []

context:
  global_dir: "~/.pmkit"
  project_dir_name: ".pmkit"
  store: "json"              # sqlite planned for v2
  max_history_messages: 25
  summarization_threshold: 20000

safety:
  redact_keys_in_logs: true
  write_files_with_backup: true
```

3. **CLI Commands**
```bash
pmkit init                   # Initialize config
pmkit sh                     # Start REPL
pmkit run "/prd 'Title'"    # Run single command
```

#### Code Samples

**cli.py** (Core REPL Implementation):
```python
from __future__ import annotations
import subprocess
from pathlib import Path
from types import SimpleNamespace
import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML

from pmkit.config.loader import load_config
from pmkit.slash.parser import parse as parse_slash
from pmkit.slash.registry import dispatch as dispatch_slash
from pmkit.utils.update_check import check_update

app = typer.Typer(help="pm-kit: PM docs with LLM superpowers")
console = Console()

SLASH_CMDS = ["/help", "/prd", "/review", "/model", "/provider", 
               "/ground", "/status", "/sandbox", "/ide", "/docs"]
completer = WordCompleter(SLASH_CMDS, ignore_case=True, sentence=True)

def git_branch(cwd: Path) -> str:
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], 
            cwd=str(cwd), 
            stderr=subprocess.DEVNULL
        )
        return result.decode().strip()
    except Exception:
        return ""

def ctx_pct(used: int, limit: int) -> str:
    if not limit: 
        return "100%"
    pct = max(0, min(100, int((1 - used/limit) * 100)))
    return f"{pct}%"

def bottom_toolbar(cwd: Path, model: str, sandbox: bool, 
                   used: int, limit: int):
    branch = git_branch(cwd)
    sand = "sandbox" if sandbox else "no sandbox"
    ctx = ctx_pct(used, limit)
    return HTML(
        f"<b>{cwd}</b> ({branch or '-'})  "
        f"<ansiyellow>{sand}</ansiyellow>   "
        f"<ansimagenta>{model}</ansimagenta>  "
        f"(<ansigreen>{ctx} context left</ansigreen>)"
    )
```

### Phase 1: PRD Generation Engine (Week 2)

#### Goals
- Implement 5-phase PRD generation pipeline
- Add deterministic content hashing and caching
- Create orchestration DAG for phase dependencies

#### Deliverables

1. **Phase Orchestrator**
```python
# pmkit/phases/orchestrator.py
from typing import Dict, List, Optional
from pathlib import Path
import hashlib
import json

class PhaseOrchestrator:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.cache_dir = project_root / ".pmkit" / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def compute_hash(self, inputs: Dict[str, str]) -> str:
        """Compute deterministic hash for phase inputs"""
        content = json.dumps(inputs, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def should_regenerate(self, phase: str, inputs: Dict) -> bool:
        """Check if phase needs regeneration based on hash"""
        current_hash = self.compute_hash(inputs)
        cache_file = self.cache_dir / f"{phase}.hash"
        
        if cache_file.exists():
            stored_hash = cache_file.read_text().strip()
            return current_hash != stored_hash
        return True
    
    def run_phase_dag(self, title: str) -> Dict[str, Path]:
        """Execute phases in dependency order"""
        phases = [
            "01_problem",
            "02_solution", 
            "03_requirements",
            "04_prototype_prompts",
            "05_final_prd"
        ]
        
        results = {}
        for phase in phases:
            if self.should_regenerate(phase, {"title": title, 
                                              "prior": results}):
                results[phase] = self.generate_phase(phase, title, results)
                self.update_cache(phase, {"title": title, 
                                         "prior": results})
        return results
```

2. **PRD File Structure**
```
/product/
  prds/<slug>/
    01_problem.md           # Problem definition
    02_solution.md          # Solution approach
    03_requirements.md      # User stories & acceptance criteria
    04_prototype_prompts.md # AI prompts for prototyping
    05_final_prd.md        # Assembled final PRD
    manifest.yaml          # Metadata and links
    .cache/               # Phase hashes
```

3. **Commands**
```bash
pm new prd "Connect Session Replay to Funnel Analysis"
pm run                      # Execute phase pipeline
pm status                   # Check PRD quality
```

### Phase 2: Quality Gates & CI Integration (Week 3)

#### Goals
- Implement comprehensive linting rules
- Integrate GitHub Actions with Reviewdog
- Add branch protection and CODEOWNERS

#### Deliverables

1. **Lint Rules Engine**
```python
# pmkit/gates/linter.py
from typing import List, Tuple
import re

class PRDLinter:
    AMBIGUOUS_TERMS = [
        "easy", "simple", "quick", "fast", "intuitive",
        "user-friendly", "seamless", "robust", "scalable"
    ]
    
    TBD_PATTERN = re.compile(r'\b(TBD|TODO|TBA|FIXME)\b', re.IGNORECASE)
    METRIC_PATTERN = re.compile(r'\d+(%|ms|s|min|hour|day|week|month)')
    
    def lint_ambiguity(self, content: str) -> List[Tuple[int, str, str]]:
        """Find ambiguous terms with line numbers"""
        issues = []
        for line_num, line in enumerate(content.split('\n'), 1):
            for term in self.AMBIGUOUS_TERMS:
                if term.lower() in line.lower():
                    issues.append((
                        line_num, 
                        f"Ambiguous term '{term}'",
                        f"Replace with specific, measurable language"
                    ))
        return issues
    
    def lint_todos(self, content: str) -> List[Tuple[int, str, str]]:
        """Find TBD/TODO markers"""
        issues = []
        for line_num, line in enumerate(content.split('\n'), 1):
            if self.TBD_PATTERN.search(line):
                issues.append((
                    line_num,
                    "Incomplete section",
                    "Complete before PR review"
                ))
        return issues
    
    def lint_metrics(self, content: str) -> List[Tuple[int, str, str]]:
        """Ensure metrics are measurable"""
        issues = []
        if "Success Metrics" in content:
            metrics_section = content.split("Success Metrics")[1].split("#")[0]
            if not self.METRIC_PATTERN.search(metrics_section):
                issues.append((
                    0,
                    "Missing measurable metrics",
                    "Add specific targets with units"
                ))
        return issues
```

2. **GitHub Actions Workflow**
```yaml
# .github/workflows/pm-gates.yml
name: PM Gates

on:
  pull_request:
    paths:
      - 'product/**'

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install PM-Kit
        run: |
          pip install -e .
      
      - name: Run PM Gates
        run: |
          pm status --format=reviewdog > results.json
      
      - name: Post Review Comments
        uses: reviewdog/action-suggester@v1
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          tool_name: pm-gates
          level: error
          fail_on_error: true
```

3. **CODEOWNERS**
```
# CODEOWNERS
/product/**/*.md @pm-team @design-lead @eng-lead
/product/prds/** @pm-team
/product/okrs/** @leadership
```

### Phase 3: Confluence Publisher (Week 4)

#### Goals
- Implement idempotent Confluence publishing
- Handle storage format and versioning
- Add hash-based skip for unchanged content

#### Deliverables

1. **Confluence Adapter**
```python
# pmkit/publishers/confluence.py
from atlassian import Confluence
from typing import Optional, Dict
import hashlib

class ConfluencePublisher:
    def __init__(self, url: str, username: str, api_token: str):
        self.confluence = Confluence(
            url=url,
            username=username,
            password=api_token
        )
        self.space_key = None
        self.parent_page_id = None
    
    def content_hash(self, content: str) -> str:
        """Generate hash for content comparison"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def publish_page(self, title: str, content: str, 
                    parent_id: Optional[str] = None) -> Dict:
        """
        Idempotent page publish - create or update
        Returns page info with URL
        """
        # Check if page exists
        existing = self.confluence.get_page_by_title(
            space=self.space_key,
            title=title
        )
        
        if existing:
            # Check if content changed
            current_hash = self.content_hash(existing['body']['storage']['value'])
            new_hash = self.content_hash(content)
            
            if current_hash == new_hash:
                return {
                    'status': 'unchanged',
                    'page_id': existing['id'],
                    'url': self._get_page_url(existing['id'])
                }
            
            # Update page (idempotent PUT operation)
            updated = self.confluence.update_page(
                page_id=existing['id'],
                title=title,
                body=content,
                version_number=existing['version']['number'] + 1
            )
            
            return {
                'status': 'updated',
                'page_id': updated['id'],
                'url': self._get_page_url(updated['id']),
                'version': updated['version']['number']
            }
        else:
            # Create new page (non-idempotent POST)
            created = self.confluence.create_page(
                space=self.space_key,
                title=title,
                body=content,
                parent_id=parent_id or self.parent_page_id
            )
            
            return {
                'status': 'created',
                'page_id': created['id'],
                'url': self._get_page_url(created['id'])
            }
    
    def publish_tree(self, root_path: Path) -> Dict[str, Dict]:
        """Publish entire PRD tree to Confluence"""
        results = {}
        
        # Create parent PRD page
        prd_content = (root_path / "05_final_prd.md").read_text()
        prd_html = self.markdown_to_storage(prd_content)
        
        parent_result = self.publish_page(
            title=f"PRD: {root_path.name}",
            content=prd_html
        )
        results['main'] = parent_result
        
        # Publish phase pages as children
        phases = ["01_problem", "02_solution", "03_requirements", 
                 "04_prototype_prompts"]
        
        for phase in phases:
            phase_file = root_path / f"{phase}.md"
            if phase_file.exists():
                phase_content = phase_file.read_text()
                phase_html = self.markdown_to_storage(phase_content)
                
                result = self.publish_page(
                    title=f"{root_path.name} - {phase.replace('_', ' ').title()}",
                    content=phase_html,
                    parent_id=parent_result['page_id']
                )
                results[phase] = result
        
        return results
```

2. **Storage Format Converter**
```python
def markdown_to_storage(self, markdown: str) -> str:
    """Convert Markdown to Confluence storage format"""
    # Basic conversions
    html = markdown
    
    # Headers
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    
    # Bold and italic
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
    
    # Code blocks
    html = re.sub(r'```(\w+)?\n(.*?)```', 
                  r'<ac:structured-macro ac:name="code">'
                  r'<ac:parameter ac:name="language">\1</ac:parameter>'
                  r'<ac:plain-text-body><![CDATA[\2]]></ac:plain-text-body>'
                  r'</ac:structured-macro>', 
                  html, flags=re.DOTALL)
    
    # Lists
    html = re.sub(r'^\* (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'(<li>.*</li>\n)+', r'<ul>\g<0></ul>', html)
    
    return html
```

### Phase 4: Jira Integration & Traceability (Week 5)

#### Goals
- Parse user stories from requirements
- Create/update Jira issues with deep links
- Maintain bidirectional traceability

#### Deliverables

1. **Story Parser**
```python
# pmkit/integrations/jira.py
from atlassian import Jira
import yaml
import re
from typing import List, Dict

class JiraIntegration:
    def __init__(self, url: str, username: str, api_token: str, project: str):
        self.jira = Jira(
            url=url,
            username=username,
            password=api_token
        )
        self.project = project
    
    def parse_stories(self, requirements_path: Path) -> List[Dict]:
        """Extract user stories from requirements.md"""
        content = requirements_path.read_text()
        stories = []
        
        # Pattern: Story ID, title, acceptance criteria
        story_pattern = re.compile(
            r'^#+\s*(?:Story\s+)?(\w+-\d+)?:?\s*(.+?)$\n'
            r'(.*?)'
            r'^(?:Acceptance|AC):?\s*\n((?:[-*]\s*.+\n?)+)',
            re.MULTILINE | re.DOTALL
        )
        
        for match in story_pattern.finditer(content):
            story_id = match.group(1) or f"STORY-{len(stories)+1}"
            title = match.group(2).strip()
            description = match.group(3).strip()
            ac_text = match.group(4)
            
            # Parse acceptance criteria
            ac_items = re.findall(r'[-*]\s*(.+)', ac_text)
            
            stories.append({
                'id': story_id,
                'title': title,
                'description': description,
                'acceptance_criteria': ac_items,
                'source_line': content[:match.start()].count('\n') + 1
            })
        
        return stories
    
    def sync_issues(self, stories: List[Dict], prd_url: str) -> Dict[str, str]:
        """
        Create or update Jira issues from stories
        Returns mapping of story_id -> issue_key
        """
        mapping = {}
        
        for story in stories:
            # Check if issue exists (search by custom field or label)
            jql = f'project = {self.project} AND labels = "prd-{story["id"]}"'
            existing = self.jira.jql(jql)
            
            if existing['issues']:
                # Update existing issue
                issue = existing['issues'][0]
                self.jira.update_issue(
                    issue['key'],
                    {
                        'summary': story['title'],
                        'description': self._format_description(story, prd_url)
                    }
                )
                mapping[story['id']] = issue['key']
            else:
                # Create new issue
                new_issue = self.jira.create_issue({
                    'project': {'key': self.project},
                    'issuetype': {'name': 'Story'},
                    'summary': story['title'],
                    'description': self._format_description(story, prd_url),
                    'labels': [f"prd-{story['id']}", 'pm-kit']
                })
                mapping[story['id']] = new_issue['key']
        
        return mapping
    
    def _format_description(self, story: Dict, prd_url: str) -> str:
        """Format story for Jira description"""
        ac_text = '\n'.join([f"* {ac}" for ac in story['acceptance_criteria']])
        
        return f"""
{story['description']}

h3. Acceptance Criteria
{ac_text}

h3. Source
[View in PRD|{prd_url}#line-{story['source_line']}]
Generated by pm-kit
"""
```

2. **Manifest Update**
```python
def update_manifest(prd_path: Path, issue_mapping: Dict[str, str]):
    """Update manifest.yaml with issue links"""
    manifest_path = prd_path / "manifest.yaml"
    
    if manifest_path.exists():
        manifest = yaml.safe_load(manifest_path.read_text()) or {}
    else:
        manifest = {}
    
    manifest['issues'] = issue_mapping
    manifest['last_sync'] = datetime.now().isoformat()
    
    manifest_path.write_text(yaml.dump(manifest, default_flow_style=False))
```

### Phase 5: Comment-Driven Review (Week 6)

#### Goals
- Fetch Confluence comments
- Map comments to phases
- Regenerate from earliest impacted phase

#### Deliverables

1. **Comment Fetcher**
```python
# pmkit/review/comment_fetcher.py
class CommentFetcher:
    def __init__(self, confluence: Confluence):
        self.confluence = confluence
    
    def fetch_comments(self, page_url: str) -> List[Dict]:
        """Fetch all comments from Confluence page"""
        page_id = self._extract_page_id(page_url)
        
        # Get footer comments (v2 API)
        footer_comments = self.confluence.get_page_comments(
            page_id, 
            expand='body.view,version'
        )
        
        # Get inline comments
        inline_comments = self.confluence.get_page_inline_comments(page_id)
        
        # Combine and normalize
        all_comments = []
        
        for comment in footer_comments['results']:
            all_comments.append({
                'type': 'footer',
                'author': comment['version']['by']['displayName'],
                'content': comment['body']['view']['value'],
                'created': comment['version']['when'],
                'context': None
            })
        
        for comment in inline_comments['results']:
            all_comments.append({
                'type': 'inline',
                'author': comment['version']['by']['displayName'],
                'content': comment['body']['view']['value'],
                'created': comment['version']['when'],
                'context': comment.get('properties', {}).get('inline-original-selection')
            })
        
        return all_comments
    
    def map_to_phases(self, comments: List[Dict]) -> Dict[str, List[Dict]]:
        """Map comments to PRD phases based on context"""
        phase_mapping = {
            '01_problem': [],
            '02_solution': [],
            '03_requirements': [],
            '04_prototype_prompts': [],
            '05_final_prd': []
        }
        
        phase_keywords = {
            '01_problem': ['problem', 'issue', 'pain', 'challenge'],
            '02_solution': ['solution', 'approach', 'design'],
            '03_requirements': ['requirement', 'story', 'acceptance'],
            '04_prototype_prompts': ['prototype', 'prompt', 'example'],
            '05_final_prd': ['summary', 'overall', 'general']
        }
        
        for comment in comments:
            # Try to match based on context or content
            matched_phase = '05_final_prd'  # Default
            
            if comment['context']:
                context_lower = comment['context'].lower()
                for phase, keywords in phase_keywords.items():
                    if any(kw in context_lower for kw in keywords):
                        matched_phase = phase
                        break
            
            phase_mapping[matched_phase].append(comment)
        
        return phase_mapping
```

### Phase 6: Release Notes Automation (Week 7)

#### Goals
- Generate release notes from PRs and issues
- Categorize changes automatically
- Inject PRD context for impact

#### Deliverables

1. **Release Notes Generator**
```python
# pmkit/releases/generator.py
from github import Github
from typing import List, Dict
import re

class ReleaseNotesGenerator:
    def __init__(self, github_token: str, repo: str):
        self.github = Github(github_token)
        self.repo = self.github.get_repo(repo)
    
    def generate_notes(self, since_tag: str = None) -> str:
        """Generate categorized release notes"""
        
        # Get PRs since last tag
        if since_tag:
            since_date = self._get_tag_date(since_tag)
        else:
            # Get last tag automatically
            tags = list(self.repo.get_tags())
            if tags:
                since_tag = tags[0].name
                since_date = self._get_tag_date(since_tag)
            else:
                since_date = None
        
        # Collect merged PRs
        prs = self._get_merged_prs(since_date)
        
        # Categorize changes
        categories = {
            'breaking': [],
            'features': [],
            'improvements': [],
            'fixes': [],
            'docs': [],
            'other': []
        }
        
        for pr in prs:
            category = self._categorize_pr(pr)
            categories[category].append(pr)
        
        # Generate markdown
        return self._format_release_notes(categories, since_tag)
    
    def _categorize_pr(self, pr) -> str:
        """Categorize PR based on labels and title"""
        labels = [l.name.lower() for l in pr.labels]
        title_lower = pr.title.lower()
        
        if 'breaking' in labels or 'breaking change' in title_lower:
            return 'breaking'
        elif 'feature' in labels or 'feat:' in title_lower:
            return 'features'
        elif 'enhancement' in labels or 'improve' in title_lower:
            return 'improvements'
        elif 'bug' in labels or 'fix:' in title_lower:
            return 'fixes'
        elif 'documentation' in labels or 'docs:' in title_lower:
            return 'docs'
        else:
            return 'other'
    
    def _format_release_notes(self, categories: Dict, since_tag: str) -> str:
        """Format release notes in markdown"""
        notes = f"# Release Notes\n\n"
        
        if since_tag:
            notes += f"Changes since {since_tag}\n\n"
        
        # Breaking changes (if any)
        if categories['breaking']:
            notes += "## ⚠️ Breaking Changes\n\n"
            for pr in categories['breaking']:
                notes += f"- {pr.title} (#{pr.number}) by @{pr.user.login}\n"
                notes += self._get_prd_context(pr)
            notes += "\n"
        
        # New features
        if categories['features']:
            notes += "## ✨ New Features\n\n"
            for pr in categories['features']:
                notes += f"- {pr.title} (#{pr.number})\n"
                notes += self._get_prd_context(pr)
            notes += "\n"
        
        # Improvements
        if categories['improvements']:
            notes += "## 🚀 Improvements\n\n"
            for pr in categories['improvements']:
                notes += f"- {pr.title} (#{pr.number})\n"
            notes += "\n"
        
        # Bug fixes
        if categories['fixes']:
            notes += "## 🐛 Bug Fixes\n\n"
            for pr in categories['fixes']:
                notes += f"- {pr.title} (#{pr.number})\n"
            notes += "\n"
        
        # Documentation
        if categories['docs']:
            notes += "## 📚 Documentation\n\n"
            for pr in categories['docs']:
                notes += f"- {pr.title} (#{pr.number})\n"
            notes += "\n"
        
        return notes
    
    def _get_prd_context(self, pr) -> str:
        """Extract PRD context from PR body"""
        # Look for PRD links in PR body
        prd_pattern = re.compile(r'PRD:\s*(.+?)(?:\n|$)')
        match = prd_pattern.search(pr.body or '')
        
        if match:
            return f"  *Impact*: {match.group(1)}\n"
        return ""
```

## File Structure

```
pmkit/
├── __init__.py
├── __main__.py
├── cli.py                      # Main CLI entry point
├── config/
│   ├── __init__.py
│   ├── loader.py              # Configuration loading
│   └── schema.py              # Config validation
├── context/
│   ├── __init__.py
│   └── store.py               # Context persistence
├── llm/
│   ├── __init__.py
│   ├── backends.py            # LLM provider abstraction
│   ├── normalize.py           # Citation normalization
│   └── grounding.py           # Web search integration
├── slash/
│   ├── __init__.py
│   ├── parser.py              # Command parsing
│   ├── registry.py            # Handler registry
│   └── handlers.py            # Command implementations
├── phases/
│   ├── __init__.py
│   ├── orchestrator.py        # Phase DAG execution
│   └── templates/             # Phase templates
│       ├── 01_problem.yaml
│       ├── 02_solution.yaml
│       ├── 03_requirements.yaml
│       ├── 04_prototype.yaml
│       └── 05_final.yaml
├── gates/
│   ├── __init__.py
│   ├── linter.py              # Lint rules
│   ├── validator.py           # Content validation
│   └── rules.yaml             # Rule definitions
├── publishers/
│   ├── __init__.py
│   ├── base.py                # Publisher interface
│   ├── confluence.py          # Confluence adapter
│   └── notion.py              # Notion adapter (future)
├── integrations/
│   ├── __init__.py
│   ├── jira.py                # Jira integration
│   ├── github.py              # GitHub integration
│   └── gitlab.py              # GitLab (future)
├── agents/
│   ├── __init__.py
│   ├── prd_agent.py           # PRD generation
│   ├── review_agent.py        # PRD review
│   └── release_agent.py       # Release notes
├── review/
│   ├── __init__.py
│   ├── comment_fetcher.py     # Comment retrieval
│   └── regenerator.py         # Phase regeneration
├── releases/
│   ├── __init__.py
│   └── generator.py           # Release notes generation
├── utils/
│   ├── __init__.py
│   ├── hashing.py             # Content hashing
│   ├── files.py               # File operations
│   ├── git.py                 # Git operations
│   └── update_check.py        # Version checking
├── ide/
│   ├── __init__.py
│   ├── ide_server.py          # WebSocket server
│   └── vscode/                # VS Code extension
│       ├── package.json
│       └── src/
│           └── extension.ts
└── tests/
    ├── __init__.py
    ├── test_cli.py
    ├── test_orchestrator.py
    ├── test_linter.py
    ├── test_confluence.py
    └── test_jira.py

pyproject.toml                  # Project configuration
README.md                       # Documentation
.github/
└── workflows/
    ├── pm-gates.yml           # PR validation
    └── release.yml            # Auto-release
```

## Dependencies (pyproject.toml)

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pmkit"
version = "0.1.0"
description = "PM-Kit: CLI-first PM docs with LLM superpowers"
authors = [{name = "Your Name", email = "your.email@example.com"}]
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}

dependencies = [
    # CLI Framework
    "typer>=0.12.0",
    "rich>=13.0.0",
    "prompt_toolkit>=3.0.52",
    
    # Configuration
    "pyyaml>=6.0.0",
    "pydantic>=2.0.0",
    
    # LLM SDKs
    "anthropic>=0.67.0",
    "openai>=1.107.1",
    "google-genai>=1.35.0",
    "ollama>=0.5.3",
    
    # Integrations
    "atlassian-python-api>=3.41.0",
    "PyGithub>=2.1.0",
    
    # Utilities
    "httpx>=0.27.0",
    "websockets>=12.0",
    "python-dotenv>=1.0.0",
    
    # Testing
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
]

[project.optional-dependencies]
dev = [
    "black>=24.0.0",
    "ruff>=0.2.0",
    "mypy>=1.8.0",
    "pre-commit>=3.6.0",
]

[project.scripts]
pmkit = "pmkit.cli:app"
pm = "pmkit.cli:app"

[tool.setuptools.packages.find]
where = ["."]
include = ["pmkit*"]

[tool.black]
line-length = 88
target-version = ["py311"]

[tool.ruff]
line-length = 88
target-version = "py311"
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
```

## Success Metrics

### 60-Day Targets
- [ ] ≥70% of PM docs authored in repo (not wiki)
- [ ] Median PRD PR review cycle ≤48h
- [ ] ≥90% of merged PRDs have synced issues
- [ ] All quality gates passing on main branch

### 90-Day Targets
- [ ] Release notes generated for every tag
- [ ] Comment-driven review operational
- [ ] Full traceability from PRD to deployed code
- [ ] 3+ teams actively using pm-kit

## Risk Mitigation

### Technical Risks

1. **Confluence API Instability**
   - **Risk**: API rate limits or breaking changes
   - **Mitigation**: Exponential backoff, request caching, API version pinning

2. **LLM Provider Outages**
   - **Risk**: Service unavailability affects PRD generation
   - **Mitigation**: Multi-provider fallback, local caching, offline mode

3. **Version Conflicts**
   - **Risk**: Concurrent edits cause conflicts
   - **Mitigation**: Optimistic locking, version tracking, conflict resolution UI

### Process Risks

1. **PM Adoption Resistance**
   - **Risk**: PMs prefer existing wiki workflow
   - **Mitigation**: Gradual rollout, training sessions, migration tools

2. **Review Bottlenecks**
   - **Risk**: Required reviewers slow down PRD merging
   - **Mitigation**: Time-boxed reviews, delegation rules, auto-approval for minor changes

3. **Template Sprawl**
   - **Risk**: Too many custom templates reduce consistency
   - **Mitigation**: Template registry, version control, deprecation policy

## Development Timeline

### Week 1: Foundation
- Set up project structure
- Implement CLI framework and REPL
- Create configuration system
- Add LLM backend abstraction

### Week 2: PRD Engine
- Build phase orchestrator
- Implement content hashing
- Create PRD templates
- Add generation pipeline

### Week 3: Quality Gates
- Implement lint rules
- Set up GitHub Actions
- Add Reviewdog integration
- Create CODEOWNERS

### Week 4: Publishing
- Build Confluence adapter
- Implement storage format conversion
- Add idempotent operations
- Create publish command

### Week 5: Traceability
- Parse user stories
- Integrate Jira API
- Implement issue sync
- Add bidirectional links

### Week 6: Review Flow
- Fetch Confluence comments
- Map to phases
- Implement regeneration
- Update publish flow

### Week 7: Release & Polish
- Generate release notes
- Add documentation
- Write tests
- Package for distribution

## Next Immediate Steps

1. **Initialize Project**
```bash
mkdir pmkit
cd pmkit
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
```

2. **Install Dependencies**
```bash
pip install typer rich prompt-toolkit pyyaml httpx
pip install anthropic openai google-generativeai ollama
pip install atlassian-python-api pygithub
pip install pytest pytest-asyncio pytest-cov
```

3. **Create Basic Structure**
```bash
mkdir -p pmkit/{cli,config,context,llm,slash,phases,gates}
mkdir -p pmkit/{publishers,integrations,agents,utils}
touch pmkit/__init__.py
touch pmkit/__main__.py
```

4. **Port Existing Code**
   - Copy CLI implementation from pmkit_final_wired_stack_and_repl.md
   - Update imports and dependencies
   - Test basic REPL functionality

5. **Begin Phase 0 Implementation**
   - Set up configuration loader
   - Implement LLM backends
   - Create slash command parser
   - Test end-to-end flow

## Conclusion

This implementation plan provides a comprehensive roadmap for building PM-Kit, a powerful CLI-first product management documentation system. By following this phased approach, we can deliver a production-ready MVP in 7 weeks that transforms how PMs work with documentation.

The plan prioritizes:
- **Practical implementation** over theoretical perfection
- **Incremental delivery** with weekly milestones
- **User value** in every phase
- **Technical excellence** through deterministic operations and quality gates

With the updated 2025 libraries and proven architectural patterns, PM-Kit will deliver on its promise: making PM docs live in Git, flow through review gates like code, and publish seamlessly to collaboration platforms—all without the complexity of traditional documentation workflows.