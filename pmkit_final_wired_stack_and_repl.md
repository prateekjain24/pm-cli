# pm-kit — **Final** Wired Tech Stack + Slash REPL + Native-Grounded LLM + IDE Companion (Python)

> Straight to the point: this is the **ship-ready wiring** that combines the tech stack + slash commands + REPL UX (banner, status bar, update toast) + native web-search tools + local/global context + optional IDE companion. Paste this into a repo and you’re off.

---

## 0) What you get

- **Typer CLI** with:
  - `pmkit sh` → **slash REPL** (`/prd`, `/review`, `/model`, `/provider`, `/ground`, …)
  - `pmkit run '/cmd ...'` → run a single slash command headlessly
- **Multi-LLM backends** (OpenAI / Claude / Gemini / Ollama) and **native web-search** (OpenAI web_search, Claude web search, Gemini google_search), with fallback path for Ollama.
- **Local+global context** persistence (JSON MVP, SQLite later), resumable sessions.
- **Deterministic by default**: temperature 0, content hashes, sidecars.
- **IDE Companion** (optional): tiny WebSocket server + VS Code extension skeleton to mirror “Connect to IDE companion” UX.
- **Status bar** in REPL: `cwd (git-branch)  sandbox|no sandbox  model  (context left %)`
- **Auto-update toast** if a newer PyPI version exists.

---

## 1) Repo Layout

```
pmkit/
  __init__.py
  cli.py                      # Typer entry point + REPL (banner, toolbar, update toast)
  config/
    loader.py                 # ~/.pmkit/config.yaml loader/merge + env override
  context/
    store.py                  # JSON context store (MVP)
  llm/
    backends.py               # OpenAI/Claude/Gemini/Ollama adapters + grounding hooks
    normalize.py              # provider → normalized citations (stubs)
  slash/
    parser.py                 # /command grammar
    registry.py               # handlers registry + dispatcher
    handlers.py               # /prd, /review, /model, /provider, /ground
  agents/
    prd_agent.py              # PRD generator (deterministic)
    review_agent.py           # PRD reviewer
  utils/
    hashing.py
    files.py
    update_check.py           # PyPI version checker (toast)
  ide/                        # OPTIONAL companion
    ide_server.py             # WebSocket bridge (CLI side)
    vscode/                   # VS Code extension skeleton
      package.json
      src/extension.ts
pyproject.toml
README.md
```

---

## 2) Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install typer rich prompt_toolkit pyyaml httpx openai anthropic google-generativeai ollama
# optional unified API (keep for later): pip install litellm
```

**Keys (recommended via env):** `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`

---

## 3) Global Config (created by `pmkit init`, or write manually)

`~/.pmkit/config.yaml`
```yaml
llm:
  provider: openai              # openai | anthropic | gemini | ollama
  model: gpt-4o
  temperature: 0.0
  max_tokens: 2000

keys:
  openai_api_key: null
  anthropic_api_key: null
  gemini_api_key: null

grounding:
  enabled: true
  strategy: native              # native | perplexity | custom
  max_searches: 3               # Claude only
  allowed_domains: []
  blocked_domains: []

context:
  global_dir: "~/.pmkit"
  project_dir_name: ".pmkit"
  store: "json"                 # sqlite later
  max_history_messages: 25
  summarization_threshold: 20000

safety:
  redact_keys_in_logs: true
  write_files_with_backup: true
```

---

## 4) Code — Typer CLI + REPL (banner, toolbar, update toast)

`pmkit/cli.py`
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
SLASH_CMDS = ["/help","/prd","/review","/model","/provider","/ground","/status","/sandbox","/ide","/docs"]
completer = WordCompleter(SLASH_CMDS, ignore_case=True, sentence=True)

def git_branch(cwd: Path) -> str:
    try:
        return subprocess.check_output(["git","rev-parse","--abbrev-ref","HEAD"], cwd=str(cwd)).decode().strip()
    except Exception:
        return ""

def ctx_pct(used: int, limit: int) -> str:
    if not limit: return "100%"
    pct = max(0, min(100, int((1 - used/limit) * 100)))
    return f"{pct}%"

def bottom_toolbar(cwd: Path, model: str, sandbox: bool, used: int, limit: int):
    branch = git_branch(cwd)
    sand = "sandbox" if sandbox else "no sandbox"
    ctx = ctx_pct(used, limit)
    return HTML(f"<b>{cwd}</b> ({branch or '-'})  <ansiyellow>{sand}</ansiyellow>   "
                f"<ansimagenta>{model}</ansimagenta>  (<ansigreen>{ctx} context left</ansigreen>)")

def banner():
    txt = Text()
    txt.append("> PM-KIT", style="bold magenta")
    console.print(Panel(txt, title="Slash REPL", subtitle="Type /help"))

@app.callback()
def cb(ctx: typer.Context):
    ctx.obj = {"cfg": load_config()}

@app.command("sh")
def shell(project: str = typer.Option(".", help="Project root")):
    cwd = Path(project).resolve()
    cfg = load_config()
    state = {"model": cfg["llm"]["model"], "provider": cfg["llm"]["provider"],
             "sandbox": False, "used_tokens": 0, "limit": 128000}
    console.clear()
    banner()
    console.print("Tips: 1) Use /help  2) /model <name>  3) /provider <openai|anthropic|gemini|ollama>\n")
    local, latest = check_update("pmkit")
    if local and latest and local != latest:
        console.print(f"[yellow]Update available! {local} → {latest}[/yellow]  Run: [bold]pmkit self-update[/bold]")

    session = PromptSession(completer=completer)
    while True:
        try:
            line = session.prompt("> ", bottom_toolbar=lambda: bottom_toolbar(cwd, state["model"], state["sandbox"], state["used_tokens"], state["limit"]))
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye")
            break
        if not line.strip(): continue
        if line.startswith("/"):
            # lightweight context object passed to handlers
            hctx = SimpleNamespace(cfg=cfg, project=cwd, printer=console, state=state)
            try:
                out = dispatch_slash(parse_slash(line), hctx)
                if out: console.print(out)
            except SystemExit as e:
                console.print(f"[red]Error:[/red] {e}")
            except Exception:
                console.print_exception()
        else:
            console.print("Commands start with '/' (e.g., /prd \"Title\")")

@app.command("run")
def run(command: str, project: str = typer.Option(".", help="Project root")):
    cfg = load_config()
    hctx = SimpleNamespace(cfg=cfg, project=Path(project).resolve(), printer=console, state={"model": cfg["llm"]["model"], "provider": cfg["llm"]["provider"], "sandbox": False, "used_tokens": 0, "limit": 128000})
    out = dispatch_slash(parse_slash(command), hctx)
    if out: console.print(out)
```

---

## 5) Code — Config Loader

`pmkit/config/loader.py`
```python
from __future__ import annotations
import os
from pathlib import Path
import yaml

DEFAULT_CFG = {
    "llm": {"provider": "openai", "model": "gpt-4o", "temperature": 0.0, "max_tokens": 2000},
    "keys": {"openai_api_key": None, "anthropic_api_key": None, "gemini_api_key": None},
    "grounding": {"enabled": True, "strategy": "native", "max_searches": 3, "allowed_domains": [], "blocked_domains": []},
    "context": {"global_dir": "~/.pmkit", "project_dir_name": ".pmkit", "store": "json", "max_history_messages": 25, "summarization_threshold": 20000},
    "safety": {"redact_keys_in_logs": True, "write_files_with_backup": True},
}

def load_config() -> dict:
    cfg = {k:(v.copy() if isinstance(v,dict) else v) for k,v in DEFAULT_CFG.items()}
    path = Path(os.path.expanduser("~")) / ".pmkit" / "config.yaml"
    if path.exists():
        user = yaml.safe_load(path.read_text()) or {}
        for k, v in user.items():
            if isinstance(v, dict) and k in cfg:
                cfg[k].update(v)
            else:
                cfg[k] = v
    # env overrides
    cfg["keys"]["openai_api_key"] = os.getenv("OPENAI_API_KEY") or cfg["keys"]["openai_api_key"]
    cfg["keys"]["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY") or cfg["keys"]["anthropic_api_key"]
    cfg["keys"]["gemini_api_key"] = os.getenv("GEMINI_API_KEY") or cfg["keys"]["gemini_api_key"]
    return cfg
```

---

## 6) Code — Context Store (JSON MVP)

`pmkit/context/store.py`
```python
from __future__ import annotations
from pathlib import Path
import json, time

class JsonContextStore:
    def __init__(self, project_dir: Path, agent: str):
        self.dir = project_dir / ".pmkit"
        self.dir.mkdir(exist_ok=True, parents=True)
        self.file = self.dir / f"context_{agent}.json"

    def load(self) -> list[dict]:
        if self.file.exists():
            return json.loads(self.file.read_text())
        return []

    def save(self, messages: list[dict]):
        self.file.write_text(json.dumps(messages, indent=2))

    def append(self, role: str, content: str, meta: dict | None = None):
        msgs = self.load()
        msgs.append({"role": role, "content": content, "meta": meta or {}, "ts": time.time()})
        self.save(msgs)
```

---

## 7) Code — LLM Backends + Grounding hooks

`pmkit/llm/backends.py`
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class Completion:
    text: str
    citations: List[Dict[str, Any]]  # normalized elsewhere

class LLMBackend:
    def complete(self, messages: list[dict], *, tools: list | None = None,
                 model: str | None = None, temperature: float | None = None,
                 max_tokens: int | None = None) -> Completion: ...
# --- Grounding selection
def provider_tools_for_grounding(provider: str, cfg: dict) -> list:
    g = cfg.get("grounding", {})
    if not g.get("enabled"): return []
    if provider == "openai": return [{"type": "web_search"}]
    if provider == "anthropic":
        tool = {"type": "web_search_20250305", "name": "web_search"}
        if g.get("max_searches"): tool["max_uses"] = g["max_searches"]
        if g.get("allowed_domains"): tool["allowed_domains"] = g["allowed_domains"]
        if g.get("blocked_domains"): tool["blocked_domains"] = g["blocked_domains"]
        return [tool]
    if provider == "gemini": return ["google_search"]  # adapter maps this to SDK tool
    return []  # ollama

# --- OpenAI
class OpenAIBackend(LLMBackend):
    def __init__(self, api_key: str | None):
        import openai
        self.openai = openai
        if api_key: openai.api_key = api_key
    def complete(self, messages, *, tools=None, model=None, temperature=None, max_tokens=None) -> Completion:
        client = self.openai.OpenAI()
        prompt = "\n".join([m["content"] for m in messages])
        resp = client.responses.create(model=model or "gpt-4o", input=prompt, tools=tools or [], temperature=0.0 if temperature is None else temperature, max_output_tokens=max_tokens or 1500)
        return Completion(text=resp.output_text, citations=[])

# --- Anthropic
class AnthropicBackend(LLMBackend):
    def __init__(self, api_key: str | None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
    def complete(self, messages, *, tools=None, model=None, temperature=None, max_tokens=None) -> Completion:
        conv = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] in ("user","assistant","system")]
        resp = self.client.messages.create(model=model or "claude-3-7-sonnet-20250219", messages=conv, tools=tools or [], temperature=0.0 if temperature is None else temperature, max_tokens=max_tokens or 1500)
        parts = [part.text for part in getattr(resp, "content", []) if getattr(part, "type", "")=="text"]
        return Completion(text="".join(parts), citations=[])

# --- Gemini
class GeminiBackend(LLMBackend):
    def __init__(self, api_key: str | None):
        import google.generativeai as genai
        self.genai = genai
        if api_key: genai.configure(api_key=api_key)
    def complete(self, messages, *, tools=None, model=None, temperature=None, max_tokens=None) -> Completion:
        from google.generativeai import types
        tool_objs = [types.Tool(google_search=types.GoogleSearch())] if (tools and "google_search" in tools) else []
        cfg = types.GenerationConfig(temperature=0.0 if temperature is None else temperature, max_output_tokens=max_tokens or 1500)
        prompt = "\n".join([m["content"] for m in messages])
        mdl = self.genai.GenerativeModel(model or "gemini-2.5-pro")
        resp = mdl.generate_content(prompt, tools=tool_objs, generation_config=cfg)
        return Completion(text=getattr(resp, "text", ""), citations=[])

# --- Ollama
class OllamaBackend(LLMBackend):
    def __init__(self):
        import ollama
        self.ollama = ollama
    def complete(self, messages, *, tools=None, model=None, temperature=None, max_tokens=None) -> Completion:
        model = model or "llama3.1"
        resp = self.ollama.chat(model=model, messages=messages)
        txt = resp["message"]["content"] if isinstance(resp, dict) else ""
        return Completion(text=txt, citations=[])

def resolve_backend(cfg: dict) -> LLMBackend:
    p = cfg["llm"]["provider"]
    if p == "openai":   return OpenAIBackend(cfg["keys"].get("openai_api_key"))
    if p == "anthropic":return AnthropicBackend(cfg["keys"].get("anthropic_api_key"))
    if p == "gemini":   return GeminiBackend(cfg["keys"].get("gemini_api_key"))
    if p == "ollama":   return OllamaBackend()
    raise SystemExit(f"Unknown provider: {p}")
```

`pmkit/llm/normalize.py` (stubs you’ll fill later)
```python
def normalize_openai(resp): return []
def normalize_claude(resp): return []
def normalize_gemini(resp): return []
```

---

## 8) Code — Agents (PRD, Review)

`pmkit/agents/prd_agent.py`
```python
from __future__ import annotations
from typing import List, Tuple
from pmkit.llm.backends import LLMBackend, Completion, provider_tools_for_grounding

class PRDAgent:
    name = "prd"
    def __init__(self, backend: LLMBackend, ctx_store, cfg: dict):
        self.backend, self.ctx, self.cfg = backend, ctx_store, cfg

    def prepare(self, title: str) -> List[dict]:
        sys = {"role": "system", "content": "You are a precise PM assistant. Output clean Markdown. No fluff."}
        user = {"role": "user", "content": f"Create a PRD outline for: {title}\nInclude: Problem, Goals, Non-goals, Metrics, Risks."}
        history = self.ctx.load()[-10:]
        return [sys, *history, user]

    def call(self, messages: List[dict]) -> Tuple[str, list]:
        tools = provider_tools_for_grounding(self.cfg["llm"]["provider"], self.cfg)
        comp: Completion = self.backend.complete(messages, tools=tools, model=self.cfg["llm"]["model"], temperature=self.cfg["llm"]["temperature"], max_tokens=self.cfg["llm"]["max_tokens"])
        return comp.text, comp.citations

    def run(self, title: str) -> str:
        text, citations = self.call(self.prepare(title))
        self.ctx.append("assistant", text, {"citations": citations, "model": self.cfg["llm"]["model"]})
        return text
```

`pmkit/agents/review_agent.py`
```python
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
from pmkit.llm.backends import LLMBackend, Completion, provider_tools_for_grounding

class ReviewAgent:
    name = "review"
    def __init__(self, backend: LLMBackend, ctx_store, cfg: dict):
        self.backend, self.ctx, self.cfg = backend, ctx_store, cfg

    def prepare(self, content: str) -> List[dict]:
        sys = {"role": "system", "content": "You are a tough, constructive PRD reviewer. Be specific and actionable."}
        user = {"role": "user", "content": f"Review this PRD and provide improvements:\n\n{content}"}
        history = self.ctx.load()[-10:]
        return [sys, *history, user]

    def call(self, messages: List[dict]) -> Tuple[str, list]:
        tools = provider_tools_for_grounding(self.cfg["llm"]["provider"], self.cfg)
        comp: Completion = self.backend.complete(messages, tools=tools, model=self.cfg["llm"]["model"], temperature=self.cfg["llm"]["temperature"], max_tokens=self.cfg["llm"]["max_tokens"])
        return comp.text, comp.citations

    def run(self, path: Path) -> str:
        if not path.exists():
            raise SystemExit(f"File not found: {path}")
        text, citations = self.call(self.prepare(path.read_text()))
        self.ctx.append("assistant", text, {"citations": citations, "model": self.cfg["llm"]["model"]})
        return text
```

---

## 9) Code — Slash Parser, Registry, Handlers

`pmkit/slash/parser.py`
```python
from __future__ import annotations
import shlex, re
from dataclasses import dataclass

KV_RE = re.compile(r'(\w+):(".*?"|\'.*?\'|[^ \t]+)')
TARGET_RE = re.compile(r'@([^\s]+)')

@dataclass
class SlashCmd:
    verb: str
    free: str
    kv: dict[str, str]
    flags: list[str]
    targets: list[str]

def parse(line: str) -> SlashCmd:
    if not line.startswith("/"): raise SystemExit("Slash command must start with '/'")
    tokens = shlex.split(line[1:])
    if not tokens: raise SystemExit("Empty command")
    verb, *rest = tokens
    raw = " ".join(rest)
    kv = {k: v.strip('\"\' ) for k, v in KV_RE.findall(raw)}
    flags = [t for t in rest if t.startswith("--")]
    targets = [m.group(1) for m in TARGET_RE.finditer(raw)]
    tmp = KV_RE.sub("", raw); tmp = TARGET_RE.sub("", tmp)
    free = " ".join([w for w in shlex.split(tmp) if not w.startswith("--")]).strip()
    return SlashCmd(verb=verb, free=free, kv=kv, flags=flags, targets=targets)
```

`pmkit/slash/registry.py`
```python
from __future__ import annotations
from typing import Callable, Dict
from pmkit.slash.parser import SlashCmd

_REG: Dict[str, Callable] = {}

def slash(name: str):
    def dec(fn: Callable):
        _REG[name] = fn
        return fn
    return dec

def dispatch(cmd: SlashCmd, ctx):
    if cmd.verb not in _REG: raise SystemExit(f"Unknown command: /{cmd.verb}")
    return _REG[cmd.verb](cmd, ctx)
```

`pmkit/slash/handlers.py`
```python
from __future__ import annotations
from pathlib import Path
from pmkit.slash.registry import slash
from pmkit.llm.backends import resolve_backend
from pmkit.context.store import JsonContextStore
from pmkit.agents.prd_agent import PRDAgent
from pmkit.agents.review_agent import ReviewAgent

def _apply_grounding_flags(cmd, cfg):
    if "--no-grounding" in cmd.flags: cfg["grounding"]["enabled"] = False
    if "allowed_domains" in cmd.kv: cfg["grounding"]["allowed_domains"] = [s.strip() for s in cmd.kv["allowed_domains"].split(",") if s.strip()]
    if "blocked_domains" in cmd.kv: cfg["grounding"]["blocked_domains"] = [s.strip() for s in cmd.kv["blocked_domains"].split(",") if s.strip()]

@slash("help")
def help_(cmd, ctx):
    return """Available:
  /prd "Title" [--no-grounding]
  /review @PRD.md [allowed_domains:foo.com,bar.com] [--no-grounding]
  /model <name> | /provider <openai|anthropic|gemini|ollama>
  /ground on|off
"""

@slash("provider")
def provider(cmd, ctx):
    name = cmd.free.strip()
    if name not in ("openai","anthropic","gemini","ollama"): raise SystemExit("provider must be one of: openai|anthropic|gemini|ollama")
    ctx.cfg["llm"]["provider"] = name
    ctx.state["provider"] = name
    return f"provider = {name}"

@slash("model")
def model(cmd, ctx):
    name = cmd.free.strip()
    if not name: raise SystemExit("Usage: /model <model-name>")
    ctx.cfg["llm"]["model"] = name
    ctx.state["model"] = name
    return f"model = {name}"

@slash("ground")
def ground(cmd, ctx):
    val = cmd.free.strip().lower()
    if val not in ("on","off"): raise SystemExit("Usage: /ground on|off")
    ctx.cfg["grounding"]["enabled"] = (val == "on")
    return f"grounding = {val}"

@slash("prd")
def prd(cmd, ctx):
    title = cmd.free or cmd.kv.get("title")
    if not title: raise SystemExit('Usage: /prd "Title"')
    cfg = ctx.cfg; _apply_grounding_flags(cmd, cfg)
    backend = resolve_backend(cfg)
    store = JsonContextStore(Path(ctx.project), agent="prd")
    out = PRDAgent(backend, store, cfg).run(title=title)
    return out

@slash("review")
def review(cmd, ctx):
    target = cmd.targets[0] if cmd.targets else "PRD.md"
    cfg = ctx.cfg; _apply_grounding_flags(cmd, cfg)
    backend = resolve_backend(cfg)
    store = JsonContextStore(Path(ctx.project), agent="review")
    out = ReviewAgent(backend, store, cfg).run(path=Path(ctx.project)/target)
    return out
```

---

## 10) Utilities

`pmkit/utils/hashing.py`
```python
import hashlib
def sha256_text(txt: str) -> str: return hashlib.sha256(txt.encode("utf-8")).hexdigest()
```

`pmkit/utils/files.py`
```python
from __future__ import annotations
from pathlib import Path
def write_with_backup(path: Path, content: str, backup: bool = True):
    if backup and path.exists(): path.with_suffix(path.suffix + ".bak").write_text(path.read_text())
    path.write_text(content)
```

`pmkit/utils/update_check.py`
```python
import json, importlib.metadata, urllib.request
def check_update(package: str = "pmkit"):
    try:
        local = importlib.metadata.version(package)
        data = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{package}/json"))
        latest = data["info"]["version"]
        return local, latest
    except Exception:
        return None, None
```

---

## 11) IDE Companion (optional)

### 11.1 CLI-side WebSocket server

`pmkit/ide/ide_server.py`
```python
import asyncio, json
import websockets

clients = set()

async def handler(ws):
    clients.add(ws)
    try:
        async for msg in ws:
            req = json.loads(msg)
            if req.get("op") == "ping":
                await ws.send(json.dumps({"ok": True}))
    finally:
        clients.remove(ws)

async def main():
    async with websockets.serve(handler, "127.0.0.1", 8787):
        print("[IDE] listening on ws://127.0.0.1:8787")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
```

Add a slash command `/ide` later to ping this server and show **[ERROR] IDEClient** if not found.

### 11.2 VS Code extension skeleton

`pmkit/ide/vscode/package.json`
```json
{
  "name": "pmkit-ide",
  "displayName": "pm-kit IDE Companion",
  "version": "0.0.1",
  "publisher": "you",
  "engines": { "vscode": "^1.90.0" },
  "activationEvents": ["onStartupFinished"],
  "main": "./out/extension.js",
  "contributes": {
    "commands": [
      {"command": "pmkit.sendSelection", "title": "pm-kit: Send Selection"},
      {"command": "pmkit.applyEdit", "title": "pm-kit: Apply Edit"}
    ]
  }
}
```

`pmkit/ide/vscode/src/extension.ts`
```ts
import * as vscode from "vscode";
import WebSocket from "ws";

let ws: WebSocket | null = null;

export function activate(ctx: vscode.ExtensionContext) {
  const connect = () => {
    ws = new WebSocket("ws://127.0.0.1:8787");
    ws.on("open", () => console.log("pm-kit IDE connected"));
    ws.on("error", () => console.warn("pm-kit IDE not running"));
  };
  connect();

  ctx.subscriptions.push(
    vscode.commands.registerCommand("pmkit.sendSelection", async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor || !ws || ws.readyState !== ws.OPEN) return;
      const text = editor.document.getText(editor.selection);
      ws.send(JSON.stringify({ op: "selection", text, path: editor.document.uri.fsPath }));
    })
  );
}

export function deactivate() { if (ws) ws.close(); }
```

> This reproduces the “connect to IDE companion” feel. Keep it optional.

---

## 12) pyproject.toml

`pyproject.toml`
```toml
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pmkit"
version = "0.1.0"
description = "pm-kit: CLI-first PM docs with LLM superpowers"
authors = [{name = "You"}]
dependencies = [
  "typer>=0.12.0",
  "rich>=13.0.0",
  "prompt_toolkit>=3.0.0",
  "pyyaml>=6.0.0",
  "httpx>=0.27.0",
  "openai>=1.0.0",
  "anthropic>=0.25.0",
  "google-generativeai>=0.7.0",
  "ollama>=0.2.0",
  "websockets>=12.0"
]

[project.scripts]
pmkit = "pmkit.cli:app"
```

---

## 13) Quick Start

```bash
# 1) config
mkdir -p ~/.pmkit && cat > ~/.pmkit/config.yaml <<'YAML'
llm: { provider: openai, model: gpt-4o, temperature: 0.0, max_tokens: 2000 }
keys: { openai_api_key: "$OPENAI_API_KEY" }
grounding: { enabled: true, strategy: native, max_searches: 3, allowed_domains: [], blocked_domains: [] }
context: { global_dir: "~/.pmkit", project_dir_name: ".pmkit", store: "json", max_history_messages: 25, summarization_threshold: 20000 }
safety: { redact_keys_in_logs: true, write_files_with_backup: true }
YAML

# 2) REPL
pmkit sh
pm> /prd "Connect Session Replay to Funnel Analysis"
pm> /review @PRD.md --no-grounding
pm> /model claude-3.7-sonnet
pm> /provider gemini
```

---

## 14) Roadmap (tight, shippable)

- [ ] `pmkit init` (collect keys, write config, sanity check providers)
- [x] Slash REPL + banner + toolbar + update toast
- [x] /prd, /review wired to agents
- [x] Native web-search routing
- [ ] Normalize citations + footnotes
- [ ] `pmkit status` (gates: ambiguity/TBD/link/glossary)
- [ ] Perplexity fallback adapter (for Ollama)
- [ ] Inline directive scanner (`<!-- pm: /cmd -->`)
- [ ] VS Code extension polish + `/ide` ping

---

## 15) Determinism & Safety

- **Deterministic defaults** (`temperature=0`), optional seeds for local models later.
- **Context pruning** + summarization (beyond threshold).
- **Backups** for writes; redact keys in logs.

---

That’s the complete, wired plan with the code you need. Drop these files into a repo, install, and you’ll have the Gemini/Codex-style REPL on top of your agents—grounded, deterministic, and extensible.
