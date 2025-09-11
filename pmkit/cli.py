"""
PM-Kit CLI - Main entry point with REPL and command handling
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

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

app = typer.Typer(
    name="pmkit",
    help="PM-Kit: PM docs with LLM superpowers",
    add_completion=False,
    rich_markup_mode="rich"
)
console = Console()

# Available slash commands
SLASH_CMDS = [
    "/help", "/prd", "/review", "/model", "/provider", 
    "/ground", "/status", "/sandbox", "/ide", "/docs",
    "/publish", "/sync", "/release", "/init"
]
completer = WordCompleter(SLASH_CMDS, ignore_case=True, sentence=True)


def git_branch(cwd: Path) -> str:
    """Get current git branch name"""
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], 
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True
        )
        return result.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def ctx_pct(used: int, limit: int) -> str:
    """Calculate context usage percentage"""
    if not limit:
        return "100%"
    pct = max(0, min(100, int((1 - used/limit) * 100)))
    return f"{pct}%"


def bottom_toolbar(cwd: Path, model: str, sandbox: bool, 
                   used: int, limit: int) -> HTML:
    """Generate bottom toolbar for REPL"""
    branch = git_branch(cwd)
    sand = "sandbox" if sandbox else "no sandbox"
    ctx = ctx_pct(used, limit)
    
    return HTML(
        f"<b>{cwd}</b> ({branch or 'no-git'})  "
        f"<ansiyellow>{sand}</ansiyellow>   "
        f"<ansimagenta>{model}</ansimagenta>  "
        f"(<ansigreen>{ctx} context left</ansigreen>)"
    )


def show_banner():
    """Display welcome banner"""
    txt = Text()
    txt.append("🚀 PM-KIT v0.1.0", style="bold magenta")
    txt.append("\nCLI-first PM docs with LLM superpowers", style="dim")
    
    console.print(Panel(
        txt, 
        title="[bold blue]Slash REPL[/bold blue]", 
        subtitle="[dim]Type /help for commands[/dim]"
    ))


@app.callback()
def callback(ctx: typer.Context):
    """Global callback to set up context"""
    ctx.obj = {"cfg": load_config()}


@app.command("init")
def init_command(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing config")
):
    """Initialize PM-Kit configuration"""
    from pmkit.commands.init import init_pmkit
    init_pmkit(force=force, console=console)


@app.command("sh", aliases=["shell", "repl"])
def shell_command(
    project: str = typer.Option(".", "--project", "-p", help="Project root directory")
):
    """Start interactive REPL shell"""
    cwd = Path(project).resolve()
    
    if not cwd.exists():
        console.print(f"[red]Error:[/red] Project directory {cwd} does not exist")
        raise typer.Exit(1)
    
    cfg = load_config()
    
    # Initialize state
    state = {
        "model": cfg["llm"]["model"],
        "provider": cfg["llm"]["provider"],
        "sandbox": False,
        "used_tokens": 0,
        "limit": 128000
    }
    
    # Clear screen and show banner
    console.clear()
    show_banner()
    console.print()
    
    # Show tips
    console.print("[dim]Tips:[/dim]")
    console.print("  • Use [bold]/help[/bold] to see available commands")
    console.print("  • Use [bold]/model <name>[/bold] to switch models")
    console.print("  • Use [bold]/provider <name>[/bold] to switch providers")
    console.print("  • Press [bold]Ctrl+D[/bold] or [bold]Ctrl+C[/bold] to exit\n")
    
    # Check for updates
    local, latest = check_update("pmkit")
    if local and latest and local != latest:
        console.print(
            f"[yellow]📦 Update available![/yellow] {local} → {latest}\n"
            f"Run: [bold]pip install --upgrade pmkit[/bold]\n"
        )
    
    # Start REPL session
    session = PromptSession(completer=completer)
    
    while True:
        try:
            line = session.prompt(
                "> ",
                bottom_toolbar=lambda: bottom_toolbar(
                    cwd, state["model"], state["sandbox"],
                    state["used_tokens"], state["limit"]
                )
            )
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye! 👋[/dim]")
            break
        
        if not line.strip():
            continue
        
        if line.startswith("/"):
            # Handle slash commands
            hctx = SimpleNamespace(
                cfg=cfg,
                project=cwd,
                printer=console,
                state=state
            )
            
            try:
                result = dispatch_slash(parse_slash(line), hctx)
                if result:
                    console.print(result)
            except SystemExit as e:
                console.print(f"[red]Error:[/red] {e}")
            except Exception as e:
                console.print(f"[red]Error:[/red] {str(e)}")
                if cfg.get("debug", False):
                    console.print_exception()
        else:
            console.print(
                "[yellow]ℹ[/yellow] Commands must start with '/' "
                "(e.g., [bold]/prd \"My Feature\"[/bold])"
            )


@app.command("run")
def run_command(
    command: str,
    project: str = typer.Option(".", "--project", "-p", help="Project root directory")
):
    """Run a single slash command"""
    cwd = Path(project).resolve()
    
    if not cwd.exists():
        console.print(f"[red]Error:[/red] Project directory {cwd} does not exist")
        raise typer.Exit(1)
    
    if not command.startswith("/"):
        command = "/" + command
    
    cfg = load_config()
    
    # Create context
    hctx = SimpleNamespace(
        cfg=cfg,
        project=cwd,
        printer=console,
        state={
            "model": cfg["llm"]["model"],
            "provider": cfg["llm"]["provider"],
            "sandbox": False,
            "used_tokens": 0,
            "limit": 128000
        }
    )
    
    try:
        result = dispatch_slash(parse_slash(command), hctx)
        if result:
            console.print(result)
    except SystemExit as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        if cfg.get("debug", False):
            console.print_exception()
        raise typer.Exit(1)


@app.command("new")
def new_command(
    resource_type: str = typer.Argument(..., help="Resource type: prd, persona, okr"),
    name: str = typer.Argument(..., help="Resource name or title"),
    project: str = typer.Option(".", "--project", "-p", help="Project root directory")
):
    """Create new PM resource (PRD, persona, OKR)"""
    from pmkit.commands.new import create_resource
    
    cwd = Path(project).resolve()
    create_resource(resource_type, name, cwd, console)


@app.command("status")
def status_command(
    project: str = typer.Option(".", "--project", "-p", help="Project root directory"),
    format: str = typer.Option("human", "--format", "-f", help="Output format: human, json, reviewdog")
):
    """Check PRD quality gates and lint status"""
    from pmkit.commands.status import check_status
    
    cwd = Path(project).resolve()
    check_status(cwd, format, console)


@app.command("publish")
def publish_command(
    target: Optional[str] = typer.Argument(None, help="Specific PRD to publish"),
    project: str = typer.Option(".", "--project", "-p", help="Project root directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without publishing")
):
    """Publish PRDs to Confluence/Notion"""
    from pmkit.commands.publish import publish_docs
    
    cwd = Path(project).resolve()
    publish_docs(cwd, target, dry_run, console)


@app.command("sync")
def sync_command(
    resource: str = typer.Argument("issues", help="Resource to sync: issues, status"),
    project: str = typer.Option(".", "--project", "-p", help="Project root directory")
):
    """Sync PRD stories with Jira/GitHub issues"""
    from pmkit.commands.sync import sync_resources
    
    cwd = Path(project).resolve()
    sync_resources(resource, cwd, console)


@app.command("release")
def release_command(
    action: str = typer.Argument("draft", help="Action: draft, publish"),
    since: Optional[str] = typer.Option(None, "--since", help="Since tag/date"),
    project: str = typer.Option(".", "--project", "-p", help="Project root directory")
):
    """Generate or publish release notes"""
    from pmkit.commands.release import handle_release
    
    cwd = Path(project).resolve()
    handle_release(action, since, cwd, console)


@app.command("version")
def version_command():
    """Show PM-Kit version"""
    from pmkit import __version__
    console.print(f"PM-Kit version {__version__}")


if __name__ == "__main__":
    app()