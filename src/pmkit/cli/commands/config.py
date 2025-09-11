"""
Configuration management commands for PM-Kit CLI.

Provides commands to view, validate, and initialize configuration files
with beautiful output and helpful error messages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from pydantic import ValidationError
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from pmkit.config import (
    clear_config,
    get_config_safe,
    init_global_config,
    init_project_config,
    load_config,
)
from pmkit.utils.console import console

# Create the config command app
app = typer.Typer(
    name="config",
    help="🔧 Configuration management for PM-Kit",
    rich_markup_mode="rich",
)


@app.command("show")
def show_config(
    format: str = typer.Option(
        "table", 
        "--format", 
        "-f",
        help="Output format: table, json, yaml"
    ),
    include_secrets: bool = typer.Option(
        False,
        "--include-secrets",
        help="Include API keys and tokens (⚠️  use with caution)"
    ),
) -> None:
    """
    📋 Show current configuration with secrets masked by default.
    
    Displays the current configuration in a beautiful format, combining
    settings from all sources (files, environment variables, defaults).
    """
    try:
        config = get_config_safe()
        
        if format.lower() == "json":
            _show_config_json(config, include_secrets)
        elif format.lower() == "yaml":
            _show_config_yaml(config, include_secrets)
        else:
            _show_config_table(config, include_secrets)
            
    except ValidationError as e:
        console.error("Configuration validation failed:")
        console.print(str(e))
        raise typer.Exit(1)
    except Exception as e:
        console.error(f"Failed to load configuration: {e}")
        raise typer.Exit(1)


@app.command("init")
def init_config(
    global_config: bool = typer.Option(
        False,
        "--global",
        "-g", 
        help="Initialize global user configuration"
    ),
    path: Optional[Path] = typer.Option(
        None,
        "--path",
        "-p",
        help="Custom path for configuration file"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite existing configuration file"
    ),
) -> None:
    """
    🚀 Initialize a new configuration file with sensible defaults.
    
    Creates either a project-specific .pmrc.yaml or global ~/.pmkit/config.yaml
    with comprehensive examples and documentation.
    """
    try:
        if global_config:
            if path:
                console.warning("--path ignored when using --global")
            
            try:
                config_path = init_global_config()
                console.success(f"Created global configuration: {config_path}")
                
            except FileExistsError as e:
                if force:
                    # Remove existing and recreate
                    existing_path = Path.home() / ".pmkit" / "config.yaml"
                    existing_path.unlink()
                    config_path = init_global_config()
                    console.success(f"Overwrote global configuration: {config_path}")
                else:
                    console.error(str(e))
                    console.info("Use --force to overwrite existing configuration")
                    raise typer.Exit(1)
        else:
            try:
                config_path = init_project_config(path)
                console.success(f"Created project configuration: {config_path}")
                
            except FileExistsError as e:
                if force:
                    # Remove existing and recreate
                    target_path = path or Path.cwd() / ".pmrc.yaml"
                    target_path.unlink()
                    config_path = init_project_config(path)
                    console.success(f"Overwrote project configuration: {config_path}")
                else:
                    console.error(str(e))
                    console.info("Use --force to overwrite existing configuration")
                    raise typer.Exit(1)
        
        # Show next steps
        _show_next_steps(config_path, global_config)
        
    except OSError as e:
        console.error(f"Failed to create configuration file: {e}")
        raise typer.Exit(1)


@app.command("validate")
def validate_config(
    fix: bool = typer.Option(
        False,
        "--fix",
        help="Attempt to fix common configuration issues"
    ),
) -> None:
    """
    ✅ Validate current configuration and check for issues.
    
    Performs comprehensive validation of the current configuration,
    checking for missing required fields, invalid values, and 
    providing helpful suggestions for fixes.
    """
    try:
        # Clear any cached config to force fresh validation
        clear_config()
        config = load_config()
        
        console.success("Configuration is valid!")
        
        # Show configuration summary
        _show_config_summary(config)
        
        # Check for potential issues
        _check_config_warnings(config)
        
    except ValidationError as e:
        console.error("Configuration validation failed:")
        console.print()
        
        # Show detailed errors with suggestions
        _show_validation_errors(e)
        
        if fix:
            console.info("Auto-fix is not yet implemented. Please fix issues manually.")
        
        raise typer.Exit(1)
        
    except Exception as e:
        console.error(f"Failed to validate configuration: {e}")
        raise typer.Exit(1)


@app.command("reset")
def reset_config(
    global_config: bool = typer.Option(
        False,
        "--global",
        "-g",
        help="Reset global configuration"
    ),
    confirm: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt"
    ),
) -> None:
    """
    🔄 Reset configuration to defaults.
    
    Removes existing configuration files and optionally recreates
    them with default values.
    """
    if global_config:
        config_path = Path.home() / ".pmkit" / "config.yaml"
        config_type = "global"
    else:
        config_path = Path.cwd() / ".pmrc.yaml"
        config_type = "project"
    
    if not config_path.exists():
        console.info(f"No {config_type} configuration file found at {config_path}")
        return
    
    if not confirm:
        response = typer.confirm(
            f"Are you sure you want to reset {config_type} configuration?"
        )
        if not response:
            console.info("Configuration reset cancelled")
            return
    
    try:
        # Remove existing config
        config_path.unlink()
        console.success(f"Removed {config_type} configuration: {config_path}")
        
        # Clear cached config
        clear_config()
        
        # Ask if they want to recreate with defaults
        if typer.confirm("Create new configuration with defaults?"):
            if global_config:
                new_path = init_global_config()
            else:
                new_path = init_project_config()
            
            console.success(f"Created new {config_type} configuration: {new_path}")
        
    except OSError as e:
        console.error(f"Failed to reset configuration: {e}")
        raise typer.Exit(1)


def _show_config_table(config: 'Config', include_secrets: bool) -> None:
    """Display configuration in a beautiful table format."""
    # Get config data (with or without secrets)
    if include_secrets:
        data = config.model_dump()
    else:
        data = config.model_dump_safe()
    
    console.print()
    console.print(
        Panel(
            _format_config_table(data),
            title="[bright]🔧 PM-Kit Configuration[/bright]",
            title_align="left",
            border_style="panel.border",
            padding=(1, 2),
        )
    )
    
    if not include_secrets:
        console.print(
            "\n[dim]💡 Use --include-secrets to show API keys and tokens[/dim]"
        )


def _format_config_table(data: dict) -> Table:
    """Format configuration data as a Rich table."""
    table = Table(show_header=True, header_style="table.header")
    table.add_column("Section", style="primary", width=15)
    table.add_column("Setting", style="info.text", width=25)
    table.add_column("Value", style="dim", width=30)
    
    def add_section(section_name: str, section_data: dict, prefix: str = "") -> None:
        for key, value in section_data.items():
            display_key = f"{prefix}{key}" if prefix else key
            
            if isinstance(value, dict):
                add_section(section_name, value, f"{display_key}.")
            else:
                # Format value for display
                if isinstance(value, Path):
                    display_value = str(value)
                elif isinstance(value, bool):
                    display_value = "✅ Yes" if value else "❌ No"
                elif value is None:
                    display_value = "[dim]Not set[/dim]"
                else:
                    display_value = str(value)
                
                table.add_row(section_name, display_key, display_value)
                section_name = ""  # Only show section name for first row
    
    for section, content in data.items():
        if isinstance(content, dict):
            add_section(section.title(), content)
        else:
            # Handle top-level settings
            display_value = str(content) if content is not None else "[dim]Not set[/dim]"
            table.add_row("General", section, display_value)
    
    return table


def _show_config_json(config: 'Config', include_secrets: bool) -> None:
    """Display configuration in JSON format."""
    if include_secrets:
        data = config.model_dump()
    else:
        data = config.model_dump_safe()
    
    # Convert Path objects to strings for JSON serialization
    def convert_paths(obj):
        if isinstance(obj, dict):
            return {k: convert_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_paths(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
    
    data = convert_paths(data)
    
    json_str = json.dumps(data, indent=2, default=str)
    console.print(json_str)


def _show_config_yaml(config: 'Config', include_secrets: bool) -> None:
    """Display configuration in YAML format."""
    try:
        import yaml
        
        if include_secrets:
            data = config.model_dump()
        else:
            data = config.model_dump_safe()
        
        # Convert Path objects to strings
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj
        
        data = convert_paths(data)
        yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False)
        console.print(yaml_str)
        
    except ImportError:
        console.error("PyYAML not available. Install with: pip install pyyaml")
        raise typer.Exit(1)


def _show_config_summary(config: 'Config') -> None:
    """Show a brief configuration summary."""
    summary = Text()
    summary.append("📊 Configuration Summary:\\n", style="info.text")
    summary.append(f"  Provider: {config.llm.provider}\\n")
    summary.append(f"  Model: {config.llm.model}\\n")
    summary.append(f"  Cache: {'Enabled' if config.cache.enabled else 'Disabled'}\\n")
    summary.append(f"  Debug: {'Enabled' if config.app.debug else 'Disabled'}\\n")
    
    if config.project_name:
        summary.append(f"  Project: {config.project_name}\\n")
    
    console.print(summary)


def _check_config_warnings(config: 'Config') -> None:
    """Check for potential configuration issues and show warnings."""
    warnings = []
    
    # Check for missing API key
    if not config.llm.api_key and config.llm.provider != 'ollama':
        warnings.append(
            f"No API key configured for {config.llm.provider}. "
            f"Set {config.llm.provider.upper()}_API_KEY environment variable."
        )
    
    # Check cache directory
    if config.cache.enabled and not config.cache.directory.exists():
        warnings.append(
            f"Cache directory does not exist: {config.cache.directory}. "
            "It will be created when needed."
        )
    
    # Check for debug mode in production
    if config.app.debug:
        warnings.append(
            "Debug mode is enabled. Disable for production use."
        )
    
    if warnings:
        console.print()
        console.warning("Potential Issues:")
        for warning in warnings:
            console.print(f"  ⚠️  {warning}")


def _show_validation_errors(error: ValidationError) -> None:
    """Show validation errors in a user-friendly format."""
    for err in error.errors():
        field_path = " → ".join(str(loc) for loc in err['loc'])
        console.print(f"[error]❌ {field_path}:[/error] {err['msg']}")


def _show_next_steps(config_path: Path, is_global: bool) -> None:
    """Show helpful next steps after creating configuration."""
    console.print()
    
    if is_global:
        steps = [
            "Edit the configuration file to set your preferences",
            "Set API keys as environment variables (see comments in file)",
            "Create project-specific .pmrc.yaml files as needed",
        ]
    else:
        steps = [
            f"Edit {config_path.name} to configure your project",
            "Set API keys in environment variables or .env file",
            "Run 'pm config validate' to check your configuration",
            "Start using PM-Kit: 'pm init' to set up your project context",
        ]
    
    console.print("[info]📋 Next Steps:[/info]")
    for i, step in enumerate(steps, 1):
        console.print(f"  {i}. {step}")


if __name__ == "__main__":
    app()