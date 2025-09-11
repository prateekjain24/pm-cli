"""
PM-Kit CLI Package.

Beautiful command-line interface following the design philosophy
from DESIGN.md with Rich theming and Typer framework.
"""

from pmkit.cli.main import app, cli_main

__all__ = ["app", "cli_main"]