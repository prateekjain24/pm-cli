"""
Rich theme configuration for PM-Kit CLI.

Defines the beautiful color palette and styling system following
the design philosophy from DESIGN.md.
"""

from rich.theme import Theme

# PM-Kit Color Palette (from DESIGN.md)
PRIMARY = "#00A6FB"    # Bright blue for primary actions
SUCCESS = "#52C41A"    # Green for successful operations  
WARNING = "#FAAD14"    # Orange for warnings
ERROR = "#FF4D4F"      # Red for errors
INFO = "#1890FF"       # Light blue for information
MUTED = "#8C8C8C"      # Gray for secondary text
ACCENT = "#722ED1"     # Purple for highlights

# Rich Theme Definition
pmkit_theme = Theme({
    # Core semantic colors
    "primary": PRIMARY,
    "success": SUCCESS,
    "warning": WARNING,
    "error": ERROR,
    "info": INFO,
    "muted": MUTED,
    "accent": ACCENT,
    
    # Status indicators
    "success.text": f"bold {SUCCESS}",
    "warning.text": f"bold {WARNING}",
    "error.text": f"bold {ERROR}",
    "info.text": f"bold {INFO}",
    
    # UI Components
    "panel.title": f"bold {PRIMARY}",
    "panel.subtitle": MUTED,
    "panel.border": PRIMARY,
    
    "progress.bar": PRIMARY,
    "progress.complete": SUCCESS,
    "progress.remaining": MUTED,
    "progress.description": INFO,
    
    "table.header": f"bold {PRIMARY}",
    "table.border": MUTED,
    
    # Command Help
    "help.command": f"bold {PRIMARY}",
    "help.option": f"bold {ACCENT}",
    "help.argument": f"italic {INFO}",
    "help.example": MUTED,
    
    # Special emphasis
    "emoji": "none",  # Preserve emoji colors
    "highlight": f"bold {ACCENT}",
    "dim": MUTED,
    "bright": f"bold {PRIMARY}",
    
    # Status messages
    "status.running": f"bold {INFO}",
    "status.complete": f"bold {SUCCESS}",
    "status.failed": f"bold {ERROR}",
    "status.pending": f"bold {WARNING}",
})

__all__ = ["pmkit_theme", "PRIMARY", "SUCCESS", "WARNING", "ERROR", "INFO", "MUTED", "ACCENT"]