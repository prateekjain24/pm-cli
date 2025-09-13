"""Context directory structure initialization for PM-Kit.

Simple, production-ready structure creation without over-engineering.
Creates the .pmkit/context directory with template YAML files.
"""

from pathlib import Path
from typing import Optional, Dict, Any

import yaml

from pmkit.utils.paths import ensure_directory, safe_write


def initialize_context_structure(project_root: Path) -> Dict[str, Any]:
    """Initialize the complete .pmkit/context directory structure.
    
    Creates directories and template YAML files with helpful comments.
    
    Args:
        project_root: Root directory of the project
        
    Returns:
        Dictionary with initialization results:
        - success: True if all operations succeeded
        - created_dirs: List of created directories
        - created_files: List of created template files
        - errors: List of any errors encountered
    """
    result = {
        "success": True,
        "created_dirs": [],
        "created_files": [],
        "errors": []
    }
    
    # Create main directories
    pmkit_dir = project_root / ".pmkit"
    context_dir = pmkit_dir / "context"
    history_dir = context_dir / "history"
    
    for directory in [pmkit_dir, context_dir, history_dir]:
        try:
            if ensure_directory(directory):
                result["created_dirs"].append(str(directory))
        except Exception as e:
            result["success"] = False
            result["errors"].append(f"Failed to create {directory}: {e}")
    
    # Create .gitignore in context directory
    gitignore_path = context_dir / ".gitignore"
    try:
        if not gitignore_path.exists():
            gitignore_content = """# Ignore history directory (contains backups)
history/

# Ignore backup files
*.backup

# Ignore temporary files
*.tmp
.version.tmp
"""
            safe_write(gitignore_path, gitignore_content, backup=False)
            result["created_files"].append(str(gitignore_path))
    except (PermissionError, OSError) as e:
        result["success"] = False
        result["errors"].append(f"Failed to create .gitignore: {e}")
    
    # Create template YAML files if they don't exist
    templates = {
        "company.yaml": _get_company_template(),
        "product.yaml": _get_product_template(),
        "market.yaml": _get_market_template(),
        "team.yaml": _get_team_template(),
        "okrs.yaml": _get_okrs_template()
    }
    
    for filename, template_data in templates.items():
        file_path = context_dir / filename
        try:
            if not file_path.exists():
                # Write YAML with helpful comments
                yaml_content = yaml.dump(
                    template_data,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True
                )
                
                # Add header comment based on file
                header = _get_file_header(filename)
                full_content = header + yaml_content
                
                safe_write(file_path, full_content, backup=False)
                result["created_files"].append(str(file_path))
        except (PermissionError, OSError) as e:
            result["success"] = False
            result["errors"].append(f"Failed to create {filename}: {e}")
    
    return result


def _get_file_header(filename: str) -> str:
    """Get the header comment for a template file."""
    headers = {
        "company.yaml": """# Company Context Configuration
# This file defines your company's basic information for PM-Kit
# Required fields: name, type, stage
# Optional fields: domain, description, target_market

""",
        "product.yaml": """# Product Context Configuration
# This file defines your product details for PM-Kit
# Required fields: name, description
# Optional fields: stage, users, pricing_model, main_metric

""",
        "market.yaml": """# Market Context Configuration
# This file defines your market positioning
# All fields are optional but recommended

""",
        "team.yaml": """# Team Context Configuration
# This file defines your team structure
# Optional but helps tailor PRDs to your team size

""",
        "okrs.yaml": """# OKRs Context Configuration
# This file defines your current objectives and key results
# Used to align PRDs with company goals

"""
    }
    return headers.get(filename, "")


def _get_company_template() -> dict:
    """Get template data for company.yaml."""
    return {
        "name": "YourCompany",
        "type": "b2b",  # or "b2c"
        "stage": "seed",  # seed, growth, scale, enterprise
        "domain": "example.com",
        "description": "Brief description of what your company does",
        "target_market": "Your target market or customer segment"
    }


def _get_product_template() -> dict:
    """Get template data for product.yaml."""
    return {
        "name": "YourProduct",
        "description": "What your product does and the problem it solves",
        "stage": "idea",  # idea, prototype, beta, pmf, growth, scale
        "users": 100,  # Number of active users
        "pricing_model": "subscription",  # subscription, one-time, usage-based, freemium
        "main_metric": "MRR"  # MRR, ARR, DAU, MAU, retention, etc.
    }


def _get_market_template() -> dict:
    """Get template data for market.yaml."""
    return {
        "competitors": [
            "Competitor1",
            "Competitor2"
        ],
        "market_size": "$1B TAM",
        "differentiator": "What makes your product unique"
    }


def _get_team_template() -> dict:
    """Get template data for team.yaml."""
    return {
        "size": 10,
        "roles": {
            "engineers": 6,
            "designers": 2,
            "pm": 1,
            "sales": 1
        }
    }


def _get_okrs_template() -> dict:
    """Get template data for okrs.yaml."""
    return {
        "quarter": "Q1 2025",
        "objectives": [
            {
                "title": "Achieve product-market fit",
                "key_results": [
                    {
                        "description": "Reach $10K MRR",
                        "target_value": "$10K",
                        "current_value": "$2K",
                        "confidence": 70
                    },
                    {
                        "description": "Get 50 paying customers",
                        "target_value": "50",
                        "current_value": "10",
                        "confidence": 80
                    }
                ]
            }
        ]
    }


def check_context_structure(project_root: Path) -> Dict[str, Any]:
    """Check the status of context directory structure.
    
    Args:
        project_root: Root directory of the project
        
    Returns:
        Dictionary with structure status:
        - initialized: True if .pmkit/context exists
        - has_templates: True if template files exist
        - missing_dirs: List of missing directories
        - missing_files: List of missing template files
    """
    pmkit_dir = project_root / ".pmkit"
    context_dir = pmkit_dir / "context"
    history_dir = context_dir / "history"
    
    status = {
        "initialized": context_dir.exists(),
        "has_templates": False,
        "missing_dirs": [],
        "missing_files": []
    }
    
    # Check directories
    for directory in [pmkit_dir, context_dir, history_dir]:
        if not directory.exists():
            status["missing_dirs"].append(str(directory))
    
    # Check template files
    template_files = ["company.yaml", "product.yaml", "market.yaml", "team.yaml", "okrs.yaml"]
    templates_found = 0
    
    for filename in template_files:
        file_path = context_dir / filename
        if not file_path.exists():
            status["missing_files"].append(str(file_path))
        else:
            templates_found += 1
    
    status["has_templates"] = templates_found > 0
    
    return status


def repair_context_structure(project_root: Path) -> Dict[str, Any]:
    """Repair missing parts of context structure.
    
    Creates any missing directories or template files without
    overwriting existing ones.
    
    Args:
        project_root: Root directory of the project
        
    Returns:
        Dictionary with repair results (same format as initialize_context_structure)
    """
    # Check what's missing
    status = check_context_structure(project_root)
    
    # If everything exists, nothing to do
    if status["initialized"] and not status["missing_dirs"] and not status["missing_files"]:
        return {
            "success": True,
            "created_dirs": [],
            "created_files": [],
            "errors": [],
            "message": "Context structure is already complete"
        }
    
    # Otherwise, initialize (it won't overwrite existing files)
    return initialize_context_structure(project_root)