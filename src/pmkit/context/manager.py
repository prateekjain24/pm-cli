"""Context manager for persisting PM-Kit context to disk.

Simple YAML-based persistence without over-engineering.
Handles atomic writes and automatic backups.
"""

import shutil
from pathlib import Path
from typing import Optional, Tuple, List
import yaml

from pmkit.context.models import (
    CompanyContext,
    Context,
    MarketContext,
    OKRContext,
    ProductContext,
    TeamContext,
)
from pmkit.context.version import ContextVersion
from pmkit.context.validator import ContextValidator, ValidationError


class ContextManager:
    """Manages context persistence to/from YAML files.
    
    Simple file-based storage with atomic writes and backups.
    """
    
    def __init__(self, context_dir: Path, validate: bool = True):
        """Initialize context manager.
        
        Args:
            context_dir: Directory to store context files (.pmkit/context)
            validate: Whether to validate context on save/load (default: True)
        """
        self.context_dir = Path(context_dir)
        self.context_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure history directory exists for backups
        self.history_dir = self.context_dir / "history"
        self.history_dir.mkdir(exist_ok=True)
        
        # Initialize validator
        self.validate = validate
        self.validator = ContextValidator() if validate else None
    
    def save_context(self, context: Context, auto_repair: bool = False) -> Tuple[bool, List[ValidationError]]:
        """Save complete context to disk.
        
        Saves each component to its own YAML file with atomic writes.
        Creates backups before overwriting existing files.
        Validates context before saving if validation is enabled.
        
        Args:
            context: Context object to save
            auto_repair: Whether to attempt auto-repair of validation warnings
            
        Returns:
            Tuple of (success, validation_errors)
        """
        # Validate if enabled
        if self.validate:
            is_valid, errors = self.validator.validate(context)
            
            # Try auto-repair if requested and possible
            if auto_repair and errors:
                if self.validator.can_auto_repair(context, errors):
                    context = self.validator.auto_repair(context, errors)
                    # Re-validate after repair
                    is_valid, errors = self.validator.validate(context)
            
            # Don't save if there are errors (warnings are OK)
            has_errors = any(e.severity == "error" for e in errors)
            if has_errors:
                return (False, errors)
        else:
            errors = []
        
        # Save each context component
        self._save_component("company.yaml", context.company)
        self._save_component("product.yaml", context.product)
        
        if context.market:
            self._save_component("market.yaml", context.market)
        if context.team:
            self._save_component("team.yaml", context.team)
        if context.okrs:
            self._save_component("okrs.yaml", context.okrs)
        
        # Update version hash
        ContextVersion.save_current(self.context_dir)
        
        return (True, errors)
    
    def load_context(self, validate: bool = True) -> Tuple[Optional[Context], List[ValidationError]]:
        """Load complete context from disk.
        
        Args:
            validate: Whether to validate loaded context
        
        Returns:
            Tuple of (context, validation_errors)
            Context is None if required files don't exist
        """
        # Check if required files exist
        company_file = self.context_dir / "company.yaml"
        product_file = self.context_dir / "product.yaml"
        
        if not company_file.exists() or not product_file.exists():
            return (None, [])
        
        # Load required components
        company = self._load_component("company.yaml", CompanyContext)
        product = self._load_component("product.yaml", ProductContext)
        
        if not company or not product:
            return (None, [ValidationError("context", "Failed to load required context files")])
        
        # Load optional components
        market = self._load_component("market.yaml", MarketContext)
        team = self._load_component("team.yaml", TeamContext)
        okrs = self._load_component("okrs.yaml", OKRContext)
        
        context = Context(
            company=company,
            product=product,
            market=market,
            team=team,
            okrs=okrs
        )
        
        # Validate if requested
        errors = []
        if validate and self.validate:
            _, errors = self.validator.validate(context)
        
        return (context, errors)
    
    def save_company(self, company: CompanyContext) -> None:
        """Save just the company context.
        
        Args:
            company: CompanyContext to save
        """
        self._save_component("company.yaml", company)
        ContextVersion.save_current(self.context_dir)
    
    def save_product(self, product: ProductContext) -> None:
        """Save just the product context.
        
        Args:
            product: ProductContext to save
        """
        self._save_component("product.yaml", product)
        ContextVersion.save_current(self.context_dir)
    
    def save_market(self, market: MarketContext) -> None:
        """Save just the market context.
        
        Args:
            market: MarketContext to save
        """
        self._save_component("market.yaml", market)
        ContextVersion.save_current(self.context_dir)
    
    def save_team(self, team: TeamContext) -> None:
        """Save just the team context.
        
        Args:
            team: TeamContext to save
        """
        self._save_component("team.yaml", team)
        ContextVersion.save_current(self.context_dir)
    
    def save_okrs(self, okrs: OKRContext) -> None:
        """Save just the OKR context.
        
        Args:
            okrs: OKRContext to save
        """
        self._save_component("okrs.yaml", okrs)
        ContextVersion.save_current(self.context_dir)
    
    def _save_component(self, filename: str, component) -> None:
        """Save a context component to a YAML file atomically.
        
        Uses atomic write (temp file + rename) to prevent corruption.
        Creates backup if file already exists.
        
        Args:
            filename: Name of the YAML file
            component: Pydantic model to save
        """
        file_path = self.context_dir / filename
        temp_path = file_path.with_suffix('.tmp')
        
        # Create backup if file exists
        if file_path.exists():
            self._create_backup(file_path)
        
        # Convert to dict and write to temp file
        data = component.model_dump(exclude_none=True, exclude_unset=True)
        
        with open(temp_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        # Atomic rename (works on POSIX and Windows)
        temp_path.replace(file_path)
    
    def _load_component(self, filename: str, model_class):
        """Load a context component from a YAML file.
        
        Args:
            filename: Name of the YAML file
            model_class: Pydantic model class to instantiate
            
        Returns:
            Instance of model_class or None if file doesn't exist
        """
        file_path = self.context_dir / filename
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
            
            if data is None:
                return None
            
            return model_class(**data)
        except Exception:
            # If we can't load, return None
            # Validator will catch this and report proper error
            return None
    
    def _create_backup(self, file_path: Path) -> None:
        """Create a backup of a file before overwriting.
        
        Args:
            file_path: Path to the file to backup
        """
        if not file_path.exists():
            return
        
        # Simple backup with .backup extension
        # For MVP, just keep one backup per file
        backup_path = file_path.with_suffix(file_path.suffix + '.backup')
        shutil.copy2(file_path, backup_path)
    
    def context_exists(self) -> bool:
        """Check if a valid context exists on disk.
        
        Returns:
            True if at least company and product files exist
        """
        company_file = self.context_dir / "company.yaml"
        product_file = self.context_dir / "product.yaml"
        return company_file.exists() and product_file.exists()
    
    def get_context_summary(self) -> dict:
        """Get a summary of what context files exist.
        
        Returns:
            Dictionary with file existence status
        """
        files = ["company.yaml", "product.yaml", "market.yaml", "team.yaml", "okrs.yaml"]
        summary = {}
        
        for filename in files:
            file_path = self.context_dir / filename
            summary[filename.replace('.yaml', '')] = file_path.exists()
        
        # Add version info
        summary['version'] = ContextVersion.load_stored(self.context_dir)
        
        return summary
    
    def get_initialization_status(self) -> dict:
        """Get detailed initialization status of the context.
        
        Checks for context completeness, validation status, and provides
        actionable information about what needs to be fixed.
        
        Returns:
            Dictionary with initialization status:
            - status: 'not_initialized', 'partial', 'complete', or 'complete_with_warnings'
            - is_valid: True if context passes validation
            - required_missing: List of missing required files
            - optional_missing: List of missing optional files
            - validation_errors: List of validation errors if any
            - suggestions: List of suggestions to fix issues
        """
        result = {
            "status": "not_initialized",
            "is_valid": False,
            "required_missing": [],
            "optional_missing": [],
            "validation_errors": [],
            "suggestions": []
        }
        
        # Check if context directory exists
        if not self.context_dir.exists():
            result["status"] = "not_initialized"
            result["suggestions"].append("Run 'pm init' to initialize your project context")
            return result
        
        # Check required files (company and product)
        company_file = self.context_dir / "company.yaml"
        product_file = self.context_dir / "product.yaml"
        
        if not company_file.exists():
            result["required_missing"].append("company.yaml")
        if not product_file.exists():
            result["required_missing"].append("product.yaml")
        
        # Check optional files
        optional_files = ["market.yaml", "team.yaml", "okrs.yaml"]
        for filename in optional_files:
            if not (self.context_dir / filename).exists():
                result["optional_missing"].append(filename)
        
        # Determine status based on what's missing
        if result["required_missing"]:
            result["status"] = "partial"
            result["suggestions"].append(
                f"Missing required files: {', '.join(result['required_missing'])}. "
                "Run 'pm init' to complete initialization."
            )
        else:
            # Try to load and validate the context
            context, validation_errors = self.load_context(validate=True)
            
            if context is None:
                result["status"] = "partial"
                result["suggestions"].append(
                    "Context files exist but cannot be loaded. Check file format and content."
                )
            else:
                # Context loads successfully
                if validation_errors:
                    # Check if there are errors or just warnings
                    has_errors = any(e.severity == "error" for e in validation_errors)
                    
                    if has_errors:
                        result["status"] = "partial"
                        result["is_valid"] = False
                        result["validation_errors"] = [
                            f"{e.field}: {e.message}" for e in validation_errors
                            if e.severity == "error"
                        ]
                        result["suggestions"].append(
                            "Fix validation errors in your context files"
                        )
                    else:
                        # Only warnings
                        result["status"] = "complete_with_warnings"
                        result["is_valid"] = True
                        result["validation_errors"] = [
                            f"{e.field}: {e.message}" for e in validation_errors
                        ]
                        result["suggestions"].append(
                            "Context is complete but has warnings. Consider addressing them for better results."
                        )
                else:
                    # No validation issues
                    result["status"] = "complete"
                    result["is_valid"] = True
                    
                    # Suggest adding optional files if missing
                    if result["optional_missing"]:
                        result["suggestions"].append(
                            f"Consider adding optional context files for better PRD generation: "
                            f"{', '.join(result['optional_missing'])}"
                        )
        
        return result
    
    def repair_context(self, auto_fix: bool = True) -> dict:
        """Attempt to repair context issues.
        
        This method can:
        - Create missing directories
        - Create template files for missing context
        - Fix validation warnings where possible
        
        Args:
            auto_fix: If True, attempts to auto-fix validation issues
            
        Returns:
            Dictionary with repair results:
            - success: True if all repairs succeeded
            - actions_taken: List of actions performed
            - remaining_issues: List of issues that couldn't be fixed
        """
        from pmkit.context.structure import repair_context_structure
        
        result = {
            "success": True,
            "actions_taken": [],
            "remaining_issues": []
        }
        
        # First, ensure directory structure is complete
        structure_result = repair_context_structure(self.context_dir.parent.parent)
        if structure_result.get("created_dirs") or structure_result.get("created_files"):
            result["actions_taken"].append("Repaired directory structure")
            for d in structure_result.get("created_dirs", []):
                result["actions_taken"].append(f"Created directory: {d}")
            for f in structure_result.get("created_files", []):
                result["actions_taken"].append(f"Created template: {f}")
        
        # Try to load and validate context
        context, validation_errors = self.load_context(validate=True)
        
        if context and auto_fix:
            # Attempt auto-repair of validation issues
            if self.validator and self.validator.can_auto_repair(context, validation_errors):
                repaired_context = self.validator.auto_repair(context, validation_errors)
                success, remaining_errors = self.save_context(repaired_context)
                
                if success:
                    result["actions_taken"].append("Auto-repaired validation warnings")
                else:
                    result["success"] = False
                    result["remaining_issues"].extend([
                        f"{e.field}: {e.message}" for e in remaining_errors
                    ])
        elif validation_errors:
            # Can't auto-fix, report issues
            result["remaining_issues"].extend([
                f"{e.field}: {e.message}" for e in validation_errors
                if e.severity == "error"
            ])
            if result["remaining_issues"]:
                result["success"] = False
        
        return result