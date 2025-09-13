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