"""Context schema migration support for PM-Kit.

Handles version compatibility and migrations between context schema versions.
Simple implementation without over-engineering - just tracks version and 
provides hooks for future migrations.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

# Current schema version - increment when breaking changes are made
SCHEMA_VERSION = "1.0.0"


class ContextMigrator:
    """Handles context schema migrations.
    
    MVP implementation that tracks schema version and provides
    framework for future migrations when needed.
    """
    
    # Migration paths - maps from_version -> to_version
    # Empty for now since we're at v1.0.0
    MIGRATIONS: Dict[str, str] = {
        # Example: "0.9.0": "1.0.0" would define a migration path
    }
    
    @staticmethod
    def get_schema_version(context_dir: Path) -> Optional[str]:
        """Get the schema version of a context directory.
        
        Args:
            context_dir: Directory containing context files
            
        Returns:
            Schema version string or None if not found
        """
        version_file = context_dir / ".schema_version"
        if version_file.exists():
            return version_file.read_text().strip()
        return None
    
    @staticmethod
    def save_schema_version(context_dir: Path, version: str = SCHEMA_VERSION) -> None:
        """Save the schema version to the context directory.
        
        Args:
            context_dir: Directory containing context files
            version: Schema version to save (defaults to current)
        """
        version_file = context_dir / ".schema_version"
        version_file.write_text(version)
    
    def check_compatibility(self, context_dir: Path) -> Tuple[bool, Optional[str]]:
        """Check if context schema version is compatible.
        
        Args:
            context_dir: Directory containing context files
            
        Returns:
            Tuple of (is_compatible, migration_message)
        """
        stored_version = self.get_schema_version(context_dir)
        
        # No version file means it's a new context or pre-versioning context
        # Assume it's compatible (will be updated on next save)
        if stored_version is None:
            return True, None
        
        # Same version = fully compatible
        if stored_version == SCHEMA_VERSION:
            return True, None
        
        # Check if migration path exists
        if stored_version in self.MIGRATIONS:
            target_version = self.MIGRATIONS[stored_version]
            return False, f"Context needs migration from v{stored_version} to v{target_version}"
        
        # Unknown version - might be newer or incompatible
        stored_parts = stored_version.split(".")
        current_parts = SCHEMA_VERSION.split(".")
        
        # Major version mismatch = incompatible
        if stored_parts[0] != current_parts[0]:
            return False, f"Incompatible context version v{stored_version} (current: v{SCHEMA_VERSION})"
        
        # Minor/patch difference = compatible (backward compatible)
        return True, f"Context version v{stored_version} is compatible with v{SCHEMA_VERSION}"
    
    def needs_migration(self, context_dir: Path) -> bool:
        """Check if context needs migration.
        
        Args:
            context_dir: Directory containing context files
            
        Returns:
            True if migration is needed
        """
        is_compatible, _ = self.check_compatibility(context_dir)
        return not is_compatible
    
    def migrate(self, context_dir: Path) -> Tuple[bool, str]:
        """Perform migration if needed.
        
        Args:
            context_dir: Directory containing context files
            
        Returns:
            Tuple of (success, message)
        """
        stored_version = self.get_schema_version(context_dir)
        
        # No migration needed
        if stored_version == SCHEMA_VERSION:
            return True, "Context is already at the current version"
        
        # No version file - just add it
        if stored_version is None:
            self.save_schema_version(context_dir)
            return True, f"Added schema version v{SCHEMA_VERSION} to context"
        
        # Check if migration path exists
        if stored_version in self.MIGRATIONS:
            # In future, actual migration logic would go here
            # For now, just update the version
            self.save_schema_version(context_dir)
            return True, f"Migrated context from v{stored_version} to v{SCHEMA_VERSION}"
        
        # No migration path available
        return False, f"No migration path from v{stored_version} to v{SCHEMA_VERSION}"