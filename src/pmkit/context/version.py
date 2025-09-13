"""Context versioning using content-based SHA256 hashing.

Minimal implementation that tracks context changes without overengineering.
Version = SHA256(company.yaml + product.yaml + market.yaml + team.yaml + okrs.yaml)
"""

import hashlib
from pathlib import Path
from typing import Optional


class ContextVersion:
    """Handles context versioning using content-based hashing.
    
    MVP implementation - just tracks if context changed, no history.
    """
    
    CONTEXT_FILES = [
        "company.yaml",
        "product.yaml", 
        "market.yaml",
        "team.yaml",
        "okrs.yaml"
    ]
    
    @staticmethod
    def compute_hash(context_dir: Path) -> str:
        """Compute SHA256 hash of all context files.
        
        Args:
            context_dir: Directory containing context YAML files
            
        Returns:
            SHA256 hash of concatenated context files
        """
        hasher = hashlib.sha256()
        
        # Process files in deterministic order
        for filename in ContextVersion.CONTEXT_FILES:
            file_path = context_dir / filename
            
            if file_path.exists():
                # Read file content and update hash
                content = file_path.read_bytes()
                hasher.update(content)
            else:
                # Use empty marker for missing files (deterministic)
                hasher.update(b"__MISSING__")
        
        return hasher.hexdigest()
    
    @staticmethod
    def has_changed(old_hash: str, new_hash: str) -> bool:
        """Check if context has changed by comparing hashes.
        
        Args:
            old_hash: Previous context hash
            new_hash: Current context hash
            
        Returns:
            True if hashes differ, False otherwise
        """
        return old_hash != new_hash
    
    @staticmethod
    def get_current(context_dir: Path) -> str:
        """Get current context version hash.
        
        Args:
            context_dir: Directory containing context files
            
        Returns:
            Current SHA256 hash
        """
        return ContextVersion.compute_hash(context_dir)
    
    @staticmethod
    def load_stored(context_dir: Path) -> Optional[str]:
        """Load previously stored version hash.
        
        Args:
            context_dir: Directory containing context files
            
        Returns:
            Stored hash or None if not found
        """
        version_file = context_dir / ".version"
        if version_file.exists():
            return version_file.read_text().strip()
        return None
    
    @staticmethod
    def save_current(context_dir: Path) -> str:
        """Compute and save current version hash.
        
        Args:
            context_dir: Directory containing context files
            
        Returns:
            Current hash that was saved
        """
        current_hash = ContextVersion.compute_hash(context_dir)
        version_file = context_dir / ".version"
        version_file.write_text(current_hash)
        return current_hash