"""Tests for context schema migration functionality."""

from pathlib import Path

import pytest
import yaml

from pmkit.context.migration import ContextMigrator, SCHEMA_VERSION
from pmkit.context.manager import ContextManager
from pmkit.context.models import Context, CompanyContext, ProductContext
from pmkit.context.validator import ContextValidator


class TestContextMigrator:
    """Test ContextMigrator functionality."""
    
    def test_get_schema_version_no_file(self, tmp_path):
        """Test getting schema version when file doesn't exist."""
        migrator = ContextMigrator()
        version = migrator.get_schema_version(tmp_path)
        assert version is None
    
    def test_save_and_get_schema_version(self, tmp_path):
        """Test saving and retrieving schema version."""
        migrator = ContextMigrator()
        
        # Save version
        migrator.save_schema_version(tmp_path, "1.0.0")
        
        # Get version
        version = migrator.get_schema_version(tmp_path)
        assert version == "1.0.0"
        
        # Verify file contents
        version_file = tmp_path / ".schema_version"
        assert version_file.exists()
        assert version_file.read_text() == "1.0.0"
    
    def test_check_compatibility_no_version(self, tmp_path):
        """Test compatibility check when no version file exists."""
        migrator = ContextMigrator()
        is_compatible, message = migrator.check_compatibility(tmp_path)
        
        # Should be compatible (assumes current version)
        assert is_compatible is True
        assert message is None
    
    def test_check_compatibility_same_version(self, tmp_path):
        """Test compatibility when versions match."""
        migrator = ContextMigrator()
        migrator.save_schema_version(tmp_path, SCHEMA_VERSION)
        
        is_compatible, message = migrator.check_compatibility(tmp_path)
        assert is_compatible is True
        assert message is None
    
    def test_check_compatibility_minor_difference(self, tmp_path):
        """Test compatibility with minor version difference."""
        migrator = ContextMigrator()
        
        # Save older minor version (backward compatible)
        migrator.save_schema_version(tmp_path, "1.0.0")
        
        # Temporarily modify SCHEMA_VERSION for testing
        import pmkit.context.migration
        original_version = pmkit.context.migration.SCHEMA_VERSION
        pmkit.context.migration.SCHEMA_VERSION = "1.1.0"
        
        try:
            is_compatible, message = migrator.check_compatibility(tmp_path)
            # Minor version difference should be compatible
            assert is_compatible is True
            assert "compatible" in message.lower()
        finally:
            pmkit.context.migration.SCHEMA_VERSION = original_version
    
    def test_check_compatibility_major_difference(self, tmp_path):
        """Test incompatibility with major version difference."""
        migrator = ContextMigrator()
        
        # Save old major version
        migrator.save_schema_version(tmp_path, "0.9.0")
        
        is_compatible, message = migrator.check_compatibility(tmp_path)
        # Major version difference = incompatible
        assert is_compatible is False
        assert "incompatible" in message.lower()
    
    def test_needs_migration(self, tmp_path):
        """Test migration detection."""
        migrator = ContextMigrator()
        
        # No version file - no migration needed
        assert migrator.needs_migration(tmp_path) is False
        
        # Same version - no migration needed
        migrator.save_schema_version(tmp_path, SCHEMA_VERSION)
        assert migrator.needs_migration(tmp_path) is False
        
        # Different major version - migration needed
        migrator.save_schema_version(tmp_path, "0.9.0")
        assert migrator.needs_migration(tmp_path) is True
    
    def test_migrate_no_version_file(self, tmp_path):
        """Test migration when no version file exists."""
        migrator = ContextMigrator()
        
        success, message = migrator.migrate(tmp_path)
        assert success is True
        assert "added schema version" in message.lower()
        
        # Verify version file was created
        assert migrator.get_schema_version(tmp_path) == SCHEMA_VERSION
    
    def test_migrate_already_current(self, tmp_path):
        """Test migration when already at current version."""
        migrator = ContextMigrator()
        migrator.save_schema_version(tmp_path, SCHEMA_VERSION)
        
        success, message = migrator.migrate(tmp_path)
        assert success is True
        assert "already at the current version" in message.lower()


class TestValidatorVersionCheck:
    """Test version compatibility checking in validator."""
    
    def test_validate_version_compatibility_current(self, tmp_path):
        """Test validation with current version."""
        validator = ContextValidator()
        migrator = ContextMigrator()
        migrator.save_schema_version(tmp_path, SCHEMA_VERSION)
        
        is_compatible, errors = validator.validate_version_compatibility(tmp_path)
        assert is_compatible is True
        assert len(errors) == 0
    
    def test_validate_version_compatibility_incompatible(self, tmp_path):
        """Test validation with incompatible version."""
        validator = ContextValidator()
        migrator = ContextMigrator()
        migrator.save_schema_version(tmp_path, "0.5.0")
        
        is_compatible, errors = validator.validate_version_compatibility(tmp_path)
        assert is_compatible is False
        assert len(errors) > 0
        assert errors[0].severity == "error"
        assert "incompatible" in errors[0].message.lower()


class TestManagerMigration:
    """Test migration integration in ContextManager."""
    
    def test_save_context_adds_schema_version(self, tmp_path):
        """Test that saving context adds schema version."""
        context_dir = tmp_path / ".pmkit" / "context"
        context_dir.mkdir(parents=True)
        
        manager = ContextManager(context_dir, validate=False)
        
        context = Context(
            company=CompanyContext(name="TestCo", type="b2b", stage="seed"),
            product=ProductContext(name="TestProduct", description="A test product")
        )
        
        success, _ = manager.save_context(context)
        assert success is True
        
        # Check schema version was saved
        version_file = context_dir / ".schema_version"
        assert version_file.exists()
        assert version_file.read_text().strip() == SCHEMA_VERSION
    
    def test_load_context_checks_compatibility(self, tmp_path):
        """Test that loading context checks version compatibility."""
        context_dir = tmp_path / ".pmkit" / "context"
        context_dir.mkdir(parents=True)
        
        # Create context files
        (context_dir / "company.yaml").write_text(
            yaml.dump({"name": "TestCo", "type": "b2b", "stage": "seed"})
        )
        (context_dir / "product.yaml").write_text(
            yaml.dump({"name": "TestProduct", "description": "A test product"})
        )
        
        # Add incompatible version
        (context_dir / ".schema_version").write_text("0.5.0")
        
        manager = ContextManager(context_dir, validate=True)
        context, errors = manager.load_context(validate=True)
        
        # Should fail to load due to incompatibility
        assert context is None
        assert len(errors) > 0
        assert any("migration" in str(e).lower() for e in errors)
    
    def test_migrate_context_success(self, tmp_path):
        """Test successful context migration."""
        context_dir = tmp_path / ".pmkit" / "context"
        context_dir.mkdir(parents=True)
        # Create history dir that ContextManager expects
        (context_dir / "history").mkdir(parents=True)
        
        manager = ContextManager(context_dir, validate=False)
        
        # No version file initially
        success, message = manager.migrate_context()
        print(f"Migration result: success={success}, message={message}")
        assert success is True
        assert "Added schema version" in message or "added schema version" in message.lower()
        
        # Version file should now exist
        version_file = context_dir / ".schema_version"
        assert version_file.exists()
        assert version_file.read_text().strip() == SCHEMA_VERSION
    
    def test_migrate_context_already_current(self, tmp_path):
        """Test migration when already at current version."""
        context_dir = tmp_path / ".pmkit" / "context"
        context_dir.mkdir(parents=True)
        
        # Set current version
        (context_dir / ".schema_version").write_text(SCHEMA_VERSION)
        
        manager = ContextManager(context_dir, validate=False)
        success, message = manager.migrate_context()
        
        assert success is True
        assert "already at the current" in message