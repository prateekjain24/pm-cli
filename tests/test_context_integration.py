"""Integration tests for the context system.

Tests the complete context system working together end-to-end,
focusing on save/load roundtrip, version changes, validation, and backups.
"""

import shutil
from pathlib import Path

import pytest
import yaml

from pmkit.context.manager import ContextManager
from pmkit.context.models import (
    CompanyContext,
    Context,
    KeyResult,
    MarketContext,
    OKRContext,
    Objective,
    ProductContext,
    TeamContext,
)
from pmkit.context.structure import initialize_context_structure
from pmkit.context.version import ContextVersion


@pytest.fixture
def full_context():
    """Create a complete context with all components for testing."""
    return Context(
        company=CompanyContext(
            name="IntegrationCo",
            type="b2b",
            stage="growth",
            domain="integration.com",
            description="A company for integration testing",
            target_market="Enterprise SaaS"
        ),
        product=ProductContext(
            name="IntegrationProduct",
            description="A comprehensive product for testing all context features",
            stage="pmf",
            users=5000,
            pricing_model="subscription",
            main_metric="MRR"
        ),
        market=MarketContext(
            competitors=["Competitor1", "Competitor2", "Competitor3"],
            market_size="$5B TAM",
            differentiator="Superior integration testing capabilities"
        ),
        team=TeamContext(
            size=50,
            roles={
                "engineers": 30,
                "designers": 5,
                "pm": 3,
                "sales": 8,
                "marketing": 4
            }
        ),
        okrs=OKRContext(
            quarter="Q1 2025",
            objectives=[
                Objective(
                    title="Achieve product-market fit",
                    key_results=[
                        KeyResult(
                            description="Reach $500K MRR",
                            target_value="$500K",
                            current_value="$250K",
                            confidence=75
                        ),
                        KeyResult(
                            description="Get 100 enterprise customers",
                            target_value="100",
                            current_value="45",
                            confidence=60
                        )
                    ]
                )
            ]
        )
    )


class TestContextIntegration:
    """Integration tests for the context system."""
    
    def test_full_context_roundtrip(self, tmp_path, full_context):
        """Test complete save/load roundtrip with all context components."""
        # Initialize structure
        initialize_context_structure(tmp_path)
        context_dir = tmp_path / ".pmkit" / "context"
        
        # Create manager and save context
        manager = ContextManager(context_dir, validate=False)
        success, errors = manager.save_context(full_context)
        assert success is True
        assert len(errors) == 0
        
        # Load context back
        loaded_context, load_errors = manager.load_context(validate=False)
        assert loaded_context is not None
        assert len(load_errors) == 0
        
        # Verify all components match
        assert loaded_context.company.name == full_context.company.name
        assert loaded_context.company.type == full_context.company.type
        assert loaded_context.company.domain == full_context.company.domain
        
        assert loaded_context.product.name == full_context.product.name
        assert loaded_context.product.users == full_context.product.users
        assert loaded_context.product.main_metric == full_context.product.main_metric
        
        assert loaded_context.market.competitors == full_context.market.competitors
        assert loaded_context.market.market_size == full_context.market.market_size
        
        assert loaded_context.team.size == full_context.team.size
        assert loaded_context.team.roles == full_context.team.roles
        
        assert loaded_context.okrs.quarter == full_context.okrs.quarter
        assert len(loaded_context.okrs.objectives) == len(full_context.okrs.objectives)
    
    def test_version_changes_on_update(self, tmp_path):
        """Test that version hash changes when context is updated."""
        # Initialize and create manager
        initialize_context_structure(tmp_path)
        context_dir = tmp_path / ".pmkit" / "context"
        manager = ContextManager(context_dir, validate=False)
        
        # Create and save initial context
        initial_context = Context(
            company=CompanyContext(name="VersionCo", type="b2b", stage="seed"),
            product=ProductContext(
                name="VersionProduct",
                description="Product for version testing"
            )
        )
        
        success, _ = manager.save_context(initial_context)
        assert success is True
        
        # Get initial version
        initial_version = ContextVersion.load_stored(context_dir)
        assert initial_version is not None
        
        # Update context
        updated_context = Context(
            company=CompanyContext(name="UpdatedVersionCo", type="b2c", stage="growth"),
            product=ProductContext(
                name="VersionProduct",
                description="Product for version testing"
            )
        )
        
        success, _ = manager.save_context(updated_context)
        assert success is True
        
        # Get new version
        new_version = ContextVersion.load_stored(context_dir)
        assert new_version is not None
        assert new_version != initial_version
        
        # Verify version changes are detected
        assert ContextVersion.has_changed(initial_version, new_version) is True
    
    def test_validation_blocks_invalid_save(self, tmp_path):
        """Test that validation prevents saving invalid context."""
        # Don't initialize structure - we want a clean directory
        context_dir = tmp_path / ".pmkit" / "context"
        context_dir.mkdir(parents=True)
        manager = ContextManager(context_dir, validate=True)
        
        # Create context with validation errors (negative team size)
        invalid_context = Context(
            company=CompanyContext(name="InvalidCo", type="b2b", stage="growth"),
            product=ProductContext(
                name="InvalidProduct",
                description="Product with validation errors"
            ),
            team=TeamContext(
                roles={"engineers": -10}  # Error: negative value
            )
        )
        
        # Try to save - should fail
        success, errors = manager.save_context(invalid_context)
        assert success is False
        assert len(errors) > 0
        assert any(e.severity == "error" for e in errors)
        
        # Verify files were not created (no real context saved, only templates if any)
        # Check that no actual context was saved by trying to load
        loaded, _ = manager.load_context(validate=False)
        assert loaded is None  # Nothing should load since save failed
    
    def test_backup_created_on_update(self, tmp_path):
        """Test that backup files are created when updating existing context."""
        # Create directories manually without templates
        context_dir = tmp_path / ".pmkit" / "context"
        context_dir.mkdir(parents=True)
        (context_dir / "history").mkdir()
        manager = ContextManager(context_dir, validate=False)
        
        # Save initial context
        initial_context = Context(
            company=CompanyContext(name="BackupCo", type="b2b", stage="seed"),
            product=ProductContext(
                name="BackupProduct",
                description="Product for backup testing"
            )
        )
        
        success, _ = manager.save_context(initial_context)
        assert success is True
        
        # Verify no backup files yet (first save, no backups)
        assert not (context_dir / "company.yaml.backup").exists()
        assert not (context_dir / "product.yaml.backup").exists()
        
        # Update context
        updated_context = Context(
            company=CompanyContext(name="UpdatedBackupCo", type="b2c", stage="growth"),
            product=ProductContext(
                name="UpdatedBackupProduct",
                description="Updated product for backup testing"
            )
        )
        
        success, _ = manager.save_context(updated_context)
        assert success is True
        
        # Verify backup files were created
        assert (context_dir / "company.yaml.backup").exists()
        assert (context_dir / "product.yaml.backup").exists()
        
        # Verify backup contains original content
        with open(context_dir / "company.yaml.backup") as f:
            backup_data = yaml.safe_load(f)
            assert backup_data["name"] == "BackupCo"
        
        # Verify current file has new content
        with open(context_dir / "company.yaml") as f:
            current_data = yaml.safe_load(f)
            assert current_data["name"] == "UpdatedBackupCo"
    
    def test_context_repair_roundtrip(self, tmp_path):
        """Test that auto-repair works during save/load roundtrip."""
        # Initialize and create manager with validation
        initialize_context_structure(tmp_path)
        context_dir = tmp_path / ".pmkit" / "context"
        manager = ContextManager(context_dir, validate=True)
        
        # Create context with repairable issues
        context_with_issues = Context(
            company=CompanyContext(name="RepairCo", type="b2b", stage="growth"),
            product=ProductContext(
                name="RepairProduct",
                description="Product that needs automatic repairs",
                stage="pmf",
                main_metric=None  # Should be auto-set to MRR for B2B
            ),
            team=TeamContext(
                size=100,  # Wrong size, should match sum of roles
                roles={"engineers": 20, "sales": 10, "pm": 2}
            )
        )
        
        # Save with auto-repair enabled
        success, errors = manager.save_context(context_with_issues, auto_repair=True)
        assert success is True
        # May have warnings but no errors
        assert all(e.severity == "warning" for e in errors if e.severity == "error")
        
        # Load back and verify repairs were applied
        loaded_context, _ = manager.load_context(validate=False)
        assert loaded_context is not None
        
        # Check repairs
        assert loaded_context.product.main_metric == "MRR"  # Auto-set for B2B
        assert loaded_context.team.size == 32  # Fixed to match sum of roles (20+10+2)