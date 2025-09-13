"""Tests for context versioning."""

import tempfile
from pathlib import Path

import pytest

from pmkit.context.version import ContextVersion


class TestContextVersion:
    """Test ContextVersion functionality."""
    
    def test_compute_hash_with_all_files(self, tmp_path):
        """Test hash computation with all context files present."""
        # Create test context files
        (tmp_path / "company.yaml").write_text("name: TestCo\ntype: b2b")
        (tmp_path / "product.yaml").write_text("name: TestProduct")
        (tmp_path / "market.yaml").write_text("competitors: []")
        (tmp_path / "team.yaml").write_text("size: 10")
        (tmp_path / "okrs.yaml").write_text("objectives: []")
        
        # Compute hash
        hash1 = ContextVersion.compute_hash(tmp_path)
        assert hash1 is not None
        assert len(hash1) == 64  # SHA256 produces 64 hex chars
        
        # Same content should produce same hash
        hash2 = ContextVersion.compute_hash(tmp_path)
        assert hash1 == hash2
    
    def test_hash_changes_with_content_change(self, tmp_path):
        """Test that hash changes when content changes."""
        # Create initial files
        company_file = tmp_path / "company.yaml"
        company_file.write_text("name: TestCo")
        (tmp_path / "product.yaml").write_text("name: TestProduct")
        (tmp_path / "market.yaml").write_text("competitors: []")
        (tmp_path / "team.yaml").write_text("size: 10")
        (tmp_path / "okrs.yaml").write_text("objectives: []")
        
        # Get initial hash
        hash_before = ContextVersion.compute_hash(tmp_path)
        
        # Change content
        company_file.write_text("name: UpdatedCo")
        
        # Hash should change
        hash_after = ContextVersion.compute_hash(tmp_path)
        assert hash_before != hash_after
    
    def test_hash_stable_without_changes(self, tmp_path):
        """Test that hash remains stable when content doesn't change."""
        # Create test files
        (tmp_path / "company.yaml").write_text("name: StableCo")
        (tmp_path / "product.yaml").write_text("name: StableProduct")
        (tmp_path / "market.yaml").write_text("competitors: [A, B]")
        (tmp_path / "team.yaml").write_text("size: 5")
        (tmp_path / "okrs.yaml").write_text("objectives: []")
        
        # Compute hash multiple times
        hashes = [ContextVersion.compute_hash(tmp_path) for _ in range(5)]
        
        # All hashes should be identical
        assert len(set(hashes)) == 1
    
    def test_missing_files_handled(self, tmp_path):
        """Test hash computation with missing files."""
        # Create only some files
        (tmp_path / "company.yaml").write_text("name: PartialCo")
        (tmp_path / "product.yaml").write_text("name: PartialProduct")
        # market.yaml, team.yaml, okrs.yaml are missing
        
        # Should still compute hash
        hash1 = ContextVersion.compute_hash(tmp_path)
        assert hash1 is not None
        
        # Adding a missing file should change hash
        (tmp_path / "market.yaml").write_text("competitors: []")
        hash2 = ContextVersion.compute_hash(tmp_path)
        assert hash1 != hash2
    
    def test_deterministic_ordering(self, tmp_path):
        """Test that file processing order is deterministic."""
        # Create files in different order across two directories
        dir1 = tmp_path / "context1"
        dir2 = tmp_path / "context2"
        dir1.mkdir()
        dir2.mkdir()
        
        # Write files in different order
        files_content = {
            "company.yaml": "name: TestCo",
            "product.yaml": "name: TestProduct",
            "market.yaml": "competitors: []",
            "team.yaml": "size: 10",
            "okrs.yaml": "objectives: []"
        }
        
        # Dir1: Write in alphabetical order
        for name, content in sorted(files_content.items()):
            (dir1 / name).write_text(content)
        
        # Dir2: Write in reverse order
        for name, content in sorted(files_content.items(), reverse=True):
            (dir2 / name).write_text(content)
        
        # Hashes should be identical
        hash1 = ContextVersion.compute_hash(dir1)
        hash2 = ContextVersion.compute_hash(dir2)
        assert hash1 == hash2
    
    def test_has_changed(self):
        """Test has_changed method."""
        hash1 = "a" * 64
        hash2 = "b" * 64
        hash3 = "a" * 64
        
        assert ContextVersion.has_changed(hash1, hash2) is True
        assert ContextVersion.has_changed(hash1, hash3) is False
        assert ContextVersion.has_changed(hash1, hash1) is False
    
    def test_get_current(self, tmp_path):
        """Test get_current method."""
        # Create test files
        (tmp_path / "company.yaml").write_text("name: CurrentCo")
        (tmp_path / "product.yaml").write_text("name: CurrentProduct")
        (tmp_path / "market.yaml").write_text("competitors: []")
        (tmp_path / "team.yaml").write_text("size: 10")
        (tmp_path / "okrs.yaml").write_text("objectives: []")
        
        # get_current should return same as compute_hash
        current = ContextVersion.get_current(tmp_path)
        computed = ContextVersion.compute_hash(tmp_path)
        assert current == computed
    
    def test_load_stored(self, tmp_path):
        """Test loading stored version."""
        # No version file initially
        assert ContextVersion.load_stored(tmp_path) is None
        
        # Create version file
        test_hash = "a" * 64
        (tmp_path / ".version").write_text(test_hash)
        
        # Should load the hash
        loaded = ContextVersion.load_stored(tmp_path)
        assert loaded == test_hash
    
    def test_save_current(self, tmp_path):
        """Test saving current version."""
        # Create test files
        (tmp_path / "company.yaml").write_text("name: SaveCo")
        (tmp_path / "product.yaml").write_text("name: SaveProduct")
        (tmp_path / "market.yaml").write_text("competitors: []")
        (tmp_path / "team.yaml").write_text("size: 10")
        (tmp_path / "okrs.yaml").write_text("objectives: []")
        
        # Save current version
        saved_hash = ContextVersion.save_current(tmp_path)
        
        # Verify it was saved correctly
        assert (tmp_path / ".version").exists()
        loaded_hash = ContextVersion.load_stored(tmp_path)
        assert loaded_hash == saved_hash
        
        # Saved hash should match computed hash
        computed_hash = ContextVersion.compute_hash(tmp_path)
        assert saved_hash == computed_hash
    
    def test_version_changes_with_content(self, tmp_path):
        """Test that version tracking works end-to-end."""
        # Create initial context
        company_file = tmp_path / "company.yaml"
        company_file.write_text("name: InitialCo")
        (tmp_path / "product.yaml").write_text("name: Product")
        (tmp_path / "market.yaml").write_text("competitors: []")
        (tmp_path / "team.yaml").write_text("size: 10")
        (tmp_path / "okrs.yaml").write_text("objectives: []")
        
        # Save initial version
        initial_hash = ContextVersion.save_current(tmp_path)
        
        # Check if changed (should be False)
        current_hash = ContextVersion.get_current(tmp_path)
        assert not ContextVersion.has_changed(initial_hash, current_hash)
        
        # Modify content
        company_file.write_text("name: UpdatedCo")
        
        # Check if changed (should be True)
        new_hash = ContextVersion.get_current(tmp_path)
        assert ContextVersion.has_changed(initial_hash, new_hash)
        
        # Save new version
        ContextVersion.save_current(tmp_path)
        
        # Load and verify
        stored_hash = ContextVersion.load_stored(tmp_path)
        assert stored_hash == new_hash