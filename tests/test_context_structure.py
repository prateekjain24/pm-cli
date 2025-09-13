"""Tests for context structure initialization."""

from pathlib import Path

import pytest
import yaml

from pmkit.context.structure import (
    initialize_context_structure,
    check_context_structure,
    repair_context_structure
)


class TestContextStructure:
    """Test context structure initialization functions."""
    
    def test_initialize_creates_all_directories(self, tmp_path):
        """Test that initialization creates all required directories."""
        result = initialize_context_structure(tmp_path)
        
        assert result["success"] is True
        assert len(result["errors"]) == 0
        
        # Check directories exist
        assert (tmp_path / ".pmkit").exists()
        assert (tmp_path / ".pmkit" / "context").exists()
        assert (tmp_path / ".pmkit" / "context" / "history").exists()
        
        # Check created_dirs list
        assert any(".pmkit" in d for d in result["created_dirs"])
        assert any("context" in d for d in result["created_dirs"])
        assert any("history" in d for d in result["created_dirs"])
    
    def test_initialize_creates_template_files(self, tmp_path):
        """Test that initialization creates all template YAML files."""
        result = initialize_context_structure(tmp_path)
        
        assert result["success"] is True
        
        context_dir = tmp_path / ".pmkit" / "context"
        
        # Check all template files exist
        expected_files = ["company.yaml", "product.yaml", "market.yaml", 
                         "team.yaml", "okrs.yaml", ".gitignore"]
        
        for filename in expected_files:
            file_path = context_dir / filename
            assert file_path.exists(), f"Missing {filename}"
            
            # Skip .gitignore for YAML validation
            if filename != ".gitignore":
                assert filename in [Path(f).name for f in result["created_files"]]
    
    def test_template_files_have_valid_yaml(self, tmp_path):
        """Test that template files contain valid YAML."""
        initialize_context_structure(tmp_path)
        
        context_dir = tmp_path / ".pmkit" / "context"
        yaml_files = ["company.yaml", "product.yaml", "market.yaml", 
                     "team.yaml", "okrs.yaml"]
        
        for filename in yaml_files:
            file_path = context_dir / filename
            
            # Read and parse YAML
            with open(file_path) as f:
                content = f.read()
                # Skip header comments
                yaml_content = '\n'.join(
                    line for line in content.split('\n') 
                    if not line.strip().startswith('#')
                )
                data = yaml.safe_load(yaml_content)
                
                assert data is not None, f"{filename} has no YAML content"
                assert isinstance(data, dict), f"{filename} should contain a dict"
    
    def test_company_template_content(self, tmp_path):
        """Test that company.yaml template has expected fields."""
        initialize_context_structure(tmp_path)
        
        company_file = tmp_path / ".pmkit" / "context" / "company.yaml"
        with open(company_file) as f:
            content = f.read()
            # Check header comment
            assert "Company Context Configuration" in content
            
            # Parse YAML
            yaml_content = '\n'.join(
                line for line in content.split('\n') 
                if not line.strip().startswith('#')
            )
            data = yaml.safe_load(yaml_content)
            
            # Check required fields
            assert data["name"] == "YourCompany"
            assert data["type"] in ["b2b", "b2c"]
            assert data["stage"] in ["seed", "growth", "scale", "enterprise"]
            
            # Check optional fields exist
            assert "domain" in data
            assert "description" in data
    
    def test_product_template_content(self, tmp_path):
        """Test that product.yaml template has expected fields."""
        initialize_context_structure(tmp_path)
        
        product_file = tmp_path / ".pmkit" / "context" / "product.yaml"
        with open(product_file) as f:
            content = f.read()
            # Check header
            assert "Product Context Configuration" in content
            
            yaml_content = '\n'.join(
                line for line in content.split('\n') 
                if not line.strip().startswith('#')
            )
            data = yaml.safe_load(yaml_content)
            
            # Check fields
            assert data["name"] == "YourProduct"
            assert "description" in data
            assert data["stage"] == "idea"
            assert data["users"] == 100
            assert data["pricing_model"] == "subscription"
            assert data["main_metric"] == "MRR"
    
    def test_okrs_template_structure(self, tmp_path):
        """Test that okrs.yaml template has proper nested structure."""
        initialize_context_structure(tmp_path)
        
        okrs_file = tmp_path / ".pmkit" / "context" / "okrs.yaml"
        with open(okrs_file) as f:
            yaml_content = '\n'.join(
                line for line in f.read().split('\n') 
                if not line.strip().startswith('#')
            )
            data = yaml.safe_load(yaml_content)
            
            assert data["quarter"] == "Q1 2025"
            assert "objectives" in data
            assert len(data["objectives"]) > 0
            
            objective = data["objectives"][0]
            assert "title" in objective
            assert "key_results" in objective
            assert len(objective["key_results"]) > 0
            
            kr = objective["key_results"][0]
            assert "description" in kr
            assert "target_value" in kr
            assert "current_value" in kr
            assert "confidence" in kr
    
    def test_gitignore_created(self, tmp_path):
        """Test that .gitignore is created with proper content."""
        initialize_context_structure(tmp_path)
        
        gitignore = tmp_path / ".pmkit" / "context" / ".gitignore"
        assert gitignore.exists()
        
        content = gitignore.read_text()
        assert "history/" in content
        assert "*.backup" in content
        assert "*.tmp" in content
    
    def test_initialize_idempotent(self, tmp_path):
        """Test that initialization is idempotent (safe to run multiple times)."""
        # First initialization
        result1 = initialize_context_structure(tmp_path)
        assert result1["success"] is True
        files_created_1 = len(result1["created_files"])
        dirs_created_1 = len(result1["created_dirs"])
        
        # Second initialization
        result2 = initialize_context_structure(tmp_path)
        assert result2["success"] is True
        
        # Should not create anything new
        assert len(result2["created_files"]) == 0
        assert len(result2["created_dirs"]) == 0
    
    def test_check_context_structure_not_initialized(self, tmp_path):
        """Test checking structure when nothing exists."""
        status = check_context_structure(tmp_path)
        
        assert status["initialized"] is False
        assert status["has_templates"] is False
        assert len(status["missing_dirs"]) == 3  # .pmkit, context, history
        assert len(status["missing_files"]) == 5  # 5 YAML files
    
    def test_check_context_structure_fully_initialized(self, tmp_path):
        """Test checking structure when fully initialized."""
        initialize_context_structure(tmp_path)
        
        status = check_context_structure(tmp_path)
        
        assert status["initialized"] is True
        assert status["has_templates"] is True
        assert len(status["missing_dirs"]) == 0
        assert len(status["missing_files"]) == 0
    
    def test_check_context_structure_partial(self, tmp_path):
        """Test checking structure when partially initialized."""
        # Create only some directories
        context_dir = tmp_path / ".pmkit" / "context"
        context_dir.mkdir(parents=True)
        
        # Create only some files
        (context_dir / "company.yaml").write_text("name: Test")
        
        status = check_context_structure(tmp_path)
        
        assert status["initialized"] is True  # context dir exists
        assert status["has_templates"] is True  # at least one template exists
        assert len(status["missing_dirs"]) == 1  # history dir missing
        assert len(status["missing_files"]) == 4  # 4 YAML files missing
    
    def test_repair_context_structure_from_scratch(self, tmp_path):
        """Test repairing when nothing exists."""
        result = repair_context_structure(tmp_path)
        
        assert result["success"] is True
        
        # Should create everything
        assert (tmp_path / ".pmkit" / "context").exists()
        assert (tmp_path / ".pmkit" / "context" / "history").exists()
        
        # Check all files created
        context_dir = tmp_path / ".pmkit" / "context"
        for filename in ["company.yaml", "product.yaml", "market.yaml", 
                        "team.yaml", "okrs.yaml"]:
            assert (context_dir / filename).exists()
    
    def test_repair_context_structure_partial(self, tmp_path):
        """Test repairing when some things exist."""
        # Create partial structure
        context_dir = tmp_path / ".pmkit" / "context"
        context_dir.mkdir(parents=True)
        (context_dir / "company.yaml").write_text("name: ExistingCompany\ntype: b2b")
        
        # Repair
        result = repair_context_structure(tmp_path)
        
        assert result["success"] is True
        
        # Should create missing items
        assert (context_dir / "history").exists()
        assert (context_dir / "product.yaml").exists()
        assert (context_dir / "market.yaml").exists()
        
        # Should NOT overwrite existing file
        company_content = (context_dir / "company.yaml").read_text()
        assert "ExistingCompany" in company_content
        assert "YourCompany" not in company_content
    
    def test_repair_when_complete(self, tmp_path):
        """Test repair when structure is already complete."""
        # Initialize first
        initialize_context_structure(tmp_path)
        
        # Try to repair
        result = repair_context_structure(tmp_path)
        
        assert result["success"] is True
        assert result["message"] == "Context structure is already complete"
        assert len(result["created_dirs"]) == 0
        assert len(result["created_files"]) == 0
    
    def test_handles_permission_errors(self, tmp_path, monkeypatch):
        """Test that permission errors are handled gracefully."""
        import os
        
        # Create read-only directory
        pmkit_dir = tmp_path / ".pmkit"
        pmkit_dir.mkdir()
        
        # Make it read-only (Unix-like systems)
        if os.name != 'nt':  # Not Windows
            os.chmod(pmkit_dir, 0o444)
            
            result = initialize_context_structure(tmp_path)
            
            # Should report errors but not crash
            assert result["success"] is False
            assert len(result["errors"]) > 0
            
            # Restore permissions for cleanup
            os.chmod(pmkit_dir, 0o755)