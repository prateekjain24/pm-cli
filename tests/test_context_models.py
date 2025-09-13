"""Tests for context models."""

import pytest
from pydantic import ValidationError

from pmkit.context.models import (
    CompanyContext,
    ProductContext,
    MarketContext,
    TeamContext,
    KeyResult,
    Objective,
    OKRContext,
    Context,
)


class TestCompanyContext:
    """Test CompanyContext model."""
    
    def test_minimal_valid_company(self):
        """Test creating company with minimal required fields."""
        company = CompanyContext(
            name="Acme Corp",
            type="b2b",
            stage="seed"
        )
        assert company.name == "Acme Corp"
        assert company.type == "b2b"
        assert company.stage == "seed"
        assert company.domain is None
        assert company.description is None
    
    def test_company_with_all_fields(self):
        """Test creating company with all fields."""
        company = CompanyContext(
            name="Acme Corp",
            type="b2b",
            stage="growth",
            domain="acme.com",
            description="Leading B2B SaaS platform",
            target_market="Enterprise"
        )
        assert company.domain == "acme.com"
        assert company.description == "Leading B2B SaaS platform"
        assert company.target_market == "Enterprise"
    
    def test_b2b_properties(self):
        """Test B2B-related properties."""
        b2b_company = CompanyContext(name="B2B Co", type="b2b", stage="growth")
        assert b2b_company.is_b2b is True
        assert b2b_company.is_b2c is False
        assert b2b_company.needs_compliance is True
        
        b2c_company = CompanyContext(name="B2C Co", type="b2c", stage="growth")
        assert b2c_company.is_b2b is False
        assert b2c_company.is_b2c is True
        assert b2c_company.needs_compliance is False
        
        b2b2c_company = CompanyContext(name="B2B2C Co", type="b2b2c", stage="seed")
        assert b2b2c_company.is_b2b is True
        assert b2b2c_company.is_b2c is True
        assert b2b2c_company.needs_compliance is False  # Not in growth/mature
    
    def test_company_validation(self):
        """Test company field validation."""
        # Name too short
        with pytest.raises(ValidationError):
            CompanyContext(name="A", type="b2b", stage="seed")
        
        # Name too long
        with pytest.raises(ValidationError):
            CompanyContext(name="A" * 101, type="b2b", stage="seed")
        
        # Invalid type
        with pytest.raises(ValidationError):
            CompanyContext(name="Test Co", type="invalid", stage="seed")
        
        # Invalid stage
        with pytest.raises(ValidationError):
            CompanyContext(name="Test Co", type="b2b", stage="invalid")


class TestProductContext:
    """Test ProductContext model."""
    
    def test_minimal_valid_product(self):
        """Test creating product with minimal required fields."""
        product = ProductContext(
            name="Awesome Product",
            description="A tool that helps teams collaborate better"
        )
        assert product.name == "Awesome Product"
        assert product.description == "A tool that helps teams collaborate better"
        assert product.stage == "mvp"  # Default
        assert product.users is None
        assert product.pricing_model is None
    
    def test_product_with_all_fields(self):
        """Test creating product with all fields."""
        product = ProductContext(
            name="Pro Tool",
            description="Enterprise collaboration platform",
            stage="scale",
            users=10000,
            pricing_model="subscription",
            main_metric="MRR"
        )
        assert product.stage == "scale"
        assert product.users == 10000
        assert product.pricing_model == "subscription"
        assert product.main_metric == "MRR"
    
    def test_product_validation(self):
        """Test product field validation."""
        # Name empty
        with pytest.raises(ValidationError):
            ProductContext(name="", description="Valid description")
        
        # Description too short
        with pytest.raises(ValidationError):
            ProductContext(name="Product", description="Short")
        
        # Description too long
        with pytest.raises(ValidationError):
            ProductContext(name="Product", description="A" * 501)
        
        # Negative users
        with pytest.raises(ValidationError):
            ProductContext(
                name="Product",
                description="Valid description here",
                users=-1
            )


class TestMarketContext:
    """Test MarketContext model."""
    
    def test_empty_market_context(self):
        """Test creating empty market context."""
        market = MarketContext()
        assert market.competitors == []
        assert market.market_size is None
        assert market.differentiator is None
        assert market.has_competitors is False
    
    def test_market_with_competitors(self):
        """Test market with competitors."""
        market = MarketContext(
            competitors=["Slack", "Teams", "Discord"],
            market_size="$10B TAM",
            differentiator="AI-powered insights"
        )
        assert len(market.competitors) == 3
        assert market.has_competitors is True
        assert market.market_size == "$10B TAM"
        assert market.differentiator == "AI-powered insights"
    
    def test_competitor_validation(self):
        """Test competitor list validation."""
        # Too many competitors
        with pytest.raises(ValidationError):
            MarketContext(competitors=["Company" + str(i) for i in range(21)])
        
        # Long competitor names get truncated
        long_name = "A" * 150
        market = MarketContext(competitors=[long_name])
        assert len(market.competitors[0]) == 100
        
        # Empty strings get filtered
        market = MarketContext(competitors=["Valid", "", "  ", "Another"])
        assert len(market.competitors) == 2


class TestTeamContext:
    """Test TeamContext model."""
    
    def test_empty_team_context(self):
        """Test creating empty team context."""
        team = TeamContext()
        assert team.size is None
        assert team.roles == {}
        assert team.total_size == 0
        assert team.is_engineering_heavy is False
    
    def test_team_with_size(self):
        """Test team with explicit size."""
        team = TeamContext(size=50)
        assert team.size == 50
        assert team.total_size == 50
    
    def test_team_with_roles(self):
        """Test team with role distribution."""
        team = TeamContext(
            roles={
                "engineers": 10,
                "designers": 2,
                "pm": 1,
                "marketing": 3
            }
        )
        assert team.total_size == 16
        assert team.is_engineering_heavy is True
        
        # Non-engineering heavy team
        team2 = TeamContext(
            roles={
                "engineers": 2,
                "sales": 10,
                "marketing": 5
            }
        )
        assert team2.is_engineering_heavy is False
    
    def test_team_validation(self):
        """Test team field validation."""
        # Size too small
        with pytest.raises(ValidationError):
            TeamContext(size=0)
        
        # Size too large
        with pytest.raises(ValidationError):
            TeamContext(size=10001)


class TestOKRContext:
    """Test OKR context models."""
    
    def test_empty_okr_context(self):
        """Test creating empty OKR context."""
        okrs = OKRContext()
        assert okrs.objectives == []
        assert okrs.quarter is None
        assert okrs.has_okrs is False
        assert okrs.total_key_results == 0
    
    def test_okr_with_objectives(self):
        """Test OKR with objectives and key results."""
        kr1 = KeyResult(
            description="Increase MRR to $100K",
            target_value="$100K",
            current_value="$75K"
        )
        kr2 = KeyResult(
            description="Achieve 95% customer retention"
        )
        
        obj = Objective(
            title="Achieve product-market fit",
            key_results=[kr1, kr2]
        )
        
        okrs = OKRContext(
            objectives=[obj],
            quarter="Q1 2025"
        )
        
        assert okrs.has_okrs is True
        assert okrs.total_key_results == 2
        assert okrs.quarter == "Q1 2025"
    
    def test_objective_validation(self):
        """Test objective validation."""
        # Title too short
        with pytest.raises(ValidationError):
            Objective(title="Bad")
        
        # Too many key results
        with pytest.raises(ValidationError):
            krs = [KeyResult(description=f"KR {i}") for i in range(6)]
            Objective(title="Valid objective", key_results=krs)
    
    def test_key_result_validation(self):
        """Test key result validation."""
        # Description too short
        with pytest.raises(ValidationError):
            KeyResult(description="Bad")
        
        # Description too long
        with pytest.raises(ValidationError):
            KeyResult(description="A" * 201)
        
        # Confidence out of range (negative)
        with pytest.raises(ValidationError):
            KeyResult(description="Valid KR", confidence=-1)
        
        # Confidence out of range (>100)
        with pytest.raises(ValidationError):
            KeyResult(description="Valid KR", confidence=101)
    
    def test_confidence_scoring(self):
        """Test confidence scoring features."""
        # Key result with confidence
        kr1 = KeyResult(
            description="Increase MRR to $100K",
            target_value="$100K",
            current_value="$75K",
            confidence=70
        )
        kr2 = KeyResult(
            description="Launch enterprise SSO",
            target_value="Shipped",
            current_value="In dev",
            confidence=40  # At risk
        )
        kr3 = KeyResult(
            description="Achieve 95% uptime",
            target_value="95%",
            current_value="92%",
            confidence=85
        )
        kr4 = KeyResult(
            description="Complete user research",
            target_value="20 interviews",
            current_value="5 interviews"
            # No confidence set
        )
        
        obj1 = Objective(
            title="Q1 Growth targets",
            key_results=[kr1, kr2, kr3]
        )
        obj2 = Objective(
            title="User insights",
            key_results=[kr4]
        )
        
        # Test average confidence at objective level
        assert obj1.average_confidence == (70 + 40 + 85) / 3
        assert obj2.average_confidence is None  # No confidence set
        
        # Test at-risk key results
        okrs = OKRContext(
            objectives=[obj1, obj2],
            quarter="Q1 2025"
        )
        
        at_risk = okrs.at_risk_key_results
        assert len(at_risk) == 1
        assert at_risk[0].description == "Launch enterprise SSO"
        assert at_risk[0].confidence == 40


class TestContext:
    """Test complete Context model."""
    
    def test_minimal_context(self):
        """Test creating context with minimal required fields."""
        context = Context(
            company=CompanyContext(
                name="Test Co",
                type="b2b",
                stage="seed"
            ),
            product=ProductContext(
                name="Test Product",
                description="A great product for testing"
            )
        )
        
        assert context.is_b2b is True
        assert context.is_b2c is False
        assert context.market is None
        assert context.team is None
        assert context.okrs is None
    
    def test_full_context(self):
        """Test creating context with all components."""
        context = Context(
            company=CompanyContext(
                name="Full Co",
                type="b2c",
                stage="growth"
            ),
            product=ProductContext(
                name="Consumer App",
                description="Mobile app for consumers"
            ),
            market=MarketContext(
                competitors=["App1", "App2"],
                market_size="$1B"
            ),
            team=TeamContext(
                size=20,
                roles={"engineers": 12, "designers": 3}
            ),
            okrs=OKRContext(
                objectives=[
                    Objective(
                        title="Reach 1M users",
                        key_results=[
                            KeyResult(description="Launch in 3 new markets")
                        ]
                    )
                ]
            )
        )
        
        assert context.is_b2c is True
        assert context.market.has_competitors is True
        assert context.team.total_size == 20
        assert context.okrs.has_okrs is True
    
    def test_default_metric(self):
        """Test smart default metric based on company type."""
        b2b_context = Context(
            company=CompanyContext(name="B2B", type="b2b", stage="growth"),
            product=ProductContext(name="Product", description="B2B product")
        )
        assert b2b_context.get_default_metric() == "MRR"
        
        b2c_context = Context(
            company=CompanyContext(name="B2C", type="b2c", stage="growth"),
            product=ProductContext(name="Product", description="B2C product")
        )
        assert b2c_context.get_default_metric() == "MAU"
    
    def test_to_yaml_dict(self):
        """Test YAML serialization."""
        context = Context(
            company=CompanyContext(
                name="YAML Co",
                type="b2b",
                stage="seed",
                domain="yaml.com"
            ),
            product=ProductContext(
                name="YAML Product",
                description="Product for YAML testing"
            )
        )
        
        yaml_dict = context.to_yaml_dict()
        
        # Check structure
        assert "company" in yaml_dict
        assert "product" in yaml_dict
        assert yaml_dict["company"]["name"] == "YAML Co"
        assert yaml_dict["company"]["domain"] == "yaml.com"
        
        # Check None values are excluded
        assert "market" not in yaml_dict
        assert "team" not in yaml_dict
        assert "okrs" not in yaml_dict