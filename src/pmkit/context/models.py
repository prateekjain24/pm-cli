"""Context models for PM-Kit.

Lean, PM-focused models that capture just enough context to generate
quality PRDs without enterprise theater.
"""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class CompanyContext(BaseModel):
    """Company context - the essentials only."""
    
    # Required fields - can't generate good PRDs without these
    name: str = Field(..., min_length=2, max_length=100, description="Company name")
    type: Literal["b2b", "b2c", "b2b2c"] = Field(..., description="Business model type")
    stage: Literal["idea", "seed", "growth", "mature"] = Field(..., description="Company stage")
    
    # Smart defaults - inferred or optional
    domain: Optional[str] = Field(None, description="Company website domain")
    description: Optional[str] = Field(None, max_length=200, description="One-line description")
    target_market: Optional[str] = Field(None, description="Primary target market (e.g., SMBs, Enterprise)")
    
    @property
    def is_b2b(self) -> bool:
        """Check if company is B2B focused."""
        return self.type in ["b2b", "b2b2c"]
    
    @property
    def is_b2c(self) -> bool:
        """Check if company is B2C focused."""
        return self.type in ["b2c", "b2b2c"]
    
    @property
    def needs_compliance(self) -> bool:
        """Check if company likely needs compliance features."""
        return self.is_b2b and self.stage in ["growth", "mature"]
    
    model_config = ConfigDict(use_enum_values=True)


class ProductContext(BaseModel):
    """Product context - what PMs actually use daily."""
    
    # Required fields
    name: str = Field(..., min_length=1, max_length=100, description="Product name")
    description: str = Field(..., min_length=10, max_length=500, description="Product description (1-2 sentences)")
    
    # Smart defaults
    stage: Literal["concept", "mvp", "pmf", "scale"] = Field("mvp", description="Product stage")
    users: Optional[int] = Field(None, ge=0, description="Current users/customers")
    pricing_model: Optional[str] = Field(None, description="Pricing model (e.g., freemium, subscription)")
    main_metric: Optional[str] = Field(None, description="North star metric")
    
    @field_validator('main_metric', mode='before')
    @classmethod
    def set_default_metric(cls, v, info):
        """Set smart default metric based on context."""
        if v is None and info.data.get('stage') in ['pmf', 'scale']:
            # Only suggest defaults for more mature products
            return None  # Will be set based on company type later
        return v
    
    model_config = ConfigDict(use_enum_values=True)


class MarketContext(BaseModel):
    """Market context - just enough market intelligence."""
    
    # All optional but useful
    competitors: List[str] = Field(default_factory=list, max_length=20, description="Competitor names")
    market_size: Optional[str] = Field(None, description="TAM/SAM/SOM if known")
    differentiator: Optional[str] = Field(None, max_length=200, description="Unique value proposition (one-liner)")
    
    @property
    def has_competitors(self) -> bool:
        """Check if competitors are defined."""
        return len(self.competitors) > 0
    
    @field_validator('competitors')
    @classmethod
    def validate_competitors(cls, v):
        """Ensure competitor names are reasonable."""
        return [name[:100] for name in v if name.strip()]  # Truncate long names


class TeamContext(BaseModel):
    """Team context - minimal but useful."""
    
    size: Optional[int] = Field(None, ge=1, le=10000, description="Total team size")
    roles: Dict[str, int] = Field(
        default_factory=dict, 
        description="Role distribution (e.g., {'engineers': 5, 'designers': 1})"
    )
    
    @property
    def total_size(self) -> int:
        """Calculate total team size from roles if not explicitly set."""
        if self.size:
            return self.size
        return sum(self.roles.values())
    
    @property
    def is_engineering_heavy(self) -> bool:
        """Check if team is engineering-heavy (useful for technical PRDs)."""
        eng_count = self.roles.get('engineers', 0) + self.roles.get('developers', 0)
        total = self.total_size
        return eng_count / total > 0.5 if total > 0 else False


class KeyResult(BaseModel):
    """Individual key result."""
    
    description: str = Field(..., min_length=5, max_length=200)
    target_value: Optional[str] = Field(None, description="Target metric value")
    current_value: Optional[str] = Field(None, description="Current metric value")
    confidence: Optional[int] = Field(None, ge=0, le=100, description="Confidence % of achieving target by quarter end")


class Objective(BaseModel):
    """Objective with key results."""
    
    title: str = Field(..., min_length=5, max_length=200, description="Objective title")
    key_results: List[KeyResult] = Field(default_factory=list, max_length=5)
    
    @field_validator('key_results')
    @classmethod
    def validate_key_results(cls, v):
        """Ensure at least one key result per objective."""
        if len(v) == 0:
            # Allow empty for initial setup
            return v
        return v
    
    @property
    def average_confidence(self) -> Optional[float]:
        """Get average confidence across all key results."""
        confidences = [kr.confidence for kr in self.key_results if kr.confidence is not None]
        return sum(confidences) / len(confidences) if confidences else None


class OKRContext(BaseModel):
    """OKR context - current objectives and key results."""
    
    objectives: List[Objective] = Field(default_factory=list, max_length=10)
    quarter: Optional[str] = Field(None, description="Current quarter (e.g., Q1 2025)")
    
    @property
    def has_okrs(self) -> bool:
        """Check if OKRs are defined."""
        return len(self.objectives) > 0
    
    @property
    def total_key_results(self) -> int:
        """Count total key results across all objectives."""
        return sum(len(obj.key_results) for obj in self.objectives)
    
    @property
    def at_risk_key_results(self) -> List[KeyResult]:
        """Get key results with low confidence (<50%)."""
        return [
            kr for obj in self.objectives 
            for kr in obj.key_results 
            if kr.confidence is not None and kr.confidence < 50
        ]


class Context(BaseModel):
    """Complete context combining all aspects."""
    
    company: CompanyContext
    product: ProductContext
    market: Optional[MarketContext] = None
    team: Optional[TeamContext] = None
    okrs: Optional[OKRContext] = None
    
    @property
    def is_b2b(self) -> bool:
        """Convenience property for B2B check."""
        return self.company.is_b2b
    
    @property
    def is_b2c(self) -> bool:
        """Convenience property for B2C check."""
        return self.company.is_b2c
    
    def get_default_metric(self) -> str:
        """Get smart default metric based on company type."""
        if self.is_b2b:
            return "MRR"  # Monthly Recurring Revenue
        else:
            return "MAU"  # Monthly Active Users
    
    def to_yaml_dict(self) -> dict:
        """Convert to YAML-friendly dictionary."""
        return self.model_dump(exclude_none=True, exclude_unset=True)
    
    model_config = ConfigDict(use_enum_values=True)