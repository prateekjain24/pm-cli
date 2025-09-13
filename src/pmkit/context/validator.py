"""Context validation for PM-Kit.

Ensures context data is valid and consistent before use.
Provides detailed validation errors and optional auto-repair.
"""

from typing import List, Optional, Tuple

from pmkit.context.models import Context, TeamContext


class ValidationError:
    """Represents a validation error with details."""
    
    def __init__(self, field: str, message: str, severity: str = "error"):
        """Initialize validation error.
        
        Args:
            field: Field path that failed validation (e.g., "company.name")
            message: Human-readable error message
            severity: "error" or "warning"
        """
        self.field = field
        self.message = message
        self.severity = severity
    
    def __str__(self) -> str:
        """Format error as string."""
        return f"[{self.severity.upper()}] {self.field}: {self.message}"


class ContextValidator:
    """Validates context data for consistency and completeness.
    
    Simple validation without over-engineering. Focus on catching
    common issues that would break PRD generation.
    """
    
    def validate(self, context: Context) -> Tuple[bool, List[ValidationError]]:
        """Validate a complete context.
        
        Args:
            context: Context object to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Required components
        if not context.company:
            errors.append(ValidationError(
                "company",
                "Company context is required"
            ))
        else:
            errors.extend(self._validate_company(context))
        
        if not context.product:
            errors.append(ValidationError(
                "product",
                "Product context is required"
            ))
        else:
            errors.extend(self._validate_product(context))
        
        # Optional components
        if context.market:
            errors.extend(self._validate_market(context))
        
        if context.team:
            errors.extend(self._validate_team(context))
        
        if context.okrs:
            errors.extend(self._validate_okrs(context))
        
        # Cross-component validation
        errors.extend(self._validate_consistency(context))
        
        # Filter out warnings for is_valid check
        has_errors = any(e.severity == "error" for e in errors)
        
        return (not has_errors, errors)
    
    def _validate_company(self, context: Context) -> List[ValidationError]:
        """Validate company context."""
        errors = []
        company = context.company
        
        # Check for suspiciously generic names
        generic_names = ["test", "demo", "example", "company", "testco"]
        if company.name.lower() in generic_names:
            errors.append(ValidationError(
                "company.name",
                f"Company name '{company.name}' appears to be a placeholder",
                severity="warning"
            ))
        
        # B2B companies at growth/mature stage should consider compliance
        if company.is_b2b and company.stage in ["growth", "mature"]:
            if not company.target_market:
                errors.append(ValidationError(
                    "company.target_market",
                    "B2B companies at growth/mature stage should specify target market (SMB/Enterprise)",
                    severity="warning"
                ))
        
        return errors
    
    def _validate_product(self, context: Context) -> List[ValidationError]:
        """Validate product context."""
        errors = []
        product = context.product
        
        # Check description quality
        if len(product.description) < 20:
            errors.append(ValidationError(
                "product.description",
                "Product description is too brief (< 20 chars). PRDs will lack context.",
                severity="warning"
            ))
        
        # Check stage consistency
        if product.stage in ["pmf", "scale"] and not product.users:
            errors.append(ValidationError(
                "product.users",
                f"Products at {product.stage} stage should have user/customer count",
                severity="warning"
            ))
        
        # Check pricing model for mature products
        if product.stage == "scale" and not product.pricing_model:
            errors.append(ValidationError(
                "product.pricing_model",
                "Products at scale should have a defined pricing model",
                severity="warning"
            ))
        
        # Check metric definition
        if product.stage in ["pmf", "scale"] and not product.main_metric:
            default_metric = "MRR" if context.is_b2b else "MAU"
            errors.append(ValidationError(
                "product.main_metric",
                f"Consider setting main metric (suggested: {default_metric})",
                severity="warning"
            ))
        
        return errors
    
    def _validate_market(self, context: Context) -> List[ValidationError]:
        """Validate market context."""
        errors = []
        market = context.market
        
        # Check for duplicate competitors
        if market.competitors:
            unique_competitors = set(c.lower() for c in market.competitors)
            if len(unique_competitors) < len(market.competitors):
                errors.append(ValidationError(
                    "market.competitors",
                    "Duplicate competitors found in list",
                    severity="warning"
                ))
        
        # Check differentiator for competitive markets
        if len(market.competitors) > 3 and not market.differentiator:
            errors.append(ValidationError(
                "market.differentiator",
                "Highly competitive market (>3 competitors) should have clear differentiator",
                severity="warning"
            ))
        
        return errors
    
    def _validate_team(self, context: Context) -> List[ValidationError]:
        """Validate team context."""
        errors = []
        team = context.team
        
        # Check team size consistency
        if team.size and team.roles:
            role_total = sum(team.roles.values())
            if role_total > 0 and abs(team.size - role_total) > 2:
                errors.append(ValidationError(
                    "team.size",
                    f"Team size ({team.size}) doesn't match role total ({role_total})",
                    severity="warning"
                ))
        
        # Check for negative values in roles
        if team.roles:
            for role, count in team.roles.items():
                if count < 0:
                    errors.append(ValidationError(
                        f"team.roles.{role}",
                        f"Role count cannot be negative ({count})"
                    ))
        
        return errors
    
    def _validate_okrs(self, context: Context) -> List[ValidationError]:
        """Validate OKR context."""
        errors = []
        okrs = context.okrs
        
        # Check for objectives without key results
        for i, obj in enumerate(okrs.objectives):
            if len(obj.key_results) == 0:
                errors.append(ValidationError(
                    f"okrs.objectives[{i}]",
                    f"Objective '{obj.title}' has no key results",
                    severity="warning"
                ))
            
            # Check for at-risk key results
            at_risk_count = sum(
                1 for kr in obj.key_results 
                if kr.confidence is not None and kr.confidence < 50
            )
            if at_risk_count > len(obj.key_results) / 2:
                errors.append(ValidationError(
                    f"okrs.objectives[{i}]",
                    f"Objective '{obj.title}' has majority of KRs at risk (<50% confidence)",
                    severity="warning"
                ))
        
        # Check quarter format
        if okrs.quarter:
            import re
            if not re.match(r'^Q[1-4]\s+20\d{2}$', okrs.quarter):
                errors.append(ValidationError(
                    "okrs.quarter",
                    f"Quarter format should be 'Q1 2025', got '{okrs.quarter}'",
                    severity="warning"
                ))
        
        return errors
    
    def _validate_consistency(self, context: Context) -> List[ValidationError]:
        """Validate cross-component consistency."""
        errors = []
        
        # B2B companies should likely have sales team
        if context.is_b2b and context.team and context.team.roles:
            has_sales = any(
                'sales' in role.lower() or 'bd' in role.lower() 
                for role in context.team.roles.keys()
            )
            if not has_sales and context.company.stage in ["growth", "mature"]:
                errors.append(ValidationError(
                    "team.roles",
                    "B2B companies at growth/mature stage typically need sales team",
                    severity="warning"
                ))
        
        # Product stage vs company stage alignment
        if context.company and context.product:
            company_stage_order = ["idea", "seed", "growth", "mature"]
            product_stage_order = ["concept", "mvp", "pmf", "scale"]
            
            company_idx = company_stage_order.index(context.company.stage)
            product_idx = product_stage_order.index(context.product.stage)
            
            # Product shouldn't be too far ahead of company
            if product_idx > company_idx + 1:
                errors.append(ValidationError(
                    "product.stage",
                    f"Product stage ({context.product.stage}) seems ahead of company stage ({context.company.stage})",
                    severity="warning"
                ))
        
        return errors
    
    def can_auto_repair(self, context: Context, errors: List[ValidationError]) -> bool:
        """Check if context can be auto-repaired.
        
        Args:
            context: Context with validation errors
            errors: List of validation errors
            
        Returns:
            True if auto-repair is possible
        """
        # For MVP, we don't auto-repair errors, only warnings
        # and only specific types of warnings
        repairable_fields = [
            "product.main_metric",
            "team.size"
        ]
        
        for error in errors:
            if error.severity == "error":
                return False
            if error.severity == "warning" and error.field not in repairable_fields:
                continue
        
        return True
    
    def auto_repair(self, context: Context, errors: List[ValidationError]) -> Context:
        """Attempt to auto-repair validation issues.
        
        Only repairs specific warnings, never errors.
        
        Args:
            context: Context to repair
            errors: List of validation errors
            
        Returns:
            Repaired context (or original if can't repair)
        """
        if not self.can_auto_repair(context, errors):
            return context
        
        repaired = context.model_copy(deep=True)
        
        for error in errors:
            # Auto-set default metric if missing
            if error.field == "product.main_metric" and not repaired.product.main_metric:
                repaired.product.main_metric = repaired.get_default_metric()
            
            # Fix team size mismatch
            elif error.field == "team.size" and repaired.team:
                if repaired.team.roles:
                    repaired.team.size = sum(repaired.team.roles.values())
        
        return repaired