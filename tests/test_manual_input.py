"""
Tests for manual input form with review/edit capabilities.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from prompt_toolkit.document import Document
from prompt_toolkit.validation import ValidationError
from rich.console import Console

from pmkit.agents.manual_input import (
    FIELD_DEFINITIONS,
    FieldMetadata,
    FieldStatus,
    ManualInputForm,
)
from pmkit.agents.validators import (
    CompanyNameSmartValidator,
    NorthStarMetricSmartValidator,
    ProductDescriptionSmartValidator,
    SmartPromptValidator,
    TeamSizeSmartValidator,
    URLSmartValidator,
    apply_autocorrect,
    validate_and_fix,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def console():
    """Create a test console."""
    return Console(force_terminal=True, width=80)


@pytest.fixture
def manual_form(console):
    """Create a ManualInputForm instance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        form = ManualInputForm(console=console)
        form.state_file = Path(tmpdir) / "test_state.json"
        form.state_file.parent.mkdir(parents=True, exist_ok=True)
        yield form


@pytest.fixture
def sample_enriched_data():
    """Sample enriched data from web search."""
    return {
        'company_name': 'Acme Corp',
        'company_type': 'b2b',
        'product_name': 'AcmeFlow',
        'product_description': 'Workflow automation for enterprise teams',
        'company_stage': 'growth',
        'target_market': 'SMBs',
        'competitors': ['Slack', 'Teams'],
        'website': 'acme.com',  # Missing protocol
    }


@pytest.fixture
def partial_enriched_data():
    """Partially enriched data with missing fields."""
    return {
        'company_name': 'Test Company',  # Suspicious name
        'company_type': 'b2b',
        'product_name': '',  # Missing
        'product_description': '',  # Missing
        'competitors': ['Test Company'],  # Self as competitor
    }


# ============================================================
# MANUAL INPUT FORM TESTS
# ============================================================

class TestManualInputForm:
    """Tests for ManualInputForm class."""

    def test_analyze_field_status_all_confirmed(self, manual_form, sample_enriched_data):
        """Test field status analysis when all fields are good."""
        # Fix the website to have protocol
        sample_enriched_data['website'] = 'https://acme.com'

        status = manual_form._analyze_field_status(sample_enriched_data, 'b2b')

        # Required fields should be confirmed
        assert status['company_name'] == FieldStatus.CONFIRMED
        assert status['product_name'] == FieldStatus.CONFIRMED
        assert status['product_description'] == FieldStatus.CONFIRMED

        # Optional fields with values should be confirmed
        assert status.get('website') == FieldStatus.CONFIRMED
        assert status.get('competitors') == FieldStatus.CONFIRMED

    def test_analyze_field_status_with_missing(self, manual_form, partial_enriched_data):
        """Test field status analysis with missing required fields."""
        status = manual_form._analyze_field_status(partial_enriched_data, 'b2b')

        # Missing required fields
        assert status['product_name'] == FieldStatus.MISSING
        assert status['product_description'] == FieldStatus.MISSING

        # Suspicious data should need review
        assert status['company_name'] == FieldStatus.REVIEW

    def test_needs_review_detects_test_data(self, manual_form):
        """Test that suspicious/test data is flagged for review."""
        assert manual_form._needs_review('company_name', 'Test Company')
        assert manual_form._needs_review('company_name', 'demo')
        assert manual_form._needs_review('company_name', 'TODO')

        # Real names should not need review
        assert not manual_form._needs_review('company_name', 'Acme Corp')
        assert not manual_form._needs_review('company_name', 'TechStart Inc')

    def test_needs_review_product_description(self, manual_form):
        """Test product description review detection."""
        # Too brief
        assert manual_form._needs_review('product_description', 'AI tool')

        # Good description
        assert not manual_form._needs_review(
            'product_description',
            'AI-powered code review tool for enterprise engineering teams'
        )

    def test_needs_review_website(self, manual_form):
        """Test website review detection."""
        # Missing protocol
        assert manual_form._needs_review('website', 'acme.com')

        # Proper URL
        assert not manual_form._needs_review('website', 'https://acme.com')

    def test_find_next_field_priority(self, manual_form):
        """Test that find_next_field respects priority (MISSING > REVIEW)."""
        status = {
            'company_name': FieldStatus.CONFIRMED,
            'product_name': FieldStatus.REVIEW,
            'product_description': FieldStatus.MISSING,
            'website': FieldStatus.OPTIONAL,
        }

        # Should return MISSING field first
        next_field = manual_form._find_next_field(status)
        assert next_field == 'product_description'

        # After fixing missing, should return REVIEW field
        status['product_description'] = FieldStatus.CONFIRMED
        next_field = manual_form._find_next_field(status)
        assert next_field == 'product_name'

        # When all confirmed, should return None
        status['product_name'] = FieldStatus.CONFIRMED
        next_field = manual_form._find_next_field(status)
        assert next_field is None

    def test_should_show_field_b2b_specific(self, manual_form):
        """Test that B2B-specific fields are shown only for B2B companies."""
        # Create a B2B-specific field
        b2b_field = FieldMetadata(
            name='pricing_model',
            display_name='Pricing Model',
            field_type='text',
            required=False,
            b2b_specific=True,
            phase=3,
        )

        # Should show for B2B
        assert manual_form._should_show_field(b2b_field, 'b2b')
        assert manual_form._should_show_field(b2b_field, 'B2B')

        # Should not show for B2C
        assert not manual_form._should_show_field(b2b_field, 'b2c')
        assert not manual_form._should_show_field(b2b_field, 'B2C')

    def test_save_and_load_progress(self, manual_form):
        """Test saving and loading progress."""
        test_data = {
            'company_name': 'Acme Corp',
            'product_name': 'AcmeFlow',
            'company_type': 'b2b',
        }

        # Save progress
        manual_form._save_progress(test_data)

        # Load progress
        loaded_data = manual_form.load_progress()

        assert loaded_data == test_data

    def test_load_progress_with_staleness_warning(self, manual_form, capsys):
        """Test that stale data triggers a warning."""
        # Create old save data
        old_date = datetime.now() - timedelta(days=10)
        save_data = {
            'data': {'company_name': 'Old Company'},
            'last_modified': old_date.isoformat(),
            'version': '1.0',
        }

        # Write old data
        with open(manual_form.state_file, 'w') as f:
            json.dump(save_data, f)

        # Load should show warning
        loaded_data = manual_form.load_progress()

        assert loaded_data == {'company_name': 'Old Company'}
        captured = capsys.readouterr()
        # Check for the warning (strip ANSI codes)
        import re
        clean_output = re.sub(r'\x1b\[[0-9;]+m', '', captured.out)
        assert '10 days old' in clean_output

    def test_clear_progress(self, manual_form):
        """Test clearing saved progress."""
        # Save some data
        manual_form._save_progress({'test': 'data'})
        assert manual_form.state_file.exists()

        # Clear progress
        manual_form.clear_progress()
        assert not manual_form.state_file.exists()

        # Loading should return None
        assert manual_form.load_progress() is None


# ============================================================
# SMART VALIDATOR TESTS
# ============================================================

class TestSmartValidators:
    """Tests for smart validators."""

    def test_company_name_validator(self):
        """Test company name validation with all levels."""
        validator = CompanyNameSmartValidator()

        # Valid name
        is_valid, messages, corrected = validator.validate('Acme Corp', {})
        assert is_valid
        assert len(messages) == 0

        # Too short (error)
        is_valid, messages, corrected = validator.validate('A', {})
        assert not is_valid
        assert any('at least 2 characters' in msg for msg in messages)

        # Test data (warning)
        is_valid, messages, corrected = validator.validate('test', {})
        assert is_valid  # Warning doesn't block
        assert any('test data' in msg for msg in messages)

        # Famous company (warning)
        is_valid, messages, corrected = validator.validate('Google', {})
        assert is_valid
        assert any('famous company' in msg for msg in messages)

        # Whitespace autocorrect
        is_valid, messages, corrected = validator.validate('  Acme  Corp  ', {})
        assert is_valid
        assert corrected == 'Acme Corp'

    def test_product_description_validator(self):
        """Test product description validation."""
        validator = ProductDescriptionSmartValidator()

        # Too brief (error)
        is_valid, messages, corrected = validator.validate('AI tool', {})
        assert not is_valid
        assert any('Too brief' in msg for msg in messages)

        # Missing target audience (error) - no 'for', 'helps', 'enables', or 'that'
        is_valid, messages, corrected = validator.validate(
            'Advanced machine learning platform using cutting-edge algorithms',
            {}
        )
        assert not is_valid
        assert any('who this product is FOR' in msg for msg in messages)

        # Good description
        is_valid, messages, corrected = validator.validate(
            'AI-powered code review tool for enterprise engineering teams',
            {}
        )
        assert is_valid
        assert len(messages) == 0

    def test_north_star_metric_validator(self):
        """Test north star metric validation with B2B/B2C alignment."""
        validator = NorthStarMetricSmartValidator()

        # Autocorrect typo
        is_valid, messages, corrected = validator.validate('MMR', {})
        assert is_valid
        assert corrected == 'MRR'

        # B2B company with B2C metric (warning)
        context = {'company_type': 'b2b'}
        is_valid, messages, corrected = validator.validate('MAU', context)
        assert is_valid  # Warning doesn't block
        assert any('unusual for B2B' in msg for msg in messages)

        # B2C company with B2B metric (warning)
        context = {'company_type': 'b2c'}
        is_valid, messages, corrected = validator.validate('MRR', context)
        assert is_valid
        assert any('unusual for B2C' in msg for msg in messages)

        # Correct alignment
        context = {'company_type': 'b2b'}
        is_valid, messages, corrected = validator.validate('MRR', context)
        assert is_valid
        assert len(messages) == 0

    def test_team_size_validator_cross_field(self):
        """Test team size validation with cross-field checks."""
        validator = TeamSizeSmartValidator()

        # Idea stage with large team (warning)
        context = {'company_stage': 'idea'}
        is_valid, messages, corrected = validator.validate(50, context)
        assert is_valid
        assert any('unusual for company stage' in msg for msg in messages)

        # Mature stage with tiny team (warning)
        context = {'company_stage': 'mature'}
        is_valid, messages, corrected = validator.validate(2, context)
        assert is_valid
        assert any('unusual for company stage' in msg for msg in messages)

        # Good alignment
        context = {'company_stage': 'growth'}
        is_valid, messages, corrected = validator.validate(25, context)
        assert is_valid
        assert len(messages) == 0

    def test_url_validator_autocorrect(self):
        """Test URL validation with autocorrect."""
        validator = URLSmartValidator()

        # Autocorrect missing protocol
        is_valid, messages, corrected = validator.validate('acme.com', {})
        assert is_valid
        assert corrected == 'https://acme.com'

        # Already has protocol
        is_valid, messages, corrected = validator.validate('https://acme.com', {})
        assert is_valid
        assert corrected == 'https://acme.com'

        # Invalid URL (error)
        is_valid, messages, corrected = validator.validate('not a url', {})
        assert not is_valid
        assert any('Invalid URL' in msg for msg in messages)

    def test_smart_prompt_validator_adapter(self):
        """Test prompt_toolkit adapter for smart validators."""
        smart_validator = CompanyNameSmartValidator()
        prompt_validator = SmartPromptValidator(smart_validator)

        # Valid input - no exception
        doc = Document('Acme Corp')
        prompt_validator.validate(doc)  # Should not raise

        # Invalid input - raises ValidationError
        doc = Document('A')  # Too short
        with pytest.raises(ValidationError) as exc_info:
            prompt_validator.validate(doc)
        assert 'at least 2 characters' in str(exc_info.value)

    def test_apply_autocorrect_function(self):
        """Test the apply_autocorrect helper function."""
        # Company name whitespace
        corrected = apply_autocorrect('company_name', '  Acme  Corp  ')
        assert corrected == 'Acme Corp'

        # Metric typo
        corrected = apply_autocorrect('north_star_metric', 'MMR')
        assert corrected == 'MRR'

        # URL missing protocol
        corrected = apply_autocorrect('website', 'acme.com')
        assert corrected == 'https://acme.com'

        # Unknown field - no change
        corrected = apply_autocorrect('unknown_field', 'test value')
        assert corrected == 'test value'

    def test_validate_and_fix_function(self):
        """Test the validate_and_fix helper function."""
        # Valid company name
        is_valid, messages, corrected = validate_and_fix(
            'company_name',
            'Acme Corp',
            {}
        )
        assert is_valid
        assert len(messages) == 0

        # Invalid with autocorrect
        is_valid, messages, corrected = validate_and_fix(
            'company_name',
            '  test  ',
            {}
        )
        assert is_valid  # Warning doesn't block
        assert corrected == 'test'
        assert any('test data' in msg for msg in messages)

        # Cross-field validation
        context = {'company_type': 'b2b'}
        is_valid, messages, corrected = validate_and_fix(
            'north_star_metric',
            'MAU',
            context
        )
        assert is_valid
        assert any('unusual for B2B' in msg for msg in messages)


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestManualInputIntegration:
    """Integration tests for manual input with onboarding."""

    def test_review_and_edit_flow(self, manual_form, sample_enriched_data):
        """Test the full review and edit flow."""
        # Mock the session.prompt directly on the instance
        mock_prompt = MagicMock()
        manual_form.session.prompt = mock_prompt

        # Mock confirm to return False (don't want to review)
        with patch('pmkit.agents.manual_input.confirm', return_value=False):
            # Simulate user fixing website and confirming done
            mock_prompt.side_effect = [
                'next',  # Go to next field needing attention
                'https://acme.com',  # Fix website (needs review due to missing protocol)
                'done',  # Finish editing
            ]

            # Run review and edit
            updated_data = manual_form.review_and_edit(
                sample_enriched_data,
                company_type='b2b',
                required_only=False
            )

            # Website should be updated with protocol
            assert 'acme.com' in str(updated_data.get('website', ''))
            assert updated_data['company_name'] == 'Acme Corp'

    def test_collect_missing_fields(self, manual_form):
        """Test collecting only missing required fields."""
        # Mock the session.prompt directly on the instance
        mock_prompt = MagicMock()
        manual_form.session.prompt = mock_prompt

        # Existing data with missing required fields
        existing_data = {
            'company_name': 'Acme Corp',
            'company_type': 'b2b',
            # Missing: product_name, product_description
        }

        # Mock user inputs for missing fields
        mock_prompt.side_effect = [
            'AcmeFlow',  # product_name
            'Workflow automation platform for enterprise engineering teams',  # product_description
        ]

        # Collect missing fields for phase 1
        updated_data = manual_form.collect_missing_fields(
            existing_data,
            company_type='b2b',
            phase=1
        )

        # Should have all required phase 1 fields
        assert updated_data['product_name'] == 'AcmeFlow'
        assert 'enterprise engineering teams' in updated_data['product_description']
        assert updated_data['company_name'] == 'Acme Corp'  # Preserved

    def test_field_metadata_definitions(self):
        """Test that field definitions are properly configured."""
        # Check phase 1 required fields
        phase1_required = [
            name for name, meta in FIELD_DEFINITIONS.items()
            if meta.phase == 1 and meta.required
        ]
        assert 'company_name' in phase1_required
        assert 'company_type' in phase1_required
        assert 'product_name' in phase1_required
        assert 'product_description' in phase1_required

        # Check phase 2 fields are optional
        phase2_fields = [
            name for name, meta in FIELD_DEFINITIONS.items()
            if meta.phase == 2
        ]
        for field_name in phase2_fields:
            assert not FIELD_DEFINITIONS[field_name].required

        # Check B2B specific fields
        b2b_fields = [
            name for name, meta in FIELD_DEFINITIONS.items()
            if meta.b2b_specific
        ]
        assert 'pricing_model' in b2b_fields