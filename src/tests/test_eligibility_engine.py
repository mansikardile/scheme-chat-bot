"""
Automated tests for the SchemeSathi eligibility engine.

Covers all 6 test cases from the specification.
Run with: .venv\\Scripts\\python.exe -m pytest src/tests/test_eligibility_engine.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from backend.eligibility_engine import (
    evaluate_scheme,
    EligibilityRule,
    EligibilityStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rule(field, operator, value, mandatory=True) -> EligibilityRule:
    return EligibilityRule(field=field, operator=operator, value=value, mandatory=mandatory)


# ---------------------------------------------------------------------------
# Test cases from specification
# ---------------------------------------------------------------------------

class TestEligibilityEngineSpec:
    """All 6 test cases from the spec document."""

    def test_case_1_eligible(self):
        """
        Test 1: Maharashtra + Female + EWS + Engineering + DSY
        vs scheme requiring Maharashtra + Female + EWS + Undergraduate
        Expected: ELIGIBLE
        """
        user = {
            'state': 'Maharashtra',
            'gender': 'Female',
            'category': 'EWS',
            'education_level': 'undergraduate',
            'study_stage': 'direct_second_year',
            'course': 'engineering',
        }
        rules = [
            make_rule('state', 'IN', ['Maharashtra']),
            make_rule('gender', 'IN', ['Female']),
            make_rule('category', 'IN', ['EWS']),
            make_rule('education_level', 'IN', ['undergraduate']),
        ]
        result = evaluate_scheme(user, rules)
        assert result.status == EligibilityStatus.ELIGIBLE
        assert len(result.failed_criteria) == 0
        assert len(result.match_reasons) >= 4

    def test_case_2_ineligible_category(self):
        """
        Test 2: Maharashtra + Female + EWS user vs ST-only scheme
        Expected: INELIGIBLE (category mismatch)
        """
        user = {
            'state': 'Maharashtra',
            'gender': 'Female',
            'category': 'EWS',
        }
        rules = [
            make_rule('category', 'IN', ['ST']),
            make_rule('state', 'IN', ['Maharashtra']),
        ]
        result = evaluate_scheme(user, rules)
        assert result.status == EligibilityStatus.INELIGIBLE
        assert 'category' in result.failed_criteria
        assert result.rejection_reason is not None
        assert 'EWS' in result.rejection_reason or 'category' in result.rejection_reason.lower()

    def test_case_3_ineligible_education_level(self):
        """
        Test 3: Maharashtra + Female + EWS + Engineering DSY
        vs scheme for 5th-7th standard (class 5-7 = primary/upper_primary)
        Expected: INELIGIBLE (education level mismatch)
        """
        user = {
            'state': 'Maharashtra',
            'gender': 'Female',
            'category': 'EWS',
            'education_level': 'undergraduate',
            'study_stage': 'direct_second_year',
        }
        rules = [
            make_rule('education_level', 'IN', ['primary', 'upper_primary']),
            make_rule('state', 'IN', ['Maharashtra']),
        ]
        result = evaluate_scheme(user, rules)
        assert result.status == EligibilityStatus.INELIGIBLE
        assert 'education_level' in result.failed_criteria

    def test_case_4_insufficient_income_unknown(self):
        """
        Test 4: Maharashtra + Female + EWS + Engineering, income UNKNOWN
        vs scheme requiring income <= 2 lakh
        Expected: INSUFFICIENT_INFORMATION
        """
        user = {
            'state': 'Maharashtra',
            'gender': 'Female',
            'category': 'EWS',
            'education_level': 'undergraduate',
            'annual_family_income': None,  # Unknown
        }
        rules = [
            make_rule('state', 'IN', ['Maharashtra']),
            make_rule('annual_family_income', '<=', 200000),
        ]
        result = evaluate_scheme(user, rules)
        assert result.status == EligibilityStatus.INSUFFICIENT_INFORMATION
        assert 'annual_family_income' in result.missing_fields

    def test_case_5_eligible_income_within_limit(self):
        """
        Test 5: Income = 1.5 lakh vs scheme requiring income <= 2 lakh
        Expected: ELIGIBLE
        """
        user = {
            'state': 'Maharashtra',
            'gender': 'Female',
            'category': 'EWS',
            'annual_family_income': 150000,  # ₹1.5 lakh
        }
        rules = [
            make_rule('state', 'IN', ['Maharashtra']),
            make_rule('annual_family_income', '<=', 200000),
        ]
        result = evaluate_scheme(user, rules)
        assert result.status == EligibilityStatus.ELIGIBLE

    def test_case_6_ineligible_income_exceeds(self):
        """
        Test 6: Income = 4 lakh vs scheme requiring income <= 2 lakh
        Expected: INELIGIBLE
        """
        user = {
            'state': 'Maharashtra',
            'gender': 'Female',
            'category': 'EWS',
            'annual_family_income': 400000,  # ₹4 lakh
        }
        rules = [
            make_rule('state', 'IN', ['Maharashtra']),
            make_rule('annual_family_income', '<=', 200000),
        ]
        result = evaluate_scheme(user, rules)
        assert result.status == EligibilityStatus.INELIGIBLE
        assert 'annual_family_income' in result.failed_criteria


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------

class TestEligibilityEngineEdgeCases:

    def test_national_overseas_scholarship_st_rejects_ews(self):
        """The exact bug from the spec: NOS for ST must reject EWS user."""
        user = {
            'state': 'Maharashtra',
            'gender': 'Female',
            'category': 'EWS',
            'education_level': 'undergraduate',
        }
        # National Overseas Scholarship for ST
        rules = [
            make_rule('category', 'IN', ['ST']),
        ]
        result = evaluate_scheme(user, rules)
        assert result.status == EligibilityStatus.INELIGIBLE, (
            "ST-only scheme MUST reject EWS user — this is the core bug being fixed"
        )

    def test_income_range_straddles_limit_is_insufficient(self):
        """If user says '3-4 lakh' and limit is 3.5 lakh → INSUFFICIENT (range straddles)."""
        user = {
            'state': 'Maharashtra',
            'annual_family_income': [300000, 400000],  # 3-4 lakh range
        }
        rules = [
            make_rule('state', 'IN', ['Maharashtra']),
            make_rule('annual_family_income', '<=', 350000),  # 3.5 lakh limit
        ]
        result = evaluate_scheme(user, rules)
        assert result.status == EligibilityStatus.INSUFFICIENT_INFORMATION

    def test_income_range_clearly_exceeds(self):
        """If user says '5-6 lakh' and limit is 3 lakh → INELIGIBLE (lower bound exceeds)."""
        user = {
            'state': 'Maharashtra',
            'annual_family_income': [500000, 600000],
        }
        rules = [
            make_rule('state', 'IN', ['Maharashtra']),
            make_rule('annual_family_income', '<=', 300000),
        ]
        result = evaluate_scheme(user, rules)
        assert result.status == EligibilityStatus.INELIGIBLE

    def test_optional_rule_does_not_block(self):
        """An optional (mandatory=False) rule that FAILS should NOT block eligibility."""
        user = {
            'state': 'Maharashtra',
            'gender': 'Male',
        }
        rules = [
            make_rule('state', 'IN', ['Maharashtra'], mandatory=True),
            make_rule('gender', 'IN', ['Female'], mandatory=False),  # preference, not required
        ]
        result = evaluate_scheme(user, rules)
        # Should be ELIGIBLE because the only mandatory rule (state) passes
        assert result.status == EligibilityStatus.ELIGIBLE

    def test_age_between(self):
        """Age BETWEEN evaluation."""
        user = {'state': 'Delhi', 'age': 25}
        rules = [
            make_rule('state', 'IN', ['Delhi']),
            make_rule('age', 'BETWEEN', [18, 50]),
        ]
        assert evaluate_scheme(user, rules).status == EligibilityStatus.ELIGIBLE

        user_too_young = {'state': 'Delhi', 'age': 15}
        result = evaluate_scheme(user_too_young, rules)
        assert result.status == EligibilityStatus.INELIGIBLE

    def test_state_mismatch(self):
        """Wrong state → INELIGIBLE."""
        user = {'state': 'Kerala', 'category': 'OBC'}
        rules = [make_rule('state', 'IN', ['Maharashtra'])]
        result = evaluate_scheme(user, rules)
        assert result.status == EligibilityStatus.INELIGIBLE

    def test_no_mandatory_rules_is_eligible(self):
        """A scheme with no mandatory rules → ELIGIBLE for any user."""
        user = {'state': 'Maharashtra'}
        rules = [make_rule('gender', 'IN', ['Female'], mandatory=False)]
        result = evaluate_scheme(user, rules)
        assert result.status == EligibilityStatus.ELIGIBLE

    def test_missing_state_is_insufficient(self):
        """If state is mandatory but user hasn't told us yet → INSUFFICIENT."""
        user = {'gender': 'Female', 'category': 'EWS'}  # no state
        rules = [
            make_rule('state', 'IN', ['Maharashtra']),
            make_rule('category', 'IN', ['EWS']),
        ]
        result = evaluate_scheme(user, rules)
        assert result.status == EligibilityStatus.INSUFFICIENT_INFORMATION
        assert 'state' in result.missing_fields

    def test_match_reasons_populated(self):
        """ELIGIBLE result should have human-readable match reasons."""
        user = {
            'state': 'Maharashtra',
            'gender': 'Female',
            'category': 'EWS',
            'annual_family_income': 250000,
        }
        rules = [
            make_rule('state', 'IN', ['Maharashtra']),
            make_rule('gender', 'IN', ['Female']),
            make_rule('category', 'IN', ['EWS']),
            make_rule('annual_family_income', '<=', 300000),
        ]
        result = evaluate_scheme(user, rules)
        assert result.status == EligibilityStatus.ELIGIBLE
        assert len(result.match_reasons) >= 4
        # Reasons should contain checkmarks
        for reason in result.match_reasons:
            assert '✓' in reason

    def test_special_condition_construction_worker_rejects_regular_student(self):
        """MBOCWW scheme requiring registered construction worker dependent must reject student without that condition."""
        user = {
            'state': 'Maharashtra',
            'gender': 'Female',
            'category': 'EWS',
            'education_level': 'undergraduate',
            'occupation': 'student',
        }
        rules = [
            make_rule('state', 'IN', ['Maharashtra']),
            make_rule('special_condition', 'IN', ['construction_worker_dependent']),
        ]
        result = evaluate_scheme(user, rules)
        assert result.status == EligibilityStatus.INELIGIBLE
        assert 'special_condition' in result.failed_criteria

    def test_special_condition_landless_labourer_rejects_regular_student(self):
        """Aam Aadmi Bima Yojana requiring landless labourer head must reject regular student."""
        user = {
            'state': 'Maharashtra',
            'gender': 'Female',
            'category': 'EWS',
            'education_level': 'undergraduate',
            'occupation': 'student',
        }
        rules = [
            make_rule('state', 'IN', ['Maharashtra']),
            make_rule('special_condition', 'IN', ['landless_labourer']),
        ]
        result = evaluate_scheme(user, rules)
        assert result.status == EligibilityStatus.INELIGIBLE
        assert 'special_condition' in result.failed_criteria

    def test_vjnt_sbc_scheme_rejects_ews(self):
        """VJNT/SBC school scholarship must reject EWS student."""
        user = {
            'state': 'Maharashtra',
            'gender': 'Female',
            'category': 'EWS',
            'education_level': 'undergraduate',
        }
        rules = [
            make_rule('state', 'IN', ['Maharashtra']),
            make_rule('category', 'IN', ['VJNT', 'SBC', 'NT']),
            make_rule('education_level', 'IN', ['primary', 'upper_primary', 'secondary']),
        ]
        result = evaluate_scheme(user, rules)
        assert result.status == EligibilityStatus.INELIGIBLE
    def test_old_age_home_rejects_young_student(self):
        """Grant in Aid to Old Age Home requires senior citizen (age >= 60) or disabled; rejects 20yo non-disabled student."""
        user = {
            'state': 'Maharashtra',
            'gender': 'Female',
            'category': 'EWS',
            'education_level': 'undergraduate',
            'disability_status': 'non_disabled',
            'age': 20,
        }
        rules = [
            make_rule('state', 'IN', ['Maharashtra']),
            make_rule('age', '>=', 60),
            make_rule('disability_status', 'IN', ['disabled']),
        ]
        result = evaluate_scheme(user, rules)
        assert result.status == EligibilityStatus.INELIGIBLE
        assert 'age' in result.failed_criteria or 'disability_status' in result.failed_criteria

    def test_disabled_scholarship_rejects_non_disabled_student(self):
        """State Pre-matric Scholarship for Disabled requires disability_status == disabled."""
        user = {
            'state': 'Maharashtra',
            'gender': 'Female',
            'category': 'EWS',
            'education_level': 'undergraduate',
            'disability_status': 'non_disabled',
            'age': 20,
        }
        rules = [
            make_rule('state', 'IN', ['Maharashtra']),
            make_rule('disability_status', 'IN', ['disabled']),
        ]
        result = evaluate_scheme(user, rules)
        assert result.status == EligibilityStatus.INELIGIBLE
        assert 'disability_status' in result.failed_criteria

    def test_majhi_ladki_bahin_rejects_20yo_or_4lpa(self):
        """Mukhyamantri Majhi Ladki Bahin Yojana requires age 21 to 65 and income <= 2.5 LPA."""
        user = {
            'state': 'Maharashtra',
            'gender': 'Female',
            'category': 'EWS',
            'education_level': 'undergraduate',
            'annual_family_income': 400000,
            'age': 20,
        }
        rules = [
            make_rule('state', 'IN', ['Maharashtra']),
            make_rule('gender', 'IN', ['Female']),
            make_rule('age', 'BETWEEN', [21, 65]),
            make_rule('annual_family_income', '<=', 250000),
        ]
        result = evaluate_scheme(user, rules)
        assert result.status == EligibilityStatus.INELIGIBLE
        assert 'age' in result.failed_criteria or 'annual_family_income' in result.failed_criteria


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

