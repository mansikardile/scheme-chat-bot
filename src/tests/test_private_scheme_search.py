import pytest
import asyncio
from backend.private_scheme_search import (
    search_private_schemes_ai,
    get_private_scheme_detail,
    get_private_scheme_context,
    PRIVATE_SCHEMES_CACHE,
)
from backend.eligibility_engine import (
    evaluate_scheme,
    EligibilityRule,
    EligibilityStatus,
)


@pytest.mark.anyio
async def test_search_private_schemes_structure():
    profile = {
        'state': 'Maharashtra',
        'gender': 'Female',
        'category': 'EWS',
        'education_level': 'undergraduate',
        'course': 'engineering',
        'study_stage': 'direct_second_year',
        'annual_family_income': 400000,
        'age': 20,
    }

    results = await search_private_schemes_ai(profile)
    assert isinstance(results, list)
    if results:
        first = results[0]
        assert 'slug' in first
        assert 'schemeName' in first
        assert first.get('level') == 'Private / Trust'
        assert first.get('is_private') is True
        assert 'briefDescription' in first

        # Check detail retrieval
        slug = first['slug']
        detail = get_private_scheme_detail(slug)
        assert detail is not None

        context = get_private_scheme_context(slug)
        assert context is not None
        assert first['schemeName'] in context


def test_private_scheme_eligibility_rules():
    # Test Lila Poonawalla rule evaluation with eligibility engine
    rules = [
        EligibilityRule(field='state', operator='IN', value=['Maharashtra'], mandatory=True, raw_text="State: Maharashtra"),
        EligibilityRule(field='gender', operator='IN', value=['Female'], mandatory=True, raw_text="Gender: Female"),
        EligibilityRule(field='annual_family_income', operator='<=', value=400000, mandatory=True, raw_text="Income <= 400000"),
        EligibilityRule(field='education_level', operator='IN', value=['undergraduate'], mandatory=True, raw_text="Education: undergraduate"),
    ]

    matching_profile = {
        'state': 'Maharashtra',
        'gender': 'Female',
        'category': 'EWS',
        'education_level': 'undergraduate',
        'course': 'engineering',
        'annual_family_income': 400000,
    }
    res = evaluate_scheme(matching_profile, rules)
    assert res.status == EligibilityStatus.ELIGIBLE

    # Male candidate should be strictly rejected
    male_profile = {
        'state': 'Maharashtra',
        'gender': 'Male',
        'category': 'EWS',
        'education_level': 'undergraduate',
        'course': 'engineering',
        'annual_family_income': 400000,
    }
    res_male = evaluate_scheme(male_profile, rules)
    assert res_male.status == EligibilityStatus.INELIGIBLE

    # Assam state candidate should be strictly rejected
    assam_profile = {
        'state': 'Assam',
        'gender': 'Female',
        'category': 'EWS',
        'education_level': 'undergraduate',
        'course': 'engineering',
        'annual_family_income': 400000,
    }
    res_assam = evaluate_scheme(assam_profile, rules)
    assert res_assam.status == EligibilityStatus.INELIGIBLE
