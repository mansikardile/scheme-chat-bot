import pytest
from backend.profile_extractor import extract_profile_fields_fast, _extract_income
from backend.intent_classifier import classify_intent_fast, Intent


def test_income_extraction_formats():
    assert _extract_income("4lpa") == 400000
    assert _extract_income("4 lpa") == 400000
    assert _extract_income("income 4 lpa") == 400000
    assert _extract_income("4.5 lpa") == 450000
    assert _extract_income("4.5lpa") == 450000
    assert _extract_income("4 l.p.a") == 400000
    assert _extract_income("4 lakh") == 400000
    assert _extract_income("4 lakhs per year") == 400000
    assert _extract_income("4 lacs") == 400000
    assert _extract_income("3 to 4 lakh") == [300000, 400000]
    assert _extract_income("3-4 lpa") == [300000, 400000]
    assert _extract_income("₹4,00,000") == 400000
    assert _extract_income("Rs 400000") == 400000
    assert _extract_income("400000") == 400000
    assert _extract_income("below 8 lakh") == 800000
    assert _extract_income("under 2.5 lakh") == 250000
    assert _extract_income("income is 4") == 400000


def test_extract_profile_fields_fast():
    fields = extract_profile_fields_fast("4lpa")
    assert fields.get("annual_family_income") == 400000

    fields = extract_profile_fields_fast("income 4 lpa")
    assert fields.get("annual_family_income") == 400000

    fields = extract_profile_fields_fast("I am from Maharashtra, female, EWS category, direct second year engineering student")
    assert fields.get("state") == "Maharashtra"
    assert fields.get("gender") == "Female"
    assert fields.get("category") == "EWS"
    assert fields.get("study_stage") == "direct_second_year"
    assert fields.get("education_level") == "undergraduate"
    assert fields.get("course") == "engineering"


def test_intent_classification():
    assert classify_intent_fast("4lpa") in (Intent.PROFILE_INFO, Intent.CLARIFICATION)
    assert classify_intent_fast("income 4 lpa") in (Intent.PROFILE_INFO, Intent.CLARIFICATION)
    assert classify_intent_fast("give me schemes") == Intent.SCHEME_REQUEST
    assert classify_intent_fast("show me the schemes") == Intent.SCHEME_REQUEST
