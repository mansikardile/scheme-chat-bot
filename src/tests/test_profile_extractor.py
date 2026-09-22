import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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


def test_occupations_and_special_conditions():
    # Weaver
    f_weaver = extract_profile_fields_fast("i am a handloom weaver from maharashtra")
    assert f_weaver.get('occupation') == 'weaver'
    assert f_weaver.get('special_condition') == 'handloom_weaver'
    assert f_weaver.get('state') == 'Maharashtra'

    # Artisan
    f_artisan = extract_profile_fields_fast("i am a traditional artisan and craftsman")
    assert f_artisan.get('occupation') == 'artisan'
    assert f_artisan.get('special_condition') == 'traditional_artisan'

    # Business owner
    f_biz = extract_profile_fields_fast("i am a shopkeeper running a small business")
    assert f_biz.get('occupation') == 'business_owner'
    assert f_biz.get('employment_status') == 'self_employed'

    # Street vendor
    f_vendor = extract_profile_fields_fast("i am a street vendor")
    assert f_vendor.get('occupation') == 'street_vendor'
    assert f_vendor.get('special_condition') == 'street_vendor'


def test_profile_advisor_adaptive_flow():
    from backend.profile_advisor import detect_intent_area, get_next_question

    # For weaver, should detect weaver_artisan and not ask for education/course
    intent = detect_intent_area([], "i am a weaver looking for schemes")
    assert intent == 'weaver_artisan'

    # Next question for weaver with state known
    profile = {'state': 'Maharashtra', 'occupation': 'weaver'}
    field, q = get_next_question(profile, [], "i am a weaver")
    assert field in ('gender', 'category', 'annual_family_income', 'age')
    assert field not in ('education_level', 'course', 'study_stage')


