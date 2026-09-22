"""
Profile Advisor for SchemeSathi.

Implements adaptive questioning — asks only the most important missing
profile field based on the user's evident intent/interest area.

Does NOT use LLM — this is pure lookup-table logic.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Field question templates (what the bot asks when a field is missing)
# ---------------------------------------------------------------------------

FIELD_QUESTIONS: dict[str, str] = {
    'state': "Which state are you from? (e.g. Maharashtra, Delhi, Karnataka…)",
    'gender': "What is your gender? (Male / Female / Transgender)",
    'category': "What is your caste category? (General / SC / ST / OBC / EWS / Minority)",
    'education_level': "What is your current education level? (e.g. Undergraduate, 12th, Diploma, Postgraduate…)",
    'course': "What course or field are you studying? (e.g. Engineering, Medical, Arts, Commerce…)",
    'study_stage': "Are you in your first year, or did you take a lateral/direct second-year admission (DSY)?",
    'year_of_study': "Which year of your course are you currently in?",
    'annual_family_income': "What is your approximate annual family income? (e.g. ₹3 lakh, ₹8 lakh per year)",
    'age': "How old are you?",
    'disability_status': "Do you have any disability (Divyang/PwD)?",
    'minority_status': "Do you belong to a minority community (Muslim, Christian, Sikh, Jain, Buddhist, Parsi)?",
    'residential_status': "Are you a permanent resident/domicile of the state you mentioned?",
    'institution_type': "Is your college/institution government-aided, private unaided, or autonomous?",
    'district': "Which district are you from? (Some schemes are district-specific)",
    'marital_status': "What is your marital status? (Single / Married / Widow / Widower)",
    'employment_status': "Are you currently employed, unemployed, or self-employed?",
    'occupation': "What is your occupation? (e.g. Farmer, Teacher, Business owner…)",
    'farmer_status': "Are you a registered farmer? Do you have a PM-KISAN beneficiary ID?",
    'land_holding': "How much agricultural land do you own (in acres or hectares)?",
    'housing_status': "What is your current housing situation? (Own house, rented, no house)",
}

# ---------------------------------------------------------------------------
# Intent → required fields (adaptive questioning)
# Fields are ordered: most important first
# ---------------------------------------------------------------------------

INTENT_REQUIRED_FIELDS: dict[str, list[str]] = {
    'weaver_artisan': [
        'state', 'gender', 'category', 'annual_family_income', 'age',
    ],
    'student_scholarship': [
        'state', 'gender', 'category', 'education_level', 'course',
        'study_stage', 'annual_family_income', 'disability_status', 'minority_status',
        'institution_type',
    ],
    'farmer': [
        'state', 'farmer_status', 'land_holding', 'annual_family_income',
        'category', 'age',
    ],
    'business': [
        'state', 'category', 'gender', 'annual_family_income', 'employment_status', 'age',
    ],
    'labour_worker': [
        'state', 'gender', 'category', 'annual_family_income', 'age',
    ],
    'women_welfare': [
        'state', 'category', 'age', 'marital_status', 'annual_family_income',
        'employment_status',
    ],
    'housing': [
        'state', 'annual_family_income', 'housing_status', 'category', 'age',
    ],
    'employment': [
        'state', 'age', 'education_level', 'category', 'employment_status',
        'annual_family_income',
    ],
    'health': [
        'state', 'annual_family_income', 'category', 'age',
    ],
    'disability': [
        'state', 'disability_status', 'category', 'annual_family_income', 'age',
    ],
    'senior_citizen': [
        'state', 'age', 'annual_family_income', 'category',
    ],
    'general': [
        'state', 'gender', 'category', 'annual_family_income', 'age',
    ],
}

# Keywords that indicate intent area (ordered by priority/specificity)
_INTENT_KEYWORDS: list[tuple[str, str]] = [
    ('weaver', 'weaver_artisan'), ('bunkar', 'weaver_artisan'), ('vankar', 'weaver_artisan'),
    ('handloom', 'weaver_artisan'), ('powerloom', 'weaver_artisan'), ('textile', 'weaver_artisan'),
    ('artisan', 'weaver_artisan'), ('craftsman', 'weaver_artisan'), ('karigar', 'weaver_artisan'),
    ('potter', 'weaver_artisan'), ('blacksmith', 'weaver_artisan'), ('carpenter', 'weaver_artisan'),
    ('sculptor', 'weaver_artisan'), ('cobbler', 'weaver_artisan'), ('tailor', 'weaver_artisan'),
    ('vishwakarma', 'weaver_artisan'),
    ('farmer', 'farmer'), ('kisan', 'farmer'), ('agri', 'farmer'), ('crop', 'farmer'),
    ('dairy', 'farmer'), ('poultry', 'farmer'), ('fisher', 'farmer'), ('matsya', 'farmer'),
    ('student', 'student_scholarship'), ('scholarship', 'student_scholarship'),
    ('education', 'student_scholarship'), ('study', 'student_scholarship'),
    ('college', 'student_scholarship'), ('engineer', 'student_scholarship'),
    ('medical', 'student_scholarship'), ('university', 'student_scholarship'),
    ('business', 'business'), ('entrepreneur', 'business'), ('startup', 'business'),
    ('shopkeeper', 'business'), ('trader', 'business'), ('merchant', 'business'),
    ('vendor', 'business'), ('street vendor', 'business'), ('hawker', 'business'),
    ('thelawala', 'business'), ('svanidhi', 'business'), ('mudra', 'business'),
    ('loan', 'business'), ('msme', 'business'),
    ('construction worker', 'labour_worker'), ('labour', 'labour_worker'), ('labor', 'labour_worker'),
    ('mazdoor', 'labour_worker'), ('daily wage', 'labour_worker'), ('mason', 'labour_worker'),
    ('eshram', 'labour_worker'), ('bocw', 'labour_worker'),
    ('women', 'women_welfare'), ('girl', 'women_welfare'), ('widow', 'women_welfare'),
    ('mahila', 'women_welfare'), ('female', 'women_welfare'),
    ('housing', 'housing'), ('house', 'housing'), ('awas', 'housing'), ('home', 'housing'),
    ('job', 'employment'), ('employ', 'employment'), ('work', 'employment'),
    ('unemployment', 'employment'), ('skill', 'employment'),
    ('health', 'health'), ('hospital', 'health'),
    ('insurance', 'health'), ('ayushman', 'health'),
    ('disabled', 'disability'), ('divyang', 'disability'), ('pwd', 'disability'),
    ('handicap', 'disability'),
    ('senior', 'senior_citizen'), ('elderly', 'senior_citizen'),
    ('pension', 'senior_citizen'), ('old age', 'senior_citizen'),
]


def detect_intent_area(conversation_history: list[dict], current_message: str) -> str:
    """Detect the user's primary interest area from conversation history."""
    combined_text = current_message.lower()
    for msg in conversation_history[-6:]:
        if msg.get('role') == 'user':
            combined_text += ' ' + msg.get('content', '').lower()

    for keyword, area in _INTENT_KEYWORDS:
        if keyword in combined_text:
            return area
    return 'general'


def get_next_question(
    user_profile: dict,
    conversation_history: list[dict],
    current_message: str,
) -> tuple[str | None, str | None]:
    """
    Determine the single most important missing profile field to ask about next.

    Returns:
        (field_name, question_text) — or (None, None) if profile is sufficiently complete.
    """
    intent_area = detect_intent_area(conversation_history, current_message)
    required_fields = INTENT_REQUIRED_FIELDS.get(intent_area, INTENT_REQUIRED_FIELDS['general'])

    for field in required_fields:
        val = user_profile.get(field)
        if val is None:
            question = FIELD_QUESTIONS.get(field, f"Could you tell me your {field.replace('_', ' ')}?")
            return field, question

    return None, None


def build_profile_summary_text(user_profile: dict) -> tuple[str, list[str]]:
    """
    Build a human-readable profile summary for display.

    Returns:
        (collected_text, missing_list) — what we know and what's still needed.
    """
    LABELS = {
        'state': 'State', 'gender': 'Gender', 'category': 'Category',
        'education_level': 'Education level', 'course': 'Course',
        'study_stage': 'Study stage', 'year_of_study': 'Year of study',
        'annual_family_income': 'Annual family income', 'age': 'Age',
        'disability_status': 'Disability', 'minority_status': 'Minority status',
        'residential_status': 'Residential status', 'institution_type': 'Institution type',
        'district': 'District', 'marital_status': 'Marital status',
        'employment_status': 'Employment status', 'occupation': 'Occupation',
        'farmer_status': 'Farmer status', 'land_holding': 'Agricultural land holding',
    }

    collected = []
    for field, label in LABELS.items():
        val = user_profile.get(field)
        if val is not None:
            if field == 'annual_family_income':
                if isinstance(val, (list, tuple)):
                    display = f"₹{val[0]//100_000:.1f}L – ₹{val[1]//100_000:.1f}L/year"
                elif isinstance(val, (int, float)):
                    display = f"₹{val/100_000:.1f}L/year"
                else:
                    display = str(val)
            elif field == 'land_holding':
                if isinstance(val, (int, float)):
                    display = f"{val} acre(s)" if val > 0 else "Landless / 0 acres"
                else:
                    display = str(val)
            elif field == 'farmer_status':
                display = "Registered Farmer" if val == 'registered_farmer' else str(val).replace('_', ' ').title()
            elif field == 'disability_status':
                display = "No (Non-disabled)" if val == 'non-disabled' else ("Yes (Divyang / PwD)" if val == 'disabled' else str(val))
            elif field == 'minority_status':
                display = "No (Non-minority)" if val == 'non-minority' else ("Yes (Minority community)" if val == 'minority' else str(val))
            elif field == 'residential_status':
                display = "Permanent Resident / Domiciled" if str(val).startswith('resident') or val == 'permanent_resident' else str(val)
            elif field == 'study_stage':
                display = str(val).replace('_', ' ').title()
            elif field == 'institution_type':
                display = str(val).replace('_', ' ').title()
            else:
                display = str(val)
            collected.append(f"✓ {label}: {display}")

    return '\n'.join(collected), []


def format_profile_for_response(user_profile: dict, next_field: str | None) -> str:
    """
    Build the acknowledgment + next-question response text shown to the user
    when in profile collection mode.
    """
    collected_text, _ = build_profile_summary_text(user_profile)

    lines = []
    if collected_text:
        lines.append("Got it! Here's what I have so far:\n")
        lines.append(collected_text)
        lines.append("")

    if next_field:
        question = FIELD_QUESTIONS.get(next_field, f"What is your {next_field.replace('_', ' ')}?")
        lines.append(question)
    else:
        lines.append("I have enough information to search for schemes. Just say **\"Show me the schemes\"** when you're ready!")

    return '\n'.join(lines)
