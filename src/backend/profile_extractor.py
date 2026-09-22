"""
Profile field extractor for SchemeSathi.

Extracts structured user profile fields from a single conversational message.
Only fields EXPLICITLY mentioned in the message are returned — this module
never infers or assumes missing values.
"""

from __future__ import annotations

import re
import json

# ---------------------------------------------------------------------------
# Fast regex extraction (no LLM) — handles common patterns instantly
# ---------------------------------------------------------------------------

# State name list (lower-case for matching)
_INDIAN_STATES = [
    'andhra pradesh', 'arunachal pradesh', 'assam', 'bihar', 'chhattisgarh',
    'goa', 'gujarat', 'haryana', 'himachal pradesh', 'jharkhand', 'karnataka',
    'kerala', 'madhya pradesh', 'maharashtra', 'manipur', 'meghalaya', 'mizoram',
    'nagaland', 'odisha', 'punjab', 'rajasthan', 'sikkim', 'tamil nadu',
    'telangana', 'tripura', 'uttar pradesh', 'uttarakhand', 'west bengal',
    'delhi', 'jammu and kashmir', 'jammu & kashmir', 'ladakh', 'puducherry',
    'chandigarh', 'andaman and nicobar', 'lakshadweep', 'dadra', 'daman',
]
_STATE_SYNONYMS = {
    'up': 'Uttar Pradesh', 'u.p.': 'Uttar Pradesh',
    'mp': 'Madhya Pradesh', 'm.p.': 'Madhya Pradesh',
    'ap': 'Andhra Pradesh', 'wb': 'West Bengal',
    'tn': 'Tamil Nadu', 'j&k': 'Jammu and Kashmir',
    'mh': 'Maharashtra', 'maharastra': 'Maharashtra',
    'uk': 'Uttarakhand',
}
_STATE_TITLE = {s: s.title() for s in _INDIAN_STATES}
_STATE_TITLE.update({
    'tamil nadu': 'Tamil Nadu', 'andhra pradesh': 'Andhra Pradesh',
    'uttar pradesh': 'Uttar Pradesh', 'madhya pradesh': 'Madhya Pradesh',
    'arunachal pradesh': 'Arunachal Pradesh', 'himachal pradesh': 'Himachal Pradesh',
    'west bengal': 'West Bengal', 'jammu and kashmir': 'Jammu and Kashmir',
    'jammu & kashmir': 'Jammu and Kashmir', 'andaman and nicobar': 'Andaman and Nicobar',
})

_CATEGORY_MAP = {
    r'\bsc\b': 'SC', 'scheduled caste': 'SC', 'dalit': 'SC',
    r'\bst\b': 'ST', 'scheduled tribe': 'ST', 'adivasi': 'ST', 'tribal': 'ST',
    r'\bobc\b': 'OBC', 'other backward': 'OBC', 'backward class': 'OBC',
    r'\bews\b': 'EWS', 'economically weaker section': 'EWS', 'economically weak': 'EWS',
    'general category': 'General', 'open category': 'General',
    r'\bvjnt\b': 'VJNT', r'\bsbc\b': 'SBC', r'\bnt\b': 'NT',
    'minority': 'Minority', 'muslim': 'Minority', 'christian': 'Minority',
    'sikh': 'Minority', 'jain': 'Minority', 'buddhist': 'Minority',
    'disabled': 'Disabled', 'divyang': 'Disabled', r'\bpwd\b': 'Disabled',
}

_GENDER_MAP = {
    r'\bgirl\b': 'Female', r'\bfemale\b': 'Female', r'\bwoman\b': 'Female',
    r'\bwomen\b': 'Female', r'\bmahila\b': 'Female', r'\blady\b': 'Female',
    r'\bboy\b': 'Male', r'\bmale\b': 'Male', r'\bman\b': 'Male',
    r'\btransgender\b': 'Transgender',
}

_EDUCATION_MAP = {
    'direct second year': 'undergraduate', r'\bdsy\b': 'undergraduate',
    'lateral entry': 'undergraduate',
    r'\bphd\b': 'doctoral', 'ph.d': 'doctoral', 'doctorate': 'doctoral',
    r'\bmtech\b': 'postgraduate', 'm.tech': 'postgraduate',
    r'\bmba\b': 'postgraduate', r'\bmsc\b': 'postgraduate', r'\bma\b': 'postgraduate',
    r'\bbtech\b': 'undergraduate', 'b.tech': 'undergraduate',
    r'\bmbbs\b': 'undergraduate', r'\bllb\b': 'undergraduate',
    r'\bbe\b': 'undergraduate', 'b.e': 'undergraduate',
    r'\bbsc\b': 'undergraduate', 'b.sc': 'undergraduate',
    r'\bbcom\b': 'undergraduate', 'b.com': 'undergraduate',
    r'\bba\b': 'undergraduate', 'b.a': 'undergraduate',
    'undergraduate': 'undergraduate', 'graduation': 'undergraduate', 'degree': 'undergraduate',
    r'\bpg\b': 'postgraduate', 'post graduate': 'postgraduate', 'masters': 'postgraduate',
    r'\bdiploma\b': 'diploma', 'polytechnic': 'diploma', r'\biti\b': 'diploma',
    'class 12': 'higher_secondary', '12th': 'higher_secondary', r'\bhsc\b': 'higher_secondary',
    'class 11': 'higher_secondary', '11th': 'higher_secondary', 'intermediate': 'higher_secondary',
    'class 10': 'secondary', '10th': 'secondary', r'\bssc\b': 'secondary', 'matric': 'secondary',
    'class 9': 'secondary', '9th': 'secondary',
    'class 8': 'upper_primary', '8th': 'upper_primary',
    'class 7': 'upper_primary', '7th': 'upper_primary',
    'class 6': 'upper_primary', '6th': 'upper_primary',
    'class 5': 'primary', '5th': 'primary',
}

_STUDY_STAGE_MAP = {
    'direct second year': 'direct_second_year',
    r'\bdsy\b': 'direct_second_year',
    'lateral entry': 'direct_second_year',
    'first year': 'first_year', '1st year': 'first_year',
    'second year': 'second_year', '2nd year': 'second_year',
    'third year': 'third_year', '3rd year': 'third_year',
    'fourth year': 'fourth_year', '4th year': 'fourth_year',
    'final year': 'fourth_year',
}

_COURSE_MAP = {
    'engineering': 'engineering', 'computer science': 'engineering',
    r'\bcs\b': 'engineering', r'\bit\b': 'engineering',
    'mechanical': 'engineering', 'electrical': 'engineering',
    'civil': 'engineering', 'electronics': 'engineering',
    'medical': 'medical', 'mbbs': 'medical', 'nursing': 'medical',
    'pharmacy': 'medical', r'\bpharm\b': 'medical',
    'arts': 'arts', r'\bba\b': 'arts',
    'science': 'science', r'\bbsc\b': 'science',
    'commerce': 'commerce', r'\bbcom\b': 'commerce',
    'law': 'law', r'\bllb\b': 'law',
    'management': 'management', r'\bmba\b': 'management',
    'architecture': 'architecture',
}


def _extract_state(text_lower: str) -> str | None:
    for s in sorted(_INDIAN_STATES, key=len, reverse=True):
        if re.search(r'\b' + re.escape(s) + r'\b', text_lower):
            return _STATE_TITLE.get(s, s.title())
    for syn, full in _STATE_SYNONYMS.items():
        if re.search(r'\b' + re.escape(syn) + r'\b', text_lower):
            return full
    return None


def _extract_category(text_lower: str) -> str | None:
    for pattern, cat in _CATEGORY_MAP.items():
        if re.search(pattern, text_lower):
            return cat
    return None


def _extract_gender(text_lower: str) -> str | None:
    for pattern, gender in _GENDER_MAP.items():
        if re.search(pattern, text_lower):
            return gender
    return None


def _extract_education_level(text_lower: str) -> str | None:
    for pattern, level in _EDUCATION_MAP.items():
        if re.search(pattern, text_lower):
            return level
    return None


def _extract_study_stage(text_lower: str) -> str | None:
    for pattern, stage in _STUDY_STAGE_MAP.items():
        if re.search(pattern, text_lower):
            return stage
    return None


def _extract_course(text_lower: str) -> str | None:
    for pattern, course in _COURSE_MAP.items():
        if re.search(pattern, text_lower):
            return course
    return None


def _extract_income(text: str) -> object:
    """
    Extract annual family income from text.
    Returns a single number, a [lo, hi] range, or None.
    """
    text_lower = text.lower()

    # Range: "3 to 4 lakh", "3-4 lakh", "3-4 lpa", "3 to 4 lpa", "3-4 lacs", "3 to 4 lac"
    m = re.search(r'([\d.]+)\s*(?:to|-)\s*([\d.]+)\s*(?:lakhs?|lacs?|lac|lpa|l\.p\.a\.?|l\b)', text_lower)
    if m:
        return [int(float(m.group(1)) * 100_000), int(float(m.group(2)) * 100_000)]

    # Range with raw numbers: "300000 to 400000" or "3,00,000 - 4,00,000"
    m = re.search(r'(?:₹|rs\.?|inr)?\s*([\d,]{4,})\s*(?:to|-)\s*(?:₹|rs\.?|inr)?\s*([\d,]{4,})', text_lower)
    if m:
        try:
            return [int(m.group(1).replace(',', '')), int(m.group(2).replace(',', ''))]
        except ValueError:
            pass

    # "below 8 lakh", "under 8 lakh", "less than 8 lakh", "< 8 lakh", "up to 8 lakh", "upto 8 lakh", "less than 4lpa", "below 4 lpa"
    m = re.search(r'(?:below|under|less than|<|up to|upto|within|max|maximum)\s*([\d.]+)\s*(?:lakhs?|lacs?|lac|lpa|l\.p\.a\.?|l\b)', text_lower)
    if m:
        return int(float(m.group(1)) * 100_000)

    # LPA / L.P.A / Lakhs / Lacs / Lac: "4lpa", "4 lpa", "4.5 lpa", "4 lakh", "4.5 lakhs", "4 lac", "4 lacs", "4 lpa"
    m = re.search(r'([\d.]+)\s*(?:lakhs?|lacs?|lac|lpa|l\.p\.a\.?)\b', text_lower)
    if m:
        return int(float(m.group(1)) * 100_000)

    # Single "4l" or "4 l" (e.g. 4L per annum, 4L)
    m = re.search(r'([\d.]+)\s*l\b', text_lower)
    if m:
        return int(float(m.group(1)) * 100_000)

    # Crore: "1 crore", "1.5 cr"
    m = re.search(r'([\d.]+)\s*(?:crores?|cr\b)', text_lower)
    if m:
        return int(float(m.group(1)) * 10_000_000)

    # Explicit Rupee amount: ₹3,00,000 or Rs. 400000 or INR 4,00,000
    m = re.search(r'(?:₹|rs\.?|inr)\s*([\d,]+)', text_lower)
    if m:
        try:
            val = int(m.group(1).replace(',', ''))
            if val >= 1000:
                return val
        except ValueError:
            pass

    # Standalone 5-8 digit number: 400000 or 4,00,000
    m = re.search(r'\b([\d,]{5,8})\b', text_lower)
    if m:
        try:
            val = int(m.group(1).replace(',', ''))
            if 10000 <= val <= 100_000_000:
                return val
        except ValueError:
            pass

    # "income 4", "income: 4.5", "income is 4"
    m = re.search(r'(?:income|annual income|family income)\s*(?:is|:|=)?\s*([\d.]+)', text_lower)
    if m:
        try:
            val = float(m.group(1))
            if val <= 100:  # User probably meant lakhs e.g. "income 4" -> 400000
                return int(val * 100_000)
            else:
                return int(val)
        except ValueError:
            pass

    return None


def _extract_age(text_lower: str) -> int | None:
    m = re.search(r'(?:i am|age is|aged?|age:?)\s+(\d{1,2})', text_lower)
    if m:
        return int(m.group(1))
    m = re.search(r'(\d{1,2})\s*(?:years?|yrs?)(?:\s*old)?', text_lower)
    if m:
        return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_profile_fields_fast(message: str) -> dict:
    """
    Fast regex-based profile field extraction from a user message.

    Returns a dict with ONLY the fields that were explicitly found.
    Missing fields are not included (caller merges with existing profile).

    This runs synchronously with no LLM calls.
    """
    text_lower = message.lower()
    fields: dict = {}

    state = _extract_state(text_lower)
    if state:
        fields['state'] = state

    category = _extract_category(text_lower)
    if category:
        fields['category'] = category

    gender = _extract_gender(text_lower)
    if gender:
        fields['gender'] = gender

    edu = _extract_education_level(text_lower)
    if edu:
        fields['education_level'] = edu

    stage = _extract_study_stage(text_lower)
    if stage:
        fields['study_stage'] = stage

    course = _extract_course(text_lower)
    if course:
        fields['course'] = course

    income = _extract_income(message)
    if income is not None:
        fields['annual_family_income'] = income

    age = _extract_age(text_lower)
    if age is not None:
        fields['age'] = age

    # Disability
    if re.search(r'\bdisabled\b|\bdivyang\b|\bpwd\b|\bhandicap\b', text_lower):
        fields['disability_status'] = 'disabled'

    # Domicile / residential
    if re.search(r'\bdomi[cs]ile\b|\bresident\b|\bliving in\b|\bfrom\b', text_lower):
        # State was already captured; flag residential status
        if state:
            fields['residential_status'] = f'resident_{state.lower().replace(" ", "_")}'

    print(f"[ProfileExtractor] Extracted from message: {fields}")
    return fields


async def extract_profile_fields_llm(
    message: str,
    conversation_history: list[dict],
    model_id: str,
    api_key: str | None = None,
    model_config: dict | None = None,
) -> dict:
    """
    LLM-based profile field extraction for complex/ambiguous messages.
    Falls back to fast extraction on error.
    """
    PROMPT = """You are extracting user eligibility profile information from a conversational message.

Extract ONLY fields that are explicitly stated. Do NOT infer or guess missing fields.
Return a JSON object with only the detected fields.

Available fields:
- state: Indian state name (string, title-case)
- gender: "Male", "Female", or "Transgender"
- category: "SC", "ST", "OBC", "EWS", "General", "Minority", "Disabled", "VJNT", "SBC", "NT"
- education_level: "pre-primary", "primary", "secondary", "higher_secondary", "diploma", "undergraduate", "postgraduate", "doctoral"
- study_stage: "first_year", "second_year", "third_year", "fourth_year", "direct_second_year"
- course: e.g. "engineering", "medical", "arts", "science", "commerce", "law", "management"
- stream: e.g. "science", "arts", "commerce"
- annual_family_income: number in rupees, or [min, max] range
- age: integer years
- disability_status: "disabled" or "non-disabled"
- minority_status: "minority" or "non-minority"
- residential_status: e.g. "resident_maharashtra"
- employment_status: "employed", "unemployed", "self-employed", "student"
- marital_status: "single", "married", "widow", "widower"
- occupation: e.g. "farmer", "teacher", "entrepreneur"
- district: district name string

Special normalizations:
- "DSY" / "Direct Second Year" → study_stage: "direct_second_year" AND education_level: "undergraduate"
- "girl student" → gender: "Female" AND education_level detected from context
- "EWS" → category: "EWS"

Conversation context (last 4 messages):
{context}

Current message: "{message}"

JSON only, no markdown:"""

    context = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Bot'}: {m['content'][:200]}"
        for m in (conversation_history[-4:] if len(conversation_history) > 4 else conversation_history)
    )

    try:
        from backend.model_registry import get_model
        from langchain_core.messages import HumanMessage

        model = get_model(model_id, api_key=api_key, model_config=model_config)
        response = await model.ainvoke([HumanMessage(
            content=PROMPT.format(context=context or "None", message=message)
        )])

        content = response.content
        if isinstance(content, list):
            content = ''.join(
                item.get('text', '') if isinstance(item, dict) else str(item)
                for item in content
                if not (isinstance(item, dict) and item.get('type') in ('thinking', 'reasoning'))
            )

        json_match = re.search(r'\{.*?\}', str(content), re.DOTALL)
        if json_match:
            fields = json.loads(json_match.group())
            fields = {k: v for k, v in fields.items() if v is not None and v != ''}
            print(f"[ProfileExtractor] LLM extracted: {fields}")
            return fields
    except Exception as e:
        print(f"[ProfileExtractor] LLM extraction failed: {e}, using fast fallback")

    return extract_profile_fields_fast(message)
