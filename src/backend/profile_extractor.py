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
    r'\bsc\b': 'SC', r'\bscheduled caste\b': 'SC', r'\bdalit\b': 'SC',
    r'\bst\b': 'ST', r'\bscheduled tribe\b': 'ST', r'\badivasi\b': 'ST', r'\btribal\b': 'ST',
    r'\bobc\b': 'OBC', r'\bother backward\b': 'OBC', r'\bbackward class\b': 'OBC',
    r'\bews\b': 'EWS', r'\beconomically weaker section\b': 'EWS', r'\beconomically weak\b': 'EWS',
    r'\bgeneral category\b': 'General', r'\bopen category\b': 'General',
    r'\bvjnt\b': 'VJNT', r'\bsbc\b': 'SBC', r'\bnt\b': 'NT',
    r'(?<!non-)(?<!not )\bminority\b': 'Minority', r'\bmuslim\b': 'Minority', r'\bchristian\b': 'Minority',
    r'\bsikh\b': 'Minority', r'\bjain\b': 'Minority', r'\bbuddhist\b': 'Minority',
    r'(?<!non-)(?<!not )\bdisabled\b': 'Disabled', r'(?<!non-)(?<!not )\bdivyang\b': 'Disabled', r'(?<!non-)(?<!not )\bpwd\b': 'Disabled',
}

_GENDER_MAP = {
    r'\bgirl\b': 'Female', r'\bfemale\b': 'Female', r'\bwoman\b': 'Female',
    r'\bwomen\b': 'Female', r'\bmahila\b': 'Female', r'\blady\b': 'Female',
    r'\bboy\b': 'Male', r'\bmale\b': 'Male', r'\bman\b': 'Male',
    r'\btransgender\b': 'Transgender',
}

_EDUCATION_MAP = {
    r'\bdirect second year\b': 'undergraduate', r'\bdsy\b': 'undergraduate',
    r'\blateral entry\b': 'undergraduate',
    r'\bphd\b': 'doctoral', r'\bph\.d\b': 'doctoral', r'\bdoctorate\b': 'doctoral',
    r'\bmtech\b': 'postgraduate', r'\bm\.tech\b': 'postgraduate',
    r'\bmba\b': 'postgraduate', r'\bmsc\b': 'postgraduate', r'\bma\b': 'postgraduate',
    r'\bbtech\b': 'undergraduate', r'\bb\.tech\b': 'undergraduate',
    r'\bmbbs\b': 'undergraduate', r'\bllb\b': 'undergraduate',
    r'\bbe\b': 'undergraduate', r'\bb\.e\b': 'undergraduate',
    r'\bbsc\b': 'undergraduate', r'\bb\.sc\b': 'undergraduate',
    r'\bbcom\b': 'undergraduate', r'\bb\.com\b': 'undergraduate',
    r'\bba\b': 'undergraduate', r'\bb\.a\b': 'undergraduate',
    r'\bundergraduate\b': 'undergraduate', r'\bgraduation\b': 'undergraduate', r'\bdegree\b': 'undergraduate',
    r'\bpg\b': 'postgraduate', r'\bpost graduate\b': 'postgraduate', r'\bmasters\b': 'postgraduate',
    r'\bdiploma\b': 'diploma', r'\bpolytechnic\b': 'diploma', r'\biti\b': 'diploma',
    r'\bclass 12\b': 'higher_secondary', r'\b12th\b': 'higher_secondary', r'\bhsc\b': 'higher_secondary',
    r'\bclass 11\b': 'higher_secondary', r'\b11th\b': 'higher_secondary', r'\bintermediate\b': 'higher_secondary',
    r'\bclass 10\b': 'secondary', r'\b10th\b': 'secondary', r'\bssc\b': 'secondary', r'\bmatric\b': 'secondary',
    r'\bclass 9\b': 'secondary', r'\b9th\b': 'secondary',
    r'\bclass 8\b': 'upper_primary', r'\b8th\b': 'upper_primary',
    r'\bclass 7\b': 'upper_primary', r'\b7th\b': 'upper_primary',
    r'\bclass 6\b': 'upper_primary', r'\b6th\b': 'upper_primary',
    r'\bclass 5\b': 'primary', r'\b5th\b': 'primary',
}

_STUDY_STAGE_MAP = {
    r'\bdirect second year\b': 'direct_second_year',
    r'\bdsy\b': 'direct_second_year',
    r'\blateral entry\b': 'direct_second_year',
    r'\bfirst year\b': 'first_year', r'\b1st year\b': 'first_year',
    r'\bsecond year\b': 'second_year', r'\b2nd year\b': 'second_year',
    r'\bthird year\b': 'third_year', r'\b3rd year\b': 'third_year',
    r'\bfourth year\b': 'fourth_year', r'\b4th year\b': 'fourth_year',
    r'\bfinal year\b': 'fourth_year',
}

_COURSE_MAP = {
    r'\bengineering\b': 'engineering', r'\bcomputer science\b': 'engineering',
    r'\bcs\b': 'engineering', r'\bit\b': 'engineering',
    r'\bmechanical\b': 'engineering', r'\belectrical\b': 'engineering',
    r'\bcivil\b': 'engineering', r'\belectronics\b': 'engineering',
    r'\bmedical\b': 'medical', r'\bmbbs\b': 'medical', r'\bnursing\b': 'medical',
    r'\bpharmacy\b': 'medical', r'\bpharm\b': 'medical',
    r'\barts\b': 'arts',
    r'\bscience\b': 'science',
    r'\bcommerce\b': 'commerce',
    r'\blaw\b': 'law',
    r'\bmanagement\b': 'management',
    r'\barchitecture\b': 'architecture',
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
    # Skip if negative disability/minority phrase
    if re.search(r'\b(?:non[- ]?disabled|not disabled|no disability|non[- ]?minority|not minority|no minority)\b', text_lower):
        return None
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

def extract_profile_fields_fast(message: str, conversation_history: list[dict] | None = None) -> dict:
    """
    Fast regex-based profile field extraction from a user message,
    including context-aware answering when the bot asks a specific question.

    Returns a dict with ONLY the fields that were explicitly found.
    """
    text_lower = message.lower().strip()
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

    # Disability (Explicit patterns)
    if re.search(r'\b(?:no disability|non[- ]?disabled|not disabled|no pwd|no divyang|not handicapped|no handicap|without disability)\b', text_lower):
        fields['disability_status'] = 'non-disabled'
    elif re.search(r'\b(?:disabled|divyang|pwd|handicap|physically handicapped)\b', text_lower) and not re.search(r'\b(?:no|not|non)\b', text_lower):
        fields['disability_status'] = 'disabled'

    # Minority (Explicit patterns)
    if re.search(r'\b(?:non[- ]?minority|not minority|no minority|majority|hindu)\b', text_lower):
        fields['minority_status'] = 'non-minority'
    elif re.search(r'\b(?:minority|muslim|christian|sikh|jain|buddhist|parsi)\b', text_lower) and not re.search(r'\b(?:no|not|non)\b', text_lower):
        fields['minority_status'] = 'minority'

    # Marital status (Explicit patterns)
    if re.search(r'\b(?:single|unmarried|bachelor|spinster|never married)\b', text_lower):
        fields['marital_status'] = 'single'
    elif re.search(r'\b(?:married)\b', text_lower) and not re.search(r'\b(?:unmarried|never married)\b', text_lower):
        fields['marital_status'] = 'married'
    elif re.search(r'\b(?:widow|widower)\b', text_lower):
        fields['marital_status'] = 'widow'

    # Institution type (Explicit patterns)
    if re.search(r'\b(?:private unaided|unaided|private college|private university|self financed)\b', text_lower):
        fields['institution_type'] = 'private_unaided'
    elif re.search(r'\b(?:government aided|govt aided|aided college|aided institution|government college|govt college)\b', text_lower):
        fields['institution_type'] = 'government_aided'
    elif re.search(r'\b(?:autonomous)\b', text_lower):
        fields['institution_type'] = 'autonomous'

    # Domicile / residential
    if re.search(r'\bdomi[cs]ile\b|\bresident\b|\bliving in\b|\bfrom\b', text_lower):
        if state:
            fields['residential_status'] = f'resident_{state.lower().replace(" ", "_")}'
        else:
            fields['residential_status'] = 'permanent_resident'

    # Contextual Answer Resolution (when answering the bot's most recent question)
    if conversation_history:
        # Find the last model message
        last_bot_msg = ""
        for m in reversed(conversation_history):
            if m.get('role') in ('model', 'assistant'):
                last_bot_msg = (m.get('content') or '').lower()
                break

        # Extract only the actual question asked at the bottom of the bot's message
        lines = [l.strip() for l in last_bot_msg.strip().splitlines() if l.strip()]
        last_question = lines[-1] if lines else ""

        is_no = bool(re.match(r'^(?:no|nope|nah|none|nil|na|no i don\'?t|no i do not|not at all|no never|nothing|no disability|not really)$', text_lower))
        is_yes = bool(re.match(r'^(?:yes|yeah|yep|yup|i have|i do|true|sure|yes i have|yes i do)$', text_lower))

        if 'disability' in last_question or 'divyang' in last_question or 'pwd' in last_question:
            if is_no or 'no' in text_lower.split():
                fields['disability_status'] = 'non-disabled'
            elif is_yes or 'yes' in text_lower.split():
                fields['disability_status'] = 'disabled'

        elif 'minority' in last_question:
            if is_no or 'no' in text_lower.split() or 'hindu' in text_lower:
                fields['minority_status'] = 'non-minority'
            elif is_yes or 'yes' in text_lower.split():
                fields['minority_status'] = 'minority'

        elif 'domicile' in last_question or 'permanent resident' in last_question:
            if is_yes or 'yes' in text_lower.split():
                fields['residential_status'] = 'permanent_resident'
            elif is_no or 'no' in text_lower.split():
                fields['residential_status'] = 'non-resident'

        elif 'marital' in last_question:
            if is_no or 'single' in text_lower or 'unmarried' in text_lower:
                fields['marital_status'] = 'single'
            elif is_yes or 'married' in text_lower:
                fields['marital_status'] = 'married'

        elif 'institution' in last_question or 'college' in last_question:
            if 'private' in text_lower or 'unaided' in text_lower:
                fields['institution_type'] = 'private_unaided'
            elif 'govt' in text_lower or 'government' in text_lower or 'aided' in text_lower:
                fields['institution_type'] = 'government_aided'
            elif 'autonomous' in text_lower:
                fields['institution_type'] = 'autonomous'

        elif 'farmer' in last_question:
            if is_no or 'not a farmer' in text_lower:
                fields['farmer_status'] = 'non-farmer'
            elif is_yes or 'farmer' in text_lower:
                fields['farmer_status'] = 'registered_farmer'

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

Extract ONLY fields that are explicitly stated or directly answered in context.
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
- residential_status: e.g. "permanent_resident" or "resident_maharashtra"
- employment_status: "employed", "unemployed", "self-employed", "student"
- marital_status: "single", "married", "widow", "widower"
- institution_type: "government_aided", "private_unaided", "autonomous"
- occupation: e.g. "farmer", "teacher", "entrepreneur"
- district: district name string

Special normalizations:
- If bot asked "Do you have any disability" and user says "no" / "no i dont" -> disability_status: "non-disabled"
- If bot asked "Do you belong to minority" and user says "no" -> minority_status: "non-minority"
- "DSY" / "Direct Second Year" -> study_stage: "direct_second_year" AND education_level: "undergraduate"
- "girl student" -> gender: "Female"
- "EWS" -> category: "EWS"

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

    return extract_profile_fields_fast(message, conversation_history)

