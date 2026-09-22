"""
Scheme Eligibility Extractor for SchemeSathi.

Uses the LLM to parse a scheme's free-text eligibility description into
a list of structured EligibilityRule objects that can be evaluated
deterministically by the eligibility engine.

Results are cached in memory (slug -> rules) so each scheme is only
parsed once per server process.
"""

from __future__ import annotations

import re
import json
import asyncio
from functools import lru_cache

from backend.eligibility_engine import EligibilityRule


# In-memory cache: slug -> list of EligibilityRule
_RULES_CACHE: dict[str, list[EligibilityRule]] = {}
_CACHE_LOCK = asyncio.Lock()

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are a precise eligibility rule extractor for Indian government schemes.

Given the eligibility description of a scheme, extract ALL eligibility conditions as structured JSON rules.

OUTPUT FORMAT — a JSON array of rule objects, each with:
[
  {{
    "field": "<field_name>",
    "operator": "<operator>",
    "value": <value>,
    "mandatory": true/false
  }}
]

ALLOWED FIELDS:
- "state"                   -> which state(s) the user must be from
- "category"                -> caste category (SC/ST/OBC/EWS/General/Minority/Disabled/VJNT/SBC/NT)
- "gender"                  -> Male/Female/Transgender
- "education_level"         -> pre-primary/primary/upper_primary/secondary/higher_secondary/diploma/undergraduate/postgraduate/doctoral
- "study_stage"             -> first_year/second_year/direct_second_year/etc.
- "course"                  -> engineering/medical/arts/science/commerce/etc.
- "annual_family_income"    -> income limit in rupees (number only, no symbols)
- "age"                     -> age in years
- "disability_status"       -> disabled/non-disabled
- "minority_status"         -> minority/non-minority
- "residential_status"      -> resident/<state>/domicile
- "special_condition"       -> construction_worker_dependent/landless_labourer/orphan/etc.
- "institution_type"        -> sainik_school/government/private/aided/unaided
- "marital_status"          -> single/married/widow/widower

ALLOWED OPERATORS:
- "IN"        -> field value must be in the list (use for category, state, gender, education_level, special_condition)
- "NOT_IN"    -> field value must NOT be in the list
- "<="        -> less than or equal (income limits, age limits)
- ">="        -> greater than or equal (minimum education, minimum age)
- "BETWEEN"   -> value is [min, max] inclusive
- "CONTAINS"  -> string contains (for course matching)
- "EXISTS"    -> field must be non-null/non-empty

MANDATORY vs OPTIONAL:
- mandatory=true  -> user MUST satisfy this to be eligible
- mandatory=false -> preferred but not disqualifying (e.g. "preference given to girls")

IMPORTANT RULES:
- If scheme is for "VJNT", "Vimukta Jatis", "Nomadic Tribes", "SBC", "Special Backward Class" -> {{"field": "category", "operator": "IN", "value": ["VJNT", "SBC"]}}
- If scheme is for "Class 1 to 10", "5th to 7th", "8th to 10th" -> {{"field": "education_level", "operator": "IN", "value": ["primary", "upper_primary", "secondary"]}}
- If scheme is for "10th to 12th", "11th and 12th" -> {{"field": "education_level", "operator": "IN", "value": ["higher_secondary"]}}
- If scheme is for "Undergraduate / Degree / Engineering" -> {{"field": "education_level", "operator": "IN", "value": ["undergraduate"]}}
- If scheme is for "Children / Family of Construction Workers / Building Workers" -> {{"field": "special_condition", "operator": "IN", "value": ["construction_worker_dependent"]}}
- If scheme is for "Landless Labourers" -> {{"field": "special_condition", "operator": "IN", "value": ["landless_labourer"]}}
- If scheme is for "Sainik School" students -> {{"field": "institution_type", "operator": "IN", "value": ["sainik_school"]}}
- Income "not exceeding 2 lakh" -> {{"field": "annual_family_income", "operator": "<=", "value": 200000}}

Eligibility description:
\"\"\"
{eligibility_text}
\"\"\"

Scheme name: {scheme_name}
Scheme categories (from DB): {categories}
Scheme states (from DB): {states}

Return ONLY the JSON array, no explanation, no markdown fences:"""


# ---------------------------------------------------------------------------
# Hard-coded fast-path rule builders (for common patterns)
# ---------------------------------------------------------------------------

def _fast_extract_rules(eligibility_text: str, states: list[str], categories: list[str], scheme_name: str = '') -> list[EligibilityRule] | None:
    """
    Extract rules using comprehensive regex patterns for common scheme types.
    """
    text_lower = (scheme_name + " " + eligibility_text).lower()
    rules: list[EligibilityRule] = []

    # --- State rules from DB metadata ---
    if states:
        rules.append(EligibilityRule(
            field='state',
            operator='IN',
            value=[s.strip() for s in states],
            mandatory=True,
            raw_text=f"State: {states}",
        ))

    # --- Category rules ---
    cat_map = {
        'scheduled caste': 'SC', r'\bsc\b': 'SC', 'dalit': 'SC',
        'scheduled tribe': 'ST', r'\bst\b': 'ST', 'tribal': 'ST', 'adivasi': 'ST',
        'other backward class': 'OBC', r'\bobc\b': 'OBC',
        'vimukta jati': 'VJNT', 'nomadic tribe': 'VJNT', 'nomadic tribes': 'VJNT',
        r'\bvjnt\b': 'VJNT', r'\bvj\b': 'VJNT', r'\bnt\b': 'VJNT',
        'special backward class': 'SBC', 'special backward classes': 'SBC', r'\bsbc\b': 'SBC',
        'economically weaker': 'EWS', r'\bews\b': 'EWS', r'\bebc\b': 'EWS',
        'minority': 'Minority', 'muslim': 'Minority', 'christian': 'Minority',
        'disabled': 'Disabled', 'divyang': 'Disabled', 'pwd': 'Disabled',
    }
    found_cats = set()
    for pattern, cat in cat_map.items():
        if re.search(pattern, text_lower):
            found_cats.add(cat)
    if found_cats:
        rules.append(EligibilityRule(
            field='category',
            operator='IN',
            value=list(found_cats),
            mandatory=True,
            raw_text=f"Category from text: {found_cats}",
        ))

    # --- Education Level ---
    if re.search(r'\b1st to 7th\b|\b1st to 10th\b|\b5th to 7th\b|\b5th to 8th\b|\bprimary\b|\bupper primary\b', text_lower):
        rules.append(EligibilityRule(
            field='education_level',
            operator='IN',
            value=['primary', 'upper_primary'],
            mandatory=True,
            raw_text="Primary/Upper Primary school only",
        ))
    elif re.search(r'\b8th to 10th\b|\b8th to 10\b|\bsecondary school\b|\bmatric\b|\bssc\b|\b9th\b|\b10th std\b|\b10th standard\b', text_lower):
        rules.append(EligibilityRule(
            field='education_level',
            operator='IN',
            value=['secondary'],
            mandatory=True,
            raw_text="Secondary school (8th-10th) only",
        ))
    elif re.search(r'\b11th and 12th\b|\b10th to 12th\b|\b11th\b|\b12th standard\b|\bhsc\b|\bhigher secondary\b|\bjunior college\b', text_lower):
        rules.append(EligibilityRule(
            field='education_level',
            operator='IN',
            value=['higher_secondary'],
            mandatory=True,
            raw_text="Higher secondary (11th-12th) only",
        ))
    elif re.search(r'\bdiploma course\b|\bdiploma\b|\bpolytechnic\b|\biti\b', text_lower) and not re.search(r'\bdegree\b|\bgraduation\b|\bundergraduate\b', text_lower):
        rules.append(EligibilityRule(
            field='education_level',
            operator='IN',
            value=['diploma'],
            mandatory=True,
            raw_text="Diploma only",
        ))

    # --- Special Conditions / Parent Occupation ---
    if re.search(r'construction worker|building and other construction|mbocww|bocw', text_lower):
        rules.append(EligibilityRule(
            field='special_condition',
            operator='IN',
            value=['construction_worker_dependent', 'construction_worker'],
            mandatory=True,
            raw_text="Parent or applicant must be registered construction worker",
        ))

    if re.search(r'landless labourer|landless agricultural', text_lower):
        rules.append(EligibilityRule(
            field='special_condition',
            operator='IN',
            value=['landless_labourer'],
            mandatory=True,
            raw_text="Must be landless labourer",
        ))

    if re.search(r'sainik school', text_lower):
        rules.append(EligibilityRule(
            field='institution_type',
            operator='IN',
            value=['sainik_school'],
            mandatory=True,
            raw_text="Must study in Sainik School",
        ))

    # --- Gender ---
    if re.search(r'\bgirl\b|\bfemale\b|\bwomen\b|\bwoman\b|\bmahila\b|\bkanyadan\b', text_lower):
        mandatory = not re.search(r'prefer|priority|encourage', text_lower)
        rules.append(EligibilityRule(
            field='gender',
            operator='IN',
            value=['Female'],
            mandatory=mandatory,
            raw_text="Gender: female",
        ))

    # --- Income ---
    income_lakh = re.search(
        r'(?:income|earning)[^₹\d]*[₹rs\.]*\s*([\d.]+)\s*(?:lakhs?|lacs?|lpa|l\b)',
        text_lower
    )
    if income_lakh:
        try:
            val = int(float(income_lakh.group(1)) * 100_000)
            rules.append(EligibilityRule(
                field='annual_family_income',
                operator='<=',
                value=val,
                mandatory=True,
                raw_text=f"Income <= {val}",
            ))
        except ValueError:
            pass
    else:
        income_match = re.search(
            r'(?:income|earning)[^₹\d]*[₹rs\.]*\s*([\d,]+)\s*(?:/-|per annum|pa|annually)?',
            text_lower
        )
        if income_match:
            try:
                income_val = int(income_match.group(1).replace(',', ''))
                if income_val <= 100:
                    income_val *= 100_000
                if income_val >= 10000:
                    rules.append(EligibilityRule(
                        field='annual_family_income',
                        operator='<=',
                        value=income_val,
                        mandatory=True,
                        raw_text=f"Income <= {income_val}",
                    ))
            except ValueError:
                pass

    # --- Age ---
    age_match = re.search(r'age[d\s]*(?:between|from)?\s*(\d{1,2})\s*(?:to|and|-)\s*(\d{1,2})', text_lower)
    if age_match:
        rules.append(EligibilityRule(
            field='age',
            operator='BETWEEN',
            value=[int(age_match.group(1)), int(age_match.group(2))],
            mandatory=True,
            raw_text=f"Age {age_match.group(1)}-{age_match.group(2)}",
        ))

    return rules if rules else None


# ---------------------------------------------------------------------------
# Main async extractor
# ---------------------------------------------------------------------------

async def extract_scheme_rules(
    slug: str,
    scheme_name: str,
    eligibility_text: str,
    states: list[str],
    categories: list[str],
    model_id: str,
    api_key: str | None = None,
    model_config: dict | None = None,
) -> list[EligibilityRule]:
    """
    Extract structured eligibility rules from free-text.

    Uses an in-memory cache — each slug is only parsed once per server process.
    Falls back to fast regex extraction if LLM call fails.

    Args:
        slug: Unique scheme identifier (used as cache key).
        scheme_name: Human-readable scheme name.
        eligibility_text: Raw markdown eligibility description.
        states: State list from DB metadata (used as a hint).
        categories: Category list from DB metadata (used as a hint).
        model_id, api_key, model_config: LLM configuration.

    Returns:
        List of EligibilityRule objects ready for the eligibility engine.
    """
    # Fast cache check (no lock needed for reads)
    if slug in _RULES_CACHE:
        return _RULES_CACHE[slug]

    async with _CACHE_LOCK:
        # Double-check under lock
        if slug in _RULES_CACHE:
            return _RULES_CACHE[slug]

        # Try fast regex first
        fast_rules = _fast_extract_rules(eligibility_text, states, categories, scheme_name=scheme_name)

        # Use fast_rules if we found substantial rules (at least state + category/education/special/income/gender)
        # or if text is trivial (no mention of caste, worker, class, school, income, etc.)
        complex_signals = ['vjnt', 'sbc', 'caste', 'worker', 'labourer', 'class', 'std', 'standard', 'sainik', 'degree', 'diploma']
        has_complex_signals = any(w in eligibility_text.lower() for w in complex_signals)

        if fast_rules and (len(fast_rules) >= 2 or not has_complex_signals):
            _RULES_CACHE[slug] = fast_rules
            print(f"[EligibilityExtractor] Fast-extracted {len(fast_rules)} rules for '{slug}'")
            return fast_rules

        # Full LLM extraction for complex eligibility text
        try:
            from backend.model_registry import get_model
            from langchain_core.messages import HumanMessage

            prompt = EXTRACTION_PROMPT.format(
                eligibility_text=eligibility_text[:3000],  # cap token usage
                scheme_name=scheme_name,
                categories=', '.join(categories) if categories else 'Not specified',
                states=', '.join(states) if states else 'All India (Central scheme)',
            )

            model = get_model(model_id, api_key=api_key, model_config=model_config)
            response = await model.ainvoke([HumanMessage(content=prompt)])

            # Extract text content
            content = response.content
            if isinstance(content, list):
                content = ''.join(
                    item.get('text', '') if isinstance(item, dict) else str(item)
                    for item in content
                    if not (isinstance(item, dict) and item.get('type') in ('thinking', 'reasoning'))
                )

            # Parse JSON - strip markdown code fences if present
            raw_text = str(content).strip()
            fence_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', raw_text, re.DOTALL)
            if fence_match:
                json_str = fence_match.group(1)
            else:
                # Find the outermost array brackets
                first_sq = raw_text.find('[')
                last_sq = raw_text.rfind(']')
                if first_sq != -1 and last_sq != -1 and last_sq > first_sq:
                    json_str = raw_text[first_sq:last_sq + 1]
                else:
                    raise ValueError("No JSON array found in LLM response")

            raw_rules = json.loads(json_str)
            rules = []
            for r in raw_rules:
                if not isinstance(r, dict) or 'field' not in r:
                    continue
                try:
                    rules.append(EligibilityRule(
                        field=r['field'],
                        operator=r.get('operator', 'IN'),
                        value=r.get('value'),
                        mandatory=bool(r.get('mandatory', True)),
                        raw_text=str(r),
                    ))
                except Exception:
                    continue

            if rules:
                _RULES_CACHE[slug] = rules
                print(f"[EligibilityExtractor] LLM-extracted {len(rules)} rules for '{slug}'")
                return rules

        except Exception as e:
            print(f"[EligibilityExtractor] LLM extraction failed for '{slug}': {e}")

        # Final fallback: fast regex rules (may be empty list)
        fallback = fast_rules or []
        _RULES_CACHE[slug] = fallback
        print(f"[EligibilityExtractor] Fallback: {len(fallback)} rules for '{slug}'")
        return fallback


def clear_cache():
    """Clear the eligibility rules cache (useful for testing)."""
    _RULES_CACHE.clear()


def get_cache_size() -> int:
    return len(_RULES_CACHE)
