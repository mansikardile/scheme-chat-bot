"""
Deterministic eligibility engine for SchemeSathi.

Given a structured user profile and a list of structured eligibility rules
(extracted from a scheme's eligibility text), evaluates whether the user is:

  ELIGIBLE                — all mandatory criteria are satisfied
  INELIGIBLE              — at least one mandatory criterion is definitively failed
  INSUFFICIENT_INFORMATION — at least one mandatory criterion cannot be evaluated
                             because the required profile field is missing

No LLM is involved here. This module is pure, deterministic Python logic.
"""

from __future__ import annotations

import re
from enum import Enum
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Canonical value maps (normalise user-supplied strings to canonical forms)
# ---------------------------------------------------------------------------

CATEGORY_ALIASES: dict[str, str] = {
    # General / Open
    'general': 'General', 'unreserved': 'General', 'open': 'General',
    # SC
    'sc': 'SC', 'scheduled caste': 'SC', 'dalit': 'SC',
    # ST
    'st': 'ST', 'scheduled tribe': 'ST', 'tribal': 'ST', 'adivasi': 'ST',
    # OBC
    'obc': 'OBC', 'other backward class': 'OBC', 'other backward caste': 'OBC',
    'vjnt': 'VJNT', 'sbc': 'SBC', 'nt': 'NT', 'nomadic': 'NT',
    # EWS
    'ews': 'EWS', 'economically weaker section': 'EWS', 'economically weak': 'EWS',
    # Minority
    'minority': 'Minority', 'muslim': 'Minority', 'christian': 'Minority',
    'sikh': 'Minority', 'jain': 'Minority', 'buddhist': 'Minority', 'parsi': 'Minority',
    # Disabled
    'disabled': 'Disabled', 'divyang': 'Disabled', 'pwd': 'Disabled',
    'handicap': 'Disabled', 'differently abled': 'Disabled',
}

GENDER_ALIASES: dict[str, str] = {
    'female': 'Female', 'girl': 'Female', 'woman': 'Female', 'women': 'Female',
    'mahila': 'Female', 'lady': 'Female', 'f': 'Female',
    'male': 'Male', 'boy': 'Male', 'man': 'Male', 'men': 'Male', 'm': 'Male',
    'transgender': 'Transgender', 'trans': 'Transgender', 'other': 'Other',
}

EDUCATION_LEVEL_ORDER: list[str] = [
    'pre-primary', 'primary', 'upper_primary', 'secondary', 'higher_secondary',
    'diploma', 'undergraduate', 'postgraduate', 'doctoral',
]

EDUCATION_ALIASES: dict[str, str] = {
    # Pre-primary
    'pre-primary': 'pre-primary', 'nursery': 'pre-primary', 'kg': 'pre-primary',
    # Primary (1-5)
    'primary': 'primary', 'class 1': 'primary', 'class 2': 'primary',
    'class 3': 'primary', 'class 4': 'primary', 'class 5': 'primary',
    '1st standard': 'primary', '2nd standard': 'primary', '3rd standard': 'primary',
    # Upper primary (6-8)
    'upper primary': 'upper_primary',
    'class 6': 'upper_primary', 'class 7': 'upper_primary', 'class 8': 'upper_primary',
    '6th standard': 'upper_primary', '7th standard': 'upper_primary', '8th standard': 'upper_primary',
    # Secondary (9-10 / matric)
    'secondary': 'secondary', 'ssc': 'secondary', 'matric': 'secondary',
    '9th': 'secondary', '10th': 'secondary', 'class 9': 'secondary', 'class 10': 'secondary',
    'post matric': 'secondary',
    # Higher secondary (11-12)
    'higher_secondary': 'higher_secondary', 'hsc': 'higher_secondary',
    'intermediate': 'higher_secondary', '11th': 'higher_secondary', '12th': 'higher_secondary',
    'class 11': 'higher_secondary', 'class 12': 'higher_secondary',
    'higher secondary': 'higher_secondary',
    # Diploma
    'diploma': 'diploma', 'polytechnic': 'diploma', 'iti': 'diploma',
    # Undergraduate
    'undergraduate': 'undergraduate', 'ug': 'undergraduate',
    'bachelor': 'undergraduate', 'b.tech': 'undergraduate', 'btech': 'undergraduate',
    'b.e': 'undergraduate', 'be': 'undergraduate', 'b.sc': 'undergraduate',
    'bsc': 'undergraduate', 'b.com': 'undergraduate', 'bcom': 'undergraduate',
    'b.a': 'undergraduate', 'ba': 'undergraduate', 'mbbs': 'undergraduate',
    'degree': 'undergraduate', 'graduation': 'undergraduate',
    'direct second year': 'undergraduate', 'dsy': 'undergraduate',
    # Postgraduate
    'postgraduate': 'postgraduate', 'pg': 'postgraduate',
    'master': 'postgraduate', 'm.tech': 'postgraduate', 'mtech': 'postgraduate',
    'm.sc': 'postgraduate', 'msc': 'postgraduate', 'm.a': 'postgraduate',
    'ma': 'postgraduate', 'mba': 'postgraduate',
    # Doctoral
    'doctoral': 'doctoral', 'phd': 'doctoral', 'ph.d': 'doctoral',
    'doctorate': 'doctoral', 'research': 'doctoral',
}

STUDY_STAGE_ALIASES: dict[str, str] = {
    'direct second year': 'direct_second_year', 'dsy': 'direct_second_year',
    'lateral entry': 'direct_second_year',
    'first year': 'first_year', '1st year': 'first_year',
    'second year': 'second_year', '2nd year': 'second_year',
    'third year': 'third_year', '3rd year': 'third_year',
    'fourth year': 'fourth_year', '4th year': 'fourth_year', 'final year': 'fourth_year',
    'fresher': 'first_year',
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class EligibilityStatus(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"


@dataclass
class EligibilityRule:
    """A single eligibility criterion for a scheme."""
    field: str           # e.g. "category", "state", "annual_family_income"
    operator: str        # IN, NOT_IN, <=, >=, <, >, BETWEEN, CONTAINS, EXISTS
    value: object        # list for IN/NOT_IN/BETWEEN, scalar otherwise
    mandatory: bool = True
    raw_text: str = ""   # Original criterion text for debugging


@dataclass
class EligibilityResult:
    """Complete eligibility evaluation result for one scheme against one user profile."""
    status: EligibilityStatus
    passed_criteria: list[str] = field(default_factory=list)
    failed_criteria: list[str] = field(default_factory=list)
    unknown_criteria: list[str] = field(default_factory=list)
    match_reasons: list[str] = field(default_factory=list)   # Human-readable ✓ bullets
    rejection_reason: str | None = None                       # First hard-fail reason
    missing_fields: list[str] = field(default_factory=list)  # Profile fields needed


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_category(value: str | None) -> str | None:
    if not value:
        return None
    return CATEGORY_ALIASES.get(value.lower().strip(), value.strip())


def _normalise_gender(value: str | None) -> str | None:
    if not value:
        return None
    return GENDER_ALIASES.get(value.lower().strip(), value.strip().capitalize())


def _normalise_education(value: str | None) -> str | None:
    if not value:
        return None
    return EDUCATION_ALIASES.get(value.lower().strip(), value.lower().strip())


def _normalise_study_stage(value: str | None) -> str | None:
    if not value:
        return None
    return STUDY_STAGE_ALIASES.get(value.lower().strip(), value.lower().strip())


def _education_level_index(level: str) -> int:
    """Return ordinal index of an education level. Returns -1 if unknown."""
    try:
        return EDUCATION_LEVEL_ORDER.index(level)
    except ValueError:
        return -1


def _parse_income(value: object) -> float | None:
    """Try to parse an income value to a float (rupees)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Remove currency symbols, commas, etc.
        cleaned = re.sub(r'[₹,\s]', '', value).lower()
        # Handle "2 lakh" / "2.5 lakh" / "2L"
        m = re.match(r'^([\d.]+)\s*l(?:akh)?$', cleaned)
        if m:
            return float(m.group(1)) * 100_000
        m = re.match(r'^([\d.]+)\s*cr(?:ore)?$', cleaned)
        if m:
            return float(m.group(1)) * 10_000_000
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Per-field evaluator
# ---------------------------------------------------------------------------

def _evaluate_rule(rule: EligibilityRule, profile: dict) -> str:
    """
    Evaluate a single rule against the user profile.

    Returns:
        "PASS"    — criterion is satisfied
        "FAIL"    — criterion is definitively violated
        "UNKNOWN" — required profile field is missing or ambiguous
    """
    field = rule.field
    op = rule.operator.upper()
    required_val = rule.value
    user_val = profile.get(field)

    # -----------------------------------------------------------------------
    # State / jurisdiction
    # -----------------------------------------------------------------------
    if field == 'state':
        if user_val is None:
            return "UNKNOWN"
        user_state = user_val.lower().strip()
        if op == 'IN':
            allowed = [str(v).lower().strip() for v in required_val]
            return "PASS" if user_state in allowed else "FAIL"
        if op == 'NOT_IN':
            excluded = [str(v).lower().strip() for v in required_val]
            return "PASS" if user_state not in excluded else "FAIL"
        return "UNKNOWN"

    # -----------------------------------------------------------------------
    # Category / caste
    # -----------------------------------------------------------------------
    if field == 'category':
        if user_val is None:
            return "UNKNOWN"
        user_cat = _normalise_category(str(user_val))
        if op == 'IN':
            allowed = [_normalise_category(str(v)) for v in required_val]
            # EWS is sometimes listed under "General" or as its own
            ews_aliases = {'EWS', 'General EWS', 'EWS (General)'}
            if user_cat in ews_aliases:
                # EWS passes General AND EWS requirements
                if any(a in ews_aliases or a == 'General' for a in allowed):
                    return "PASS"
            return "PASS" if user_cat in allowed else "FAIL"
        if op == 'NOT_IN':
            excluded = [_normalise_category(str(v)) for v in required_val]
            return "FAIL" if user_cat in excluded else "PASS"
        return "UNKNOWN"

    # -----------------------------------------------------------------------
    # Gender
    # -----------------------------------------------------------------------
    if field == 'gender':
        if user_val is None:
            return "UNKNOWN"
        user_gender = _normalise_gender(str(user_val))
        if op == 'IN':
            allowed = [_normalise_gender(str(v)) for v in required_val]
            return "PASS" if user_gender in allowed else "FAIL"
        if op == 'NOT_IN':
            excluded = [_normalise_gender(str(v)) for v in required_val]
            return "FAIL" if user_gender in excluded else "PASS"
        return "UNKNOWN"

    # -----------------------------------------------------------------------
    # Education level (ordinal comparison)
    # -----------------------------------------------------------------------
    if field == 'education_level':
        if user_val is None:
            return "UNKNOWN"
        user_edu = _normalise_education(str(user_val))
        user_idx = _education_level_index(user_edu)

        if op == 'IN':
            allowed = [_normalise_education(str(v)) for v in required_val]
            return "PASS" if user_edu in allowed else "FAIL"
        if op == 'NOT_IN':
            excluded = [_normalise_education(str(v)) for v in required_val]
            return "FAIL" if user_edu in excluded else "PASS"
        if op == 'GTE' or op == '>=':
            min_edu = _normalise_education(str(required_val))
            min_idx = _education_level_index(min_edu)
            if user_idx == -1 or min_idx == -1:
                return "UNKNOWN"
            return "PASS" if user_idx >= min_idx else "FAIL"
        if op == 'LTE' or op == '<=':
            max_edu = _normalise_education(str(required_val))
            max_idx = _education_level_index(max_edu)
            if user_idx == -1 or max_idx == -1:
                return "UNKNOWN"
            return "PASS" if user_idx <= max_idx else "FAIL"
        return "UNKNOWN"

    # -----------------------------------------------------------------------
    # Study stage (direct second year, first year, etc.)
    # -----------------------------------------------------------------------
    if field == 'study_stage':
        if user_val is None:
            return "UNKNOWN"
        user_stage = _normalise_study_stage(str(user_val))
        if op == 'IN':
            allowed = [_normalise_study_stage(str(v)) for v in required_val]
            if user_stage == 'direct_second_year' and ('second_year' in allowed or 'direct_second_year' in allowed):
                return "PASS"
            return "PASS" if user_stage in allowed else "FAIL"
        if op == 'NOT_IN':
            excluded = [_normalise_study_stage(str(v)) for v in required_val]
            return "FAIL" if user_stage in excluded else "PASS"
        return "UNKNOWN"

    # -----------------------------------------------------------------------
    # Annual family income (numeric)
    # -----------------------------------------------------------------------
    if field == 'annual_family_income':
        raw_user = profile.get('annual_family_income')
        if raw_user is None:
            return "UNKNOWN"
        # Handle range: [min, max] — use upper bound conservatively
        if isinstance(raw_user, (list, tuple)) and len(raw_user) == 2:
            user_income_lo = _parse_income(raw_user[0])
            user_income_hi = _parse_income(raw_user[1])
            if user_income_lo is None or user_income_hi is None:
                return "UNKNOWN"
            # For <= limit: if even the lower bound exceeds limit → FAIL
            # If upper bound is under limit → PASS. Otherwise UNKNOWN (range straddles limit)
            if op in ('<=', 'LTE'):
                limit = _parse_income(required_val)
                if limit is None:
                    return "UNKNOWN"
                if user_income_lo > limit:
                    return "FAIL"
                if user_income_hi <= limit:
                    return "PASS"
                return "UNKNOWN"  # Range straddles the limit — ask for exact income
            if op in ('>=', 'GTE'):
                limit = _parse_income(required_val)
                if limit is None:
                    return "UNKNOWN"
                return "PASS" if user_income_hi >= limit else "FAIL"
        else:
            user_income = _parse_income(raw_user)
            if user_income is None:
                return "UNKNOWN"
            if op in ('<=', 'LTE'):
                limit = _parse_income(required_val)
                return "PASS" if (limit and user_income <= limit) else "FAIL"
            if op in ('<', 'LT'):
                limit = _parse_income(required_val)
                return "PASS" if (limit and user_income < limit) else "FAIL"
            if op in ('>=', 'GTE'):
                limit = _parse_income(required_val)
                return "PASS" if (limit and user_income >= limit) else "FAIL"
            if op in ('>', 'GT'):
                limit = _parse_income(required_val)
                return "PASS" if (limit and user_income > limit) else "FAIL"
            if op == 'BETWEEN':
                lo, hi = _parse_income(required_val[0]), _parse_income(required_val[1])
                if lo is None or hi is None:
                    return "UNKNOWN"
                return "PASS" if lo <= user_income <= hi else "FAIL"
        return "UNKNOWN"

    # -----------------------------------------------------------------------
    # Age (numeric, years)
    # -----------------------------------------------------------------------
    if field == 'age':
        user_age = profile.get('age')
        if user_age is None:
            return "UNKNOWN"
        try:
            ua = float(user_age)
        except (ValueError, TypeError):
            return "UNKNOWN"
        if op in ('<=', 'LTE'):
            return "PASS" if ua <= float(required_val) else "FAIL"
        if op in ('<', 'LT'):
            return "PASS" if ua < float(required_val) else "FAIL"
        if op in ('>=', 'GTE'):
            return "PASS" if ua >= float(required_val) else "FAIL"
        if op in ('>', 'GT'):
            return "PASS" if ua > float(required_val) else "FAIL"
        if op == 'BETWEEN':
            lo, hi = float(required_val[0]), float(required_val[1])
            return "PASS" if lo <= ua <= hi else "FAIL"
        return "UNKNOWN"

    # -----------------------------------------------------------------------
    # Residential status / domicile
    # -----------------------------------------------------------------------
    if field == 'residential_status':
        user_res = profile.get('residential_status') or profile.get('state')
        if not user_res:
            return "UNKNOWN"
        user_res_str = str(user_res).lower()
        user_state_str = str(profile.get('state', '')).lower()
        if op == 'IN':
            allowed = [str(v).lower() for v in required_val]
            return "PASS" if any(a in user_res_str or user_res_str in a or a in user_state_str or user_state_str in a for a in allowed) else "FAIL"
        if op == 'EXISTS':
            return "PASS"
        return "UNKNOWN"

    # -----------------------------------------------------------------------
    # Boolean/existence flags: disability, minority, etc.
    # -----------------------------------------------------------------------
    if field in ('disability_status', 'minority_status',
                 'marital_status', 'employment_status', 'farmer_status'):
        user_flag = profile.get(field)
        if user_flag is None:
            # These are often optional prerequisites, not always mandatory
            return "UNKNOWN"
        if op == 'IN':
            allowed = [str(v).lower() for v in required_val]
            return "PASS" if str(user_flag).lower() in allowed else "FAIL"
        if op == 'EXISTS':
            return "PASS" if user_flag else "FAIL"
        return "UNKNOWN"

    # -----------------------------------------------------------------------
    # Course / stream / institution_type
    # -----------------------------------------------------------------------
    if field in ('course', 'stream', 'institution_type', 'college_type'):
        user_course = profile.get(field)
        if user_course is None:
            return "UNKNOWN"
        user_str = str(user_course).lower()
        if op == 'IN':
            allowed = [str(v).lower() for v in required_val]
            return "PASS" if any(a in user_str or user_str in a for a in allowed) else "FAIL"
        if op == 'CONTAINS':
            keyword = str(required_val).lower()
            return "PASS" if keyword in user_str else "FAIL"
        return "UNKNOWN"

    # -----------------------------------------------------------------------
    # Special conditions / Beneficiary requirements (construction worker, landless labourer, etc.)
    # -----------------------------------------------------------------------
    if field in ('special_condition', 'specific_special_conditions', 'occupation'):
        user_conds = profile.get('specific_special_conditions') or []
        user_occ = profile.get('occupation') or ''
        user_items = [str(c).lower() for c in user_conds]
        if user_occ:
            user_items.append(str(user_occ).lower())
        if op == 'IN':
            allowed = [str(v).lower() for v in required_val]
            return "PASS" if any(a in user_items or any(a in item for item in user_items) for a in allowed) else "FAIL"
        if op == 'EXISTS':
            return "PASS" if user_items else "FAIL"
        return "UNKNOWN"

    # Unknown field — treat as unknown so we don't block unfairly
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Friendly label generators
# ---------------------------------------------------------------------------

_FIELD_LABELS = {
    'state': 'State',
    'category': 'Category',
    'gender': 'Gender',
    'education_level': 'Education level',
    'study_stage': 'Study stage',
    'annual_family_income': 'Annual family income',
    'age': 'Age',
    'disability_status': 'Disability status',
    'minority_status': 'Minority status',
    'residential_status': 'Residential status',
    'course': 'Course/stream',
    'institution_type': 'Institution type',
}


def _match_reason(rule: EligibilityRule, profile: dict) -> str:
    """Generate a human-readable ✓ match reason for a passed criterion."""
    f = rule.field
    user_val = profile.get(f)
    label = _FIELD_LABELS.get(f, f.replace('_', ' ').title())

    if f == 'state':
        return f"✓ {user_val} resident"
    if f == 'gender':
        return f"✓ {_normalise_gender(str(user_val))} student/applicant"
    if f == 'category':
        return f"✓ {_normalise_category(str(user_val))} category"
    if f == 'education_level':
        return f"✓ {str(user_val).title()} education"
    if f == 'study_stage':
        stage = str(user_val).replace('_', ' ').title()
        return f"✓ {stage}"
    if f == 'annual_family_income':
        if isinstance(rule.value, (int, float)):
            limit_lakh = rule.value / 100_000
            return f"✓ Family income within ₹{limit_lakh:.1f}L limit"
        return f"✓ Family income within required limit"
    if f == 'age':
        return f"✓ Age {user_val} within required range"
    return f"✓ {label}: {user_val}"


def _fail_reason(rule: EligibilityRule, profile: dict) -> str:
    """Generate a concise rejection reason string."""
    f = rule.field
    user_val = profile.get(f)
    label = _FIELD_LABELS.get(f, f.replace('_', ' ').title())

    if f == 'category':
        req = rule.value if not isinstance(rule.value, list) else '/'.join(str(v) for v in rule.value)
        return f"Category mismatch: user={_normalise_category(str(user_val))}, required={req}"
    if f == 'state':
        req = rule.value if not isinstance(rule.value, list) else '/'.join(str(v) for v in rule.value)
        return f"State mismatch: user={user_val}, required={req}"
    if f == 'gender':
        req = rule.value if not isinstance(rule.value, list) else '/'.join(str(v) for v in rule.value)
        return f"Gender mismatch: user={_normalise_gender(str(user_val))}, required={req}"
    if f == 'education_level':
        req = rule.value if not isinstance(rule.value, list) else '/'.join(str(v) for v in rule.value)
        return f"Education level mismatch: user={user_val}, required={req}"
    if f == 'annual_family_income':
        return f"Income exceeds limit: user={user_val}, limit={rule.value}"
    if f == 'age':
        return f"Age out of range: user={user_val}, requirement={rule.operator} {rule.value}"
    return f"{label} criterion not met (user={user_val}, required={rule.value})"


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def evaluate_scheme(
    user_profile: dict,
    rules: list[EligibilityRule],
) -> EligibilityResult:
    """
    Evaluate whether a user profile satisfies a scheme's eligibility rules.

    Args:
        user_profile: The accumulated user profile dict from ChatSession.
        rules: List of EligibilityRule objects extracted from the scheme's eligibility text.

    Returns:
        EligibilityResult with status ELIGIBLE, INELIGIBLE, or INSUFFICIENT_INFORMATION.

    Algorithm:
        1. Evaluate every mandatory rule.
        2. If ANY mandatory rule → FAIL: overall result is INELIGIBLE (stop).
        3. If ANY mandatory rule → UNKNOWN: overall result is INSUFFICIENT_INFORMATION.
        4. If ALL mandatory rules → PASS: overall result is ELIGIBLE.
    """
    result = EligibilityResult(status=EligibilityStatus.ELIGIBLE)
    has_unknown = False

    mandatory_rules = [r for r in rules if r.mandatory]
    optional_rules = [r for r in rules if not r.mandatory]

    for rule in mandatory_rules:
        outcome = _evaluate_rule(rule, user_profile)
        label = _FIELD_LABELS.get(rule.field, rule.field)

        if outcome == "PASS":
            result.passed_criteria.append(rule.field)
            result.match_reasons.append(_match_reason(rule, user_profile))

        elif outcome == "FAIL":
            reason = _fail_reason(rule, user_profile)
            result.failed_criteria.append(rule.field)
            result.rejection_reason = reason
            result.status = EligibilityStatus.INELIGIBLE
            # Log the failed criterion but keep checking to collect all failures
            print(f"[EligibilityEngine] FAIL — {reason}")
            # Return immediately on first hard fail (fast-path)
            return result

        else:  # UNKNOWN
            has_unknown = True
            result.unknown_criteria.append(rule.field)
            result.missing_fields.append(rule.field)

    # Evaluate optional rules (these never cause INELIGIBLE but contribute to match reasons)
    for rule in optional_rules:
        outcome = _evaluate_rule(rule, user_profile)
        if outcome == "PASS":
            result.match_reasons.append(_match_reason(rule, user_profile))

    if has_unknown:
        result.status = EligibilityStatus.INSUFFICIENT_INFORMATION

    return result


def evaluate_scheme_batch(
    user_profile: dict,
    scheme_rules: list[tuple[str, str, list[EligibilityRule]]],
) -> list[tuple[str, str, EligibilityResult]]:
    """
    Evaluate multiple schemes at once.

    Args:
        user_profile: The accumulated user profile dict.
        scheme_rules: List of (slug, scheme_name, rules) tuples.

    Returns:
        List of (slug, scheme_name, EligibilityResult) tuples — all statuses included.
        Caller filters to ELIGIBLE only.
    """
    results = []
    for slug, name, rules in scheme_rules:
        er = evaluate_scheme(user_profile, rules)
        print(f"[EligibilityEngine] {slug!r} -> {er.status.value}"
              + (f" | FAIL: {er.rejection_reason}" if er.rejection_reason else ""))
        results.append((slug, name, er))
    return results
