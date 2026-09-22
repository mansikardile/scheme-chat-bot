"""
Private & CSR Scheme Search for SchemeSathi.

Discovers verified private, trust, NGO, and corporate CSR scholarships and grants (e.g. Lila
Poonawalla Foundation, Tata Trusts, Reliance Foundation, Kotak Kanya, HDFC
Parivartan, Santoor, Sitaram Jindal, SEWA, etc.) via dynamic AI knowledge matching
and a curated foundation repository.

All discovered schemes are formatted into standard scheme summary objects and
subsequently evaluated by the deterministic EligibilityEngine.
"""

from __future__ import annotations

import re
import json
import asyncio
from typing import Optional
from backend.eligibility_engine import EligibilityRule


# In-memory cache for discovered private schemes: slug -> full scheme data dict
PRIVATE_SCHEMES_CACHE: dict[str, dict] = {}
_CACHE_LOCK = asyncio.Lock()

# Profile query cache: signature -> list of slugs
_PROFILE_QUERY_CACHE: dict[str, list[str]] = {}


# ---------------------------------------------------------------------------
# Curated Verified Private Foundations & CSR Scholarships Repository
# ---------------------------------------------------------------------------

CURATED_PRIVATE_SCHEMES = [
    {
        'slug': 'pvt-lila-poonawalla-undergraduate-scholarship',
        'scheme_name': 'Lila Poonawalla Foundation Undergraduate Scholarship for Girls',
        'short_title': 'LPF Scholarship',
        'foundation_name': 'Lila Poonawalla Foundation (LPF)',
        'states': ['Maharashtra'],
        'categories': ['General', 'EWS', 'OBC', 'SC', 'ST', 'VJNT', 'SBC', 'Minority'],
        'gender': ['Female'],
        'education_level': ['undergraduate'],
        'course': 'engineering',
        'max_income': 400000,
        'brief_description': 'Merit-cum-need scholarship providing up to ₹60,000/year financial aid, corporate mentorship, leadership training, and placement support for girl students pursuing BE/B.Tech degrees (including 1st year and Direct Second Year / DSY) in Maharashtra.',
        'eligibility_md': """1. Must be a female candidate with domicile in Maharashtra (Pune, Amravati, Wardha, Nagpur, etc.).
2. Enrolled in 1st year BE/B.Tech or Direct Second Year (DSY) engineering degree course in a recognized college in Maharashtra.
3. Total annual family income must not exceed ₹4,00,000 per annum.
4. Minimum 60% marks in 10th, 12th, or Diploma.""",
        'application_url': 'https://www.lilapoonawallafoundation.com',
        'rules': [
            EligibilityRule(field='state', operator='IN', value=['Maharashtra'], mandatory=True, raw_text='Maharashtra resident'),
            EligibilityRule(field='gender', operator='IN', value=['Female'], mandatory=True, raw_text='Female only'),
            EligibilityRule(field='education_level', operator='IN', value=['undergraduate'], mandatory=True, raw_text='Undergraduate course'),
            EligibilityRule(field='annual_family_income', operator='<=', value=400000, mandatory=True, raw_text='Income <= ₹4 LPA'),
        ]
    },
    {
        'slug': 'pvt-kotak-kanya-scholarship',
        'scheme_name': 'Kotak Kanya Scholarship',
        'short_title': 'Kotak Kanya',
        'foundation_name': 'Kotak Education Foundation & Kotak Mahindra Group CSR',
        'states': [],  # All India
        'categories': ['General', 'EWS', 'OBC', 'SC', 'ST', 'VJNT', 'SBC', 'Minority'],
        'gender': ['Female'],
        'education_level': ['undergraduate'],
        'course': 'engineering',
        'max_income': 600000,
        'brief_description': 'Flagship CSR initiative providing up to ₹1,50,000 per academic year to meritorious girl students pursuing professional graduation courses (Engineering, MBBS, Architecture, Design, Integrated LLB).',
        'eligibility_md': """1. Open to girl students across India.
2. Must have scored >= 75% marks in Class 12 board examinations.
3. Enrolled in 1st year or direct admission in recognized professional graduation courses (BE/B.Tech, MBBS, etc.).
4. Annual family income from all sources must be ₹6,00,000 or less.""",
        'application_url': 'https://kotakeducation.org/kotak-kanya-scholarship/',
        'rules': [
            EligibilityRule(field='gender', operator='IN', value=['Female'], mandatory=True, raw_text='Female only'),
            EligibilityRule(field='education_level', operator='IN', value=['undergraduate'], mandatory=True, raw_text='Undergraduate course'),
            EligibilityRule(field='annual_family_income', operator='<=', value=600000, mandatory=True, raw_text='Income <= ₹6 LPA'),
        ]
    },
    {
        'slug': 'pvt-reliance-foundation-undergraduate-scholarship',
        'scheme_name': 'Reliance Foundation Undergraduate Scholarship',
        'short_title': 'Reliance Foundation Scholarship',
        'foundation_name': 'Reliance Foundation',
        'states': [],  # All India
        'categories': ['General', 'EWS', 'OBC', 'SC', 'ST', 'VJNT', 'SBC', 'Minority'],
        'gender': ['Female', 'Male'],
        'education_level': ['undergraduate'],
        'course': 'general',
        'max_income': 1500000,
        'brief_description': 'Merit-cum-means scholarship grant of up to ₹2,00,000 for undergraduate students in any stream over the duration of their degree programme, along with leadership development and alumni networking.',
        'eligibility_md': """1. Resident Indian citizen.
2. Passed standard 12th with minimum 60% marks.
3. Currently enrolled in a full-time undergraduate degree program in India.
4. Household annual income up to ₹15,00,000 (preference given to income under ₹2.5L and ₹6L).""",
        'application_url': 'https://www.scholarships.reliancefoundation.org',
        'rules': [
            EligibilityRule(field='education_level', operator='IN', value=['undergraduate'], mandatory=True, raw_text='Undergraduate degree'),
            EligibilityRule(field='annual_family_income', operator='<=', value=1500000, mandatory=True, raw_text='Income <= ₹15 LPA'),
        ]
    },
    {
        'slug': 'pvt-sitaram-jindal-foundation-scholarship',
        'scheme_name': 'Sitaram Jindal Foundation Scholarship Scheme',
        'short_title': 'Jindal Scholarship',
        'foundation_name': 'Sitaram Jindal Foundation (SJF)',
        'states': [],  # All India
        'categories': ['General', 'EWS', 'OBC', 'SC', 'ST', 'VJNT', 'SBC', 'Minority'],
        'gender': ['Female', 'Male'],
        'education_level': ['undergraduate', 'diploma', 'higher_secondary', 'postgraduate'],
        'course': 'general',
        'max_income': 400000,
        'brief_description': 'Monthly stipend grant (₹1,500 to ₹3,200 per month) for students pursuing engineering, medicine, diploma, graduation, and post-graduation courses from economically disadvantaged backgrounds.',
        'eligibility_md': """1. Open to students studying in recognized institutions across India.
2. Applicable for Higher Secondary, Diploma, Engineering, Medical, and General Degree courses.
3. Family income must not exceed ₹4,00,000 per annum (relaxable up to ₹4.5L in special cases).
4. Minimum 60% to 65% aggregate in qualifying exam.""",
        'application_url': 'https://www.sitaramjindalfoundation.org/scholarships.php',
        'rules': [
            EligibilityRule(field='education_level', operator='IN', value=['higher_secondary', 'diploma', 'undergraduate', 'postgraduate'], mandatory=True, raw_text='Must be an enrolled student (11th/12th/Diploma/UG/PG)'),
            EligibilityRule(field='annual_family_income', operator='<=', value=400000, mandatory=True, raw_text='Income <= ₹4 LPA'),
        ]
    },
    {
        'slug': 'pvt-tata-trusts-higher-education-scholarship',
        'scheme_name': 'Tata Trusts Means Grant for Higher Education',
        'short_title': 'Tata Trusts Grant',
        'foundation_name': 'Tata Trusts & Philanthropic Grants',
        'states': [],  # All India
        'categories': ['General', 'EWS', 'OBC', 'SC', 'ST', 'VJNT', 'SBC', 'Minority'],
        'gender': ['Female', 'Male'],
        'education_level': ['undergraduate', 'postgraduate'],
        'course': 'general',
        'max_income': 500000,
        'brief_description': 'Need-based tuition fee assistance and study grants for students pursuing undergraduate and postgraduate degree courses in India with demonstrated financial need.',
        'eligibility_md': """1. Indian nationals enrolled in recognized colleges or universities.
2. Applicable for undergraduate and postgraduate degree courses.
3. Annual family income must not exceed ₹5,00,000.
4. Consistent satisfactory academic track record.""",
        'application_url': 'https://www.tatatrusts.org',
        'rules': [
            EligibilityRule(field='education_level', operator='IN', value=['undergraduate', 'postgraduate'], mandatory=True, raw_text='Undergraduate or Postgraduate degree'),
            EligibilityRule(field='annual_family_income', operator='<=', value=500000, mandatory=True, raw_text='Income <= ₹5 LPA'),
        ]
    },
    {
        'slug': 'pvt-tata-trusts-craft-artisan-grant',
        'scheme_name': 'Tata Trusts Craft-Based Livelihood & Artisan Grant',
        'short_title': 'Tata Trusts Artisan Grant',
        'foundation_name': 'Tata Trusts',
        'states': [],  # All India
        'categories': ['General', 'EWS', 'OBC', 'SC', 'ST', 'VJNT', 'SBC', 'Minority'],
        'gender': ['Female', 'Male'],
        'education_level': ['none', 'primary', 'secondary', 'undergraduate', 'diploma'],
        'max_income': 500000,
        'brief_description': 'Grants, skill enhancement, loom upgrades, and direct market access linkages for traditional handloom weavers and craft artisans across India.',
        'eligibility_md': """1. Open to traditional handloom weavers, craftspersons, and rural artisans in India.
2. Preference given to women artisans, SHG members, and master craftsmen.
3. Family income within ₹5,00,000 per annum.""",
        'application_url': 'https://www.tatatrusts.org/our-work/livelihood/crafts',
        'rules': [
            EligibilityRule(field='special_condition', operator='IN', value=['handloom_weaver', 'traditional_artisan', 'weaver', 'artisan'], mandatory=True, raw_text='Traditional weaver or artisan'),
            EligibilityRule(field='annual_family_income', operator='<=', value=500000, mandatory=True, raw_text='Income <= ₹5 LPA'),
        ]
    },
    {
        'slug': 'pvt-reliance-foundation-rural-transformation',
        'scheme_name': 'Reliance Foundation Rural Livelihoods & Women Artisan Support',
        'short_title': 'Reliance Foundation Artisan Support',
        'foundation_name': 'Reliance Foundation CSR',
        'states': [],  # All India
        'categories': ['General', 'EWS', 'OBC', 'SC', 'ST', 'VJNT', 'SBC', 'Minority'],
        'gender': ['Female', 'Male'],
        'education_level': ['none', 'primary', 'secondary', 'undergraduate', 'diploma'],
        'max_income': 600000,
        'brief_description': 'Empowers rural handloom weavers, women self-help groups, and agricultural producers with working capital, modern toolkits, and cooperative marketing.',
        'eligibility_md': """1. Practicing handloom weavers, rural artisans, or small farmers.
2. Both male and female applicants eligible, with priority for women collectives.
3. Annual family income up to ₹6,00,000.""",
        'application_url': 'https://www.reliancefoundation.org',
        'rules': [
            EligibilityRule(field='special_condition', operator='IN', value=['handloom_weaver', 'traditional_artisan', 'weaver', 'artisan', 'shg_member', 'farmer'], mandatory=True, raw_text='Rural artisan, handloom weaver or farmer'),
            EligibilityRule(field='annual_family_income', operator='<=', value=600000, mandatory=True, raw_text='Income <= ₹6 LPA'),
        ]
    },
    {
        'slug': 'pvt-mahindra-krishi-samriddhi-csr',
        'scheme_name': 'Mahindra Krishi Samriddhi CSR Farmer Support Initiative',
        'short_title': 'Mahindra Krishi Samriddhi',
        'foundation_name': 'Mahindra & Mahindra CSR Foundation',
        'states': [],  # All India
        'categories': ['General', 'EWS', 'OBC', 'SC', 'ST', 'VJNT', 'SBC', 'Minority'],
        'gender': ['Female', 'Male'],
        'max_income': 600000,
        'brief_description': 'Provides small and marginal farmers with subsidized modern agricultural implements, soil health enhancement kits, drip irrigation assistance, and digital farming guidance.',
        'eligibility_md': """1. Small and marginal farmers with landholdings up to 2.5 hectares (approx. 5 acres) or tenant cultivators.
2. Both male and female farmers eligible, with priority for women farmers.
3. Annual family income up to ₹6,00,000.""",
        'application_url': 'https://www.mahindra.com/sustainability/csr',
        'rules': [
            EligibilityRule(field='farmer_status', operator='IN', value=['farmer', 'registered_farmer'], mandatory=True, raw_text='Must be an active farmer'),
            EligibilityRule(field='annual_family_income', operator='<=', value=600000, mandatory=True, raw_text='Income <= ₹6 LPA'),
        ]
    },
    {
        'slug': 'pvt-itc-mission-sunehra-kal-agriculture',
        'scheme_name': 'ITC Mission Sunehra Kal Sustainable Agriculture & Farmer Empowerment',
        'short_title': 'ITC Choupal CSR',
        'foundation_name': 'ITC Limited CSR (e-Choupal Initiative)',
        'states': [],  # All India
        'categories': ['General', 'EWS', 'OBC', 'SC', 'ST', 'VJNT', 'SBC', 'Minority'],
        'gender': ['Female', 'Male'],
        'max_income': 600000,
        'brief_description': 'Assists smallholder farmers with climate-smart agriculture practices, certified high-yield seed assistance, water-saving technologies, and direct market linkage.',
        'eligibility_md': """1. Small and marginal farmers or agricultural laborers.
2. Engaged in crop cultivation, horticulture, or agro-forestry.
3. Annual family income up to ₹6,00,000.""",
        'application_url': 'https://www.itcportal.com/sustainability/mission-sunehra-kal.aspx',
        'rules': [
            EligibilityRule(field='farmer_status', operator='IN', value=['farmer', 'registered_farmer'], mandatory=True, raw_text='Must be an active farmer'),
            EligibilityRule(field='annual_family_income', operator='<=', value=600000, mandatory=True, raw_text='Income <= ₹6 LPA'),
        ]
    },
]


PRIVATE_SEARCH_PROMPT = """You are an expert scheme & grant advisor for Indian citizens, farmers, artisans, and students.

Find verified, legitimate PRIVATE grants, NGO/Trust support, and Corporate CSR initiatives in India (e.g., Mahindra Krishi Samriddhi, ITC Mission Sunehra Kal, Tata Trusts Livelihoods, Reliance Foundation Rural Transformation, Lila Poonawalla Foundation, Kotak Kanya, HDFC Parivartan, Sitaram Jindal Foundation, etc.) that match this candidate's specific profile and domain.

Candidate Profile:
- State: {state}
- Gender: {gender}
- Caste Category: {category}
- Occupation: {occupation}
- Farmer Status: {farmer_status}
- Land Holding: {land_holding}
- Education Level: {education_level}
- Course/Stream: {course}
- Study Stage: {study_stage}
- Age: {age}
- Annual Family Income: {income}
- Disability: {disability}
- Minority: {minority}

Instructions:
1. If candidate is a Farmer, find agricultural and rural livelihood CSR initiatives (NOT student scholarships).
2. If candidate is a Student, find scholarships and education grants.
3. If candidate is an Artisan/Weaver, find craft/handloom livelihood programs.
4. Identify 3 to 6 prominent, verified REAL private/trust/CSR schemes matching this exact domain.
5. Write precise, factual eligibility criteria for each scheme in markdown under `eligibility_md`.
6. Include the official organization / foundation website URL where applicants can apply.

Return ONLY a valid JSON array of objects with this exact structure:
[
  {{
    "slug": "unique-kebab-slug",
    "scheme_name": "Full Official Name of Grant, Initiative or Scholarship",
    "short_title": "Short Name / Acronym",
    "foundation_name": "Trust / Corporate / NGO Name",
    "states": ["State1", "State2"],
    "categories": ["Category1", "Category2"],
    "gender": ["Female"],
    "education_level": "none",
    "max_income": 600000,
    "brief_description": "2-3 sentences explaining the grant amount/support and target candidates.",
    "eligibility_md": "Numbered list of mandatory eligibility conditions.",
    "application_url": "https://official-portal-url.org"
  }}
]

No markdown fences, no explanatory text, return ONLY the JSON array:"""


def _get_profile_signature(profile: dict) -> str:
    """Generate a stable cache key from relevant profile attributes."""
    keys = [
        str(profile.get('state') or ''),
        str(profile.get('gender') or ''),
        str(profile.get('category') or ''),
        str(profile.get('education_level') or ''),
        str(profile.get('course') or ''),
        str(profile.get('study_stage') or ''),
        str(profile.get('occupation') or ''),
        str(profile.get('special_condition') or ''),
        str(profile.get('annual_family_income') or ''),
    ]
    return '|'.join(keys).lower()


def _register_scheme_in_cache(item: dict, rules: list[EligibilityRule] | None = None) -> dict:
    """Format and register a private scheme into global caches."""
    slug = item.get('slug') or re.sub(r'[^a-z0-9]+', '-', item.get('scheme_name', '').lower()).strip('-')
    if not slug.startswith("pvt-"):
        slug = f"pvt-{slug}"

    summary = {
        'slug': slug,
        'schemeName': item.get('scheme_name', ''),
        'schemeShortTitle': item.get('short_title', item.get('scheme_name', '')),
        'level': 'Private / Trust',
        'beneficiaryState': item.get('states', []),
        'schemeCategory': ['Private & CSR Scholarships', 'Trusts & Foundations'],
        'nodalMinistryName': item.get('foundation_name', 'Private Trust / NGO Foundation'),
        'tags': ['private', 'scholarship', 'csr', 'trust', item.get('course', '')],
        'schemeFor': 'Students & Citizens',
        'briefDescription': item.get('brief_description', ''),
        'applicationUrl': item.get('application_url', 'https://www.google.com'),
        'is_private': True,
        'source_type': 'private',
        'rules': rules or item.get('rules') or [],
    }

    detailed_raw = {
        'slug': slug,
        'en': {
            'basicDetails': {
                'schemeName': item.get('scheme_name', ''),
                'schemeShortTitle': item.get('short_title', ''),
                'level': {'value': 'Private', 'label': 'Private / Trust'},
                'state': {'value': item.get('states', [''])[0] if item.get('states') else '', 'label': ', '.join(item.get('states', [])) or 'All India'},
                'nodalMinistryName': {'value': item.get('foundation_name', ''), 'label': item.get('foundation_name', '')},
                'tags': summary['tags'],
            },
            'schemeContent': {
                'briefDescription': item.get('brief_description', ''),
                'benefits': f"Financial support, mentorship, and grant assistance via {item.get('foundation_name', 'Trust Foundation')}.",
                'applicationProcess': [
                    {
                        'mode': 'Online Application Portal',
                        'url': item.get('application_url', ''),
                        'process_md': f"Apply directly on the official {item.get('foundation_name', 'Foundation')} portal at [{item.get('application_url', 'website')}]({item.get('application_url', '')})."
                    }
                ],
                'documentsRequired': [
                    {'name': 'Identity & Address Proof (Aadhaar / Voter ID)'},
                    {'name': 'Income Certificate / Self Declaration'},
                    {'name': 'Academic Marksheets / Admission Proof (for scholarships)'},
                    {'name': 'Artisan / Trade Card or Passbook (for trade schemes)'},
                ],
                'eligibilityCriteria': {
                    'eligibilityDescription_md': item.get('eligibility_md', item.get('brief_description', ''))
                }
            }
        }
    }

    PRIVATE_SCHEMES_CACHE[slug] = summary
    PRIVATE_SCHEMES_CACHE[f"{slug}__raw"] = detailed_raw
    return summary


# Pre-register curated schemes into cache at import time
for curated in CURATED_PRIVATE_SCHEMES:
    _register_scheme_in_cache(curated, curated.get('rules'))


async def search_private_schemes_ai(
    user_profile: dict,
    model_id: str = 'gemini-flash',
    api_key: str | None = None,
    model_config: dict | None = None,
) -> list[dict]:
    """
    Dynamically discover matching private/trust/CSR scholarships using curated
    repository + AI knowledge matching.

    Returns candidate scheme dicts with pre-attached rules and portal links.
    """
    candidates: list[dict] = []
    seen_slugs = set()

    # 1. Match from Curated Verified Private Schemes (Instant, zero latency)
    user_state = (user_profile.get('state') or '').lower()
    user_gender = (user_profile.get('gender') or '').lower()
    user_occ = (user_profile.get('occupation') or '').lower()
    user_farmer = bool(user_profile.get('farmer_status') or user_occ == 'farmer')
    user_edu = user_profile.get('education_level')
    user_course = user_profile.get('course')
    is_student = (user_occ == 'student' or bool(user_edu) or bool(user_course))

    user_income = user_profile.get('annual_family_income')
    if isinstance(user_income, (list, tuple)):
        user_income = max(user_income)

    for item in CURATED_PRIVATE_SCHEMES:
        # State check
        item_states = [s.lower() for s in item.get('states', [])]
        if item_states and user_state and user_state not in item_states:
            continue
        # Gender check
        item_genders = [g.lower() for g in item.get('gender', [])]
        if item_genders and user_gender and user_gender not in item_genders:
            continue
        # Income check
        item_max_inc = item.get('max_income')
        if item_max_inc and user_income and user_income > item_max_inc:
            continue

        # Domain check: If user is explicitly a farmer and not a student, do not match student scholarships
        if user_farmer and not is_student:
            if item.get('education_level') and 'scholarship' in item.get('slug', ''):
                continue
        # If user is explicitly a student and not a farmer, do not match exclusive farmer/artisan schemes
        if is_student and not user_farmer:
            if item.get('slug') in ('pvt-mahindra-krishi-samriddhi-csr', 'pvt-itc-mission-sunehra-kal-agriculture'):
                continue

        slug = item['slug']
        summary = PRIVATE_SCHEMES_CACHE.get(slug)
        if summary and slug not in seen_slugs:
            candidates.append(summary)
            seen_slugs.add(slug)

    # If we already have strong curated candidates (>= 3), return immediately to save LLM credits & latency
    if len(candidates) >= 3:
        return candidates

    # 2. Dynamic AI Discovery for additional CSR / Trust scholarships
    sig = _get_profile_signature(user_profile)
    if sig in _PROFILE_QUERY_CACHE:
        cached_slugs = _PROFILE_QUERY_CACHE[sig]
        for s in cached_slugs:
            if s in PRIVATE_SCHEMES_CACHE and s not in seen_slugs:
                candidates.append(PRIVATE_SCHEMES_CACHE[s])
                seen_slugs.add(s)
        return candidates

    income = user_profile.get('annual_family_income')
    if isinstance(income, (list, tuple)):
        income_str = f"₹{income[0]//100_000}L–₹{income[1]//100_000}L/year"
    elif isinstance(income, (int, float)):
        income_str = f"₹{income//100_000}L/year (₹{income:,.0f})"
    else:
        income_str = str(income or 'Not specified')

    prompt = PRIVATE_SEARCH_PROMPT.format(
        state=user_profile.get('state') or 'All India',
        gender=user_profile.get('gender') or 'Any',
        category=user_profile.get('category') or 'General / All',
        occupation=user_profile.get('occupation') or ('Farmer' if user_farmer else ('Student' if is_student else 'Citizen')),
        farmer_status='Farmer' if user_farmer else 'No',
        land_holding=f"{user_profile.get('land_holding')} acres" if user_profile.get('land_holding') else 'Not specified',
        education_level=user_profile.get('education_level') or ('N/A' if user_farmer and not is_student else 'Undergraduate'),
        course=user_profile.get('course') or ('Agriculture / Farming' if user_farmer and not is_student else 'General'),
        study_stage=user_profile.get('study_stage') or 'N/A',
        age=user_profile.get('age') or 'Not specified',
        income=income_str,
        disability=user_profile.get('disability_status') or 'None',
        minority=user_profile.get('minority_status') or 'None',
    )

    try:
        from backend.model_registry import get_model
        from langchain_core.messages import HumanMessage

        model = get_model(model_id, api_key=api_key, model_config=model_config)
        response = await model.ainvoke([HumanMessage(content=prompt)])

        content = response.content
        if isinstance(content, list):
            content = ''.join(
                item.get('text', '') if isinstance(item, dict) else str(item)
                for item in content
                if not (isinstance(item, dict) and item.get('type') in ('thinking', 'reasoning'))
            )

        raw_text = str(content).strip()
        fence_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', raw_text, re.DOTALL)
        if fence_match:
            json_str = fence_match.group(1)
        else:
            first_sq = raw_text.find('[')
            last_sq = raw_text.rfind(']')
            if first_sq != -1 and last_sq != -1 and last_sq > first_sq:
                json_str = raw_text[first_sq:last_sq + 1]
            else:
                json_str = "[]"

        items = json.loads(json_str)
        async with _CACHE_LOCK:
            ai_slugs = []
            for item in items:
                if not isinstance(item, dict) or not item.get('scheme_name'):
                    continue

                # Build automatic eligibility rules from the AI's structured response
                rules = []
                if item.get('states'):
                    rules.append(EligibilityRule(field='state', operator='IN', value=item['states'], mandatory=True, raw_text=f"States: {item['states']}"))
                if item.get('gender'):
                    rules.append(EligibilityRule(field='gender', operator='IN', value=item['gender'], mandatory=True, raw_text=f"Gender: {item['gender']}"))
                if item.get('max_income'):
                    rules.append(EligibilityRule(field='annual_family_income', operator='<=', value=int(item['max_income']), mandatory=True, raw_text=f"Income <= ₹{item['max_income']}"))
                if item.get('education_level'):
                    rules.append(EligibilityRule(field='education_level', operator='IN', value=[item['education_level']], mandatory=True, raw_text=f"Education: {item['education_level']}"))

                summary = _register_scheme_in_cache(item, rules=rules)
                s_slug = summary['slug']
                ai_slugs.append(s_slug)
                if s_slug not in seen_slugs:
                    candidates.append(summary)
                    seen_slugs.add(s_slug)

            _PROFILE_QUERY_CACHE[sig] = list(seen_slugs)

        print(f"[PrivateSchemeSearch] Total {len(candidates)} private/trust schemes available for profile")
        return candidates

    except Exception as e:
        print(f"[PrivateSchemeSearch] Dynamic search error: {e}")
        return candidates


def get_private_scheme_detail(slug: str) -> Optional[dict]:
    """Retrieve full raw detailed scheme dict for a cached private scheme."""
    return PRIVATE_SCHEMES_CACHE.get(f"{slug}__raw")


def get_private_scheme_context(slug: str) -> Optional[str]:
    """Format detailed context for a cached private scheme."""
    raw = get_private_scheme_detail(slug)
    if not raw:
        return None

    en = raw.get('en', {})
    basic = en.get('basicDetails', {})
    content = en.get('schemeContent', {})
    elig = en.get('eligibilityCriteria', {})

    parts = [
        f"# {basic.get('schemeName', '')}",
        f"**Organization / Foundation**: {basic.get('nodalMinistryName', {}).get('label', '')}",
        f"**Category**: Private / Trust CSR Scholarship",
        f"\n## Overview\n{content.get('briefDescription', '')}",
        f"\n## Eligibility Criteria\n{elig.get('eligibilityDescription_md', '')}",
        f"\n## Benefits\n{content.get('benefits', '')}",
        f"\n## How to Apply\n{content.get('applicationProcess', '')}",
        f"\n## Documents Required\n{content.get('documentsRequired', '')}",
    ]
    return '\n'.join(parts)
