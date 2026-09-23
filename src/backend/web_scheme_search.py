"""
Web Scheme Search for SchemeSathi.

Searches the web (DuckDuckGo) for government welfare schemes matching user profile,
then uses LLM to extract structured scheme information and verify eligibility.
"""

from __future__ import annotations

import json
import re
import asyncio

# Cache: query_key -> list of web-found scheme dicts
_WEB_SEARCH_CACHE: dict[str, list[dict]] = {}


def _build_search_query(user_profile: dict) -> str:
    parts = []
    state = user_profile.get("state") or ""
    course = user_profile.get("course") or ""
    occupation = user_profile.get("occupation") or ""
    category = user_profile.get("category") or ""
    gender = user_profile.get("gender") or ""

    if state:
        parts.append(f"{state} government")
    parts.append("scholarship scheme 2024 2025")

    if course and course not in ("general", "student"):
        parts.append(f"{course} student")
    elif occupation and occupation == "student":
        parts.append("student")

    if category and category != "General":
        parts.append(category)

    if gender == "Female":
        parts.append("girl women")

    query = " ".join(parts)
    query += " site:myscheme.gov.in OR site:scholarships.gov.in OR site:buddy4study.com"
    return query


def _profile_cache_key(user_profile: dict) -> str:
    return "|".join([
        str(user_profile.get("state") or ""),
        str(user_profile.get("course") or ""),
        str(user_profile.get("occupation") or ""),
        str(user_profile.get("category") or ""),
        str(user_profile.get("gender") or ""),
    ])


WEB_SCHEME_EXTRACT_PROMPT = """You are extracting government scheme information from web search results.

Given these web search snippets about government welfare schemes, extract ONLY real, verifiable schemes.

USER PROFILE:
{profile}

WEB SEARCH RESULTS:
{search_results}

TASK:
1. Identify actual government/trust schemes mentioned in the results
2. For EACH scheme, extract structured information
3. Critically assess: Is this scheme ACTUALLY applicable to this specific user?

Return a JSON array. Each scheme must have ALL these fields:
[
  {{
    "scheme_name": "<exact scheme name>",
    "slug": "<snake_case_unique_id>",
    "level": "Central" or "State" or "Private / Trust",
    "state": "<state name or 'All India'>",
    "brief_description": "<2-3 sentence description of what the scheme provides>",
    "eligibility_summary": "<key eligibility criteria as a brief text>",
    "benefit": "<what user gets: amount, type>",
    "application_url": "<URL if found, else null>",
    "is_eligible": true or false or null,
    "eligible_reason": "<why eligible or not>",
    "confidence": 0.0 to 1.0
  }}
]

STRICT RULES:
- Only include schemes where is_eligible=true AND confidence >= 0.7
- Do NOT invent schemes - only include ones clearly mentioned in the search results
- Do NOT include schemes for wrong beneficiary type (elderly pension for young student etc.)
- If search results have no relevant schemes, return []

Return ONLY the JSON array, no explanation:"""


async def search_schemes_web(
    user_profile: dict,
    model_id: str,
    api_key: str | None = None,
    model_config: dict | None = None,
    max_results: int = 5,
) -> list[dict]:
    """Search the web for government schemes matching user profile."""
    cache_key = _profile_cache_key(user_profile)
    if cache_key in _WEB_SEARCH_CACHE:
        print(f"[WebSearch] Cache hit")
        return _WEB_SEARCH_CACHE[cache_key]

    try:
        from duckduckgo_search import DDGS
    except ImportError:
        print("[WebSearch] duckduckgo_search not installed, skipping")
        return []

    query = _build_search_query(user_profile)
    print(f"[WebSearch] Query: {query}")

    try:
        results = await asyncio.to_thread(_ddg_search, query, max_results)
    except Exception as e:
        print(f"[WebSearch] DuckDuckGo failed: {e}")
        return []

    if not results:
        print("[WebSearch] No results")
        return []

    search_text = ""
    for i, r in enumerate(results, 1):
        search_text += f"\n[{i}] {r.get('title','')}\n{r.get('body','')}\nURL: {r.get('href','')}\n"

    profile_lines = []
    for field, label in [
        ("state", "State"), ("gender", "Gender"), ("category", "Category"),
        ("age", "Age"), ("annual_family_income", "Income"),
        ("education_level", "Education"), ("course", "Course"),
        ("occupation", "Occupation"),
    ]:
        val = user_profile.get(field)
        if val is not None:
            profile_lines.append(f"- {label}: {val}")
    profile_str = "\n".join(profile_lines)

    prompt = WEB_SCHEME_EXTRACT_PROMPT.format(
        profile=profile_str,
        search_results=search_text[:3000],
    )

    try:
        from backend.model_registry import get_model
        from langchain_core.messages import HumanMessage

        model = get_model(model_id, api_key=api_key, model_config=model_config)
        response = await model.ainvoke([HumanMessage(content=prompt)])

        content = response.content
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
                if not (isinstance(item, dict) and item.get("type") in ("thinking", "reasoning"))
            )

        json_match = re.search(r"\[.*\]", str(content), re.DOTALL)
        if json_match:
            schemes_raw = json.loads(json_match.group())
            schemes = []
            for s in schemes_raw:
                if not s.get("is_eligible"):
                    continue
                if (s.get("confidence") or 0) < 0.7:
                    continue

                slug = s.get("slug") or re.sub(r"[^a-z0-9]+", "-", s.get("scheme_name", "").lower())[:50]
                if not slug.startswith("web-"):
                    slug = f"web-{slug}"

                scheme = {
                    "slug": slug,
                    "schemeName": s.get("scheme_name", ""),
                    "briefDescription": s.get("brief_description", ""),
                    "level": s.get("level", "Central"),
                    "beneficiaryState": [s["state"]] if s.get("state") and s["state"] != "All India" else [],
                    "schemeCategory": ["Education", "Scholarship"],
                    "tags": ["web-search", "live"],
                    "applicationUrl": s.get("application_url") or "",
                    "eligibility_text": s.get("eligibility_summary", ""),
                    "is_web_result": True,
                    "rules": [],
                    "llm_verified": True,
                    "llm_confidence": s.get("confidence", 0.7),
                    "llm_reason": s.get("eligible_reason", ""),
                }
                schemes.append(scheme)
                print(f"[WebSearch] Found: {s.get('scheme_name')} (conf={s.get('confidence')})")

            _WEB_SEARCH_CACHE[cache_key] = schemes
            print(f"[WebSearch] {len(schemes)} eligible schemes found")
            return schemes

    except Exception as e:
        print(f"[WebSearch] LLM extraction failed: {e}")

    _WEB_SEARCH_CACHE[cache_key] = []
    return []


def _ddg_search(query: str, max_results: int) -> list[dict]:
    """Synchronous DuckDuckGo search (run in thread)."""
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results, region="in-en", safesearch="off"))


def build_web_scheme_card(scheme: dict) -> dict:
    """Build a scheme card for a web-search result."""
    return {
        "slug": scheme.get("slug", ""),
        "name": scheme.get("schemeName", ""),
        "brief": scheme.get("briefDescription", ""),
        "level": scheme.get("level", "Central"),
        "states": scheme.get("beneficiaryState", []),
        "categories": scheme.get("schemeCategory", ["Education"]),
        "tags": scheme.get("tags", ["web-search"]),
        "has_details": False,
        "match_reasons": [f"✓ {scheme.get('llm_reason', 'Matches your profile')}"],
        "eligibility_status": "ELIGIBLE",
        "application_url": scheme.get("applicationUrl", ""),
        "is_private": False,
        "is_web_result": True,
    }
