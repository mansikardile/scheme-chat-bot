"""
RAG Pipeline for SchemeSathi — Strict Eligibility-First Architecture.

Pipeline modes:
  1. PROFILE_INFO    → Extract fields, update profile, ask next question. No cards.
  2. SCHEME_REQUEST  → Retrieve candidates, extract eligibility rules, run deterministic
                       eligibility engine, return ONLY verified ELIGIBLE schemes.
  3. DETAIL_REQUEST  → Existing behaviour: fetch full scheme detail and explain.
  4. GREETING / OTHER → Friendly response, no cards.
"""

import re
import asyncio

from backend.scheme_loader import scheme_loader
from backend.llm_service import (
    generate_response, generate_response_stream,
    extract_profile_and_intent, generate_profile_conversation, llm_verify_eligibility,
)
from backend.intent_classifier import classify_intent_fast, classify_intent_llm, Intent
from backend.profile_extractor import extract_profile_fields_fast, extract_profile_fields_llm
from backend.profile_advisor import get_next_question, format_profile_for_response
from backend.eligibility_engine import (
    evaluate_scheme_batch, EligibilityStatus, EligibilityRule,
)
from backend.scheme_eligibility_extractor import extract_scheme_rules
from backend.private_scheme_search import (
    search_private_schemes_ai,
    get_private_scheme_detail,
    get_private_scheme_context,
    PRIVATE_SCHEMES_CACHE,
)
from backend.web_scheme_search import search_schemes_web, build_web_scheme_card


# ---------------------------------------------------------------------------
# Helper: build scheme context string for LLM
# ---------------------------------------------------------------------------

def _build_scheme_context(slugs: list[str], user_message: str, session_history: list[dict]) -> str:
    parts = []
    for slug in slugs[:8]:  # Cap at 8 to avoid token overflow
        if slug.startswith('pvt-') or slug in PRIVATE_SCHEMES_CACHE:
            ctx = get_private_scheme_context(slug)
        else:
            ctx = scheme_loader.get_scheme_context(slug)
        if ctx:
            parts.append(ctx)
    return '\n\n---\n\n'.join(parts) if parts else "No specific scheme data available."


def _build_detail_context(slug: str) -> str:
    if slug.startswith('pvt-') or slug in PRIVATE_SCHEMES_CACHE:
        pvt_ctx = get_private_scheme_context(slug)
        if pvt_ctx:
            return pvt_ctx
    return scheme_loader.get_scheme_context(slug) or "Scheme details not found."


# ---------------------------------------------------------------------------
# Candidate retrieval from DuckDB
# ---------------------------------------------------------------------------

def _retrieve_candidates(user_profile: dict, user_message: str = '', limit: int = 40) -> list[dict]:
    """
    Retrieve candidate schemes from DuckDB using the user profile and query.
    This is a BROAD retrieval across all citizen domains (education, agriculture,
    weavers, artisans, business, employment, housing, health, women, disability, etc.)
    which are then strictly filtered by the deterministic eligibility engine.
    """
    from backend.config import SCHEMES_DB_PATH
    import duckdb

    state = user_profile.get('state') or ''
    course = user_profile.get('course') or ''
    edu = user_profile.get('education_level') or ''
    stage = user_profile.get('study_stage') or ''
    occupation = user_profile.get('occupation') or ''
    spec_cond = user_profile.get('special_condition') or ''
    gender = user_profile.get('gender') or ''
    msg_lower = (user_message or '').lower()

    try:
        conn = duckdb.connect(SCHEMES_DB_PATH, read_only=True)

        conditions = []
        params = []

        # 1. State filter: User's state OR Central
        if state:
            conditions.append("(states ILIKE ? OR level = 'Central')")
            params.append(f"%{state}%")

        # 2. Broad domain conditions
        domain_conditions = []

        # Weavers & Artisans
        if occupation in ('weaver', 'artisan') or spec_cond in ('handloom_weaver', 'traditional_artisan') or any(w in msg_lower for w in ['weaver', 'bunkar', 'vankar', 'handloom', 'powerloom', 'artisan', 'craftsman', 'karigar', 'vishwakarma']):
            domain_conditions.extend([
                "search_text ILIKE '%weaver%'",
                "search_text ILIKE '%handloom%'",
                "search_text ILIKE '%powerloom%'",
                "search_text ILIKE '%artisan%'",
                "search_text ILIKE '%craftsman%'",
                "search_text ILIKE '%textile%'",
                "search_text ILIKE '%vishwakarma%'",
                "categories ILIKE '%Handicrafts%'",
                "categories ILIKE '%Textiles%'",
                "categories ILIKE '%Skills%'",
            ])

        # Education & Students
        if edu or course or stage or occupation == 'student' or any(w in msg_lower for w in ['student', 'scholarship', 'study', 'college', 'engineering']):
            domain_conditions.extend([
                "categories ILIKE '%Education%'",
                "search_text ILIKE '%scholarship%'",
                "search_text ILIKE '%student%'",
                "search_text ILIKE '%education%'",
                "search_text ILIKE '%fellowship%'",
                "tags ILIKE '%scholarship%'",
            ])
            if course:
                domain_conditions.append("search_text ILIKE ?")
                params.append(f"%{course}%")

        # Farmers & Agriculture
        if user_profile.get('farmer_status') or occupation == 'farmer' or any(w in msg_lower for w in ['farmer', 'kisan', 'krishi', 'crop', 'dairy']):
            domain_conditions.extend([
                "categories ILIKE '%Agriculture%'",
                "search_text ILIKE '%farmer%'",
                "search_text ILIKE '%kisan%'",
                "search_text ILIKE '%krishi%'",
                "search_text ILIKE '%crop%'",
            ])

        # Business / MSME / Street Vendors
        if occupation in ('business_owner', 'street_vendor') or user_profile.get('employment_status') == 'self_employed' or any(w in msg_lower for w in ['business', 'msme', 'loan', 'vendor', 'shop', 'svanidhi', 'mudra', 'startup']):
            domain_conditions.extend([
                "categories ILIKE '%Business%'",
                "categories ILIKE '%Banking%'",
                "search_text ILIKE '%msme%'",
                "search_text ILIKE '%business%'",
                "search_text ILIKE '%vendor%'",
                "search_text ILIKE '%svanidhi%'",
                "search_text ILIKE '%mudra%'",
                "search_text ILIKE '%enterprise%'",
                "search_text ILIKE '%loan%'",
            ])

        # Construction & Unorganized Workers
        if occupation == 'construction_worker' or spec_cond in ('construction_worker', 'landless_labourer') or any(w in msg_lower for w in ['worker', 'labour', 'construction', 'mazdoor', 'eshram']):
            domain_conditions.extend([
                "search_text ILIKE '%construction%'",
                "search_text ILIKE '%labour%'",
                "search_text ILIKE '%worker%'",
                "search_text ILIKE '%bocw%'",
                "search_text ILIKE '%unorganized%'",
                "search_text ILIKE '%eshram%'",
            ])

        # Teachers, Faculty & Researchers
        if occupation == 'teacher' or any(w in msg_lower for w in ['teacher', 'faculty', 'professor', 'lecturer', 'educator']):
            domain_conditions.extend([
                "search_text ILIKE '%teacher%'",
                "search_text ILIKE '%faculty%'",
                "search_text ILIKE '%professor%'",
                "search_text ILIKE '%training%'",
                "search_text ILIKE '%research%'",
                "categories ILIKE '%Education%'",
                "categories ILIKE '%Skills%'",
            ])

        # Housing
        if user_profile.get('housing_status') or any(w in msg_lower for w in ['housing', 'house', 'awas', 'home']):
            domain_conditions.extend([
                "categories ILIKE '%Housing%'",
                "search_text ILIKE '%awas%'",
                "search_text ILIKE '%housing%'",
            ])

        # Health
        if any(w in msg_lower for w in ['health', 'hospital', 'treatment', 'ayushman', 'medical', 'insurance']):
            domain_conditions.extend([
                "categories ILIKE '%Health%'",
                "search_text ILIKE '%ayushman%'",
                "search_text ILIKE '%health%'",
                "search_text ILIKE '%medical%'",
            ])

        # Women specific
        if gender == 'Female' or any(w in msg_lower for w in ['women', 'girl', 'mahila', 'widow', 'female', 'kanya']):
            domain_conditions.extend([
                "categories ILIKE '%Women%'",
                "search_text ILIKE '%women%'",
                "search_text ILIKE '%mahila%'",
                "search_text ILIKE '%kanya%'",
            ])

        # Senior Citizens
        if (user_profile.get('age') and int(user_profile.get('age', 0)) >= 60) or any(w in msg_lower for w in ['senior', 'pension', 'elderly', 'old age']):
            domain_conditions.extend([
                "search_text ILIKE '%senior%'",
                "search_text ILIKE '%pension%'",
                "search_text ILIKE '%elderly%'",
                "search_text ILIKE '%old age%'",
            ])

        # Disability
        if user_profile.get('disability_status') == 'disabled' or any(w in msg_lower for w in ['disabled', 'divyang', 'pwd', 'handicap']):
            domain_conditions.extend([
                "search_text ILIKE '%disabled%'",
                "search_text ILIKE '%divyang%'",
                "search_text ILIKE '%pwd%'",
            ])

        if domain_conditions:
            conditions.append("(" + " OR ".join(domain_conditions) + ")")

        sql = "SELECT slug FROM schemes"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        if state:
            sql += f" ORDER BY CASE WHEN states ILIKE '%{state}%' THEN 0 ELSE 1 END, scheme_name ASC"
        else:
            sql += " ORDER BY scheme_name ASC"

        sql += f" LIMIT {limit}"

        rows = conn.execute(sql, params).fetchall()
        conn.close()

        candidates = []
        for r in rows:
            slug = r[0]
            summary = scheme_loader.slug_to_index.get(slug)
            if summary:
                candidates.append(summary)

        if state and candidates:
            candidates = scheme_loader._filter_geo_restricted(candidates, state)

        print(f"[RAGPipeline] Retrieved {len(candidates)} candidates for eligibility checking")
        return candidates
    except Exception as e:
        print(f"[RAGPipeline] DuckDB candidate retrieval error ({e}), falling back to scheme_loader.search_schemes")
        return scheme_loader.search_schemes(limit=limit)


# ---------------------------------------------------------------------------
# No-match response
# ---------------------------------------------------------------------------

def _no_match_response(user_profile: dict, checked_count: int, language: str = 'en') -> str:
    fields_checked = []
    if user_profile.get('state'):
        fields_checked.append(f"State: {user_profile['state']}")
    if user_profile.get('category'):
        fields_checked.append(f"Category: {user_profile['category']}")
    if user_profile.get('gender'):
        fields_checked.append(f"Gender: {user_profile['gender']}")
    if user_profile.get('education_level'):
        fields_checked.append(f"Education: {user_profile['education_level']}")
    if user_profile.get('annual_family_income') is not None:
        income = user_profile['annual_family_income']
        if isinstance(income, (list, tuple)):
            fields_checked.append(f"Income: ₹{income[0]//100_000}L–₹{income[1]//100_000}L")
        else:
            fields_checked.append(f"Income: ₹{income//100_000}L")

    criteria_text = '\n'.join(f"• {f}" for f in fields_checked) if fields_checked else "• General criteria"
    return (
        f"I checked {checked_count} schemes against your profile but couldn't find a "
        f"verified scheme that satisfies all your eligibility criteria.\n\n"
        f"Criteria I used:\n{criteria_text}\n\n"
        f"**What you can do:**\n"
        f"• Double-check your details (category, income, education level)\n"
        f"• Ask me to search with relaxed criteria\n"
        f"• Try mentioning a specific scheme name if you have one in mind"
    )


# ---------------------------------------------------------------------------
# Format eligible scheme results
# ---------------------------------------------------------------------------

def _format_eligible_results(
    eligible: list[tuple[str, str, object]],
    user_profile: dict,
) -> str:
    """Build the conversational reply when eligible schemes are found."""
    gov_list = [(s, n, r) for s, n, r in eligible if not (s.startswith('pvt-') or s in PRIVATE_SCHEMES_CACHE)]
    pvt_list = [(s, n, r) for s, n, r in eligible if (s.startswith('pvt-') or s in PRIVATE_SCHEMES_CACHE)]

    user_occ = (user_profile.get('occupation') or '').lower()
    user_farmer = bool(user_profile.get('farmer_status') or user_occ == 'farmer')
    pvt_label = "Private & CSR Trust initiative(s)" if user_farmer else "Private & CSR Trust scholarship(s)"
    pvt_header = "### 🏢 Private & CSR Trust Initiatives" if user_farmer else "### 🏢 Private & CSR Trust Scholarships"

    summary_parts = []
    if gov_list:
        summary_parts.append(f"**{len(gov_list)} Government scheme(s)**")
    if pvt_list:
        summary_parts.append(f"**{len(pvt_list)} {pvt_label}**")

    breakdown = " and ".join(summary_parts) if summary_parts else f"**{len(eligible)} verified scheme(s)**"
    lines = [f"I found {breakdown} matching your complete profile:\n"]

    if gov_list:
        lines.append("### 🏛️ Government Schemes")
        for i, (slug, name, result) in enumerate(gov_list, 1):
            lines.append(f"**{i}. {name}**")
            if result.match_reasons:
                for reason in result.match_reasons[:4]:
                    lines.append(f"   {reason}")
            lines.append("")

    if pvt_list:
        lines.append(pvt_header)
        for i, (slug, name, result) in enumerate(pvt_list, 1):
            lines.append(f"**{i}. {name}**")
            if result.match_reasons:
                for reason in result.match_reasons[:4]:
                    lines.append(f"   {reason}")
            lines.append("")

    lines.append("Tap any scheme card below to view full details, benefits, and direct application portals.")
    return '\n'.join(lines)



# ---------------------------------------------------------------------------
# Main RAG Pipeline class
# ---------------------------------------------------------------------------

class RAGPipeline:
    """Strict eligibility-first RAG pipeline for SchemeSathi."""

    # ------------------------------------------------------------------
    # PROFILE COLLECTION mode
    # ------------------------------------------------------------------

    async def handle_profile_info(
        self,
        user_message: str,
        session_history: list[dict],
        session,
        language: str = 'en',
        model_id: str = 'gemini-flash',
        api_key: str | None = None,
        model_config: dict | None = None,
    ) -> tuple[str, list[dict], dict]:
        """
        Handle a profile-info message — fully LLM-powered.

        1. Single LLM call extracts intent + new fields + fields to clear (handles
           corrections, negations, conversational context — no regex)
        2. Clears corrected fields, merges new ones into session profile
        3. Determines the next missing field via profile_advisor logic
        4. LLM generates a warm, natural conversational reply asking that one question
        """
        # Step 1: LLM extracts intent + profile fields + corrections in one call
        llm_result = await extract_profile_and_intent(
            user_message=user_message,
            conversation_history=session_history,
            model_id=model_id,
            api_key=api_key,
            model_config=model_config,
        )

        new_fields = llm_result.get('fields') or {}
        clear_fields = llm_result.get('clear_fields') or []
        llm_intent = llm_result.get('intent', 'PROFILE_INFO')

        # If LLM classified this as SCHEME_REQUEST, hand off to scheme handler
        if llm_intent == 'SCHEME_REQUEST':
            return await self.handle_scheme_request(
                user_message, session_history, session, language, model_id, api_key, model_config
            )

        # Step 2a: Clear corrected fields first
        if clear_fields:
            session.clear_profile_fields(clear_fields)

        # Step 2b: Merge new fields into profile
        if new_fields:
            session.update_profile(new_fields)

        profile = session.get_profile()

        # Step 3: Determine next missing field using advisor logic
        next_field, next_question = get_next_question(
            profile, session_history, user_message
        )

        # Step 4: LLM generates natural conversational reply
        reply = await generate_profile_conversation(
            user_message=user_message,
            user_profile=profile,
            next_question=next_question,
            conversation_history=session_history,
            language=language,
            model_id=model_id,
            api_key=api_key,
            model_config=model_config,
        )

        return reply, [], profile  # No scheme cards in profile collection mode

    def _detect_contradiction(self, new_fields: dict, existing_profile: dict) -> str | None:
        """
        Detect contradictions between new fields and existing profile.
        Returns a clarification question if contradiction detected, else None.
        """
        CATEGORY_GROUPS = {
            'SC': {'SC'},
            'ST': {'ST'},
            'OBC': {'OBC', 'VJNT', 'SBC', 'NT'},
            'EWS': {'EWS'},
            'General': {'General'},
            'Minority': {'Minority'},
        }

        existing_cat = existing_profile.get('category')
        new_cat = new_fields.get('category')

        if existing_cat and new_cat and existing_cat != new_cat:
            # Check if they're in the same group
            existing_group = next((g for g, members in CATEGORY_GROUPS.items() if existing_cat in members), existing_cat)
            new_group = next((g for g, members in CATEGORY_GROUPS.items() if new_cat in members), new_cat)
            if existing_group != new_group:
                return (
                    f"You previously mentioned **{existing_cat}** category, "
                    f"but now you've mentioned **{new_cat}**. "
                    f"Which one should I use for your eligibility check?"
                )
        return None

    # ------------------------------------------------------------------
    # SCHEME SEARCH mode
    # ------------------------------------------------------------------

    async def handle_scheme_request(
        self,
        user_message: str,
        session_history: list[dict],
        session,
        language: str = 'en',
        model_id: str = 'gemini-flash',
        api_key: str | None = None,
        model_config: dict | None = None,
    ) -> tuple[str, list[dict], dict]:
        """
        Full eligibility pipeline with LLM final verification:
        1. LLM extracts any profile fields embedded in the request
        2. Retrieve broad candidates from DB
        3. LLM extracts eligibility rules per scheme (cached)
        4. Deterministic eligibility engine filters candidates
        5. LLM final verification cross-checks each ELIGIBLE scheme vs full profile
        6. Return only 100%-verified eligible schemes
        """
        # Extract any profile fields embedded in the request (LLM-powered)
        llm_result = await extract_profile_and_intent(
            user_message=user_message,
            conversation_history=session_history,
            model_id=model_id,
            api_key=api_key,
            model_config=model_config,
        )
        new_fields = llm_result.get('fields') or {}
        clear_fields = llm_result.get('clear_fields') or []

        if clear_fields:
            session.clear_profile_fields(clear_fields)
        if new_fields:
            session.update_profile(new_fields)

        profile = session.get_profile()
        session.scheme_search_requested = True
        session.conversation_mode = 'scheme_search'

        if not session.has_minimum_profile():
            return (
                "I don't have enough information about you yet to search schemes reliably. "
                "Please tell me at least your **state** and one more detail "
                "(category, education level, or income) so I can check eligibility properly.",
                [], profile
            )

        # 1. Broad candidate retrieval: Government (DuckDB) + Private/CSR + Web Search (parallel)
        gov_task = asyncio.to_thread(_retrieve_candidates, profile, user_message, 40)
        pvt_task = search_private_schemes_ai(profile, model_id=model_id, api_key=api_key, model_config=model_config)
        web_task = search_schemes_web(profile, model_id=model_id, api_key=api_key, model_config=model_config)

        gov_candidates, pvt_candidates, web_candidates = await asyncio.gather(gov_task, pvt_task, web_task)

        # Web candidates are already LLM-verified — separate them out before deterministic pipeline
        # They will be added directly to eligible results after the deterministic pipeline runs
        db_candidates = list(pvt_candidates) + list(gov_candidates)
        if not db_candidates and not web_candidates:
            return _no_match_response(profile, 0), [], profile

        # 2. Extract eligibility rules for each candidate in parallel (LLM, cached)
        async def _fetch_candidate_rules(candidate):
            slug = candidate.get('slug', '')
            name = candidate.get('schemeName', slug)
            states = candidate.get('beneficiaryState', [])
            categories = candidate.get('schemeCategory', [])

            if candidate.get('rules'):
                session.eligibility_cache[slug] = candidate['rules']
                return slug, name, candidate['rules'], ''

            if slug in session.eligibility_cache:
                return slug, name, session.eligibility_cache[slug], ''

            # Get eligibility text
            if slug.startswith('pvt-') or slug in PRIVATE_SCHEMES_CACHE:
                detailed_raw = get_private_scheme_detail(slug) or {}
                elig_data = detailed_raw.get('en', {}).get('eligibilityCriteria', {})
                eligibility_text = elig_data.get('eligibilityDescription_md', '') if isinstance(elig_data, dict) else str(elig_data)
                if not eligibility_text:
                    eligibility_text = candidate.get('briefDescription', '')
            else:
                detailed = scheme_loader.detailed_schemes.get(slug, {})
                en_data = detailed.get('en', {})
                elig_data = en_data.get('eligibilityCriteria', {})
                if isinstance(elig_data, dict):
                    eligibility_text = elig_data.get('eligibilityDescription_md', '')
                else:
                    eligibility_text = str(elig_data)
                if not eligibility_text:
                    eligibility_text = candidate.get('briefDescription', '')

            rules = await extract_scheme_rules(
                slug=slug,
                scheme_name=name,
                eligibility_text=eligibility_text or '',
                states=states,
                categories=categories,
                model_id=model_id,
                api_key=api_key,
                model_config=model_config,
            )
            session.eligibility_cache[slug] = rules
            # Return eligibility_text so we can use it for LLM verification
            return slug, name, rules, eligibility_text or ''

        # Only process db_candidates through deterministic + LLM pipeline
        if db_candidates:
            scheme_rules_list = await asyncio.gather(*[_fetch_candidate_rules(c) for c in db_candidates])
        else:
            scheme_rules_list = []

        # 3. Deterministic evaluation
        # scheme_rules_list now has tuples of (slug, name, rules, eligibility_text)
        evaluation_results = evaluate_scheme_batch(
            profile,
            [(slug, name, rules) for slug, name, rules, _ in scheme_rules_list]
        )

        # 4. Separate deterministic results
        eligible_det = [
            (slug, name, result)
            for slug, name, result in evaluation_results
            if result.status == EligibilityStatus.ELIGIBLE
        ]
        insufficient = [
            (slug, name, result)
            for slug, name, result in evaluation_results
            if result.status == EligibilityStatus.INSUFFICIENT_INFORMATION
        ]

        print(f"[RAGPipeline] Deterministic: "
              f"{len(eligible_det)} ELIGIBLE, "
              f"{len([r for _, _, r in evaluation_results if r.status == EligibilityStatus.INELIGIBLE])} INELIGIBLE, "
              f"{len(insufficient)} INSUFFICIENT")

        # 5. LLM final verification — cross-check each deterministically-eligible scheme
        # Build a lookup for eligibility text
        elig_text_map = {slug: elig_text for slug, _, _, elig_text in scheme_rules_list}

        async def _llm_verify_scheme(slug, name, det_result):
            elig_text = elig_text_map.get(slug, '')
            llm_check = await llm_verify_eligibility(
                user_profile=profile,
                scheme_name=name,
                eligibility_text=elig_text,
                model_id=model_id,
                api_key=api_key,
                model_config=model_config,
            )
            return slug, name, det_result, llm_check

        if eligible_det:
            verified_results = await asyncio.gather(
                *[_llm_verify_scheme(slug, name, result) for slug, name, result in eligible_det]
            )
        else:
            verified_results = []

        # 6. Keep only schemes that pass LLM verification (confidence >= 0.75, strict)
        eligible = []
        llm_rejected = []
        for slug, name, det_result, llm_check in verified_results:
            status = llm_check.get('status', 'ELIGIBLE')
            confidence = llm_check.get('confidence', 0.7)
            if status == 'ELIGIBLE' and confidence >= 0.75:
                eligible.append((slug, name, det_result))
            elif status == 'INELIGIBLE':
                print(f"[LLMVerify] Removed '{name}' — {llm_check.get('reason', '')}")
                llm_rejected.append((slug, name, det_result))
            else:
                # INSUFFICIENT or low confidence — only keep if confidence is reasonably high
                if confidence >= 0.60:
                    eligible.append((slug, name, det_result))
                else:
                    print(f"[LLMVerify] Excluded '{name}' (low confidence {confidence:.2f}) — {llm_check.get('reason', '')}")
                    llm_rejected.append((slug, name, det_result))


        # 7. Add web-search results (pre-verified by LLM) to the eligible set
        web_scheme_cards = []
        if web_candidates:
            from backend.eligibility_engine import EligibilityResult, EligibilityStatus as ES
            for wc in web_candidates:
                slug = wc.get('slug', '')
                name = wc.get('schemeName', '')
                # Create a synthetic ELIGIBLE result for web schemes
                web_result = EligibilityResult(
                    status=ES.ELIGIBLE,
                    match_reasons=[f"✓ {wc.get('llm_reason', 'Matches your profile (web)')}"],
                )
                eligible.append((slug, name, web_result))
                session.last_eligible_slugs.append(slug)
                web_scheme_cards.append(build_web_scheme_card(wc))

        session.last_eligible_slugs = [slug for slug, _, _ in eligible]

        print(f"[RAGPipeline] After LLM verification: {len(eligible)} confirmed ELIGIBLE "
              f"({len(llm_rejected)} removed by LLM, {len(web_candidates)} from web)")

        # Build response
        if eligible:
            reply = _format_eligible_results(eligible, profile)
            db_scheme_cards = [
                self._build_card(slug, name, result)
                for slug, name, result in eligible
                if not any(wc['slug'] == slug for wc in web_candidates)
            ]
            all_cards = db_scheme_cards + web_scheme_cards
            return reply, all_cards, profile
        elif insufficient:
            missing_counts: dict[str, int] = {}
            for _, _, result in insufficient:
                for f in result.missing_fields:
                    missing_counts[f] = missing_counts.get(f, 0) + 1
            top_missing = max(missing_counts, key=missing_counts.get) if missing_counts else None

            if top_missing:
                from backend.profile_advisor import FIELD_QUESTIONS
                question = FIELD_QUESTIONS.get(top_missing, f"What is your {top_missing.replace('_', ' ')}?")
                reply = (
                    f"I found {len(insufficient)} potentially matching scheme(s), but I need one more detail "
                    f"to confirm your eligibility:\n\n{question}"
                )
            else:
                reply = _no_match_response(profile, len(db_candidates))
            return reply, [], profile
        else:
            return _no_match_response(profile, len(db_candidates)), [], profile

    def _build_card(self, slug: str, name: str, result) -> dict:
        """Build a scheme card dict with match reasons."""
        if slug.startswith('pvt-') or slug in PRIVATE_SCHEMES_CACHE:
            summary = PRIVATE_SCHEMES_CACHE.get(slug, {})
            return {
                'slug': slug,
                'name': name,
                'brief': summary.get('briefDescription', ''),
                'level': summary.get('level', 'Private / Trust'),
                'states': summary.get('beneficiaryState', []),
                'categories': summary.get('schemeCategory', ['Private & CSR Scholarships']),
                'tags': summary.get('tags', []),
                'has_details': True,
                'match_reasons': result.match_reasons,
                'eligibility_status': result.status.value,
                'application_url': summary.get('applicationUrl') or ('https://www.google.com/search?q=' + name.replace(' ', '+')),
                'is_private': True,
            }

        summary = scheme_loader.slug_to_index.get(slug, {})
        return {
            'slug': slug,
            'name': name,
            'brief': summary.get('briefDescription', ''),
            'level': summary.get('level', ''),
            'states': summary.get('beneficiaryState', []),
            'categories': summary.get('schemeCategory', []),
            'tags': summary.get('tags', []),
            'has_details': bool(scheme_loader.detailed_schemes.get(slug)),
            'match_reasons': result.match_reasons,
            'eligibility_status': result.status.value,
            'application_url': f"https://www.myscheme.gov.in/schemes/{slug}",
            'is_private': False,
        }

    # ------------------------------------------------------------------
    # DETAIL REQUEST mode
    # ------------------------------------------------------------------

    async def handle_detail_request(
        self,
        user_message: str,
        session_history: list[dict],
        session,
        language: str = 'en',
        model_id: str = 'gemini-flash',
        api_key: str | None = None,
        model_config: dict | None = None,
    ) -> tuple[str, list[dict], dict]:
        """Handle requests for specific scheme details."""
        profile = session.get_profile()

        # Try to find the scheme the user is asking about
        slug = None
        # Check if asking about one of the last eligible schemes
        if session.last_eligible_slugs:
            for s in session.last_eligible_slugs:
                if s.startswith('pvt-') or s in PRIVATE_SCHEMES_CACHE:
                    s_name = PRIVATE_SCHEMES_CACHE.get(s, {}).get('schemeName', '')
                else:
                    s_name = scheme_loader.slug_to_index.get(s, {}).get('schemeName', '')
                if s_name and (s_name.lower()[:12] in user_message.lower() or any(w in user_message.lower() for w in s_name.lower().split() if len(w) > 4)):
                    slug = s
                    break
            if not slug and len(session.last_eligible_slugs) == 1:
                slug = session.last_eligible_slugs[0]

        # Try to match scheme name in private cache
        if not slug:
            for s, s_data in PRIVATE_SCHEMES_CACHE.items():
                if not s.endswith('__raw'):
                    s_name = s_data.get('schemeName', '')
                    if s_name and any(w in user_message.lower() for w in s_name.lower().split() if len(w) > 4):
                        slug = s
                        break

        if not slug:
            # Find by text matching in DuckDB
            found = scheme_loader.search_schemes(query=user_message, limit=1)
            slug = found[0].get('slug') if found else None

        if slug:
            scheme_context = _build_detail_context(slug)
            reply = await generate_response(
                model_id, session_history, user_message, scheme_context,
                language=language, api_key=api_key, model_config=model_config,
                user_profile=profile,
            )
            return reply, [], profile
        else:
            reply = await generate_response(
                model_id, session_history, user_message,
                "No specific scheme context available.",
                language=language, api_key=api_key, model_config=model_config,
                user_profile=profile,
            )
            return reply, [], profile

    # ------------------------------------------------------------------
    # Unified entry points
    # ------------------------------------------------------------------

    async def process_query(
        self,
        session_history: list[dict],
        user_message: str,
        language: str = 'en',
        model_id: str = 'gemini-flash',
        api_key: str | None = None,
        model_config: dict | None = None,
        user_profile: dict | None = None,
        session=None,
    ) -> tuple[str, list[dict], dict]:
        """
        Full RAG pipeline (non-streaming) — LLM-first intent classification.

        Returns:
            (reply_text, scheme_cards, updated_profile)
        """
        profile = (session.get_profile() if session else {}) or user_profile or {}

        # Fast check for greetings (no LLM needed)
        msg_lower = user_message.lower().strip()
        import re as _re
        if _re.match(r'^(?:hi|hello|hey|namaste|namaskar|hii+|heyyy*)[!.\s]*$', msg_lower):
            reply = (
                "Hello! 👋 I'm SchemeSathi — I help you find government schemes and scholarships "
                "you're actually eligible for.\n\n"
                "To get started, tell me a bit about yourself — which state are you from, "
                "and what kind of help are you looking for? (education, farming, business, health...)"
            )
            return reply, [], profile

        # Fast check for explicit scheme requests
        intent_fast = classify_intent_fast(user_message, session_history)
        if intent_fast == Intent.SCHEME_REQUEST:
            if session:
                return await self.handle_scheme_request(
                    user_message, session_history, session, language, model_id, api_key, model_config
                )
            return "Please start a new chat session to search for schemes.", [], profile

        if intent_fast == Intent.DETAIL_REQUEST:
            if session:
                return await self.handle_detail_request(
                    user_message, session_history, session, language, model_id, api_key, model_config
                )
            return await self._generic_response(
                user_message, session_history, profile, language, model_id, api_key, model_config
            )

        # For all other messages (PROFILE_INFO, CLARIFICATION, OTHER) — use LLM-first handler
        if session:
            reply, cards, updated_profile = await self.handle_profile_info(
                user_message, session_history, session, language, model_id, api_key, model_config
            )
        else:
            reply, cards, updated_profile = await self._generic_response(
                user_message, session_history, profile, language, model_id, api_key, model_config
            )

        return reply, cards, updated_profile

    async def process_query_stream(
        self,
        session_history: list[dict],
        user_message: str,
        language: str = 'en',
        model_id: str = 'gemini-flash',
        api_key: str | None = None,
        model_config: dict | None = None,
        user_profile: dict | None = None,
        session=None,
    ):
        """
        Streaming RAG pipeline — LLM-first intent classification.

        Yields dicts:
          {"type": "text",    "content": "<chunk>"}
          {"type": "cards",   "content": [<card_dict>, ...]}
          {"type": "profile", "content": <profile_dict>}
        """
        profile = (session.get_profile() if session else {}) or user_profile or {}

        # Fast check for greetings
        msg_lower = user_message.lower().strip()
        import re as _re
        if _re.match(r'^(?:hi|hello|hey|namaste|namaskar|hii+|heyyy*)[!.\s]*$', msg_lower):
            greeting = (
                "Hello! 👋 I'm SchemeSathi — I help you find government schemes and scholarships "
                "you're actually eligible for.\n\n"
                "To get started, tell me a bit about yourself — which state are you from, "
                "and what kind of help are you looking for? (education, farming, business, health...)"
            )
            yield {"type": "text", "content": greeting}
            yield {"type": "cards", "content": []}
            yield {"type": "profile", "content": profile}
            return

        # Fast check for explicit scheme requests
        intent_fast = classify_intent_fast(user_message, session_history)

        if intent_fast == Intent.SCHEME_REQUEST:
            if session:
                reply, cards, updated_profile = await self.handle_scheme_request(
                    user_message, session_history, session, language, model_id, api_key, model_config
                )
                chunk_size = 50
                for i in range(0, len(reply), chunk_size):
                    yield {"type": "text", "content": reply[i:i+chunk_size]}
                    await asyncio.sleep(0)
                yield {"type": "cards", "content": cards}
                yield {"type": "profile", "content": updated_profile}
            else:
                yield {"type": "text", "content": "Please start a new chat session."}
                yield {"type": "cards", "content": []}
                yield {"type": "profile", "content": profile}
            return

        if intent_fast == Intent.DETAIL_REQUEST:
            if session:
                reply, cards, updated_profile = await self.handle_detail_request(
                    user_message, session_history, session, language, model_id, api_key, model_config
                )
                for chunk in [reply[i:i+50] for i in range(0, len(reply), 50)]:
                    yield {"type": "text", "content": chunk}
                    await asyncio.sleep(0)
                yield {"type": "cards", "content": cards}
                yield {"type": "profile", "content": updated_profile}
            else:
                async for chunk in generate_response_stream(
                    model_id, session_history, user_message, "No context.",
                    language=language, api_key=api_key, model_config=model_config,
                    user_profile=profile,
                ):
                    yield {"type": "text", "content": chunk}
                yield {"type": "cards", "content": []}
                yield {"type": "profile", "content": profile}
            return

        # All other messages → LLM-first profile handler
        if session:
            reply, cards, updated_profile = await self.handle_profile_info(
                user_message, session_history, session, language, model_id, api_key, model_config
            )
        else:
            reply = await generate_response(
                model_id, session_history, user_message, "",
                language=language, api_key=api_key, model_config=model_config,
                user_profile=profile,
            )
            cards, updated_profile = [], profile

        chunk_size = 50
        for i in range(0, len(reply), chunk_size):
            yield {"type": "text", "content": reply[i:i+chunk_size]}
            await asyncio.sleep(0)
        yield {"type": "cards", "content": []}  # NO cards in profile collection
        yield {"type": "profile", "content": updated_profile}

    async def _generic_response(
        self, user_message, session_history, profile, language, model_id, api_key, model_config
    ) -> tuple[str, list, dict]:
        reply = await generate_response(
            model_id, session_history, user_message, "",
            language=language, api_key=api_key, model_config=model_config,
            user_profile=profile,
        )
        return reply, [], profile


rag_pipeline = RAGPipeline()
