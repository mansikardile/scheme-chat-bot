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
from backend.llm_service import generate_response, generate_response_stream
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

def _retrieve_candidates(user_profile: dict, limit: int = 35) -> list[dict]:
    """
    Retrieve candidate schemes from DuckDB using the user profile.
    This is a BROAD retrieval — we get candidates based on state and domain
    (education, agriculture, employment, etc.) and then filter deterministically.
    """
    from backend.config import SCHEMES_DB_PATH
    import duckdb

    state = user_profile.get('state') or ''
    course = user_profile.get('course') or ''
    edu = user_profile.get('education_level') or ''
    stage = user_profile.get('study_stage') or ''
    occupation = user_profile.get('occupation') or ''

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
        if edu or course or stage:
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

        if user_profile.get('farmer_status') or occupation == 'farmer':
            domain_conditions.extend([
                "categories ILIKE '%Agriculture%'",
                "search_text ILIKE '%farmer%'",
                "search_text ILIKE '%kisan%'",
            ])

        if user_profile.get('housing_status'):
            domain_conditions.extend([
                "categories ILIKE '%Housing%'",
                "search_text ILIKE '%awas%'",
                "search_text ILIKE '%housing%'",
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

    summary_parts = []
    if gov_list:
        summary_parts.append(f"**{len(gov_list)} Government scheme(s)**")
    if pvt_list:
        summary_parts.append(f"**{len(pvt_list)} Private & CSR Trust scholarship(s)**")

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
        lines.append("### 🏢 Private & CSR Trust Scholarships")
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
        Handle a profile-info message:
        1. Extract new fields
        2. Merge into session profile
        3. Determine next missing field
        4. Return acknowledgment + question (no cards)
        """
        # Extract new fields from this message
        new_fields = extract_profile_fields_fast(user_message)
        if not new_fields and session_history:
            try:
                new_fields = await extract_profile_fields_llm(
                    user_message, session_history, model_id, api_key=api_key, model_config=model_config
                )
            except Exception as e:
                print(f"[RAGPipeline] LLM profile extraction error: {e}")
                new_fields = {}

        # Handle contradictions
        contradiction = self._detect_contradiction(new_fields, session.get_profile())
        if contradiction:
            reply = contradiction
            return reply, [], session.get_profile()

        session.update_profile(new_fields)
        profile = session.get_profile()

        # Determine next missing field
        next_field, next_question = get_next_question(
            profile, session_history, user_message
        )

        reply = format_profile_for_response(profile, next_field)
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
        Full eligibility pipeline:
        1. Retrieve broad candidates
        2. Extract eligibility rules per scheme (LLM, cached)
        3. Run deterministic eligibility engine
        4. Return ALL eligible schemes
        """
        # Extract any profile fields embedded in the request message (e.g. "I'm a student from Assam, show schemes")
        new_fields = extract_profile_fields_fast(user_message)
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

        # 1. Broad candidate retrieval: Government (DuckDB) + Private/CSR (Curated + AI Search)
        gov_task = asyncio.to_thread(_retrieve_candidates, profile, limit=25)
        pvt_task = search_private_schemes_ai(profile, model_id=model_id, api_key=api_key, model_config=model_config)

        gov_candidates, pvt_candidates = await asyncio.gather(gov_task, pvt_task)
        # Place private and government candidates together
        candidates = list(pvt_candidates) + list(gov_candidates)
        if not candidates:
            return _no_match_response(profile, 0), [], profile

        # 2. Extract eligibility rules for each candidate in parallel
        async def _fetch_candidate_rules(candidate):
            slug = candidate.get('slug', '')
            name = candidate.get('schemeName', slug)
            states = candidate.get('beneficiaryState', [])
            categories = candidate.get('schemeCategory', [])

            # Check if candidate already has pre-attached rules
            if candidate.get('rules'):
                session.eligibility_cache[slug] = candidate['rules']
                return slug, name, candidate['rules']

            # Check session cache
            if slug in session.eligibility_cache:
                return slug, name, session.eligibility_cache[slug]

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
            return slug, name, rules

        scheme_rules_list = await asyncio.gather(*[_fetch_candidate_rules(c) for c in candidates])

        # 3. Deterministic evaluation
        evaluation_results = evaluate_scheme_batch(profile, scheme_rules_list)

        # 4. Separate results
        eligible = [
            (slug, name, result)
            for slug, name, result in evaluation_results
            if result.status == EligibilityStatus.ELIGIBLE
        ]
        insufficient = [
            (slug, name, result)
            for slug, name, result in evaluation_results
            if result.status == EligibilityStatus.INSUFFICIENT_INFORMATION
        ]

        session.last_eligible_slugs = [slug for slug, _, _ in eligible]

        print(f"[RAGPipeline] Eligibility results: "
              f"{len(eligible)} ELIGIBLE, "
              f"{len([r for _, _, r in evaluation_results if r.status == EligibilityStatus.INELIGIBLE])} INELIGIBLE, "
              f"{len(insufficient)} INSUFFICIENT")

        # Build response
        if eligible:
            reply = _format_eligible_results(eligible, profile)
            scheme_cards = [
                self._build_card(slug, name, result)
                for slug, name, result in eligible
            ]
            return reply, scheme_cards, profile
        elif insufficient:
            # Find the most common missing field across INSUFFICIENT results
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
                reply = _no_match_response(profile, len(candidates))
            return reply, [], profile
        else:
            return _no_match_response(profile, len(candidates)), [], profile

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
        Full RAG pipeline (non-streaming).

        Returns:
            (reply_text, scheme_cards, updated_profile)
        """
        profile = user_profile or {}

        # Classify intent
        intent = await classify_intent_llm(
            user_message, session_history, model_id, api_key, model_config
        )
        print(f"[RAGPipeline] Intent: {intent.value}")

        if intent == Intent.SCHEME_REQUEST:
            if session:
                reply, cards, updated_profile = await self.handle_scheme_request(
                    user_message, session_history, session, language, model_id, api_key, model_config
                )
            else:
                reply = "Please start a new chat session to search for schemes."
                cards, updated_profile = [], profile
        elif intent == Intent.DETAIL_REQUEST:
            if session:
                reply, cards, updated_profile = await self.handle_detail_request(
                    user_message, session_history, session, language, model_id, api_key, model_config
                )
            else:
                reply, cards, updated_profile = await self._generic_response(
                    user_message, session_history, profile, language, model_id, api_key, model_config
                )
        elif intent == Intent.GREETING:
            reply = (
                "Hello! 👋 I'm SchemeSathi — I help you find government schemes and scholarships "
                "you're actually eligible for.\n\n"
                "To get started, tell me a bit about yourself:\n"
                "• Which state are you from?\n"
                "• Are you a student, farmer, entrepreneur, or something else?\n"
                "• Any specific type of scheme you're looking for?"
            )
            cards, updated_profile = [], profile
        else:
            # PROFILE_INFO, CLARIFICATION, OTHER
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
        Streaming RAG pipeline.

        Yields dicts:
          {"type": "text",    "content": "<chunk>"}
          {"type": "cards",   "content": [<card_dict>, ...]}
          {"type": "profile", "content": <profile_dict>}
        """
        # Classify intent synchronously for speed
        intent = classify_intent_fast(user_message, session_history)
        print(f"[RAGPipeline] Stream intent (fast): {intent.value}")

        # For ambiguous cases, refine with LLM (non-blocking prefetch)
        if intent in (Intent.OTHER, Intent.CLARIFICATION):
            intent = await classify_intent_llm(
                user_message, session_history, model_id, api_key, model_config
            )
            print(f"[RAGPipeline] Stream intent (LLM): {intent.value}")

        profile = (session.get_profile() if session else {}) or user_profile or {}

        if intent == Intent.SCHEME_REQUEST:
            if session:
                # Run full eligibility pipeline (non-streaming computation)
                reply, cards, updated_profile = await self.handle_scheme_request(
                    user_message, session_history, session, language, model_id, api_key, model_config
                )
                # Stream the text character-by-character for smooth UX
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

        elif intent == Intent.DETAIL_REQUEST:
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

        elif intent == Intent.GREETING:
            greeting = (
                "Hello! 👋 I'm SchemeSathi — I help you find government schemes and scholarships "
                "you're actually eligible for.\n\n"
                "To get started, tell me a bit about yourself:\n"
                "• Which state are you from?\n"
                "• Are you a student, farmer, entrepreneur, or something else?\n"
                "• Any specific type of scheme you're looking for?"
            )
            yield {"type": "text", "content": greeting}
            yield {"type": "cards", "content": []}
            yield {"type": "profile", "content": profile}

        else:
            # PROFILE_INFO / CLARIFICATION / OTHER → collect profile, no cards
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
