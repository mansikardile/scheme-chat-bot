"""
RAG pipeline for SchemeSathi.

Orchestration layer that wires together:
  1. Custom multilingual query-expansion (domain logic — kept as-is)
  2. LangChain-backed OllamaEmbeddings (via embedding_service)
  3. LangChain-backed Chroma vector store (via vector_store)
  4. Provider-agnostic LLM service (via llm_service)
  5. Custom intent-gated card-selection (domain logic — kept as-is)
"""

from backend.embedding_service import embedding_service
from backend.vector_store import vector_store
from backend.scheme_loader import scheme_loader
from backend.llm_service import generate_response, generate_response_stream, extract_filters, classify_detail_query
import re


# Regional to English Keyword Mapping for high precision vector retrieval
MULTILINGUAL_KEYWORD_MAP = {
    # Categories & Caste
    'ओबीसी': 'OBC category Other Backward Class',
    'obc': 'OBC category Other Backward Class',
    'एससी': 'SC Scheduled Caste',
    'sc': 'SC Scheduled Caste',
    'एसटी': 'ST Scheduled Tribe',
    'st': 'ST Scheduled Tribe',
    'जनरल': 'General Category',
    'माइनॉरिटी': 'Minority Muslim Christian Sikh Jain Buddhist',
    
    # Occupations & Roles
    'छात्र': 'student education scholarship college university school',
    'छात्रा': 'female student girl scholarship',
    'विद्यार्थी': 'student education scholarship',
    'மாணவர்': 'student education scholarship',
    'విద్యార్థి': 'student education scholarship',
    'किसान': 'farmer agriculture crop cultivation land loan PM-KISAN',
    'शेतकरी': 'farmer agriculture crop cultivation',
    'விவசாயி': 'farmer agriculture crop',
    'రైతు': 'farmer agriculture crop',
    'महिला': 'women female entrepreneur widow mother girl child',
    'महिलाएं': 'women female entrepreneur',
    'பெண்கள்': 'women female entrepreneur',
    'స్త్రీలు': 'women female entrepreneur',
    'व्यापारी': 'business entrepreneur MSME shopkeeper loan vendor',
    'उद्यमी': 'entrepreneur business MSME startup loan',
    'बेरोजगार': 'unemployed youth employment skill training job',
    'मजदूर': 'worker laborer unorganized sector pension',
    'श्रमिक': 'worker laborer unorganized sector pension',

    # Needs & Services
    'छात्रवृत्ति': 'scholarship stipend financial assistance education fee',
    'स्कॉलरशिप': 'scholarship stipend financial assistance',
    'உதவித்தொகை': 'scholarship stipend',
    'लोन': 'loan credit subsidy interest subvention bank financial',
    'ऋण': 'loan credit subsidy interest subvention bank',
    'कर्ज': 'loan credit subsidy interest subvention',
    'கடன்': 'loan credit subsidy',
    'రుణం': 'loan credit subsidy',
    'पेंशन': 'pension senior citizen elderly old age monthly allowance',
    'ओल्ड एज': 'old age senior citizen pension',
    'आवास': 'housing house construction Pradhan Mantri Awas Yojana PMAY shelter',
    'घर': 'housing house construction PMAY',
    'स्वास्थ्य': 'health healthcare hospital medical insurance Ayushman Bharat',
    'इलाज': 'health healthcare medical insurance treatment',
    'बीमा': 'insurance life accidental health policy cover',

    # Common States
    'असम': 'Assam',
    'महाराष्ट्र': 'Maharashtra',
    'उत्तर प्रदेश': 'Uttar Pradesh UP',
    'बिहार': 'Bihar',
    'राजस्थान': 'Rajasthan',
    'मध्य प्रदेश': 'Madhya Pradesh MP',
    'गुजरात': 'Gujarat',
    'पंजाब': 'Punjab',
    'हरियाणा': 'Haryana',
    'तमिलनाडु': 'Tamil Nadu',
    'कर्नाटक': 'Karnataka',
    'तेलंगाना': 'Telangana',
    'आंध्र प्रदेश': 'Andhra Pradesh',
    'पश्चिम बंगाल': 'West Bengal',
    'ओडिशा': 'Odisha',
}


class RAGPipeline:
    """Orchestrates query translation, vector search, LLM response, and UI card rendering."""

    async def process_query(
        self,
        session_history: list[dict],
        user_message: str,
        language: str = 'en',
        model_id: str = 'gemini-flash',
        api_key: str | None = None,
        model_config: dict | None = None,
    ) -> tuple[str, list[dict]]:
        """Full RAG pipeline: filter extraction → SQL retrieval → compact context → LLM response."""

        # 1. Extract structured filters from user message + history via a fast LLM call
        filters = {}
        try:
            filters = await extract_filters(
                model_id, user_message, session_history,
                api_key=api_key, model_config=model_config
            )
        except Exception as e:
            print(f"[RAG Pipeline] Filter extraction failed ({e}), proceeding without filters")

        # 2. Targeted DuckDB SQL search using extracted filters
        results = []
        has_filters = any(filters.get(k) for k in ('state', 'category', 'keywords', 'level'))

        if has_filters:
            matched = scheme_loader.search_with_filters(filters, limit=8)
            results = [{'slug': s['slug']} for s in matched]

        # 3. Fallback chain: vector search → DuckDB BM25 keyword search
        if not results:
            english_search_query = self._build_english_search_query(session_history, user_message, language)
            print(f"[RAG Pipeline] No filter results — trying vector search: '{english_search_query}'")
            try:
                if embedding_service._lc_embeddings is not None:
                    query_embedding = await embedding_service.embed_text(english_search_query)
                    results = vector_store.search(query_embedding, top_k=8)
            except Exception as e:
                print(f"[RAG Pipeline] Vector search failed ({e}), falling back to BM25")

        if not results:
            print("[RAG Pipeline] Using DuckDB BM25 keyword fallback")
            db_results = scheme_loader.search_schemes(query=user_message, limit=8)
            results = [{'slug': s['slug']} for s in db_results]

        retrieved_slugs = [r['slug'] for r in results]
        print(f"[RAG Pipeline] Retrieved {len(results)} scheme(s): {retrieved_slugs}")

        # 4. Classify: is the user asking a specific factual question about one scheme?
        #    Pass the actual retrieved scheme names so implicit references ("it", "this scheme")
        #    can be resolved reliably from the list.
        scheme_context = await self._build_context(
            results, user_message, session_history, model_id, api_key, model_config
        )

        # 5. Generate response using provider-agnostic llm_service
        reply_text = await generate_response(
            model_id, session_history, user_message, scheme_context,
            language=language, api_key=api_key, model_config=model_config
        )

        # 6. Intent-gated Card Selection logic (domain logic — kept as custom post-processing)
        scheme_cards = self._select_cards_if_needed(reply_text, user_message, results)

        return reply_text, scheme_cards


    async def process_query_stream(
        self,
        session_history: list[dict],
        user_message: str,
        language: str = 'en',
        model_id: str = 'gemini-flash',
        api_key: str | None = None,
        model_config: dict | None = None,
    ):
        """Full RAG pipeline streaming LLM response: filter extraction → SQL retrieval → compact context."""

        # 1. Extract structured filters from user message + history via a fast LLM call
        filters = {}
        try:
            filters = await extract_filters(
                model_id, user_message, session_history,
                api_key=api_key, model_config=model_config
            )
        except Exception as e:
            print(f"[RAG Pipeline] Filter extraction failed ({e}), proceeding without filters")

        # 2. Targeted DuckDB SQL search using extracted filters
        results = []
        has_filters = any(filters.get(k) for k in ('state', 'category', 'keywords', 'level'))

        if has_filters:
            matched = scheme_loader.search_with_filters(filters, limit=8)
            results = [{'slug': s['slug']} for s in matched]

        # 3. Fallback chain: vector search → DuckDB BM25 keyword search
        if not results:
            english_search_query = self._build_english_search_query(session_history, user_message, language)
            print(f"[RAG Pipeline] No filter results — trying vector search (Stream): '{english_search_query}'")
            try:
                if embedding_service._lc_embeddings is not None:
                    query_embedding = await embedding_service.embed_text(english_search_query)
                    results = vector_store.search(query_embedding, top_k=8)
            except Exception as e:
                print(f"[RAG Pipeline] Vector search failed ({e}), falling back to BM25")

        if not results:
            print("[RAG Pipeline] Using DuckDB BM25 keyword fallback (Stream)")
            db_results = scheme_loader.search_schemes(query=user_message, limit=8)
            results = [{'slug': s['slug']} for s in db_results]

        retrieved_slugs = [r['slug'] for r in results]
        print(f"[RAG Pipeline] Retrieved {len(results)} scheme(s) (Stream): {retrieved_slugs}")

        # 4. Classify: detail request or broad query? Build context accordingly.
        scheme_context = await self._build_context(
            results, user_message, session_history, model_id, api_key, model_config
        )

        # 5. Stream response using provider-agnostic llm_service
        full_reply_text = ""
        async for chunk in generate_response_stream(
            model_id, session_history, user_message, scheme_context,
            language=language, api_key=api_key, model_config=model_config
        ):
            clean_chunk = re.sub(r'https?://[^\s)]+', '', chunk)
            clean_chunk = re.sub(r'\[Link\]\(\)', '', clean_chunk)
            if clean_chunk:
                full_reply_text += clean_chunk
                yield {"type": "text", "content": clean_chunk}

        # 6. Intent-gated Card Selection logic based on the full generated response
        scheme_cards = self._select_cards_if_needed(full_reply_text, user_message, results)
        yield {"type": "cards", "content": scheme_cards}


    async def _build_context(
        self,
        results: list[dict],
        user_message: str,
        session_history: list[dict],
        model_id: str,
        api_key: str | None,
        model_config: dict | None,
    ) -> str:
        """Build the scheme context string to inject into the LLM prompt.

        Runs classify_detail_query() with the retrieved scheme names, then routes to one
        of two context strategies:

        - Detail path: user is asking a factual question about a specific named scheme.
          Injects get_scheme_context() (full) for that scheme + compact summaries for
          up to 3 others.  Total: ~3-6 KB.

        - Broad path (default): user is discovering / listing schemes.
          Injects get_scheme_summary_context() (compact) for all results.
          Total: ~2-4 KB.

        Both are far smaller than the old 15-full-context approach (~45 KB).
        """
        # Collect names from results for the classifier (must exist in slug_to_index)
        retrieved_names = []
        for r in results:
            name = scheme_loader.slug_to_index.get(r['slug'], {}).get('schemeName', '')
            if name:
                retrieved_names.append(name)

        # Classify
        classification = await classify_detail_query(
            model_id, user_message, session_history, retrieved_names,
            api_key=api_key, model_config=model_config
        )

        context_parts = []

        if classification.get('detail_request') and classification.get('scheme_name'):
            target_name = classification['scheme_name']
            target_slug = scheme_loader.find_scheme_slug_by_name(target_name)

            if target_slug:
                # Full detail for the target scheme
                full_ctx = scheme_loader.get_scheme_context(target_slug)
                context_parts.append(f'--- Scheme 1 (Full Detail) ---\n{full_ctx}')
                print(f"[RAG Pipeline] Detail path: full context for '{target_slug}'")

                # Compact summaries for the remaining results (up to 3)
                others = [r for r in results if r['slug'] != target_slug][:3]
                for i, res in enumerate(others, 2):
                    ctx = scheme_loader.get_scheme_summary_context(res['slug'])
                    context_parts.append(f'--- Scheme {i} (Summary) ---\n{ctx}')
            else:
                # Slug not found — fall through to broad path
                print(f"[RAG Pipeline] Detail path: could not resolve slug for '{target_name}', "
                      "falling back to summaries")
                for i, res in enumerate(results, 1):
                    ctx = scheme_loader.get_scheme_summary_context(res['slug'])
                    context_parts.append(f'--- Scheme {i} ---\n{ctx}')
        else:
            # Broad path — compact summaries for all results
            for i, res in enumerate(results, 1):
                ctx = scheme_loader.get_scheme_summary_context(res['slug'])
                context_parts.append(f'--- Scheme {i} ---\n{ctx}')

        scheme_context = '\n\n'.join(context_parts)
        print(f"[RAG Pipeline] Injected Context Length: {len(scheme_context)} characters "
              f"(~{len(scheme_context) // 4} tokens)\n")
        return scheme_context

    def _build_english_search_query(self, history: list[dict], current_message: str, language: str) -> str:
        """Translates regional terms into English concepts for ChromaDB search."""
        context_parts = []
        recent = history[-4:] if len(history) > 4 else history
        for msg in recent:
            if msg['role'] == 'user':
                context_parts.append(msg['content'])
        context_parts.append(current_message)
        raw_query = ' '.join(context_parts)

        mapped_concepts = []
        raw_lower = raw_query.lower()
        for term, en_concept in MULTILINGUAL_KEYWORD_MAP.items():
            if term in raw_lower or term in raw_query:
                mapped_concepts.append(en_concept)

        english_search_text = raw_query
        if mapped_concepts:
            english_search_text += " " + " ".join(mapped_concepts)

        return english_search_text

    def _select_cards_if_needed(
        self, reply_text: str, user_message: str, results: list[dict]
    ) -> list[dict]:
        """Return UI Card Boxes ONLY when a relevant scheme is recommended or explicitly requested."""
        if not results:
            return []

        reply_lower = reply_text.lower()
        user_lower = user_message.lower()

        # 1. Check if AI mentions scheme names / titles in text response.
        #    Uses bidirectional matching:
        #    (a) Full DB name is a substring of the reply (exact)
        #    (b) Slug appears in reply
        #    (c) 2+ significant words (>5 chars) from the scheme name appear in the reply
        #        — catches abbreviated names like "Weaver MUDRA Scheme" vs full DB name
        mentioned_slugs = []
        for res in results:
            slug = res['slug']
            idx = scheme_loader.slug_to_index.get(slug, {})
            name = idx.get('schemeName', '')
            if not name:
                continue

            # (a) exact slug or full name match
            if slug in reply_text or name.lower() in reply_lower:
                mentioned_slugs.append(slug)
                continue

            # (b) keyword overlap: count significant words from scheme name in reply
            if len(name) > 6:
                sig_words = [w for w in name.split() if len(w) > 5 and w.isalpha()]
                if sig_words:
                    matches = sum(1 for w in sig_words if w.lower() in reply_lower)
                    # Require at least 3 distinctive word matches to avoid false positives
                    # (2 was too loose — common words like "education"+"scholarship" caused
                    # premature cards before the LLM had even asked for the user's state)
                    if matches >= 3:
                        mentioned_slugs.append(slug)

        if mentioned_slugs:
            return self._build_scheme_cards_for_slugs(mentioned_slugs[:4])

        # 2. Check if user explicitly asked for schemes / a list
        user_explicitly_requested_list = any(
            kw in user_lower for kw in [
                'list', 'show list', 'give me list', 'show schemes', 'all schemes',
                'what schemes', 'which schemes', 'schemes for me', 'schemes do you',
                'yojana list', 'yojna list', 'recommend schemes', 'which scheme',
                'schemes available', 'available schemes', 'tell me schemes',
                'लिस्ट', 'सूची', 'दिखाएं', 'दीजिए', 'योजना दिखाओ', 'योजना की सूची'
            ]
        )

        ai_recommends_scheme = any(
            kw in reply_lower for kw in [
                'suitable scheme', 'best scheme', 'recommended scheme',
                'schemes available', 'following scheme', 'several scheme',
                'wonderful scheme', 'great scheme', 'support program',
                'उपयुक्त योजना', 'प्रमुख सरकारी योजना', 'नीचे दी गई योजना', 'निम्नलिखित योजना'
            ]
        )

        if user_explicitly_requested_list or ai_recommends_scheme:
            top_slugs = [r['slug'] for r in results[:4]]
            return self._build_scheme_cards_for_slugs(top_slugs)

        # 3. Otherwise (during intake Q&A like "I am a student" or "OBC"), DO NOT display card boxes yet
        return []


    def _build_scheme_cards_for_slugs(self, slugs: list[str]) -> list[str]:
        cards = []
        seen = set()
        for slug in slugs:
            if slug in seen:
                continue
            seen.add(slug)
            idx = scheme_loader.slug_to_index.get(slug, {})
            if not idx:
                continue
            brief = idx.get('briefDescription', '')
            cards.append({
                'slug': slug,
                'name': idx.get('schemeName', 'Unknown'),
                'brief': brief[:200] + ('...' if len(brief) > 200 else ''),
                'level': idx.get('level', ''),
                'states': idx.get('beneficiaryState', []),
                'categories': idx.get('schemeCategory', []),
                'tags': idx.get('tags', []),
                'has_details': True,
            })
        return cards


rag_pipeline = RAGPipeline()
