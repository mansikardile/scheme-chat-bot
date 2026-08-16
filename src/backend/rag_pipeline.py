"""
RAG pipeline for SchemeSathi.

Orchestration layer that wires together:
  1. Custom multilingual query-expansion (domain logic â€” kept as-is)
  2. LangChain-backed OllamaEmbeddings (via embedding_service)
  3. LangChain-backed Chroma vector store (via vector_store)
  4. LangChain-backed ChatGoogleGenerativeAI (via gemini_service)
  5. Custom intent-gated card-selection (domain logic â€” kept as-is)

The custom pre/post-processing steps make a pure LCEL pipe chain awkward,
so the pipeline is implemented as a Python class that invokes LangChain
components in sequence â€” a deliberately pragmatic choice.
"""

from backend.embedding_service import embedding_service
from backend.vector_store import vector_store
from backend.scheme_loader import scheme_loader
from backend.gemini_service import gemini_service
import re


# Regional to English Keyword Mapping for high precision vector retrieval
MULTILINGUAL_KEYWORD_MAP = {
    # Categories & Caste
    'à¤“à¤¬à¥€à¤¸à¥€': 'OBC category Other Backward Class',
    'obc': 'OBC category Other Backward Class',
    'à¤à¤¸à¤¸à¥€': 'SC Scheduled Caste',
    'sc': 'SC Scheduled Caste',
    'à¤à¤¸à¤Ÿà¥€': 'ST Scheduled Tribe',
    'st': 'ST Scheduled Tribe',
    'à¤œà¤¨à¤°à¤²': 'General Category',
    'à¤®à¤¾à¤‡à¤¨à¥‰à¤°à¤¿à¤Ÿà¥€': 'Minority Muslim Christian Sikh Jain Buddhist',
    
    # Occupations & Roles
    'à¤›à¤¾à¤¤à¥à¤°': 'student education scholarship college university school',
    'à¤›à¤¾à¤¤à¥à¤°à¤¾': 'female student girl scholarship',
    'à¤µà¤¿à¤¦à¥à¤¯à¤¾à¤°à¥à¤¥à¥€': 'student education scholarship',
    'à®®à®¾à®£à®µà®°à¯': 'student education scholarship',
    'à°µà°¿à°¦à±à°¯à°¾à°°à±à°¥à°¿': 'student education scholarship',
    'à¤•à¤¿à¤¸à¤¾à¤¨': 'farmer agriculture crop cultivation land loan PM-KISAN',
    'à¤¶à¥‡à¤¤à¤•à¤°à¥€': 'farmer agriculture crop cultivation',
    'à®µà®¿à®µà®šà®¾à®¯à®¿': 'farmer agriculture crop',
    'à°°à±ˆà°¤à±': 'farmer agriculture crop',
    'à¤®à¤¹à¤¿à¤²à¤¾': 'women female entrepreneur widow mother girl child',
    'à¤®à¤¹à¤¿à¤²à¤¾à¤à¤‚': 'women female entrepreneur',
    'à®ªà¯†à®£à¯à®•à®³à¯': 'women female entrepreneur',
    'à°¸à±à°¤à±à°°à±€à°²à±': 'women female entrepreneur',
    'à¤µà¥à¤¯à¤¾à¤ªà¤¾à¤°à¥€': 'business entrepreneur MSME shopkeeper loan vendor',
    'à¤‰à¤¦à¥à¤¯à¤®à¥€': 'entrepreneur business MSME startup loan',
    'à¤¬à¥‡à¤°à¥‹à¤œà¤—à¤¾à¤°': 'unemployed youth employment skill training job',
    'à¤®à¤œà¤¦à¥‚à¤°': 'worker laborer unorganized sector pension',
    'à¤¶à¥à¤°à¤®à¤¿à¤•': 'worker laborer unorganized sector pension',

    # Needs & Services
    'à¤›à¤¾à¤¤à¥à¤°à¤µà¥ƒà¤¤à¥à¤¤à¤¿': 'scholarship stipend financial assistance education fee',
    'à¤¸à¥à¤•à¥‰à¤²à¤°à¤¶à¤¿à¤ª': 'scholarship stipend financial assistance',
    'à¤‰à¤¦à°µà°¿à°—à°¾à¤ˆ': 'scholarship stipend',
    'à¤²à¥‹à¤¨': 'loan credit subsidy interest subvention bank financial',
    'à¤‹à¤£': 'loan credit subsidy interest subvention bank',
    'à¤•à¤°à¥à¤œ': 'loan credit subsidy interest subvention',
    'à®•à®Ÿàµ¯': 'loan credit subsidy',
    'à°°à±à°£à°‚': 'loan credit subsidy',
    'à¤ªà¥‡à¤‚à¤¶à¤¨': 'pension senior citizen elderly old age monthly allowance',
    'à¤“à¤²à¥à¤¡ à¤à¤œ': 'old age senior citizen pension',
    'à¤†à¤µà¤¾à¤¸': 'housing house construction Pradhan Mantri Awas Yojana PMAY shelter',
    'à¤˜à¤°': 'housing house construction PMAY',
    'à¤¸à¥à¤µà¤¾à¤¸à¥à¤¥à¥à¤¯': 'health healthcare hospital medical insurance Ayushman Bharat',
    'à¤‡à¤²à¤¾à¤œ': 'health healthcare medical insurance treatment',
    'à¤¬à¥€à¤®à¤¾': 'insurance life accidental health policy cover',

    # Common States
    'à¤…à¤¸à¤®': 'Assam',
    'à¤®à¤¹à¤¾à¤°à¤¾à¤·à¥à¤Ÿà¥à¤°': 'Maharashtra',
    'à¤‰à¤¤à¥à¤¤à¤° à¤ªà¥à¤°à¤¦à¥‡à¤¶': 'Uttar Pradesh UP',
    'à¤¬à¤¿à¤¹à¤¾à¤°': 'Bihar',
    'à¤°à¤¾à¤œà¤¸à¥à¤¥à¤¾à¤¨': 'Rajasthan',
    'à¤®à¤§à¥à¤¯ à¤ªà¥à¤°à¤¦à¥‡à¤¶': 'Madhya Pradesh MP',
    'à¤—à¥à¤œà¤°à¤¾à¤¤': 'Gujarat',
    'à¤ªà¤‚à¤œà¤¾à¤¬': 'Punjab',
    'à¤¹à¤°à¤¿à¤¯à¤¾à¤£à¤¾': 'Haryana',
    'à¤¤à¤®à¤¿à¤²à¤¨à¤¾à¤¡à¥': 'Tamil Nadu',
    'à¤•à¤°à¥à¤¨à¤¾à¤Ÿà¤•': 'Karnataka',
    'à¤¤à¥‡à¤²à¤‚à¤—à¤¾à¤¨à¤¾': 'Telangana',
    'à¤†à¤‚à¤§à¥à¤° à¤ªà¥à¤°à¤¦à¥‡à¤¶': 'Andhra Pradesh',
    'à¤ªà¤¶à¥à¤šà¤¿à¤® à¤¬à¤‚à¤—à¤¾à¤²': 'West Bengal',
    'à¤“à¤¡à¤¿à¤¶à¤¾': 'Odisha',
}


class RAGPipeline:
    """Orchestrates query translation, vector search, LLM response, and UI card rendering."""

    async def process_query(
        self, session_history: list[dict], user_message: str, language: str = 'en'
    ) -> tuple[str, list[dict]]:
        """Full RAG pipeline with multilingual query translation and intent-gated card display."""

        # 1. Translate / Enrich search query to English for high precision vector retrieval
        english_search_query = self._build_english_search_query(session_history, user_message, language)
        print(f"\n[RAG Pipeline] Enriched English Search Query: '{english_search_query}'")

        # 2. Embed English search query (LangChain OllamaEmbeddings, async via thread)
        try:
            query_embedding = await embedding_service.embed_text(english_search_query)
        except Exception as e:
            print(f'Embedding error: {e}')
            return 'Sorry, I could not process your request right now.', []

        # 3. Vector search in ChromaDB via LangChain-backed store (retrieve top 15 schemes)
        results = vector_store.search(query_embedding, top_k=15)
        retrieved_slugs = [r['slug'] for r in results]
        print(f"[RAG Pipeline] ChromaDB Top 15 retrieved scheme slugs: {retrieved_slugs}")

        # 4. Build rich scheme context for Gemini
        context_parts = []
        for i, res in enumerate(results, 1):
            ctx = scheme_loader.get_scheme_context(res['slug'])
            context_parts.append(f'--- Scheme {i} ---\n{ctx}')
        scheme_context = '\n\n'.join(context_parts)
        print(f"[RAG Pipeline] Injected Context Length: {len(scheme_context)} characters\n")

        # 5. Generate response using LangChain ChatGoogleGenerativeAI in user's requested language
        reply_text = await gemini_service.generate_response(
            session_history, user_message, scheme_context, language=language
        )

        # 6. Intent-gated Card Selection logic (domain logic â€” kept as custom post-processing)
        scheme_cards = self._select_cards_if_needed(reply_text, user_message, results)

        return reply_text, scheme_cards

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

        # 1. Check if AI mentions scheme names / titles explicitly in text response
        mentioned_slugs = []
        for res in results:
            slug = res['slug']
            idx = scheme_loader.slug_to_index.get(slug, {})
            name = idx.get('schemeName', '')
            if slug in reply_text or (name and len(name) > 6 and name.lower() in reply_text.lower()):
                mentioned_slugs.append(slug)

        if mentioned_slugs:
            return self._build_scheme_cards_for_slugs(mentioned_slugs[:4])

        # 2. Check if user explicitly asked for a list or scheme recommendations
        user_explicitly_requested_list = any(
            kw in user_message.lower() for kw in [
                'list', 'show list', 'give me list', 'show schemes', 'all schemes',
                'yojana list', 'yojna list', 'recommend schemes', 'which scheme',
                'à¤²à¤¿à¤¸à¥à¤Ÿ', 'à¤¸à¥‚à¤šà¥€', 'à¤¦à¤¿à¤–à¤¾à¤à¤‚', 'à¤¦à¥€à¤œà¤¿à¤', 'à¤¯à¥‹à¤œà¤¨à¤¾ à¤¦à¤¿à¤–à¤¾à¤“', 'à¤¯à¥‹à¤œà¤¨à¤¾ à¤•à¥€ à¤¸à¥‚à¤šà¥€'
            ]
        )

        ai_recommends_scheme = any(
            kw in reply_text.lower() for kw in [
                'suitable scheme', 'best scheme', 'recommended scheme',
                'à¤‰à¤ªà¤¯à¥à¤•à¥à¤¤ à¤¯à¥‹à¤œà¤¨à¤¾', 'à¤ªà¥à¤°à¤®à¥à¤– à¤¸à¤°à¤•à¤¾à¤°à¥€ à¤¯à¥‹à¤œà¤¨à¤¾', 'à¤¨à¥€à¤šà¥‡ à¤¦à¥€ à¤—à¤ˆ à¤¯à¥‹à¤œà¤¨à¤¾', 'à¤¨à¤¿à¤®à¥à¤¨à¤²à¤¿à¤–à¤¿à¤¤ à¤¯à¥‹à¤œà¤¨à¤¾'
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
