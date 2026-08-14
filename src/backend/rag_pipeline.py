"""RAG pipeline for SchemeSathi."""

from backend.embedding_service import embedding_service
from backend.vector_store import vector_store
from backend.scheme_loader import scheme_loader
from backend.gemini_service import gemini_service
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
    'उदవిगाई': 'scholarship stipend',
    'लोन': 'loan credit subsidy interest subvention bank financial',
    'ऋण': 'loan credit subsidy interest subvention bank',
    'कर्ज': 'loan credit subsidy interest subvention',
    'கட൯': 'loan credit subsidy',
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
        self, session_history: list[dict], user_message: str, language: str = 'en'
    ) -> tuple[str, list[dict]]:
        """Full RAG pipeline with multilingual query translation and intent-gated card display."""

        # 1. Translate / Enrich search query to English for high precision vector retrieval
        english_search_query = self._build_english_search_query(session_history, user_message, language)
        print(f"\n[RAG Pipeline] Enriched English Search Query: '{english_search_query}'")

        # 2. Embed English search query
        try:
            query_embedding = await embedding_service.embed_text(english_search_query)
        except Exception as e:
            print(f'Embedding error: {e}')
            return 'Sorry, I could not process your request right now.', []

        # 3. Vector search in ChromaDB (retrieve top 15 schemes)
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

        # 5. Generate response using Gemini in user's requested language
        reply_text = await gemini_service.generate_response(
            session_history, user_message, scheme_context, language=language
        )

        # 6. Intent-gated Card Selection logic
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
                'लिस्ट', 'सूची', 'दिखाएं', 'दीजिए', 'योजना दिखाओ', 'योजना की सूची'
            ]
        )

        ai_recommends_scheme = any(
            kw in reply_text.lower() for kw in [
                'suitable scheme', 'best scheme', 'recommended scheme',
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
