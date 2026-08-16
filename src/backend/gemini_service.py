"""Gemini AI service for SchemeSathi."""

import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from backend.config import GEMINI_API_KEY
import os

# Language display names for the system prompt
LANGUAGE_NAMES = {
    'en': 'English',
    'hi': 'Hindi',
    'ta': 'Tamil',
    'te': 'Telugu',
    'bn': 'Bengali',
    'mr': 'Marathi',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'pa': 'Punjabi',
    'or': 'Odia',
    'as': 'Assamese',
    'ur': 'Urdu',
}

SYSTEM_PROMPT = """You are "SchemeSathi" — a warm, patient, and helpful AI assistant that helps Indian citizens discover government welfare schemes they are eligible for.

You MUST respond in {language_name} language.

## CRITICAL BEHAVIOR RULES

1. **NO RAW URLS OR LINKS IN CHAT TEXT**:
   - NEVER write raw URLs or web links (like https://www.myscheme.gov.in/...) in your message.
   - Interactive Scheme Card Boxes with "View Details" buttons will be displayed automatically below your message.
   - Refer to schemes by their names naturally in your text.

2. **CONVERSATIONAL INTAKE (One question at a time)**:
   - Do NOT dump a list of schemes immediately when conversation starts or when user info is incomplete.
   - Ask ONE relevant follow-up question per message to build their profile (e.g. State, Category/Caste like OBC/SC/ST, Education level, Income).
   - Keep chat text concise, conversational, and warm (2-3 sentences max).

3. **POINTING TO THE MOST USEFUL SCHEME**:
   - When user profile is clear or user asks for schemes/list ("give list", "लिस्ट दीजिए", "show schemes"), highlight the MOST USEFUL scheme(s) naturally in text and ask if they'd like more details.
   - The UI will render the Scheme Card Boxes directly below your text response.

4. **STRICT ACCURACY**:
   - Recommend ONLY schemes present in the provided [SCHEME DATA] context.
"""


class GeminiService:
    """Service for Gemini API interactions via LangChain."""

    def __init__(self):
        self._chain = None
        self.model_name = 'gemini-flash-latest'
        self.fallback_models = ['gemini-flash-lite-latest', 'gemini-2.0-flash']

    def init(self):
        """Initialize the LangChain Gemini chain with fallbacks."""
        api_key = GEMINI_API_KEY or os.getenv('GEMINI_API_KEY', '')
        if not api_key:
            print('WARNING: GEMINI_API_KEY not set!')
            return

        primary = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=api_key,
            temperature=0.6,
            max_output_tokens=1024,
        )
        fallbacks = [
            ChatGoogleGenerativeAI(
                model=m,
                google_api_key=api_key,
                temperature=0.6,
                max_output_tokens=1024,
            )
            for m in self.fallback_models
        ]
        self._chain = primary.with_fallbacks(fallbacks)
        print('Gemini service (LangChain) initialized.')

    async def generate_response(
        self,
        conversation_history: list[dict],
        user_message: str,
        scheme_context: str,
        language: str = 'en',
    ) -> str:
        """Generate a conversational response grounded in scheme data."""
        if not self._chain:
            self.init()
            if not self._chain:
                return 'Error: Gemini API key not configured. Please add GEMINI_API_KEY to your .env file.'

        lang_name = LANGUAGE_NAMES.get(language, 'English')
        system_text = SYSTEM_PROMPT.format(language_name=lang_name)

        # Build LangChain message list: [SystemMessage] + history + augmented user turn
        messages: list = [SystemMessage(content=system_text)]

        for msg in conversation_history:
            if msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['content']))
            else:
                messages.append(AIMessage(content=msg['content']))

        augmented_user = (
            f"{user_message}\n\n"
            f"[SCHEME DATA]\n{scheme_context}\n[END SCHEME DATA]\n\n"
            f"Remember: Respond in {lang_name}. Do NOT write raw URLs. "
            f"Ask 1 follow-up question if profile incomplete."
        )
        messages.append(HumanMessage(content=augmented_user))

        try:
            response = await self._chain.ainvoke(messages)
            text = response.content
        except Exception as e:
            print(f"Gemini API error (all models exhausted): {e}")
            error_msgs = {
                'hi': 'क्षमा करें, फ्री कोटा/रेट लिमिट भर गया है। कृपया कुछ सेकंड बाद फिर से प्रयास करें।',
                'en': "I'm sorry, rate limit exceeded for free tier. Please wait a few seconds and try again.",
            }
            return error_msgs.get(language, error_msgs['en'])

        # Extra safety: strip any raw URL links the model might have generated
        text = re.sub(r'https?://[^\s)]+', '', text)
        text = re.sub(r'\[Link\]\(\)', '', text)
        return text.strip()


gemini_service = GeminiService()
