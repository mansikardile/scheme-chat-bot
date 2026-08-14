"""Gemini AI service for SchemeSathi."""

from google import genai
from google.genai import types
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
    """Service for Gemini API interactions."""

    def __init__(self):
        self.client = None
        self.model_name = 'gemini-flash-latest'
        self.fallback_models = ['gemini-flash-lite-latest', 'gemini-2.0-flash']

    def init(self):
        """Initialize the Gemini client."""
        api_key = GEMINI_API_KEY or os.getenv('GEMINI_API_KEY', '')
        if not api_key:
            print('WARNING: GEMINI_API_KEY not set!')
            return
        self.client = genai.Client(api_key=api_key)
        print('Gemini service initialized.')

    async def generate_response(
        self,
        conversation_history: list[dict],
        user_message: str,
        scheme_context: str,
        language: str = 'en',
    ) -> str:
        """Generate a conversational response grounded in scheme data."""
        if not self.client:
            self.init()
            if not self.client:
                return 'Error: Gemini API key not configured. Please add GEMINI_API_KEY to your .env file.'

        lang_name = LANGUAGE_NAMES.get(language, 'English')
        system = SYSTEM_PROMPT.format(language_name=lang_name)

        history = []
        for msg in conversation_history:
            role = 'user' if msg['role'] == 'user' else 'model'
            history.append(
                types.Content(role=role, parts=[types.Part.from_text(text=msg['content'])])
            )

        augmented = (
            f"{user_message}\n\n"
            f"[SCHEME DATA]\n{scheme_context}\n[END SCHEME DATA]\n\n"
            f"Remember: Respond in {lang_name}. Do NOT write raw URLs. Ask 1 follow-up question if profile incomplete."
        )

        models_to_try = [self.model_name] + self.fallback_models
        for m_name in models_to_try:
            try:
                chat = self.client.chats.create(
                    model=m_name,
                    history=history,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=0.6,
                        max_output_tokens=1024,
                    ),
                )
                response = chat.send_message(augmented)
                text = response.text
                # Extra safety: strip any raw URL links Gemini might have generated
                import re
                text = re.sub(r'https?://[^\s)]+', '', text)
                text = re.sub(r'\[Link\]\(\)', '', text)
                return text.strip()
            except Exception as e:
                print(f"Gemini API error with model {m_name}: {e}")
                continue

        error_msgs = {
            'hi': 'क्षमा करें, फ्री कोटा/रेट लिमिट भर गया है। कृपया कुछ सेकंड बाद फिर से प्रयास करें।',
            'en': "I'm sorry, rate limit exceeded for free tier. Please wait a few seconds and try again.",
        }
        return error_msgs.get(language, error_msgs['en'])


gemini_service = GeminiService()
