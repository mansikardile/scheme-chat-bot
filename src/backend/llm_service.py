"""Provider-agnostic LLM service for SchemeSathi."""

import re
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from backend.model_registry import get_model

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


def _extract_text_content(content) -> str:
    """Extract clean string text from LangChain message content (str, list, or dict)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get('type') == 'text' and 'text' in item:
                    parts.append(str(item['text']))
                elif 'text' in item:
                    parts.append(str(item['text']))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(content, dict):
        if content.get('type') == 'text' and 'text' in content:
            return str(content['text'])
        if 'text' in content:
            return str(content['text'])
    return str(content)


async def generate_response(
    model_id: str,
    conversation_history: list[dict],
    user_message: str,
    scheme_context: str,
    language: str = 'en',
    api_key: str | None = None,
    model_config: dict | None = None,
) -> str:
    """Generate a conversational response grounded in scheme data using a model from model_registry."""
    print(f"\n[LLM Service] Processing chat request using model_id: '{model_id}'")
    try:
        model = get_model(model_id, api_key=api_key, model_config=model_config)
    except Exception as e:
        print(f"Error instantiating model '{model_id}': {e}")
        err = str(e)
        if "GEMINI_API_KEY" in err:
            return "🔑 **Gemini API Key Required**: Please click the **⚙️ Settings** icon in the top right corner of the page to enter your Gemini API Key, or add `GEMINI_API_KEY` to your `.env` file."
        return f'⚠️ Model configuration error: {err}'

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
        response = await model.ainvoke(messages)
        text = _extract_text_content(response.content)
    except Exception as e:
        print(f"LLM execution error with model '{model_id}': {e}")
        error_msgs = {
            'hi': 'क्षमा करें, अनुरोध प्रोसेस करने में त्रुटि हुई। कृपया सेटिंग्स या मॉडल कॉन्फ़िगरेशन जांचें।',
            'en': f"Error generating response from model ({str(e)}). Please check your model settings.",
        }
        return error_msgs.get(language, error_msgs['en'])

    # Extra safety: strip any raw URL links the model might have generated
    text = re.sub(r'https?://[^\s)]+', '', text)
    text = re.sub(r'\[Link\]\(\)', '', text)
    return text.strip()


async def generate_response_stream(
    model_id: str,
    conversation_history: list[dict],
    user_message: str,
    scheme_context: str,
    language: str = 'en',
    api_key: str | None = None,
    model_config: dict | None = None,
):
    """Generate a conversational response stream grounded in scheme data using a model from model_registry."""
    print(f"\n[LLM Service] Processing streaming chat request using model_id: '{model_id}'")
    try:
        model = get_model(model_id, api_key=api_key, model_config=model_config)
    except Exception as e:
        print(f"Error instantiating model '{model_id}': {e}")
        err = str(e)
        if "GEMINI_API_KEY" in err:
            yield "🔑 **Gemini API Key Required**: Please click the **⚙️ Settings** icon in the top right corner to enter your Gemini API Key, or add `GEMINI_API_KEY` to your `.env` file."
        else:
            yield f'⚠️ Model configuration error: {err}'
        return

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
        async for chunk in model.astream(messages):
            text = _extract_text_content(chunk.content)
            if text:
                yield text
    except Exception as e:
        print(f"LLM execution error with model '{model_id}': {e}")
        error_msgs = {
            'hi': 'क्षमा करें, अनुरोध प्रोसेस करने में त्रुटि हुई। कृपया सेटिंग्स या मॉडल कॉन्फ़िगरेशन जांचें।',
            'en': f"Error generating response from model ({str(e)}). Please check your model settings.",
        }
        yield error_msgs.get(language, error_msgs['en'])

