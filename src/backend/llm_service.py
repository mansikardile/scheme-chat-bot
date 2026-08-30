"""Provider-agnostic LLM service for SchemeSathi."""

import re
import json
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


# ---------------------------------------------------------------------------
# Filter extraction — used as a fast pre-pass to build structured SQL queries
# ---------------------------------------------------------------------------

FILTER_EXTRACTION_PROMPT = """Extract search filters from a user query about Indian government welfare schemes.
Return ONLY a JSON object. Use null for anything not mentioned or unclear.

Fields:
- "state": Indian state name (string, e.g. "Maharashtra", "Bihar"), or null
- "category": beneficiary group — one of "SC", "OBC", "ST", "General", "Women", "Farmer",
  "Student", "Minority", "Disabled", "Senior Citizen", "BPL", "Youth", "Entrepreneur", or null
- "keywords": 2-4 space-separated English keywords capturing the core need (e.g. "education scholarship children"), or null
- "level": "Central" or "State" or null

Recent conversation (user messages only): {context}
Current user message: {user_message}

JSON only, no explanation, no markdown fences:"""


async def extract_filters(
    model_id: str,
    user_message: str,
    conversation_history: list[dict],
    api_key: str | None = None,
    model_config: dict | None = None,
) -> dict:
    """Use the LLM to extract structured search filters from the user's message + history.

    Returns a dict with optional keys: state, category, keywords, level.
    Returns an empty dict on any failure so the caller can gracefully fall back.
    """
    recent_user_msgs = [m['content'] for m in conversation_history[-4:] if m['role'] == 'user']
    context = " | ".join(recent_user_msgs) if recent_user_msgs else "None"

    prompt = FILTER_EXTRACTION_PROMPT.format(context=context, user_message=user_message)

    try:
        model = get_model(model_id, api_key=api_key, model_config=model_config)
        response = await model.ainvoke([HumanMessage(content=prompt)])
        text = _extract_text_content(response.content).strip()
        # Models sometimes wrap output in ```json ... ``` fences — strip them
        json_match = re.search(r'\{.*?\}', text, re.DOTALL)
        if json_match:
            filters = json.loads(json_match.group())
            # Normalise: drop keys with null / empty values
            filters = {k: v for k, v in filters.items() if v}
            print(f"[Filter Extraction] Extracted: {filters}")
            return filters
    except Exception as e:
        print(f"[Filter Extraction] Failed ({e}), pipeline will use keyword fallback")
    return {}


# ---------------------------------------------------------------------------
# Detail query classification — runs AFTER retrieval, with scheme names in hand
# ---------------------------------------------------------------------------

DETAIL_CLASSIFICATION_PROMPT = """You are deciding whether a user is asking a specific factual question about one particular government scheme.

Schemes available in this conversation:
{scheme_list}

Recent conversation (user messages only): {context}
Current user message: {user_message}

Is the user asking a specific factual/detail question (eligibility details, what it covers, whether it allows X, benefits, how to apply, limits, exclusions) about ONE of the schemes listed above?

- Detail question examples: "can it be used for travel?", "what are the income limits?", "is furniture allowed?", "how do I apply for the Research Grant?"
- NOT a detail question: "list schemes for SC", "show me scholarships in UP", "what schemes exist for farmers?"

Return ONLY JSON, no explanation:
{{"detail_request": true or false, "scheme_name": "<exact name from the list above, or null>"}}"""


async def classify_detail_query(
    model_id: str,
    user_message: str,
    conversation_history: list[dict],
    retrieved_scheme_names: list[str],
    api_key: str | None = None,
    model_config: dict | None = None,
) -> dict:
    """Classify whether the user is asking a factual detail question about a specific scheme.

    Called AFTER retrieval so we can pass the actual scheme names from the results,
    making implicit reference resolution (e.g. "can IT be used for travel?") reliable.

    Args:
        retrieved_scheme_names: List of scheme names from the retrieval results.

    Returns:
        Dict with keys:
            - detail_request (bool): True if a specific factual question about one scheme.
            - scheme_name (str | None): Exact name from retrieved_scheme_names, or None.
        Returns {"detail_request": False, "scheme_name": None} on any failure.
    """
    default = {"detail_request": False, "scheme_name": None}
    if not retrieved_scheme_names:
        return default

    recent_user_msgs = [m['content'] for m in conversation_history[-4:] if m['role'] == 'user']
    context = " | ".join(recent_user_msgs) if recent_user_msgs else "None"

    numbered_list = "\n".join(f"{i+1}. {name}" for i, name in enumerate(retrieved_scheme_names))
    prompt = DETAIL_CLASSIFICATION_PROMPT.format(
        scheme_list=numbered_list,
        context=context,
        user_message=user_message,
    )

    try:
        model = get_model(model_id, api_key=api_key, model_config=model_config)
        response = await model.ainvoke([HumanMessage(content=prompt)])
        text = _extract_text_content(response.content).strip()
        json_match = re.search(r'\{.*?\}', text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            # Validate: scheme_name must be from our list (prevent hallucinated names)
            scheme_name = result.get("scheme_name")
            if scheme_name and scheme_name not in retrieved_scheme_names:
                # Try a case-insensitive match as fallback
                lower_map = {n.lower(): n for n in retrieved_scheme_names}
                scheme_name = lower_map.get(scheme_name.lower())
            detail = bool(result.get("detail_request")) and scheme_name is not None
            print(f"[Detail Classifier] detail_request={detail}, scheme_name={scheme_name!r}")
            return {"detail_request": detail, "scheme_name": scheme_name}
    except Exception as e:
        print(f"[Detail Classifier] Failed ({e}), defaulting to summary context")
    return default


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

