"""
Intent classifier for SchemeSathi.

Classifies each incoming user message into one of several intents,
which determines what action the pipeline takes next.
"""

from __future__ import annotations

import re
from enum import Enum


class Intent(str, Enum):
    PROFILE_INFO   = "PROFILE_INFO"    # User is sharing personal details
    SCHEME_REQUEST = "SCHEME_REQUEST"  # User explicitly asks for scheme recommendations
    DETAIL_REQUEST = "DETAIL_REQUEST"  # User asks for details about a specific scheme
    GREETING       = "GREETING"        # Hi, hello, etc.
    CLARIFICATION  = "CLARIFICATION"   # User is answering a follow-up question
    OTHER          = "OTHER"           # Unrelated or unclear


# ---------------------------------------------------------------------------
# Fast keyword-based classification (no LLM)
# ---------------------------------------------------------------------------

_SCHEME_REQUEST_PATTERNS = [
    r'\b(?:s+h+o+w+|sow|sho|find|display|list|give|get|recommend|suggest)\s*(?:me\s+)?(?:the\s+)?(?:all\s+)?(?:schemes?|scheems?|skemes?|scemes?|shemes?|yojanas?|yojnas?|scholarships?|results?)\b',
    r'\b(?:s+h+o+w+|sow|sho)\s+(?:all\s+|my\s+)?(?:schemes?|shemes?|skemes?|scemes?|results?|scholarships?)\b',
    r'\bgive\s*(?:me\s+)?(?:the\s+)?(?:schemes?|scheems?|skemes?)\b',
    r'\blist\s*(?:of\s+)?(?:schemes?|scheems?|skemes?)\b',
    r'\bfind\s*(?:me\s+)?(?:schemes?|scheems?|skemes?)\b',
    r'\bsearch\s*(?:for\s+)?(?:schemes?|scheems?|skemes?)\b',
    r'\bwhat\s+(?:schemes?|shemes?|skemes?)\b',
    r'\bwhich\s+(?:schemes?|shemes?|skemes?)\b',
    r'\b(?:schemes?|shemes?|skemes?)\s+(?:available|for\s+me|do\s+you\s+have|i\s+(?:can|could)\s+get)\b',
    r'\beligible\s+(?:for\s+)?(?:which|what|any)\b',
    r'\bam\s+i\s+eligible\b',
    r'\bscholars?hips?\s+(?:for\s+me|available|i\s+can\s+get)\b',
    r'\byojna(?:s)?\s+(?:list|show|give|find)\b',
    r'\bलिस्ट\s+(?:दो|दीजिए|दिखाओ)\b',
    r'\bयोजना\s+(?:दिखाओ|बताओ|दीजिए)\b',
    r'\bसूची\b',
    r'\bwhat\s+(?:government\s+)?(?:benefits?|support|assistance|help)\s+(?:can|do)\s+i\s+get\b',
    r'\bgive\s+(?:me\s+)?(?:a\s+)?list\b',
    r'\ball\s+(?:matching\s+)?(?:schemes?|scheems?|skemes?)\b',
    r'\bshow\s+results?\b',
]

_DETAIL_REQUEST_PATTERNS = [
    r'\bhow\s+(?:do\s+i\s+)?apply\b',
    r'\beligibility\s+(?:criteria|details?|for)\b',
    r'\bdocuments?\s+(?:required|needed)\b',
    r'\bbenefits?\s+of\b',
    r'\bwhat\s+is\s+(?:the\s+)?\w+\s+scheme\b',
    r'\btell\s+me\s+(?:more\s+)?about\b',
    r'\bdetails?\s+(?:of|about|for)\b',
    r'\bmore\s+(?:information|info|details?)\b',
    r'\bapplication\s+(?:process|procedure|form|link)\b',
    r'\bofficial\s+(?:website|portal|link)\b',
    r'\bdeadline\b', r'\blast\s+date\b',
    r'\bview\s+details?\b',
    r'\bincome\s+limit\b', r'\bage\s+limit\b',
]

_GREETING_PATTERNS = [
    r'^(?:hi|hello|hey|namaste|namaskar|hii+|heyyy*)[!.\s]*$',
    r'^(?:good\s+(?:morning|afternoon|evening|night))[!.\s]*$',
    r'^(?:hola|salut|bonjour)[!.\s]*$',
]

_PROFILE_INFO_SIGNALS = [
    # State mentions
    r'\bi\s+(?:am|live|stay|reside|belong)\s+(?:in|from|to)\b',
    r'\bmy\s+(?:state|city|district|home|native)\b',
    r'\bi\s+am\s+(?:a|an)\b',
    r'\b(?:my\s+)?(?:income|caste|category|age|course|degree|qualification|family)\b',
    r'\bmy\s+annual\b', r'\bper\s+year\b', r'\bper\s+annum\b',
    r'\bi\s+study\b', r'\bi\s+am\s+studying\b', r'\bstudying\s+in\b',
    r'\bi\s+have\s+(?:completed|passed|done)\b',
    r'\bfemale\b', r'\bgirl\b', r'\bboy\b',
    r'\b(?:sc|st|obc|ews|general|open|vjnt)\s*(?:category|caste|quota)?\b',
    r'\bdirect\s+second\s+year\b', r'\bdsy\b',
    r'\blakhs?\b', r'\blacs?\b', r'\blpa\b', r'\bl\.p\.a\b', r'\bcrores?\b', r'[₹]',
    r'\bdisabled\b', r'\bdivyang\b',
    r'\bminority\b', r'\bmuslim\b', r'\bchristian\b',
]


def classify_intent_fast(message: str, conversation_history: list[dict] | None = None) -> Intent:
    """
    Fast regex-based intent classification. No LLM calls.

    Args:
        message: The user's current message.
        conversation_history: Previous turns (used to detect clarification context).

    Returns:
        Intent enum value.
    """
    msg_lower = message.lower().strip()

    # Greetings
    for pattern in _GREETING_PATTERNS:
        if re.match(pattern, msg_lower):
            return Intent.GREETING

    # Explicit scheme requests (highest priority)
    for pattern in _SCHEME_REQUEST_PATTERNS:
        if re.search(pattern, msg_lower):
            return Intent.SCHEME_REQUEST

    # Detail requests (second priority)
    for pattern in _DETAIL_REQUEST_PATTERNS:
        if re.search(pattern, msg_lower):
            return Intent.DETAIL_REQUEST

    # Profile info signals
    for pattern in _PROFILE_INFO_SIGNALS:
        if re.search(pattern, msg_lower):
            return Intent.PROFILE_INFO

    # Short responses in context of a question → likely clarification / profile info
    if conversation_history and len(message.strip().split()) <= 6:
        return Intent.CLARIFICATION

    # Single numeric value (likely answering income/age question)
    if re.match(r'^[\d,. ]+(?:lakhs?|lacs?|lac|lpa|l\.p\.a\.?|l|cr|crore|k|rs|₹)?$', msg_lower):
        return Intent.CLARIFICATION

    return Intent.OTHER


async def classify_intent_llm(
    message: str,
    conversation_history: list[dict],
    model_id: str,
    api_key: str | None = None,
    model_config: dict | None = None,
) -> Intent:
    """
    LLM-based intent classification for ambiguous messages.
    Falls back to fast classification on error.
    """
    PROMPT = """Classify this user message into exactly ONE of these intents:

PROFILE_INFO    → User is sharing personal info (name, state, category, income, education, gender, age, disability, etc.)
SCHEME_REQUEST  → User explicitly asks to see/find/list schemes, scholarships, benefits, yojanas they qualify for
DETAIL_REQUEST  → User asks for more details, eligibility criteria, how to apply, documents for a SPECIFIC scheme
GREETING        → Simple hello/hi/namaste with no other content
CLARIFICATION   → User is answering a specific question the assistant just asked
OTHER           → None of the above

Recent conversation (last 4 turns):
{context}

User message: "{message}"

Reply with ONLY one of: PROFILE_INFO, SCHEME_REQUEST, DETAIL_REQUEST, GREETING, CLARIFICATION, OTHER"""

    context = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Bot'}: {m['content'][:150]}"
        for m in (conversation_history[-4:] if len(conversation_history) > 4 else conversation_history)
    )

    # First try fast classification
    fast = classify_intent_fast(message, conversation_history)
    if fast not in (Intent.OTHER, Intent.CLARIFICATION):
        return fast

    try:
        from backend.model_registry import get_model
        from langchain_core.messages import HumanMessage

        model = get_model(model_id, api_key=api_key, model_config=model_config)
        response = await model.ainvoke([HumanMessage(
            content=PROMPT.format(context=context or "None", message=message)
        )])

        content = str(response.content).strip().upper()
        # Extract just the intent keyword
        for intent_name in Intent.__members__:
            if intent_name in content:
                result = Intent[intent_name]
                print(f"[IntentClassifier] LLM → {result.value}")
                return result
    except Exception as e:
        print(f"[IntentClassifier] LLM failed: {e}")

    print(f"[IntentClassifier] Fast → {fast.value}")
    return fast
