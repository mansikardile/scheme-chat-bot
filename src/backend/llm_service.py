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

5. **CONVERSATION MEMORY — CRITICAL**:
   - Read the ENTIRE conversation history before responding.
   - The MOST RECENTLY stated information ALWAYS takes precedence. For example:
     - If the user said "I am from Assam" and later said "I am from Maharashtra", treat them as being from **Maharashtra**.
     - If the user said "I am a student" and later said "I am a girl student", they are BOTH a student AND female — accumulate the profile.
     - If the user changes their category (e.g., first said EWS, then said DSY), the latest one is the active category.
   - NEVER contradict or ignore something the user just told you.
   - NEVER assume or hallucinate profile details not explicitly stated in the conversation.
"""


# ---------------------------------------------------------------------------
# Combined Intent + Profile Extraction — single LLM call for all profile turns
# ---------------------------------------------------------------------------

PROFILE_AND_INTENT_PROMPT = """You are the brain of SchemeSathi, an AI assistant that helps Indian citizens find government welfare schemes.

Your task: Analyse the user's latest message in context of the full conversation and return a JSON object with:
1. The intent of the message
2. Any new profile fields the user is sharing
3. Any fields that need to be CLEARED because the user corrected themselves

## INTENT OPTIONS
- "PROFILE_INFO"    → User is sharing personal info (state, age, gender, income, occupation, etc.)
- "SCHEME_REQUEST"  → User explicitly asks to SEE/FIND/LIST schemes they qualify for
- "DETAIL_REQUEST"  → User asks for details about a specific scheme
- "GREETING"        → Simple hello/hi/namaste
- "OTHER"           → Unclear or unrelated

## PROFILE FIELDS you can extract:
- state: Indian state name (title-case string), e.g. "Goa", "Maharashtra"
- gender: "Male", "Female", or "Transgender"
- category: "SC", "ST", "OBC", "EWS", "General", "Minority", "Disabled", "VJNT", "SBC", "NT"
- education_level: "primary", "secondary", "higher_secondary", "diploma", "undergraduate", "postgraduate", "doctoral"
- study_stage: "first_year", "second_year", "third_year", "fourth_year", "direct_second_year"
- course: e.g. "law", "engineering", "medical", "arts", "science", "commerce", "management"
- annual_family_income: integer in rupees (e.g. 400000 for 4 lakh / 4 LPA)
- age: integer years
- disability_status: "disabled" or "non-disabled"
- minority_status: "minority" or "non-minority"
- residential_status: "permanent_resident" or "resident_<state>"
- employment_status: "employed", "unemployed", "self_employed", "student"
- marital_status: "single", "married", "widow", "widower"
- occupation: e.g. "student", "farmer", "teacher", "lawyer", "doctor", "business_owner", "weaver", "artisan"
- farmer_status: "registered_farmer", "farmer", "non_farmer"
- land_holding: float in acres, or 0 for landless
- institution_type: "government_aided", "private_unaided", "autonomous"
- district: district name string
- housing_status: "own_house", "rented", "no_house"

## CLEAR FIELDS
If the user CORRECTS a previous answer or says they are NOT something, list those field names in "clear_fields".

Examples of corrections that require clears:
- "sorry I am a law student" (was previously said to be a farmer) → clear_fields: ["occupation", "farmer_status", "land_holding"]
- "not a farmer" → clear_fields: ["occupation", "farmer_status", "land_holding"]
- "I am not disabled" → fields: {{disability_status: "non-disabled"}}  (no clear needed, just update)
- "actually I'm from Maharashtra not Goa" → fields: {{state: "Maharashtra"}}, clear_fields: [] (state just overwrites)

## INCOME PARSING (critical)
- "4 LPA" / "4 lakh" / "4L" → 400000
- "4.5 lakh" → 450000
- "8 lakhs per year" → 800000
- "1 crore" → 10000000

## IMPORTANT RULES
- Extract ONLY what is explicitly stated in the current message or clearly answering the bot's last question
- NEVER infer or assume fields not mentioned
- If user says "no" to a yes/no question the bot asked, set the appropriate negative value
- If user says "yes" to a yes/no question, set the appropriate positive value
- The latest statement ALWAYS wins (e.g. if they previously said farmer and now say law student, clear farmer fields)
- For occupation = "student" also set employment_status = "student"

## AUTO-INFERENCE RULES (CRITICAL — always apply these):
- If user mentions "law student", "law college", "llb", "legal studies" → ALWAYS set course="law" AND education_level="undergraduate"
- If user mentions "engineering student", "btech", "b.tech", "be student" → set course="engineering" AND education_level="undergraduate"
- If user mentions "medical student", "mbbs student" → set course="medical" AND education_level="undergraduate"
- If user says "1st year", "first year", "2nd year" etc. at college → set study_stage accordingly AND education_level="undergraduate"
- If user says "direct second year" or "DSY" → set study_stage="direct_second_year" AND education_level="undergraduate"
- If occupation/course clearly indicates a degree student, ALWAYS include education_level="undergraduate" in fields


Conversation history (most recent last):
{history}

User's LATEST message: "{message}"

Return ONLY this JSON, no explanation, no markdown fences:
{{
  "intent": "<one of the intent options above>",
  "fields": {{<field_name>: <value>, ...}},
  "clear_fields": ["<field1>", "<field2>", ...]
}}"""


async def extract_profile_and_intent(
    user_message: str,
    conversation_history: list[dict],
    model_id: str,
    api_key: str | None = None,
    model_config: dict | None = None,
) -> dict:
    """Single LLM call that simultaneously classifies intent AND extracts/clears profile fields.

    Returns dict with keys:
      - intent: str (e.g. "PROFILE_INFO", "SCHEME_REQUEST")
      - fields: dict of new/updated profile fields
      - clear_fields: list of field names to set to None (user corrections)
    Falls back to empty fields + "OTHER" intent on error.
    """
    # Build compact history string (last 8 turns to keep context manageable)
    history_lines = []
    for m in conversation_history[-8:]:
        role = "User" if m.get('role') == 'user' else "Bot"
        content = (m.get('content') or '')[:300].replace('\n', ' ')
        history_lines.append(f"{role}: {content}")
    history_str = "\n".join(history_lines) if history_lines else "None"

    prompt = PROFILE_AND_INTENT_PROMPT.format(
        history=history_str,
        message=user_message,
    )

    try:
        model = get_model(model_id, api_key=api_key, model_config=model_config)
        response = await model.ainvoke([HumanMessage(content=prompt)])
        text = _extract_text_content(response.content).strip()

        # Strip markdown fences if model wrapped the JSON
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            intent = parsed.get('intent', 'OTHER')
            fields = parsed.get('fields') or {}
            clear_fields = parsed.get('clear_fields') or []

            # Sanitise: remove null/empty/zero income fields
            fields = {k: v for k, v in fields.items()
                      if v is not None and v != '' and v != [] }
            if 'annual_family_income' in fields and fields['annual_family_income'] in (0, '0', ''):
                del fields['annual_family_income']

            print(f"[LLM IntentExtract] intent={intent}, fields={fields}, clear={clear_fields}")
            return {'intent': intent, 'fields': fields, 'clear_fields': clear_fields}
    except Exception as e:
        print(f"[LLM IntentExtract] Failed: {e}")

    return {'intent': 'OTHER', 'fields': {}, 'clear_fields': []}


# ---------------------------------------------------------------------------
# Profile Conversation Generation — natural LLM response during intake
# ---------------------------------------------------------------------------

PROFILE_CONVERSATION_PROMPT = """You are SchemeSathi — a warm, patient, and helpful AI assistant helping Indian citizens find government welfare schemes.

## YOUR TASK
The user just shared some information. Generate ONE short, natural, conversational response that:
1. Briefly acknowledges what they said (warmly, 1 sentence max)
2. Asks the SINGLE MOST IMPORTANT missing question listed below — just that one, no more

## RULES
- Respond in {language_name}
- Keep it SHORT: 2–3 sentences max
- Sound warm and human, NOT robotic
- Do NOT list bullet points of what you know — just acknowledge naturally in prose
- Do NOT ask multiple questions — just ONE
- If the user corrected themselves (e.g. said "not a farmer"), acknowledge the correction gently
- If user said something contradictory in the past, do NOT bring it up — just work with current profile
- Do NOT make up any schemes or information
- If profile is complete (next_question is null), tell them they can say "show me schemes" when ready

## CURRENT USER PROFILE (accumulated so far):
{profile_summary}

## NEXT QUESTION TO ASK (ask exactly this, but phrase it naturally):
{next_question}

## RECENT CONVERSATION (last 6 turns):
{history}

## USER'S LATEST MESSAGE:
"{user_message}"

Now write your response:"""


async def generate_profile_conversation(
    user_message: str,
    user_profile: dict,
    next_question: str | None,
    conversation_history: list[dict],
    language: str = 'en',
    model_id: str = 'gemini-flash',
    api_key: str | None = None,
    model_config: dict | None = None,
) -> str:
    """Generate a natural conversational reply during profile collection.

    This replaces the hardcoded 'Got it! Here's what I have so far:' template.
    The LLM will acknowledge what the user said and ask the next question naturally.
    """
    from backend.profile_advisor import build_profile_summary_text

    # Build readable profile summary for LLM
    collected_text, _ = build_profile_summary_text(user_profile)
    if not collected_text:
        profile_summary = "(No profile information collected yet)"
    else:
        profile_summary = collected_text

    # Build recent history
    history_lines = []
    for m in conversation_history[-6:]:
        role = "User" if m.get('role') == 'user' else "Bot"
        content = (m.get('content') or '')[:200].replace('\n', ' ')
        history_lines.append(f"{role}: {content}")
    history_str = "\n".join(history_lines) if history_lines else "None"

    next_q = next_question or "null — profile is sufficiently complete. Tell them to say 'show me schemes'."
    lang_name = LANGUAGE_NAMES.get(language, 'English')

    prompt = PROFILE_CONVERSATION_PROMPT.format(
        language_name=lang_name,
        profile_summary=profile_summary,
        next_question=next_q,
        history=history_str,
        user_message=user_message,
    )

    try:
        model = get_model(model_id, api_key=api_key, model_config=model_config)
        response = await model.ainvoke([HumanMessage(content=prompt)])
        text = _extract_text_content(response.content).strip()
        # Strip URLs just in case
        text = re.sub(r'https?://[^\s)]+', '', text)
        return text
    except Exception as e:
        print(f"[ProfileConversation] LLM failed: {e}")
        # Graceful fallback
        if next_question:
            return f"Got it! {next_question}"
        return "I have enough information to search for schemes. Just say **\"Show me the schemes\"** when you're ready!"


# ---------------------------------------------------------------------------
# LLM Final Eligibility Verification — double-checks scheme vs user profile
# ---------------------------------------------------------------------------

LLM_ELIGIBILITY_VERIFY_PROMPT = """You are a strict eligibility checker for Indian government welfare schemes.

Your job: Verify if this specific user is actually eligible for the scheme. Be VERY strict — reject any scheme that clearly isn't meant for this type of person.

## USER PROFILE:
{profile}

## SCHEME NAME: {scheme_name}

## SCHEME ELIGIBILITY CRITERIA:
{eligibility_text}

## MANDATORY CHECKS (check ALL of these):

### 1. BENEFICIARY TYPE CHECK (most important)
Look at the scheme name and description — what type of person is it ACTUALLY for?
- Pension for elderly/senior citizens (60+) → INELIGIBLE if user is young (under 50)
- HIV/AIDS support schemes → INELIGIBLE if user has not mentioned HIV/AIDS
- Widow/destitute women pension → INELIGIBLE if user is single/young student
- Nursing training → INELIGIBLE if user is not a nursing student
- Sports scholarship → INELIGIBLE if user is not an athlete/sportsperson
- Construction worker/BOCW → INELIGIBLE if user is a student, not a construction worker
- Farmer schemes → INELIGIBLE if user's occupation is not farmer

### 2. AGE CHECK
If the scheme has an age requirement (min/max) and user's age clearly doesn't fit → INELIGIBLE

### 3. INCOME CHECK
If the scheme has an income limit and user's income clearly exceeds it → INELIGIBLE

### 4. STATE CHECK
If the scheme is for a specific state and user is from a different state → INELIGIBLE

### 5. RELEVANCE CHECK
Is this scheme actually relevant to what the user is? If it's completely irrelevant to this person's situation, mark as INELIGIBLE.

## DECISION RULES:
- ELIGIBLE: User clearly matches this scheme's target beneficiary group AND all verifiable conditions pass
- INELIGIBLE: Scheme is clearly NOT meant for this person OR a verifiable condition fails
- INSUFFICIENT: Scheme seems relevant but key eligibility info is missing from the profile

Return ONLY this JSON, no explanation, no markdown:
{{
  "status": "ELIGIBLE" | "INELIGIBLE" | "INSUFFICIENT",
  "confidence": <0.0 to 1.0>,
  "reason": "<one sentence: why eligible or why rejected>",
  "failed_conditions": ["<specific condition that fails, if any>"]
}}"""


async def llm_verify_eligibility(
    user_profile: dict,
    scheme_name: str,
    eligibility_text: str,
    model_id: str,
    api_key: str | None = None,
    model_config: dict | None = None,
) -> dict:
    """LLM final pass to verify a user's eligibility for a specific scheme.

    This is called AFTER the deterministic engine marks a scheme as ELIGIBLE,
    to catch false positives from incomplete regex rules.

    Returns:
        dict with keys: status ('ELIGIBLE'/'INELIGIBLE'/'INSUFFICIENT'),
                        confidence (float), reason (str), failed_conditions (list)
    """
    if not eligibility_text or len(eligibility_text.strip()) < 10:
        # No eligibility text to verify against — trust deterministic engine
        return {'status': 'ELIGIBLE', 'confidence': 0.7, 'reason': 'No detailed eligibility text available', 'failed_conditions': []}

    # Build human-readable profile string
    profile_lines = []
    field_labels = {
        'state': 'State', 'gender': 'Gender', 'category': 'Category/Caste',
        'age': 'Age', 'annual_family_income': 'Annual Family Income (₹)',
        'education_level': 'Education Level', 'course': 'Course/Field',
        'study_stage': 'Study Stage', 'occupation': 'Occupation',
        'farmer_status': 'Farmer Status', 'land_holding': 'Land Holding (acres)',
        'marital_status': 'Marital Status', 'disability_status': 'Disability Status',
        'minority_status': 'Minority Status', 'residential_status': 'Residential Status',
        'employment_status': 'Employment Status', 'institution_type': 'Institution Type',
        'housing_status': 'Housing Status', 'district': 'District',
    }
    for field, label in field_labels.items():
        val = user_profile.get(field)
        if val is not None:
            profile_lines.append(f"- {label}: {val}")
    profile_str = "\n".join(profile_lines) if profile_lines else "No profile information available"

    prompt = LLM_ELIGIBILITY_VERIFY_PROMPT.format(
        profile=profile_str,
        scheme_name=scheme_name,
        eligibility_text=eligibility_text[:2000],  # Cap to avoid token overflow
    )

    try:
        model = get_model(model_id, api_key=api_key, model_config=model_config)
        response = await model.ainvoke([HumanMessage(content=prompt)])
        text = _extract_text_content(response.content).strip()

        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            status = result.get('status', 'ELIGIBLE')
            confidence = float(result.get('confidence', 0.7))
            reason = result.get('reason', '')
            failed = result.get('failed_conditions', [])
            print(f"[LLMVerify] {scheme_name}: {status} (conf={confidence:.2f}) — {reason}")
            return {'status': status, 'confidence': confidence, 'reason': reason, 'failed_conditions': failed}
    except Exception as e:
        print(f"[LLMVerify] Failed for {scheme_name}: {e}")

    # On failure, trust deterministic engine
    return {'status': 'ELIGIBLE', 'confidence': 0.6, 'reason': 'LLM verification unavailable', 'failed_conditions': []}


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

Recent conversation: {context}
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

    Returns a dict with optional keys: state, category, keywords.
    (level is intentionally excluded — the state SQL condition already handles
    Central+State scheme discovery correctly via OR level='Central'.)
    Returns an empty dict on any failure so the caller can gracefully fall back.
    """
    # Build context from recent user messages AND the last AI response (truncated).
    # The AI response often contains scheme names that the user refers to with
    # vague pronouns ("these schemes", "it", "this") in the next turn.
    context_parts = []
    for m in conversation_history[-6:]:
        if m['role'] == 'user':
            context_parts.append(f"User: {m['content']}")
        elif m['role'] == 'assistant':
            # Truncate AI responses to avoid blowing up the prompt
            snippet = m['content'][:300].replace('\n', ' ')
            context_parts.append(f"Assistant: {snippet}")
    context = " | ".join(context_parts) if context_parts else "None"

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

    # Include both user messages and the last AI response (truncated) so that
    # vague references like "these schemes" or "can IT be used for X" can be
    # resolved against what the assistant just recommended.
    context_parts = []
    for m in conversation_history[-6:]:
        if m['role'] == 'user':
            context_parts.append(f"User: {m['content']}")
        elif m['role'] == 'assistant':
            snippet = m['content'][:300].replace('\n', ' ')
            context_parts.append(f"Assistant: {snippet}")
    context = " | ".join(context_parts) if context_parts else "None"

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
    """Extract clean string text from LangChain message content (str, list, or dict), ignoring thinking/reasoning blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                item_type = item.get('type', '')
                if item_type in ('thinking', 'reasoning', 'thought') or 'thinking' in item or 'thought' in item:
                    continue
                if item_type == 'text' and 'text' in item:
                    parts.append(str(item['text']))
                elif 'text' in item:
                    parts.append(str(item['text']))
        return "".join(parts)
    if isinstance(content, dict):
        item_type = content.get('type', '')
        if item_type in ('thinking', 'reasoning', 'thought') or 'thinking' in content or 'thought' in content:
            return ""
        if item_type == 'text' and 'text' in content:
            return str(content['text'])
        if 'text' in content:
            return str(content['text'])
        return ""
    return ""


def _build_profile_summary(user_profile: dict | None) -> str:
    """Build a concise human-readable profile summary from the accumulated session profile.

    Returns an empty string when no profile fields have been set yet.
    """
    if not user_profile:
        return ""
    labels = {
        'state': 'State',
        'category': 'Category/Caste',
        'gender': 'Gender',
        'level': 'Scheme Level',
        'keywords': 'Additional context',
    }
    lines = []
    for key, label in labels.items():
        val = user_profile.get(key)
        if val:
            lines.append(f"- {label}: {val}")
    return "\n".join(lines)


async def generate_response(
    model_id: str,
    conversation_history: list[dict],
    user_message: str,
    scheme_context: str,
    language: str = 'en',
    api_key: str | None = None,
    model_config: dict | None = None,
    user_profile: dict | None = None,
) -> str:
    """Generate a conversational response grounded in scheme data using a model from model_registry."""
    print(f"\n[LLM Service] Processing chat request using model_id: '{model_id}'")
    try:
        model = get_model(model_id, api_key=api_key, model_config=model_config)
    except Exception as e:
        print(f"Error instantiating model '{model_id}': {e}")
        err = str(e)
        if "GEMINI_API_KEY" in err:
            return "Required: Please add GEMINI_API_KEY to your .env file or enter it via Settings."
        return f'Model configuration error: {err}'

    lang_name = LANGUAGE_NAMES.get(language, 'English')
    system_text = SYSTEM_PROMPT.format(language_name=lang_name)

    # Build LangChain message list: [SystemMessage] + history + augmented user turn
    messages: list = [SystemMessage(content=system_text)]

    for msg in conversation_history:
        if msg['role'] == 'user':
            messages.append(HumanMessage(content=msg['content']))
        else:
            messages.append(AIMessage(content=msg['content']))

    # Inject structured profile summary so the LLM always has authoritative accumulated facts
    profile_summary = _build_profile_summary(user_profile)
    profile_block = (
        f"[USER PROFILE - ACCUMULATED ACROSS CONVERSATION]\n{profile_summary}\n[END USER PROFILE]\n\n"
        if profile_summary else ""
    )
    augmented_user = (
        f"{user_message}\n\n"
        + profile_block
        + f"[SCHEME DATA]\n{scheme_context}\n[END SCHEME DATA]\n\n"
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
    user_profile: dict | None = None,
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

    # Inject structured profile summary so the LLM always has authoritative accumulated facts
    profile_summary = _build_profile_summary(user_profile)
    profile_block = (
        f"[USER PROFILE - ACCUMULATED ACROSS CONVERSATION]\n{profile_summary}\n[END USER PROFILE]\n\n"
        if profile_summary else ""
    )
    augmented_user = (
        f"{user_message}\n\n"
        + profile_block
        + f"[SCHEME DATA]\n{scheme_context}\n[END SCHEME DATA]\n\n"
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

