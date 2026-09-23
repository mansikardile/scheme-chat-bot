"""Chat session and manager for SchemeSathi."""

import uuid
from datetime import datetime
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage


# Full profile schema — all fields start as None
_EMPTY_PROFILE = {
    # Location
    'state': None,
    'district': None,
    # Demographics
    'age': None,
    'gender': None,
    # Caste / category
    'category': None,
    'subcategory': None,
    # Income
    'annual_family_income': None,
    # Education
    'education_level': None,   # pre-primary/primary/secondary/higher_secondary/undergraduate/postgraduate/doctoral
    'course': None,            # engineering/medical/arts/science/commerce/law/management etc.
    'stream': None,
    'study_stage': None,       # first_year/second_year/third_year/fourth_year/direct_second_year
    'year_of_study': None,
    'college_type': None,      # government/private/aided
    'institution_type': None,
    # Residency
    'residential_status': None,
    # Special conditions
    'disability_status': None,
    'minority_status': None,
    # Employment
    'employment_status': None,
    # Personal
    'marital_status': None,
    'occupation': None,
    # Farmer-specific
    'farmer_status': None,
    'land_holding': None,
    # Housing
    'housing_status': None,
    # Misc
    'specific_special_conditions': [],
}


class ChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.language: str = 'en'
        self.created_at = datetime.now()
        self.last_active = datetime.now()

        # LangChain in-memory message store for this session
        self._history = InMemoryChatMessageHistory()

        # Accumulated user profile — updated after every user turn.
        # Most-recent mention always wins.
        self.user_profile: dict = dict(_EMPTY_PROFILE)

        # Conversation state
        self.conversation_mode: str = 'profile_collection'  # 'profile_collection' | 'scheme_search'
        self.scheme_search_requested: bool = False

        # Per-session eligibility rules cache: slug -> list[EligibilityRule]
        # Avoids re-calling LLM for the same scheme in the same session
        self.eligibility_cache: dict = {}

        # Last set of retrieved+eligible scheme slugs (for detail follow-ups)
        self.last_eligible_slugs: list[str] = []

    def update_profile(self, new_fields: dict):
        """
        Merge new_fields into the persistent profile.
        Non-null values from new_fields overwrite existing ones.
        """
        for key, value in new_fields.items():
            if key == 'specific_special_conditions':
                if isinstance(value, list):
                    existing = self.user_profile.get('specific_special_conditions') or []
                    merged = list(set(existing + value))
                    self.user_profile['specific_special_conditions'] = merged
            elif value is not None and value != '' and value != []:
                # Safety guard: never overwrite a previously recorded income (>0) with 0 or empty
                if key == 'annual_family_income' and (value == 0 or value == '0') and (self.user_profile.get('annual_family_income') or 0) > 0:
                    continue
                self.user_profile[key] = value

    def clear_profile_fields(self, field_names: list[str]):
        """
        Explicitly set the given fields back to None.
        Used when the LLM detects the user corrected a previous answer
        (e.g. "I am not a farmer" after previously saying they were).
        """
        for field in field_names:
            if field in self.user_profile:
                self.user_profile[field] = None
                print(f"[ChatSession] Cleared profile field: {field}")

    def get_profile_for_llm(self) -> str:
        """Return the current profile as a readable multi-line string for LLM prompts."""
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
        lines = []
        for field, label in field_labels.items():
            val = self.user_profile.get(field)
            if val is not None:
                lines.append(f"- {label}: {val}")
        return '\n'.join(lines) if lines else '(No profile information yet)'

    def get_profile(self) -> dict:
        """Return a copy of the current accumulated user profile."""
        return dict(self.user_profile)

    def has_minimum_profile(self) -> bool:
        """
        Check if we have at least the state and one other field,
        sufficient to start a scheme search.
        """
        p = self.user_profile
        has_state = bool(p.get('state'))
        other_fields = ['category', 'gender', 'education_level', 'course',
                        'annual_family_income', 'age', 'occupation']
        has_other = any(bool(p.get(f)) for f in other_fields)
        return has_state and has_other

    def add_message(self, role: str, content: str):
        """Append a message and update the last_active timestamp."""
        if role == 'user':
            self._history.add_message(HumanMessage(content=content))
        else:
            self._history.add_message(AIMessage(content=content))
        self.last_active = datetime.now()

    def get_history(self) -> list[dict]:
        """Return history as [{'role': ..., 'content': ...}] — same shape as before."""
        result = []
        for msg in self._history.messages:
            role = 'user' if isinstance(msg, HumanMessage) else 'model'
            result.append({'role': role, 'content': msg.content})
        return result


class ChatManager:
    def __init__(self):
        self.sessions: dict[str, ChatSession] = {}

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = ChatSession(session_id)
        return session_id

    def get_session(self, session_id: str):
        return self.sessions.get(session_id)

    def cleanup_stale(self, max_age_hours: int = 1):
        now = datetime.now()
        stale = [
            sid for sid, s in self.sessions.items()
            if (now - s.last_active).total_seconds() > max_age_hours * 3600
        ]
        for sid in stale:
            del self.sessions[sid]


chat_manager = ChatManager()
