import uuid
from datetime import datetime
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage


class ChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.language: str = 'en'
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        # LangChain in-memory message store for this session
        self._history = InMemoryChatMessageHistory()

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
