import uuid
from datetime import datetime


class ChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: list[dict] = []
        self.language: str = 'en'
        self.created_at = datetime.now()
        self.last_active = datetime.now()

    def add_message(self, role: str, content: str):
        self.messages.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
        })
        self.last_active = datetime.now()

    def get_history(self) -> list[dict]:
        return self.messages


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
