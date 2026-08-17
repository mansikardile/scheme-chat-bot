from pydantic import BaseModel

class ChatRequest(BaseModel):
    session_id: str
    message: str
    language: str = 'en'  # Language code: en, hi, ta, te, bn, mr, gu, kn, ml, pa, etc.
    model: str = 'gemini-flash'  # Selected AI model ID
    api_key: str | None = None  # Optional user-provided API key



class SchemeCard(BaseModel):
    slug: str
    name: str
    brief: str
    level: str
    states: list[str] = []
    categories: list[str] = []
    tags: list[str] = []
    has_details: bool = False

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    schemes: list[SchemeCard] = []
