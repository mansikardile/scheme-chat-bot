"""Model registry for provider-agnostic LLM configurations."""

import os
from dataclasses import dataclass
from typing import Callable
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.config import GEMINI_API_KEY


@dataclass
class ModelInfo:
    id: str
    label: str
    factory: Callable[[str | None], BaseChatModel]
    requires_api_key: bool


def _create_gemini_model(api_key: str | None = None) -> BaseChatModel:
    """Factory function to build a Gemini ChatGoogleGenerativeAI instance with fallbacks."""
    key = api_key or GEMINI_API_KEY or os.getenv('GEMINI_API_KEY', '')
    if not key:
        raise ValueError('GEMINI_API_KEY not configured. Please add GEMINI_API_KEY to your .env file.')

    primary = ChatGoogleGenerativeAI(
        model='gemini-flash-latest',
        google_api_key=key,
        temperature=0.6,
        max_output_tokens=1024,
    )
    fallback_models = ['gemini-flash-lite-latest', 'gemini-2.0-flash']
    fallbacks = [
        ChatGoogleGenerativeAI(
            model=m,
            google_api_key=key,
            temperature=0.6,
            max_output_tokens=1024,
        )
        for m in fallback_models
    ]
    return primary.with_fallbacks(fallbacks)


MODEL_REGISTRY: dict[str, ModelInfo] = {
    'gemini-flash': ModelInfo(
        id='gemini-flash',
        label='Google Gemini Flash',
        factory=_create_gemini_model,
        requires_api_key=True,
    ),
}


def get_model(model_id: str, api_key: str | None = None) -> BaseChatModel:
    """Retrieve a configured LangChain BaseChatModel instance for the given model_id."""
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_id '{model_id}'. Registered models: {list(MODEL_REGISTRY.keys())}")
    info = MODEL_REGISTRY[model_id]
    return info.factory(api_key)
