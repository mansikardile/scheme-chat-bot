"""Model registry for provider-agnostic LLM configurations with local/server routing."""

import os
import httpx
from dataclasses import dataclass
from typing import Callable, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from backend.config import GEMINI_API_KEY, OLLAMA_HOST, SERVER_OLLAMA_HOST, EMBEDDING_API_KEY, CONFIGURED_MODELS


@dataclass
class ModelInfo:
    id: str
    label: str
    factory: Callable[[str | None], BaseChatModel]
    requires_api_key: bool


LOCAL_OLLAMA_HOST = "http://localhost:11434"

def is_model_downloaded_locally(model_name: str) -> bool:
    """Checks if local Ollama instance is running and has the specified model downloaded.
    Always queries localhost:11434 regardless of OLLAMA_HOST env var."""
    try:
        url = f"{LOCAL_OLLAMA_HOST}/api/tags"
        response = httpx.get(url, timeout=1.5)
        if response.status_code == 200:
            models_data = response.json().get('models', [])
            # Compare full names like 'gemma3:4b' or just base name 'gemma3'
            installed_names = [m.get('name', '').lower() for m in models_data]
            installed_bases = [n.split(':')[0] for n in installed_names]
            target = model_name.lower()
            base_target = target.split(':')[0]
            return target in installed_names or base_target in installed_bases
    except Exception:
        pass
    return False


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


def _create_ollama_model(model_name: str) -> BaseChatModel:
    """Factory function to build a ChatOllama model with local check and ai.11022006.xyz server fallback."""
    is_local = is_model_downloaded_locally(model_name)
    target_host = LOCAL_OLLAMA_HOST if is_local else SERVER_OLLAMA_HOST
    print(f"[Model Registry] Instantiating Ollama model '{model_name}'. Downloaded locally: {is_local} -> Using host: {target_host}")

    client_kwargs: dict[str, Any] = {}
    if not is_local and EMBEDDING_API_KEY:
        client_kwargs["headers"] = {"Authorization": f"Bearer {EMBEDDING_API_KEY}"}

    return ChatOllama(
        model=model_name,
        base_url=target_host,
        temperature=0.6,
        client_kwargs=client_kwargs if client_kwargs else None,
    )


def get_available_models_info() -> list[dict]:
    """Return available models with their current execution status (local vs server vs cloud)."""
    model_list = []
    for cfg in CONFIGURED_MODELS:
        m_id = cfg['id']
        m_name = cfg['name']
        provider = cfg['provider']
        model_name = cfg.get('model_name', m_id)
        desc = cfg.get('description', '')

        if provider == 'gemini':
            location = 'cloud'
            location_label = 'Cloud (Google API)'
        else:
            is_local = is_model_downloaded_locally(model_name)
            if is_local:
                location = 'local'
                location_label = 'Local (Downloaded)'
            else:
                location = 'server'
                location_label = 'Server (ai.11022006.xyz)'

        model_list.append({
            'id': m_id,
            'name': m_name,
            'provider': provider,
            'location': location,
            'location_label': location_label,
            'description': desc,
        })
    return model_list


def get_model(model_id: str, api_key: str | None = None) -> BaseChatModel:
    """Retrieve a configured LangChain BaseChatModel instance for the given model_id."""
    # Find in CONFIGURED_MODELS
    matched_cfg = next((c for c in CONFIGURED_MODELS if c['id'] == model_id), None)
    if not matched_cfg:
        # Fallback to default gemini-flash if unknown ID
        print(f"[Model Registry] Unknown model_id '{model_id}', falling back to 'gemini-flash'")
        return _create_gemini_model(api_key)

    provider = matched_cfg.get('provider', 'gemini')
    model_name = matched_cfg.get('model_name', model_id)

    if provider == 'gemini':
        return _create_gemini_model(api_key)
    else:
        return _create_ollama_model(model_name)

