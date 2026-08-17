"""Model registry for provider-agnostic LLM configurations with local/server routing."""

import os
import httpx
from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from backend.config import GEMINI_API_KEY, OLLAMA_HOST, SERVER_OLLAMA_HOST, EMBEDDING_API_KEY, CONFIGURED_MODELS

LOCAL_OLLAMA_HOST = "http://localhost:11434"


def get_local_ollama_models() -> dict[str, Any]:
    """Query local Ollama instance (localhost:11434) to list all downloaded models."""
    try:
        url = f"{LOCAL_OLLAMA_HOST}/api/tags"
        response = httpx.get(url, timeout=2.0)
        if response.status_code == 200:
            data = response.json()
            raw_models = data.get("models", [])
            model_list = []
            for m in raw_models:
                name = m.get("name", "")
                size_bytes = m.get("size", 0)
                size_gb = f"{size_bytes / (1024 ** 3):.1f} GB" if size_bytes else ""
                details = m.get("details", {})
                param_size = details.get("parameter_size", "")
                model_list.append({
                    "name": name,
                    "size": size_gb,
                    "parameter_size": param_size,
                    "modified_at": m.get("modified_at", "")
                })
            return {
                "running": True,
                "models": model_list
            }
    except Exception as e:
        pass
    return {
        "running": False,
        "models": []
    }


def is_model_downloaded_locally(model_name: str) -> bool:
    """Checks if local Ollama instance is running and has the specified model downloaded."""
    local_info = get_local_ollama_models()
    if not local_info["running"]:
        return False
    installed_names = [m["name"].lower() for m in local_info["models"]]
    installed_bases = [n.split(":")[0] for n in installed_names]
    target = model_name.lower()
    base_target = target.split(":")[0]
    return target in installed_names or base_target in installed_bases


def _create_gemini_model(model_name: str = "gemini-flash-latest", api_key: str | None = None) -> BaseChatModel:
    """Factory function to build a Gemini ChatGoogleGenerativeAI instance with fallbacks."""
    key = api_key or GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("GEMINI_API_KEY not configured. Please enter a Gemini API Key in Settings or set GEMINI_API_KEY.")

    primary = ChatGoogleGenerativeAI(
        model=model_name or "gemini-flash-latest",
        google_api_key=key,
        temperature=0.6,
        max_output_tokens=1024,
    )
    fallback_models = ["gemini-flash-lite-latest", "gemini-2.0-flash"]
    fallbacks = [
        ChatGoogleGenerativeAI(
            model=m,
            google_api_key=key,
            temperature=0.6,
            max_output_tokens=1024,
        )
        for m in fallback_models if m != model_name
    ]
    if fallbacks:
        return primary.with_fallbacks(fallbacks)
    return primary


def _create_ollama_model(
    model_name: str,
    base_url: str | None = None,
    api_key: str | None = None,
    is_local: bool = True
) -> BaseChatModel:
    """Factory function to build a ChatOllama instance (local or remote custom server)."""
    if is_local:
        target_host = LOCAL_OLLAMA_HOST
    elif base_url:
        target_host = base_url.rstrip("/")
    else:
        # Default server fallback
        target_host = SERVER_OLLAMA_HOST

    print(f"[Model Registry] Instantiating Ollama model '{model_name}' on host: {target_host} (is_local={is_local})")

    client_kwargs: dict[str, Any] = {}
    auth_key = api_key or (EMBEDDING_API_KEY if not is_local and target_host == SERVER_OLLAMA_HOST else None)
    if auth_key and not is_local:
        client_kwargs["headers"] = {"Authorization": f"Bearer {auth_key}"}

    return ChatOllama(
        model=model_name,
        base_url=target_host,
        temperature=0.6,
        client_kwargs=client_kwargs if client_kwargs else None,
    )


def get_available_models_info() -> list[dict]:
    """Return default configured models."""
    model_list = []
    for cfg in CONFIGURED_MODELS:
        m_id = cfg["id"]
        m_name = cfg["name"]
        provider = cfg.get("provider", "gemini")
        model_name = cfg.get("model_name", m_id)
        desc = cfg.get("description", "")
        is_local = cfg.get("is_local", False)

        model_list.append({
            "id": m_id,
            "name": m_name,
            "provider": provider,
            "model_name": model_name,
            "is_local": is_local,
            "description": desc,
        })
    return model_list


def get_model(
    model_id: str,
    api_key: str | None = None,
    model_config: dict[str, Any] | None = None
) -> BaseChatModel:
    """Retrieve a configured LangChain BaseChatModel instance based on model_id or custom model_config."""
    # 1. If explicit custom model_config is provided (e.g. user added custom model in browser)
    if model_config and isinstance(model_config, dict):
        provider = model_config.get("provider", "ollama")
        is_local = model_config.get("is_local", False)
        model_name = model_config.get("model_name") or model_config.get("name", "gemini-flash-latest")
        custom_key = model_config.get("api_key") or api_key
        base_url = model_config.get("base_url") or model_config.get("link")

        if provider == "gemini":
            return _create_gemini_model(model_name=model_name, api_key=custom_key)
        else:
            return _create_ollama_model(
                model_name=model_name,
                base_url=base_url,
                api_key=custom_key,
                is_local=is_local
            )

    # 2. Match from CONFIGURED_MODELS
    matched_cfg = next((c for c in CONFIGURED_MODELS if c["id"] == model_id), None)
    if matched_cfg:
        provider = matched_cfg.get("provider", "gemini")
        model_name = matched_cfg.get("model_name", model_id)
        is_local = matched_cfg.get("is_local", False)
        base_url = matched_cfg.get("base_url")

        if provider == "gemini":
            return _create_gemini_model(model_name=model_name, api_key=api_key)
        else:
            return _create_ollama_model(
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
                is_local=is_local
            )

    # 3. Default fallback to Gemini
    print(f"[Model Registry] Unknown model_id '{model_id}', defaulting to Gemini")
    return _create_gemini_model("gemini-flash-latest", api_key=api_key)
