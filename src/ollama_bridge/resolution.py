"""
Deterministic Model Resolution for Ollama MCP Bridge.
License: GNU AGPLv3
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import requests

from .config import (
    DEFAULT_FALLBACK_MODEL,
    OLLAMA_HOST,
    PREFERRED_MODELS,
    get_configured_model,
)

logger = logging.getLogger("ollama_bridge.resolution")


def _find_by_keyword(names: List[str], keyword: str) -> Optional[str]:
    """Find the first model name containing the given keyword."""
    for name in names:
        if keyword in name.lower():
            return name
    return None


def select_preferred_model(available_names: List[str]) -> Optional[str]:
    """
    Select preferred coding model from available names according to priority list.
    Filters out empty/whitespace-only model strings.
    McCabe Complexity M <= 5.
    """
    valid_names = [n.strip() for n in available_names if n and n.strip()]
    if not valid_names:
        return None

    for pref in PREFERRED_MODELS:
        if pref in valid_names:
            return pref

    coder_match = _find_by_keyword(valid_names, "coder")
    if coder_match:
        return coder_match

    code_match = _find_by_keyword(valid_names, "code")
    if code_match:
        return code_match

    return valid_names[0]


def _fetch_tags_names() -> List[str]:
    """Fetch model names directly from Ollama endpoint."""
    try:
        url = f"{OLLAMA_HOST}/api/tags"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        return [m.get("name", "") for m in models if isinstance(m, dict)]
    except Exception as e:
        logger.warning(f"Failed to query Ollama tags during resolution: {e}")
        return []


def resolve_model(
    requested_model: str = "",
    available_models: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Resolve target model based on strict 4-level precedence (REQ-001):
    1. Explicit tool argument
    2. Environment variable LOCAL_LLM_MODEL
    3. Auto-detected preferred local model
    4. Fallback default ('qwen2.5-coder:14b')
    McCabe Complexity M <= 5.
    """
    explicit = requested_model.strip() if requested_model else ""
    if explicit:
        return explicit

    configured = get_configured_model()
    if configured:
        return configured

    if available_models is not None:
        names = [m.get("name", "") for m in available_models if isinstance(m, dict)]
    else:
        names = _fetch_tags_names()

    auto_model = select_preferred_model(names)
    if auto_model:
        return auto_model

    return DEFAULT_FALLBACK_MODEL
