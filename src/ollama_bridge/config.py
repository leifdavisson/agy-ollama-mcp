"""
Configuration management for Ollama MCP Bridge.
License: GNU AGPLv3
"""
from __future__ import annotations

import logging
import os
import sys
from typing import List

OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
CONFIGURED_MODEL: str = os.getenv("LOCAL_LLM_MODEL", "").strip()
DEFAULT_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "180"))
DEFAULT_NUM_CTX: int = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
DEFAULT_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
DEFAULT_FALLBACK_MODEL: str = "qwen2.5-coder:14b"

PREFERRED_MODELS: List[str] = [
    "qwen2.5-coder:14b",
    "qwen2.5-coder:7b",
    "deepseek-r1:14b",
    "deepseek-r1:8b",
    "dolphin3-tools:latest",
    "qwen3.5:latest",
    "qwen3.5:27b",
    "llama3.2:latest",
]


def configure_logging(level: int = logging.WARNING) -> logging.Logger:
    """Configure runtime logging ensuring stderr isolation (REQ-008)."""
    logger = logging.getLogger("ollama_bridge")
    logger.setLevel(level)
    logger.handlers.clear()
    handler = logging.StreamHandler(stream=sys.stderr)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_configured_model() -> str:
    """Retrieve the configured model from environment (REQ-001)."""
    return os.getenv("LOCAL_LLM_MODEL", "").strip()
