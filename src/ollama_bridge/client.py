"""
Ollama Client and Prewarm Module.

This module provides the OllamaClient class and helper functions for interacting
with a local Ollama daemon, inspecting installed model inventory, converting
byte metrics, and prewarming models into memory/VRAM with keep_alive=-1.

Requirements:
    - REQ-006: Model Inventory & Status Enumeration
    - REQ-007: Memory Pre-warming & Keep-Alive
Specification:
    - features/model_inventory.feature
License: GNU AGPLv3
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional
import requests

from .resolution import resolve_model

logger = logging.getLogger("ollama_bridge.client")


def format_bytes_to_gb(bytes_val: int) -> str:
    """Convert raw byte size to gigabytes formatted to two decimal places.

    Example:
        8988112209 -> "8.37 GB"
    """
    gb_val = round(bytes_val / (1024 ** 3), 2)
    return f"{gb_val:.2f} GB"


@dataclass
class ModelInfo:
    """Information about an installed Ollama model."""
    name: str
    size: int
    size_gb: str
    parameter_size: str
    quantization_level: str
    is_default: bool = False

    def to_display_string(self) -> str:
        """Format model information as a single-line summary."""
        return f"- {self.name} (size: {self.size_gb}, params: {self.parameter_size}, quant: {self.quantization_level})"


class OllamaClient:
    """Client for querying Ollama API and managing model prewarming."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        timeout: int = 30,
        configured_model: str = "",
    ) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.configured_model = configured_model.strip()
        self._last_error: Optional[str] = None

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Query /api/tags endpoint to retrieve raw model inventory dictionary."""
        self._last_error = None
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            models = data.get("models", [])
            return [m for m in models if isinstance(m, dict)]
        except requests.exceptions.ConnectionError as e:
            self._last_error = f"Unable to connect: {e}"
            return []
        except requests.exceptions.Timeout as e:
            self._last_error = f"Request timed out: {e}"
            return []
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else "Unknown"
            text = e.response.text if e.response is not None else str(e)
            self._last_error = f"HTTP {code} error: {text}"
            return []
        except Exception as e:
            self._last_error = f"Error: {e}"
            return []

    def resolve_model(self, requested_model: str = "") -> str:
        """Resolve model name following precedence rules."""
        if requested_model and requested_model.strip():
            return requested_model.strip()
        if self.configured_model:
            return self.configured_model
        models = self.get_available_models()
        return resolve_model(requested_model, available_models=models)

    def list_models(self) -> List[ModelInfo]:
        """Fetch and parse model list from Ollama into ModelInfo objects."""
        raw_models = self.get_available_models()
        if not raw_models:
            return []
        active_default = self.resolve_model()
        model_infos: List[ModelInfo] = []
        for m in raw_models:
            if not isinstance(m, dict):
                continue
            name = m.get("name", "unknown")
            size = m.get("size", 0)
            size_gb = format_bytes_to_gb(size)
            details = m.get("details")
            params = "unknown"
            quant = "unknown"
            if isinstance(details, dict):
                params = details.get("parameter_size") or "unknown"
                quant = details.get("quantization_level") or "unknown"
            is_default = (name == active_default)
            model_infos.append(
                ModelInfo(
                    name=name,
                    size=size,
                    size_gb=size_gb,
                    parameter_size=params,
                    quantization_level=quant,
                    is_default=is_default,
                )
            )
        return model_infos

    def format_inventory(self, models: Optional[List[ModelInfo]] = None) -> str:
        """Return human-readable inventory of models including active default model."""
        if models is None:
            models = self.list_models()
        if not models:
            if self._last_error:
                return f"Unable to connect to Ollama at {self.host}. {self._last_error}"
            return f"No models found or unable to connect to Ollama at {self.host}"
        active_default = self.resolve_model()
        lines = [
            f"Active Default Model: {active_default}",
            f"Installed Models ({len(models)}):",
        ]
        for m in models:
            lines.append(m.to_display_string())
        return "\n".join(lines)

    def prewarm_model(self, model: str = "") -> str:
        """Pin model into memory by issuing keep_alive: -1 POST request to /api/generate."""
        target_model = self.resolve_model(model)
        url = f"{self.host}/api/generate"
        payload = {
            "model": target_model,
            "keep_alive": -1,
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return f"Successfully pre-warmed model '{target_model}' into memory."
        except requests.exceptions.ConnectionError as e:
            return f"Error pre-warming model '{target_model}': Unable to connect to Ollama ({e})"
        except requests.exceptions.Timeout as e:
            return f"Error pre-warming model '{target_model}': Request timed out ({e})"
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else "Unknown"
            text = e.response.text if e.response is not None else str(e)
            return f"Error pre-warming model '{target_model}': HTTP {code} - {text}"
        except Exception as e:
            return f"Error pre-warming model '{target_model}': {e}"


def local_list_models(client: Optional[OllamaClient] = None) -> str:
    """Convenience helper to format inventory using default or supplied client."""
    c = client or OllamaClient()
    return c.format_inventory()


def local_prewarm_model(model: str = "", client: Optional[OllamaClient] = None) -> str:
    """Convenience helper to prewarm model using default or supplied client."""
    c = client or OllamaClient()
    return c.prewarm_model(model)
