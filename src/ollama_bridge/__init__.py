"""
Ollama MCP Bridge Package.
Tiered Edge-Cloud Architecture: Local Models as Zero-Cost Semantic Pre-Filters.
License: GNU AGPLv3
"""
from .config import (
    CONFIGURED_MODEL,
    DEFAULT_FALLBACK_MODEL,
    DEFAULT_NUM_CTX,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    OLLAMA_HOST,
    PREFERRED_MODELS,
    configure_logging,
    get_configured_model,
)
from .resolution import resolve_model, select_preferred_model
from .client import (
    ModelInfo,
    OllamaClient,
    format_bytes_to_gb,
    local_list_models,
    local_prewarm_model,
)
from .engine import (
    chunk_lines_overlap,
    local_chunked_summary,
    local_draft_code,
    local_extract_json,
    local_map_reduce_file,
    local_summarize_and_extract,
    map_worker,
    partition_chunks,
    query_ollama,
)
from .server import mcp, main

__all__ = [
    "CONFIGURED_MODEL",
    "DEFAULT_FALLBACK_MODEL",
    "DEFAULT_NUM_CTX",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT",
    "OLLAMA_HOST",
    "PREFERRED_MODELS",
    "configure_logging",
    "get_configured_model",
    "resolve_model",
    "select_preferred_model",
    "ModelInfo",
    "OllamaClient",
    "format_bytes_to_gb",
    "local_list_models",
    "local_prewarm_model",
    "chunk_lines_overlap",
    "local_chunked_summary",
    "local_draft_code",
    "local_extract_json",
    "local_map_reduce_file",
    "local_summarize_and_extract",
    "map_worker",
    "partition_chunks",
    "query_ollama",
    "mcp",
    "main",
]
