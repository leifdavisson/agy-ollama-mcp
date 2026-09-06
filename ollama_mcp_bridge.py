#!/usr/bin/env python3
"""
Ollama MCP Bridge for Google Antigravity CLI (agy).
Tiered Edge-Cloud Architecture: Local Models as Zero-Cost Semantic Pre-Filters.

Exposes local Ollama models via Model Context Protocol (MCP) to delegate
token-heavy preliminary operations (code drafting, massive log summaries,
AST extraction, schema inference) to local models without consuming cloud quota.

License: GNU AGPLv3
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path so modular package is always available
SRC_PATH = str(Path(__file__).resolve().parent / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from ollama_bridge.config import (
    DEFAULT_FALLBACK_MODEL,
    DEFAULT_NUM_CTX,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    OLLAMA_HOST,
    PREFERRED_MODELS,
    configure_logging,
    get_configured_model,
)
from ollama_bridge import resolution
from ollama_bridge.resolution import select_preferred_model
from ollama_bridge.client import (
    ModelInfo,
    OllamaClient,
    format_bytes_to_gb,
    local_list_models,
    local_prewarm_model,
)
from ollama_bridge.engine import partition_chunks
from ollama_bridge import engine
from ollama_bridge.server import mcp, main

CONFIGURED_MODEL: str = os.getenv("LOCAL_LLM_MODEL", "").strip()


def resolve_model(
    requested_model: str = "",
    available_models: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Resolve model taking into account module CONFIGURED_MODEL override."""
    explicit = requested_model.strip() if requested_model else ""
    if explicit:
        return explicit
    if CONFIGURED_MODEL:
        return CONFIGURED_MODEL
    return resolution.resolve_model(requested_model, available_models=available_models)


def query_ollama(
    prompt: str,
    system: str = "",
    model: str = "",
    temperature: Optional[float] = None,
    num_ctx: Optional[int] = None,
) -> str:
    return engine.query_ollama(prompt, system=system, model=model, temperature=temperature, num_ctx=num_ctx)


def local_draft_code(
    task_description: str,
    context: str = "",
    language: str = "",
    model: str = "",
) -> str:
    system = (
        "You are an expert software engineer. Generate clean, modular, production-ready code based on instructions. "
        "Adhere to best practices, robust type annotations, and idiomatic conventions. "
        "Provide code directly with concise comments where essential."
    )
    parts = []
    if language:
        parts.append(f"Target Language/Framework: {language}")
    if context:
        parts.append(f"Context / Existing Code:\n{context}")
    parts.append(f"Task Description:\n{task_description}")
    prompt = "\n\n".join(parts)
    return query_ollama(prompt, system=system, model=model)


def local_summarize_and_extract(
    content: str,
    extraction_goal: str,
    model: str = "",
) -> str:
    system = (
        "You are a dense technical extraction model. Remove all noise, boilerplate, repetitive lines, and timestamps. "
        "Extract only key architectural points, error messages, stack traces, schemas, or critical facts matching the goal."
    )
    prompt = f"Target Goal: {extraction_goal}\n\nContent to reduce:\n{content}"
    return query_ollama(prompt, system=system, model=model)


def local_chunked_summary(
    content: str,
    extraction_goal: str = "Extract key technical points, errors, and relevant logic",
    chunk_chars: int = 12000,
    model: str = "",
) -> str:
    if len(content) <= chunk_chars:
        return local_summarize_and_extract(content, extraction_goal, model=model)
    chunks = partition_chunks(content, chunk_chars)
    if not chunks:
        return ""
    summaries = []
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        sub_goal = f"Part {i+1}/{total} of content. {extraction_goal}"
        sub_summary = local_summarize_and_extract(chunk, sub_goal, model=model)
        summaries.append(f"--- Chunk {i+1}/{total} Summary ---\n{sub_summary}")
    if len(chunks) <= 1:
        return summaries[0]
    combined = "\n\n".join(summaries)
    synthesis_prompt = (
        f"Consolidate and synthesize the following chunk summaries into a single cohesive, non-redundant brief.\n"
        f"Goal: {extraction_goal}\n\n"
        f"Chunk Summaries:\n{combined}"
    )
    system = "You are a master technical editor. Synthesize multi-part notes into a clean, concise, unified report."
    return query_ollama(synthesis_prompt, system=system, model=model)


def local_extract_json(
    content: str,
    schema_description: str,
    model: str = "",
) -> str:
    system = (
        "You are a strict data extraction engine. Extract data from the input content and output ONLY a valid JSON object or array matching the requested schema. "
        "Do not include markdown conversational filler, explanations, or commentary outside the JSON block."
    )
    prompt = f"Target Schema / Fields:\n{schema_description}\n\nContent:\n{content}"
    return query_ollama(prompt, system=system, model=model, temperature=0.1)


def get_available_models():
    """Retrieve raw available models dictionary list from default client."""
    client = OllamaClient(host=OLLAMA_HOST, timeout=5)
    return client.get_available_models()


def chunk_lines_overlap(lines: List[str], chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Slice lines into overlapping chunks preserving line boundaries."""
    return engine.chunk_lines_overlap(lines, chunk_size=chunk_size, overlap=overlap)


def local_map_reduce_file(
    file_path: str,
    extraction_goal: str,
    chunk_size: int = 400,
    overlap: int = 50,
    concurrency: int = 2,
    model: str = "",
) -> str:
    """Compress large files, traces, or logs using local Ollama Map-Reduce before ingesting into context."""
    return engine.local_map_reduce_file(
        file_path=file_path,
        extraction_goal=extraction_goal,
        chunk_size=chunk_size,
        overlap=overlap,
        concurrency=concurrency,
        model=model,
    )


if __name__ == "__main__":
    main()
