"""
Ollama MCP Bridge - Engine Module.
Prompt synthesis, code drafting, dense summarization, line-aware chunking,
and structured JSON extraction.

Target Implementation Path: src/ollama_bridge/engine.py
License: GNU AGPLv3
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import requests

from .config import (
    DEFAULT_NUM_CTX,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    OLLAMA_HOST,
)
from .resolution import resolve_model

logger = logging.getLogger("ollama_bridge.engine")


def query_ollama(
    prompt: str,
    system: str = "",
    model: str = "",
    temperature: Optional[float] = None,
    num_ctx: Optional[int] = None,
) -> str:
    """Execute generate query against local Ollama daemon."""
    active_model = resolve_model(model)
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": active_model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": DEFAULT_TEMPERATURE if temperature is None else temperature,
            "num_ctx": DEFAULT_NUM_CTX if num_ctx is None else num_ctx,
        },
    }
    logger.info(f"Querying Ollama: model={active_model}, prompt_len={len(prompt)}")
    try:
        response = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return str(data.get("response", ""))
    except requests.exceptions.ConnectionError:
        return (
            f"Error: Unable to connect to Ollama at {OLLAMA_HOST}. "
            "Please ensure Ollama is running (`ollama serve` or `systemctl status ollama`)."
        )
    except requests.exceptions.Timeout:
        return f"Error: Request to Ollama timed out after {DEFAULT_TIMEOUT}s for model '{active_model}'."
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else "Unknown"
        text = e.response.text if e.response is not None else str(e)
        return f"Error from Ollama ({status_code}): {text}"
    except Exception as e:
        return f"Unexpected error during Ollama query: {str(e)}"


def partition_chunks(content: str, chunk_chars: int) -> List[str]:
    """
    Partition content into chunks on line boundaries without splitting lines.
    REQ-004: Invariant "".join(chunks) == content.
    McCabe Complexity M <= 4.
    """
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")

    if not content:
        return []

    raw_lines = content.split("\n")
    lines = [l + "\n" for l in raw_lines[:-1]]
    if raw_lines[-1]:
        lines.append(raw_lines[-1])

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for line in lines:
        if (current_len + len(line) > chunk_chars) and bool(current_chunk):
            chunks.append("".join(current_chunk))
            current_chunk = [line]
            current_len = len(line)
        else:
            current_chunk.append(line)
            current_len += len(line)

    chunks.append("".join(current_chunk))
    return chunks


def local_draft_code(
    task_description: str,
    context: str = "",
    language: str = "",
    model: str = "",
) -> str:
    """
    Draft code implementation locally via Ollama.
    REQ-002: Formats prompt with language, context, task; enforces clean, modular production code.
    McCabe Complexity M <= 3.
    """
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
    return query_ollama(prompt=prompt, system=system, model=model)


def local_summarize_and_extract(
    content: str,
    extraction_goal: str,
    model: str = "",
) -> str:
    """
    Compress raw text/logs into high-density technical summary stripping noise and boilerplate.
    REQ-003: Instructs model to remove noise, boilerplate, repetitive lines, timestamps.
    """
    system = (
        "You are a dense technical extraction model. Remove all noise, boilerplate, repetitive lines, and timestamps. "
        "Extract only key architectural points, error messages, stack traces, schemas, or critical facts matching the goal."
    )
    prompt = f"Target Goal: {extraction_goal}\n\nContent to reduce:\n{content}"
    return query_ollama(prompt=prompt, system=system, model=model)


def local_chunked_summary(
    content: str,
    extraction_goal: str = "Extract key technical points, errors, and relevant logic",
    chunk_chars: int = 12000,
    model: str = "",
) -> str:
    """
    Map-reduce chunked summarization for massive content.
    REQ-004: Single-pass if len <= chunk_chars; multi-pass partitioned map-reduce if > chunk_chars.
    McCabe Complexity M <= 4.
    """
    if len(content) <= chunk_chars:
        return local_summarize_and_extract(content, extraction_goal, model=model)

    chunks = partition_chunks(content, chunk_chars)
    if not chunks:
        return ""

    summaries: List[str] = []
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
    """
    Extract structured JSON matching a schema description.
    REQ-005: Clamped temperature 0.1 and strict JSON output instructions.
    """
    system = (
        "You are a strict data extraction engine. Extract data from the input content and output ONLY a valid JSON object or array matching the requested schema. "
        "Do not include markdown conversational filler, explanations, or commentary outside the JSON block."
    )
    prompt = f"Target Schema / Fields:\n{schema_description}\n\nContent:\n{content}"
    return query_ollama(prompt=prompt, system=system, model=model, temperature=0.1)
