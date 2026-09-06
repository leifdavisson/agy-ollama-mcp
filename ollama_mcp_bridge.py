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

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional
import requests

# FastMCP / MCPServer compatibility across MCP Python SDK versions (v1 & v2)
try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        try:
            from fastmcp import FastMCP
        except ImportError:
            sys.stderr.write(
                "Error: MCP Python SDK not found. Install via `pip install --user mcp requests`.\n"
            )
            sys.exit(1)

# Ensure logging writes exclusively to stderr so stdout is 100% clean for JSON-RPC
logging.basicConfig(
    level=logging.INFO if os.getenv("OLLAMA_MCP_DEBUG") else logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("ollama-mcp-bridge")

# Runtime Configurations
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
CONFIGURED_MODEL = os.getenv("LOCAL_LLM_MODEL", "").strip()
DEFAULT_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))
DEFAULT_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
DEFAULT_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))

mcp = FastMCP("local-ollama")


def get_available_models() -> List[Dict[str, Any]]:
    """Query Ollama API to retrieve list of installed models."""
    try:
        url = f"{OLLAMA_HOST}/api/tags"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return resp.json().get("models", [])
    except Exception as e:
        logger.warning(f"Unable to query Ollama models at {OLLAMA_HOST}: {e}")
        return []


def resolve_model(requested_model: str = "") -> str:
    """
    Resolve which model to use.
    Precedence:
    1. Explicit parameter passed to tool
    2. LOCAL_LLM_MODEL environment variable
    3. Auto-detected preferred local model
    4. Fallback default
    """
    if requested_model and requested_model.strip():
        return requested_model.strip()
    if CONFIGURED_MODEL:
        return CONFIGURED_MODEL

    available = get_available_models()
    names = [m.get("name", "") for m in available]

    preferred = [
        "qwen2.5-coder:14b",
        "qwen2.5-coder:7b",
        "deepseek-r1:14b",
        "deepseek-r1:8b",
        "dolphin3-tools:latest",
        "qwen3.5:latest",
        "qwen3.5:27b",
        "llama3.2:latest",
    ]
    for pref in preferred:
        if pref in names:
            return pref
    for name in names:
        if "coder" in name.lower() or "code" in name.lower():
            return name
    if names:
        return names[0]
    return "qwen2.5-coder:14b"


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
        return data.get("response", "")
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


@mcp.tool()
def local_draft_code(
    task_description: str,
    context: str = "",
    language: str = "",
    model: str = "",
) -> str:
    """Generate initial code drafts, boilerplate, unit tests, or scaffolding locally via Ollama without consuming cloud tokens."""
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


@mcp.tool()
def local_summarize_and_extract(
    content: str,
    extraction_goal: str,
    model: str = "",
) -> str:
    """Compress massive files, logs, terminal traces, or documentation into high-density summaries before cloud reasoning."""
    system = (
        "You are a dense technical extraction model. Remove all noise, boilerplate, repetitive lines, and timestamps. "
        "Extract only key architectural points, error messages, stack traces, schemas, or critical facts matching the goal."
    )
    prompt = f"Target Goal: {extraction_goal}\n\nContent to reduce:\n{content}"
    return query_ollama(prompt, system=system, model=model)


@mcp.tool()
def local_chunked_summary(
    content: str,
    extraction_goal: str = "Extract key technical points, errors, and relevant logic",
    chunk_chars: int = 12000,
    model: str = "",
) -> str:
    """Map-reduce chunked summarization for massive files or logs that exceed single context limits. Splits content into chunks, summarizes each locally, and synthesizes into a cohesive high-density brief."""
    if len(content) <= chunk_chars:
        return local_summarize_and_extract(content, extraction_goal, model=model)

    lines = content.splitlines(keepends=True)
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for line in lines:
        if current_len + len(line) > chunk_chars and current_chunk:
            chunks.append("".join(current_chunk))
            current_chunk = [line]
            current_len = len(line)
        else:
            current_chunk.append(line)
            current_len += len(line)
    if current_chunk:
        chunks.append("".join(current_chunk))

    summaries: List[str] = []
    for i, chunk in enumerate(chunks):
        sub_goal = f"Part {i+1}/{len(chunks)} of content. {extraction_goal}"
        sub_summary = local_summarize_and_extract(chunk, sub_goal, model=model)
        summaries.append(f"--- Chunk {i+1}/{len(chunks)} Summary ---\n{sub_summary}")

    if len(chunks) <= 1:
        return summaries[0] if summaries else ""

    combined = "\n\n".join(summaries)
    synthesis_prompt = (
        f"Consolidate and synthesize the following chunk summaries into a single cohesive, non-redundant brief.\n"
        f"Goal: {extraction_goal}\n\n"
        f"Chunk Summaries:\n{combined}"
    )
    system = "You are a master technical editor. Synthesize multi-part notes into a clean, concise, unified report."
    return query_ollama(synthesis_prompt, system=system, model=model)


@mcp.tool()
def local_extract_json(
    content: str,
    schema_description: str,
    model: str = "",
) -> str:
    """Extract structured JSON from unstructured text, logs, or documentation according to a target schema description."""
    system = (
        "You are a strict data extraction engine. Extract data from the input content and output ONLY a valid JSON object or array matching the requested schema. "
        "Do not include markdown conversational filler, explanations, or commentary outside the JSON block."
    )
    prompt = f"Target Schema / Fields:\n{schema_description}\n\nContent:\n{content}"
    return query_ollama(prompt, system=system, model=model, temperature=0.1)


@mcp.tool()
def local_list_models() -> str:
    """List all models currently installed and available in the local Ollama instance."""
    models = get_available_models()
    if not models:
        return f"No models found or unable to connect to Ollama at {OLLAMA_HOST}"
    results = []
    for m in models:
        name = m.get("name", "unknown")
        size_gb = round(m.get("size", 0) / (1024 ** 3), 2)
        details = m.get("details", {})
        quant = details.get("quantization_level", "unknown")
        params = details.get("parameter_size", "unknown")
        results.append(f"- {name} (size: {size_gb} GB, params: {params}, quant: {quant})")
    active = resolve_model()
    return f"Active Default Model: {active}\nInstalled Models ({len(models)}):\n" + "\n".join(results)


@mcp.tool()
def local_prewarm_model(model: str = "") -> str:
    """Pre-warm a local model into memory/VRAM with keep_alive=-1 so subsequent calls have zero cold-start delay."""
    target_model = resolve_model(model)
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": target_model,
        "keep_alive": -1,
    }
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        return f"Successfully pre-warmed model '{target_model}' into memory."
    except Exception as e:
        return f"Error pre-warming model '{target_model}': {str(e)}"


def main():
    """Run the MCP server over STDIO transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
