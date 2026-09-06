"""
Ollama MCP Bridge - Server Module.
Exposes MCP tools via FastMCP / MCPServer over STDIO transport.
License: GNU AGPLv3
"""
from __future__ import annotations

import logging
import sys
from typing import Optional

from mcp.server.mcpserver import MCPServer as FastMCP

from .config import configure_logging
from .client import local_list_models as client_list_models, local_prewarm_model as client_prewarm_model
from .engine import (
    local_draft_code as engine_draft_code,
    local_summarize_and_extract as engine_summarize,
    local_chunked_summary as engine_chunked_summary,
    local_extract_json as engine_extract_json,
    local_map_reduce_file as engine_map_reduce_file,
)

logger = configure_logging()

mcp = FastMCP("local-ollama")


@mcp.tool()
def local_draft_code(
    task_description: str,
    context: str = "",
    language: str = "",
    model: str = "",
) -> str:
    """Generate initial code drafts, boilerplate, unit tests, or scaffolding locally via Ollama without consuming cloud tokens."""
    return engine_draft_code(
        task_description=task_description,
        context=context,
        language=language,
        model=model,
    )


@mcp.tool()
def local_summarize_and_extract(
    content: str,
    extraction_goal: str,
    model: str = "",
) -> str:
    """Compress massive files, logs, terminal traces, or documentation into high-density summaries before cloud reasoning."""
    return engine_summarize(
        content=content,
        extraction_goal=extraction_goal,
        model=model,
    )


@mcp.tool()
def local_chunked_summary(
    content: str,
    extraction_goal: str = "Extract key technical points, errors, and relevant logic",
    chunk_chars: int = 12000,
    model: str = "",
) -> str:
    """Map-reduce chunked summarization for massive files or logs that exceed single context limits."""
    return engine_chunked_summary(
        content=content,
        extraction_goal=extraction_goal,
        chunk_chars=chunk_chars,
        model=model,
    )


@mcp.tool()
def local_extract_json(
    content: str,
    schema_description: str,
    model: str = "",
) -> str:
    """Extract structured JSON from unstructured text, logs, or documentation according to a target schema description."""
    return engine_extract_json(
        content=content,
        schema_description=schema_description,
        model=model,
    )


@mcp.tool()
def local_list_models() -> str:
    """List all models currently installed and available in the local Ollama instance."""
    return client_list_models()


@mcp.tool()
def local_prewarm_model(model: str = "") -> str:
    """Pre-warm a local model into memory/VRAM with keep_alive=-1 so subsequent calls have zero cold-start delay."""
    return client_prewarm_model(model=model)


@mcp.tool()
def local_map_reduce_file(
    file_path: str,
    extraction_goal: str,
    chunk_size: int = 400,
    overlap: int = 50,
    concurrency: int = 2,
    model: str = "",
) -> str:
    """Compress large files, traces, or logs using local Ollama Map-Reduce before ingesting into context."""
    return engine_map_reduce_file(
        file_path=file_path,
        extraction_goal=extraction_goal,
        chunk_size=chunk_size,
        overlap=overlap,
        concurrency=concurrency,
        model=model,
    )


def main() -> None:
    """Run the MCP server over STDIO transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
