"""
Unit and Integration tests for Ollama MCP Bridge.
Tests cover model resolution, query formatting, tool definitions,
and JSON-RPC STDIO transport.
License: GNU AGPLv3
"""
import json
import os
import subprocess
import sys
from typing import Any, Callable
from unittest.mock import patch, MagicMock
import pytest

# Ensure parent directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ollama_mcp_bridge as bridge


def verifies(req_id: str) -> Callable[[Any], Any]:
    """Decorator linking test to functional requirement ID for AST RTM parser."""
    def decorator(fn: Any) -> Any:
        if not hasattr(fn, "_verifies"):
            fn._verifies = []
        fn._verifies.append(req_id)
        setattr(fn, "__req_id__", req_id)
        return fn
    return decorator


@verifies("REQ-001")
def test_resolve_model_explicit():
    """Explicit requested model should always take precedence."""
    resolved = bridge.resolve_model("custom-model:latest")
    assert resolved == "custom-model:latest"


@verifies("REQ-001")
def test_resolve_model_from_env(monkeypatch):
    """LOCAL_LLM_MODEL env var should be respected if set."""
    monkeypatch.setattr(bridge, "CONFIGURED_MODEL", "env-model:7b")
    resolved = bridge.resolve_model("")
    assert resolved == "env-model:7b"


@verifies("REQ-001")
def test_resolve_model_autodetect(monkeypatch):
    """Auto-detect should pick coding models from available tags."""
    monkeypatch.setattr(bridge, "CONFIGURED_MODEL", "")
    mock_models = [
        {"name": "llama3.2:latest"},
        {"name": "qwen2.5-coder:14b"},
    ]
    with patch.object(bridge, "get_available_models", return_value=mock_models):
        resolved = bridge.resolve_model("")
        assert resolved == "qwen2.5-coder:14b"


@verifies("REQ-002")
def test_query_ollama_success():
    """Test successful query response handling."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "def add(a, b): return a + b"}
    mock_resp.raise_for_status.return_value = None

    with patch("requests.post", return_value=mock_resp):
        res = bridge.query_ollama("Write an add function", model="qwen2.5-coder:14b")
        assert "def add" in res


@verifies("REQ-002")
def test_query_ollama_connection_error():
    """Test graceful handling of Ollama connection errors."""
    import requests
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        res = bridge.query_ollama("Hello")
        assert "Unable to connect to Ollama" in res


@verifies("REQ-002")
def test_local_draft_code_formatting():
    """Test prompt assembly in local_draft_code."""
    with patch.object(bridge, "query_ollama", return_value="class User: pass") as mock_query:
        result = bridge.local_draft_code(
            task_description="Create User class",
            context="import pydantic",
            language="python",
        )
        assert result == "class User: pass"
        args, kwargs = mock_query.call_args
        prompt = args[0]
        assert "Target Language/Framework: python" in prompt
        assert "import pydantic" in prompt
        assert "Create User class" in prompt


@verifies("REQ-003")
def test_local_summarize_and_extract_formatting():
    """Test prompt assembly in local_summarize_and_extract."""
    with patch.object(bridge, "query_ollama", return_value="Timeout at line 42") as mock_query:
        result = bridge.local_summarize_and_extract(
            content="Full log file content",
            extraction_goal="Find timeout line",
        )
        assert result == "Timeout at line 42"
        args, kwargs = mock_query.call_args
        prompt = args[0]
        assert "Target Goal: Find timeout line" in prompt
        assert "Full log file content" in prompt


@verifies("REQ-005")
def test_local_extract_json():
    """Test local_extract_json system prompt and temperature."""
    with patch.object(bridge, "query_ollama", return_value='{"status": "ok"}') as mock_query:
        res = bridge.local_extract_json("raw text", "{status: string}")
        assert res == '{"status": "ok"}'
        assert mock_query.call_args[1].get("temperature") == 0.1


@verifies("REQ-004")
def test_local_chunked_summary_small_content():
    """Small content under chunk_chars should call local_summarize_and_extract directly."""
    with patch.object(bridge, "local_summarize_and_extract", return_value="Short summary") as mock_sub:
        res = bridge.local_chunked_summary("short content", chunk_chars=1000)
        assert res == "Short summary"
        assert mock_sub.call_count == 1


@verifies("REQ-004")
def test_local_chunked_summary_large_content():
    """Large content exceeding chunk_chars should chunk and synthesize."""
    large_content = ("Line of log data that is repeating.\n" * 100)
    with patch.object(bridge, "local_summarize_and_extract", return_value="Chunk summary"), \
         patch.object(bridge, "query_ollama", return_value="Final synthesized summary"):
        res = bridge.local_chunked_summary(large_content, chunk_chars=200)
        assert res == "Final synthesized summary"


@verifies("REQ-008")
def test_mcp_stdio_handshake_and_tools():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    bridge_path = os.path.join(repo_root, "ollama_mcp_bridge.py")
    env = dict(os.environ)
    src_dir = os.path.join(repo_root, "src")
    root_dir = repo_root
    env["PYTHONPATH"] = f"{src_dir}:{root_dir}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.Popen(
        [sys.executable, bridge_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=root_dir,
    )

    try:
        # 1. Initialize
        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "pytest-client", "version": "1.0"},
            },
        }
        proc.stdin.write(json.dumps(init_req) + "\n")
        proc.stdin.flush()
        init_res = json.loads(proc.stdout.readline())
        assert init_res.get("jsonrpc") == "2.0"
        assert init_res["result"]["serverInfo"]["name"] == "local-ollama"

        # 2. Initialized Notification
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}) + "\n")
        proc.stdin.flush()

        # 3. List Tools
        tools_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        proc.stdin.write(json.dumps(tools_req) + "\n")
        proc.stdin.flush()
        tools_res = json.loads(proc.stdout.readline())
        tools = [t["name"] for t in tools_res["result"]["tools"]]

        expected_tools = [
            "local_draft_code",
            "local_summarize_and_extract",
            "local_chunked_summary",
            "local_extract_json",
            "local_list_models",
            "local_prewarm_model",
        ]
        for t in expected_tools:
            assert t in tools
    finally:
        proc.terminate()
        proc.wait(timeout=5)
