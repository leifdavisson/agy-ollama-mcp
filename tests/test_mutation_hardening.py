"""
Comprehensive Mutation Hardening Tests.
Kills surviving mutants by asserting exact payloads, default arguments,
string templates, formatters, boundaries, and exception contracts.

Specifications:
- REQ-001: Model Selection & Precedence Hierarchy
- REQ-002: Local Code Drafting Execution
- REQ-003: Dense Summarization & Noise Extraction
- REQ-004: Chunked Map-Reduce Summarization
- REQ-005: Strict Structured JSON Extraction
- REQ-006: Model Inventory & Capabilities
- REQ-007: Speculative Model Prewarming
- REQ-008: Protocol Conformance & Stdio Isolation

License: GNU AGPLv3
"""
import logging
import os
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, call, patch
import pytest
import requests

# Ensure src is in sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(repo_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from ollama_bridge import client, config, engine, resolution, server
from ollama_bridge.client import (
    ModelInfo,
    OllamaClient,
    format_bytes_to_gb,
    local_list_models,
    local_prewarm_model,
)
from ollama_bridge.config import (
    DEFAULT_FALLBACK_MODEL,
    DEFAULT_NUM_CTX,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    OLLAMA_HOST,
    PREFERRED_MODELS,
    configure_logging,
)
from ollama_bridge.engine import (
    local_chunked_summary,
    local_draft_code,
    local_extract_json,
    local_summarize_and_extract,
    partition_chunks,
    query_ollama,
)
from ollama_bridge.resolution import (
    _fetch_tags_names,
    get_configured_model,
    resolve_model,
    select_preferred_model,
)


def verifies(req_id: str):
    def decorator(fn):
        fn.verified_requirement = req_id
        fn = pytest.mark.requirement(req_id)(fn)
        return pytest.mark.verifies(req_id)(fn)
    return decorator


# ==============================================================================
# REQ-008: Config & Logging Mutation Hardening
# ==============================================================================

@verifies("REQ-008")
def test_config_logging_attributes_and_defaults():
    """Kill logging mutants: name, level, handlers, formatter, stream, propagate."""
    # Test with default argument
    logger_default = configure_logging()
    assert logger_default.name == "ollama_bridge"
    assert logger_default.level == logging.WARNING
    assert logger_default.propagate is False
    assert len(logger_default.handlers) == 1

    handler = logger_default.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert handler.stream == sys.stderr
    assert handler.level == logging.WARNING
    assert handler.formatter is not None

    # Test formatter format output
    record = logging.LogRecord(
        name="ollama_bridge",
        level=logging.WARNING,
        pathname="test.py",
        lineno=1,
        msg="test message",
        args=(),
        exc_info=None,
    )
    formatted = handler.formatter.format(record)
    assert "[WARNING] ollama_bridge: test message" in formatted

    # Test with explicit level
    logger_custom = configure_logging(level=logging.DEBUG)
    assert logger_custom.level == logging.DEBUG
    assert logger_custom.handlers[0].level == logging.DEBUG


# ==============================================================================
# REQ-001: Resolution Mutation Hardening
# ==============================================================================

@verifies("REQ-001")
def test_resolution_fetch_tags_names_success_and_edge_cases():
    """Kill mutants in _fetch_tags_names: URL, timeout, status_code, JSON parsing."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "models": [
            {"name": "qwen2.5-coder:14b"},
            {"name": "deepseek-r1:14b"},
            {"not_name": "unknown"},
            "invalid_non_dict_element",
        ]
    }
    mock_resp.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_resp) as mock_get:
        names = _fetch_tags_names()
        mock_get.assert_called_once_with(f"{OLLAMA_HOST}/api/tags", timeout=5)
        mock_resp.raise_for_status.assert_called_once()
        assert names == ["qwen2.5-coder:14b", "deepseek-r1:14b", ""]

    # Test when models key is empty or missing
    mock_resp.json.return_value = {}
    with patch("requests.get", return_value=mock_resp):
        assert _fetch_tags_names() == []

    # Test when requests.get raises exception
    with patch("requests.get", side_effect=requests.exceptions.RequestException("boom")):
        with patch.object(resolution.logger, "warning") as mock_warn:
            assert _fetch_tags_names() == []
            mock_warn.assert_called_once()
            assert "Failed to query Ollama tags" in mock_warn.call_args[0][0]


@verifies("REQ-001")
def test_resolution_select_preferred_model_filtering():
    """Kill mutants in select_preferred_model: whitespace and candidate matching."""
    # Only empty/whitespace strings
    assert select_preferred_model(["", "   ", "\t\n"]) is None

    # Preferred order adherence
    available = ["llama3.2:latest", "qwen2.5-coder:14b", "deepseek-r1:14b"]
    assert select_preferred_model(available) == "qwen2.5-coder:14b"

    # Match lowercasing
    assert select_preferred_model(["  QWEN2.5-CODER:7B  "]) == "QWEN2.5-CODER:7B"

    # Unrecognized model returns first valid
    assert select_preferred_model(["  custom-alpha:latest  "]) == "custom-alpha:latest"


@verifies("REQ-001")
def test_resolution_resolve_model_defaults_and_whitespace():
    """Kill mutants in resolve_model: default argument, empty string fallthrough."""
    # Test resolve_model default arguments
    with patch.dict("os.environ", {}, clear=True):
        with patch("ollama_bridge.resolution._fetch_tags_names", return_value=[]):
            resolved = resolve_model()
            assert resolved == DEFAULT_FALLBACK_MODEL
            assert resolved == "qwen2.5-coder:14b"
            assert not resolved.startswith("XXXX")

    # Whitespace only requested_model must fall through
    with patch.dict("os.environ", {"LOCAL_LLM_MODEL": "env-model"}):
        assert resolve_model("   ") == "env-model"

    # Available models with non-dict and missing name
    avail: List[Dict[str, Any]] = [
        {"name": "deepseek-r1:14b"},
        {"not_name": "x"},
    ]
    with patch.dict("os.environ", {}, clear=True):
        assert resolve_model("", available_models=avail) == "deepseek-r1:14b"


# ==============================================================================
# REQ-006 & REQ-007: Client Mutation Hardening
# ==============================================================================

@verifies("REQ-006")
def test_client_init_and_default_attributes():
    """Kill mutants in OllamaClient.__init__: host, timeout, configured_model, _last_error."""
    c_default = OllamaClient()
    assert c_default.host == "http://localhost:11434"
    assert c_default.timeout == 30
    assert c_default.configured_model == ""
    assert c_default._last_error is None

    # Custom attributes with trailing slash and spaces
    c_custom = OllamaClient(
        host="http://custom-host:11434/",
        timeout=45,
        configured_model="  my-model  ",
    )
    assert c_custom.host == "http://custom-host:11434"
    assert c_custom.timeout == 45
    assert c_custom.configured_model == "my-model"
    assert c_custom._last_error is None


@verifies("REQ-006")
def test_client_format_bytes_to_gb_exact_values():
    """Kill mutants in format_bytes_to_gb: rounding, power of 1024, suffix."""
    assert format_bytes_to_gb(0) == "0.00 GB"
    assert format_bytes_to_gb(1024 * 1024 * 1024) == "1.00 GB"
    assert format_bytes_to_gb(2 * 1024 * 1024 * 1024) == "2.00 GB"
    assert format_bytes_to_gb(8988112209) == "8.37 GB"
    assert format_bytes_to_gb(1555000000) == "1.45 GB"


@verifies("REQ-006")
def test_client_get_available_models_call_contract():
    """Kill mutants in get_available_models: exact URL, timeout, and state."""
    c = OllamaClient(host="http://test-host:11434", timeout=12)
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"models": [{"name": "mod1"}]}
    mock_resp.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_resp) as mock_get:
        res = c.get_available_models()
        mock_get.assert_called_once_with("http://test-host:11434/api/tags", timeout=12)
        mock_resp.raise_for_status.assert_called_once()
        assert res == [{"name": "mod1"}]
        assert c._last_error is None


@verifies("REQ-006")
def test_client_list_models_dataclass_fields():
    """Kill mutants in list_models: ModelInfo field assignments and default detection."""
    c = OllamaClient(configured_model="target:model")
    raw_models = [
        {
            "name": "target:model",
            "size": 8988112209,
            "details": {
                "parameter_size": "14B",
                "quantization_level": "Q4_K_M",
            },
            "modified_at": "2026-03-01T12:00:00Z",
        },
        {
            "name": "other:model",
            "size": 4294967296,
            "details": {
                "parameter_size": "7B",
                "quantization_level": "Q8_0",
            },
            "modified_at": "2026-03-02T12:00:00Z",
        },
    ]

    with patch.object(c, "get_available_models", return_value=raw_models):
        models = c.list_models()
        assert len(models) == 2

        m0 = models[0]
        assert m0.name == "target:model"
        assert m0.size == 8988112209
        assert m0.size_gb == "8.37 GB"
        assert m0.parameter_size == "14B"
        assert m0.quantization_level == "Q4_K_M"
        assert m0.is_default is True

        m1 = models[1]
        assert m1.name == "other:model"
        assert m1.size == 4294967296
        assert m1.size_gb == "4.00 GB"
        assert m1.parameter_size == "7B"
        assert m1.quantization_level == "Q8_0"
        assert m1.is_default is False


@verifies("REQ-007")
def test_client_prewarm_model_call_contract():
    """Kill mutants in prewarm_model: URL, keep_alive=-1, payload, timeout, output text."""
    c = OllamaClient(host="http://test-host:11434", timeout=20)
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None

    with patch("requests.post", return_value=mock_resp) as mock_post:
        msg = c.prewarm_model("  deepseek-r1:14b  ")
        mock_post.assert_called_once_with(
            "http://test-host:11434/api/generate",
            json={"model": "deepseek-r1:14b", "keep_alive": -1},
            timeout=20,
        )
        assert msg == "Successfully pre-warmed model 'deepseek-r1:14b' into memory."


# ==============================================================================
# REQ-002, REQ-003, REQ-004, REQ-005: Engine Mutation Hardening
# ==============================================================================

@verifies("REQ-002")
def test_engine_query_ollama_exact_payload_and_defaults():
    """Kill mutants in query_ollama: payload structure, options, timeout, defaults."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "mock generated code"}
    mock_resp.raise_for_status.return_value = None

    # Test with all explicit parameters
    with patch("requests.post", return_value=mock_resp) as mock_post:
        res = query_ollama(
            prompt="my_prompt",
            system="my_system",
            model="custom:14b",
            temperature=0.7,
            num_ctx=8192,
        )
        assert res == "mock generated code"
        mock_post.assert_called_once_with(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": "custom:14b",
                "prompt": "my_prompt",
                "system": "my_system",
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_ctx": 8192,
                },
            },
            timeout=DEFAULT_TIMEOUT,
        )

    # Test with default parameters
    with patch("requests.post", return_value=mock_resp) as mock_post:
        query_ollama(prompt="default_prompt")
        called_payload = mock_post.call_args[1]["json"]
        assert called_payload["system"] == ""
        assert called_payload["stream"] is False
        assert called_payload["options"]["temperature"] == DEFAULT_TEMPERATURE
        assert called_payload["options"]["temperature"] == 0.2
        assert called_payload["options"]["num_ctx"] == DEFAULT_NUM_CTX
        assert called_payload["options"]["num_ctx"] == 16384


@verifies("REQ-002")
def test_engine_local_draft_code_prompt_and_system_exact():
    """Kill mutants in local_draft_code: prompt assembly, default args, system prompt."""
    expected_system = (
        "You are an expert software engineer. Generate clean, modular, production-ready code based on instructions. "
        "Adhere to best practices, robust type annotations, and idiomatic conventions. "
        "Provide code directly with concise comments where essential."
    )

    # Test task only (default args context="", language="", model="")
    with patch("ollama_bridge.engine.query_ollama", return_value="def foo(): pass") as mock_q:
        code = local_draft_code(task_description="Build an AGPLv3 CLI")
        assert code == "def foo(): pass"
        mock_q.assert_called_once()
        kwargs = mock_q.call_args[1]
        assert kwargs["model"] == ""
        assert kwargs["system"] == expected_system
        prompt = kwargs["prompt"]
        assert "Target Language/Framework" not in prompt
        assert "Context / Existing Code" not in prompt
        assert prompt == "Task Description:\nBuild an AGPLv3 CLI"

    # Test with context and language
    with patch("ollama_bridge.engine.query_ollama", return_value="class Bar: pass") as mock_q:
        local_draft_code(
            task_description="Refactor Bar",
            context="class Bar:\n    pass",
            language="python",
            model="qwen2.5-coder:14b",
        )
        kwargs = mock_q.call_args[1]
        assert kwargs["model"] == "qwen2.5-coder:14b"
        prompt = kwargs["prompt"]
        assert prompt == "Target Language/Framework: python\n\nContext / Existing Code:\nclass Bar:\n    pass\n\nTask Description:\nRefactor Bar"


@verifies("REQ-003")
def test_engine_local_summarize_and_extract_exact():
    """Kill mutants in local_summarize_and_extract: system prompt, extraction prompt."""
    expected_system = (
        "You are a dense technical extraction model. Remove all noise, boilerplate, repetitive lines, and timestamps. "
        "Extract only key architectural points, error messages, stack traces, schemas, or critical facts matching the goal."
    )

    with patch("ollama_bridge.engine.query_ollama", return_value="Extracted facts") as mock_q:
        res = local_summarize_and_extract(
            content="Error log line 1\nError log line 2",
            extraction_goal="Find root cause",
            model="custom:summarizer",
        )
        assert res == "Extracted facts"
        mock_q.assert_called_once()
        kwargs = mock_q.call_args[1]
        assert kwargs["model"] == "custom:summarizer"
        assert kwargs["system"] == expected_system
        assert kwargs["prompt"] == "Target Goal: Find root cause\n\nContent to reduce:\nError log line 1\nError log line 2"


@verifies("REQ-004")
def test_engine_partition_chunks_multiline_join_and_boundary():
    """Kill mutant _partition_chunks__mutmut_25 (\"XXXX\".join) and chunk size validation."""
    lines = ["chunk1_line1\n", "chunk1_line2\n", "chunk2_line1\n", "chunk2_line2\n"]
    content = "".join(lines)
    chunks = partition_chunks(content, chunk_chars=26)
    assert len(chunks) == 2
    assert chunks[0] == "chunk1_line1\nchunk1_line2\n"
    assert chunks[1] == "chunk2_line1\nchunk2_line2\n"
    assert "".join(chunks) == content
    assert "XXXX" not in chunks[0]
    assert "XXXX" not in chunks[1]

    with pytest.raises(ValueError, match="chunk_chars must be positive"):
        partition_chunks("some text", 0)

    with pytest.raises(ValueError, match="chunk_chars must be positive"):
        partition_chunks("some text", -5)


@verifies("REQ-004")
def test_engine_local_chunked_summary_map_reduce_exact():
    """Kill mutants in local_chunked_summary: synthesis prompt, headers, system prompt."""
    content = "Line1: data\n" * 100
    chunk_chars = 200

    expected_synthesis_system = (
        "You are a master technical editor. Synthesize multi-part notes into a clean, concise, unified report."
    )

    with patch("ollama_bridge.engine.local_summarize_and_extract") as mock_extract, \
         patch("ollama_bridge.engine.query_ollama") as mock_query:
        mock_extract.side_effect = lambda c, g, model="": f"Summary for chunk len {len(c)}"
        mock_query.return_value = "Unified Map-Reduce Summary"

        res = local_chunked_summary(
            content=content,
            extraction_goal="Consolidate system events",
            chunk_chars=chunk_chars,
            model="qwen:synthesis",
        )
        assert res == "Unified Map-Reduce Summary"
        mock_query.assert_called_once()
        kwargs = mock_query.call_args[1]
        assert kwargs["model"] == "qwen:synthesis"
        assert kwargs["system"] == expected_synthesis_system

        prompt = mock_query.call_args[0][0] if mock_query.call_args[0] else kwargs.get("prompt", "")
        assert "Consolidate and synthesize the following chunk summaries into a single cohesive, non-redundant brief." in prompt
        assert "Goal: Consolidate system events" in prompt
        assert "--- Chunk 1/" in prompt


@verifies("REQ-005")
def test_engine_local_extract_json_exact_prompts_and_markdown_stripping():
    """Kill mutants in local_extract_json: system prompt, schema formatting, code fence stripping."""
    expected_system = (
        "You are a strict data extraction engine. Extract data from the input content and output ONLY a valid JSON object or array matching the requested schema. "
        "Do not include markdown conversational filler, explanations, or commentary outside the JSON block."
    )

    with patch("ollama_bridge.engine.query_ollama", return_value='{"name": "test"}') as mock_q:
        res = local_extract_json("raw content", schema_description='{"type": "object"}', model="custom:json")
        assert res == '{"name": "test"}'
        kwargs = mock_q.call_args[1]
        assert kwargs["model"] == "custom:json"
        assert kwargs["system"] == expected_system
        assert kwargs["temperature"] == 0.1
        assert kwargs["prompt"] == 'Target Schema / Fields:\n{"type": "object"}\n\nContent:\nraw content'


# ==============================================================================
# Additional Specific Targeted Error & String Mutants
# ==============================================================================

@verifies("REQ-001")
def test_resolution_coder_keyword_fallback():
    """Kill mutants in coder keyword selection."""
    assert select_preferred_model(["starcoder:7b", "mistral:7b"]) == "starcoder:7b"
    assert select_preferred_model(["magic-coder:latest"]) == "magic-coder:latest"


@verifies("REQ-006")
def test_client_get_available_models_error_messages():
    """Kill mutants in OllamaClient error message formatting."""
    c = OllamaClient(host="http://test-host:11434///")
    assert c.host == "http://test-host:11434"

    # ConnectionError
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("conn fail")):
        res = c.get_available_models()
        assert res == []
        assert c._last_error == "Unable to connect: conn fail"

    # Timeout
    with patch("requests.get", side_effect=requests.exceptions.Timeout("timed out")):
        res = c.get_available_models()
        assert res == []
        assert c._last_error == "Request timed out: timed out"

    # HTTPError with response=None
    err_no_resp = requests.exceptions.HTTPError("no resp")
    err_no_resp.response = None
    with patch("requests.get", side_effect=err_no_resp):
        res = c.get_available_models()
        assert res == []
        assert c._last_error == "HTTP Unknown error: no resp"

    # HTTPError with response
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.text = "Service Unavailable"
    err_resp = requests.exceptions.HTTPError("with resp", response=mock_resp)
    with patch("requests.get", side_effect=err_resp):
        res = c.get_available_models()
        assert res == []
        assert c._last_error == "HTTP 503 error: Service Unavailable"

    # models key missing
    mock_empty = MagicMock()
    mock_empty.json.return_value = {}
    with patch("requests.get", return_value=mock_empty):
        res = c.get_available_models()
        assert res == []


@verifies("REQ-007")
def test_client_prewarm_model_error_messages():
    """Kill mutants in prewarm_model error message formatting."""
    c = OllamaClient(host="http://test-host:11434")

    # ConnectionError
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("conn fail")):
        msg = c.prewarm_model("model1")
        assert msg == "Error pre-warming model 'model1': Unable to connect to Ollama (conn fail)"

    # Timeout
    with patch("requests.post", side_effect=requests.exceptions.Timeout("timed out")):
        msg = c.prewarm_model("model1")
        assert msg == "Error pre-warming model 'model1': Request timed out (timed out)"

    # HTTPError without response
    err_no_resp = requests.exceptions.HTTPError("no resp")
    err_no_resp.response = None
    with patch("requests.post", side_effect=err_no_resp):
        msg = c.prewarm_model("model1")
        assert msg == "Error pre-warming model 'model1': HTTP Unknown - no resp"

    # HTTPError with response
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"
    err_resp = requests.exceptions.HTTPError("with resp", response=mock_resp)
    with patch("requests.post", side_effect=err_resp):
        msg = c.prewarm_model("model1")
        assert msg == "Error pre-warming model 'model1': HTTP 500 - Internal Server Error"


@verifies("REQ-002")
def test_engine_query_ollama_exact_error_messages_and_empty_response():
    """Kill mutants in query_ollama error messages and missing response key."""
    # Empty response dict
    mock_resp = MagicMock()
    mock_resp.json.return_value = {}
    mock_resp.raise_for_status.return_value = None
    with patch("requests.post", return_value=mock_resp):
        res = query_ollama("prompt", model="qwen2.5-coder:14b")
        assert res == ""

    # ConnectionError
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("offline")):
        res = query_ollama("prompt", model="qwen2.5-coder:14b")
        expected = (
            f"Error: Unable to connect to Ollama at {OLLAMA_HOST}. "
            "Please ensure Ollama is running (`ollama serve` or `systemctl status ollama`)."
        )
        assert res == expected

    # Timeout
    with patch("requests.post", side_effect=requests.exceptions.Timeout("t/o")):
        res = query_ollama("prompt", model="qwen2.5-coder:14b")
        expected = f"Error: Request to Ollama timed out after {DEFAULT_TIMEOUT}s for model 'qwen2.5-coder:14b'."
        assert res == expected

    # HTTPError without response
    err_no_resp = requests.exceptions.HTTPError("http fail")
    err_no_resp.response = None
    with patch("requests.post", side_effect=err_no_resp):
        res = query_ollama("prompt", model="qwen2.5-coder:14b")
        assert res == "Error from Ollama (Unknown): http fail"

    # HTTPError with response
    mock_err_resp = MagicMock()
    mock_err_resp.status_code = 404
    mock_err_resp.text = "Model Not Found"
    err_with_resp = requests.exceptions.HTTPError("not found", response=mock_err_resp)
    with patch("requests.post", side_effect=err_with_resp):
        res = query_ollama("prompt", model="qwen2.5-coder:14b")
        assert res == "Error from Ollama (404): Model Not Found"


@verifies("REQ-006")
def test_client_list_models_unknown_fallbacks():
    """Kill mutants in list_models default fallback values for name, size, details."""
    c = OllamaClient()
    with patch.object(c, "get_available_models", return_value=[{}]):
        models = c.list_models()
        assert len(models) == 1
        m = models[0]
        assert m.name == "unknown"
        assert m.size == 0
        assert m.size_gb == "0.00 GB"
        assert m.parameter_size == "unknown"
        assert m.quantization_level == "unknown"


@verifies("REQ-001")
def test_client_resolve_model_delegation_to_available_models():
    """Kill mutants in OllamaClient.resolve_model argument wiring and get_available_models call."""
    c = OllamaClient()
    with patch.object(c, "get_available_models", return_value=[{"name": "deepseek-r1:14b"}]) as mock_avail:
        res = c.resolve_model("")
        assert res == "deepseek-r1:14b"
        mock_avail.assert_called_once()

    # When requested_model is explicitly given
    assert c.resolve_model("custom-explicit:7b") == "custom-explicit:7b"
