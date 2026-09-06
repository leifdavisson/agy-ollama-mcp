"""
Unit, Property-Based, and MC/DC Test Suite for Ollama Client & Prewarm Module.

Requirements Verified:
    - REQ-006: Model Inventory & Status Enumeration
    - REQ-007: Memory Pre-warming & Keep-Alive
Specification:
    - features/model_inventory.feature
Target Implementation:
    - src/ollama_bridge/client.py
"""
import os
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import pytest
import requests
from hypothesis import given, strategies as st, settings

# Ensure src/ and root are in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ollama_bridge.client import (
    OllamaClient,
    ModelInfo,
    format_bytes_to_gb,
    local_list_models,
    local_prewarm_model,
)


def verifies(req_id: str):
    """Decorator marking a test function as verifying a specific requirement ID.

    Attaches requirement metadata to the function and marks it with pytest markers.
    """
    def decorator(fn):
        setattr(fn, "__req_id__", req_id)
        if not hasattr(fn, "_verifies"):
            fn._verifies = []
        fn._verifies.append(req_id)
        fn = pytest.mark.requirement(req_id)(fn)
        return pytest.mark.verifies(req_id)(fn)
    return decorator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_deepseek_tag_payload() -> Dict[str, Any]:
    """Sample /api/tags JSON response containing deepseek-r1:14b from specification."""
    return {
        "models": [
            {
                "name": "deepseek-r1:14b",
                "model": "deepseek-r1:14b",
                "modified_at": "2024-11-20T10:00:00Z",
                "size": 8988112209,
                "digest": "abcdef1234567890",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "qwen2",
                    "families": ["qwen2"],
                    "parameter_size": "14.8B",
                    "quantization_level": "Q4_K_M",
                },
            }
        ]
    }


@pytest.fixture
def mock_multi_tag_payload() -> Dict[str, Any]:
    """Sample /api/tags JSON response with multiple models and variations."""
    return {
        "models": [
            {
                "name": "deepseek-r1:14b",
                "size": 8988112209,
                "details": {
                    "parameter_size": "14.8B",
                    "quantization_level": "Q4_K_M",
                },
            },
            {
                "name": "qwen2.5-coder:14b",
                "size": 9123456789,
                "details": {
                    "parameter_size": "14.7B",
                    "quantization_level": "Q4_K_M",
                },
            },
            {
                "name": "llama3.2:latest",
                "size": 2048000000,
                "details": {
                    "parameter_size": "3B",
                    "quantization_level": "Q8_0",
                },
            },
        ]
    }


# ===========================================================================
# Section 1: REQ-006 Inventory Parsing & Unit Tests
# ===========================================================================

@verifies("REQ-006")
def test_format_bytes_to_gb_spec_vector():
    """Verify exact byte conversion from specification feature: 8988112209 -> '8.37 GB'."""
    raw_bytes = 8988112209
    formatted = format_bytes_to_gb(raw_bytes)
    assert formatted == "8.37 GB"


@verifies("REQ-006")
def test_format_bytes_to_gb_boundary_zero():
    """Verify 0 bytes formats correctly to '0.00 GB'."""
    assert format_bytes_to_gb(0) == "0.00 GB"


@verifies("REQ-006")
def test_format_bytes_to_gb_exact_gib():
    """Verify exact powers of 2 (1 GiB, 10 GiB) formatting."""
    one_gib = 1024 ** 3
    ten_gib = 10 * (1024 ** 3)
    assert format_bytes_to_gb(one_gib) == "1.00 GB"
    assert format_bytes_to_gb(ten_gib) == "10.00 GB"


@verifies("REQ-006")
def test_format_bytes_to_gb_fractional_rounding():
    """Verify standard round-half-even behavior on fractional sizes."""
    # 500,000,000 bytes / (1024^3) = 0.46566... -> rounded to 0.47 GB
    assert format_bytes_to_gb(500000000) == "0.47 GB"


@verifies("REQ-006")
def test_parse_model_inventory_deepseek_spec(mock_deepseek_tag_payload):
    """Verify BDD scenario: Given deepseek-r1:14b with 8988112209 bytes, parses size, params, quant."""
    client = OllamaClient(host="http://localhost:11434")
    mock_resp = MagicMock()
    mock_resp.json.return_value = mock_deepseek_tag_payload
    mock_resp.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_resp):
        models = client.list_models()
        inventory_text = client.format_inventory(models)

    assert len(models) == 1
    m = models[0]
    assert m.name == "deepseek-r1:14b"
    assert m.size == 8988112209
    assert m.size_gb == "8.37 GB"
    assert m.parameter_size == "14.8B"
    assert m.quantization_level == "Q4_K_M"

    assert "deepseek-r1:14b" in inventory_text
    assert "8.37 GB" in inventory_text
    assert "14.8B" in inventory_text
    assert "Q4_K_M" in inventory_text
    assert "Active Default Model:" in inventory_text


@verifies("REQ-006")
def test_parse_model_inventory_multiple_models_with_active_default(mock_multi_tag_payload):
    """Verify inventory with multiple models identifies active default model and counts."""
    client = OllamaClient(host="http://localhost:11434", configured_model="qwen2.5-coder:14b")
    mock_resp = MagicMock()
    mock_resp.json.return_value = mock_multi_tag_payload
    mock_resp.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_resp):
        models = client.list_models()
        inventory_text = client.format_inventory(models)

    assert len(models) == 3
    assert "Installed Models (3):" in inventory_text
    assert "Active Default Model: qwen2.5-coder:14b" in inventory_text
    assert "qwen2.5-coder:14b" in inventory_text
    assert "llama3.2:latest" in inventory_text

    # Verify is_default flag on ModelInfo
    coder_model = next(m for m in models if m.name == "qwen2.5-coder:14b")
    assert coder_model.is_default is True
    llama_model = next(m for m in models if m.name == "llama3.2:latest")
    assert llama_model.is_default is False


@verifies("REQ-006")
def test_parse_model_inventory_missing_optional_details():
    """Verify fallback to 'unknown' when model payload lacks details or fields."""
    client = OllamaClient(host="http://localhost:11434")
    payload = {
        "models": [
            {
                "name": "bare-model:latest",
                "size": 1073741824,
                # 'details' omitted entirely
            }
        ]
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = payload
    mock_resp.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_resp):
        models = client.list_models()
        display = client.format_inventory(models)

    assert len(models) == 1
    assert models[0].parameter_size == "unknown"
    assert models[0].quantization_level == "unknown"
    assert "params: unknown" in display
    assert "quant: unknown" in display


@verifies("REQ-006")
def test_parse_model_inventory_empty_list():
    """Verify graceful handling when /api/tags returns empty models list."""
    client = OllamaClient(host="http://localhost:11434")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"models": []}
    mock_resp.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_resp):
        models = client.list_models()
        display = client.format_inventory(models)

    assert len(models) == 0
    assert "No models found" in display


# ===========================================================================
# Section 2: REQ-006 Network Error Handling for Inventory
# ===========================================================================

@verifies("REQ-006")
def test_inventory_connection_error_graceful_handling():
    """Verify ConnectionError returns a human-readable error message without uncaught exception."""
    client = OllamaClient(host="http://localhost:11434")
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        models = client.list_models()
        assert models == []
        display = client.format_inventory()
        assert "unable to connect" in display.lower() or "connection refused" in display.lower()


@verifies("REQ-006")
def test_inventory_timeout_error_graceful_handling():
    """Verify Timeout returns a graceful error message indicating timeout."""
    client = OllamaClient(host="http://localhost:11434", timeout=5)
    with patch("requests.get", side_effect=requests.exceptions.Timeout("Read timed out")):
        models = client.list_models()
        assert models == []
        display = client.format_inventory()
        assert "timed out" in display.lower() or "timeout" in display.lower() or "unable to connect" in display.lower()


@verifies("REQ-006")
def test_inventory_http_4xx_error_handling():
    """Verify HTTP 404/4xx error is captured with status code and error details."""
    client = OllamaClient(host="http://localhost:11434")
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.text = "Not Found"
    error = requests.exceptions.HTTPError(response=mock_resp)

    with patch("requests.get", side_effect=error):
        models = client.list_models()
        assert models == []
        display = client.format_inventory()
        assert "404" in display or "not found" in display.lower() or "unable" in display.lower()


@verifies("REQ-006")
def test_inventory_http_5xx_error_handling():
    """Verify HTTP 500/5xx error is captured gracefully with status code."""
    client = OllamaClient(host="http://localhost:11434")
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"
    error = requests.exceptions.HTTPError(response=mock_resp)

    with patch("requests.get", side_effect=error):
        models = client.list_models()
        assert models == []
        display = client.format_inventory()
        assert "500" in display or "internal server error" in display.lower() or "unable" in display.lower()


# ===========================================================================
# Section 3: REQ-007 Memory Pre-warming & Keep-Alive
# ===========================================================================

@verifies("REQ-007")
def test_prewarm_explicit_model_payload_and_endpoint():
    """Verify BDD scenario: POST /api/generate with keep_alive=-1 and explicit target model."""
    client = OllamaClient(host="http://localhost:11434")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "success"}
    mock_resp.raise_for_status.return_value = None

    with patch("requests.post", return_value=mock_resp) as mock_post:
        res = client.prewarm_model("qwen2.5-coder:14b")

    mock_post.assert_called_once()
    called_url, called_kwargs = mock_post.call_args
    assert called_url[0].endswith("/api/generate")
    payload = called_kwargs.get("json", {})
    assert payload.get("model") == "qwen2.5-coder:14b"
    assert payload.get("keep_alive") == -1
    assert "Successfully pre-warmed model 'qwen2.5-coder:14b'" in res


@verifies("REQ-007")
def test_prewarm_empty_model_uses_resolved_default():
    """Verify prewarm with empty model parameter resolves to active default model."""
    client = OllamaClient(host="http://localhost:11434", configured_model="env-coder:7b")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "success"}
    mock_resp.raise_for_status.return_value = None

    with patch("requests.post", return_value=mock_resp) as mock_post:
        res = client.prewarm_model("")

    called_kwargs = mock_post.call_args[1]
    payload = called_kwargs.get("json", {})
    assert payload.get("model") == "env-coder:7b"
    assert payload.get("keep_alive") == -1
    assert "env-coder:7b" in res


@verifies("REQ-007")
def test_prewarm_keep_alive_constant_is_strict_negative_one():
    """Verify keep_alive is strictly integer -1 (not True, not float, not string)."""
    client = OllamaClient(host="http://localhost:11434")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None

    with patch("requests.post", return_value=mock_resp) as mock_post:
        client.prewarm_model("deepseek-r1:14b")

    payload = mock_post.call_args[1].get("json", {})
    val = payload.get("keep_alive")
    assert val == -1
    assert isinstance(val, int)
    assert not isinstance(val, bool)


# ===========================================================================
# Section 4: REQ-007 Network Error Handling for Prewarm
# ===========================================================================

@verifies("REQ-007")
def test_prewarm_connection_error_graceful_handling():
    """Verify prewarm returns error string on ConnectionError without crash."""
    client = OllamaClient(host="http://localhost:11434")
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        res = client.prewarm_model("qwen2.5-coder:14b")
    assert "Error pre-warming model 'qwen2.5-coder:14b'" in res
    assert "Connection refused" in res or "Unable to connect" in res or "Error" in res


@verifies("REQ-007")
def test_prewarm_timeout_error_graceful_handling():
    """Verify prewarm returns error string on Timeout."""
    client = OllamaClient(host="http://localhost:11434")
    with patch("requests.post", side_effect=requests.exceptions.Timeout("Request timed out")):
        res = client.prewarm_model("qwen2.5-coder:14b")
    assert "Error pre-warming model 'qwen2.5-coder:14b'" in res
    assert "timed out" in res.lower() or "timeout" in res.lower()


@verifies("REQ-007")
def test_prewarm_http_4xx_error_handling():
    """Verify prewarm returns error string with HTTP 404 (model not found)."""
    client = OllamaClient(host="http://localhost:11434")
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.text = "model 'missing:latest' not found"
    err = requests.exceptions.HTTPError(response=mock_resp)

    with patch("requests.post", side_effect=err):
        res = client.prewarm_model("missing:latest")
    assert "Error pre-warming model 'missing:latest'" in res
    assert "404" in res or "not found" in res.lower()


@verifies("REQ-007")
def test_prewarm_http_5xx_error_handling():
    """Verify prewarm returns error string with HTTP 503 (server overloaded)."""
    client = OllamaClient(host="http://localhost:11434")
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.text = "Service Unavailable"
    err = requests.exceptions.HTTPError(response=mock_resp)

    with patch("requests.post", side_effect=err):
        res = client.prewarm_model("qwen2.5-coder:14b")
    assert "Error pre-warming model 'qwen2.5-coder:14b'" in res
    assert "503" in res or "Service Unavailable" in res


# ===========================================================================
# Section 5: MC/DC Test Vectors (Modified Condition / Decision Coverage)
# ===========================================================================

# Decision 1: Model Resolution Hierarchy
# Conditions:
#   C1: Explicit requested_model is non-empty
#   C2: CONFIGURED_MODEL / LOCAL_LLM_MODEL is non-empty
#   C3: Preferred model is found in available models / tags
# Fallback: "qwen2.5-coder:14b"

@verifies("REQ-006")
def test_mcdc_model_resolution_vector_1_explicit():
    """MC/DC Vector 1: C1=True, C2=True, C3=True -> Outcome: Explicit Model."""
    client = OllamaClient(configured_model="env-model:7b")
    with patch.object(client, "get_available_models", return_value=[{"name": "autodetect-model:14b"}]):
        res = client.resolve_model("explicit-model:latest")
    assert res == "explicit-model:latest"


@verifies("REQ-006")
def test_mcdc_model_resolution_vector_2_configured_env():
    """MC/DC Vector 2: C1=False, C2=True, C3=True -> Outcome: Configured Model (Isolates C1)."""
    client = OllamaClient(configured_model="env-model:7b")
    with patch.object(client, "get_available_models", return_value=[{"name": "autodetect-model:14b"}]):
        res = client.resolve_model("")
    assert res == "env-model:7b"


@verifies("REQ-006")
def test_mcdc_model_resolution_vector_3_autodetect():
    """MC/DC Vector 3: C1=False, C2=False, C3=True -> Outcome: Auto-detected Model (Isolates C2)."""
    client = OllamaClient(configured_model="")
    with patch.object(client, "get_available_models", return_value=[{"name": "qwen2.5-coder:14b"}]):
        res = client.resolve_model("")
    assert res == "qwen2.5-coder:14b"


@verifies("REQ-006")
def test_mcdc_model_resolution_vector_4_fallback():
    """MC/DC Vector 4: C1=False, C2=False, C3=False -> Outcome: Fallback Default (Isolates C3)."""
    client = OllamaClient(configured_model="")
    with patch.object(client, "get_available_models", return_value=[]):
        res = client.resolve_model("")
    assert res == "qwen2.5-coder:14b"


# Decision 2: Model Inventory Detail Parsing
# Conditions:
#   C4: 'details' dict present in raw model dict
#   C5: 'parameter_size' present in details dict
#   C6: 'quantization_level' present in details dict

@verifies("REQ-006")
def test_mcdc_details_vector_5_all_present():
    """MC/DC Vector 5: C4=True, C5=True, C6=True -> Full parameters extracted."""
    client = OllamaClient()
    payload = {"models": [{"name": "m1", "size": 1000, "details": {"parameter_size": "7B", "quantization_level": "Q4_0"}}]}
    mock_resp = MagicMock(json=lambda: payload, raise_for_status=lambda: None)
    with patch("requests.get", return_value=mock_resp):
        m = client.list_models()[0]
    assert m.parameter_size == "7B"
    assert m.quantization_level == "Q4_0"


@verifies("REQ-006")
def test_mcdc_details_vector_6_missing_params():
    """MC/DC Vector 6: C4=True, C5=False, C6=True -> params='unknown', quant='Q4_0' (Isolates C5)."""
    client = OllamaClient()
    payload = {"models": [{"name": "m1", "size": 1000, "details": {"quantization_level": "Q4_0"}}]}
    mock_resp = MagicMock(json=lambda: payload, raise_for_status=lambda: None)
    with patch("requests.get", return_value=mock_resp):
        m = client.list_models()[0]
    assert m.parameter_size == "unknown"
    assert m.quantization_level == "Q4_0"


@verifies("REQ-006")
def test_mcdc_details_vector_7_missing_quant():
    """MC/DC Vector 7: C4=True, C5=True, C6=False -> params='7B', quant='unknown' (Isolates C6)."""
    client = OllamaClient()
    payload = {"models": [{"name": "m1", "size": 1000, "details": {"parameter_size": "7B"}}]}
    mock_resp = MagicMock(json=lambda: payload, raise_for_status=lambda: None)
    with patch("requests.get", return_value=mock_resp):
        m = client.list_models()[0]
    assert m.parameter_size == "7B"
    assert m.quantization_level == "unknown"


@verifies("REQ-006")
def test_mcdc_details_vector_8_missing_details_dict():
    """MC/DC Vector 8: C4=False -> params='unknown', quant='unknown' (Isolates C4)."""
    client = OllamaClient()
    payload = {"models": [{"name": "m1", "size": 1000}]}
    mock_resp = MagicMock(json=lambda: payload, raise_for_status=lambda: None)
    with patch("requests.get", return_value=mock_resp):
        m = client.list_models()[0]
    assert m.parameter_size == "unknown"
    assert m.quantization_level == "unknown"


# Decision 3: HTTP Response / Error Classification
# Conditions:
#   E1: Success HTTP 200
#   E2: ConnectionError
#   E3: Timeout
#   E4: HTTPError 4xx
#   E5: HTTPError 5xx

@verifies("REQ-007")
def test_mcdc_network_vector_9_success_200():
    """MC/DC Vector 9: E1=True -> Success confirmation."""
    client = OllamaClient()
    mock_resp = MagicMock(status_code=200, raise_for_status=lambda: None)
    with patch("requests.post", return_value=mock_resp):
        res = client.prewarm_model("test-model")
    assert "Successfully pre-warmed" in res


@verifies("REQ-007")
def test_mcdc_network_vector_10_conn_error():
    """MC/DC Vector 10: E2=True -> Connection error handled."""
    client = OllamaClient()
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Refused")):
        res = client.prewarm_model("test-model")
    assert "Error pre-warming" in res
    assert "Refused" in res or "Unable" in res or "Error" in res


@verifies("REQ-007")
def test_mcdc_network_vector_11_timeout_error():
    """MC/DC Vector 11: E3=True -> Timeout error handled."""
    client = OllamaClient()
    with patch("requests.post", side_effect=requests.exceptions.Timeout("Timed out")):
        res = client.prewarm_model("test-model")
    assert "Error pre-warming" in res
    assert "timed out" in res.lower() or "timeout" in res.lower()


@verifies("REQ-007")
def test_mcdc_network_vector_12_client_4xx():
    """MC/DC Vector 12: E4=True -> HTTP 400/404 handled with status code."""
    client = OllamaClient()
    resp = MagicMock(status_code=400, text="Bad Request")
    with patch("requests.post", side_effect=requests.exceptions.HTTPError(response=resp)):
        res = client.prewarm_model("test-model")
    assert "Error pre-warming" in res
    assert "400" in res or "Bad Request" in res


@verifies("REQ-007")
def test_mcdc_network_vector_13_server_5xx():
    """MC/DC Vector 13: E5=True -> HTTP 500/502 handled with status code."""
    client = OllamaClient()
    resp = MagicMock(status_code=502, text="Bad Gateway")
    with patch("requests.post", side_effect=requests.exceptions.HTTPError(response=resp)):
        res = client.prewarm_model("test-model")
    assert "Error pre-warming" in res
    assert "502" in res or "Bad Gateway" in res


# ===========================================================================
# Section 6: Hypothesis Property-Based Tests
# ===========================================================================

@verifies("REQ-006")
@given(st.integers(min_value=0, max_value=10**18))
@settings(max_examples=100)
def test_hypothesis_format_bytes_to_gb_properties(byte_val: int):
    """Property test: For all non-negative integers, format_bytes_to_gb conforms to schema and bounds.

    Properties:
    1. Returns a string ending with ' GB'.
    2. Numerical portion parses to a non-negative float.
    3. Has at most 2 decimal digits.
    4. Numerical value matches round(byte_val / 1024^3, 2) within float precision.
    """
    res = format_bytes_to_gb(byte_val)
    assert isinstance(res, str)
    assert res.endswith(" GB")

    num_str = res[:-3]
    num_val = float(num_str)
    assert num_val >= 0.0

    if "." in num_str:
        decimals = len(num_str.split(".")[1])
        assert decimals <= 2

    expected_val = round(byte_val / (1024 ** 3), 2)
    assert abs(num_val - expected_val) < 1e-5


@verifies("REQ-006")
@given(
    st.integers(min_value=0, max_value=10**15),
    st.integers(min_value=0, max_value=10**15),
)
@settings(max_examples=50)
def test_hypothesis_format_bytes_monotonic(b1: int, b2: int):
    """Property test: format_bytes_to_gb is monotonically non-decreasing.

    If b1 <= b2, then float_val(b1) <= float_val(b2).
    """
    small = min(b1, b2)
    large = max(b1, b2)

    val_small = float(format_bytes_to_gb(small).replace(" GB", ""))
    val_large = float(format_bytes_to_gb(large).replace(" GB", ""))

    assert val_small <= val_large


@verifies("REQ-006")
@given(st.just(8988112209))
def test_hypothesis_format_bytes_exact_spec_invariance(exact_bytes: int):
    """Property test: Invariance of the specific benchmark test vector from model_inventory.feature."""
    assert format_bytes_to_gb(exact_bytes) == "8.37 GB"
