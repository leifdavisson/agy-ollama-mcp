from ollama_bridge.client import local_list_models as client_list_models, local_prewarm_model as client_prewarm_model
"""
Additional targeted tests to achieve 100% statement and branch coverage across all modules.
Requirements: REQ-001, REQ-002, REQ-004, REQ-006, REQ-007, REQ-008.
License: GNU AGPLv3
"""
from unittest.mock import patch, MagicMock
import requests
import pytest

from ollama_bridge.client import OllamaClient
from ollama_bridge.engine import query_ollama, partition_chunks, local_chunked_summary
from ollama_bridge.resolution import _fetch_tags_names


def verifies(req_id: str):
    def decorator(fn):
        fn.verified_requirement = req_id
        return fn
    return decorator


@verifies("REQ-006")
def test_client_generic_exception_in_available_models():
    client = OllamaClient()
    with patch("requests.get", side_effect=RuntimeError("Unexpected socket crash")):
        res = client.get_available_models()
        assert res == []
        assert "Error: Unexpected socket crash" in (client._last_error or "")


@verifies("REQ-006")
def test_client_list_models_non_dict_element_ignored():
    client = OllamaClient()
    with patch.object(client, "get_available_models", return_value=["invalid_string_element", {"name": "valid:7b"}]):
        models = client.list_models()
        assert len(models) == 1
        assert models[0].name == "valid:7b"


@verifies("REQ-007")
def test_client_prewarm_exception_branches():
    client = OllamaClient()
    with patch("requests.post", side_effect=requests.exceptions.Timeout("Timed out")):
        res = client.prewarm_model("m1")
        assert "Request timed out" in res

    mock_resp = MagicMock()
    mock_resp.status_code = 502
    mock_resp.text = "Bad Gateway"
    with patch("requests.post", side_effect=requests.exceptions.HTTPError(response=mock_resp)):
        res = client.prewarm_model("m1")
        assert "HTTP 502" in res

    with patch("requests.post", side_effect=ValueError("Invalid payload")):
        res = client.prewarm_model("m1")
        assert "Error pre-warming model 'm1': Invalid payload" in res


@verifies("REQ-002")
def test_engine_query_ollama_exception_branches():
    with patch("requests.post", side_effect=requests.exceptions.Timeout("Timeout")):
        res = query_ollama("prompt")
        assert "timed out after" in res

    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.text = "Bad Request"
    with patch("requests.post", side_effect=requests.exceptions.HTTPError(response=mock_resp)):
        res = query_ollama("prompt")
        assert "Error from Ollama (400)" in res

    with patch("requests.post", side_effect=RuntimeError("Unexpected")):
        res = query_ollama("prompt")
        assert "Unexpected error during Ollama query" in res


@verifies("REQ-004")
def test_engine_partition_chunks_no_trailing_newline():
    content = "Line1\nLine2\nLine3"  # No trailing newline
    chunks = partition_chunks(content, chunk_chars=10)
    assert "".join(chunks) == content
    assert len(chunks) == 3


@verifies("REQ-004")
def test_engine_chunked_summary_empty_chunks():
    with patch("ollama_bridge.engine.partition_chunks", return_value=[]):
        res = local_chunked_summary("long enough text to partition", chunk_chars=5)
        assert res == ""


@verifies("REQ-001")
def test_resolution_fetch_tags_names_exception():
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Offline")):
        names = _fetch_tags_names()
        assert names == []

@verifies("REQ-006")
def test_client_local_list_models_default_client():
    with patch("ollama_bridge.client.OllamaClient.list_models", return_value=[]):
        res = client_list_models()
        assert "No models found" in res or "Unable to connect" in res


@verifies("REQ-007")
def test_client_local_prewarm_model_default_client():
    with patch("ollama_bridge.client.OllamaClient.prewarm_model", return_value="prewarmed ok"):
        res = client_prewarm_model("model_x")
        assert res == "prewarmed ok"


@verifies("REQ-004")
def test_engine_partition_chunks_trailing_newline_empty_last_element():
    res = partition_chunks("first\nsecond\n", chunk_chars=20)
    assert res == ["first\nsecond\n"]

@verifies("REQ-004")
def test_engine_partition_chunks_both_branches_of_last_line():
    assert partition_chunks("no_newline", 10) == ["no_newline"]
    assert partition_chunks("with_newline\n", 20) == ["with_newline\n"]
