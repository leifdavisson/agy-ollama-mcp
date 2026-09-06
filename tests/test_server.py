"""
Unit tests for server.py FastMCP tools and entrypoint (REQ-008).
License: GNU AGPLv3
"""
from unittest.mock import patch, MagicMock
import pytest
from ollama_bridge import server


def verifies(req_id: str):
    def decorator(fn):
        fn.verified_requirement = req_id
        return fn
    return decorator


@verifies("REQ-008")
def test_server_tools_delegation():
    """Verify server tools properly delegate to underlying engine and client modules."""
    with patch("ollama_bridge.server.engine_draft_code", return_value="code") as m_draft, \
         patch("ollama_bridge.server.engine_summarize", return_value="summary") as m_sum, \
         patch("ollama_bridge.server.engine_chunked_summary", return_value="chunked") as m_chunk, \
         patch("ollama_bridge.server.engine_extract_json", return_value="{}") as m_json, \
         patch("ollama_bridge.server.engine_map_reduce_file", return_value="brief") as m_mrf, \
         patch("ollama_bridge.server.client_list_models", return_value="models") as m_list, \
         patch("ollama_bridge.server.client_prewarm_model", return_value="prewarmed") as m_prewarm:

        assert server.local_draft_code("task", "ctx", "py", "model") == "code"
        m_draft.assert_called_once_with(task_description="task", context="ctx", language="py", model="model")

        assert server.local_summarize_and_extract("content", "goal", "model") == "summary"
        m_sum.assert_called_once_with(content="content", extraction_goal="goal", model="model")

        assert server.local_chunked_summary("content", "goal", 1000, "model") == "chunked"
        m_chunk.assert_called_once_with(content="content", extraction_goal="goal", chunk_chars=1000, model="model")

        assert server.local_extract_json("content", "schema", "model") == "{}"
        m_json.assert_called_once_with(content="content", schema_description="schema", model="model")

        assert server.local_map_reduce_file("file.log", "goal", 200, 20, 3, "model") == "brief"
        m_mrf.assert_called_once_with(file_path="file.log", extraction_goal="goal", chunk_size=200, overlap=20, concurrency=3, model="model")

        assert server.local_list_models() == "models"
        m_list.assert_called_once()

        assert server.local_prewarm_model("model") == "prewarmed"
        m_prewarm.assert_called_once_with(model="model")


@verifies("REQ-008")
def test_server_main_entrypoint():
    """Verify server.main() calls mcp.run(transport='stdio')."""
    with patch.object(server.mcp, "run") as mock_run:
        server.main()
        mock_run.assert_called_once_with(transport="stdio")
