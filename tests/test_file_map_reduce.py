"""
Unit and Integration tests for File-Direct Sliding-Window Map-Reduce (REQ-009).
Specification: features/file_map_reduce.feature
License: GNU AGPLv3
"""
import os
import sys
from unittest.mock import MagicMock, patch
import pytest

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(repo_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from ollama_bridge.engine import (
    chunk_lines_overlap,
    local_map_reduce_file,
    map_worker,
)


def verifies(req_id: str):
    def decorator(fn):
        fn.verified_requirement = req_id
        fn = pytest.mark.requirement(req_id)(fn)
        return pytest.mark.verifies(req_id)(fn)
    return decorator


@verifies("REQ-009")
def test_chunk_lines_overlap_validation_and_empty():
    """Verify input validation and empty lines handling in chunk_lines_overlap."""
    assert chunk_lines_overlap([]) == []

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_lines_overlap(["a\n"], chunk_size=0)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_lines_overlap(["a\n"], chunk_size=-10)

    with pytest.raises(ValueError, match="overlap must be non-negative"):
        chunk_lines_overlap(["a\n"], chunk_size=10, overlap=-1)

    with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
        chunk_lines_overlap(["a\n"], chunk_size=10, overlap=10)

    with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
        chunk_lines_overlap(["a\n"], chunk_size=10, overlap=15)


@verifies("REQ-009")
def test_chunk_lines_overlap_sliding_window():
    """Verify sliding window lines overlap and preservation across chunk boundaries."""
    lines = [f"Line {i}\n" for i in range(1, 11)]  # 10 lines
    chunk_size = 4
    overlap = 2

    # step = 4 - 2 = 2
    # chunk 0: lines 0..4 (Lines 1..4)
    # chunk 1: lines 2..6 (Lines 3..6)
    # chunk 2: lines 4..8 (Lines 5..8)
    # chunk 3: lines 6..10 (Lines 7..10)
    chunks = chunk_lines_overlap(lines, chunk_size=chunk_size, overlap=overlap)
    assert len(chunks) == 4

    assert chunks[0] == "Line 1\nLine 2\nLine 3\nLine 4\n"
    assert chunks[1] == "Line 3\nLine 4\nLine 5\nLine 6\n"
    assert chunks[2] == "Line 5\nLine 6\nLine 7\nLine 8\n"
    assert chunks[3] == "Line 7\nLine 8\nLine 9\nLine 10\n"

    # Verify overlap at boundaries
    assert "Line 3\nLine 4\n" in chunks[0] and chunks[1].startswith("Line 3\nLine 4\n")
    assert "Line 5\nLine 6\n" in chunks[1] and chunks[2].startswith("Line 5\nLine 6\n")


@verifies("REQ-009")
def test_map_worker_prompt_and_system():
    """Verify map_worker prompt construction and system instruction."""
    chunk = "2026-09-06 [ERROR] DB Connection Timeout at PID 421\n2026-09-06 [INFO] Healthcheck OK"
    goal = "Extract database timeouts"

    with patch("ollama_bridge.engine.query_ollama", return_value="- DB Timeout PID 421") as mock_q:
        res = map_worker(chunk, goal, model="custom:map")
        assert res == "- DB Timeout PID 421"
        mock_q.assert_called_once()
        kwargs = mock_q.call_args[1]
        args = mock_q.call_args[0]
        prompt = args[0] if args else kwargs.get("prompt", "")
        assert kwargs["model"] == "custom:map"
        assert kwargs["temperature"] == 0.1
        assert "GOAL:\nExtract database timeouts" in prompt
        assert "CONTENT:\n" + chunk in prompt
        assert "dense extraction and noise-filtering engine" in kwargs["system"]
        assert "NO_SIGNAL" in kwargs["system"]


@verifies("REQ-009")
def test_local_map_reduce_file_nonexistent_and_dir(tmp_path):
    """Verify error handling for missing files and directories."""
    missing = tmp_path / "does_not_exist.log"
    res = local_map_reduce_file(str(missing), "goal")
    assert "Error: File" in res
    assert "does not exist" in res

    directory = tmp_path / "somedir"
    directory.mkdir()
    res_dir = local_map_reduce_file(str(directory), "goal")
    assert "Error: Path" in res_dir
    assert "is a directory" in res_dir


@verifies("REQ-009")
def test_local_map_reduce_file_empty(tmp_path):
    """Verify empty file handling."""
    empty_file = tmp_path / "empty.log"
    empty_file.write_text("")
    res = local_map_reduce_file(str(empty_file), "goal")
    assert res == "File is empty."


@verifies("REQ-009")
def test_local_map_reduce_file_single_pass(tmp_path):
    """Verify small file executes single-pass without chunking."""
    small_file = tmp_path / "small.log"
    content = "Line 1: Error 500\nLine 2: Warning\n"
    small_file.write_text(content)

    with patch("ollama_bridge.engine.query_ollama", return_value="Single pass summary") as mock_q:
        res = local_map_reduce_file(
            file_path=str(small_file),
            extraction_goal="Find Error 500",
            chunk_size=10,
            model="qwen:test",
        )
        assert res == "Single pass summary"
        mock_q.assert_called_once()
        kwargs = mock_q.call_args[1]
        args = mock_q.call_args[0]
        prompt = args[0] if args else kwargs.get("prompt", "")
        assert kwargs["model"] == "qwen:test"
        assert kwargs["temperature"] == 0.1
        assert "GOAL: Find Error 500" in prompt
        assert content in prompt


@verifies("REQ-009")
def test_local_map_reduce_file_multi_chunk_with_no_signal(tmp_path):
    """Verify multi-chunk map-reduce when all chunks yield NO_SIGNAL."""
    large_file = tmp_path / "large_empty.log"
    lines = [f"Routine log line {i}\n" for i in range(1, 21)]
    large_file.write_text("".join(lines))

    # With chunk_size=5 and overlap=1, 20 lines yields multiple chunks
    with patch("ollama_bridge.engine.map_worker", return_value="NO_SIGNAL"):
        res = local_map_reduce_file(
            file_path=str(large_file),
            extraction_goal="Find memory leaks",
            chunk_size=5,
            overlap=1,
            concurrency=2,
        )
        assert "No signal matching 'Find memory leaks' found across" in res


@verifies("REQ-009")
def test_local_map_reduce_file_multi_chunk_synthesis(tmp_path):
    """Verify multi-chunk map-reduce successfully runs concurrent workers and synthesizes brief."""
    large_file = tmp_path / "large_system.log"
    lines = [f"Log trace line {i}\n" for i in range(1, 25)]
    large_file.write_text("".join(lines))

    def mock_worker(chunk, goal, model=""):
        if "trace line 5" in chunk:
            return "- Database Connection Refused at line 5"
        if "trace line 15" in chunk:
            return "- OOM Killer invoked at line 15"
        return "NO_SIGNAL"

    with patch("ollama_bridge.engine.map_worker", side_effect=mock_worker) as m_work, \
         patch("ollama_bridge.engine.query_ollama", return_value="Final Executive Diagnostic Brief") as m_reduce:

        res = local_map_reduce_file(
            file_path=str(large_file),
            extraction_goal="Identify outages and kill events",
            chunk_size=6,
            overlap=2,
            concurrency=4,
            model="qwen2.5-coder:14b",
        )
        assert res == "Final Executive Diagnostic Brief"
        assert m_work.call_count > 1
        m_reduce.assert_called_once()
        kwargs = m_reduce.call_args[1]
        assert kwargs["model"] == "qwen2.5-coder:14b"
        assert kwargs["temperature"] == 0.1
        prompt = m_reduce.call_args[0][0] if m_reduce.call_args[0] else kwargs.get("prompt", "")
        assert "PRIMARY OBJECTIVE: Identify outages and kill events" in prompt
        assert "Database Connection Refused at line 5" in prompt
        assert "OOM Killer invoked at line 15" in prompt


@verifies("REQ-009")
def test_local_map_reduce_file_relative_path(tmp_path, monkeypatch):
    """Verify local_map_reduce_file handles relative paths."""
    monkeypatch.chdir(tmp_path)
    rel_file = "relative_test.log"
    with open(rel_file, "w") as f:
        f.write("Line 1\nLine 2\n")

    with patch("ollama_bridge.engine.query_ollama", return_value="Summary of rel"):
        res = local_map_reduce_file(rel_file, "Goal")
        assert res == "Summary of rel"

