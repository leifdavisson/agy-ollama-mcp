"""
Test Engine Module: Drafting, Summarization, Line-aware Chunking & Map-Reduce, JSON Extraction.
Specifications:
- REQ-002: Local Code Drafting Execution
- REQ-003: Dense Summarization & Noise Extraction
- REQ-004: Chunked Map-Reduce Summarization
- REQ-005: Strict Structured JSON Extraction
Features:
- features/code_drafting.feature
- features/summarization.feature
- features/json_extraction.feature

Includes:
1. @verifies(req_id) traceability decorator.
2. Spec-first unit tests mapped to Gherkin scenarios.
3. MC/DC test vectors for compound boolean decision branches.
4. Hypothesis property-based invariant tests for line-aware chunking.

License: GNU AGPLv3
"""
import os
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, call
import pytest
from hypothesis import given, settings, strategies as st

# Ensure src and root are available in sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(repo_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Import target implementation module
try:
    from ollama_bridge import engine
    from ollama_bridge.engine import (
        local_draft_code,
        local_summarize_and_extract,
        local_chunked_summary,
        local_extract_json,
        partition_chunks,
    )
except ImportError:
    # Fallback for strict test harness initialization prior to module presence
    engine = None
    def partition_chunks(content: str, chunk_chars: int) -> List[str]:
        raise NotImplementedError("partition_chunks not implemented (RED phase)")

    def local_draft_code(task_description: str, context: str = "", language: str = "", model: str = "") -> str:
        raise NotImplementedError("local_draft_code not implemented (RED phase)")

    def local_summarize_and_extract(content: str, extraction_goal: str, model: str = "") -> str:
        raise NotImplementedError("local_summarize_and_extract not implemented (RED phase)")

    def local_chunked_summary(content: str, extraction_goal: str = "", chunk_chars: int = 12000, model: str = "") -> str:
        raise NotImplementedError("local_chunked_summary not implemented (RED phase)")

    def local_extract_json(content: str, schema_description: str, model: str = "") -> str:
        raise NotImplementedError("local_extract_json not implemented (RED phase)")


def verifies(req_id: str):
    """
    Decorator to trace test cases back to requirements.
    Attaches requirement metadata and registers pytest mark.
    """
    def decorator(fn):
        fn.verified_requirement = req_id
        return pytest.mark.requirement(req_id)(fn)
    return decorator


# ==============================================================================
# REQ-002: Local Code Drafting Execution
# Feature: features/code_drafting.feature
# ==============================================================================

@verifies("REQ-002")
def test_draft_code_prompt_assembly_gherkin_scenario():
    """
    Given a task description "Write a thread-safe singleton in Python"
    And a target language "python"
    And an existing context "import threading"
    When the local_draft_code tool is executed
    Then the generated prompt must contain "Target Language/Framework: python"
    And the generated prompt must contain "import threading"
    And the generated prompt must contain "Write a thread-safe singleton in Python"
    And the system prompt must require production quality code.
    """
    task = "Write a thread-safe singleton in Python"
    language = "python"
    context = "import threading"
    expected_code = "class Singleton:\n    _lock = threading.Lock()"

    with patch.object(engine, "query_ollama", return_value=expected_code) as mock_query:
        result = local_draft_code(
            task_description=task,
            context=context,
            language=language,
        )

        assert result == expected_code
        assert mock_query.call_count == 1
        call_args, call_kwargs = mock_query.call_args

        # Verify Prompt Assembly
        prompt = call_args[0] if call_args else call_kwargs.get("prompt", "")
        assert "Target Language/Framework: python" in prompt
        assert "import threading" in prompt
        assert "Write a thread-safe singleton in Python" in prompt

        # Verify System Prompt requires clean, modular production code
        system = call_kwargs.get("system", "")
        assert "expert software engineer" in system.lower()
        assert "clean" in system.lower()
        assert "production" in system.lower()


@verifies("REQ-002")
def test_draft_code_returns_model_response_on_http_200():
    """Verifies that on HTTP 200, the raw generated code response string is returned."""
    mock_code = "def solve(): return 42"
    with patch.object(engine, "query_ollama", return_value=mock_code):
        res = local_draft_code(task_description="Implement solver")
        assert res == mock_code


@verifies("REQ-002")
def test_draft_code_error_propagation_on_failure():
    """Verifies that connection or HTTP errors return descriptive error messages."""
    error_msg = "Error: Unable to connect to Ollama at http://localhost:11434"
    with patch.object(engine, "query_ollama", return_value=error_msg):
        res = local_draft_code(task_description="Implement solver")
        assert "Error: Unable to connect to Ollama" in res


# ==============================================================================
# MC/DC Vectors: REQ-002 local_draft_code Prompt Assembly
# Compound Decision: (has_language) AND (has_context)
# ==============================================================================

@verifies("REQ-002")
def test_mcdc_draft_code_vector1_both_language_and_context():
    """MC/DC Vector 1: Language=True, Context=True -> Both blocks included in prompt."""
    with patch.object(engine, "query_ollama", return_value="ok") as mock_query:
        local_draft_code(task_description="Task A", language="rust", context="fn main() {}")
        prompt = mock_query.call_args[0][0] if mock_query.call_args[0] else mock_query.call_args[1]["prompt"]
        assert "Target Language/Framework: rust" in prompt
        assert "Context / Existing Code:\nfn main() {}" in prompt
        assert "Task Description:\nTask A" in prompt


@verifies("REQ-002")
def test_mcdc_draft_code_vector2_language_only_no_context():
    """MC/DC Vector 2: Language=True, Context=False -> Language included, Context excluded."""
    with patch.object(engine, "query_ollama", return_value="ok") as mock_query:
        local_draft_code(task_description="Task B", language="rust", context="")
        prompt = mock_query.call_args[0][0] if mock_query.call_args[0] else mock_query.call_args[1]["prompt"]
        assert "Target Language/Framework: rust" in prompt
        assert "Context / Existing Code:" not in prompt
        assert "Task Description:\nTask B" in prompt


@verifies("REQ-002")
def test_mcdc_draft_code_vector3_context_only_no_language():
    """MC/DC Vector 3: Language=False, Context=True -> Language excluded, Context included."""
    with patch.object(engine, "query_ollama", return_value="ok") as mock_query:
        local_draft_code(task_description="Task C", language="", context="const x = 10;")
        prompt = mock_query.call_args[0][0] if mock_query.call_args[0] else mock_query.call_args[1]["prompt"]
        assert "Target Language/Framework:" not in prompt
        assert "Context / Existing Code:\nconst x = 10;" in prompt
        assert "Task Description:\nTask C" in prompt


@verifies("REQ-002")
def test_mcdc_draft_code_vector4_neither_language_nor_context():
    """MC/DC Vector 4: Language=False, Context=False -> Neither block included."""
    with patch.object(engine, "query_ollama", return_value="ok") as mock_query:
        local_draft_code(task_description="Task D", language="", context="")
        prompt = mock_query.call_args[0][0] if mock_query.call_args[0] else mock_query.call_args[1]["prompt"]
        assert "Target Language/Framework:" not in prompt
        assert "Context / Existing Code:" not in prompt
        assert "Task Description:\nTask D" in prompt


# ==============================================================================
# REQ-003: Dense Summarization & Noise Extraction
# Feature: features/summarization.feature
# ==============================================================================

@verifies("REQ-003")
def test_summarize_and_extract_prompt_assembly_and_noise_stripping():
    """
    Verifies local_summarize_and_extract prompt construction and noise-stripping
    system prompt instructions.
    """
    raw_content = "2026-09-06T12:00:01 INFO [main] [trace] Database connected\n2026-09-06T12:00:02 ERROR [main] FATAL: Connection pool exhausted"
    goal = "Identify fatal database exceptions"
    expected_summary = "FATAL: Connection pool exhausted"

    with patch.object(engine, "query_ollama", return_value=expected_summary) as mock_query:
        result = local_summarize_and_extract(
            content=raw_content,
            extraction_goal=goal,
            model="qwen2.5-coder:14b",
        )

        assert result == expected_summary
        assert mock_query.call_count == 1
        call_args, call_kwargs = mock_query.call_args

        prompt = call_args[0] if call_args else call_kwargs.get("prompt", "")
        assert f"Target Goal: {goal}" in prompt
        assert raw_content in prompt

        # Verify noise stripping instructions in system prompt
        system = call_kwargs.get("system", "")
        assert "dense technical extraction" in system.lower()
        assert "noise" in system.lower()
        assert "boilerplate" in system.lower()
        assert "timestamps" in system.lower()

        # Verify model parameter passing
        assert call_kwargs.get("model") == "qwen2.5-coder:14b"


@verifies("REQ-003")
def test_summarize_and_extract_returns_distilled_content():
    """Verifies that distilled summary is returned cleanly without conversational filler."""
    with patch.object(engine, "query_ollama", return_value="Distilled: OOM Killed at PID 102"):
        res = local_summarize_and_extract("Long trace", "Find OOM")
        assert res == "Distilled: OOM Killed at PID 102"


# ==============================================================================
# REQ-004: Chunked Map-Reduce Summarization & Line-Aware Chunking
# Feature: features/summarization.feature
# ==============================================================================

@verifies("REQ-004")
def test_chunked_summary_single_pass_when_len_le_chunk_chars():
    """
    Scenario: Single pass summarization for small content
    Given content of length 500 characters
    And an extraction goal "Identify fatal database exceptions"
    When local_chunked_summary is called with chunk limit 1000
    Then exactly 1 extraction pass is executed
    And the prompt includes the extraction goal "Identify fatal database exceptions"
    """
    content = "Line of log output: system healthy.\n" * 14  # ~504 chars
    content = content[:500]
    assert len(content) == 500
    goal = "Identify fatal database exceptions"

    with patch.object(engine, "local_summarize_and_extract", return_value="Pass 1 summary") as mock_extract, \
         patch.object(engine, "query_ollama") as mock_query:
        result = local_chunked_summary(
            content=content,
            extraction_goal=goal,
            chunk_chars=1000,
        )

        assert result == "Pass 1 summary"
        assert mock_extract.call_count == 1
        assert mock_extract.call_args[0][0] == content
        assert mock_extract.call_args[0][1] == goal

        # Exactly 1 extraction pass; synthesis pass must NOT execute
        assert mock_query.call_count == 0


@verifies("REQ-004")
def test_chunked_summary_multi_pass_map_reduce_large_content():
    """
    Scenario: Map-reduce chunking for large content exceeding chunk size
    Given content of length 2500 characters with 10 lines
    And an extraction goal "Identify memory leak stack traces"
    When local_chunked_summary is called with chunk limit 500
    Then the content is partitioned into multiple chunks without line splitting
    And an intermediate summary is generated for each chunk
    And a synthesis pass combines all intermediate chunk summaries into a final brief.
    """
    # 10 lines of 250 characters each (including newline) -> 2500 characters total
    lines = [f"Line {i:02d}: " + ("x" * 240) + "\n" for i in range(10)]
    content = "".join(lines)
    assert len(content) == 2500
    assert len(content.splitlines()) == 10

    goal = "Identify memory leak stack traces"

    def fake_extract(chunk: str, sub_goal: str, model: str = ""):
        return f"Summary of {chunk[:10].strip()}"

    with patch.object(engine, "local_summarize_and_extract", side_effect=fake_extract) as mock_extract, \
         patch.object(engine, "query_ollama", return_value="Final synthesized brief") as mock_query:
        result = local_chunked_summary(
            content=content,
            extraction_goal=goal,
            chunk_chars=500,
            model="qwen2.5-coder:14b",
        )

        assert result == "Final synthesized brief"

        # Intermediate extraction calls: 2500 / 500 = 5 chunks (2 lines per chunk = 500 chars)
        assert mock_extract.call_count == 5

        # Verify intermediate sub-goals contain part indexing
        for i, call_item in enumerate(mock_extract.call_args_list):
            chunk_arg, sub_goal_arg = call_item[0][:2]
            assert f"Part {i+1}/5 of content. {goal}" in sub_goal_arg
            # Verify line splitting did not occur within chunks
            assert chunk_arg.endswith("\n")

        # Verify synthesis pass
        assert mock_query.call_count == 1
        synthesis_prompt = mock_query.call_args[0][0]
        synthesis_system = mock_query.call_args[1].get("system", "")

        assert "Consolidate and synthesize" in synthesis_prompt
        assert f"Goal: {goal}" in synthesis_prompt
        assert "Chunk Summaries:" in synthesis_prompt
        assert "--- Chunk 1/5 Summary ---" in synthesis_prompt
        assert "master technical editor" in synthesis_system.lower()


# ==============================================================================
# MC/DC Vectors: REQ-004 Chunked Summary Decision Logic
# Decision 1: len(content) <= chunk_chars (Single vs Multi-pass)
# Decision 2: len(chunks) <= 1 (Single chunk after partition -> bypass synthesis)
# ==============================================================================

@verifies("REQ-004")
def test_mcdc_chunked_summary_vector1_strictly_less_than_limit():
    """MC/DC Vector 1: len(content) < chunk_chars -> Single pass directly."""
    content = "short single line log\n"
    with patch.object(engine, "local_summarize_and_extract", return_value="Single") as mock_extract, \
         patch.object(engine, "query_ollama") as mock_query:
        res = local_chunked_summary(content, chunk_chars=100)
        assert res == "Single"
        assert mock_extract.call_count == 1
        assert mock_query.call_count == 0


@verifies("REQ-004")
def test_mcdc_chunked_summary_vector2_exact_boundary_len_eq_chunk_chars():
    """MC/DC Vector 2: Boundary len(content) == chunk_chars -> Single pass directly."""
    content = "x" * 100
    with patch.object(engine, "local_summarize_and_extract", return_value="Boundary Single") as mock_extract, \
         patch.object(engine, "query_ollama") as mock_query:
        res = local_chunked_summary(content, chunk_chars=100)
        assert res == "Boundary Single"
        assert mock_extract.call_count == 1
        assert mock_query.call_count == 0


@verifies("REQ-004")
def test_mcdc_chunked_summary_vector3_boundary_len_plus_one_triggers_multipass():
    """
    MC/DC Vector 3: Boundary len(content) == chunk_chars + 1 -> Multi-pass triggered.
    Proves len(content) <= chunk_chars independently controls execution mode.
    """
    # 2 lines of 51 chars each = 102 chars; chunk_chars = 100
    line1 = ("a" * 50) + "\n"
    line2 = ("b" * 50) + "\n"
    content = line1 + line2
    assert len(content) == 102

    with patch.object(engine, "local_summarize_and_extract", return_value="Chunk summary") as mock_extract, \
         patch.object(engine, "query_ollama", return_value="Synthesized") as mock_query:
        res = local_chunked_summary(content, chunk_chars=100)
        assert res == "Synthesized"
        assert mock_extract.call_count == 2
        assert mock_query.call_count == 1


@verifies("REQ-004")
def test_mcdc_chunked_summary_vector4_single_chunk_after_partition_bypasses_synthesis():
    """
    MC/DC Vector 4: len(content) > chunk_chars BUT partition produces exactly 1 chunk
    (e.g., a single indivisible line of 150 chars with chunk_chars = 100).
    Outcome: Returns intermediate chunk summary directly; suppresses redundant synthesis.
    """
    indivisible_content = "Z" * 150  # 150 chars without newline
    with patch.object(engine, "local_summarize_and_extract", return_value="Only Chunk Summary") as mock_extract, \
         patch.object(engine, "query_ollama") as mock_query:
        res = local_chunked_summary(indivisible_content, chunk_chars=100)
        assert "Only Chunk Summary" in res
        assert mock_extract.call_count == 1
        assert mock_query.call_count == 0


# ==============================================================================
# MC/DC Vectors: partition_chunks Line Accumulation
# Predicate: (current_len + len(line) > chunk_chars) AND bool(current_chunk)
# ==============================================================================

@verifies("REQ-004")
def test_mcdc_partition_chunks_vector1_overflow_with_existing_chunk():
    """MC/DC Vector 1: Overflow=True, CurrentChunk=True -> Flushes chunk, starts new chunk."""
    content = "Line 1\nLine 2\nLine 3\n"
    # "Line 1\n" is 7 chars. "Line 2\n" is 7 chars. chunk_chars=10 forces flush after line 1.
    chunks = partition_chunks(content, chunk_chars=10)
    assert len(chunks) == 3
    assert chunks == ["Line 1\n", "Line 2\n", "Line 3\n"]
    assert "".join(chunks) == content


@verifies("REQ-004")
def test_mcdc_partition_chunks_vector2_overflow_with_empty_chunk():
    """
    MC/DC Vector 2: Overflow=True, CurrentChunk=False -> Line alone exceeds chunk_chars.
    Must NOT flush empty chunk; accepts line into current chunk without dropping it.
    """
    long_line = ("L" * 30) + "\n"
    short_line = "S\n"
    content = long_line + short_line
    chunks = partition_chunks(content, chunk_chars=10)
    assert len(chunks) == 2
    assert chunks[0] == long_line
    assert chunks[1] == short_line
    assert "".join(chunks) == content


@verifies("REQ-004")
def test_mcdc_partition_chunks_vector3_fits_with_existing_chunk():
    """MC/DC Vector 3: Overflow=False, CurrentChunk=True -> Appends to existing chunk."""
    content = "A\nB\nC\n"
    chunks = partition_chunks(content, chunk_chars=20)
    assert len(chunks) == 1
    assert chunks[0] == content


@verifies("REQ-004")
def test_mcdc_partition_chunks_vector4_empty_input():
    """MC/DC Vector 4: Empty input content string -> Returns empty list."""
    chunks = partition_chunks("", chunk_chars=100)
    assert chunks == []


# ==============================================================================
# REQ-005: Strict Structured JSON Extraction
# Feature: features/json_extraction.feature
# ==============================================================================

@verifies("REQ-005")
def test_extract_json_gherkin_scenario():
    """
    Given unstructured text "Server started at port 8080 with worker count 4"
    And target schema "{port: int, workers: int}"
    When local_extract_json is executed
    Then the query temperature must be 0.1
    And the system prompt must strictly forbid commentary outside JSON.
    """
    unstructured_text = "Server started at port 8080 with worker count 4"
    schema = "{port: int, workers: int}"
    expected_json = '{"port": 8080, "workers": 4}'

    with patch.object(engine, "query_ollama", return_value=expected_json) as mock_query:
        result = local_extract_json(
            content=unstructured_text,
            schema_description=schema,
            model="qwen2.5-coder:14b",
        )

        assert result == expected_json
        assert mock_query.call_count == 1
        call_args, call_kwargs = mock_query.call_args

        # Verify strict low temperature for deterministic schema conformity
        assert call_kwargs.get("temperature") == 0.1

        # Verify prompt assembly
        prompt = call_args[0] if call_args else call_kwargs.get("prompt", "")
        assert f"Target Schema / Fields:\n{schema}" in prompt
        assert f"Content:\n{unstructured_text}" in prompt

        # Verify system prompt strictly forbids commentary
        system = call_kwargs.get("system", "")
        assert "strict data extraction" in system.lower()
        assert "only a valid json" in system.lower()
        assert "do not include markdown conversational filler" in system.lower()


@verifies("REQ-005")
def test_extract_json_temperature_always_0_1_regardless_of_defaults():
    """Verifies that temperature 0.1 is enforced deterministically for schema conformity."""
    with patch.object(engine, "query_ollama", return_value='{}') as mock_query:
        local_extract_json("any content", "any schema")
        assert mock_query.call_args[1].get("temperature") == 0.1


# ==============================================================================
# Hypothesis Property-Based Invariant Tests (Rule 6)
# Invariant: Chunked partition never loses lines or characters when joined
# ==============================================================================

@verifies("REQ-004")
@given(
    content=st.text(min_size=0, max_size=5000),
    chunk_chars=st.integers(min_value=1, max_value=2000),
)
@settings(max_examples=100)
def test_hypothesis_partition_lossless_characters_and_lines_invariant(content: str, chunk_chars: int):
    """
    Property: For ANY arbitrary text (unicode, multiline, empty, whitespace) and any chunk_chars >= 1,
    joining the partitioned chunks MUST exactly equal the original text without losing a single character.
    Invariant: "".join(partition_chunks(content, chunk_chars)) == content.
    """
    chunks = partition_chunks(content, chunk_chars)
    reconstructed = "".join(chunks)
    assert reconstructed == content, f"Character loss detected! Original len={len(content)}, Reconstructed len={len(reconstructed)}"


@verifies("REQ-004")
@given(
    lines=st.lists(
        st.text(alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\n\r"), max_size=80),
        min_size=1,
        max_size=30,
    ),
    chunk_chars=st.integers(min_value=20, max_value=500),
)
@settings(max_examples=60)
def test_hypothesis_partition_never_cuts_lines_invariant(lines: List[str], chunk_chars: int):
    """
    Property: Lines are never cut in half when lines fit within chunk_chars.
    Every chunk boundary must fall on a newline boundary.
    """
    content = "\n".join(lines) + "\n"
    chunks = partition_chunks(content, chunk_chars)

    # 1. Lossless reconstruction
    assert "".join(chunks) == content

    # 2. Line boundary preservation: each chunk must end with a newline
    for chunk in chunks:
        assert chunk.endswith("\n"), f"Chunk did not end at line boundary: {chunk!r}"

    # 3. Line split parity: splitlines of reconstructed chunks must equal original splitlines
    original_split = content.splitlines(keepends=True)
    chunked_split = [line for c in chunks for line in c.splitlines(keepends=True)]
    assert chunked_split == original_split


@verifies("REQ-004")
@given(
    content=st.text(min_size=1, max_size=1000),
    chunk_chars=st.integers(min_value=1001, max_value=5000),
)
@settings(max_examples=40)
def test_hypothesis_partition_single_chunk_under_limit(content: str, chunk_chars: int):
    """
    Property: If len(content) <= chunk_chars, partition_chunks must yield at most 1 chunk.
    """
    chunks = partition_chunks(content, chunk_chars)
    assert len(chunks) <= 1
    if chunks:
        assert chunks[0] == content
