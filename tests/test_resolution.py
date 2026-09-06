"""
Spec-First Unit Tests, Hypothesis Property-Based Tests, and MC/DC Test Vectors
for Model Resolution & Config Module (REQ-001, REQ-008).

Specifications:
- features/model_resolution.feature
- features/protocol_isolation.feature

License: GNU AGPLv3
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, strategies as st

# Ensure src/ directory and workspace root are available on sys.path
SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ollama_bridge.config import (
    CONFIGURED_MODEL,
    DEFAULT_FALLBACK_MODEL,
    DEFAULT_NUM_CTX,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    OLLAMA_HOST,
    PREFERRED_MODELS,
    configure_logging,
    get_configured_model,
)
from ollama_bridge.resolution import (
    resolve_model,
    select_preferred_model,
)


def verifies(req_id: str) -> Callable[[Any], Any]:
    """
    Traceability decorator linking tests to functional requirement IDs.
    Attaches __req_id__ and _verifies metadata, and registers pytest mark.
    Directly compatible with scripts/generate_rtm.py AST visitor.
    """
    def decorator(fn: Any) -> Any:
        if not hasattr(fn, "_verifies"):
            fn._verifies = []
        fn._verifies.append(req_id)
        setattr(fn, "__req_id__", req_id)
        return pytest.mark.verifies(req_id)(fn)
    return decorator


# ==============================================================================
# SPEC-FIRST TESTS: Gherkin Scenarios from features/model_resolution.feature
# ==============================================================================

@verifies("REQ-001")
def test_spec_scenario_1_explicit_model_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Scenario: Explicit model parameter takes highest precedence
    Given the environment variable "LOCAL_LLM_MODEL" is "qwen2.5-coder:14b"
    When model resolution is requested with explicit model "deepseek-r1:14b"
    Then the resolved model must be "deepseek-r1:14b"
    """
    monkeypatch.setenv("LOCAL_LLM_MODEL", "qwen2.5-coder:14b")
    available_models = [
        {"name": "llama3.2:latest"},
        {"name": "qwen2.5-coder:14b"},
    ]
    resolved = resolve_model("deepseek-r1:14b", available_models=available_models)
    assert resolved == "deepseek-r1:14b"


@verifies("REQ-001")
def test_spec_scenario_2_env_var_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Scenario: Configured environment variable takes precedence when no explicit model is provided
    Given the environment variable "LOCAL_LLM_MODEL" is "dolphin3-tools:latest"
    When model resolution is requested with explicit model ""
    Then the resolved model must be "dolphin3-tools:latest"
    """
    monkeypatch.setenv("LOCAL_LLM_MODEL", "dolphin3-tools:latest")
    available_models = [
        {"name": "qwen2.5-coder:14b"},
        {"name": "llama3.2:latest"},
    ]
    resolved = resolve_model("", available_models=available_models)
    assert resolved == "dolphin3-tools:latest"


@verifies("REQ-001")
def test_spec_scenario_3_autodetect_preferred_coder_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Scenario: Auto-detection selects preferred coder model when available
    Given the environment variable "LOCAL_LLM_MODEL" is ""
    And the available models in Ollama are "llama3.2:latest,qwen2.5-coder:14b,dolphin3:latest"
    When model resolution is requested with explicit model ""
    Then the resolved model must be "qwen2.5-coder:14b"
    """
    monkeypatch.setenv("LOCAL_LLM_MODEL", "")
    available_models = [
        {"name": "llama3.2:latest"},
        {"name": "qwen2.5-coder:14b"},
        {"name": "dolphin3:latest"},
    ]
    resolved = resolve_model("", available_models=available_models)
    assert resolved == "qwen2.5-coder:14b"


@verifies("REQ-001")
def test_spec_scenario_4_fallback_model_when_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Scenario: Fallback model is selected when Ollama is unreachable
    Given the environment variable "LOCAL_LLM_MODEL" is ""
    And Ollama is unreachable (available_models=None or network error)
    When model resolution is requested with explicit model ""
    Then the resolved model must be "qwen2.5-coder:14b"
    """
    monkeypatch.setenv("LOCAL_LLM_MODEL", "")
    resolved = resolve_model("", available_models=None)
    assert resolved == "qwen2.5-coder:14b"


# ==============================================================================
# UNIT TESTS: Precedence Boundary Cases & Auto-Detection Hierarchy (REQ-001)
# ==============================================================================

@verifies("REQ-001")
def test_explicit_model_whitespace_is_trimmed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit argument with surrounding whitespace is stripped and honored."""
    monkeypatch.setenv("LOCAL_LLM_MODEL", "qwen2.5-coder:14b")
    resolved = resolve_model("   deepseek-r1:14b \t\n  ")
    assert resolved == "deepseek-r1:14b"


@verifies("REQ-001")
def test_env_var_whitespace_is_trimmed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configured environment variable with surrounding whitespace is stripped."""
    monkeypatch.setenv("LOCAL_LLM_MODEL", "   dolphin3-tools:latest \t  ")
    resolved = resolve_model("")
    assert resolved == "dolphin3-tools:latest"


@verifies("REQ-001")
def test_autodetect_respects_preferred_hierarchy_ranking(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    When multiple preferred models are available, resolution chooses the one
    with the highest rank in PREFERRED_MODELS order.
    qwen2.5-coder:7b precedes deepseek-r1:14b in preferred hierarchy.
    """
    monkeypatch.setenv("LOCAL_LLM_MODEL", "")
    available = [
        {"name": "deepseek-r1:14b"},
        {"name": "qwen2.5-coder:7b"},
        {"name": "llama3.2:latest"},
    ]
    resolved = resolve_model("", available_models=available)
    assert resolved == "qwen2.5-coder:7b"


@verifies("REQ-001")
def test_autodetect_heuristic_coder_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    When no exact match exists in PREFERRED_MODELS, any model containing 'coder'
    is selected before non-coding models.
    """
    monkeypatch.setenv("LOCAL_LLM_MODEL", "")
    available = [
        {"name": "general-ai:latest"},
        {"name": "starcoder2:15b"},
        {"name": "other-agent:latest"},
    ]
    resolved = resolve_model("", available_models=available)
    assert resolved == "starcoder2:15b"


@verifies("REQ-001")
def test_autodetect_heuristic_code_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    When no exact match exists in PREFERRED_MODELS, any model containing 'code'
    is selected before non-coding models.
    """
    monkeypatch.setenv("LOCAL_LLM_MODEL", "")
    available = [
        {"name": "general-assistant:latest"},
        {"name": "magic-code:7b"},
    ]
    resolved = resolve_model("", available_models=available)
    assert resolved == "magic-code:7b"


@verifies("REQ-001")
def test_autodetect_first_available_when_no_coder_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    When available models do not match preferred list or coder heuristics,
    the first available model is selected.
    """
    monkeypatch.setenv("LOCAL_LLM_MODEL", "")
    available = [
        {"name": "custom-domain-model:8b"},
        {"name": "arbitrary-backup:latest"},
    ]
    resolved = resolve_model("", available_models=available)
    assert resolved == "custom-domain-model:8b"


@verifies("REQ-001")
def test_fallback_when_available_list_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty available model list falls back to DEFAULT_FALLBACK_MODEL."""
    monkeypatch.setenv("LOCAL_LLM_MODEL", "")
    resolved = resolve_model("", available_models=[])
    assert resolved == "qwen2.5-coder:14b"


@verifies("REQ-001")
def test_select_preferred_model_helper_standalone() -> None:
    """Unit test for select_preferred_model helper function."""
    candidates = ["llama3.2:latest", "qwen2.5-coder:14b", "misc:latest"]
    selected = select_preferred_model(candidates)
    assert selected == "qwen2.5-coder:14b"


# ==============================================================================
# MC/DC TRUTH-TABLE VECTORS: Compound Boolean Predicates (REQ-001)
# Decision: Model Source Resolution
# Predicate: Direct Specification = (A: has_explicit or B: has_env)
#             Else: (C: has_autodetect) vs Fallback
# ==============================================================================

@verifies("REQ-001")
@pytest.mark.parametrize(
    "vector_id,requested_model,env_model,available_models,expected_model,branch_desc",
    [
        # Vector 1: A=True, B=True, C=True -> Explicit wins (A independent of B & C)
        (
            "MCDC-V1",
            "deepseek-r1:14b",
            "dolphin3-tools:latest",
            [{"name": "llama3.2:latest"}],
            "deepseek-r1:14b",
            "Explicit argument overrides configured env and autodetect",
        ),
        # Vector 2: A=True, B=False, C=False -> Explicit wins in isolation
        (
            "MCDC-V2",
            "deepseek-r1:14b",
            "",
            [],
            "deepseek-r1:14b",
            "Explicit argument in isolation with empty env and no models",
        ),
        # Vector 3: A=False, B=True, C=True -> Env wins (A toggles T->F from V1)
        (
            "MCDC-V3",
            "",
            "dolphin3-tools:latest",
            [{"name": "llama3.2:latest"}],
            "dolphin3-tools:latest",
            "Configured env var overrides auto-detection when explicit is empty",
        ),
        # Vector 4: A=False, B=False, C=True -> Autodetect wins (B toggles T->F from V3)
        (
            "MCDC-V4",
            "",
            "",
            [{"name": "llama3.2:latest"}],
            "llama3.2:latest",
            "Auto-detection resolves available model when explicit and env are empty",
        ),
        # Vector 5: A=False, B=False, C=False -> Fallback wins (C toggles T->F from V4)
        (
            "MCDC-V5",
            "",
            "",
            [],
            "qwen2.5-coder:14b",
            "Default fallback selected when no model is specified or available",
        ),
    ],
)
def test_mcdc_model_resolution_truth_table(
    vector_id: str,
    requested_model: str,
    env_model: str,
    available_models: List[Dict[str, Any]],
    expected_model: str,
    branch_desc: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MC/DC compound boolean predicate truth-table vector test."""
    monkeypatch.setenv("LOCAL_LLM_MODEL", env_model)
    resolved = resolve_model(requested_model, available_models=available_models)
    assert resolved == expected_model


# ==============================================================================
# HYPOTHESIS PROPERTY-BASED TESTS (REQ-001)
# ==============================================================================

@verifies("REQ-001")
@given(
    model_name=st.text(
        alphabet=st.characters(blacklist_categories=("Cs", "Cc")),
        min_size=1,
        max_size=64,
    ).filter(lambda s: bool(s.strip()))
)
@settings(max_examples=50)
def test_property_arbitrary_explicit_model_strings(model_name: str) -> None:
    """
    Property: Any arbitrary non-empty string provided as an explicit argument
    must resolve exactly to its stripped version without mutation.
    """
    resolved = resolve_model(model_name)
    assert resolved == model_name.strip()
    assert len(resolved) > 0
    assert not resolved.startswith(" ")
    assert not resolved.endswith(" ")


@verifies("REQ-001")
@given(
    whitespace_explicit=st.text(alphabet=" \t\n\r", min_size=0, max_size=20),
    env_target=st.sampled_from(["dolphin3-tools:latest", "qwen2.5-coder:7b", "deepseek-coder:6.7b"]),
)
@settings(max_examples=30)
def test_property_empty_whitespace_explicit_falls_through_to_env(
    whitespace_explicit: str,
    env_target: str,
) -> None:
    """
    Property: Empty or whitespace-only explicit argument strings must fall through
    to the configured environment variable.
    """
    with patch.dict(os.environ, {"LOCAL_LLM_MODEL": env_target}):
        resolved = resolve_model(whitespace_explicit)
        assert resolved == env_target
        assert resolved != whitespace_explicit


@verifies("REQ-001")
@given(
    arbitrary_explicit=st.one_of(st.none(), st.text(max_size=30)),
    arbitrary_env=st.one_of(st.none(), st.text(alphabet=st.characters(blacklist_characters="\x00"), max_size=30)),
    available_names=st.lists(st.text(min_size=1, max_size=30), max_size=4),
)
@settings(max_examples=50)
def test_property_resolution_invariant_always_non_empty(
    arbitrary_explicit: Optional[str],
    arbitrary_env: Optional[str],
    available_names: List[str],
) -> None:
    """
    Property Invariant: resolve_model MUST ALWAYS return a non-empty, stripped string
    representing a valid model identifier under all inputs and environment configurations.
    """
    env_dict = {"LOCAL_LLM_MODEL": arbitrary_env} if arbitrary_env is not None else {}
    with patch.dict(os.environ, env_dict, clear=(arbitrary_env is None)):
        req = "" if arbitrary_explicit is None else arbitrary_explicit
        models = [{"name": n} for n in available_names]
        resolved = resolve_model(req, available_models=models)

        assert isinstance(resolved, str)
        assert len(resolved) > 0
        assert resolved == resolved.strip()


# ==============================================================================
# PROTOCOL ISOLATION & LOGGING (REQ-008, features/protocol_isolation.feature)
# ==============================================================================

@verifies("REQ-008")
def test_protocol_isolation_logging_routes_exclusively_to_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Scenario: Logging output is routed to standard error
    Given the logger is configured
    When internal diagnostic events occur
    Then nothing is written to stdout except JSON-RPC formatted lines
    And diagnostic logs appear exclusively on stderr
    """
    logger = configure_logging(level=logging.INFO)
    logger.info("Diagnostic event: resolving model")
    logger.warning("Diagnostic warning: Ollama latency high")
    logger.error("Diagnostic error: model not found")

    captured = capsys.readouterr()
    # Stdout must remain completely empty to protect JSON-RPC transport integrity
    assert captured.out == ""
    # Stderr must receive the diagnostic messages
    assert "Diagnostic event: resolving model" in captured.err
    assert "Diagnostic warning: Ollama latency high" in captured.err
    assert "Diagnostic error: model not found" in captured.err


@verifies("REQ-008")
def test_protocol_isolation_no_stdout_pollution_during_resolution(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Model resolution must never write debug or print statements to stdout.
    """
    monkeypatch.setenv("LOCAL_LLM_MODEL", "")
    _ = resolve_model("test-model:latest", available_models=[])
    captured = capsys.readouterr()
    assert captured.out == ""


# ==============================================================================
# CONFIGURATION UNIT TESTS (REQ-001, REQ-008)
# ==============================================================================

@verifies("REQ-001")
def test_config_runtime_constants_defaults() -> None:
    """Verify runtime configuration constants and defaults match specifications."""
    assert OLLAMA_HOST == "http://localhost:11434"
    assert DEFAULT_TIMEOUT == 180
    assert DEFAULT_NUM_CTX == 16384
    assert DEFAULT_TEMPERATURE == 0.2
    assert DEFAULT_FALLBACK_MODEL == "qwen2.5-coder:14b"
    assert "qwen2.5-coder:14b" in PREFERRED_MODELS
    assert PREFERRED_MODELS[0] == "qwen2.5-coder:14b"


@verifies("REQ-001")
def test_config_get_configured_model_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify get_configured_model reads dynamically from LOCAL_LLM_MODEL."""
    monkeypatch.setenv("LOCAL_LLM_MODEL", "starcoder:latest")
    assert get_configured_model() == "starcoder:latest"
    monkeypatch.setenv("LOCAL_LLM_MODEL", "   ")
    assert get_configured_model() == ""
