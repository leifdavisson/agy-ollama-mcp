# Contributing to Ollama MCP Bridge (`agy-ollama-mcp`)

Thank you for your interest in contributing to the **Ollama MCP Bridge for Google Antigravity CLI**!

This project adheres to high-assurance software engineering practices inspired by **INCOSE Requirements Engineering** and **DO-178C Level A** Verification & Validation standards.

---

## 1. Governance & Quality Gates

To maintain production stability and high assurance, every contribution must satisfy these non-negotiable gates:

1. **License**: All contributions are licensed under the [GNU AGPLv3](file:///data/agy_ollama_mcp/LICENSE) ([Open LICENSE](file:///data/agy_ollama_mcp/LICENSE) (file:///data/agy_ollama_mcp/LICENSE)).
2. **Requirements Traceability**: Every feature must be formalized in [`requirements.json`](file:///data/agy_ollama_mcp/requirements.json) with acceptance criteria and a corresponding Gherkin feature file in [`features/`](file:///data/agy_ollama_mcp/features/).
3. **Spec-First Test Driven Development**: Tests must be written before implementation code and tagged with `@verifies("REQ-XXX")`.
4. **100% Statement and Branch Coverage**: All source modules in `src/ollama_bridge/` must maintain 100% statement and 100% branch coverage (`pytest --cov=src --cov-branch --cov-fail-under=100`).
5. **MC/DC Truth-Table Verification**: Any compound boolean decisions must demonstrate condition independence in [`scripts/verify_mcdc.py`](file:///data/agy_ollama_mcp/scripts/verify_mcdc.py).
6. **Strict Static Typing**: All code must pass `mypy --strict src/` with zero errors.
7. **Protocol Cleanliness**: The MCP daemon communicates via JSON-RPC 2.0 over STDIO. Stdout must never be polluted by print statements or logger messages; all diagnostic logging must route strictly to stderr.

---

## 2. Development Setup

Clone and install the repository in editable mode:

```bash
git clone https://github.com/leifdavisson/agy-ollama-mcp.git
cd agy-ollama-mcp

# Install runtime and development dependencies
pip install -e .
pip install -r requirements-dev.txt
```

---

## 3. Local Verification Suite

Before submitting a Pull Request, run the full verification battery:

```bash
# 1. Static type checking
mypy --strict src/

# 2. Test suite with 100% statement and branch coverage
pytest --cov=src --cov-branch --cov-fail-under=100 --cov-report=term-missing tests/

# 3. DO-178C Level A MC/DC truth-table verification
python3 scripts/verify_mcdc.py

# 4. AST Requirements Traceability Matrix generation
python3 scripts/generate_rtm.py
```

---

## 4. Companion Projects & Attribution

- **[Ollama](https://ollama.com)**: Official documentation and runtime daemon.
- **[Model Context Protocol (MCP)](https://modelcontextprotocol.io)**: MCP specification and SDKs.
- **[FastMCP](https://github.com/jlowin/fastmcp)**: Pythonic MCP server framework.
- **[Qwen2.5-Coder](https://github.com/QwenLM/Qwen2.5-Coder)** & **[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)**: Local coding and reasoning models.
- **[Google Antigravity CLI](file:///home/leifdavisson/.gemini/config/skills/antigravity-guide/SKILL.md)**: Agentic coding harness.
