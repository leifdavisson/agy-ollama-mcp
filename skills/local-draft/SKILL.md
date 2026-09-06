---
name: local-draft
description: Forces code drafting, AST parsing, or summarization onto the local Ollama instance to conserve cloud quota and token limits.
---

# Local Ollama Offloading Skill for Antigravity CLI

This skill instructs the Antigravity agent to delegate token-heavy preliminary tasks (boilerplate code drafting, log parsing, documentation summaries, data extraction) to the local Ollama MCP bridge (`local-ollama`) before routing high-signal results to cloud reasoning.

## When to Invoke This Skill
- The user uses `/local-draft <task>`
- Large files, stack traces, or terminal logs (> 2,000 tokens) need summarization or error extraction
- Boilerplate code, unit test suites, or initial file scaffolding is being generated
- Context limits or quota exhaustion need to be minimized

## Instructions for Antigravity Agent:
1. **Pre-filter Large Contexts Locally:**
   - Do NOT pass large raw logs, database dumps, or complete file contents directly into your cloud context.
   - For logs or files under 12,000 characters, invoke `local-ollama`'s `local_summarize_and_extract(content, extraction_goal)`.
   - For multi-megabyte files or massive logs, invoke `local-ollama`'s `local_chunked_summary(content, extraction_goal)`.
2. **Draft Boilerplate & Scaffolding Locally:**
   - Call `local-ollama`'s `local_draft_code(task_description, context, language)`.
   - Specify the target language or framework explicitly when known.
3. **Structured Data Extraction:**
   - Use `local_extract_json(content, schema_description)` to parse unstructured outputs into strict JSON schemas.
4. **Cloud Review & Integration:**
   - Take the locally generated output and review it critically for accuracy, workspace conformity, and edge cases.
   - Present the polished code or concise architectural insight to the user.
