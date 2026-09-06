Feature: Model Resolution
  As the Antigravity local bridge
  I want deterministic resolution of Ollama models
  So that execution targets the most capable available local model without ambiguity

  Scenario: Explicit model parameter takes highest precedence
    Given the environment variable "LOCAL_LLM_MODEL" is "qwen2.5-coder:14b"
    When model resolution is requested with explicit model "deepseek-r1:14b"
    Then the resolved model must be "deepseek-r1:14b"

  Scenario: Configured environment variable takes precedence when no explicit model is provided
    Given the environment variable "LOCAL_LLM_MODEL" is "dolphin3-tools:latest"
    When model resolution is requested with explicit model ""
    Then the resolved model must be "dolphin3-tools:latest"

  Scenario: Auto-detection selects preferred coder model when available
    Given the environment variable "LOCAL_LLM_MODEL" is ""
    And the available models in Ollama are "llama3.2:latest,qwen2.5-coder:14b,dolphin3:latest"
    When model resolution is requested with explicit model ""
    Then the resolved model must be "qwen2.5-coder:14b"

  Scenario: Fallback model is selected when Ollama is unreachable
    Given the environment variable "LOCAL_LLM_MODEL" is ""
    And Ollama is unreachable
    When model resolution is requested with explicit model ""
    Then the resolved model must be "qwen2.5-coder:14b"
