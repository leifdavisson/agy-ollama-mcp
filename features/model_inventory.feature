Feature: Model Inventory and Prewarming
  As an operator
  I want to inspect local Ollama models and prewarm them in memory
  So that execution is transparent and latency is eliminated

  Scenario: Listing models parses sizes and parameters
    Given Ollama returns a model "deepseek-r1:14b" with size 8988112209 bytes
    When local_list_models is executed
    Then the output contains "deepseek-r1:14b"
    And the reported size is "8.37 GB"
    And the active default model is indicated

  Scenario: Prewarming issues keep-alive request
    Given a target model "qwen2.5-coder:14b"
    When local_prewarm_model is executed
    Then a POST request is sent to "/api/generate" with keep_alive set to -1
    And a confirmation message is returned
