Feature: Structured JSON Extraction
  As a structured data pipeline
  I want to extract typed JSON from unstructured logs
  So that downstream tools receive deterministic schemas

  Scenario: Extracting JSON matching schema
    Given unstructured text "Server started at port 8080 with worker count 4"
    And target schema "{port: int, workers: int}"
    When local_extract_json is executed
    Then the query temperature must be 0.1
    And the system prompt must strictly forbid commentary outside JSON
