Feature: Local Code Drafting
  As an autonomous coding agent
  I want to draft code implementations locally
  So that cloud tokens are preserved for high-level architecture

  Scenario: Drafting code with language and context
    Given a task description "Write a thread-safe singleton in Python"
    And a target language "python"
    And an existing context "import threading"
    When the local_draft_code tool is executed
    Then the generated prompt must contain "Target Language/Framework: python"
    And the generated prompt must contain "import threading"
    And the generated prompt must contain "Write a thread-safe singleton in Python"
    And the system prompt must require production quality code
