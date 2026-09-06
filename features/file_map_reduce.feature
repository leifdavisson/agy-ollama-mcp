Feature: File-Direct Sliding-Window Map-Reduce
  As an autonomous engineering agent harness
  I want to process multi-megabyte log and source code files directly from the filesystem
  So that I can extract dense diagnostic briefs without loading massive strings into JSON-RPC STDIO transport.

  @REQ-009
  Scenario: Non-existent file path handling
    Given a non-existent file path "/tmp/does_not_exist.log"
    When local_map_reduce_file is invoked with the invalid path
    Then the system returns a descriptive error indicating the file does not exist.

  @REQ-009
  Scenario: Small file single-pass execution
    Given a file with line count less than or equal to chunk_size
    When local_map_reduce_file is invoked with goal "Extract error codes"
    Then the system executes a single-pass local Ollama distillation without chunking.

  @REQ-009
  Scenario: Multi-chunk sliding window with overlap
    Given a large log file exceeding chunk_size
    When local_map_reduce_file partitions the content with chunk_size 400 and overlap 50
    Then consecutive chunks share exactly 50 overlapping lines
    And no lines are truncated or lost.

  @REQ-009
  Scenario: Concurrent Map phase with NO_SIGNAL filtering
    Given multiple chunks to evaluate in parallel
    When map workers execute across worker threads
    Then chunks returning "NO_SIGNAL" are filtered out
    And only relevant findings are passed to the Reduce synthesis pass.
