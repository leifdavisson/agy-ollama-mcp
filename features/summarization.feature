Feature: High-Density Summarization and Chunking
  As a reasoning engine
  I want to compress large logs and files into high-density summaries
  So that large context windows do not inflate cloud reasoning costs

  Scenario: Single pass summarization for small content
    Given content of length 500 characters
    And an extraction goal "Identify fatal database exceptions"
    When local_chunked_summary is called with chunk limit 1000
    Then exactly 1 extraction pass is executed
    And the prompt includes the extraction goal "Identify fatal database exceptions"

  Scenario: Map-reduce chunking for large content exceeding chunk size
    Given content of length 2500 characters with 10 lines
    And an extraction goal "Identify memory leak stack traces"
    When local_chunked_summary is called with chunk limit 500
    Then the content is partitioned into multiple chunks without line splitting
    And an intermediate summary is generated for each chunk
    And a synthesis pass combines all intermediate chunk summaries into a final brief
