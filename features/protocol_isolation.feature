Feature: Protocol Isolation and Logging
  As an MCP client
  I want STDIO output to contain only valid JSON-RPC
  So that logging never corrupts the communication channel

  Scenario: Logging output is routed to standard error
    Given the MCP server is initialized
    When internal diagnostic events occur
    Then nothing is written to stdout except JSON-RPC formatted lines
    And diagnostic logs appear exclusively on stderr
