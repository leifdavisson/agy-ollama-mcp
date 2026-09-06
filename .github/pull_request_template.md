## Description
<!-- Provide a clear summary of your changes and why they are needed. -->

## Related Requirements
<!-- List requirement IDs from requirements.json verified or added by this PR. E.g. REQ-009 -->
- Verified: `REQ-XXX`

## Verification Checklist
- [ ] Gherkin feature file updated or created in `features/`
- [ ] Spec-first tests decorated with `@verifies("REQ-XXX")`
- [ ] `mypy --strict src/` passes with 0 errors
- [ ] 100% statement and branch coverage maintained (`pytest --cov=src --cov-branch --cov-fail-under=100`)
- [ ] MC/DC condition independence verified (`python3 scripts/verify_mcdc.py`)
- [ ] AST RTM regenerated with 0 uncovered requirements (`python3 scripts/generate_rtm.py`)
- [ ] STDIO protocol cleanly isolated (stdout is pure JSON-RPC frames)
- [ ] Licensed under GNU AGPLv3
