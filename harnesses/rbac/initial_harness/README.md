# Initial RBAC harness

The seed harness is deliberately conservative for Spider v3 column-level RBAC. It skips
the LLM for malformed inputs, a role/policy mismatch, a policy that explicitly
grants every column in every table, or a literal unavailable multi-token schema
column with complete per-table policy evidence. Unknown formats and ambiguous
requests defer to the LLM and force exactly one
`final_answer(access_granted: bool)` tool call. It does not contain fuzzy/keyword
rules.

Candidates may evolve `__init__.py` and any other local module while preserving the
`build_harness(policy_rows)` public contract and required runtime behavior. The seed
separates evolvable behavior into `routing.py`, `harness.py`, `prompts.py`, and
`response.py`; `llm.py` owns model calls plus usage, latency, and error normalization.
