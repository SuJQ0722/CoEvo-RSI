"""RBAC-specific MetaEvolver instructions."""

from __future__ import annotations

from typing import Any, Mapping


def build(config: Mapping[str, Any]) -> str:
    selection = config.get("selection") or {}
    safe_ex_tol = float(selection.get("safe_ex_tolerance", 0.0))
    ac_f1_tol = float(selection.get("ac_f1_tolerance", 0.0))
    objective = (
        "Training and validation each contain 50% Spider and 50% BIRD. Aim to improve "
        "SafeEX and AC-F1 on both datasets together; inspect each subset's failures "
        "instead of hiding a regression in one subset behind an aggregate gain. "
        "The existing aggregate selection thresholds remain unchanged.\n\n"
        if (config.get("data") or {}).get("balanced_spider_bird") else ""
    )
    return objective + f"""Domain: column-level Text2SQL RBAC with rule routing and agentic LLM fallback.

Headline metrics (in selection priority order):
  safe_ex      — correct SQL on allowed samples / all allowed samples (primary gate)
  ac_f1        — access-control F1 (primary gate)
  safe_deny    — correct refusal rate on denied samples
  violation_rate    — security breach rate (lower is better)
  over_refusal_rate — unnecessary refusal rate (lower is better)
  llm_rate     — fraction of samples that invoke the LLM (lower is better)
  rule_rate    — fraction decided by deterministic rules
  rule_accuracy — correctness of rule decisions

Selection: safe_ex tolerance {safe_ex_tol:g}, ac_f1 tolerance {ac_f1_tol:g}. A child
must not regress safe_ex or ac_f1, then must improve at least one of: safe_ex, ac_f1,
or llm_rate.

The harness is an agentic multi-turn system. The LLM receives two tools:
  test_sql(sql) — execute SQL against the sample database, returns rows or error
  final_answer(access_granted, sql) — submit the authorization decision with optional SQL

When access is granted, the LLM should generate SQL and may use test_sql to verify
correctness before submitting. When access is denied, the LLM calls final_answer
with access_granted=false. The LLM may iterate over multiple turns to refine its SQL.

Evolution possibilities are broad. Consider improvements to:
  - Rule routing: better column linking, schema normalization, policy coverage
  - Prompt engineering: clearer instructions, chain-of-thought, few-shot examples
  - Agent loop patterns: multi-turn refinement, self-verification strategies
  - SQL generation quality: schema-aware prompting, error recovery loops
  - Any other structural innovations that improve the measured metrics

Prevent false filtering: do not add broad keyword deny/allow lists, default-deny
requests merely to save calls, infer column access from the role name alone, or
optimize coverage without measuring rule accuracy. Unknown roles and unseen database
schemas are valid inputs and must normally defer to the LLM.

Never encode validation/test instructions, sample IDs, gold labels, reference
answers, SQL, metadata, or test_type-specific answers. Validation is a fixed public
subset of the full test manifest; use it to learn general mechanisms, never memorize
its questions or policies. Preserve build_harness(policy_rows),
route(), messages(), call_llm(model, instruction, test_role, route, db_path=...),
parse_access(), the final_answer tool, both formal ACCESS markers, and 0/1 outputs.
Keep LLM request execution inside the imported llm.py component.
The candidate may evolve any local module, including __init__.py, as long as the
public build_harness contract and required runtime behavior remain intact.
Only the configured training policy rows may be compiled into rule/index state."""
