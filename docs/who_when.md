# Who&When: Multi-Agent Failure Attribution

Given the trajectory of a multi-agent system that failed its task, the agent must identify the **responsible agent** (Who) and the **decisive error step** (When).

## Harness interaction

```text
normalized trajectory
        |
        v
structured summary + task  ---> model
                                  |
                      inspect(step_id_list)
                                  |
                   selected full step records
                                  |
              final_answer(responsible agent, decisive step)
```

- **Summary.** The summary lists step IDs, agent/role metadata, text shapes, and payload field names. It never contains step text, labels, or full nested records.
- **`inspect(step_id_list)`.** This is the only way to see a step's full record.
- **`final_answer`.** This tool submits the attribution and ends the episode. Exactly one tool call is executed per turn.

The seed harness (`harnesses/who_when/initial_harness`) has no planner, memory module, or sub-agent. Its only state is the set of inspected step IDs and the final result. The final evolved harness (`harnesses/who_when/generated`) keeps the same two-tool interface and observation boundary. Within that boundary, it changes:

- prompting;
- inspection coverage;
- budget management.

## Harness contract

Every evolved harness must preserve four properties:

- the failure-attribution task;
- the summary-first observation boundary, so no labels or step text leak into the initial context;
- the normalized data contract;
- the exact two-tool interface.

Each candidate is smoke-tested before evaluation. The checks cover import/build, harness identity, the tool interface, initial-context leakage, `inspect`, and `final_answer`.

## Evaluation

Validation and test follow the released Who&When protocol. Agent Accuracy and Step Accuracy are percentages over every reference record, and missing or malformed answers count as incorrect. Hand-Crafted and Algorithm-Generated results are reported separately. Who&When Pro and TraceElephant are evaluated as separate held-out test sets.

## Selection

Selection uses the validation split only; the test sets are evaluated once, for the final pair. The ranking metric weights Step above Agent:

```
Score = 0.3 * AgentAccuracy + 0.7 * StepAccuracy
```

- **Tie-breaking.** If every quality term is within epsilon, TIMR decides, and token cost after that.
- **Admission.** The overall Agent/Step regression tolerances control admission to the candidate pool.
- **Where it is used.** This ranking drives harness shortlisting, historical model selection, merge-candidate selection, and pre/post-training comparison.
- **Training data.** Before SFT, verified hard trajectories may be repeated so that trace length, decisive-step position, and agent count move toward the validation distribution.

## What evolved: `initial_harness` → `generated`

Both harnesses expose the same two tools and the same summary-first boundary. Everything that changed sits inside the tools and the prompt.

| Aspect | `initial_harness` | `generated` |
| --- | --- | --- |
| Turn budget | 10 | 12 |
| `inspect` request | Up to 3 exact steps | Up to 5 anchors, each returned with ±2 neighboring steps |
| `inspect` payload | Full raw records, truncated at 16k characters | Compact records within a 10k-character budget: field-aware head/tail clipping, component markers (`<memory>`, `<plan>`, `<action>`, …), and sub-step outlines |
| Coverage of long traces | Up to the model | Automatic scouting and a coverage gate before `final_answer` (below) |
| Answer schema | Decisive step taken from `evidence_step_ids[0]` | Explicit `decisive_step` field, which must be a step that was inspected |
| Attribution post-processing | None | Blame on observers is moved to the causal actor; role suffixes are normalized |

**Inspection.** Each requested anchor now comes back with its local context, so the model sees what led to a step and what followed it.

**Automatic scouting.** Suppose the first inspection stays within the opening 25% of a trace of six or more steps. The tool then also returns scout windows at about 20%, 45%, 75%, and 100% of the trajectory. It also suggests anchors at the start or midpoint of the largest uninspected gaps.

**Coverage gate.** For traces of 20 or more steps, `final_answer` is held back while a large chronological region is still uninspected. A region counts as large at eight or more steps, or three or more at the tail. The tool returns suggested anchors instead, at most twice per episode. This replaced the seed's tendency to stop after reading only the opening setup.

**Attribution rules.** Some answers blame `ComputerTerminal` or a tool step, which usually only reports the result of an earlier action. Such blame is moved to the preceding non-observer agent. Role labels such as `Agent (thought)` or `Agent (-> Tool)` are normalized to the base agent name, and boolean strings are parsed.

**Prompt.** The system prompt adds explicit guidance in three areas:

- **Inspection strategy:** read chronological windows, and sample early, middle, and late regions of long traces.
- **Attribution rules:** pick the earliest step that made success impossible rather than a later symptom. Check coordinator or verifier acceptance steps. Do not blame observer agents. Use component labels such as `<memory>` or `<plan>` when one of those sections is the failure locus.
- **Final-answer protocol:** submit `decisive_step` as an integer step ID that was inspected.
