# HomeBench: Smart-Home Instruction Following

Given a household's devices and a user request, the agent must map the request to device-control commands and reject sub-instructions that cannot be carried out. We use the multi-operation subsets, in which each request contains 2–10 sub-instructions:

- VM: all sub-instructions are valid;
- IM: all sub-instructions are invalid;
- MM: valid and invalid sub-instructions are mixed.

Each sample is an independent request, and no real device is ever operated.

## Harness interaction

The model input has five parts:

1. **System prompt** (`prompt.txt`). It restricts the model to public device state and methods, explains relative changes and invalid input, and requires submission through `final_answer.commands`.
2. **Household index.** A compact room-to-device mapping for the current household.
3. **In-context examples.** Four evolvable synthetic examples, included when `harness.prompt_mode: icl`.
4. **User instruction.** The original request.
5. **Tool messages.** The model calls `inspect_home` and `list_methods` to read current values, ranges, options, and exact method signatures. It then calls `final_answer` exactly once.

The harness never sees scoring information such as the reference output, sample ID, or category. Tool schemas and execution are owned by the trusted domain runtime.

The seed harness (`harnesses/homebench/initial_harness`) is organized into a few modules:

- `settings.py`: options;
- `context.py`: the household index;
- `prompts.py`: prompt assembly;
- `harness.py`: conversation state;
- `factory.py`: the `build_agent` entry point.

The final evolved harness (`harnesses/homebench/generated`) adds three things:

- grounding of device and attribute references (`grounding.py`);
- pre-submission validation of commands (`validation.py`);
- refined prompts and examples.

## Evaluation

Scoring follows the official `compute_accuracy`:

1. strip spaces and newlines;
2. extract the content inside braces;
3. split on commas;
4. compute multiset (Counter) exact match together with micro precision, recall, and F1.

`submission.py` serializes the final `final_answer.commands` into the official text format, after checking that it comes from a genuine terminal tool call. Success rate and F1 are reported separately for VM and IM.

## Selection

A candidate harness must strictly improve EM on the same tasks without lowering F1. After training, a pair is accepted only if neither EM nor F1 regresses. Any infrastructure error makes an evaluation ineligible for promotion.

## What evolved: `initial_harness` → `generated`

Both harnesses keep the same module layout, tools, prompt mode, and the same four in-context examples. The evolved harness adds three things: grounding of names, an ordered frame for multi-operation requests, and a deterministic preflight before submission. None of these read gold labels.

| Aspect | `initial_harness` | `generated` |
| --- | --- | --- |
| Room/device names | Exact match against the household index | `grounding.py` folds case, punctuation, underscores, spacing, and repeated letters, and tolerates a one-letter difference for longer names. A phrasing that denotes exactly one delivered room is mapped to that room's spelling and shown as a hint |
| Multi-operation structure | Free-form | An ordered list of operations matching the declared arity (`multiN`), so each sub-instruction gets exactly one output slot, in request order, with no merging, dropping, or repeating |
| Reading observations | Not specified | Bounded rules for reusing earlier `inspect_home` / `list_methods` results instead of re-reading |
| Enum arguments | Not specified | Options are written exactly as printed and unquoted |
| Submission | Accepted as given | `validation.py` preflights every `final_answer` (below) |

**Submission preflight.** `SubmissionValidator` checks each submitted command against the public household data. It verifies:

- the number of slots against the declared arity;
- the room, device, and method names, with a hint naming the delivered spelling when a room is misspelled;
- argument types;
- enum options and numeric bounds;
- for relative requests such as "increase by 10", whether the value equals the observed current value plus or minus the delta.

When any check fails, the errors are returned to the model, which corrects only those slots and calls `final_answer` again. The system prompt states this protocol explicitly.
