def build(config):
    return """Domain: HomeBench multi-operation household instruction generation (2–10
sub-instructions per query, VM/IM/MM). Improve exact-match EM
while retaining micro command F1. Preserve build_agent(HarnessSettings, client=),
initialize(instruction, public_home, tools), step(), receive_tool_result(message).
Each step makes exactly one model request and returns its actual message with
one tool call. final_answer.commands must contain brace-enclosed instructions.
The runtime supplies read-only inspect_home/list_methods and scores the final
string using the fixed official Counter semantics (including repeated errors).
Improve context selection, state grounding for relative changes, room/device
resolution, compound request coverage, bound handling and error classification.
Keep tool observations authentic and preserve the user request. Read only public
home state/methods delivered through initialize and runtime tool observations.
Only edit this harness. Never access dataset files, target/reference labels,
label conflict reports, evaluator internals or other samples; never hardcode IDs.
The initial harness defaults to a compact room/device index and four bundled
official ICL examples in icl_examples.txt; detailed state and method signatures
are fetched through read-only tools. harness.prompt_mode also supports zero_shot.
Examples illustrate format under other household states, not current-task facts.
You may edit the bundled examples, but never retrieve examples or answers from
other dataset samples. Ground current commands in the supplied current home.
Training is disjoint from evaluation. Validation freezes 2,000 structurally
checked, observed official-rerun failures drawn from the 4,414-item full multi
test, so validation is intentionally a test subset rather than an independent
held-out split. Every candidate uses the full frozen validation set."""
