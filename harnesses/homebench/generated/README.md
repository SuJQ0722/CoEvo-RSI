# HomeBench initial harness

The seed harness is intentionally small but structured:

- `settings.py` validates evolvable harness options.
- `context.py` builds a room/device inventory without dumping all state and methods into the first prompt.
- `prompts.py` owns prompt assets and initial-message assembly.
- `harness.py` owns conversation lifecycle and history.
- `factory.py` provides the stable `build_agent` runtime entry point.
- `agent.py` remains a compatibility import surface.

The trusted domain runtime owns tool schemas and execution. The harness receives only public
household data, initially exposes a compact inventory, and asks the model to retrieve relevant
state or method details with `inspect_home` and `list_methods` before submitting `final_answer`.
