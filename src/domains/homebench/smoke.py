import importlib

from src.config import harness_module_suffix, harness_relative_path
from src.domains.base import SmokeCheck, SmokeReport


def test(harness_name, *, package, root):
    checks = []
    name = f"{package}.{harness_module_suffix(harness_name)}"
    try:
        module = importlib.import_module(name)

        class Client:
            def complete(self, messages, tools):
                assert any("Public request" in str(m.get("content", "")) for m in messages)
                return {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "",
                    "tool_calls": [
                        {
                            "id": "smoke1",
                            "type": "function",
                            "function": {
                                "name": "final_answer",
                                "arguments": '{"commands":"{error_input}"}',
                            },
                        }
                    ],
                }

        from .runtime.execution import TOOLS

        agent = module.build_agent(module.HarnessSettings({}), client=Client())
        agent.initialize("Public request", {"home_status": {}, "method": []}, TOOLS)
        message = agent.step()
        checks.append(SmokeCheck("public_agent_contract", message.get("role") == "assistant", name))
        path = root / harness_relative_path(harness_name)
        source = "\n".join(p.read_text() for p in path.rglob("*.py"))
        forbidden = [
            s
            for s in (
                "homebench.evaluator",
                "homebench.data",
                "data/HomeBench",
                "label_conflicts",
                "target[",
            )
            if s in source
        ]
        checks.append(SmokeCheck("private_api_check", not forbidden, repr(forbidden)))
    except Exception as exc:  # noqa: BLE001 -- record candidate/provider failures
        checks.append(SmokeCheck("public_agent_contract", False, f"{type(exc).__name__}: {exc}"))
    return SmokeReport(harness_name, all(c.passed for c in checks), tuple(checks))
