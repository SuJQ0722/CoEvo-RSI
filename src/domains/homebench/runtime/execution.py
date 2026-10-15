"""Trusted, bounded read-only tools and transcript capture for HomeBench."""

import importlib
import json
from copy import deepcopy

from src.config import harness_module_suffix, harness_spec
from src.diagnostics.observe import ObservedModel
from src.training.message_schema import canonicalize_assistant_message, tool_result_message


def schema(name, description, properties, required=()):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


TOOLS = [
    schema(
        "inspect_home",
        "Read household device state, supported attributes, bounds and options. Omit room for the full state.",
        {"room": {"type": "string"}},
    ),
    schema(
        "list_methods",
        "Read available control methods. Optional exact room/device filters.",
        {"room": {"type": "string"}, "device": {"type": "string"}},
    ),
    schema(
        "final_answer",
        "Submit all machine instructions enclosed in braces, e.g. {room.light.turn_on(),error_input}. Commands are scored, never executed on real devices.",
        {"commands": {"type": "string"}},
        ("commands",),
    ),
]


def observation(home, name, args):
    if name == "inspect_home":
        if set(args) - {"room"} or any(not isinstance(v, str) for v in args.values()):
            return {"error": "room must be a string"}
        state = home["home_status"]
        return deepcopy(
            state
            if "room" not in args
            else {args["room"]: state.get(args["room"], {"error": "unknown room"})}
        )
    if name == "list_methods":
        if set(args) - {"room", "device"} or any(not isinstance(v, str) for v in args.values()):
            return {"error": "room/device must be strings"}
        return deepcopy(
            [
                m
                for m in home["method"]
                if ("room" not in args or m.get("room_name") == args["room"])
                and ("device" not in args or m.get("device_name") == args["device"])
            ]
        )
    return {"error": "unknown tool"}


def run_sample(sample, config, harness_name, client, *, max_turns=8):
    client = ObservedModel(client)
    spec = harness_spec(config)
    module = importlib.import_module(
        f"{spec.package}.{harness_module_suffix(harness_name, spec.base)}"
    )
    # Client proxy captures the actual prompts sent to the model, independently
    # of the evolvable agent's mutable history.
    requests, responses = [], []
    request_limit = 0

    class ClientProxy:
        def complete(self, messages, tools):
            if len(requests) >= request_limit:
                raise ValueError("one model call per step is allowed")
            # Prompt policy belongs to the selected evolvable harness. The
            # trusted runtime records and forwards it without hidden injection.
            effective_messages = deepcopy(messages)
            requests.append(
                {"messages": deepcopy(effective_messages), "tools": deepcopy(tools)}
            )
            response = client.complete(deepcopy(effective_messages), deepcopy(tools))
            responses.append(deepcopy(response))
            return response

    agent = module.build_agent(module.HarnessSettings(config), client=ClientProxy())
    public = {
        "home_status": deepcopy(sample.home["home_status"]),
        "method": deepcopy(sample.home["method"]),
    }
    agent.initialize(sample.instruction, public, deepcopy(TOOLS))
    configure_task = getattr(agent, "set_task_context", None)
    if callable(configure_task):
        configure_task(task_type=sample.record["task_type"])
    transcript, generated, error, finished = [], "", None, False
    error_detail = None
    termination = "budget_exhausted"
    seen, calls, usage = set(), 0, 0
    usage_available = True
    for turn in range(max_turns):
        before = len(requests)
        request_limit = before + 1
        try:
            message = canonicalize_assistant_message(agent.step())
        except Exception as exc:  # noqa: BLE001 -- record candidate/provider failures
            error = f"model_or_agent_error:{type(exc).__name__}"
            error_detail = str(exc)[:2000]
            termination = (
                "provider_error"
                if any("error_type" in r for r in client.requests)
                else "runtime_error"
            )
            break
        if len(requests) != before + 1 or len(responses) != len(requests):
            error = "agent_must_make_one_model_request_per_step"
            termination = "harness_protocol_error"
            break
        if message != canonicalize_assistant_message(responses[-1]):
            error = "agent_modified_model_response"
            termination = "harness_protocol_error"
            break
        if turn == 0:
            transcript = deepcopy(requests[0]["messages"])
        model_usage = message.pop("_coevox_usage", {})
        usage_available = usage_available and bool(model_usage)
        usage += int(model_usage.get("total_tokens", 0))
        transcript.append(deepcopy(message))
        tool_calls = message.get("tool_calls") or []
        if len(tool_calls) != 1:
            termination = "protocol_error"
            break  # Model protocol failure, not an infrastructure error.
        call = tool_calls[0]
        name = (call.get("function") or {}).get("name", "")
        cid = call.get("id")
        if not cid or cid in seen:
            termination = "protocol_error"
            break
        seen.add(cid)
        calls += 1
        try:
            args = call["function"]["arguments"]
            args = json.loads(args) if isinstance(args, str) else args
            if not isinstance(args, dict):
                raise TypeError("arguments must be an object")
        except (KeyError, TypeError, ValueError):
            result = {"error": "invalid JSON object arguments"}
        else:
            if (
                name == "final_answer"
                and set(args) == {"commands"}
                and isinstance(args["commands"], str)
            ):
                validate = getattr(agent, "validate_submission", None)
                preflight = validate(args["commands"]) if callable(validate) else {"valid": True}
                if not isinstance(preflight, dict) or not isinstance(
                    preflight.get("valid"), bool
                ):
                    raise TypeError("harness submission validator returned an invalid result")
                if preflight["valid"]:
                    generated, finished = args["commands"], True
                    termination = "submitted"
                    break
                result = {
                    "status": "rejected_by_preflight",
                    "errors": preflight.get("errors") or ["submission failed preflight"],
                    "instruction": "Correct the listed issues and call final_answer again.",
                }
            else:
                result = observation(sample.home, name, args)
        reply = tool_result_message(call, json.dumps(result, ensure_ascii=False))
        transcript.append(reply)
        try:
            agent.receive_tool_result(deepcopy(reply))
        except Exception as exc:  # noqa: BLE001 -- record candidate/provider failures
            error = f"agent_observation_error:{type(exc).__name__}"
            termination = "runtime_error"
            break
    return {
        "sample_id": sample.sample_id,
        "task_digest": sample.digest,
        "instruction": sample.instruction,
        "protocol": sample.record["environment"]["protocol"],
        "task_type": sample.record["task_type"],
        "messages": transcript,
        "requests": requests,
        "tools": deepcopy(TOOLS),
        "generated": generated,
        "finished": finished,
        "error": error,
        "error_detail": error_detail,
        "interface_trace": client.trace(termination),
        "tool_calls": calls,
        "total_tokens": usage,
        "token_usage_available": usage_available and bool(requests),
    }
