import json
from io import BytesIO
from urllib.error import HTTPError

from src.runner.model_client import OpenAICompatibleClient


class _Response:
    def __init__(self, message=None, usage=None):
        self.message = message or {"role": "assistant", "tool_calls": []}
        self.usage = usage

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        payload = {"choices": [{"message": self.message}]}
        if self.usage is not None:
            payload["usage"] = self.usage
        return json.dumps(payload).encode()


class _RawResponse(_Response):
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode()


def test_client_sends_bounded_reasoning_controls(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["payload"] = json.loads(req.data)
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("src.runner.model_client.request.urlopen", fake_urlopen)
    client = OpenAICompatibleClient(model="Qwen3.5-4B")
    client.complete([{"role": "user", "content": "x"}], [])

    assert captured["payload"]["reasoning_effort"] == "medium"
    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert captured["payload"]["max_tokens"] == 4096
    assert captured["payload"]["parallel_tool_calls"] is False
    assert captured["payload"]["tool_choice"] == "required"
    assert captured["timeout"] == 300


def test_client_allows_auto_tool_choice_for_reasoning_provider(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["payload"] = json.loads(req.data)
        return _Response()

    monkeypatch.setattr("src.runner.model_client.request.urlopen", fake_urlopen)
    client = OpenAICompatibleClient(model="reasoning-provider", tool_choice="auto")
    client.complete([{"role": "user", "content": "x"}], [])

    assert captured["payload"]["tool_choice"] == "auto"


def test_client_retries_context_rejection_with_smaller_output_budget(monkeypatch):
    budgets = []

    def fake_urlopen(req, timeout):
        payload = json.loads(req.data)
        budgets.append(payload["max_tokens"])
        if len(budgets) == 1:
            detail = json.dumps(
                {
                    "error": {
                        "message": (
                            "This model's maximum context length is 40960 tokens. "
                            "However, you requested 6144 output tokens and your prompt "
                            "contains at least 34817 input tokens."
                        )
                    }
                }
            ).encode()
            raise HTTPError(req.full_url, 400, "Bad Request", {}, BytesIO(detail))
        return _Response()

    monkeypatch.setattr("src.runner.model_client.request.urlopen", fake_urlopen)
    client = OpenAICompatibleClient(model="Qwen3.5-4B", max_tokens=6144)
    client.complete([{"role": "user", "content": "x"}], [])

    assert budgets == [6144, 3072]


def test_client_recovers_qwen_tool_call_from_reasoning(monkeypatch):
    reasoning = """<tool_call>
<function=inspect>
<parameter=step_id_list>[1, 2]</parameter>
</function>
</tool_call>"""

    monkeypatch.setattr(
        "src.runner.model_client.request.urlopen",
        lambda _req, timeout: _Response(
            {"role": "assistant", "content": "", "reasoning": reasoning}
        ),
    )
    client = OpenAICompatibleClient(model="Qwen3.5-4B")
    message = client.complete(
        [{"role": "user", "content": "x"}],
        [{"type": "function", "function": {"name": "inspect"}}],
    )
    assert message["tool_calls"][0]["function"] == {
        "name": "inspect",
        "arguments": '{"step_id_list": [1, 2]}',
    }
    next_message = client.complete(
        [{"role": "user", "content": "x"}],
        [{"type": "function", "function": {"name": "inspect"}}],
    )
    assert next_message["tool_calls"][0]["id"] != message["tool_calls"][0]["id"]


def test_client_returns_usage_as_internal_metadata(monkeypatch):
    monkeypatch.setattr(
        "src.runner.model_client.request.urlopen",
        lambda _req, timeout: _Response(
            usage={"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
        ),
    )
    message = OpenAICompatibleClient(model="Qwen3.5-4B").complete(
        [{"role": "user", "content": "x"}], []
    )
    assert message["_coevox_usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "total_tokens": 15,
    }


def test_observer_sees_provider_format_before_client_recovery(monkeypatch):
    from src.diagnostics.mismatch import analyze_record, summarize
    from src.diagnostics.observe import ObservedModel
    from src.domains.who_when.mismatch import WhoWhenMismatch

    reasoning = "<tool_call><function=inspect><parameter=step_id_list>[1]</parameter></function></tool_call>"
    monkeypatch.setattr(
        "src.runner.model_client.request.urlopen",
        lambda _req, timeout: _Response({"role": "assistant", "reasoning": reasoning}),
    )
    observed = ObservedModel(OpenAICompatibleClient(model="fixture"))
    response = observed.complete(
        [],
        [
            {
                "type": "function",
                "function": {
                    "name": "inspect",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "step_id_list": {
                                "type": "array",
                                "items": {"type": "integer"},
                            }
                        },
                        "required": ["step_id_list"],
                    },
                },
            }
        ],
    )
    assert response["tool_calls"][0]["function"]["name"] == "inspect"
    assert observed.model == "fixture"
    raw = observed.trace()["requests"][0]
    assert "tool_calls" not in raw["response"]
    assert "reasoning" not in raw["response"]
    assert raw["transport_repaired"]
    assert raw["effective_response"]["tool_calls"] == response["tool_calls"]
    result = summarize(
        [analyze_record({"sample_id": "x", "interface_trace": observed.trace()}, WhoWhenMismatch())]
    )
    assert result["rates"]["response_protocol_error_rate"]["rate"] == 0
    assert result["rates"]["interface_mismatch_turn_rate"]["rate"] == 0
    assert result["rates"]["transport_repair_rate"]["rate"] == 1
    assert result["transport_repaired_response_count"] == 1


def test_client_supports_responses_api_tool_round_trip(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data)
        captured["timeout"] = timeout
        return _RawResponse(
            {
                "output": [
                    {
                        "type": "reasoning",
                        "summary": [
                            {"type": "summary_text", "text": "Checked permissions."}
                        ],
                    },
                    {
                        "type": "function_call",
                        "call_id": "call-final",
                        "name": "final_answer",
                        "arguments": '{"access_granted":false}',
                    },
                ],
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 4,
                    "total_tokens": 24,
                },
            }
        )

    monkeypatch.setattr("src.runner.model_client.request.urlopen", fake_urlopen)
    client = OpenAICompatibleClient(
        model="gpt-test",
        base_url="https://example.test/v1",
        api_mode="responses",
        reasoning_effort="high",
    )
    message = client.complete(
        [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "Check access."},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "tool_calls": [
                    {
                        "id": "call-test",
                        "type": "function",
                        "function": {
                            "name": "test_sql",
                            "arguments": '{"sql":"SELECT 1"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-test",
                "name": "test_sql",
                "content": '{"rows":[[1]]}',
            },
        ],
        [
            {
                "type": "function",
                "function": {
                    "name": "final_answer",
                    "description": "Finish.",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )

    assert captured["url"] == "https://example.test/v1/responses"
    assert captured["payload"]["reasoning"] == {
        "effort": "high",
        "summary": "auto",
    }
    assert captured["payload"]["tools"][0]["name"] == "final_answer"
    assert captured["payload"]["input"][-1] == {
        "type": "function_call_output",
        "call_id": "call-test",
        "output": '{"rows":[[1]]}',
    }
    assert message["reasoning_content"] == "Checked permissions."
    assert message["tool_calls"][0]["function"]["name"] == "final_answer"
    assert message["_coevox_usage"]["total_tokens"] == 24


def test_responses_client_replays_reasoning_items_after_tool_result(monkeypatch):
    payloads = []
    bodies = [
        {
            "output": [
                {
                    "type": "reasoning",
                    "id": "reasoning-1",
                    "encrypted_content": "opaque",
                    "summary": [],
                },
                {
                    "type": "function_call",
                    "call_id": "call-test",
                    "name": "test_sql",
                    "arguments": '{"sql":"SELECT 1"}',
                },
            ]
        },
        {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call-final",
                    "name": "final_answer",
                    "arguments": '{"access_granted":true,"sql":"SELECT 1"}',
                }
            ]
        },
        {"output": []},
    ]

    def fake_urlopen(req, timeout):
        payloads.append(json.loads(req.data))
        return _RawResponse(bodies.pop(0))

    monkeypatch.setattr("src.runner.model_client.request.urlopen", fake_urlopen)
    client = OpenAICompatibleClient(model="gpt-test", api_mode="responses")
    messages = [{"role": "user", "content": "query"}]
    first = client.complete(messages, [])
    messages.extend(
        [
            first,
            {
                "role": "tool",
                "tool_call_id": "call-test",
                "name": "test_sql",
                "content": '{"rows":[[1]]}',
            },
        ]
    )
    second = client.complete(messages, [])

    replayed = payloads[1]["input"]
    assert replayed[1]["type"] == "reasoning"
    assert replayed[1]["encrypted_content"] == "opaque"
    assert replayed[2]["type"] == "function_call"
    assert replayed[3]["type"] == "function_call_output"

    messages.extend([
        second,
        {"role": "tool", "tool_call_id": "call-final", "content": "validation failed; retry"},
    ])
    client.complete(messages, [])
    # The curator makes many tool turns and may correct a rejected plan.
    # Earlier reasoning must not disappear on the third request.
    assert payloads[2]["input"][:4] == replayed


def test_responses_client_exposes_truncated_reasoning():
    client = OpenAICompatibleClient(model="meta", api_mode="responses")
    result = client._responses_message({
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [],
        "usage": {"input_tokens": 20, "output_tokens": 8192},
    })
    assert result["_coevox_incomplete_reason"] == "max_output_tokens"
    assert result["_coevox_usage"]["completion_tokens"] == 8192
