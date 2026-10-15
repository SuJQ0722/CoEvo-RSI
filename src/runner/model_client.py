"""Small OpenAI Chat Completions/Responses client used by rollout collection."""

from __future__ import annotations

import json
import logging
import os
import re
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Protocol
from urllib import error, request

from src.training.message_schema import canonicalize_assistant_message

_QWEN_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_QWEN_FUNCTION_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
_QWEN_PARAMETER_RE = re.compile(
    r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL
)
_CONTEXT_LENGTH_RE = re.compile(
    r"maximum context length is\s+(?P<context>\d+)\s+tokens.*?"
    r"requested\s+(?P<output>\d+)\s+output tokens.*?"
    r"prompt contains at least\s+(?P<input>\d+)\s+input tokens",
    re.IGNORECASE | re.DOTALL,
)
_MAX_CONTEXT_RETRIES = 3
_MIN_RETRY_OUTPUT_TOKENS = 256
LOGGER = logging.getLogger(__name__)


def _parse_qwen_tool_calls(
    text: str,
    tools: list[dict[str, Any]],
    *,
    call_id_prefix: str = "recovered-tool-call",
) -> list[dict[str, Any]]:
    """Recover Qwen XML tool calls when the server leaves them in reasoning."""
    allowed_names = {
        str(tool.get("function", {}).get("name") or "") for tool in tools
    }
    calls: list[dict[str, Any]] = []
    for block in _QWEN_TOOL_CALL_RE.findall(text):
        match = _QWEN_FUNCTION_RE.fullmatch(block.strip())
        if match is None:
            continue
        name, body = match.groups()
        name = name.strip()
        if name not in allowed_names:
            continue
        arguments: dict[str, Any] = {}
        for parameter_name, raw_value in _QWEN_PARAMETER_RE.findall(body):
            raw_value = raw_value.strip()
            try:
                value = json.loads(raw_value)
            except (json.JSONDecodeError, TypeError):
                value = raw_value
            arguments[parameter_name.strip()] = value
        calls.append(
            {
                "id": f"{call_id_prefix}-{len(calls)}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )
    return calls


class ChatModel(Protocol):
    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]: ...


class TextModel(Protocol):
    def complete_text(self, messages: list[dict[str, Any]]) -> dict[str, Any]: ...


@dataclass
class OpenAICompatibleClient:
    model: str
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str | None = None
    temperature: float = 0.6
    top_p: float | None = 0.95
    top_k: int | None = 20
    max_tokens: int | None = 4096
    reasoning_effort: str | None = "medium"
    enable_thinking: bool | None = True
    timeout: float = 300.0
    seed: int | None = None
    api_mode: str = "chat_completions"
    tool_choice: str = "required"
    _response_items: OrderedDict[str, list[dict[str, Any]]] = field(
        default_factory=OrderedDict, init=False, repr=False, compare=False
    )
    _response_items_lock: Lock = field(
        default_factory=Lock, init=False, repr=False, compare=False
    )
    _recovered_call_sequence: int = field(default=0, init=False, repr=False, compare=False)
    _recovered_call_lock: Lock = field(
        default_factory=Lock, init=False, repr=False, compare=False
    )

    def _next_recovered_call_prefix(self) -> str:
        with self._recovered_call_lock:
            sequence = self._recovered_call_sequence
            self._recovered_call_sequence += 1
        return f"recovered-tool-call-{sequence}"

    @staticmethod
    def _smaller_output_budget(
        payload: dict[str, Any], detail: str
    ) -> tuple[dict[str, Any], str, int, int] | None:
        """Reduce only an output reservation rejected by the context window.

        vLLM stops counting the prompt as soon as the requested output budget
        can no longer fit, so the reported input count is only a lower bound.
        Halving leaves useful headroom and allows a bounded second/third retry
        without truncating the conversation or any tool evidence.
        """
        match = _CONTEXT_LENGTH_RE.search(detail)
        if match is None:
            return None
        key = "max_output_tokens" if "max_output_tokens" in payload else "max_tokens"
        current = payload.get(key)
        if not isinstance(current, int) or current <= _MIN_RETRY_OUTPUT_TOKENS:
            return None
        context = int(match.group("context"))
        minimum_input = int(match.group("input"))
        available = context - minimum_input - 1
        reduced = min(current // 2, available)
        if reduced < _MIN_RETRY_OUTPUT_TOKENS or reduced >= current:
            return None
        retry_payload = dict(payload)
        retry_payload[key] = reduced
        return retry_payload, key, current, reduced

    def _request(
        self, payload: dict[str, Any], *, expected_status: int | None = None
    ) -> dict[str, Any]:
        if self.api_mode not in {"chat_completions", "responses"}:
            raise ValueError(f"unsupported OpenAI API mode: {self.api_mode!r}")
        api_key = self.api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
        endpoint = "responses" if self.api_mode == "responses" else "chat/completions"
        request_payload = payload
        for context_retry in range(_MAX_CONTEXT_RETRIES + 1):
            req = request.Request(
                f"{self.base_url.rstrip('/')}/{endpoint}",
                data=json.dumps(request_payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=self.timeout) as response:
                    if expected_status is not None and response.status != expected_status:
                        raise RuntimeError(
                            f"model endpoint returned HTTP {response.status}: "
                            f"expected HTTP {expected_status}"
                        )
                    return json.load(response)
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if api_key:
                    detail = detail.replace(api_key, "[REDACTED]")
                retry = self._smaller_output_budget(request_payload, detail)
                if retry is None or context_retry >= _MAX_CONTEXT_RETRIES:
                    raise RuntimeError(
                        f"model endpoint returned HTTP {exc.code}: {detail}"
                    ) from exc
                request_payload, key, previous, reduced = retry
                LOGGER.warning(
                    "context window rejected %s=%d; retrying request with %s=%d",
                    key,
                    previous,
                    key,
                    reduced,
                )
        raise AssertionError("unreachable context retry loop")

    def preflight(self) -> int:
        """Probe the actual generation endpoint with the configured tool controls."""
        messages = [{"role": "user", "content": "Call connectivity_check with ok=true."}]
        tools = [{
            "type": "function",
            "function": {
                "name": "connectivity_check",
                "description": "Confirm that the generation endpoint is available.",
                "parameters": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
            },
        }]
        if self.api_mode == "responses":
            payload = self._responses_payload(messages, tools)
        else:
            payload = self._sampling_payload(messages)
            payload.update({
                "tools": tools,
                "tool_choice": self.tool_choice,
                "parallel_tool_calls": False,
            })
        self._request(payload, expected_status=200)
        return 200

    def _sampling_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {
                "enable_thinking": self.enable_thinking
            }
        return payload

    def _responses_input(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Translate a canonical chat transcript into Responses API input items."""
        items: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role") or "")
            if role in {"system", "user"}:
                items.append({"role": role, "content": str(message.get("content") or "")})
                continue
            if role == "assistant":
                tool_calls = message.get("tool_calls") or []
                call_ids = [str(call.get("id") or "") for call in tool_calls]
                cached: list[dict[str, Any]] | None = None
                with self._response_items_lock:
                    if call_ids and all(
                        call_id in self._response_items for call_id in call_ids
                    ):
                        cached = deepcopy(self._response_items[call_ids[0]])
                        # Each request replays the full conversation. Keep the
                        # original reasoning items for later tool turns too;
                        # _responses_message bounds the cache size.
                if cached is not None:
                    items.extend(cached)
                    continue
                content = str(message.get("content") or "")
                if content:
                    items.append({"role": "assistant", "content": content})
                for tool_call in tool_calls:
                    function = tool_call.get("function") or {}
                    arguments = function.get("arguments", "{}")
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments, ensure_ascii=False)
                    call_id = str(tool_call.get("id") or "")
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": str(function.get("name") or ""),
                            "arguments": arguments,
                        }
                    )
                continue
            if role == "tool":
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(message.get("tool_call_id") or ""),
                        "output": str(message.get("content") or ""),
                    }
                )
                continue
            raise ValueError(f"unsupported message role for Responses API: {role!r}")
        return items

    @staticmethod
    def _responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for tool in tools:
            function = tool.get("function") or {}
            result.append(
                {
                    "type": "function",
                    "name": str(function.get("name") or ""),
                    "description": str(function.get("description") or ""),
                    "parameters": function.get("parameters") or {},
                    "strict": bool(function.get("strict", False)),
                }
            )
        return result

    def _responses_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": self._responses_input(messages),
        }
        if tools is not None:
            payload.update(
                {
                    "tools": self._responses_tools(tools),
                    "tool_choice": self.tool_choice,
                    "parallel_tool_calls": False,
                }
            )
        if self.max_tokens is not None:
            payload["max_output_tokens"] = self.max_tokens
        if self.reasoning_effort is not None:
            payload["reasoning"] = {
                "effort": self.reasoning_effort,
                "summary": "auto",
            }
        return payload

    def _responses_message(self, body: dict[str, Any]) -> dict[str, Any]:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for item in body.get("output") or []:
            item_type = item.get("type")
            if item_type == "function_call":
                tool_calls.append(
                    {
                        "id": str(item.get("call_id") or item.get("id") or ""),
                        "type": "function",
                        "function": {
                            "name": str(item.get("name") or ""),
                            "arguments": item.get("arguments") or "{}",
                        },
                    }
                )
            elif item_type == "message":
                for part in item.get("content") or []:
                    if part.get("type") in {"output_text", "text"}:
                        content_parts.append(str(part.get("text") or ""))
            elif item_type == "reasoning":
                for part in item.get("summary") or []:
                    if part.get("type") in {"summary_text", "text"}:
                        reasoning_parts.append(str(part.get("text") or ""))
        result = canonicalize_assistant_message(
            {
                "role": "assistant",
                "content": "\n".join(part for part in content_parts if part),
                "reasoning_content": "\n".join(
                    part for part in reasoning_parts if part
                ),
                "tool_calls": tool_calls,
            }
        )
        response_items = body.get("output") or []
        call_ids = [
            str(item.get("call_id") or item.get("id") or "")
            for item in response_items
            if item.get("type") == "function_call"
        ]
        if call_ids:
            with self._response_items_lock:
                for call_id in call_ids:
                    self._response_items[call_id] = deepcopy(response_items)
                    self._response_items.move_to_end(call_id)
                while len(self._response_items) > 1024:
                    self._response_items.popitem(last=False)
        usage = body.get("usage")
        incomplete = body.get("incomplete_details")
        if isinstance(incomplete, dict) and incomplete.get("reason"):
            result["_coevox_incomplete_reason"] = incomplete["reason"]
        if isinstance(usage, dict):
            input_tokens = int(usage.get("input_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or 0)
            result["_coevox_usage"] = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": int(
                    usage.get("total_tokens") or input_tokens + output_tokens
                ),
            }
        return result

    @staticmethod
    def _message_with_usage(body: dict[str, Any]) -> dict[str, Any]:
        message = body["choices"][0]["message"]
        result = canonicalize_assistant_message(message)
        usage = body.get("usage")
        if isinstance(usage, dict):
            result["_coevox_usage"] = {
                key: int(usage.get(key) or 0)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
        return result

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return self._complete(messages, tools)

    def complete_observed(self, messages, tools, *, observe_response):
        """Observe the provider message before normalization or tool recovery."""
        return self._complete(messages, tools, observe_response=observe_response)

    def _complete(self, messages, tools, *, observe_response=None):
        if self.api_mode == "responses":
            body = self._request(self._responses_payload(messages, tools))
            if observe_response is not None:
                observe_response(body)
            return self._responses_message(body)
        payload = self._sampling_payload(messages)
        payload.update({
            "tools": tools,
            "tool_choice": self.tool_choice,
            "parallel_tool_calls": False,
        })
        body = self._request(payload)
        if observe_response is not None:
            observe_response(body["choices"][0]["message"])
        result = self._message_with_usage(body)
        if not result.get("tool_calls"):
            reasoning = str(result.get("reasoning") or result.get("reasoning_content") or "")
            recovered = _parse_qwen_tool_calls(
                reasoning,
                tools,
                call_id_prefix=self._next_recovered_call_prefix(),
            )
            if recovered:
                result["tool_calls"] = recovered
        return result

    def complete_text(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate a normal assistant message without forcing a tool call."""
        if self.api_mode == "responses":
            return self._responses_message(
                self._request(self._responses_payload(messages))
            )
        return self._message_with_usage(self._request(self._sampling_payload(messages)))
