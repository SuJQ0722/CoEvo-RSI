"""Canonical chat-message contract shared by every rejection-sampling domain."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any


_LEGACY_ASSISTANT_MARKER = re.compile(r"(?im)^\s*(Thought|Action)\s*:")


def validate_chat_messages(
    messages: Any,
    *,
    require_tool_calls: bool = False,
    require_reasoning_content: bool = True,
) -> None:
    """Validate the OpenAI-style transcript persisted for SFT.

    Domain-specific samplers may decide which rollout is correct, but they must
    all persist the same message schema: assistant reasoning is separate from
    content, actions are ``tool_calls``, and observations are ``tool`` messages.
    """

    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    known_call_ids: set[str] = set()
    previous_role: str | None = None
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} must be an object")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"message {index} has unsupported role {role!r}")
        if role == "system" and index != 0:
            raise ValueError("system message must be first")
        if role == "user" and previous_role == "tool":
            raise ValueError(
                f"message {index} has user role immediately after a tool result; "
                "merge the guidance into the tool result for ms-swift compatibility"
            )
        content = message.get("content", "")
        if not isinstance(content, str):
            raise ValueError(f"message {index} content must be a string")
        if role == "assistant":
            if "reasoning" in message:
                raise ValueError(
                    f"assistant message {index} must use reasoning_content, not reasoning"
                )
            reasoning = message.get("reasoning_content")
            if require_reasoning_content and not isinstance(reasoning, str):
                raise ValueError(
                    f"assistant message {index} reasoning_content must be a string"
                )
            if reasoning is not None and not isinstance(reasoning, str):
                raise ValueError(f"assistant message {index} reasoning must be a string")
            if _LEGACY_ASSISTANT_MARKER.search(str(content or "")):
                raise ValueError(f"assistant message {index} uses legacy Thought/Action text")
            tool_calls = message.get("tool_calls") or []
            if require_tool_calls and len(tool_calls) != 1:
                raise ValueError(f"assistant message {index} must contain exactly one tool call")
            for tool_call in tool_calls:
                call_id = str(tool_call.get("id") or "")
                function = tool_call.get("function") or {}
                name = str(function.get("name") or "")
                arguments = function.get("arguments", {})
                if not call_id or not name:
                    raise ValueError(f"assistant message {index} has an invalid tool call")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"assistant message {index} has invalid tool arguments JSON"
                        ) from exc
                if not isinstance(arguments, dict):
                    raise ValueError(f"assistant message {index} tool arguments must be an object")
                if call_id in known_call_ids:
                    raise ValueError(f"duplicate tool call id {call_id!r}")
                known_call_ids.add(call_id)
        elif role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            if call_id not in known_call_ids:
                raise ValueError(f"tool message {index} references unknown call id {call_id!r}")
        previous_role = role


def validate_training_record(
    record: Any,
    *,
    require_tool_calls: bool = True,
    require_reasoning_content: bool = True,
) -> None:
    """Validate one domain-independent rejection-sampling output record."""

    if not isinstance(record, dict):
        raise ValueError("training record must be an object")
    validate_chat_messages(
        record.get("messages"),
        require_tool_calls=require_tool_calls,
        require_reasoning_content=require_reasoning_content,
    )
    tools = record.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError("training record tools must be a non-empty list")
    declared_names: set[str] = set()
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ValueError(f"tool schema {index} must be an object")
        name = str((tool.get("function") or {}).get("name") or "")
        if not name:
            raise ValueError(f"tool schema {index} has no function name")
        if name in declared_names:
            raise ValueError(f"duplicate tool schema {name!r}")
        declared_names.add(name)
    called_names = {
        str((call.get("function") or {}).get("name") or "")
        for message in record["messages"]
        if message.get("role") == "assistant"
        for call in (message.get("tool_calls") or [])
    }
    undeclared = called_names - declared_names
    if undeclared:
        raise ValueError(f"messages call undeclared tools: {sorted(undeclared)!r}")
    if not isinstance(record.get("sampling"), dict):
        raise ValueError("training record sampling must be an object")


def canonicalize_chat_messages(messages: Any) -> list[dict[str, Any]]:
    """Return a detached transcript using the one canonical assistant schema."""

    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    normalized: list[dict[str, Any]] = []
    for index, message in enumerate(deepcopy(messages)):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} must be an object")
        if message.get("role") == "assistant":
            message = canonicalize_assistant_message(message)
        normalized.append(message)
    return normalized


def canonicalize_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    """Map provider variants onto stable assistant fields."""

    result = {key: value for key, value in message.items() if value is not None}
    result["role"] = "assistant"
    result.setdefault("content", "")
    reasoning = result.pop("reasoning", result.get("reasoning_content", ""))
    result["reasoning_content"] = str(reasoning or "")
    return result


def assistant_tool_message(
    tool_calls: list[dict[str, Any]],
    *,
    reasoning_content: Any = "",
    content: Any = "",
) -> dict[str, Any]:
    """Build the assistant half of a tool interaction for any domain runtime."""

    return canonicalize_assistant_message({
        "role": "assistant",
        "content": content,
        "reasoning_content": reasoning_content,
        "tool_calls": tool_calls,
    })


def tool_result_message(tool_call: dict[str, Any], content: Any) -> dict[str, Any]:
    """Build the observation half of a tool interaction for any domain runtime."""

    function = tool_call.get("function") or {}
    call_id = str(tool_call.get("id") or "")
    name = str(function.get("name") or "")
    if not call_id or not name:
        raise ValueError("cannot build tool result for an invalid tool call")
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": str(content),
    }
