"""Stateful model/tool conversation for the seed HomeBench harness."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .prompts import initial_messages
from .settings import HarnessSettings


class HomeBenchHarness:
    """Keep conversation state while the trusted runtime executes tools."""

    name = "initial_harness"

    def __init__(self, settings: HarnessSettings, client: Any):
        self.settings = settings
        self.client = client
        self.messages: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] = []
        self._initialized = False

    def initialize(
        self,
        instruction: str,
        home: dict[str, Any],
        tools: list[dict[str, Any]],
    ) -> None:
        if self._initialized:
            raise RuntimeError("HomeBench harness is already initialized")
        self.messages = initial_messages(self.settings, instruction, home)
        self.tools = deepcopy(tools)
        self._initialized = True

    def step(self) -> dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("HomeBench harness must be initialized before step")
        message = self.client.complete(self.messages, self.tools)
        if not isinstance(message, dict):
            raise TypeError("model response must be an object")
        # Provider telemetry belongs in the returned record, not in the next
        # model request's conversational history.
        history_message = {
            key: deepcopy(value)
            for key, value in message.items()
            if not key.startswith("_coevox_")
        }
        self.messages.append(history_message)
        return message

    def receive_tool_result(self, message: dict[str, Any]) -> None:
        if not self._initialized:
            raise RuntimeError("HomeBench harness must be initialized before tool results")
        if not isinstance(message, dict) or message.get("role") != "tool":
            raise ValueError("tool result must be a tool-role message")
        self.messages.append(deepcopy(message))


# Preserve the public class name used by existing runs and external callers.
PromptAgent = HomeBenchHarness
