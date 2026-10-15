"""Stateful model/tool conversation for the seed HomeBench harness."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .prompts import build_user_prompt, initial_messages
from .settings import HarnessSettings
from .validation import SubmissionValidator, declared_arity


class HomeBenchHarness:
    """Keep conversation state while the trusted runtime executes tools."""

    name = "initial_harness"

    def __init__(self, settings: HarnessSettings, client: Any):
        self.settings = settings
        self.client = client
        self.messages: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] = []
        self._initialized = False
        self._validator: SubmissionValidator | None = None
        self._instruction = ""
        self._public_home: Mapping[str, Any] = {}
        self._steps = 0

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
        self._validator = SubmissionValidator(instruction=instruction, home=home)
        self._instruction = str(instruction)
        self._public_home = home
        self._steps = 0
        self._initialized = True

    def set_task_context(self, *, task_type: str) -> None:
        """Receive public structural metadata from the trusted runtime."""

        if self._validator is None:
            raise RuntimeError("HomeBench harness must be initialized before task context")
        self._validator.set_task_type(task_type)
        self._apply_slot_ladder(declared_arity(task_type))

    def _apply_slot_ladder(self, declared_slots: int | None) -> None:
        """Render the ordered rung frame into this harness's own opening message.

        Only the two messages this harness built are touched, and only while no
        step has been taken, so the model's own messages and every tool
        observation stay verbatim. Any rendering failure leaves the incumbent
        prompt in place, and nothing here issues a client request.
        """

        if not self._initialized or self._steps or len(self.messages) != 2:
            return
        if not isinstance(declared_slots, int) or isinstance(declared_slots, bool):
            return
        if declared_slots <= 0 or self.messages[1].get("role") != "user":
            return
        try:
            content = build_user_prompt(
                self._instruction,
                self._public_home,
                prompt_mode=self.settings.prompt_mode,
                declared_slots=declared_slots,
            )
        except (TypeError, ValueError, KeyError, OSError):
            return
        self.messages[1]["content"] = content

    def validate_submission(self, commands: str) -> dict[str, Any]:
        """Preflight a proposed answer without reading reference labels."""

        if self._validator is None:
            raise RuntimeError("HomeBench harness must be initialized before validation")
        return self._validator.validate(commands)

    def step(self) -> dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("HomeBench harness must be initialized before step")
        self._steps += 1
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
