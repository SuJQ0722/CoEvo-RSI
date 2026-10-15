"""Composable RBAC harness assembled from evolvable routing and prompt modules."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .llm import LLMCaller, LLMCallResult
from .prompts import access_messages, preauthorized_messages
from .response import parse_access
from .routing import Route, RuleRouter


class RBACHarness:
    """Rule-first access control with a generative-LLM fallback."""

    name = "initial_harness"

    def __init__(self, policy_rows: Sequence[Mapping[str, Any]]):
        self.router = RuleRouter(policy_rows)
        self.llm = LLMCaller()

    @property
    def tools(self) -> list[dict[str, Any]]:
        return self.llm.tools

    @property
    def known_roles(self) -> frozenset[str]:
        return self.router.known_roles

    @property
    def exact_policy(self) -> dict[str, str]:
        return self.router.exact_policy

    def route(self, instruction: str, test_role: str) -> Route:
        return self.router.route(instruction, test_role)

    def messages(
        self, instruction: str, test_role: str, route: Route
    ) -> list[dict[str, str]]:
        if route.decision == 1:
            return preauthorized_messages(instruction, test_role)
        return access_messages(instruction, test_role)

    def call_llm(
        self,
        model: Any,
        instruction: str,
        test_role: str,
        route: Route,
        *,
        db_path: str | None = None,
    ) -> LLMCallResult:
        return self.llm.call(
            model,
            self.messages(instruction, test_role, route),
            db_path=db_path,
            policy_instruction=instruction,
        )

    @staticmethod
    def parse_access(response: object) -> int:
        return parse_access(response)
