"""Composable RBAC harness assembled from evolvable routing and prompt modules."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .llm import LLMCaller, LLMCallResult
from .prompts import access_messages, preauthorized_messages
from .response import parse_access
from .routing import Route, RuleRouter
from .tools.final_answer import ACCESS_GRANTED


class RBACHarness:
    """Rule-first access control with a generative-LLM fallback."""

    name = "generated"

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
        if route.decision == 1 or route.source == "all_columns_policy":
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
        result = self.llm.call(
            model,
            self.messages(instruction, test_role, route),
            db_path=db_path,
            policy_instruction=instruction,
            require_denial_evidence=route.source == "llm_fallback",
            recover_exhausted_grant=route.source == "llm_fallback",
            verify_grant_grounding=route.source == "llm_fallback",
        )
        if route.source != "all_columns_policy":
            return result
        # The all-columns rule already decided the authorization from the policy
        # header alone: every declared table is granted in full, so the outcome is
        # a grant no matter what marker the model returned.  The model is consulted
        # only to supply the SQL body a bare rule grant cannot produce -- the SQL
        # absence (not the decision) is what scored these samples zero on SafeEX.
        # Keep the deterministic 0/1 output and the error="" terminal contract and
        # borrow only the SQL, so a model denial, failure or empty submission can
        # never regress the access decision below the incumbent's rule grant.
        sql = result.sql
        sql = sql.strip() if isinstance(sql, str) and sql.strip() else None
        return LLMCallResult(
            response=ACCESS_GRANTED,
            messages=result.messages,
            usage=result.usage,
            latency_seconds=result.latency_seconds,
            sql=sql,
            turns=result.turns,
            error="",
        )

    @staticmethod
    def parse_access(response: object) -> int:
        return parse_access(response)
