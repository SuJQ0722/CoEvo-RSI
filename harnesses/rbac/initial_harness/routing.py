"""Evolvable conservative routing for column-level Text2SQL RBAC."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


_DECLARED_ROLE = re.compile(r"(?mi)^Role:\s*(.+?)\s*$")
_TABLE_COUNT = re.compile(r"(?mi)^Total Tables:\s*(\d+)\s*$")
_ACCESS_LINE = re.compile(r"(?mi)^Accessible Columns:\s*(.+?)\s*$")
_SCHEMA_TABLE = re.compile(
    r"(?mi)^Table\s+([^\n]+?)\s+has columns such as\s+(.+?)\.\s*$"
)
_REQUEST_MARKER = "##Access Request:"


def normalize_text(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def _tokens(value: object) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", str(value).casefold()))


def _contains_token_sequence(
    tokens: tuple[str, ...], sequence: tuple[str, ...]
) -> bool:
    width = len(sequence)
    return width > 0 and any(
        tokens[index : index + width] == sequence
        for index in range(len(tokens) - width + 1)
    )


@dataclass(frozen=True)
class Route:
    """A rule decision, or ``None`` when the generative model must decide."""

    decision: int | None
    source: str
    confidence: float
    required_role: str | None = None


class RuleRouter:
    """Minimal safe rules; unseen database roles always defer to the LLM."""

    def __init__(self, policy_rows: Sequence[Mapping[str, Any]]):
        self.known_roles = frozenset(
            str(row.get("test_role") or "").strip()
            for row in policy_rows
            if str(row.get("test_role") or "").strip()
        )
        # Kept as part of the stable harness interface. Text2SQL train and eval
        # databases are disjoint, so memorized instruction labels are not rules.
        self.exact_policy: dict[str, str] = {}

    @staticmethod
    def _all_database_columns_allowed(instruction: str) -> bool:
        table_match = _TABLE_COUNT.search(instruction)
        access_match = _ACCESS_LINE.search(instruction)
        if not table_match or not access_match:
            return False
        entries = [item.strip() for item in access_match.group(1).split(";") if item.strip()]
        return (
            len(entries) == int(table_match.group(1))
            and all(item.upper().endswith(": ALL COLUMNS") for item in entries)
        )

    @staticmethod
    def _explicit_inaccessible_multitoken_column(instruction: str) -> bool:
        """Recognize a narrow, high-precision deterministic denial.

        A request qualifies only when it literally contains a multi-token schema
        column name and that column is unavailable in every schema table where it
        occurs.  Single-token semantic guesses (for example, mapping "popular" to
        ``percentage``) intentionally remain with the LLM.
        """
        header, separator, request = instruction.partition(_REQUEST_MARKER)
        if (
            not separator
            or "##Role Access Policy" not in header
            or not _DECLARED_ROLE.search(header)
        ):
            return False

        access_match = _ACCESS_LINE.search(header)
        if not access_match:
            return False

        accessible: dict[tuple[str, ...], set[tuple[str, ...]] | None] = {}
        for entry in access_match.group(1).split(";"):
            if ":" not in entry:
                return False
            table, columns = entry.split(":", 1)
            table_key = _tokens(table)
            if not table_key or table_key in accessible:
                return False
            if normalize_text(columns) == "all columns":
                accessible[table_key] = None
                continue
            column_keys = {_tokens(column) for column in columns.split(",")}
            if not column_keys or () in column_keys:
                return False
            accessible[table_key] = column_keys

        occurrences: dict[tuple[str, ...], set[tuple[str, ...]]] = {}
        for match in _SCHEMA_TABLE.finditer(header):
            table_key = _tokens(match.group(1))
            for column in match.group(2).split(","):
                column_key = _tokens(column)
                if table_key and column_key:
                    occurrences.setdefault(column_key, set()).add(table_key)
        if not occurrences:
            return False

        request_tokens = _tokens(request)
        for column_key, tables in occurrences.items():
            if len(column_key) < 2 or not _contains_token_sequence(
                request_tokens, column_key
            ):
                continue
            if all(
                table in accessible
                and accessible[table] is not None
                and column_key not in accessible.get(table, set())
                for table in tables
            ):
                return True
        return False

    def route(self, instruction: str, test_role: str) -> Route:
        instruction = str(instruction or "").strip()
        test_role = str(test_role or "").strip()
        if not instruction or not test_role:
            return Route(0, "invalid_input", 1.0)
        declared = _DECLARED_ROLE.search(instruction)
        if declared and normalize_text(declared.group(1)) != normalize_text(test_role):
            return Route(0, "role_policy_mismatch", 1.0)
        if self._all_database_columns_allowed(instruction):
            return Route(1, "all_columns_policy", 1.0, test_role)
        if self._explicit_inaccessible_multitoken_column(instruction):
            return Route(0, "explicit_inaccessible_column", 1.0)
        return Route(None, "llm_fallback", 0.0)
