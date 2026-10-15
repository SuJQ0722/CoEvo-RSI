"""Gold-free deterministic preflight for HomeBench final submissions."""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .grounding import delivered_rooms, resolve_room_name


_TASK_ARITY = re.compile(r"^multi(\d+)_")

# Envelope transport payload. The envelope predicate below is unchanged: the
# diagnostic only observes the string that was already rejected, so the
# accept/reject decision set cannot move. Both halves are task-free and the echo
# is bounded by a fixed fragment limit; the assembled note is emitted on every
# envelope rejection, with the incumbent sentence kept as the leading text.
_ENVELOPE_SHAPE = "{room.device.operation(value),room.device.operation(value)}"
_ENVELOPE_ECHO_LIMIT = 64


def _fragment(text: str, limit: int = _ENVELOPE_ECHO_LIMIT) -> str:
    """Collapse whitespace and truncate, so a rejection echoes what was sent.

    Only runs of whitespace are collapsed and only the tail beyond ``limit`` is
    dropped: no character is added, removed, reordered or repaired, so the result
    observes the submission instead of proposing a rewritten one.
    """

    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."
_RELATIVE = re.compile(
    r"\b(increase|raise|decrease|lower|reduce)\b.*?\bby\s+(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def declared_arity(task_type: Any) -> int | None:
    """Return the sub-instruction count a multiN_* task type declares."""

    match = _TASK_ARITY.match(str(task_type))
    return int(match.group(1)) if match else None


def _literal(node: ast.AST) -> Any:
    # Observed option lists are printed without quotes, so an unquoted bare
    # identifier denotes the option name it spells: `set_speed(high)` carries
    # the string "high". Literal values keep their own types, and the observed
    # options list plus the declared parameter type still gate the result; only
    # identifiers are admitted here, never calls, subscripts or other
    # expressions, which keep raising the visible literal error below.
    if isinstance(node, ast.Name):
        return node.id
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise ValueError("arguments must be literal numbers, strings, or tuples") from exc


def _call_parts(node: ast.AST) -> tuple[str, str, str, list[Any]]:
    if not isinstance(node, ast.Call) or node.keywords:
        raise ValueError("each non-error slot must be a positional method call")
    method = node.func
    if not (
        isinstance(method, ast.Attribute)
        and isinstance(method.value, ast.Attribute)
        and isinstance(method.value.value, ast.Name)
    ):
        raise ValueError("method calls must use room.device.operation(...) syntax")
    return (
        method.value.value.id,
        method.value.attr,
        method.attr,
        [_literal(arg) for arg in node.args],
    )


def _slots(commands: str) -> list[ast.AST]:
    text = commands.strip()
    if not (text.startswith("{") and text.endswith("}")):
        raise ValueError(
            "commands must use exactly one enclosing pair of braces; saw start "
            f"{text[:1]!r} and end {text[-1:]!r} around {len(text)} characters "
            f"{_fragment(text)!r}; resubmit exactly one brace-enclosed "
            f"comma-separated list, e.g. {_ENVELOPE_SHAPE}"
        )
    try:
        expression = ast.parse(text, mode="eval").body
    except SyntaxError as exc:
        raise ValueError("commands are not valid HomeBench call syntax") from exc
    if not isinstance(expression, ast.Set):
        raise ValueError("commands must be a non-empty brace-enclosed slot list")
    return list(expression.elts)


def _clauses(instruction: str, expected: int | None) -> list[str]:
    pieces = [
        part.strip(" .")
        for part in re.split(r",\s*(?:and\s+)?|\s+(?:and then|then|and)\s+", instruction)
        if part.strip(" .")
    ]
    return pieces if expected is not None and len(pieces) == expected else []


def _matches_type(value: Any, declared: str) -> bool:
    kind = declared.replace(" ", "")
    if kind == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "str":
        return isinstance(value, str)
    if "Tuple[int,int,int]" in kind:
        return (
            isinstance(value, tuple)
            and len(value) == 3
            and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        )
    return True


class SubmissionValidator:
    """Validate structure/schema and only high-confidence relative arithmetic."""

    def __init__(self, *, instruction: str, home: Mapping[str, Any]):
        self.instruction = str(instruction)
        self.home = home
        self.expected_slots: int | None = None
        self.methods = {
            (str(row.get("room_name")), str(row.get("device_name")), str(row.get("operation"))): row
            for row in home.get("method", [])
            if isinstance(row, Mapping)
        }
        # The delivered room names, used only to name a spelling in the text of a
        # rejection; they never widen or narrow the decision below.
        self.rooms = delivered_rooms(home)

    def set_task_type(self, task_type: str) -> None:
        self.expected_slots = declared_arity(task_type)

    def _attribute(self, room: str, device: str, parameter: str) -> Mapping[str, Any] | None:
        state = self.home.get("home_status", {}).get(room, {}).get(device, {})
        attributes = state.get("attributes", {}) if isinstance(state, Mapping) else {}
        for name, value in attributes.items():
            if str(name).strip() == parameter and isinstance(value, Mapping):
                return value
        return None

    def _room_spelling_hint(self, room: str) -> str:
        """Name the delivered spelling when a submitted room denotes one room.

        Message-only: the submission is still rejected exactly as before, and the
        hint is empty whenever the room already is a delivered name or resolves to
        no single room, so the remaining rejection text stays byte-identical.
        """

        resolved = resolve_room_name(room, self.rooms)
        if resolved is None:
            return ""
        return (
            f"; household_index spells that room {resolved!r}, so submit the slot "
            "with that spelling"
        )

    def validate(self, commands: str) -> dict[str, Any]:
        errors: list[str] = []
        try:
            nodes = _slots(commands)
        except ValueError as exc:
            return {"valid": False, "errors": [str(exc)]}
        if self.expected_slots is not None and len(nodes) != self.expected_slots:
            errors.append(
                f"slot count is {len(nodes)} but task_type requires {self.expected_slots}; "
                "preserve repeated error_input slots"
            )
        clauses = _clauses(self.instruction, self.expected_slots)
        for index, node in enumerate(nodes, start=1):
            if isinstance(node, ast.Name) and node.id == "error_input":
                continue
            try:
                room, device, operation, arguments = _call_parts(node)
            except ValueError as exc:
                errors.append(f"slot {index}: {exc}")
                continue
            signature = self.methods.get((room, device, operation))
            if signature is None:
                errors.append(
                    f"slot {index}: unavailable method {room}.{device}.{operation}"
                    + self._room_spelling_hint(room)
                )
                continue
            parameters = signature.get("parameters") or []
            if len(arguments) != len(parameters):
                errors.append(
                    f"slot {index}: {operation} expects {len(parameters)} arguments, got {len(arguments)}"
                )
                continue
            for argument, parameter in zip(arguments, parameters):
                name = str(parameter.get("name"))
                if not _matches_type(argument, str(parameter.get("type", ""))):
                    errors.append(f"slot {index}: parameter {name} has the wrong type")
                    continue
                attribute = self._attribute(room, device, name)
                if attribute is not None:
                    options = attribute.get("options")
                    if isinstance(options, Sequence) and not isinstance(options, (str, bytes)):
                        if argument not in options:
                            errors.append(f"slot {index}: {name}={argument!r} is outside observed options")
                    try:
                        below = "lowest" in attribute and float(argument) < float(attribute["lowest"])
                        above = "highest" in attribute and float(argument) > float(attribute["highest"])
                    except (TypeError, ValueError):
                        below = above = False
                    if below or above:
                        errors.append(f"slot {index}: {name}={argument!r} is outside observed bounds")
                    if index <= len(clauses) and isinstance(argument, (int, float)):
                        relative = _RELATIVE.search(clauses[index - 1])
                        current = attribute.get("value")
                        if relative and isinstance(current, (int, float)):
                            delta = float(relative.group(2))
                            sign = -1 if relative.group(1).lower() in {"decrease", "lower", "reduce"} else 1
                            expected = current + sign * delta
                            if float(argument) != expected:
                                errors.append(
                                    f"slot {index}: relative {name} should be {expected:g} "
                                    f"from observed {current!r}, got {argument!r}"
                                )
        return {"valid": not errors, "errors": errors}
