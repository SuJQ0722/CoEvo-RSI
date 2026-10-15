"""Prompt resources and deterministic initial-message construction."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .context import build_household_index
from .grounding import request_room_hints
from .settings import HarnessSettings

RESOURCE_DIR = Path(__file__).resolve().parent

# The ordered rung frame is a compact order reminder beside the request, never a
# second copy of the task: every rung is trimmed and the rung count is capped, so
# a request of any length renders a bounded block.
_RUNG_LIMIT = 64
_RUNG_CAP = 12
_NEUTRAL_RUNG = "(clause not isolated)"

# The preflight aligns one request clause with one output slot through the same
# separators, so the frame and the clause-indexed checks agree on where a single
# requested operation ends.
_CLAUSE_SPLIT = re.compile(r",\s*(?:and\s+)?|\s+(?:and then|then|and)\s+")
_LEADING_THEN = re.compile(r"^then\s+", re.IGNORECASE)

# The closed-slot grounding rule is rendered from two independently droppable
# halves: the target-resolution test that decides closure and the authority split
# for everything the index cannot answer.
_GROUNDING_RULES: tuple[str, ...] = (
    "Resolve each slot's room and device against household_index before closing anything: an "
    "exact index name wins, any phrasing that section resolves to exactly one index name is "
    "that name in the index's own spelling, and only a room (or device in that exact room) "
    "denoting no index name closes as error_input in its original position; never invent or "
    "substitute a target.",
    "The index proves existence only; list_methods owns operation support; inspect_home "
    "owns values, options and bounds.",
)


def _read_resource(name: str) -> str:
    return (RESOURCE_DIR / name).read_text(encoding="utf-8").strip()


def _render_grounding_authority_clause() -> str:
    """Render the closure-grounding rule as one bounded clause."""

    return " ".join(_GROUNDING_RULES)


# Read economy is rendered from independently droppable clauses: the identity
# test that decides whether a read is already answered, the scope rule that
# decides how far one answer reaches, and the explicit non-cap guard that keeps
# a slot whose evidence no observation covers served by exactly one read. The
# clauses restate only the request's own reuse decision, so no read is ever
# suppressed and no read budget is introduced.
_READ_IDENTITY_CLAUSES: tuple[str, ...] = (
    "Search the observations already in this conversation before reading again: "
    "one observation covers the whole scope it returned, so a room-level "
    "inspect_home result already holds the values, bounds and options it reported "
    "for every device in that room, and a room-level list_methods result already "
    "holds every operation it reported for that room.",
    "A read is identified by the room it targets together with the evidence it "
    "requests, so keep a private list of the (room, evidence target) pair of every "
    "read already answered here and never issue a read whose pair duplicates that "
    "list; do not re-read merely to confirm evidence you already hold, and resolve "
    "that slot from the observation already in this conversation instead.",
)
_READ_NON_CAP_CLAUSE = (
    "This is not a cap on reads: a slot that still lacks evidence that no "
    "observation here covers keeps exactly one relevant read for its missing room "
    "or device, and prefer a read whose scope supports several open slots."
)


def _render_read_reuse_clauses() -> str:
    """Render the read-identity and observation-scope rule as bounded clauses."""

    clauses = [*_READ_IDENTITY_CLAUSES, _READ_NON_CAP_CLAUSE]
    return "\n".join(f"- {clause}" for clause in clauses)


def _render_request_room_hints(
    instruction: str,
    index: Mapping[str, list[str]],
) -> str:
    """Name the delivered spelling for the request phrasings the fold resolves.

    The pairs come from the same delivered index that is rendered above them, so
    the block is empty unless this task's own request and its own household data
    produce a match, and it is capped by the resolver.
    """

    hints = request_room_hints(instruction, tuple(index.keys()))
    if not hints:
        return ""
    lines = "".join(
        f"\n- {phrase!r} denotes {room!r}"
        for phrase, room in hints
    )
    return (
        "\nRequest phrasings below are not exact index names, but under a fold that ignores "
        "case, punctuation, spacing and letters repeated inside a name, or a single-letter "
        "difference, each resolves to exactly one room above; use the index spelling:"
        + lines
    )


def _clause_fragments(instruction: str) -> list[str]:
    """Split the request into clause fragments in request order."""

    fragments: list[str] = []
    for part in _CLAUSE_SPLIT.split(str(instruction)):
        fragment = _LEADING_THEN.sub("", part.strip(" ."))
        if fragment:
            fragments.append(fragment)
    return fragments


def _trim_rung(text: str, limit: int = _RUNG_LIMIT) -> str:
    """Collapse whitespace and truncate, so one rung stays bounded."""

    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    cut = collapsed[: limit - 3].rstrip()
    boundary = cut.rfind(" ")
    if boundary >= 12:
        cut = cut[:boundary].rstrip(" ,.;:")
    return cut + "..."


def _rungs(fragments: Sequence[str], arity: int) -> list[str]:
    """Give every declared slot one rung, in request order.

    Surplus wording folds into the last declared rung, so no arriving clause is
    dropped; a short split pads with a neutral marker rather than an invented
    operation, because the frame may only restate the request that arrived.
    """

    if arity <= 0 or not fragments:
        return []
    if len(fragments) >= arity:
        return list(fragments[: arity - 1]) + [" ".join(fragments[arity - 1 :])]
    return list(fragments) + [_NEUTRAL_RUNG] * (arity - len(fragments))


def render_slot_ladder(instruction: str, declared_slots: int | None) -> str:
    """Render the ordered rung frame for a request of declared arity.

    Empty whenever the declared slot count is unknown or out of the bounded
    range, so the caller keeps the prompt it would otherwise have rendered.
    """

    if not isinstance(declared_slots, int) or isinstance(declared_slots, bool):
        return ""
    if declared_slots <= 0 or declared_slots > _RUNG_CAP:
        return ""
    rungs = _rungs(_clause_fragments(instruction), declared_slots)
    if not rungs:
        return ""
    body = " ".join(f"{index}) {_trim_rung(rung)}" for index, rung in enumerate(rungs, 1))
    return (
        f"Ordered requested operations (declared arity {declared_slots}), one rung per output "
        f"slot, in request order: {body}. These rungs are an order frame, not task facts: keep "
        "exactly one output slot per rung in this order, never merge, drop or repeat a rung, "
        "and resolve every rung from the request itself and the read observations rather than "
        "from this frame."
    )


def build_user_prompt(
    instruction: str,
    home: Mapping[str, Any],
    *,
    prompt_mode: str,
    declared_slots: int | None = None,
) -> str:
    grounding_clause = _render_grounding_authority_clause()
    index = build_household_index(home)
    sections = [
        "<household_index>\n"
        "Available room and device names (details must be retrieved with tools):\n"
        + json.dumps(
            index,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + _render_request_room_hints(instruction, index)
        + "\n</household_index>"
    ]
    if prompt_mode == "icl":
        sections.append(
            "<icl_examples>\n"
            "These examples show output syntax for other household states; never copy "
            "their device availability or values into the current task.\n"
            + _read_resource("icl_examples.txt")
            + "\n</icl_examples>"
        )
    ladder = render_slot_ladder(instruction, declared_slots)
    sections.append(
        "<User instructions>\n" + str(instruction) + (f"\n\n{ladder}" if ladder else "")
    )
    # Keep the completion decision adjacent to the task so it remains salient
    # after longer inventory and ICL sections.
    sections.append(
        "<evidence_sufficiency_checkpoint>\n"
        "Before each tool call, re-check the ordered slots: a slot is resolved only by "
        "evidence already observed here, and error_input only as below. "
        + grounding_clause
        + "\n- If a slot lacks required evidence, call exactly one relevant read tool.\n"
        + _render_read_reuse_clauses()
        + "\n"
        "- If every slot is resolved, call final_answer now, exactly once, and never read "
        "again. After any read, re-check the ordered slots against the observation just "
        "returned: if it was the last missing piece, this turn must be final_answer, not "
        "another read, and never add a confirmation read once all slots are resolved.\n"
        "Make exactly one tool call on this assistant turn; never issue sibling tool calls.\n"
        "</evidence_sufficiency_checkpoint>"
    )
    return "\n\n".join(sections)


def initial_messages(
    settings: HarnessSettings,
    instruction: str,
    home: Mapping[str, Any],
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _read_resource("prompt.txt")},
        {
            "role": "user",
            "content": build_user_prompt(
                instruction,
                home,
                prompt_mode=settings.prompt_mode,
            ),
        },
    ]
