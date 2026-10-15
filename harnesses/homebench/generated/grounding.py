"""Resolve request phrasing to delivered index names without hardcoded spellings.

The household index handed to the harness at ``initialize`` is the only spelling
authority available here, so every comparison form is derived at runtime from the
public home data that task delivers: case, punctuation, underscores, internal
spacing and letters repeated inside a name carry no meaning for matching, and a
request phrasing that denotes exactly one delivered name is reported as that
name so the caller can reuse the delivered spelling instead of closing a slot.
Nothing in this module knows a household noun.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .context import build_household_index

_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_REPEATED = re.compile(r"(.)\1+")
_WORDS = re.compile(r"[0-9A-Za-z]+")

_MAX_PHRASE_WORDS = 3
_HINT_LIMIT = 4
_NEAR_MATCH_MIN_LENGTH = 6


def comparison_form(name: Any) -> str:
    """Fold one name to the form used for candidate matching.

    A letter repeated inside a name is folded to a single occurrence, so a
    delivered spelling and a request phrasing that differ only by those surface
    conventions share one form. Every character comes from the name handed in.
    """

    folded = _NON_ALNUM.sub("", str(name).casefold())
    return _REPEATED.sub(r"\1", folded)


def _one_letter_apart(left: str, right: str) -> bool:
    """Return True when two comparison forms differ by at most one letter."""

    if len(left) > len(right):
        left, right = right, left
    if len(right) - len(left) > 1:
        return False
    for index, character in enumerate(left):
        if character != right[index]:
            if len(left) == len(right):
                return left[index + 1 :] == right[index + 1 :]
            return left[index:] == right[index + 1 :]
    return len(left) != len(right)


def delivered_rooms(home: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the room names the public home data delivers, in index order."""

    try:
        return tuple(build_household_index(home).keys())
    except (TypeError, AttributeError):
        return ()


def resolve_room_name(name: Any, rooms: Sequence[str]) -> str | None:
    """Return the one delivered room a phrasing denotes, otherwise None.

    A phrasing that already is a delivered name resolves to nothing, because
    there is no spelling to correct. Otherwise a delivered name is returned only
    when exactly one of them shares the phrasing's comparison form, or - once the
    phrasing is long enough to carry meaning - sits one letter away from it. No
    candidate, several candidates and an empty comparison form all return None,
    so the caller keeps whatever decision it would have made on its own.
    """

    if name in rooms:
        return None
    form = comparison_form(name)
    if not form:
        return None
    folded = [room for room in rooms if comparison_form(room) == form]
    if folded:
        return folded[0] if len(folded) == 1 else None
    if len(form) < _NEAR_MATCH_MIN_LENGTH:
        return None
    near = [
        room
        for room in rooms
        if abs(len(comparison_form(room)) - len(form)) <= 1
        and _one_letter_apart(form, comparison_form(room))
    ]
    return near[0] if len(near) == 1 else None


def request_room_hints(
    instruction: str,
    rooms: Sequence[str],
    *,
    limit: int = _HINT_LIMIT,
) -> list[tuple[str, str]]:
    """List the (request phrasing, delivered name) pairs this request resolves.

    Only contiguous word runs of the request are considered, the longest run at
    each position first, each delivered name is reported at most once, and the
    result is truncated at ``limit``, so both the work and the rendered text stay
    bounded for any instruction.
    """

    words = _WORDS.findall(str(instruction))
    hints: list[tuple[str, str]] = []
    claimed: set[str] = set()
    inspected: set[str] = set()
    for start in range(len(words)):
        for size in range(_MAX_PHRASE_WORDS, 0, -1):
            if start + size > len(words):
                continue
            phrase = " ".join(words[start : start + size])
            if phrase in inspected:
                continue
            inspected.add(phrase)
            room = resolve_room_name(phrase, rooms)
            if room is None or room in claimed:
                continue
            claimed.add(room)
            hints.append((phrase, room))
            if len(hints) >= limit:
                return hints
    return hints
