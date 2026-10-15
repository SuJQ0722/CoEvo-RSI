"""Strict formal marker-to-access normalization."""

from __future__ import annotations

DENIAL_MARKER = "ACCESS_DENIED"
GRANT_MARKER = "ACCESS_GRANTED"


def parse_access(response: object) -> int:
    """Parse only the two protocol markers; malformed output fails closed."""
    text = str(response or "").strip()
    marker = text.upper()
    if marker == GRANT_MARKER:
        return 1
    return 0
