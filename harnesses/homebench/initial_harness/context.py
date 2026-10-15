"""Build the bounded public context shown before the first model call."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _public_home_parts(home: Mapping[str, Any]) -> tuple[Mapping[str, Any], Sequence[Any]]:
    status = home.get("home_status")
    methods = home.get("method")
    if not isinstance(status, Mapping):
        raise TypeError("home.home_status must be an object")
    if not isinstance(methods, Sequence) or isinstance(methods, (str, bytes)):
        raise TypeError("home.method must be a list")
    return status, methods


def build_household_index(home: Mapping[str, Any]) -> dict[str, list[str]]:
    """Return room/device names without eagerly exposing state or method details."""

    status, methods = _public_home_parts(home)
    devices_by_room: dict[str, list[str]] = {}

    for raw_room, raw_state in status.items():
        room = str(raw_room)
        devices: list[str] = []
        if isinstance(raw_state, Mapping):
            devices.extend(str(name) for name in raw_state if name != "room_name")
        devices_by_room[room] = list(dict.fromkeys(devices))

    # Include public method-only entries defensively, while keeping the initial
    # context to an inventory rather than the full operation schema.
    for raw_method in methods:
        if not isinstance(raw_method, Mapping):
            continue
        room = str(raw_method.get("room_name") or "").strip()
        device = str(raw_method.get("device_name") or "").strip()
        if not room or not device:
            continue
        devices = devices_by_room.setdefault(room, [])
        if device not in devices:
            devices.append(device)

    return devices_by_room
