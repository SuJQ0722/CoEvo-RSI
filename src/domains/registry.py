"""Lazy domain composition root; the only shared module naming built-in domains."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Any

from .base import DomainPlugin

_BUILTINS = {
    "homebench": "src.domains.homebench.plugin",
    "who_when": "src.domains.who_when.plugin",
    "rbac": "src.domains.rbac.plugin",
}
_ALIASES: dict[str, str] = {"whowhen": "who_when"}
_FACTORIES: dict[str, Callable[[], DomainPlugin]] = {}


def normalize_domain_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    return _ALIASES.get(normalized, normalized)


def register_domain(name: str, factory: Callable[[], DomainPlugin]) -> None:
    """Register an additional domain without editing orchestration or training."""
    normalized = name.strip().lower().replace("-", "_")
    if (
        not normalized
        or normalized in _BUILTINS
        or normalized in _FACTORIES
        or normalized in _ALIASES
    ):
        raise ValueError(f"empty or already registered domain: {name!r}")
    _FACTORIES[normalized] = factory


def get_domain_plugin(config_or_name: Mapping[str, Any] | str) -> DomainPlugin:
    name = (
        config_or_name
        if isinstance(config_or_name, str)
        else str((config_or_name.get("project") or {}).get("domain") or "")
    )
    name = normalize_domain_name(name)
    if name in _FACTORIES:
        return _FACTORIES[name]()
    if name not in _BUILTINS:
        raise ValueError(f"unsupported domain plugin: {name!r}; select project.domain explicitly")
    return import_module(_BUILTINS[name]).PLUGIN
