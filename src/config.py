"""Load and compose CoEvoX YAML configuration with OmegaConf."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.utils.persistence import atomic_text

_LEGACY_ENV_REFERENCE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")
_HARNESS_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# The shipped final evolved harness lives directly in ``<root>/generated``; new
# candidates produced by co-evolution runs are written to ``<root>/candidates/<name>``.
FINAL_HARNESS = "generated"
CANDIDATE_DIR = "candidates"
_TOP_LEVEL_HARNESSES = {"initial_harness", "system_zero", FINAL_HARNESS}


def harness_relative_path(name: str, base: str = "initial_harness") -> Path:
    """Return a harness directory relative to its domain harness root."""
    if name == base or name in _TOP_LEVEL_HARNESSES:
        return Path(name)
    return Path(CANDIDATE_DIR) / name


def harness_module_suffix(name: str, base: str = "initial_harness") -> str:
    """Return the dotted module path of a harness below its domain package."""
    return ".".join(harness_relative_path(name, base).parts)
_SECRET_CONFIG_KEYS = frozenset(
    {"api_key", "apikey", "access_token", "password", "secret", "token"}
)
_REDACTED_SECRET = "[REDACTED_SECRET]"


@dataclass(frozen=True)
class HarnessSpec:
    """Resolved domain-to-harness mapping."""

    domain: str
    kind: str
    package: str
    root: Path
    base: str
    active: str

    def path_for(self, name: str) -> Path:
        if not _HARNESS_NAME.fullmatch(name):
            raise ValueError(f"invalid harness name: {name!r}")
        return self.root / harness_relative_path(name, self.base)

    @property
    def active_path(self) -> Path:
        return self.path_for(self.active)


def _load_project_dotenv(path: Path) -> None:
    """Load a minimal shell-style .env without overriding the process environment."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        # Empty example overrides mean "use the YAML default".
        if value:
            os.environ.setdefault(name, value)


def _read_omegaconf(path: Path) -> DictConfig:
    # Existing configs use ${NAME} for environment variables. Translate that
    # legacy spelling to OmegaConf's resolver while retaining normal
    # lower-case/config-key interpolations.
    source = path.read_text(encoding="utf-8")
    source = _LEGACY_ENV_REFERENCE.sub(r"${oc.env:\1,''}", source)
    config = OmegaConf.create(source)
    if not isinstance(config, DictConfig):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return config


def _load_composed(path: Path, stack: tuple[Path, ...] = ()) -> DictConfig:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"configuration inheritance cycle: {chain}")
    config = _read_omegaconf(path)
    parent = config.pop("extends", None)
    if not parent:
        return config
    parent_path = Path(str(parent))
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return OmegaConf.merge(_load_composed(parent_path, (*stack, path)), config)


def load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    _load_project_dotenv(path.parent.parent / ".env")
    config = OmegaConf.to_container(_load_composed(path), resolve=True)
    if not isinstance(config, dict):  # Defensive: _load_composed enforces this.
        raise ValueError(f"configuration root must be a mapping: {path}")
    return config


def resolve_output_root(config: dict[str, Any], project_root: Path) -> Path:
    """Resolve the configured shared runtime-output directory."""

    configured = (config.get("paths") or {}).get("output", "output")
    path = Path(str(configured)).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def redact_config_secrets(value: Any) -> Any:
    """Return a detached config tree with credential values removed."""

    if isinstance(value, dict):
        return {
            key: (
                _REDACTED_SECRET
                if str(key).casefold() in _SECRET_CONFIG_KEYS and item not in (None, "")
                else redact_config_secrets(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_config_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_config_secrets(item) for item in value)
    return value


def write_resolved_config(config: dict[str, Any], path: Path) -> None:
    """Persist a resolved runtime config without copying secrets to artifacts."""

    safe_config = redact_config_secrets(config)
    with atomic_text(path) as stream:
        stream.write(OmegaConf.to_yaml(OmegaConf.create(safe_config), resolve=True))


_HARNESS_OVERRIDE_KEYS = {
    "vllm": {"max_model_len"},
    "model": {
        "max_tokens",
        "temperature",
        "top_p",
        "enable_thinking",
        "reasoning_effort",
    },
    "agent": {"max_steps", "max_memory_length"},
    "execution": {"action_timeout_seconds", "max_observation_chars"},
    "harness": {"max_turns"},
}


def apply_harness_overrides(config: dict[str, Any], harness_path: Path) -> dict[str, Any]:
    """Merge a harness-local, deliberately narrow runtime override fragment."""

    path = harness_path.resolve() / "config.yaml"
    if not path.is_file():
        return config
    local = load_config(path)
    overrides = local.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError(f"harness overrides must be a mapping: {path}")
    violations: list[str] = []
    for section, values in overrides.items():
        allowed = _HARNESS_OVERRIDE_KEYS.get(section)
        if allowed is None or not isinstance(values, dict):
            violations.append(str(section))
            continue
        violations.extend(f"{section}.{key}" for key in values if key not in allowed)
    if violations:
        raise ValueError(
            f"unsupported harness overrides in {path}: {', '.join(sorted(violations))}"
        )
    merged = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.create(config), OmegaConf.create(overrides)),
        resolve=True,
    )
    if not isinstance(merged, dict):  # pragma: no cover - merge preserves mapping
        raise TypeError("merged harness configuration must be a mapping")
    return merged


def load_domain_config(domain: str, config_dir: Path | None = None) -> dict[str, Any]:
    """Load ``base.yaml`` merged with the selected domain configuration."""
    config_dir = (config_dir or Path(__file__).resolve().parents[1] / "configs").resolve()
    from src.domains.registry import normalize_domain_name

    normalized = normalize_domain_name(domain)
    path = config_dir / f"{normalized}.yaml"
    if not path.is_file():
        choices = ", ".join(sorted(item.stem for item in config_dir.glob("*.yaml")))
        raise ValueError(f"unknown domain {domain!r}; available configs: {choices}")
    return load_config(path)


def active_harness(config: dict[str, Any]) -> str:
    harness = config.get("harness")
    if not isinstance(harness, dict) or not harness.get("active"):
        return "initial_harness"
    return str(harness["active"])


def harness_spec(
    config: dict[str, Any],
    *,
    project_root: Path | None = None,
    require_active: bool = True,
) -> HarnessSpec:
    """Resolve and validate the harness selected by one domain config."""

    project = config.get("project") or {}
    harness = config.get("harness") or {}
    if not isinstance(project, dict) or not isinstance(harness, dict):
        raise ValueError("project and harness configuration must be mappings")
    domain = str(project.get("domain") or "").strip()
    kind = str(harness.get("kind") or "").strip()
    package = str(harness.get("package") or "").strip()
    root_value = str(harness.get("root") or "").strip()
    base = str(harness.get("base") or "initial_harness").strip()
    active = str(harness.get("active") or base).strip()
    missing = [
        name
        for name, value in (
            ("project.domain", domain),
            ("harness.kind", kind),
            ("harness.package", package),
            ("harness.root", root_value),
        )
        if not value
    ]
    if missing:
        raise ValueError("missing domain harness mapping: " + ", ".join(missing))
    if not package.startswith("harnesses."):
        raise ValueError(f"harness.package must be under harnesses: {package!r}")
    if not _HARNESS_NAME.fullmatch(base) or not _HARNESS_NAME.fullmatch(active):
        raise ValueError(f"invalid base/active harness: base={base!r} active={active!r}")
    root = Path(root_value)
    if not root.is_absolute():
        root = (project_root or Path.cwd()) / root
    spec = HarnessSpec(domain, kind, package, root.resolve(), base, active)
    base_path = spec.path_for(base)
    if not base_path.is_dir():
        raise FileNotFoundError(f"base harness directory not found: {base_path}")
    if require_active and not spec.active_path.is_dir():
        raise FileNotFoundError(f"active harness directory not found: {spec.active_path}")
    return spec


def set_active_harness(config_path: Path, harness_name: str) -> None:
    lines = config_path.read_text(encoding="utf-8").splitlines()
    harness_start: int | None = None
    harness_end = len(lines)
    for index, line in enumerate(lines):
        if line == "harness:":
            harness_start = index
            continue
        if harness_start is not None and line and not line[0].isspace():
            harness_end = index
            break
    if harness_start is None:
        raise ValueError(f"configuration has no top-level harness section: {config_path}")
    active_pattern = re.compile(r"^(\s+)active\s*:")
    for index in range(harness_start + 1, harness_end):
        match = active_pattern.match(lines[index])
        if match:
            lines[index] = f"{match.group(1)}active: {harness_name}"
            break
    else:
        lines.insert(harness_start + 1, f"  active: {harness_name}")
    with atomic_text(config_path) as stream:
        stream.write("\n".join(lines) + "\n")
