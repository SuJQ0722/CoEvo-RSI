"""Stable ports implemented by every CoEvoX benchmark domain."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from src.training.sampling import SamplingStats

    from .round_adapter import RoundAdapter


@dataclass(frozen=True)
class SmokeCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class SmokeReport:
    harness: str
    passed: bool
    checks: tuple[SmokeCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "harness": self.harness,
            "passed": self.passed,
            "checks": [check.__dict__ for check in self.checks],
        }


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float]
    per_sample: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectionDecision:
    accepted: bool
    reason: str
    deltas: dict[str, float]


@dataclass(frozen=True)
class SamplingRequest:
    """Resolved policy sampling settings, independent of task and acceptance semantics."""

    config_path: Path
    data_path: Path
    output_dir: Path
    policy_model: str
    base_url: str
    harness_name: str
    attempts_per_item: int
    accepted_per_item: int
    concurrency: int
    max_turns: int
    quality_tier: str
    limit: int | None = None
    resume: bool = False
    temperature: float | None = None
    on_attempt: Callable[[dict[str, Any]], None] | None = None

    def kwargs(self) -> dict[str, Any]:
        # Do not deepcopy callback objects as dataclasses.asdict would.
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class TrainingSourceRequest:
    accepted_path: Path
    data_path: Path
    output_dir: Path
    harness_name: str
    quality_tier: str
    sampling_config: Mapping[str, Any]


class BaseDomainPlugin:
    """Neutral defaults; domain implementations own any curation of accepted data."""

    def prepare_training_source(self, request: TrainingSourceRequest) -> Path:
        return request.accepted_path

    def curation_profile(self, sample: Any) -> dict:
        raise NotImplementedError("domain must define its training curation profile")

    def validate_curation_record(self, record: dict, sample: Any) -> None:
        """Optional domain acceptance check on the immutable raw trajectory."""


class DomainPlugin(Protocol):
    name: str
    harness_kind: str

    def round_adapter(self) -> RoundAdapter: ...

    def collect_training(self, request: SamplingRequest) -> SamplingStats: ...

    def prepare_training_source(self, request: TrainingSourceRequest) -> Path: ...

    def curation_profile(self, sample: Any) -> dict: ...

    def validate_curation_record(self, record: dict, sample: Any) -> None: ...

    def load_data(self, path: Path, **kwargs: Any) -> Any: ...

    def build_evidence(self, source: Path, output: Path) -> Path: ...

    def evaluate(self, value: Any, **kwargs: Any) -> EvaluationResult: ...

    def meta_user_prompt(self, config: Mapping[str, Any]) -> str: ...

    def smoke_test(self, harness_name: str, *, package: str, root: Path) -> SmokeReport: ...

    def select(
        self,
        parent: EvaluationResult,
        child: EvaluationResult,
        config: Mapping[str, Any],
    ) -> SelectionDecision: ...
