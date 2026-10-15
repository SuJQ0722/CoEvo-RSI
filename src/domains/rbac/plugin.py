"""RBAC domain composition root."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.domains.base import EvaluationResult, SmokeReport

from . import data, evaluator, evidence, prompt, selection, smoke


from src.domains.base import BaseDomainPlugin


class RBACPlugin(BaseDomainPlugin):
    name = "rbac"
    harness_kind = "rbac"

    def round_adapter(self):
        from .round_adapter import RBACRoundAdapter
        return RBACRoundAdapter()

    def collect_training(self, request):
        from .rejection_sampling import collect
        return collect(request)

    def prepare_training_source(self, request):
        from .rejection_sampling import prepare_training_source
        return prepare_training_source(request)

    def curation_profile(self, sample):
        from .curation import profile
        return profile(sample)

    def validate_curation_record(self, record, sample):
        from .curation import validate_record
        validate_record(record, sample)

    def load_data(self, path: Path, **kwargs: Any):
        return data.load(path, **kwargs)

    def build_evidence(self, source: Path, output: Path) -> Path:
        return evidence.build(source, output)

    def evaluate(self, value: Any, **kwargs: Any) -> EvaluationResult:
        return evaluator.evaluate(value, **kwargs)

    def meta_user_prompt(self, config: Mapping[str, Any]) -> str:
        return prompt.build(config)

    def smoke_test(self, harness_name: str, *, package: str, root: Path) -> SmokeReport:
        return smoke.test(harness_name, package=package, root=root)

    def select(self, parent, child, config):
        return selection.select(parent, child, config)


PLUGIN = RBACPlugin()
