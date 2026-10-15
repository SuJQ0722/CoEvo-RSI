"""HomeBench composition root for the shared evolution pipeline."""

from src.domains.base import BaseDomainPlugin

from . import data, evaluator, evidence, prompt, selection, smoke


class HomeBenchPlugin(BaseDomainPlugin):
    name = "homebench"
    harness_kind = "homebench"

    def round_adapter(self):
        from .round_adapter import HomeBenchRoundAdapter

        return HomeBenchRoundAdapter()

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

    def load_data(self, path, **kwargs):
        return data.load(path, **kwargs)

    def build_evidence(self, source, output):
        return evidence.build(source, output)

    def evaluate(self, value, **kwargs):
        return evaluator.evaluate(value, **kwargs)

    def meta_user_prompt(self, config):
        return prompt.build(config)

    def smoke_test(self, harness_name, *, package, root):
        return smoke.test(harness_name, package=package, root=root)

    def select(self, parent, child, config):
        return selection.select(parent, child, config)


PLUGIN = HomeBenchPlugin()
