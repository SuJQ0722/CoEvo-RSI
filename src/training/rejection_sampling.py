"""Domain-independent rejection-sampling dispatch."""

from src.config import load_config
from src.domains import get_domain_plugin
from src.domains.base import SamplingRequest
from src.training.sampling import SamplingStats


def collect_to_directory(request: SamplingRequest) -> SamplingStats:
    plugin = get_domain_plugin(load_config(request.config_path))
    return plugin.collect_training(request)


if __name__ == "__main__":
    raise SystemExit(
        "Use python -m src.training.pipeline --config PATH --sampling-only (see --help)"
    )
