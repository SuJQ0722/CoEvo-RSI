"""Domain curation and canonical SFT conversion shared by every training entry."""

from pathlib import Path

from src.domains import get_domain_plugin
from src.domains.base import TrainingSourceRequest
from src.training.ms_swift import prepare_ms_swift_dataset
from src.utils.logging import get_logger


def prepare_training_dataset(
    *,
    domain: str,
    quality_tier: str,
    accepted_rollouts: Path,
    training_dataset: Path,
    train_data: Path,
    harness_name: str,
    sampling_config: dict,
    training_config: dict,
    config: dict | None = None,
    val_data: Path | None = None,
    project_root: Path | None = None,
    validation_python: str | None = None,
) -> tuple[int, Path]:
    """Prepare the exact SFT source used by both fresh and resumed candidates."""
    plugin = get_domain_plugin(domain)
    if (training_config.get("data_curation") or {}).get("mode", "heuristic") == "agent":
        from src.training.curation import curate_dataset

        if config is None or val_data is None or project_root is None:
            raise ValueError("agent data curation requires config, val_data and project_root")
        return curate_dataset(
            plugin=plugin,
            accepted_path=accepted_rollouts,
            training_dataset=training_dataset,
            train_data=train_data,
            val_data=val_data,
            harness_name=harness_name,
            quality_tier=quality_tier,
            sampling_config=sampling_config,
            training_config=training_config,
            config=config,
            project_root=project_root,
            validation_python=validation_python,
        )
    training_source = plugin.prepare_training_source(
        TrainingSourceRequest(
            accepted_path=accepted_rollouts,
            data_path=train_data,
            output_dir=training_dataset.parent,
            harness_name=harness_name,
            quality_tier=quality_tier,
            sampling_config=sampling_config,
        )
    )
    prepared = prepare_ms_swift_dataset(
        training_source,
        training_dataset,
        tool_policy="preserve",
        sample_id_prefix_repeats=training_config.get("sample_id_prefix_repeats"),
    )
    if not prepared:
        raise RuntimeError("no usable SFT records after domain curation")
    get_logger("training.ms_swift").info(
        "prepared dataset records=%d source=%s output=%s quality_tier=%s",
        prepared,
        training_source,
        training_dataset,
        quality_tier,
    )
    return prepared, training_source
