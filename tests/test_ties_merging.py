"""Numerical and provenance checks using tiny dense checkpoints on CPU."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from src.meta_evolver.merging import merge_models, merge_recipe, ties_tensor
from src.meta_evolver.weights import TensorFiles, describe_model


def test_ties_resolves_signs_trims_and_preserves_zero_coordinates():
    base = torch.full((4,), 10.0)
    donors = [
        base + torch.tensor(delta) for delta in ([4, -2, 0.1, 0], [-2, -6, 0.2, 0], [1, 3, 0.3, 0])
    ]
    merged = ties_tensor(base, lambda: iter(donors), [1, 1, 1], density=1, scale=1)
    torch.testing.assert_close(merged, torch.tensor([12.5, 6.0, 10.2, 10.0]))
    trimmed = ties_tensor(base, lambda: iter(donors), [1, 1, 1], density=0.5, scale=1)
    torch.testing.assert_close(trimmed, torch.tensor([12.5, 6.0, 10.0, 10.0]))


def test_ties_sign_tie_and_scale():
    base = torch.tensor([1.0, 1.0])
    result = ties_tensor(
        base,
        lambda: iter([torch.tensor([3.0, 5.0]), torch.tensor([-1.0, 3.0])]),
        [1, 1],
        density=1,
        scale=0.5,
    )
    torch.testing.assert_close(result, torch.tensor([1.0, 2.5]))


def checkpoint(root, name, values):
    path = root / name
    path.mkdir()
    (path / "config.json").write_text('{"model_type":"tiny"}')
    (path / "tokenizer.json").write_text('{"vocab":"same"}')
    save_file(
        {"w": torch.tensor(values, dtype=torch.bfloat16), "buffer": torch.tensor([1])},
        path / "model.safetensors",
    )
    return describe_model(path, project_root=root, cache_root=root / "cache")


def test_dense_ties_cache_recipe_base_and_weights(tmp_path):
    base = checkpoint(tmp_path, "base", [10, 10, 10, 10])
    base["common_base_id"] = base["effective_weights_hash"]
    a = checkpoint(tmp_path, "a", [14, 8, 10, 10])
    b = checkpoint(tmp_path, "b", [8, 4, 10, 10])
    for parent in (a, b):
        parent["common_base_id"] = base["common_base_id"]
    options = {
        "base": base,
        "project_root": tmp_path,
        "cache_root": tmp_path / "cache",
        "density": 1,
    }
    mixed = merge_models([a, b], **options)
    tensor = TensorFiles(Path(mixed["path"])).tensor("w")
    torch.testing.assert_close(tensor, torch.tensor([14, 6, 10, 10], dtype=torch.bfloat16))
    again = merge_models([b, a], **options)
    assert again["path"] == mixed["path"]
    assert again["effective_weights_hash"] == mixed["effective_weights_hash"]
    scaled = merge_models([a, b], **options, scale=0.5)
    assert scaled["path"] != mixed["path"]
    linear = merge_models(
        [a, b],
        base=base,
        project_root=tmp_path,
        cache_root=tmp_path / "cache",
        method="linear",
        weights=[3, 1],
    )
    torch.testing.assert_close(
        TensorFiles(Path(linear["path"])).tensor("w"),
        torch.tensor([12.5, 7, 10, 10], dtype=torch.bfloat16),
    )
    metadata = json.loads((Path(mixed["path"]) / "lineage_merge.json").read_text())
    assert metadata["recipe"]["base_weights_hash"] == base["effective_weights_hash"]
    assert metadata["recipe"]["trim_scope"] == "tensor"
    with pytest.raises(ValueError, match="duplicate"):
        merge_recipe([a, a], base=base)
    with pytest.raises(ValueError, match="common base"):
        merge_recipe([a, b], base=None)
    with pytest.raises(ValueError, match="non-base"):
        merge_recipe([a, base], base=base)
    with pytest.raises(ValueError, match="positive"):
        merge_recipe([a, b], base=base, weights=[1, -1])
    with pytest.raises(ValueError, match="initialization"):
        merge_recipe([a, dict(b, common_base_id="other")], base=base)


def test_merge_rejects_changed_donor(tmp_path):
    base = checkpoint(tmp_path, "base", [0, 0])
    a = checkpoint(tmp_path, "a", [1, 2])
    b = checkpoint(tmp_path, "b", [3, 4])
    for parent in (a, b):
        parent["common_base_id"] = base["effective_weights_hash"]
    save_file(
        {"w": torch.tensor([4, 4], dtype=torch.bfloat16), "buffer": torch.tensor([1])},
        Path(a["path"]) / "model.safetensors",
    )
    with pytest.raises(ValueError, match="input changed"):
        merge_models([a, b], base=base, project_root=tmp_path, cache_root=tmp_path / "cache")
