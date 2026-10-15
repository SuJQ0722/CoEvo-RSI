"""Dense linear/TIES merging with explicit recipes and bounded tensor-level memory.

TIES uses task vectors relative to the common base: trim, elect the weighted
sign, then take a sign-consistent weighted mean. Trimming is per tensor, not
global across a flattened checkpoint. See https://arxiv.org/abs/2306.01708.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from src.utils.persistence import atomic_json

from .lineage import digest
from .weights import TensorFiles, compatible, copy_assets, describe_model


def merge_recipe(parents, *, base=None, method="ties", weights=None, density=0.2, scale=1.0):
    if method not in {"ties", "linear"}:
        raise ValueError("merge method must be ties or linear")
    if len(parents) < 2:
        raise ValueError("merging requires at least two distinct models")
    hashes = [parent["effective_weights_hash"] for parent in parents]
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate effective weights in merge parents")
    common = parents[0].get("common_base_id")
    if not common or any(parent.get("common_base_id") != common for parent in parents):
        raise ValueError("unproven_common_initialization")
    if any(not compatible(parents[0], parent) for parent in parents[1:]):
        raise ValueError("incompatible_architecture_tokenizer_or_layout")
    weights = [1.0] * len(parents) if weights is None else weights
    if not isinstance(weights, (list, tuple)) or len(weights) != len(parents):
        raise ValueError("one positive weight is required per merge parent")
    if any(
        isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0
        for w in weights
    ):
        raise ValueError("merge weights must be finite and positive")
    total = sum(weights)
    if not math.isfinite(total):
        raise ValueError("merge weight sum is not finite")
    recipe = {
        "version": "dense_merge_v1",
        "method": method,
        "common_base_id": common,
        "parents": sorted(
            [
                {
                    "model_id": parent["id"],
                    "weights_hash": parent["effective_weights_hash"],
                    "weight": weight / total,
                }
                for parent, weight in zip(parents, weights)
            ],
            key=lambda item: item["weights_hash"],
        ),
    }
    if method == "ties":
        if (
            base is None
            or base["effective_weights_hash"] != common
            or not compatible(parents[0], base)
        ):
            raise ValueError("TIES requires the materialized common base checkpoint")
        if base["effective_weights_hash"] in hashes:
            raise ValueError("TIES donors must be non-base checkpoints")
        if (
            isinstance(density, bool)
            or not isinstance(density, (int, float))
            or not math.isfinite(density)
            or not 0 < density <= 1
        ):
            raise ValueError("TIES density must be in (0, 1]")
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(scale)
            or scale <= 0
        ):
            raise ValueError("TIES scale must be finite and positive")
        recipe.update(
            base_weights_hash=base["effective_weights_hash"],
            density=float(density),
            scale=float(scale),
            trim_scope="tensor",
        )
    return recipe


def _trim(delta, density):
    import torch

    if not torch.isfinite(delta).all():
        raise ValueError("non-finite task vector")
    if density == 1 or delta.numel() == 0:
        return delta
    flat = delta.reshape(-1)
    keep = max(1, math.ceil(flat.numel() * density))
    indices = flat.abs().topk(keep, sorted=False).indices
    trimmed = torch.zeros_like(flat)
    trimmed[indices] = flat[indices]
    return trimmed.reshape(delta.shape)


def ties_tensor(base, donors, weights, *, density, scale):
    """donors is a repeatable iterator factory: two passes avoid stacking K tensors."""
    import torch

    base = base.float()
    if not torch.isfinite(base).all():
        raise ValueError("non-finite base tensor")
    elected = torch.zeros_like(base)
    for tensor, weight in zip(donors(), weights):
        delta = _trim(tensor.float() - base, density)
        elected.add_(delta, alpha=weight)
    elected.sign_()
    numerator, denominator = torch.zeros_like(base), torch.zeros_like(base)
    for tensor, weight in zip(donors(), weights):
        delta = _trim(tensor.float() - base, density)
        mask = (delta != 0) & (delta.sign() == elected)
        numerator.add_(delta * mask, alpha=weight)
        denominator.add_(mask.to(base.dtype), alpha=weight)
    # Exact sign ties have no donor; preserve the base at that coordinate.
    safe_denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    return base + scale * numerator / safe_denominator


def merge_models(parents, *, base, cache_root: Path, project_root: Path, **options):
    recipe = merge_recipe(parents, base=base, **options)
    by_hash = {p["effective_weights_hash"]: p for p in parents}
    ordered = [by_hash[p["weights_hash"]] for p in recipe["parents"]]
    weights = [p["weight"] for p in recipe["parents"]]
    checked = ordered + ([base] if recipe["method"] == "ties" else [])
    for model in checked:
        current = describe_model(model["path"], project_root=project_root, cache_root=cache_root)
        if current["effective_weights_hash"] != model["effective_weights_hash"] or not compatible(
            current, model
        ):
            raise ValueError("merge input changed after the model inventory was frozen")

    output = cache_root / f"{recipe['method']}_{digest(recipe)}"
    marker = output / "lineage_merge.json"
    receipt = json.loads(marker.read_text()) if marker.is_file() else None
    if receipt and receipt.get("recipe") != recipe:
        raise ValueError("merge cache recipe mismatch")
    if receipt is None:
        import torch
        from safetensors.torch import save_file

        output.mkdir(parents=True, exist_ok=True)
        readers = [TensorFiles(Path(model["path"])) for model in ordered]
        anchor = TensorFiles(Path(base["path"])) if recipe["method"] == "ties" else readers[0]
        if any(set(reader.mapping) != set(anchor.mapping) for reader in readers):
            raise ValueError("merge tensor names differ")
        weight_map, total_size = {}, 0
        for index, key in enumerate(sorted(anchor.mapping), 1):
            template = anchor.tensor(key)

            def donors(key=key, template=template):
                for reader in readers:
                    tensor = reader.tensor(key)
                    if (
                        tensor.shape != template.shape
                        or tensor.is_floating_point() != template.is_floating_point()
                    ):
                        raise ValueError(f"merge tensor layout/type mismatch: {key}")
                    yield tensor

            if not template.is_floating_point():
                if any(not torch.equal(tensor, template) for tensor in donors()):
                    raise ValueError(f"non-floating buffers differ: {key}")
                merged = template
            elif recipe["method"] == "ties":
                merged = ties_tensor(
                    template, donors, weights, density=recipe["density"], scale=recipe["scale"]
                )
            else:
                merged = torch.zeros_like(template, dtype=torch.float32)
                for tensor, weight in zip(donors(), weights):
                    merged.add_(tensor.float(), alpha=weight)
            merged = merged.to(template.dtype)
            if merged.is_floating_point() and not torch.isfinite(merged).all():
                raise ValueError(f"non-finite merged tensor: {key}")
            shard = f"model-{index:05d}-of-{len(anchor.mapping):05d}.safetensors"
            save_file({key: merged.contiguous()}, output / shard, metadata={"format": "pt"})
            weight_map[key] = shard
            total_size += merged.numel() * merged.element_size()
        copy_assets(Path(ordered[0]["path"]), output)
        atomic_json(
            output / "model.safetensors.index.json",
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
        )

    model = describe_model(
        output,
        project_root=project_root,
        cache_root=cache_root,
        common_base_id=recipe["common_base_id"],
        parent_ids=[p["id"] for p in ordered],
        route="merge",
    )
    if receipt and receipt["result_weights_hash"] != model["effective_weights_hash"]:
        raise ValueError("cached merged weights changed after completion")
    model.update(parent_weights=weights, merge_recipe=recipe)
    if receipt is None:
        atomic_json(
            marker, {"recipe": recipe, "result_weights_hash": model["effective_weights_hash"]}
        )
    return model
