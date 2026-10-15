"""Content-addressed dense model identities and one FP32 50/50 merge.

Model evolution uses full-parameter SFT checkpoints. Hashing and interpolation
operate directly on complete dense weights.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from src.utils.persistence import atomic_json
from src.utils.python_source import normalized_python

from .lineage import digest


def file_hash(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tree_hash(path: Path, *, suffixes=None):
    return digest(
        {
            p.relative_to(path).as_posix(): file_hash(p)
            for p in sorted(path.rglob("*"))
            if p.is_file()
            and "__pycache__" not in p.parts
            and p.suffix not in {".pyc", ".pyo"}
            and (suffixes is None or p.suffix in suffixes)
        }
    )


def harness_hash(path: Path):
    """Hash the complete executable bundle for evaluation-cache provenance.

    All Python (including dynamic/unreachable imports) uses normalization v1;
    non-Python assets use raw bytes. Reports, lineage metadata and caches are
    excluded. Preserve the per-file digests and sorted-JSON aggregation, which
    intentionally differ from semantic_harness_hash's reachable-code stream.
    """
    contents = {}
    for source in sorted(path.rglob("*")):
        if (
            not source.is_file()
            or "__pycache__" in source.parts
            or source.suffix in {".pyc", ".pyo"}
        ):
            continue
        if source.name in {"evolution_report.md", "lineage.json"}:
            continue
        if source.suffix == ".py":
            value = digest(
                normalized_python(source.read_text(encoding="utf-8"), filename=str(source))
            )
        else:
            value = file_hash(source)
        contents[source.relative_to(path).as_posix()] = value
    return digest(contents)


def resolve_model(source, project_root: Path) -> Path:
    path = Path(source).expanduser()
    local = path if path.is_absolute() else project_root / path
    if local.is_dir():
        return local.resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(source), local_files_only=True)).resolve()


class TensorFiles:
    """Read one safetensor at a time, or at most one legacy PyTorch shard."""

    def __init__(self, path: Path):
        from safetensors import safe_open

        self.path = path
        self.mapping = {}
        self._bin_name, self._bin = None, None
        shards = sorted(path.glob("model*.safetensors"))
        if shards:
            for shard in shards:
                with safe_open(shard, framework="pt", device="cpu") as stream:
                    for key in stream.keys():  # noqa: SIM118 -- safetensors handle, not a mapping
                        if key in self.mapping:
                            raise ValueError(f"duplicate tensor {key}")
                        self.mapping[key] = shard
        else:
            import torch

            shards = sorted(path.glob("pytorch_model*.bin"))
            for shard in shards:
                tensors = torch.load(shard, map_location="cpu", weights_only=True)
                for key in tensors:
                    if key in self.mapping:
                        raise ValueError(f"duplicate tensor {key}")
                    self.mapping[key] = shard
                del tensors
        if not self.mapping:
            raise ValueError(f"no dense model weights at {path}")

    def tensor(self, key):
        shard = self.mapping[key]
        if shard.suffix == ".safetensors":
            from safetensors import safe_open

            with safe_open(shard, framework="pt", device="cpu") as stream:
                return stream.get_tensor(key)
        if self._bin_name != shard:
            import torch

            self._bin = torch.load(shard, map_location="cpu", weights_only=True)
            self._bin_name = shard
        return self._bin[key]


def _tokenizer_identity(path: Path):
    names = (
        "tokenizer.json",
        "tokenizer.model",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "added_tokens.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
    )
    files = {name: file_hash(path / name) for name in names if (path / name).is_file()}
    if not files:
        raise ValueError(f"model has no tokenizer assets: {path}")
    # Trainer output can change name_or_path without changing the tokenizer.
    if (path / "tokenizer_config.json").is_file():
        cfg = json.loads((path / "tokenizer_config.json").read_text())
        for name in ("name_or_path", "_name_or_path"):
            cfg.pop(name, None)
        files["tokenizer_config.json"] = digest(cfg)
    return digest(files)


def _config_identity(path: Path):
    config = json.loads((path / "config.json").read_text())

    def clean(item):
        if isinstance(item, dict):
            return {
                k: clean(v)
                for k, v in item.items()
                if k
                not in {
                    "_name_or_path",
                    "transformers_version",
                    "torch_dtype",
                    "dtype",
                    "_commit_hash",
                }
            }
        if isinstance(item, list):
            return [clean(v) for v in item]
        return item

    return config, digest(clean(config))


def copy_assets(source: Path, output: Path):
    for path in source.iterdir():
        if path.is_file() and (
            path.suffix in {".json", ".model", ".txt", ".jinja", ".py", ".tiktoken"}
        ):
            if path.name.startswith(("trainer", "training", "optimizer", "scheduler", "lineage")):
                continue
            if ".index.json" in path.name:
                continue
            shutil.copy2(path, output / path.name)


def describe_model(
    source,
    *,
    project_root: Path,
    cache_root: Path,
    common_base_id=None,
    parent_ids=(),
    route="single",
):
    path = resolve_model(source, project_root)
    cfg, architecture = _config_identity(path)
    if cfg.get("quantization_config"):
        # Cannot hash integer tensors as effective dense weights.
        raise ValueError("quantized weights must be materialized to dense weights before LINEAGE")
    tensors = TensorFiles(path)
    files = sorted(
        set(tensors.mapping.values())
        | {
            p
            for p in path.iterdir()
            if p.is_file() and p.suffix in {".json", ".model", ".txt", ".jinja", ".tiktoken"}
        }
    )
    signature = {
        str(p): [p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns] for p in files
    }
    index_path = cache_root / "identities" / f"{digest(str(path))}.json"
    cached = json.loads(index_path.read_text()) if index_path.is_file() else {}
    if cached.get("file_signature") == signature:
        weights_hash, layout = cached["effective_weights_hash"], cached["layout"]
    else:
        h = hashlib.sha256()
        layout = {}
        for key in sorted(tensors.mapping):
            tensor = tensors.tensor(key)
            layout[key] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            h.update(key.encode() + b"\0" + json.dumps(list(tensor.shape)).encode() + b"\0")
            # Normalize actual values to FP32; same dense values in bf16/fp32 dedup.
            canonical = (
                tensor.float().contiguous() if tensor.is_floating_point() else tensor.contiguous()
            )
            h.update(str(canonical.dtype).encode() + b"\0")
            h.update(canonical.view(-1).view(__import__("torch").uint8).numpy().tobytes())
        weights_hash = h.hexdigest()
        atomic_json(
            index_path,
            {"file_signature": signature, "effective_weights_hash": weights_hash, "layout": layout},
        )
    return {
        "id": f"m_{weights_hash}",
        "path": str(path),
        "source": str(path),
        "effective_weights_hash": weights_hash,
        "common_base_id": common_base_id,
        "parent_ids": list(parent_ids),
        "route": route,
        "architecture_hash": architecture,
        "tokenizer_hash": _tokenizer_identity(path),
        "layout_hash": digest({k: v["shape"] for k, v in layout.items()}),
        "precision": sorted({v["dtype"] for v in layout.values()}),
        "file_signature": signature,
    }


def compatible(left, right):
    return all(
        left[key] == right[key] for key in ("architecture_hash", "tokenizer_hash", "layout_hash")
    )


def merge_reason(left, right):
    if left["effective_weights_hash"] == right["effective_weights_hash"]:
        return "identical_effective_weights"
    if not left.get("common_base_id") or left["common_base_id"] != right.get("common_base_id"):
        return "unproven_common_initialization"
    if not compatible(left, right):
        return "incompatible_architecture_tokenizer_or_layout"
    return None


def merge_half(left, right, *, cache_root: Path, project_root: Path):
    reason = merge_reason(left, right)
    if reason:
        raise ValueError(reason)
    merge_id = digest(sorted([left["effective_weights_hash"], right["effective_weights_hash"]]))
    output = cache_root / f"merge_{merge_id}"
    marker = output / "lineage_merge.json"
    if not marker.is_file():
        import torch
        from safetensors.torch import save_file

        output.mkdir(parents=True, exist_ok=True)
        a, b = TensorFiles(Path(left["path"])), TensorFiles(Path(right["path"]))
        weight_map, total_size = {}, 0
        # One tensor per shard bounds peak memory independently of model size.
        for index, key in enumerate(sorted(a.mapping), 1):
            x, y = a.tensor(key), b.tensor(key)
            if x.shape != y.shape:
                raise ValueError(f"tensor layout mismatch: {key}")
            if x.is_floating_point() and y.is_floating_point():
                merged = (x.float() * 0.5 + y.float() * 0.5).to(x.dtype)
                if not torch.isfinite(merged).all():
                    raise ValueError(f"non-finite merged tensor: {key}")
            elif not x.is_floating_point() and not y.is_floating_point() and torch.equal(x, y):
                merged = x
            else:
                raise ValueError(f"non-floating buffers differ: {key}")
            shard = f"model-{index:05d}-of-{len(a.mapping):05d}.safetensors"
            save_file({key: merged.contiguous()}, output / shard, metadata={"format": "pt"})
            weight_map[key] = shard
            total_size += merged.numel() * merged.element_size()
        copy_assets(Path(left["path"]), output)
        atomic_json(
            output / "model.safetensors.index.json",
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
        )
        atomic_json(marker, {"parent_ids": [left["id"], right["id"]], "parent_weights": [0.5, 0.5]})
    model = describe_model(
        output,
        project_root=project_root,
        cache_root=cache_root,
        common_base_id=left["common_base_id"],
        parent_ids=[left["id"], right["id"]],
        route="merge",
    )
    model["parent_weights"] = [0.5, 0.5]
    return model
