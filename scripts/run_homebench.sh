#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
# Use the global/container CUDA runtime; project .venvs can contain an
# incompatible PyTorch/vLLM build.
# shellcheck source=scripts/lib/environment.sh
source "${PROJECT_ROOT}/scripts/lib/environment.sh"
for arg in "$@"; do
  if [[ "$arg" == --help || "$arg" == -h ]]; then exec bash "${PROJECT_ROOT}/scripts/run.sh" --help; fi
done
coevox_load_env
coevox_global_runtime "HomeBench"

export DOMAIN=homebench
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/qwen35_9b}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export TRAINING_MAX_LENGTH="${TRAINING_MAX_LENGTH:-65536}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
export MAX_TURNS="${MAX_TURNS:-20}"
export SAMPLING_CONCURRENCY="${SAMPLING_CONCURRENCY:-32}"
export INFERENCE_CONCURRENCY="${INFERENCE_CONCURRENCY:-32}"

exec bash "${PROJECT_ROOT}/scripts/run.sh" --domain homebench "$@"
