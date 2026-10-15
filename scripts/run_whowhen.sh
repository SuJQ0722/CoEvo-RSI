#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Same Python/environment selection as run_rbac.sh. Keep the Who&When context
# and training defaults explicit while allowing caller overrides.
# shellcheck source=scripts/lib/environment.sh
source "${PROJECT_ROOT}/scripts/lib/environment.sh"
for arg in "$@"; do
  if [[ "$arg" == --help || "$arg" == -h ]]; then exec bash "${PROJECT_ROOT}/scripts/run.sh" --help; fi
done
coevox_load_env
coevox_global_runtime "WhoWhen"

export DOMAIN=who_when
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/qwen35_9b}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export TRAINING_MAX_LENGTH="${TRAINING_MAX_LENGTH:-65536}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
export MAX_TURNS="${MAX_TURNS:-20}"
export SAMPLING_CONCURRENCY="${SAMPLING_CONCURRENCY:-32}"
export INFERENCE_CONCURRENCY="${INFERENCE_CONCURRENCY:-32}"

exec bash "${PROJECT_ROOT}/scripts/run.sh" --domain who_when "$@"
