#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
# Use the same global/container Python and vLLM runtime selection as
# HomeBench. The RBAC-only environment is restricted to evaluation subprocesses.
# shellcheck source=scripts/lib/environment.sh
source "${PROJECT_ROOT}/scripts/lib/environment.sh"
for arg in "$@"; do
  if [[ "$arg" == --help || "$arg" == -h ]]; then exec bash "${PROJECT_ROOT}/scripts/run.sh" --help; fi
done
coevox_load_env
# This only selects the orchestrator/vLLM runtime. RBAC task execution keeps
# using execution.environment_prefix (.venvs/rbac) in its isolated subprocess.
coevox_global_runtime "RBAC"

export DOMAIN=rbac
export ROUNDS="${ROUNDS:-${ROUND:-10}}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export TRAINING_MAX_LENGTH="${TRAINING_MAX_LENGTH:-8192}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export MAX_TURNS="${MAX_TURNS:-10}"
export SAMPLING_CONCURRENCY="${SAMPLING_CONCURRENCY:-32}"
export INFERENCE_CONCURRENCY="${INFERENCE_CONCURRENCY:-32}"

exec bash "${PROJECT_ROOT}/scripts/run.sh" --domain rbac "$@"
