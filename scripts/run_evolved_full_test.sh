#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# Use the same CUDA/vLLM runtime isolation as the domain evolution launchers.
# shellcheck source=scripts/lib/environment.sh
source "${PROJECT_ROOT}/scripts/lib/environment.sh"
coevox_load_env
coevox_python
FALLBACK_PYTHON="$(command -v "${PYTHON}")"
# Let the sanitizer reselect Python after it removes active virtual/Conda PATH
# entries; retaining a relative `python` name here can become unresolvable.
unset PYTHON
coevox_global_runtime "Full test"

# A CPU/login container may not have vLLM installed globally. In that case the
# shared runtime sanitizer can land on a bare system Python; keep dry-run/help
# and dependency diagnostics usable through the caller's working interpreter.
if ! "${PYTHON}" -c 'import omegaconf' >/dev/null 2>&1; then
  PYTHON="${FALLBACK_PYTHON}"
fi

exec "${PYTHON}" -m src.runner.full_test "$@"
