#!/usr/bin/env bash
# Shared launcher setup. Caller exports override .env; empty values use defaults.
coevox_load_env() {
  local evo_key
  local -A evo_saved=()
  while IFS= read -r evo_key; do
    evo_saved["${evo_key}"]="${!evo_key}"
  done < <(compgen -e)
  if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.env"
    set +a
  fi
  for evo_key in "${!evo_saved[@]}"; do
    if [[ "${!evo_key-}" != "${evo_saved[${evo_key}]}" ]]; then
      export "${evo_key}=${evo_saved[${evo_key}]}"
    fi
  done
  # OmegaConf's environment interpolation distinguishes empty from absent.
  for evo_key in MODEL_PATH POLICY_MODEL VLLM_COMMAND POLICY_BASE_URL MAX_MODEL_LEN \
    GPU_MEMORY_UTILIZATION TENSOR_PARALLEL_SIZE DATA_PARALLEL_SIZE RBAC_ENVIRONMENT_PREFIX; do
    if [[ -z "${!evo_key:-}" ]]; then unset "${evo_key}"; fi
  done
}

coevox_python() {
  # Use only an explicit executable or the caller/container PATH.
  if [[ -z "${PYTHON:-}" ]]; then
    if command -v python >/dev/null 2>&1; then
      export PYTHON=python
    elif command -v python3 >/dev/null 2>&1; then
      export PYTHON=python3
    else
      echo "Python executable not found on PATH" >&2
      return 127
    fi
  fi
  command -v "${PYTHON}" >/dev/null || {
    echo "Python executable not found: ${PYTHON}" >&2
    return 127
  }
}

coevox_global_runtime() {
  local runtime_label="${1:-CoEvoX}"
  local clean_path=""
  local clean_library_path=""
  local active_virtual_env="${VIRTUAL_ENV:-}"
  local active_conda_prefix="${CONDA_PREFIX:-}"
  local path_entry library_entry resolved_entry runtime_variable runtime_value resolved_runtime
  local default_vllm override_vllm resolved_vllm resolved_override resolved_python
  local vllm_entrypoint vllm_shebang vllm_python
  local -a path_entries=() library_entries=()

  # GPU images provide the compatible PyTorch/vLLM stack globally.  A
  # project-local environment may contain a different CUDA build, so never
  # let activation metadata or a stale PATH entry select it for orchestration.
  unset VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONHOME
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

  IFS=':' read -r -a path_entries <<< "${PATH:-}"
  for path_entry in "${path_entries[@]}"; do
    resolved_entry="$(readlink -f "${path_entry}" 2>/dev/null || true)"
    if [[ "${path_entry}" == "${PROJECT_ROOT}"/.venvs/* || "${resolved_entry}" == "${PROJECT_ROOT}"/.venvs/* ]]; then
      continue
    fi
    if [[ -n "${active_virtual_env}" && (
      "${path_entry}" == "${active_virtual_env}/bin" ||
      "${resolved_entry}" == "$(readlink -f "${active_virtual_env}/bin" 2>/dev/null || true)"
    ) ]]; then
      continue
    fi
    if [[ -n "${active_conda_prefix}" && (
      "${path_entry}" == "${active_conda_prefix}/bin" ||
      "${resolved_entry}" == "$(readlink -f "${active_conda_prefix}/bin" 2>/dev/null || true)"
    ) ]]; then
      continue
    fi
    clean_path="${clean_path:+${clean_path}:}${path_entry}"
  done
  export PATH="${clean_path}"

  IFS=':' read -r -a library_entries <<< "${LD_LIBRARY_PATH:-}"
  for library_entry in "${library_entries[@]}"; do
    [[ -n "${library_entry}" ]] || continue
    resolved_entry="$(readlink -f "${library_entry}" 2>/dev/null || true)"
    if [[ "${library_entry}" == "${PROJECT_ROOT}"/.venvs/* || "${resolved_entry}" == "${PROJECT_ROOT}"/.venvs/* ]]; then
      continue
    fi
    if [[ -n "${active_virtual_env}" && (
      "${library_entry}" == "${active_virtual_env}"/* ||
      "${resolved_entry}" == "${active_virtual_env}"/*
    ) ]]; then
      continue
    fi
    if [[ -n "${active_conda_prefix}" && (
      "${library_entry}" == "${active_conda_prefix}"/* ||
      "${resolved_entry}" == "${active_conda_prefix}"/*
    ) ]]; then
      continue
    fi
    clean_library_path="${clean_library_path:+${clean_library_path}:}${library_entry}"
  done
  if [[ -n "${clean_library_path}" ]]; then
    export LD_LIBRARY_PATH="${clean_library_path}"
  else
    unset LD_LIBRARY_PATH
  fi

  for runtime_variable in CUDA_HOME CUDA_PATH; do
    runtime_value="${!runtime_variable:-}"
    if [[ "${runtime_value}" == "${PROJECT_ROOT}"/.venvs/* ||
      (-n "${active_virtual_env}" && "${runtime_value}" == "${active_virtual_env}"/*) ||
      (-n "${active_conda_prefix}" && "${runtime_value}" == "${active_conda_prefix}"/*)
    ]]; then
      unset "${runtime_variable}"
    fi
  done

  for runtime_variable in PYTHON VLLM_COMMAND; do
    runtime_value="${!runtime_variable:-}"
    [[ -n "${runtime_value}" ]] || continue
    resolved_runtime="$(command -v "${runtime_value}" 2>/dev/null || true)"
    resolved_runtime="$(readlink -f "${resolved_runtime}" 2>/dev/null || true)"
    if [[ "${runtime_value}" == "${PROJECT_ROOT}"/.venvs/* ||
      "${resolved_runtime}" == "${PROJECT_ROOT}"/.venvs/* ||
      (-n "${active_virtual_env}" && "${resolved_runtime}" == "${active_virtual_env}"/*) ||
      (-n "${active_conda_prefix}" && "${resolved_runtime}" == "${active_conda_prefix}"/*)
    ]]; then
      unset "${runtime_variable}"
    fi
  done

  # Keep frozen configs stable when the default executable is given by its
  # absolute path. LINEAGE intentionally compares configuration values exactly.
  if [[ -n "${VLLM_COMMAND:-}" ]]; then
    default_vllm="$(command -v vllm 2>/dev/null || true)"
    override_vllm="$(command -v "${VLLM_COMMAND}" 2>/dev/null || true)"
    resolved_vllm="$(readlink -f "${default_vllm}" 2>/dev/null || true)"
    resolved_override="$(readlink -f "${override_vllm}" 2>/dev/null || true)"
    if [[ -n "${default_vllm}" && (
      "${override_vllm}" == "${default_vllm}" ||
      (-n "${resolved_vllm}" && "${resolved_override}" == "${resolved_vllm}")
    ) ]]; then
      unset VLLM_COMMAND
    fi
  fi

  # vLLM console scripts carry the interpreter of the container environment
  # that installed vLLM. Use that same interpreter for orchestration so a
  # caller's unrelated Python/Conda environment cannot mix CUDA runtimes.
  vllm_entrypoint="$(command -v "${VLLM_COMMAND:-vllm}" 2>/dev/null || true)"
  if [[ -n "${vllm_entrypoint}" && -f "${vllm_entrypoint}" ]]; then
    IFS= read -r vllm_shebang < "${vllm_entrypoint}" || true
    vllm_python="${vllm_shebang#\#!}"
    if [[ "${vllm_python}" == /* && -x "${vllm_python}" ]]; then
      export PYTHON="${vllm_python}"
    fi
  fi

  coevox_python
  resolved_python="$(readlink -f "$(command -v "${PYTHON}")")"
  if [[ "${resolved_python}" == "${PROJECT_ROOT}"/.venvs/* ]]; then
    echo "${runtime_label} cannot use a project .venvs Python: ${resolved_python}" >&2
    return 2
  fi
  echo "${runtime_label} runtime: python=${resolved_python} vllm=$(command -v "${VLLM_COMMAND:-vllm}" 2>/dev/null || echo missing)"
}
