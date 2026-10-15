#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
ORIGINAL_ARGS=("$@")

# shellcheck source=scripts/lib/environment.sh
source "${PROJECT_ROOT}/scripts/lib/environment.sh"
usage() {
  cat <<'HELP'
Usage: bash scripts/run.sh [DOMAIN] [--domain DOMAIN | --config PATH] [options]
Default domain: who_when (alias: whowhen). Also supported: rbac, homebench.
  --dry-run          Preview the full-val LINEAGE plan without models, GPUs or APIs
  --resume [ROUND]   Resume the latest run in this config's output, or a named round
  --fresh            Compatibility no-op; starting a new round is the default
  --monitor          Supervise failures; frozen rounds do not permit code repairs
  --no-monitor       Disable supervision
  -h, --help         Show this help
Configuration: caller environment > .env > domain YAML. CLI domain/config overrides
DOMAIN/CONFIG in the environment. Set MODEL_PATH and CUDA_VISIBLE_DEVICES as needed.
ROUNDS=1 limits execution to one round. Other options go to run_one_round.py.
With --resume, ROUNDS is the absolute target round. Automatic selection uses run
timestamps, then round numbers; a completed or incomplete latest run reports an
error instead of silently falling back to an older run. Use --dry-run to preview.
HELP
}
for arg in "$@"; do
  if [[ "$arg" == --help || "$arg" == -h ]]; then usage; exit 0; fi
done
coevox_load_env
coevox_python
# All model/data artifacts used by the pipeline are pre-cached. Keep Hugging
# Face clients from probing the network during vLLM startup or SFT.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

PYTHON="${PYTHON:-python}"
DOMAIN="${DOMAIN:-who_when}"
CONFIG="${CONFIG:-}"
BACKEND="${BACKEND:-codex}"
MONITOR_FAILURES="${MONITOR_FAILURES:-0}"
MAX_CODE_REPAIRS="${MAX_CODE_REPAIRS:-0}"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen3.5-9B}"
MODEL_PATH="${MODEL_PATH:-}"
POLICY_MODEL="${POLICY_MODEL:-${BASE_MODEL_NAME}}"
ROUNDS="${ROUNDS:-}"
# NO_POLICY_UPDATE_PATIENCE remains a compatibility alias for older .env files.
POLICY_ROLLBACK_PATIENCE="${POLICY_ROLLBACK_PATIENCE:-${NO_POLICY_UPDATE_PATIENCE:-}}"
MAX_HARNESS_ATTEMPTS="${MAX_HARNESS_ATTEMPTS:-}"
ATTEMPTS_PER_ITEM="${ATTEMPTS_PER_ITEM:-}"
SAMPLING_CONCURRENCY="${SAMPLING_CONCURRENCY:-}"
INFERENCE_CONCURRENCY="${INFERENCE_CONCURRENCY:-}"
SAMPLING_QUALITY_TIER="${SAMPLING_QUALITY_TIER:-}"
MAX_TURNS="${MAX_TURNS:-}"
TRAIN_DATA="${TRAIN_DATA:-}"
REBUILD_SPLITS="${REBUILD_SPLITS:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-}"
TRAINING_NUM_PROCESSES="${TRAINING_NUM_PROCESSES:-}"
TRAINING_SEQUENCE_PARALLEL_SIZE="${TRAINING_SEQUENCE_PARALLEL_SIZE:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
TRAINING_MAX_LENGTH="${TRAINING_MAX_LENGTH:-}"
DRY_RUN=0
RESUME_ROUND=""
AUTO_RESUME=0
forwarded_args=()
cli_domain=""
cli_config=""

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --domain)
      if [[ "$#" -lt 2 || -z "$2" ]]; then
        echo "Usage: $0 [DOMAIN] [--domain DOMAIN] [pipeline options]" >&2
        exit 2
      fi
      DOMAIN="$2"
      cli_domain="$2"
      CONFIG="${cli_config}"
      shift 2
      ;;
    --config)
      if [[ "$#" -lt 2 || -z "$2" ]]; then
        echo "--config requires a path" >&2
        exit 2
      fi
      CONFIG="$2"
      cli_config="$2"
      shift 2
      ;;
    --resume)
      AUTO_RESUME=0
      RESUME_ROUND=""
      if [[ "$#" -ge 2 && "$2" != -* ]]; then
        RESUME_ROUND="$2"
        shift 2
      else
        AUTO_RESUME=1
        shift
      fi
      ;;
    --fresh)
      # Compatibility no-op. Only --resume ROUND enables artifact reuse.
      shift
      ;;
    --stage|--selection-harness)
      echo "run.sh executes complete evolution rounds; standalone stages are not supported" >&2
      exit 2
      ;;
    --continue-candidate|--accepted-rollouts|--reuse-parent-val-dir|--reuse-parent-test-dir)
      echo "$1 is retired from LINEAGE; resume with --resume ROUND" >&2
      exit 2
      ;;
    --dry-run)
      DRY_RUN=1
      forwarded_args+=("$1")
      shift
      ;;
    --monitor)
      MONITOR_FAILURES=1
      shift
      ;;
    --no-monitor)
      MONITOR_FAILURES=0
      shift
      ;;
    *)
      # A positional domain is accepted only as the first original argument.
      if [[ "$1" != -* && "$#" -eq "${#ORIGINAL_ARGS[@]}" ]]; then
        DOMAIN="$1"
        cli_domain="$1"
        CONFIG=""
      else
        forwarded_args+=("$1")
      fi
      shift
      ;;
  esac
done

if [[ -n "${DOMAIN}" ]]; then
  DOMAIN="$("${PYTHON}" -c 'import sys; from src.domains.registry import normalize_domain_name; print(normalize_domain_name(sys.argv[1]))' "${DOMAIN}")"
fi
CONFIG="${CONFIG:-configs/${DOMAIN}.yaml}"
if [[ ! -f "${CONFIG}" ]]; then
  echo "Domain config not found: ${CONFIG} (domain=${DOMAIN})" >&2
  exit 2
fi

command -v "${PYTHON}" >/dev/null || {
  echo "Python executable not found: ${PYTHON}" >&2
  exit 127
}

configured_domain="$("${PYTHON}" -c 'import sys; from pathlib import Path; from src.config import load_config; from src.domains.registry import get_domain_plugin; print(get_domain_plugin(load_config(Path(sys.argv[1]))).name)' "${CONFIG}")"
if [[ -n "${cli_domain}" && "${DOMAIN}" != "${configured_domain}" ]]; then
  echo "--domain ${DOMAIN} conflicts with ${CONFIG} (domain=${configured_domain})" >&2
  exit 2
fi
DOMAIN="${configured_domain}"

mapfile -t runtime_config < <(
  "${PYTHON}" -c \
    'import sys; from pathlib import Path; from src.config import load_config, resolve_output_root; c=load_config(Path(sys.argv[1])); d=c.get("evolution", {}); r=c.get("rejection_sampling", {}); o=c.get("rollout", {}); t=c.get("training", {}); v=c.get("vllm", {}); print(d.get("max_rounds", 10)); print(d.get("policy_rollback_patience", 3)); print((c.get("model") or {}).get("base", "")); print(c.get("data", {}).get("splits", {}).get("train", "")); print(r.get("attempts_per_item", 3)); print(r.get("concurrency", 16)); print(o.get("concurrency", 16)); print(c.get("harness", {}).get("max_turns", 10)); print(r.get("quality_tier", "strict")); print(v.get("tensor_parallel_size", 1)); print(v.get("data_parallel_size", "")); print(t.get("num_processes", 1)); print(t.get("sequence_parallel_size", 1)); print(resolve_output_root(c, Path.cwd())); print(c.get("meta_evolver", {}).get("max_attempts_per_round", 9)); print(v.get("gpu_memory_utilization", 0.8)); print(v.get("max_model_len", 32768)); print(t.get("max_length", 32768))' \
    "${CONFIG}"
)
if [[ "${#runtime_config[@]}" -ne 18 ]]; then
  echo "Could not read runtime settings from ${CONFIG}" >&2
  exit 2
fi
ROUNDS="${ROUNDS:-${runtime_config[0]}}"
POLICY_ROLLBACK_PATIENCE="${POLICY_ROLLBACK_PATIENCE:-${runtime_config[1]}}"
MODEL_PATH="${MODEL_PATH:-${runtime_config[2]}}"
TRAIN_DATA="${TRAIN_DATA:-${runtime_config[3]}}"
ATTEMPTS_PER_ITEM="${ATTEMPTS_PER_ITEM:-${runtime_config[4]}}"
SAMPLING_CONCURRENCY="${SAMPLING_CONCURRENCY:-${runtime_config[5]}}"
INFERENCE_CONCURRENCY="${INFERENCE_CONCURRENCY:-${runtime_config[6]}}"
MAX_TURNS="${MAX_TURNS:-${runtime_config[7]}}"
SAMPLING_QUALITY_TIER="${SAMPLING_QUALITY_TIER:-${runtime_config[8]}}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${runtime_config[9]}}"
CONFIGURED_DATA_PARALLEL_SIZE="${runtime_config[10]}"
CONFIGURED_TRAINING_NUM_PROCESSES="${runtime_config[11]}"
TRAINING_SEQUENCE_PARALLEL_SIZE="${TRAINING_SEQUENCE_PARALLEL_SIZE:-${runtime_config[12]}}"
OUTPUT_ROOT="${runtime_config[13]}"
if ((AUTO_RESUME)); then
  RESUME_ROUND="$("${PYTHON}" "${PROJECT_ROOT}/scripts/latest_resume_round.py" "${OUTPUT_ROOT}")"
  echo "Auto-resume: ${RESUME_ROUND} (output=${OUTPUT_ROOT})"
fi
MAX_HARNESS_ATTEMPTS="${MAX_HARNESS_ATTEMPTS:-${runtime_config[14]}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${runtime_config[15]}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${runtime_config[16]}}"
TRAINING_MAX_LENGTH="${TRAINING_MAX_LENGTH:-${runtime_config[17]}}"
if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is unset and model.base is missing from ${CONFIG}" >&2
  exit 2
fi

if [[ ! "${TENSOR_PARALLEL_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TENSOR_PARALLEL_SIZE must be a positive integer, got: ${TENSOR_PARALLEL_SIZE}" >&2
  exit 2
fi

visible_gpu_ids="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "${visible_gpu_ids}" || "${visible_gpu_ids,,}" == "all" ]]; then
  detected_gpu_ids=()
  if command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t detected_gpu_ids < <(
      nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true
    )
  fi
  valid_detected_gpu_ids=()
  for gpu_id in "${detected_gpu_ids[@]}"; do
    gpu_id="${gpu_id//[[:space:]]/}"
    if [[ "${gpu_id}" =~ ^[0-9]+$ ]]; then
      valid_detected_gpu_ids+=("${gpu_id}")
    fi
  done
  if ((${#valid_detected_gpu_ids[@]})); then
    visible_gpu_ids="$(IFS=,; echo "${valid_detected_gpu_ids[*]}")"
  elif ((DRY_RUN)); then
    dry_run_gpu_count="${TRAINING_NUM_PROCESSES:-${CONFIGURED_TRAINING_NUM_PROCESSES}}"
    visible_gpu_ids="0"
    for ((gpu_index = 1; gpu_index < dry_run_gpu_count; gpu_index++)); do
      visible_gpu_ids+=",${gpu_index}"
    done
    echo "dry-run: no GPUs detected; simulating CUDA=${visible_gpu_ids} from training.num_processes"
  else
    echo "Cannot discover visible GPUs: set CUDA_VISIBLE_DEVICES or provide a working nvidia-smi" >&2
    exit 2
  fi
fi
visible_gpu_ids="${visible_gpu_ids//[[:space:]]/}"
IFS=',' read -r -a visible_gpus <<< "${visible_gpu_ids}"
if [[ -z "${visible_gpu_ids}" || "${visible_gpu_ids}" == "-1" || "${visible_gpu_ids,,}" == "none" || "${#visible_gpus[@]}" -eq 0 ]]; then
  echo "No visible GPUs were found (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})" >&2
  exit 2
fi
VISIBLE_GPU_COUNT="${#visible_gpus[@]}"
if ((VISIBLE_GPU_COUNT % TENSOR_PARALLEL_SIZE != 0)); then
  echo "Visible GPU count ${VISIBLE_GPU_COUNT} is not divisible by TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}" >&2
  exit 2
fi
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-${CONFIGURED_DATA_PARALLEL_SIZE:-$((VISIBLE_GPU_COUNT / TENSOR_PARALLEL_SIZE))}}"
TRAINING_NUM_PROCESSES="${TRAINING_NUM_PROCESSES:-${CONFIGURED_TRAINING_NUM_PROCESSES:-${VISIBLE_GPU_COUNT}}}"

for value_name in ROUNDS POLICY_ROLLBACK_PATIENCE MAX_HARNESS_ATTEMPTS ATTEMPTS_PER_ITEM SAMPLING_CONCURRENCY INFERENCE_CONCURRENCY MAX_TURNS DATA_PARALLEL_SIZE TRAINING_NUM_PROCESSES TRAINING_SEQUENCE_PARALLEL_SIZE MAX_MODEL_LEN TRAINING_MAX_LENGTH; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name} must be a positive integer, got: ${value}" >&2
    exit 2
  fi
done
if ((DATA_PARALLEL_SIZE * TENSOR_PARALLEL_SIZE > VISIBLE_GPU_COUNT)); then
  echo "vLLM TP * DP exceeds visible GPU count ${VISIBLE_GPU_COUNT}" >&2
  exit 2
fi
if ((TRAINING_NUM_PROCESSES > VISIBLE_GPU_COUNT)); then
  echo "TRAINING_NUM_PROCESSES=${TRAINING_NUM_PROCESSES} exceeds visible GPU count ${VISIBLE_GPU_COUNT}" >&2
  exit 2
fi
if ((TRAINING_NUM_PROCESSES % TRAINING_SEQUENCE_PARALLEL_SIZE != 0)); then
  echo "TRAINING_NUM_PROCESSES=${TRAINING_NUM_PROCESSES} must be divisible by training.sequence_parallel_size=${TRAINING_SEQUENCE_PARALLEL_SIZE}" >&2
  exit 2
fi
if [[ "${SAMPLING_QUALITY_TIER}" != "strict" && "${SAMPLING_QUALITY_TIER}" != "balanced" ]]; then
  echo "SAMPLING_QUALITY_TIER must be strict or balanced, got: ${SAMPLING_QUALITY_TIER}" >&2
  exit 2
fi
if ((ROUNDS > runtime_config[0])); then
  echo "ROUNDS cannot exceed the configured evolution limit of ${runtime_config[0]}, got: ${ROUNDS}" >&2
  exit 2
fi
if [[ "${MONITOR_FAILURES}" != "0" && "${MONITOR_FAILURES}" != "1" ]]; then
  echo "MONITOR_FAILURES must be 0 or 1, got: ${MONITOR_FAILURES}" >&2
  exit 2
fi
if [[ ! "${MAX_CODE_REPAIRS}" =~ ^[0-9]+$ ]]; then
  echo "MAX_CODE_REPAIRS must be a non-negative integer, got: ${MAX_CODE_REPAIRS}" >&2
  exit 2
fi

export COEVOX_COLOR="${COEVOX_COLOR:-1}"
export CUDA_VISIBLE_DEVICES="${visible_gpu_ids}"

# Validate resume before installing backends or starting services.
start_round=1
if [[ -n "${RESUME_ROUND}" ]]; then
  if [[ "${RESUME_ROUND}" =~ ^round_([0-9]+)_[A-Za-z0-9_-]+$ ]]; then
    start_round=$((10#${BASH_REMATCH[1]}))
  else
    echo "Cannot determine round number from --resume ${RESUME_ROUND}" >&2
    exit 2
  fi
  if ((start_round < 1 || start_round > ROUNDS)); then
    echo "Resume round ${start_round} is outside the configured round limit ${ROUNDS}" >&2
    exit 2
  fi
  for artifact in snapshot.json journal.json; do
    if [[ ! -s "${OUTPUT_ROOT}/rounds/${RESUME_ROUND}/${artifact}" ]]; then
      echo "Missing frozen resume artifact: ${RESUME_ROUND}/${artifact}" >&2
      exit 2
    fi
  done
fi

LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/pipeline_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee "${LOG_FILE}") 2>&1

ensure_backend() {
  export PATH="${HOME}/.local/bin:${PATH}"
  if command -v "${BACKEND}" >/dev/null 2>&1; then
    return
  fi
  if [[ "${BACKEND}" != "codex" ]]; then
    echo "Backend executable not found: ${BACKEND}" >&2
    exit 127
  fi
  echo "codex not found, installing..."
  curl -fsSL https://chatgpt.com/codex/install.sh | sh
  hash -r
  if ! command -v codex >/dev/null 2>&1; then
    echo "Codex installation completed but 'codex' is still not on PATH=${PATH}" >&2
    exit 127
  fi
  echo "Codex installed: $(command -v codex)"
}

if ((!DRY_RUN)); then
  ensure_backend
fi

if ((MONITOR_FAILURES)) && ((MAX_CODE_REPAIRS > 0)); then
  echo "LINEAGE does not allow external code repair during a frozen round; use MAX_CODE_REPAIRS=0" >&2
  exit 2
fi

if ((MONITOR_FAILURES)) && ((!DRY_RUN)) && [[ "${COEVOX_MONITOR_ACTIVE:-0}" != "1" ]]; then
  export COEVOX_MONITOR_ACTIVE=1
  exec "${PYTHON}" scripts/monitor_evolution.py \
    --config "${CONFIG}" \
    --max-code-repairs "${MAX_CODE_REPAIRS}" \
    -- bash scripts/run.sh "${ORIGINAL_ARGS[@]}" --no-monitor
fi

if ((!DRY_RUN)) && [[ ! -f "${TRAIN_DATA}" && "${REBUILD_SPLITS}" != "1" ]]; then
  echo "Full training dataset not found: ${TRAIN_DATA}" >&2
  echo "Set REBUILD_SPLITS=1 to rebuild it or set TRAIN_DATA to an existing repository-relative JSONL file." >&2
  exit 2
fi

training_records="unknown"
if [[ -f "${TRAIN_DATA}" ]]; then
  training_records="$(wc -l < "${TRAIN_DATA}")"
  if [[ "${TRAIN_DATA,,}" == *.csv ]]; then
    # CSV inputs contain one header row; rejection sampling counts data rows.
    training_records="$((training_records - 1))"
  fi
fi

echo "Starting CoEvoX pipeline (domain=${DOMAIN}, config=${CONFIG}, training_data=${TRAIN_DATA}, training_records=${training_records}, CUDA=${visible_gpu_ids}, GPUs=${VISIBLE_GPU_COUNT}, vLLM_TP=${TENSOR_PARALLEL_SIZE}, vLLM_DP=${DATA_PARALLEL_SIZE}, training_processes=${TRAINING_NUM_PROCESSES}, training_SP=${TRAINING_SEQUENCE_PARALLEL_SIZE}, memory=${GPU_MEMORY_UTILIZATION}, inference_context=${MAX_MODEL_LEN}, inference_concurrency=${INFERENCE_CONCURRENCY}, training_context=${TRAINING_MAX_LENGTH}, max_turns=${MAX_TURNS}, sampling_attempts=${ATTEMPTS_PER_ITEM}, sampling_concurrency=${SAMPLING_CONCURRENCY}, sampling_quality=${SAMPLING_QUALITY_TIER}, log=${LOG_FILE})"
if [[ -n "${RESUME_ROUND}" ]]; then
  echo "Resume: reusing frozen LINEAGE state for round=${RESUME_ROUND} (identity checks follow resume_validation)"
fi

run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -n "${COEVOX_RUN_SUFFIX:-}" ]]; then
  if [[ ! "${COEVOX_RUN_SUFFIX}" =~ ^[A-Za-z0-9_]+$ ]]; then
    echo "COEVOX_RUN_SUFFIX must contain only letters, digits or underscores" >&2
    exit 2
  fi
  run_stamp+="_${COEVOX_RUN_SUFFIX}"
fi
current_policy_model="${POLICY_MODEL}"
current_model_path="${MODEL_PATH}"
policy_rollback_streak=0
completed_rounds=0
stop_reason="round_limit"

for ((round_number = start_round; round_number <= ROUNDS; round_number++)); do
  if [[ "${round_number}" == "${start_round}" && -n "${RESUME_ROUND}" ]]; then
    candidate_name="${RESUME_ROUND}"
  else
    candidate_name="round_$(printf '%02d' "${round_number}")_${run_stamp}"
  fi
  round_command=(
    "${PYTHON}" scripts/run_one_round.py
    --config "${CONFIG}"
    --backend "${BACKEND}"
    --round-number "${round_number}"
    --candidate-name "${candidate_name}"
    --model "${current_policy_model}"
    --base-model-name "${BASE_MODEL_NAME}"
    --model-path "${current_model_path}"
    --train-data "${TRAIN_DATA}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --data-parallel-size "${DATA_PARALLEL_SIZE}"
    --training-num-processes "${TRAINING_NUM_PROCESSES}"
    --training-sequence-parallel-size "${TRAINING_SEQUENCE_PARALLEL_SIZE}"
    --training-max-length "${TRAINING_MAX_LENGTH}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-turns "${MAX_TURNS}"
    --max-harness-attempts "${MAX_HARNESS_ATTEMPTS}"
    --attempts-per-item "${ATTEMPTS_PER_ITEM}"
    --sampling-concurrency "${SAMPLING_CONCURRENCY}"
    --inference-concurrency "${INFERENCE_CONCURRENCY}"
    --sampling-quality-tier "${SAMPLING_QUALITY_TIER}"
  )

  if [[ "${round_number}" == "${start_round}" && -n "${RESUME_ROUND}" ]]; then
    round_command+=(--resume-parent-artifacts)
  fi
  if [[ "${REBUILD_SPLITS}" == "1" && "${round_number}" == "1" ]]; then
    round_command+=(--rebuild-splits)
  fi
  if ((round_number < ROUNDS)); then
    round_command+=(--stop-after-round)
  fi
  round_command+=("${forwarded_args[@]}")

  echo "Starting CoEvoX round ${round_number}/${ROUNDS}: ${candidate_name} (CUDA=${visible_gpu_ids}, GPUs=${VISIBLE_GPU_COUNT}, vLLM_TP=${TENSOR_PARALLEL_SIZE}, vLLM_DP=${DATA_PARALLEL_SIZE}, training_processes=${TRAINING_NUM_PROCESSES}, training_SP=${TRAINING_SEQUENCE_PARALLEL_SIZE}, memory=${GPU_MEMORY_UTILIZATION})"
  if "${round_command[@]}"; then
    :
  else
    round_status=$?
    echo "Round failed: ${candidate_name}; rerun with --resume ${candidate_name} to continue its frozen state" >&2
    exit "${round_status}"
  fi

  if ((DRY_RUN)); then
    continue
  fi

  completed_rounds="${round_number}"

  summary_path="${OUTPUT_ROOT}/rounds/${candidate_name}/round_summary.json"
  if [[ ! -f "${summary_path}" ]]; then
    echo "Round summary was not created: ${summary_path}" >&2
    exit 1
  fi
  mapfile -t next_state < <(
    "${PYTHON}" -c \
      'import json, sys; s=json.load(open(sys.argv[1], encoding="utf-8")); d=s["deployment"]; print(d["model"]); print(d["model_path"]); print("1" if s.get("harness_updated") or s.get("policy_updated") else "0"); print(s.get("selection_decision", "unknown")); print(len(s.get("harness_attempts", []))); print(int(s.get("policy_rollbacks", s.get("parent_rollbacks", 0))))' \
      "${summary_path}"
  )
  current_policy_model="${next_state[0]}"
  current_model_path="${next_state[1]}"
  if [[ "${next_state[2]}" == "1" ]]; then
    policy_rollback_streak=0
  else
    policy_rollback_streak=$((policy_rollback_streak + next_state[5]))
    if ((policy_rollback_streak > POLICY_ROLLBACK_PATIENCE)); then
      policy_rollback_streak="${POLICY_ROLLBACK_PATIENCE}"
    fi
  fi
  if ((policy_rollback_streak >= POLICY_ROLLBACK_PATIENCE)); then
    stop_reason="policy_rollback_limit"
    echo "Pipeline stop reason=${stop_reason} round=${round_number}/${ROUNDS} selection_decision=${next_state[3]} meta_attempts=${next_state[4]} policy_rollbacks=${policy_rollback_streak} rollback_limit=${POLICY_ROLLBACK_PATIENCE}"
    break
  fi
done

if ((DRY_RUN)); then
  completed_rounds="${ROUNDS}"
fi
echo "Pipeline complete stop_reason=${stop_reason} completed_rounds=${completed_rounds}/${ROUNDS} policy=${current_policy_model} model_path=${current_model_path}"
