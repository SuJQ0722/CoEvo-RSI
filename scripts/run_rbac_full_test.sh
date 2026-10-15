#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

RBAC_PYTHON="${RBAC_PYTHON:-${PROJECT_ROOT}/.venvs/rbac/bin/python}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/rbac.yaml}"
TEST_DATA="${TEST_DATA:-${PROJECT_ROOT}/data/rbac/test.jsonl}"
DB_DIR="${DB_DIR:-${PROJECT_ROOT}/data/spider/database}"
BASE_URL="${POLICY_BASE_URL:-http://127.0.0.1:8000/v1}"
CONCURRENCY="${INFERENCE_CONCURRENCY:-32}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/rbac/full_test}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_rbac_full_test.sh HARNESS [HARNESS ...]

Example:
  scripts/run_rbac_full_test.sh generated

The script reuses the model currently served at POLICY_BASE_URL, runs the
complete RBAC test split, scores it, and writes one directory per harness.

Optional environment variables:
  POLICY_BASE_URL       OpenAI-compatible endpoint (default: http://127.0.0.1:8000/v1)
  POLICY_MODEL          Served model ID; auto-detected from /models when unset
  INFERENCE_CONCURRENCY Evaluation concurrency (default: 32)
  OUTPUT_ROOT           Result root (default: output/rbac/full_test)
  DB_DIR                Spider database directory (default: data/spider/database)
EOF
}

if [[ "$#" -eq 0 ]]; then
  usage >&2
  exit 2
fi
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi
if [[ ! -x "${RBAC_PYTHON}" ]]; then
  echo "RBAC Python environment is missing: ${RBAC_PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" || ! -f "${TEST_DATA}" || ! -d "${DB_DIR}" ]]; then
  echo "Missing RBAC config, test data, or database directory" >&2
  exit 2
fi
if ! [[ "${CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
  echo "INFERENCE_CONCURRENCY must be a positive integer, got: ${CONCURRENCY}" >&2
  exit 2
fi

MODEL="${POLICY_MODEL:-}"
MODEL="$("${RBAC_PYTHON}" - "${BASE_URL}" "${MODEL}" <<'PY'
import json
import sys
from urllib.error import URLError
from urllib.request import urlopen

base_url, requested_model = sys.argv[1:]
try:
    with urlopen(base_url.rstrip("/") + "/models", timeout=10) as response:
        payload = json.load(response)
except (OSError, URLError, ValueError) as exc:
    raise SystemExit(f"Model endpoint is not ready at {base_url}: {exc}")

model_ids = [str(item.get("id")) for item in payload.get("data", []) if item.get("id")]
if requested_model:
    if requested_model not in model_ids:
        raise SystemExit(
            f"POLICY_MODEL={requested_model!r} is not served; available models: {model_ids}"
        )
    print(requested_model)
elif len(model_ids) == 1:
    print(model_ids[0])
else:
    raise SystemExit(
        f"Cannot select one served model from {model_ids}; set POLICY_MODEL explicitly"
    )
PY
)"

mkdir -p "${OUTPUT_ROOT}"
printf '%s\n' \
  "RBAC full-test evaluation" \
  "model=${MODEL}" \
  "endpoint=${BASE_URL}" \
  "test_data=${TEST_DATA}" \
  "concurrency=${CONCURRENCY}" \
  "output_root=${OUTPUT_ROOT}"

for harness_name in "$@"; do
  case "${harness_name}" in
    initial_harness|generated) harness_dir="${PROJECT_ROOT}/harnesses/rbac/${harness_name}" ;;
    *) harness_dir="${PROJECT_ROOT}/harnesses/rbac/candidates/${harness_name}" ;;
  esac
  if [[ ! -d "${harness_dir}" ]]; then
    echo "RBAC harness does not exist: ${harness_dir}" >&2
    exit 2
  fi

  output_dir="${OUTPUT_ROOT}/${harness_name}"
  predictions="${output_dir}/samples.jsonl"
  metrics="${output_dir}/metrics.json"
  mkdir -p "${output_dir}"

  echo "Starting full test: harness=${harness_name} output=${output_dir}"
  "${RBAC_PYTHON}" -m src.domains.rbac.runtime.infer \
    --config "${CONFIG}" \
    --data "${TEST_DATA}" \
    --db-dir "${DB_DIR}" \
    --model "${MODEL}" \
    --base-url "${BASE_URL}" \
    --harness "${harness_name}" \
    --concurrency "${CONCURRENCY}" \
    --output "${predictions}" \
    --resume

  echo "Scoring full test: harness=${harness_name}"
  "${RBAC_PYTHON}" -m src.domains.rbac.runtime.evaluation \
    --predictions "${predictions}" \
    --output "${metrics}" \
    --harness "${harness_name}"

  "${RBAC_PYTHON}" - "${metrics}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
metrics = json.loads(path.read_text(encoding="utf-8"))
print(
    "RESULT "
    f"harness={metrics['harness']} "
    f"samples={metrics['total']} "
    f"SafeEX={metrics['safe_ex']:.2%} "
    f"AC-F1={metrics['ac_f1']:.2%} "
    f"SafeDeny={metrics['safe_deny']:.2%} "
    f"violation={metrics['violation_rate']:.2%} "
    f"over_refusal={metrics['over_refusal_rate']:.2%} "
    f"LLM={metrics['llm_rate']:.2%} "
    f"rule={metrics['rule_rate']:.2%} "
    f"rule_accuracy={metrics['rule_accuracy']:.2%} "
    f"protocol_invalid={metrics.get('protocol_invalid', 0)} "
    f"metrics={path}"
)
PY
done

echo "All requested RBAC full-test evaluations completed."
