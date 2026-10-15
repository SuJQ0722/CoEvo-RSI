#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-python3}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/homebench.yaml}"
TEST_DATA="${TEST_DATA:-${PROJECT_ROOT}/data/homebench/test.jsonl}"
BASE_URL="${POLICY_BASE_URL:-http://127.0.0.1:8000/v1}"
CONCURRENCY="${INFERENCE_CONCURRENCY:-16}"
MAX_TURNS="${MAX_TURNS:-20}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/homebench/full_test}"

usage() {
  cat <<'EOF'
Usage: scripts/run_homebench_full_test.sh HARNESS [HARNESS ...]

Reuses the model served at POLICY_BASE_URL and evaluates each HomeBench harness
on the complete test split. Set POLICY_MODEL when multiple models are served.
EOF
}

if [[ "$#" -eq 0 ]]; then usage >&2; exit 2; fi
if [[ "$1" == "-h" || "$1" == "--help" ]]; then usage; exit 0; fi
if [[ ! -f "${CONFIG}" || ! -f "${TEST_DATA}" ]]; then
  echo "Missing HomeBench config or test data" >&2; exit 2
fi
if ! [[ "${CONCURRENCY}" =~ ^[1-9][0-9]*$ && "${MAX_TURNS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "INFERENCE_CONCURRENCY and MAX_TURNS must be positive integers" >&2; exit 2
fi

MODEL="$("${PYTHON}" - "${BASE_URL}" "${POLICY_MODEL:-}" <<'PY'
import json, sys
from urllib.request import urlopen
base_url, requested = sys.argv[1:]
try:
    with urlopen(base_url.rstrip("/") + "/models", timeout=10) as response:
        ids = [str(x["id"]) for x in json.load(response).get("data", []) if x.get("id")]
except (OSError, ValueError) as exc:
    raise SystemExit(f"Model endpoint is not ready at {base_url}: {exc}")
if requested:
    if requested not in ids:
        raise SystemExit(f"POLICY_MODEL={requested!r} is not served; available models: {ids}")
    print(requested)
elif len(ids) == 1:
    print(ids[0])
else:
    raise SystemExit(f"Cannot select one served model from {ids}; set POLICY_MODEL explicitly")
PY
)"

mkdir -p "${OUTPUT_ROOT}"
for harness_name in "$@"; do
  case "${harness_name}" in
    initial_harness|generated) harness_dir="${PROJECT_ROOT}/harnesses/homebench/${harness_name}" ;;
    *) harness_dir="${PROJECT_ROOT}/harnesses/homebench/candidates/${harness_name}" ;;
  esac
  if [[ ! -d "${harness_dir}" ]]; then
    echo "HomeBench harness does not exist: ${harness_name}" >&2; exit 2
  fi
  output_dir="${OUTPUT_ROOT}/${harness_name}"
  mkdir -p "${output_dir}"
  "${PYTHON}" -m src.domains.homebench.runtime.infer \
    --config "${CONFIG}" --data "${TEST_DATA}" --output-dir "${output_dir}" \
    --model "${MODEL}" --base-url "${BASE_URL}" --harness "${harness_name}" \
    --max-turns "${MAX_TURNS}" --concurrency "${CONCURRENCY}" --resume
  "${PYTHON}" - "${harness_name}" "${output_dir}/metrics.json" <<'PY'
import json, sys
from pathlib import Path
harness, path = sys.argv[1], Path(sys.argv[2])
m = json.loads(path.read_text())
print(f"RESULT harness={harness} samples={m['sample_count']} EM={m['em']:.2%} F1={m['f1']:.2%} completed={m['completed_count']} infrastructure_errors={m['infrastructure_errors']} metrics={path}")
PY
done
