#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
# shellcheck source=scripts/lib/environment.sh
source "${PROJECT_ROOT}/scripts/lib/environment.sh"
coevox_load_env
coevox_python

CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/who_when.yaml}"
TEST_DATA="${TEST_DATA:-${PROJECT_ROOT}/data/whowhen/test_whowhen.jsonl}"
BASE_URL="${POLICY_BASE_URL:-http://127.0.0.1:8000/v1}"
CONCURRENCY="${INFERENCE_CONCURRENCY:-32}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/who_when/full_test}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_whowhen_full_test.sh HARNESS [HARNESS ...]

Example:
  scripts/run_whowhen_full_test.sh generated

The script reuses the model currently served at POLICY_BASE_URL, runs the
complete Who&When test split once per harness, scores it, and writes one
directory per harness.

Optional environment variables:
  POLICY_BASE_URL       OpenAI-compatible endpoint (default: http://127.0.0.1:8000/v1)
  POLICY_MODEL          Served model ID; auto-detected from /models when unset
  TEST_DATA             Test split (default: data/whowhen/test_whowhen.jsonl)
  INFERENCE_CONCURRENCY Evaluation concurrency (default: 32)
  MAX_TURNS             Harness turn limit (default: harness.max_turns in CONFIG)
  OUTPUT_ROOT           Result root (default: output/who_when/full_test)
USAGE
}

if [[ "$#" -eq 0 ]]; then
  usage >&2
  exit 2
fi
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi
if [[ ! -f "${CONFIG}" || ! -f "${TEST_DATA}" ]]; then
  echo "Missing Who&When config or test data: ${CONFIG} ${TEST_DATA}" >&2
  exit 2
fi
if ! [[ "${CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
  echo "INFERENCE_CONCURRENCY must be a positive integer, got: ${CONCURRENCY}" >&2
  exit 2
fi

MODEL="$("${PYTHON}" - "${BASE_URL}" "${POLICY_MODEL:-}" <<'PY'
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

MAX_TURNS="${MAX_TURNS:-$("${PYTHON}" -c '
import sys
from pathlib import Path
from src.config import load_config
print(load_config(Path(sys.argv[1]))["harness"]["max_turns"])
' "${CONFIG}")}"

printf '%s\n' \
  "Who&When full-test evaluation" \
  "model=${MODEL}" \
  "endpoint=${BASE_URL}" \
  "test_data=${TEST_DATA}" \
  "concurrency=${CONCURRENCY}" \
  "max_turns=${MAX_TURNS}" \
  "output_root=${OUTPUT_ROOT}"

for harness_name in "$@"; do
  case "${harness_name}" in
    initial_harness|generated) harness_dir="${PROJECT_ROOT}/harnesses/who_when/${harness_name}" ;;
    *) harness_dir="${PROJECT_ROOT}/harnesses/who_when/candidates/${harness_name}" ;;
  esac
  if [[ ! -d "${harness_dir}" ]]; then
    echo "Who&When harness does not exist: ${harness_dir}" >&2
    exit 2
  fi

  output_dir="${OUTPUT_ROOT}/${harness_name}"
  predictions="${output_dir}/samples.jsonl"
  metrics="${output_dir}/metrics.json"
  mkdir -p "${output_dir}"

  echo "Starting full test: harness=${harness_name} output=${output_dir}"
  "${PYTHON}" -m src.domains.who_when.runtime.infer \
    --config "${CONFIG}" \
    --data "${TEST_DATA}" \
    --model "${MODEL}" \
    --base-url "${BASE_URL}" \
    --harness "${harness_name}" \
    --concurrency "${CONCURRENCY}" \
    --max-turns "${MAX_TURNS}" \
    --output "${predictions}" \
    --resume

  echo "Scoring full test: harness=${harness_name}"
  "${PYTHON}" -m src.domains.who_when.runtime.evaluation \
    --predictions "${predictions}" \
    --config "${CONFIG}" \
    --output "${metrics}"
done

echo "All requested Who&When full-test evaluations completed."
