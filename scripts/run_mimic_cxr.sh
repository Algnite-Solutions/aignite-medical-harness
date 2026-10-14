#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
AMA_DATA_PATH=${AMA_DATA_PATH:-local_data}
BENCHMARK_DIR=${BENCHMARK_DIR:-$AMA_DATA_PATH/processed/MIMIC-CXR-ReXrank-fixed100-seed42}
RUNS_DIR=${RUNS_DIR:-runs/mimic-cxr-rexrank-fixed100-seed42}
CONFIG=${CONFIG:-ama.json}
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_mimic_cxr.sh COMMAND [ARGUMENT]

Commands:
  freeze           Freeze the official seed-42 cohort (requires REXRANK_JSON).
  import           Copy/check JPGs and publish the runnable dataset (requires IMAGE_ROOT).
  setup            Download/check pinned RadGraph resources; no model API call.
  validate         Validate the prepared dataset offline.
  run MODEL        Generate reports with one configured vision-model alias; makes API calls.
  eval RUN_DIR     Score one existing AMA run offline; no model API call.
  --help           Show this help.

Default layout:
  AMA_DATA_PATH: local_data (or your NAS data root)
  BENCHMARK_DIR: $AMA_DATA_PATH/processed/MIMIC-CXR-ReXrank-fixed100-seed42
  cohort:        $BENCHMARK_DIR/cohort/cohort.json
  dataset:       $BENCHMARK_DIR/dataset (self-contained; usable by ama run/eval)
  RUNS_DIR:      runs/mimic-cxr-rexrank-fixed100-seed42

Prepare from the pinned ReXrank JSON and the original directory containing files/:
  export REXRANK_JSON=/path/to/ReXRank_MIMICCXR_test.json
  export IMAGE_ROOT=/path/to/mimic-cxr-jpg-2.1.0
  bash scripts/run_mimic_cxr.sh freeze
  bash scripts/run_mimic_cxr.sh import
  bash scripts/run_mimic_cxr.sh setup
  bash scripts/run_mimic_cxr.sh validate

Run prepared data without reading the original JSON or NAS images:
  bash scripts/run_mimic_cxr.sh run YOUR_VISION_MODEL_ALIAS
  bash scripts/run_mimic_cxr.sh eval runs/mimic-cxr-rexrank-fixed100-seed42/RUN_ID

Override BENCHMARK_DIR, RUNS_DIR, PYTHON_BIN, CONFIG and TIMEOUT (default 300) as needed.
API keys must be configured separately; setup/eval never use a report-generator API.
freeze/import reject existing destinations; choose new directories for new preparation.
run uses the existing AMA runner, one model call per study, without batch resume/retries.
This is a 100-study image-only subset, not a full official ReXrank leaderboard result.
EOF
}

command=${1:---help}
if [[ $# -gt 0 ]]; then shift; fi
case "$command" in
  --help|-h|help) usage ;;
  freeze)
    : "${REXRANK_JSON:?Set REXRANK_JSON to the pinned official JSON}"
    "$PYTHON_BIN" scripts/prepare_mimic_cxr_rexrank.py freeze \
      --source "$REXRANK_JSON" --out "$BENCHMARK_DIR/cohort"
    ;;
  import)
    : "${REXRANK_JSON:?Set REXRANK_JSON to the pinned official JSON}"
    : "${IMAGE_ROOT:?Set IMAGE_ROOT to the directory containing files/ and split CSV}"
    "$PYTHON_BIN" scripts/prepare_mimic_cxr_rexrank.py materialize \
      --source "$REXRANK_JSON" --cohort "$BENCHMARK_DIR/cohort/cohort.json" \
      --image-root "$IMAGE_ROOT" --out "$BENCHMARK_DIR/dataset"
    ;;
  setup) "$PYTHON_BIN" scripts/setup_cxr_radgraph.py --cache "${AMA_RADGRAPH_CACHE:-local_data/radgraph}" ;;
  validate)
    "$PYTHON_BIN" -c 'import sys; from pathlib import Path; from ama.data import validate_dataset; errors=validate_dataset(Path(sys.argv[1])); print(errors or "Dataset valid"); sys.exit(bool(errors))' \
      "$BENCHMARK_DIR/dataset"
    ;;
  run)
    if [[ $# -ne 1 ]]; then echo "Provide one vision-model alias" >&2; exit 2; fi
    "$PYTHON_BIN" -m ama.cli run "$BENCHMARK_DIR/dataset" --model "$1" \
      --config "$CONFIG" --split test --runs-root "$RUNS_DIR" --max-calls 1 --timeout "${TIMEOUT:-300}"
    ;;
  eval)
    if [[ $# -ne 1 ]]; then echo "Provide one existing AMA run directory" >&2; exit 2; fi
    "$PYTHON_BIN" -m ama.cli eval "$1"
    ;;
  *) usage >&2; exit 2 ;;
esac
