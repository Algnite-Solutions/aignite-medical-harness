#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
AMA_DATA_PATH=${AMA_DATA_PATH:-local_data}
DATASET_DIR=${DATASET_DIR:-$AMA_DATA_PATH/processed/MIMIC-CXR-AutoMedBench-findings100-v1/dataset}
CONFIG=${CONFIG:-ama.json}
RUNS_DIR=${RUNS_DIR:-local_data/automedbench/runs/mimic-cxr-ama}

usage() {
  cat <<'EOF'
Usage: bash scripts/run_mimic_cxr.sh COMMAND [ARGS...]

  import SOURCE [OPTIONS...]  Import a standalone raw package as findings-only.
  reuse SOURCE [OPTIONS...]   Reuse a frozen AMA CXR dataset as findings-only.
  run MODEL [OPTIONS...]     Call native ama run; extra options pass through.
  eval RUN_DIR [OPTIONS...]  Call native ama eval; e.g. --scorer mimic_cxr_automedbench_v1.

AMA_DATA_PATH defaults to local_data; the runnable dataset is stored under
  $AMA_DATA_PATH/processed/MIMIC-CXR-AutoMedBench-findings100-v1/dataset
Set DATASET_DIR, CONFIG, RUNS_DIR or PYTHON_BIN to override paths.
The dataset directory contains dataset.json, episodes.jsonl, targets.jsonl,
instructions.txt, eval.json and copied artifacts. Run records stay separate.
import/reuse default to SPLIT=test; set SPLIT=validate for validation data.
run defaults to TIMEOUT=180 and MAX_CALLS=1. Import never overwrites data.
Default findings-only scoring uses the pinned AutoMedBench lightweight backend.
No Docker or clinical model download is required.

Example (verify the restricted snapshot as described in the data card first):
  DATASET_DIR=/path/to/processed/mimic-cxr-jpg-2.1.0/findings100-v1/dataset \
    bash scripts/run_mimic_cxr.sh run glm-5.3-flash-coding --split test
  bash scripts/run_mimic_cxr.sh eval local_data/automedbench/runs/mimic-cxr-ama/<run-id>
EOF
}

command=${1:---help}
if [[ $# -gt 0 ]]; then shift; fi
case "$command" in
  help|--help|-h) usage ;;
  import|reuse)
    if [[ $# -lt 1 ]]; then usage >&2; exit 2; fi
    source_dir=$1; shift
    source_format=raw
    if [[ $command == reuse ]]; then source_format=ama; fi
    exec "$PYTHON_BIN" -m ama.cli import mimic-cxr --source "$source_dir" \
      --out "$DATASET_DIR" --split "${SPLIT:-test}" --source-format "$source_format" \
      --report-mode findings-only "$@"
    ;;
  run)
    if [[ $# -lt 1 ]]; then usage >&2; exit 2; fi
    model=$1; shift
    exec "$PYTHON_BIN" -m ama.cli run "$DATASET_DIR" --model "$model" --config "$CONFIG" \
      --runs-root "$RUNS_DIR" --max-calls "${MAX_CALLS:-1}" --timeout "${TIMEOUT:-180}" "$@"
    ;;
  eval)
    if [[ $# -lt 1 ]]; then usage >&2; exit 2; fi
    exec "$PYTHON_BIN" -m ama.cli eval "$@"
    ;;
  *) usage >&2; exit 2 ;;
esac
