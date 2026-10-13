#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
CONFIG=${CONFIG:-ama.json}
RUNS_DIR=${RUNS_DIR:-runs/symptom2disease-fixed100-seed42}

usage() {
  cat <<'EOF'
Usage: bash scripts/run_symptom2disease.sh COMMAND [MODEL ...]

Commands:
  import             Import the complete CSV into processed/Symptom2Disease.
  sample             Prepare the seed-42, balanced 100-case cohort under processed/.
  run MODEL ...      Run prepared cases sequentially and score each model.
  eval MODEL ...     Re-score existing merged AMA runs without API calls.
  compare MODEL ...  Compare completed batches without API calls.
  all MODEL ...      Prepare the cohort, then run, score and compare in one command.
  --help             Show this help.

Set AMA_DATA_PATH to the directory containing raw/ and processed/:
  export AMA_DATA_PATH=/Volumes/10.214.105.104/data
  bash scripts/run_symptom2disease.sh import
  bash scripts/run_symptom2disease.sh all deepseek-v4-flash antangelmed2 qwen36

Default paths:
  SOURCE_CSV:   $AMA_DATA_PATH/raw/Symptom2Disease/Symptom2Disease.csv
  PROCESSED_DIR:$AMA_DATA_PATH/processed/Symptom2Disease (full dataset)
  BENCHMARK_DIR:$AMA_DATA_PATH/processed/Symptom2Disease-fixed100-seed42
  RUNS_DIR:     runs/symptom2disease-fixed100-seed42 (records and comparison.json)

The prepared runnable dataset is $BENCHMARK_DIR/dataset:
  ama run "$BENCHMARK_DIR/dataset" --model qwen36
  ama eval runs/<run-directory>

Override the named paths, PYTHON_BIN or CONFIG as needed. run and compare need
only BENCHMARK_DIR and RUNS_DIR; they do not read the raw CSV.
Model aliases come from CONFIG; set their api_key_env variables before run/all.
TIMEOUT=300, DELAY=1, MAX_ATTEMPTS=3 control requests and transport retries.
Repeating run/all resumes transport failures and keeps recorded answers frozen,
including invalid answers. An incomplete orphan shard stops for inspection.
import rejects an existing destination. sample reuses only an unchanged cohort.
For a fresh experiment, choose new BENCHMARK_DIR and RUNS_DIR paths.
EOF
}

source_csv() {
  if [[ -n ${SOURCE_CSV:-} ]]; then printf '%s\n' "$SOURCE_CSV"; else
    : "${AMA_DATA_PATH:?Set SOURCE_CSV or AMA_DATA_PATH}"
    printf '%s\n' "$AMA_DATA_PATH/raw/Symptom2Disease/Symptom2Disease.csv"
  fi
}

processed_dir() {
  if [[ -n ${PROCESSED_DIR:-} ]]; then printf '%s\n' "$PROCESSED_DIR"; else
    : "${AMA_DATA_PATH:?Set PROCESSED_DIR or AMA_DATA_PATH}"
    printf '%s\n' "$AMA_DATA_PATH/processed/Symptom2Disease"
  fi
}

benchmark_dir() {
  if [[ -n ${BENCHMARK_DIR:-} ]]; then printf '%s\n' "$BENCHMARK_DIR"; else
    : "${AMA_DATA_PATH:?Set BENCHMARK_DIR or AMA_DATA_PATH}"
    printf '%s\n' "$AMA_DATA_PATH/processed/Symptom2Disease-fixed100-seed42"
  fi
}

sample() {
  PYTHONPATH=src "$PYTHON_BIN" -m scripts.symptom2disease_benchmark \
    --source "$(source_csv)" --benchmark "$(benchmark_dir)" --prepare-only
}

run_models() {
  local args=(--benchmark "$(benchmark_dir)" --out "$RUNS_DIR" --config "$CONFIG" --run-only
    --timeout "${TIMEOUT:-300}" --delay "${DELAY:-1}" --max-attempts "${MAX_ATTEMPTS:-3}")
  local model
  for model in "$@"; do args+=(--model "$model"); done
  PYTHONPATH=src "$PYTHON_BIN" -m scripts.symptom2disease_benchmark "${args[@]}"
}

compare_models() {
  local args=(--benchmark "$(benchmark_dir)" --out "$RUNS_DIR" --compare-only)
  local model
  for model in "$@"; do args+=(--model "$model"); done
  PYTHONPATH=src "$PYTHON_BIN" -m scripts.symptom2disease_benchmark "${args[@]}"
}

command=${1:---help}
if [[ $# -gt 0 ]]; then shift; fi
case "$command" in
  --help|-h|help) usage ;;
  import)
    PYTHONPATH=src "$PYTHON_BIN" -m ama.cli import symptom2disease \
      --source "$(source_csv)" --out "$(processed_dir)"
    ;;
  sample) sample ;;
  run|all|eval|compare)
    if [[ $# -eq 0 ]]; then echo "Provide at least one model alias" >&2; exit 2; fi
    case "$command" in
      all) sample; run_models "$@"; compare_models "$@" ;;
      run) run_models "$@" ;;
      eval)
        for model in "$@"; do
          if [[ ! $model =~ ^[A-Za-z0-9._-]+$ || $model == . || $model == .. ]]; then
            echo "Invalid model alias: $model" >&2; exit 2
          fi
          PYTHONPATH=src "$PYTHON_BIN" -m ama.cli eval "$RUNS_DIR/$model/merged"
        done
        ;;
      compare)
        compare_models "$@"
        ;;
    esac
    ;;
  *) usage >&2; exit 2 ;;
esac
