#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN=${PYTHON_BIN:-python3.12}
BENCHMARK_DIR=${BENCHMARK_DIR:-runs/mimic-cdm-open-v1-100-seed42}
RUNS_DIR=${RUNS_DIR:-runs/mimic-cdm-open-v1-100-live}
TOOL_ADAPTER=src/ama/importers/mimic_cdm_tools.py
VIEWS=(hpi interactive full_info)

usage() {
  cat <<'EOF'
Usage: bash scripts/run_mimic_ext.sh {import|sample|run MODEL|eval MODEL|compare MODEL_A MODEL_B}

Set AMA_DATA_PATH to the directory containing raw/ and processed/ for import or sample.
Set BENCHMARK_DIR, RUNS_DIR, or PYTHON_BIN to override the output paths or Python.
For a fresh repeat, use new BENCHMARK_DIR and RUNS_DIR paths; sample will not overwrite a cohort.

Reproduce the seed-42, 100-case experiment (25 cases per source group):
  AMA_DATA_PATH=/path/to/data bash scripts/run_mimic_ext.sh sample
  bash scripts/run_mimic_ext.sh run antangelmed2
  bash scripts/run_mimic_ext.sh run deepseek-v4-flash
  bash scripts/run_mimic_ext.sh run glm-5.3-flash
  bash scripts/run_mimic_ext.sh compare antangelmed2 deepseek-v4-flash
  bash scripts/run_mimic_ext.sh compare antangelmed2 glm-5.3-flash
  bash scripts/run_mimic_ext.sh compare deepseek-v4-flash glm-5.3-flash

The run command evaluates each merged view. Repeating it resumes failed cases.
Run the models sequentially if they share an endpoint or quota.
EOF
}

processed_dir() {
  : "${AMA_DATA_PATH:?Set AMA_DATA_PATH to the directory containing raw/ and processed/}"
  printf '%s\n' "$AMA_DATA_PATH/processed/mimic-iv-ext-clinical-decision-making"
}

case ${1:-} in
  import)
    : "${AMA_DATA_PATH:?Set AMA_DATA_PATH to the directory containing raw/ and processed/}"
    PYTHONPATH=src "$PYTHON_BIN" -m ama.importers.mimic_cdm \
      --source "$AMA_DATA_PATH/raw/mimic-iv-ext-clinical-decision-making" \
      --out "$(processed_dir)"
    ;;
  sample)
    source_dir=$(processed_dir)
    PYTHONPATH=src "$PYTHON_BIN" -m ama.importers.mimic_cdm_benchmark \
      --full-info "$source_dir/mimic_cdm_full_info" \
      --interactive "$source_dir/mimic_cdm_interactive" \
      --out "$BENCHMARK_DIR" --per-class 25 --seed 42
    ;;
  run)
    model=${2:?Provide a model key from ama.json}
    for view in "${VIEWS[@]}"; do
      args=(--dataset "$BENCHMARK_DIR/mimic_cdm_open_$view" \
        --model "$model" --out "$RUNS_DIR/$model/$view" \
        --timeout 120 --max-calls 24 --delay 3)
      if [[ $view == interactive ]]; then
        args+=(--tools "$TOOL_ADAPTER")
      fi
      PYTHONPATH=src "$PYTHON_BIN" -m scripts.mimic_cdm_batch "${args[@]}"
      PYTHONPATH=src "$PYTHON_BIN" -m ama.cli eval "$RUNS_DIR/$model/$view/merged"
      status=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
        "$RUNS_DIR/$model/$view/batch.json")
      if [[ $status != completed ]]; then
        echo "$model $view: $status; rerun this command to retry failed cases" >&2
        exit 1
      fi
    done
    ;;
  eval)
    model=${2:?Provide a model key from ama.json}
    for view in "${VIEWS[@]}"; do
      PYTHONPATH=src "$PYTHON_BIN" -m ama.cli eval "$RUNS_DIR/$model/$view/merged"
    done
    ;;
  compare)
    model_a=${2:?Provide the first model key}
    model_b=${3:?Provide the second model key}
    PYTHONPATH=src "$PYTHON_BIN" -m scripts.mimic_cdm_compare \
      --model "$model_a" "$RUNS_DIR/$model_a/hpi/merged" \
        "$RUNS_DIR/$model_a/interactive/merged" "$RUNS_DIR/$model_a/full_info/merged" \
      --model "$model_b" "$RUNS_DIR/$model_b/hpi/merged" \
        "$RUNS_DIR/$model_b/interactive/merged" "$RUNS_DIR/$model_b/full_info/merged" \
      --out "$RUNS_DIR/compare-$model_a-vs-$model_b.json"
    ;;
  *)
    usage
    exit 1
    ;;
esac
