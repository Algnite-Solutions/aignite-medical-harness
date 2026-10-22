#!/usr/bin/env bash
set -euo pipefail

# DATASET and RUN_ROOT point at controlled storage; only code and aggregates enter Git.
RUN_ROOT=${RUN_ROOT:-runs/pathvqa-pr100-v2}
PYTHON_BIN=${PYTHON_BIN:-python3}
COMMAND=${1:-help}
shift || true

case "$COMMAND" in
  import)
    RAW=${RAW:?Set RAW to the pinned PathVQA Parquet snapshot}
    OUT=${OUT:?Set OUT to a new AMA dataset directory}
    "$PYTHON_BIN" -m ama.cli import pathvqa --source "$RAW" --out "$OUT" "$@"
    ;;
  prepare|verify|run|eval|compare)
    DATASET=${DATASET:?Set DATASET to the imported PathVQA AMA dataset}
    "$PYTHON_BIN" -m scripts.pathvqa_v2 "$COMMAND" --dataset "$DATASET" --root "$RUN_ROOT" "$@"
    ;;
  smoke)
    DATASET=${DATASET:?Set DATASET to the imported PathVQA AMA dataset}
    "$PYTHON_BIN" -m scripts.pathvqa_v2 smoke --dataset "$DATASET" --root "$RUN_ROOT" "$@"
    ;;
  all)
    DATASET=${DATASET:?Set DATASET to the imported PathVQA AMA dataset}
    "$PYTHON_BIN" -m scripts.pathvqa_v2 verify --dataset "$DATASET" --root "$RUN_ROOT"
    "$PYTHON_BIN" -m scripts.pathvqa_v2 compare --dataset "$DATASET" --root "$RUN_ROOT" \
      --alias qwen36-thinking --alias qwen36-nothink --alias glm-5.3-flash "$@"
    ;;
  *)
    printf 'Usage: DATASET=/path/to/ama/pathvqa-task [RAW=... OUT=...] %s {import|prepare|verify|run|eval|compare|all} [options]\n' "$0" >&2
    exit 2
    ;;
esac
