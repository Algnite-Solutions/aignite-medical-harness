#!/usr/bin/env bash
# SRRSH ICC 临床表处理入口：一键导入 + 结构校验 + 计数对账。
# ⚠ 真实 xlsx 为 PHI：只在本地/NAS 处理；产物目录已被 .gitignore（datasets/srrsh_icc*）排除。
#
# 用法：
#   bash scripts/process_srrsh_icc.sh                    # 默认读 ../ICC临床数据.xlsx，写 datasets/srrsh_icc
#   ICC_XLSX=/path/to/ICC临床数据.xlsx ICC_OUT=/path/to/out bash scripts/process_srrsh_icc.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ICC_XLSX="${ICC_XLSX:-$REPO_ROOT/../ICC临床数据.xlsx}"
ICC_OUT="${ICC_OUT:-$REPO_ROOT/datasets/srrsh_icc}"

if [ ! -f "$ICC_XLSX" ]; then
  echo "source xlsx not found: $ICC_XLSX (set ICC_XLSX to override)" >&2
  exit 1
fi

cd "$REPO_ROOT"
PYTHONPATH=src python -m ama.cli import icc-xlsx --source "$ICC_XLSX" --out "$ICC_OUT"
ICC_OUT="$ICC_OUT" PYTHONPATH=src python -c "import os; from pathlib import Path; from ama.data import validate_dataset; errors = validate_dataset(Path(os.environ['ICC_OUT'])); print('OK' if not errors else errors); raise SystemExit(1 if errors else 0)"
