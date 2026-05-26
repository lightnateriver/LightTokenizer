#!/bin/bash
# 一键启动穷举搜索
set -e
TOKENIZER=${1:-"/mnt/sfs_turbo/models/qwen3.5-35b-a3b"}
LANG=${2:-"zh"}
cd "$(dirname "$0")/.."
python src/lopt_sweep.py --lang "$LANG" --tokenizer "$TOKENIZER" --md "results/lopt_${LANG}_results.md"
