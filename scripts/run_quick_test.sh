#!/bin/bash
# 快速验证 LoPT
set -e
TOKENIZER=${1:-"/mnt/sfs_turbo/models/qwen3.5-35b-a3b"}
cd "$(dirname "$0")/.."
python3 -c "
from src.lopt_core import run_lopt
from src.tokenizer_utils import load_tokenizer
from src.corpora import ZH_BLOCKS

tok = load_tokenizer('$TOKENIZER')
text = ''.join(ZH_BLOCKS) * 2000
messages = [{'role': 'user', 'content': text}]
ser_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
prompt_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print(f'Text: {len(prompt_str)} chars, {len(ser_ids)} tokens')
r = run_lopt(prompt_str, ser_ids, Lc=65536, LO=16, n_threads=4)
print(f'LoPT: {r["total_ms"]:.1f}ms (pool={r["pool_ms"]:.1f} + merge={r["merge_ms"]:.1f})')
print(f'Correct: {r["correct"]}, Fallback: {r["has_fallback"]}')
"
