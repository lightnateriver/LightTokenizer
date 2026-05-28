#!/usr/bin/env python3
"""LoPT v2 stage1 tokenizer entrypoint."""

from benchmarks.lopt_v2 import (
    LoPTV2Config,
    LoPTV2ParallelTokenizer,
    LoPTV2Result,
    explain_token_mismatch,
    recommended_process_count,
    token_ids_sha256,
)

LoPTConfig = LoPTV2Config
LoPTParallelTokenizer = LoPTV2ParallelTokenizer
LoPTResult = LoPTV2Result

__all__ = [
    "LoPTConfig",
    "LoPTParallelTokenizer",
    "LoPTResult",
    "LoPTV2Config",
    "LoPTV2ParallelTokenizer",
    "LoPTV2Result",
    "explain_token_mismatch",
    "recommended_process_count",
    "token_ids_sha256",
]
