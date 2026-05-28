"""Configuration for the LoPT v2 stage1 implementation."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LoPTV2Config:
    tokenizer_path: str
    processes: int
    overlap_chars: int = 1024
    min_match_tokens: int = 2
    initial_chunk_chars: int | None = None
    max_retry_rounds: int = 0
    record_boundary_diagnostics: bool = False
    stage_label: str = "v2-stage2"


def recommended_process_count() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(32, cpu_count))
