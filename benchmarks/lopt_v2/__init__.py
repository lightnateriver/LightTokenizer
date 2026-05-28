"""LoPT v2 stage1 tokenizer package."""

from .config import LoPTV2Config, recommended_process_count
from .engine import LoPTV2ParallelTokenizer
from .types import (
    AbsoluteTokenSpan,
    BoundaryDiagnostics,
    BoundaryMatch,
    ChunkResult,
    ChunkTask,
    LoPTV2Result,
)
from .validator import explain_token_mismatch, token_ids_sha256

__all__ = [
    "AbsoluteTokenSpan",
    "BoundaryDiagnostics",
    "BoundaryMatch",
    "ChunkResult",
    "ChunkTask",
    "LoPTV2Config",
    "LoPTV2ParallelTokenizer",
    "LoPTV2Result",
    "explain_token_mismatch",
    "recommended_process_count",
    "token_ids_sha256",
]
