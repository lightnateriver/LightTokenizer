"""Shared datatypes for the LoPT v2 tokenizer implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

BoundaryTokenEntry = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class ChunkTask:
    index: int
    start: int
    core_end: int
    end: int
    overlap_chars: int
    text: str


@dataclass(frozen=True, slots=True)
class AbsoluteTokenSpan:
    token_index: int
    token_id: int
    local_start: int
    local_end: int
    abs_start: int
    abs_end: int


@dataclass(slots=True)
class ChunkResult:
    index: int
    start: int
    end: int
    input_ids: Sequence[int]
    offsets: list[tuple[int, int]] = field(default_factory=list)
    left_boundary_entries: list[BoundaryTokenEntry] = field(default_factory=list)
    right_boundary_entries: list[BoundaryTokenEntry] = field(default_factory=list)
    worker_pid: int = 0
    worker_started_at_s: float = 0.0
    worker_finished_at_s: float = 0.0
    parent_received_at_s: float = 0.0
    worker_encode_time_s: float = 0.0
    worker_materialize_time_s: float = 0.0
    absolute_spans: list[AbsoluteTokenSpan] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class DispatchMetrics:
    submit_time_s: float
    collect_time_s: float
    total_time_s: float
    child_compute_makespan_s: float
    result_return_tail_s: float
    result_receive_lag_max_s: float
    result_receive_lag_avg_s: float
    worker_encode_time_s_sum: float
    worker_encode_time_s_max: float
    worker_materialize_time_s_sum: float
    worker_materialize_time_s_max: float
    task_count: int


@dataclass(frozen=True, slots=True)
class BoundaryDiagnostics:
    left_chunk_index: int
    right_chunk_index: int
    overlap_start: int
    overlap_end: int
    left_overlap_tokens: int
    right_overlap_tokens: int
    matched_pairs: int
    longest_run: int
    status: str
    message: str
    sample_left_keys: tuple[tuple[int, int, int], ...] = ()
    sample_right_keys: tuple[tuple[int, int, int], ...] = ()
    sample_match_keys: tuple[tuple[int, int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class BoundaryMatch:
    left_chunk_index: int
    right_chunk_index: int
    left_start_index: int
    right_start_index: int
    length: int
    overlap_start: int
    overlap_end: int
    status: str = "valid"
    message: str = ""
    diagnostics: BoundaryDiagnostics | None = None

    @property
    def is_valid(self) -> bool:
        return (
            self.status == "valid"
            and self.left_start_index >= 0
            and self.right_start_index >= 0
            and self.length > 0
        )


@dataclass(frozen=True, slots=True)
class LoPTV2Result:
    token_ids: list[int]
    raw_token_ids: list[int]
    chat_template_time_s: float
    dispatch_submit_time_s: float
    dispatch_collect_time_s: float
    dispatch_process_collect_time_s: float
    collect_child_compute_makespan_s: float
    collect_result_return_tail_s: float
    collect_result_receive_lag_max_s: float
    collect_result_receive_lag_avg_s: float
    chunk_dedup_time_s: float
    worker_encode_time_s_sum: float
    worker_encode_time_s_max: float
    worker_materialize_time_s_sum: float
    worker_materialize_time_s_max: float
    process_only_time_s: float
    e2e_time_s: float
    merge_time_s: float
    retry_rounds: int
    chunk_chars: int
    chunk_count: int
    boundary_matches: tuple[BoundaryMatch, ...] = ()
    boundary_diagnostics: tuple[BoundaryDiagnostics, ...] = ()
