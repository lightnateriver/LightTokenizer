"""Boundary matching for the LoPT v2 tokenizer."""

from __future__ import annotations

from .types import BoundaryDiagnostics, BoundaryMatch, BoundaryTokenEntry, ChunkResult

_TokenKey = tuple[int, int, int]
_OverlapEntry = BoundaryTokenEntry


def overlap_window(left: ChunkResult, right: ChunkResult) -> tuple[int, int]:
    return right.start, min(left.end, right.end)


def _entry_key(entry: _OverlapEntry) -> _TokenKey:
    return (entry[0], entry[1], entry[2])


def _sample_keys(
    entries: list[_OverlapEntry],
    *,
    limit: int = 4,
) -> tuple[_TokenKey, ...]:
    return tuple(_entry_key(entry) for entry in entries[:limit])


def collect_overlap_entries(
    result: ChunkResult,
    overlap_start: int,
    overlap_end: int,
) -> list[_OverlapEntry]:
    if overlap_start == result.start and result.left_boundary_entries:
        return [
            entry
            for entry in result.left_boundary_entries
            if entry[0] >= overlap_start and entry[1] <= overlap_end
        ]
    if result.right_boundary_entries:
        return [
            entry
            for entry in result.right_boundary_entries
            if entry[0] >= overlap_start and entry[1] <= overlap_end
        ]
    return []


def _build_invalid_match(
    *,
    left: ChunkResult,
    right: ChunkResult,
    overlap_start: int,
    overlap_end: int,
    status: str,
    message: str,
    left_entries: list[_OverlapEntry],
    right_entries: list[_OverlapEntry],
    matched_pairs: int,
    longest_run: int,
    sample_match_keys: tuple[_TokenKey, ...],
    record_diagnostics: bool,
) -> BoundaryMatch:
    diagnostics = None
    if record_diagnostics:
        diagnostics = BoundaryDiagnostics(
            left_chunk_index=left.index,
            right_chunk_index=right.index,
            overlap_start=overlap_start,
            overlap_end=overlap_end,
            left_overlap_tokens=len(left_entries),
            right_overlap_tokens=len(right_entries),
            matched_pairs=matched_pairs,
            longest_run=longest_run,
            status=status,
            message=message,
            sample_left_keys=_sample_keys(left_entries),
            sample_right_keys=_sample_keys(right_entries),
            sample_match_keys=sample_match_keys,
        )
    return BoundaryMatch(
        left_chunk_index=left.index,
        right_chunk_index=right.index,
        left_start_index=-1,
        right_start_index=-1,
        length=0,
        overlap_start=overlap_start,
        overlap_end=overlap_end,
        status=status,
        message=message,
        diagnostics=diagnostics,
    )


def build_boundary_match(
    left: ChunkResult,
    right: ChunkResult,
    *,
    min_match_tokens: int,
    record_diagnostics: bool = False,
) -> BoundaryMatch:
    overlap_start, overlap_end = overlap_window(left, right)
    left_entries = collect_overlap_entries(left, overlap_start, overlap_end)
    right_entries = collect_overlap_entries(right, overlap_start, overlap_end)

    if overlap_end <= overlap_start:
        return _build_invalid_match(
            left=left,
            right=right,
            overlap_start=overlap_start,
            overlap_end=overlap_end,
            status="empty_overlap_window",
            message="Adjacent chunks do not have a valid overlap window after planning.",
            left_entries=left_entries,
            right_entries=right_entries,
            matched_pairs=0,
            longest_run=0,
            sample_match_keys=(),
            record_diagnostics=record_diagnostics,
        )

    matched_pairs: list[tuple[int, int]] = []
    sample_match_keys: list[_TokenKey] = []
    i = 0
    j = 0
    while i < len(left_entries) and j < len(right_entries):
        left_key = _entry_key(left_entries[i])
        right_key = _entry_key(right_entries[j])
        if left_key == right_key:
            matched_pairs.append((left_entries[i][3], right_entries[j][3]))
            if len(sample_match_keys) < 4:
                sample_match_keys.append(left_key)
            i += 1
            j += 1
        elif left_key < right_key:
            i += 1
        else:
            j += 1

    if not matched_pairs:
        return _build_invalid_match(
            left=left,
            right=right,
            overlap_start=overlap_start,
            overlap_end=overlap_end,
            status="no_match",
            message="No exact token/span pairs were found inside the overlap window.",
            left_entries=left_entries,
            right_entries=right_entries,
            matched_pairs=0,
            longest_run=0,
            sample_match_keys=(),
            record_diagnostics=record_diagnostics,
        )

    best_start = 0
    best_len = 1
    run_start = 0
    run_len = 1
    for idx in range(1, len(matched_pairs)):
        prev_left, prev_right = matched_pairs[idx - 1]
        cur_left, cur_right = matched_pairs[idx]
        if cur_left == prev_left + 1 and cur_right == prev_right + 1:
            run_len += 1
        else:
            if run_len > best_len:
                best_start = run_start
                best_len = run_len
            run_start = idx
            run_len = 1

    if run_len > best_len:
        best_start = run_start
        best_len = run_len

    left_start_index, right_start_index = matched_pairs[best_start]
    status = "valid" if best_len >= min_match_tokens else "short_match"
    message = (
        "Boundary match is valid."
        if status == "valid"
        else (
            "Matched overlap is shorter than min_match_tokens: "
            f"{best_len} < {min_match_tokens}."
        )
    )
    diagnostics = None
    if record_diagnostics:
        diagnostics = BoundaryDiagnostics(
            left_chunk_index=left.index,
            right_chunk_index=right.index,
            overlap_start=overlap_start,
            overlap_end=overlap_end,
            left_overlap_tokens=len(left_entries),
            right_overlap_tokens=len(right_entries),
            matched_pairs=len(matched_pairs),
            longest_run=best_len,
            status=status,
            message=message,
            sample_left_keys=_sample_keys(left_entries),
            sample_right_keys=_sample_keys(right_entries),
            sample_match_keys=tuple(sample_match_keys),
        )
    return BoundaryMatch(
        left_chunk_index=left.index,
        right_chunk_index=right.index,
        left_start_index=left_start_index,
        right_start_index=right_start_index,
        length=best_len,
        overlap_start=overlap_start,
        overlap_end=overlap_end,
        status=status,
        message=message,
        diagnostics=diagnostics,
    )
