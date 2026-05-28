"""Chunk merge logic for LoPT v2 stage1."""

from __future__ import annotations

from .types import BoundaryMatch, ChunkResult


def summarize_invalid_matches(matches: list[BoundaryMatch]) -> str:
    if not matches:
        return "No invalid boundary matches."
    first = matches[0]
    return (
        "Invalid boundary match at chunks "
        f"{first.left_chunk_index}->{first.right_chunk_index}: "
        f"{first.status}; {first.message}"
    )


def validate_matches(
    results: list[ChunkResult],
    matches: list[BoundaryMatch],
    *,
    min_match_tokens: int,
) -> None:
    expected = max(0, len(results) - 1)
    if len(matches) != expected:
        raise RuntimeError(
            f"Expected {expected} boundary matches but got {len(matches)}."
        )
    invalid = [
        match
        for match in matches
        if (not match.is_valid) or match.length < min_match_tokens
    ]
    if invalid:
        raise RuntimeError(summarize_invalid_matches(invalid))


def merge_chunk_results(
    results: list[ChunkResult],
    matches: list[BoundaryMatch],
    *,
    min_match_tokens: int,
) -> list[int]:
    if len(results) == 1:
        return list(results[0].input_ids)

    validate_matches(results, matches, min_match_tokens=min_match_tokens)

    merged: list[int] = []
    first_tokens = results[0].input_ids
    first_match = matches[0]
    merged.extend(first_tokens[: first_match.left_start_index + first_match.length])

    for index in range(1, len(results) - 1):
        tokens = results[index].input_ids
        previous = matches[index - 1]
        current = matches[index]
        start = previous.right_start_index + previous.length
        end = current.left_start_index + current.length
        merged.extend(tokens[start:end])

    last_tokens = results[-1].input_ids
    last_match = matches[-1]
    merged.extend(last_tokens[last_match.right_start_index + last_match.length :])
    return merged
