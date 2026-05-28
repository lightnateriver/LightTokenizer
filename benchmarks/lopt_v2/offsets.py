"""Absolute character-offset materialization for LoPT v2 stage1."""

from __future__ import annotations

from .types import AbsoluteTokenSpan, ChunkResult


def materialize_absolute_offsets(result: ChunkResult) -> ChunkResult:
    absolute_spans: list[AbsoluteTokenSpan] = []
    for token_index, ((local_start, local_end), token_id) in enumerate(
        zip(result.offsets, result.input_ids, strict=True)
    ):
        if local_end <= local_start:
            continue
        absolute_spans.append(
            AbsoluteTokenSpan(
                token_index=token_index,
                token_id=token_id,
                local_start=local_start,
                local_end=local_end,
                abs_start=result.start + local_start,
                abs_end=result.start + local_end,
            )
        )
    return ChunkResult(
        index=result.index,
        start=result.start,
        end=result.end,
        input_ids=list(result.input_ids),
        offsets=list(result.offsets),
        absolute_spans=absolute_spans,
    )
