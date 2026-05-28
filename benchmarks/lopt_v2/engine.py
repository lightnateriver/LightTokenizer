"""LoPT v2 parallel tokenizer engine."""

from __future__ import annotations

import math
import time

from transformers import PreTrainedTokenizerFast

from .config import LoPTV2Config
from .matcher import build_boundary_match
from .merger import merge_chunk_results, summarize_invalid_matches
from .planner import build_chunk_plan
from .types import BoundaryMatch, LoPTV2Result
from .worker import TokenizerWorkerPool


class LoPTV2ParallelTokenizer:
    def __init__(self, config: LoPTV2Config) -> None:
        self.config = config
        self._pool = TokenizerWorkerPool(
            tokenizer_path=config.tokenizer_path,
            processes=config.processes,
        )
        self._special_token_tokenizer = PreTrainedTokenizerFast.from_pretrained(
            config.tokenizer_path
        )

    def close(self) -> None:
        self._pool.close()

    def __enter__(self) -> "LoPTV2ParallelTokenizer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _initial_chunk_chars(self, text: str) -> int:
        configured = self.config.initial_chunk_chars
        if configured is not None:
            return max(1, configured)
        return max(1, math.ceil(len(text) / max(self.config.processes, 1)))

    def _format_failed_boundaries(self, invalid: list[BoundaryMatch]) -> str:
        message = summarize_invalid_matches(invalid)
        if not invalid:
            return message
        first = invalid[0]
        if first.diagnostics is None:
            return message
        return (
            f"{message} overlap=({first.diagnostics.overlap_start}, "
            f"{first.diagnostics.overlap_end}) "
            f"left_overlap_tokens={first.diagnostics.left_overlap_tokens} "
            f"right_overlap_tokens={first.diagnostics.right_overlap_tokens} "
            f"matched_pairs={first.diagnostics.matched_pairs} "
            f"longest_run={first.diagnostics.longest_run}"
        )

    def tokenize(
        self,
        text: str,
        add_special_tokens: bool = True,
        *,
        chunk_count: int | None = None,
        overlap_chars: int | None = None,
        chat_template_time_s: float = 0.0,
    ) -> LoPTV2Result:
        if not text:
            raw_ids: list[int] = []
            final_ids = (
                self._special_token_tokenizer.build_inputs_with_special_tokens(raw_ids)
                if add_special_tokens
                else raw_ids
            )
            return LoPTV2Result(
                token_ids=list(final_ids),
                raw_token_ids=raw_ids,
                chat_template_time_s=chat_template_time_s,
                dispatch_submit_time_s=0.0,
                dispatch_collect_time_s=0.0,
                dispatch_process_collect_time_s=0.0,
                collect_child_compute_makespan_s=0.0,
                collect_result_return_tail_s=0.0,
                collect_result_receive_lag_max_s=0.0,
                collect_result_receive_lag_avg_s=0.0,
                chunk_dedup_time_s=0.0,
                worker_encode_time_s_sum=0.0,
                worker_encode_time_s_max=0.0,
                worker_materialize_time_s_sum=0.0,
                worker_materialize_time_s_max=0.0,
                process_only_time_s=0.0,
                e2e_time_s=0.0,
                merge_time_s=0.0,
                retry_rounds=0,
                chunk_chars=0,
                chunk_count=0,
                boundary_matches=(),
                boundary_diagnostics=(),
            )

        return self._tokenize_once(
            text,
            add_special_tokens=add_special_tokens,
            chunk_count=chunk_count,
            overlap_chars=overlap_chars,
            chat_template_time_s=chat_template_time_s,
        )

    def _tokenize_once(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        chunk_count: int | None,
        overlap_chars: int | None,
        chat_template_time_s: float,
    ) -> LoPTV2Result:
        active_overlap_chars = (
            self.config.overlap_chars if overlap_chars is None else overlap_chars
        )
        initial_chunk_chars = (
            None if chunk_count is not None else self._initial_chunk_chars(text)
        )
        tasks, resolved_chunk_chars = build_chunk_plan(
            text,
            chunk_count=chunk_count,
            chunk_chars=initial_chunk_chars,
            overlap_chars=active_overlap_chars,
        )
        results: list = []
        try:
            results, dispatch_submit_time_s, dispatch_collect_time_s = self._pool.dispatch(
                tasks
            )
            dispatch_process_collect_time_s = (
                dispatch_submit_time_s + dispatch_collect_time_s
            )
            worker_encode_time_s_sum = sum(
                result.worker_encode_time_s for result in results
            )
            worker_encode_time_s_max = max(
                (result.worker_encode_time_s for result in results),
                default=0.0,
            )
            worker_materialize_time_s_sum = sum(
                result.worker_materialize_time_s for result in results
            )
            worker_materialize_time_s_max = max(
                (result.worker_materialize_time_s for result in results),
                default=0.0,
            )
            first_worker_start = min(
                (
                    result.worker_started_at_s
                    for result in results
                    if result.worker_started_at_s
                ),
                default=0.0,
            )
            last_worker_finish = max(
                (
                    result.worker_finished_at_s
                    for result in results
                    if result.worker_finished_at_s
                ),
                default=0.0,
            )
            last_parent_receive = max(
                (
                    result.parent_received_at_s
                    for result in results
                    if result.parent_received_at_s
                ),
                default=0.0,
            )
            receive_lags = [
                result.parent_received_at_s - result.worker_finished_at_s
                for result in results
                if result.parent_received_at_s and result.worker_finished_at_s
            ]
            collect_child_compute_makespan_s = max(
                0.0,
                last_worker_finish - first_worker_start,
            ) if first_worker_start and last_worker_finish else 0.0
            collect_result_return_tail_s = max(
                0.0,
                last_parent_receive - last_worker_finish,
            ) if last_parent_receive and last_worker_finish else 0.0
            collect_result_receive_lag_max_s = max(receive_lags, default=0.0)
            collect_result_receive_lag_avg_s = (
                sum(receive_lags) / len(receive_lags) if receive_lags else 0.0
            )

            boundary_matches: list[BoundaryMatch] = []
            boundary_diagnostics = []
            if len(results) == 1:
                raw_ids = list(results[0].input_ids)
                chunk_dedup_time_s = 0.0
            else:
                postprocess_start = time.perf_counter()
                for left, right in zip(results, results[1:]):
                    match = build_boundary_match(
                        left,
                        right,
                        min_match_tokens=self.config.min_match_tokens,
                        record_diagnostics=self.config.record_boundary_diagnostics,
                    )
                    boundary_matches.append(match)
                    if match.diagnostics is not None:
                        boundary_diagnostics.append(match.diagnostics)
                invalid = [
                    match
                    for match in boundary_matches
                    if (not match.is_valid)
                    or match.length < self.config.min_match_tokens
                ]
                if invalid:
                    raise RuntimeError(self._format_failed_boundaries(invalid))
                raw_ids = merge_chunk_results(
                    results,
                    boundary_matches,
                    min_match_tokens=self.config.min_match_tokens,
                )
                chunk_dedup_time_s = time.perf_counter() - postprocess_start

            final_ids = (
                self._special_token_tokenizer.build_inputs_with_special_tokens(raw_ids)
                if add_special_tokens
                else raw_ids
            )
            e2e_time_s = (
                chat_template_time_s
                + dispatch_process_collect_time_s
                + chunk_dedup_time_s
            )
            return LoPTV2Result(
                token_ids=list(final_ids),
                raw_token_ids=raw_ids,
                chat_template_time_s=chat_template_time_s,
                dispatch_submit_time_s=dispatch_submit_time_s,
                dispatch_collect_time_s=dispatch_collect_time_s,
                dispatch_process_collect_time_s=dispatch_process_collect_time_s,
                collect_child_compute_makespan_s=collect_child_compute_makespan_s,
                collect_result_return_tail_s=collect_result_return_tail_s,
                collect_result_receive_lag_max_s=collect_result_receive_lag_max_s,
                collect_result_receive_lag_avg_s=collect_result_receive_lag_avg_s,
                chunk_dedup_time_s=chunk_dedup_time_s,
                worker_encode_time_s_sum=worker_encode_time_s_sum,
                worker_encode_time_s_max=worker_encode_time_s_max,
                worker_materialize_time_s_sum=worker_materialize_time_s_sum,
                worker_materialize_time_s_max=worker_materialize_time_s_max,
                process_only_time_s=dispatch_process_collect_time_s,
                e2e_time_s=e2e_time_s,
                merge_time_s=chunk_dedup_time_s,
                retry_rounds=0,
                chunk_chars=resolved_chunk_chars,
                chunk_count=len(tasks),
                boundary_matches=tuple(boundary_matches),
                boundary_diagnostics=tuple(boundary_diagnostics),
            )
        finally:
            if results:
                self._pool.cleanup_results(results)
