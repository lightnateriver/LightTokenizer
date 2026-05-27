#!/usr/bin/env python3
"""Compare replay benchmark outputs against a reference replay result file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def case_key(record: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        record["tokenizer_family"],
        record["language"],
        record["length_label"],
        int(record["input_chars"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--max-time-delta-ms", type=float, default=200.0)
    parser.add_argument("--subset-only", action="store_true")
    args = parser.parse_args()

    reference = json.loads(args.reference_json.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate_json.read_text(encoding="utf-8"))
    ref_map = {case_key(row): row for row in reference}
    cand_map = {case_key(row): row for row in candidate}

    shared_keys = sorted(set(ref_map) & set(cand_map))
    missing_in_candidate = sorted(set(ref_map) - set(cand_map))
    extra_in_candidate = sorted(set(cand_map) - set(ref_map))

    mismatches = []
    for key in shared_keys:
        ref = ref_map[key]
        cand = cand_map[key]
        problems = []
        for field in (
            "exact_match",
            "native_output_tokens",
            "lopt_output_tokens",
            "native_token_hash",
            "lopt_token_hash",
        ):
            if ref.get(field) != cand.get(field):
                problems.append(
                    {
                        "field": field,
                        "reference": ref.get(field),
                        "candidate": cand.get(field),
                    }
                )
        time_deltas = {}
        for field in (
            "native_e2e_time_ms",
            "native_tokenizer_time_ms",
            "mp_dispatch_process_collect_time_ms",
            "chunk_dedup_time_ms",
            "lopt_e2e_time_ms",
        ):
            if ref.get(field) is None or cand.get(field) is None:
                continue
            delta = abs(float(ref[field]) - float(cand[field]))
            time_deltas[field] = delta
            if delta > args.max_time_delta_ms:
                problems.append(
                    {
                        "field": field,
                        "reference": ref.get(field),
                        "candidate": cand.get(field),
                        "delta_ms": round(delta, 3),
                    }
                )
        if problems:
            mismatches.append(
                {
                    "case": {
                        "tokenizer_family": key[0],
                        "language": key[1],
                        "length_label": key[2],
                        "input_chars": key[3],
                    },
                    "problems": problems,
                    "time_deltas_ms": time_deltas,
                }
            )

    payload = {
        "shared_case_count": len(shared_keys),
        "missing_in_candidate": [
            {
                "tokenizer_family": key[0],
                "language": key[1],
                "length_label": key[2],
                "input_chars": key[3],
            }
            for key in missing_in_candidate
        ],
        "extra_in_candidate": [
            {
                "tokenizer_family": key[0],
                "language": key[1],
                "length_label": key[2],
                "input_chars": key[3],
            }
            for key in extra_in_candidate
        ],
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    missing_for_exit = [] if args.subset_only else missing_in_candidate
    if missing_for_exit or extra_in_candidate or mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
