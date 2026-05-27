#!/usr/bin/env python3
"""Collect benchmark host environment information for reproducible reports."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


def run_command(argv: list[str]) -> str:
    completed = subprocess.run(
        argv,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def find_value(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip()


def parse_kb_value(text: str, key: str) -> int | None:
    value = find_value(rf"^{re.escape(key)}:\s+(\d+)\s+kB$", text)
    return int(value) if value is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    hostname = run_command(["hostname"])
    lscpu_raw = run_command(["lscpu"])
    free_h_raw = run_command(["free", "-h"])
    meminfo_raw = run_command(
        [
            "bash",
            "-lc",
            "grep -E 'MemTotal|MemFree|MemAvailable|SwapTotal|SwapFree' /proc/meminfo",
        ]
    )
    try:
        numactl_raw = run_command(["numactl", "--hardware"])
    except Exception:
        numactl_raw = ""

    hardware_summary: dict[str, Any] = {
        "architecture": find_value(r"^Architecture:\s+(.+)$", lscpu_raw),
        "vendor_id": find_value(r"^Vendor ID:\s+(.+)$", lscpu_raw),
        "cpu_count": int(find_value(r"^CPU\(s\):\s+(\d+)$", lscpu_raw) or 0),
        "threads_per_core": int(
            find_value(r"^Thread\(s\) per core:\s+(\d+)$", lscpu_raw) or 0
        ),
        "clusters": int(find_value(r"^Cluster\(s\):\s+(\d+)$", lscpu_raw) or 0),
        "numa_nodes": int(
            find_value(r"^NUMA node\(s\):\s+(\d+)$", lscpu_raw) or 0
        ),
        "cpu_max_mhz": float(
            find_value(r"^CPU max MHz:\s+([0-9.]+)$", lscpu_raw) or 0.0
        ),
        "cpu_min_mhz": float(
            find_value(r"^CPU min MHz:\s+([0-9.]+)$", lscpu_raw) or 0.0
        ),
        "l1d_cache": find_value(r"^L1d cache:\s+(.+)$", lscpu_raw),
        "l2_cache": find_value(r"^L2 cache:\s+(.+)$", lscpu_raw),
        "l3_cache": find_value(r"^L3 cache:\s+(.+)$", lscpu_raw),
    }

    payload = {
        "hostname": hostname,
        "hardware_summary": hardware_summary,
        "memory_total_human": find_value(r"^Mem:\s+(\S+)", free_h_raw),
        "memory_available_human": find_value(
            r"^Mem:\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+(\S+)$",
            free_h_raw,
        ),
        "memory_total_kb": parse_kb_value(meminfo_raw, "MemTotal"),
        "memory_available_kb": parse_kb_value(meminfo_raw, "MemAvailable"),
        "lscpu_raw": lscpu_raw,
        "free_h_raw": free_h_raw,
        "meminfo_raw": meminfo_raw,
        "numactl_raw": numactl_raw,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_json": str(args.output_json)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
