#!/usr/bin/env python3
"""Generate a collapsible HTML benchmark report for native vLLM vs LoPT."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Iterable


FAMILY_ORDER = {
    "DeepSeek-V4-Pro": 0,
    "Qwen3.5": 1,
}
VERSION_ORDER = {
    "v1": 0,
    "v2": 1,
}
LANGUAGE_ORDER = {
    "zh": 0,
    "en": 1,
}
LANGUAGE_DISPLAY = {
    "zh": "中文",
    "en": "英文",
}
SERIES_COLORS = {
    ("DeepSeek-V4-Pro", "zh"): "#1f6feb",
    ("DeepSeek-V4-Pro", "en"): "#0f9f9a",
    ("Qwen3.5", "zh"): "#f59e0b",
    ("Qwen3.5", "en"): "#13835b",
}
BREAKDOWN_COLORS = {
    "chat": "#9ca3af",
    "mp": "#1f6feb",
    "dedup": "#0f9f9a",
    "native": "#f59e0b",
    "lopt": "#13835b",
}
STATUS_DISPLAY = {
    "valid": "有效",
    "fallback": "回退",
    "mismatch": "不一致",
    "error": "错误",
}


@dataclass(frozen=True)
class CaseKey:
    tokenizer_family: str
    language: str
    length_label: str
    input_chars: int


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def sort_case_key(key: CaseKey) -> tuple[int, int, int, str]:
    return (
        FAMILY_ORDER.get(key.tokenizer_family, 99),
        LANGUAGE_ORDER.get(key.language, 99),
        key.input_chars,
        key.length_label,
    )


def record_case_key(record: dict[str, Any]) -> CaseKey:
    return CaseKey(
        tokenizer_family=record["tokenizer_family"],
        language=record["language"],
        length_label=record["length_label"],
        input_chars=int(record["input_chars"]),
    )


def fmt_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def fmt_ms(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):,.3f}"


def fmt_x(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}x"


def fmt_pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2f}%"


def fmt_int(value: Any) -> str:
    if value is None:
        return "-"
    return f"{int(value):,}"


def language_label(language: str) -> str:
    return LANGUAGE_DISPLAY.get(language, language.upper())


def family_language_label(family: str, language: str) -> str:
    return f"{family} / {language_label(language)}"


def version_label(version: str) -> str:
    return {"v1": "v1", "v2": "v2.1"}.get(version, version)


def family_language_version_label(family: str, language: str, version: str) -> str:
    return f"{family} / {language_label(language)} / {version_label(version)}"


def status_label(status: str | None) -> str:
    if not status:
        return "-"
    return STATUS_DISPLAY.get(status, status)


def enrich_detail_record(record: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(record)
    native_e2e = enriched.get("native_e2e_time_ms")
    native_tok = enriched.get("native_tokenizer_time_ms")
    lopt_e2e = enriched.get("lopt_e2e_time_ms")
    lopt_mp = enriched.get("mp_dispatch_process_collect_time_ms")
    enriched["e2e_speedup_x"] = (
        round(float(native_e2e) / float(lopt_e2e), 6)
        if native_e2e and lopt_e2e
        else None
    )
    enriched["tokenizer_speedup_x"] = (
        round(float(native_tok) / float(lopt_mp), 6)
        if native_tok and lopt_mp
        else None
    )
    enriched["tokenizer_time_drop_pct"] = (
        round((1.0 - (float(lopt_mp) / float(native_tok))) * 100.0, 6)
        if native_tok and lopt_mp
        else None
    )
    return enriched


def json_script_payload(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")


def badge_class_speedup(value: float | None) -> str:
    if value is None:
        return "neutral"
    if value >= 1.2:
        return "good"
    if value >= 1.0:
        return "ok"
    return "warn"


def details_id(prefix: str, key: CaseKey) -> str:
    raw = f"{prefix}-{key.tokenizer_family}-{key.language}-{key.length_label}-{key.input_chars}"
    return "".join(ch if ch.isalnum() else "-" for ch in raw)


def render_ascii_block(title: str, content: str, accent: str = "blue") -> str:
    return f"""
    <div class="ascii-card {accent}">
      <div class="ascii-title">{escape(title)}</div>
      <pre>{escape(content.strip())}</pre>
    </div>
    """


def render_table(headers: list[str], rows: Iterable[list[str]], class_name: str = "") -> str:
    row_html = []
    for row in rows:
        cells = "".join(f"<td>{cell}</td>" for cell in row)
        row_html.append(f"<tr>{cells}</tr>")
    return f"""
    <table class="{class_name}">
      <thead><tr>{"".join(f"<th>{header}</th>" for header in headers)}</tr></thead>
      <tbody>
        {"".join(row_html)}
      </tbody>
    </table>
    """


def parse_cpuset_count(cpuset: str) -> int:
    total = 0
    for chunk in cpuset.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            total += end - start + 1
        else:
            total += 1
    return total


def extract_ascii_flow(flow_doc: str) -> str:
    marker = "## ASCII Flowchart"
    if marker not in flow_doc:
        return flow_doc.strip()
    tail = flow_doc.split(marker, 1)[1]
    start = tail.find("```text")
    if start == -1:
        return tail.strip()
    tail = tail[start + len("```text") :]
    end = tail.find("```")
    if end == -1:
        return tail.strip()
    return tail[:end].strip()


def extract_key_source_lines(flow_doc: str) -> list[str]:
    marker = "## Key Source Files"
    if marker not in flow_doc:
        return []
    section = flow_doc.split(marker, 1)[1]
    end = section.find("## ")
    if end != -1:
        section = section[:end]
    lines = []
    for raw in section.splitlines():
        raw = raw.strip()
        if raw.startswith("- "):
            lines.append(raw[2:])
    return lines


def load_corpus_sources(meta_dir: Path | None) -> dict[str, dict[str, Any]]:
    if meta_dir is None:
        return {}
    result = {}
    for language in ("zh", "en"):
        meta_path = meta_dir / f"{language}_sources.json"
        if meta_path.exists():
            result[language] = load_json(meta_path)
    return result


def load_env_info(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return load_json(path)


def build_principle_text() -> list[str]:
    return [
        "LoPT v1 保留了论文的核心思想：先把长文本按字符切成带 overlap 的多个 chunk，再在多个进程中并行执行 tokenization，最后基于位置感知的 overlap 匹配把各 chunk 的 token IDs 合并，确保最终 token 序列与串行 tokenization 完全一致。",
        "LoPT v2.1 继续保留 v1 的分块、并行和 overlap 去重主思路，但把父进程侧的调度与收集链路拆得更细，显式记录 submit / collect / child compute / return tail / receive lag 等阶段，便于把瓶颈定位到更细粒度。",
        "在本次 benchmark 实现里，每个 worker process 内部各自持有一个本地 HF fast tokenizer。父进程负责分发 chunk 文本、回收每个 chunk 的 token IDs 和 offset mappings，再利用全局字符位置去识别并删除重叠区域中的重复 token。",
        "相较论文版本，这一版工程实现不做在线 chunk-length retry。按照你的要求，候选配置一旦失败就直接回退到原始串行逻辑；而离线 search 则穷举 worker 数、chunk 数和 overlap 大小，以找到稳定且 exact-match 的配置。",
    ]


def build_native_bottleneck_points() -> list[str]:
    return [
        "在通用场景下，Tokenizer 结果无法跨请求缓存，因此每个长输入都必须从原始文本重新执行 tokenization。",
        "vLLM 原生链路虽然已经使用了 AsyncMicrobatchTokenizer 和共享线程池，但单个超长 prompt 本质上仍然对应一次底层 tokenizer 调用。线程池提升的是请求间并发能力，而不是单请求内部 1M 级长文本的切分并行。",
        "当长上下文模型、Prefill 高 Cache 命中率以及 PD 分离共同降低了后续推理阶段压力后，CPU 侧 Tokenizer 就会成为单请求的主要串行热点。",
        "在 1M 级输入下，哪怕 overlap 去重或 merge 只增加少量额外 CPU 开销，都可能抵消并行收益，所以优化必须同时满足两点：显著降低 tokenization 主耗时，并且把 merge 开销控制在较低水平，同时保证精度完全一致。",
        "v1 已经证明了多进程切分 + overlap 去重这条路是可行的；v2.1 进一步把主耗时拆成 dispatch / child compute / collect tail / dedup 四个可观察阶段，用于继续找真正的瓶颈所在。",
    ]


def build_background_points() -> list[str]:
    return [
        "场景：Agent 应用向 vLLM 推理服务发送超长 prompt 请求。",
        "当前系统背景：Tokenizer 异步线程池已经启用；主流大模型普遍支持最高 1M 上下文；Prefill 阶段假设具有很高的 Cache 命中率；PD 分离后推理侧已经不再主导整体时延预算。",
        "核心痛点：Tokenizer 结果不可缓存，每个请求都必须重新计算；当输入文本达到超长规模时，Tokenizer 延迟会成为新的 CPU 瓶颈。",
        "本次 benchmark 的目标：在完全一致的模型、真实语料与精度校验条件下，对比 vLLM 原生 Tokenizer 链路与 LoPT v1 / v2.1 风格多进程并行实现的性能差异。",
    ]


def render_metric_legend() -> str:
    rows = [
        [r"<code>c_in</code>", "输入字符数 (chars)", "当前测试用例的 prompt 文本长度。"],
        [r"<code>n_out</code>", "输出 token 数 (tok)", "tokenization 完成后的最终 token 数量。"],
        [r"<code>p</code>", "worker 进程数 (proc)", "LoPT Tokenizer worker processes 的数量。"],
        [r"<code>k</code>", "chunk 数量 (count)", "分发到 process pool 的文本 chunk 总数。"],
        [r"<code>o</code>", "overlap 大小 (chars)", "相邻 chunk 之间的字符重叠长度。"],
        [r"<code>t_native_e2e</code>", "原生 E2E 耗时 (ms)", "从请求进入到原生 token IDs 返回的完整端到端耗时。"],
        [r"<code>t_native_tok</code>", "原生 Tokenizer 纯耗时 (ms)", "进入 <code>AsyncMicrobatchTokenizer.encode</code> 后到返回结果的耗时。"],
        [r"<code>t_chat</code>", "chat template 耗时 (ms)", "LoPT 分发前的模板处理耗时；本次 completion-style benchmark 中为 0.0 ms。"],
        [r"<code>t_submit</code>", "dispatch submit 耗时 (ms)", "父进程把 chunk 提交到进程池所花的时间。"],
        [r"<code>t_collect</code>", "dispatch collect 耗时 (ms)", "父进程等待并收集所有 worker 返回值所花的时间。"],
        [r"<code>t_lopt_mp</code>", "LoPT 多进程分发/处理/回收耗时 (ms)", "父进程分发 chunk 到所有 worker process 并回收结果的总耗时。"],
        [r"<code>t_worker_encode_sum</code>", "worker encode 总时长 (ms)", "所有 worker 子进程 encode 时间的求和。"],
        [r"<code>t_worker_encode_max</code>", "worker encode 最大时长 (ms)", "最慢 chunk 的 encode 时间上界。"],
        [r"<code>t_worker_mat_sum</code>", "worker materialize 总时长 (ms)", "所有 worker 对 input_ids / offsets 物化的总时间。"],
        [r"<code>t_worker_mat_max</code>", "worker materialize 最大时长 (ms)", "最慢 chunk 的物化时间上界。"],
        [r"<code>t_dedup</code>", "chunk 去重耗时 (ms)", "父进程对 overlap 区域执行匹配和去冗余的耗时。"],
        [r"<code>t_lopt_e2e</code>", "LoPT E2E 耗时 (ms)", r"<code>t_chat + t_lopt_mp + t_dedup</code>。"],
        [r"<code>S_e2e</code>", "E2E 加速比 (x)", r"<code>t_native_e2e / t_lopt_e2e</code>。"],
        [r"<code>S_tok</code>", "Tokenizer 加速比 (x)", r"<code>t_native_tok / t_lopt_mp</code>。"],
        [r"<code>D_tok</code>", "Tokenizer 耗时下降幅度 (%)", r"<code>(1 - t_lopt_mp / t_native_tok) * 100</code>。"],
    ]
    return render_table(
        ["符号", "单位 / 含义", "定义"],
        rows,
        class_name="legend-table",
    )


def render_source_panel(corpus_sources: dict[str, dict[str, Any]]) -> str:
    if not corpus_sources:
        return "<p class=\"muted\">本次报告未提供语料来源元数据。</p>"

    sections = []
    for language in ("zh", "en"):
        meta = corpus_sources.get(language)
        if not meta:
            continue
        source_rows = []
        for source in meta.get("sources", []):
            source_rows.append(
                [
                    escape(source.get("url", "-")),
                    fmt_int(source.get("extracted_chars")),
                    fmt_int(source.get("kept_chars")),
                ]
            )
        info = [
            f"<span><strong>语言</strong>: {language_label(language)}</span>",
            f"<span><strong>最终字符数</strong>: {fmt_int(meta.get('final_chars'))} chars</span>",
            f"<span><strong>目标字符数</strong>: {fmt_int(meta.get('target_chars'))} chars</span>",
            f"<span><strong>是否由种子扩展</strong>: {'是' if meta.get('expanded_from_existing_seed') else '否'}</span>",
        ]
        sections.append(
            f"""
            <details class="fold">
              <summary>{language_label(language)}真实网页语料来源</summary>
              <div class="body">
                <div class="chip-row">{''.join(f'<span class="chip">{item}</span>' for item in info)}</div>
                <p class="muted">本次 benchmark 使用来自真实公开网页的纯{language_label(language)}文本，并做了可见文本抽取与规范化处理。当远端抓取不稳定时，使用真实网页文本块循环扩展语料，直到覆盖所需的超长输入长度。</p>
                {render_table(
                    ["来源 URL", "抽取可见文本 (chars)", "保留目标语言文本 (chars)"],
                    source_rows or [["-", "-", "-"]],
                    class_name="dense-table",
                )}
              </div>
            </details>
            """
        )
    return "".join(sections)


def case_title(record: dict[str, Any]) -> str:
    return (
        f"{record['tokenizer_family']} / {record['language'].upper()} / "
        f"{record['length_label']} ({fmt_int(record['input_chars'])} chars)"
    )


def summarize_replay_records(replay_records: list[dict[str, Any]]) -> dict[str, Any]:
    if not replay_records:
        return {}
    avg_native = sum(float(r["native_e2e_time_ms"]) for r in replay_records) / len(replay_records)
    avg_lopt = sum(float(r["lopt_e2e_time_ms"]) for r in replay_records) / len(replay_records)
    avg_speedup = sum(float(r["e2e_speedup_x"]) for r in replay_records) / len(replay_records)
    max_speedup = max(replay_records, key=lambda r: float(r["e2e_speedup_x"]))
    min_speedup = min(replay_records, key=lambda r: float(r["e2e_speedup_x"]))
    exact_count = sum(bool(r.get("exact_match")) for r in replay_records)
    return {
        "case_count": len(replay_records),
        "avg_native": avg_native,
        "avg_lopt": avg_lopt,
        "avg_speedup": avg_speedup,
        "exact_count": exact_count,
        "max_speedup": max_speedup,
        "min_speedup": min_speedup,
    }


def compare_replay_records(
    v1_records: list[dict[str, Any]],
    v2_records: list[dict[str, Any]],
) -> dict[str, Any]:
    v1_by_key = {
        (r["tokenizer_family"], r["language"], r["length_label"]): r
        for r in v1_records
    }
    rows = []
    for r2 in v2_records:
        key = (r2["tokenizer_family"], r2["language"], r2["length_label"])
        r1 = v1_by_key.get(key)
        if not r1:
            continue
        rows.append(
            {
                "tokenizer_family": r2["tokenizer_family"],
                "language": r2["language"],
                "length_label": r2["length_label"],
                "input_chars": int(r2["input_chars"]),
                "v1_e2e": float(r1["lopt_e2e_time_ms"]),
                "v2_e2e": float(r2["lopt_e2e_time_ms"]),
                "speedup_x": float(r1["lopt_e2e_time_ms"]) / float(r2["lopt_e2e_time_ms"]),
                "delta_ms": float(r1["lopt_e2e_time_ms"]) - float(r2["lopt_e2e_time_ms"]),
                "v1_exact": bool(r1.get("exact_match")),
                "v2_exact": bool(r2.get("exact_match")),
                "v1_worker_processes": int(r1["worker_processes"]),
                "v2_worker_processes": int(r2["worker_processes"]),
                "v1_chunk_count": int(r1["chunk_count"]),
                "v2_chunk_count": int(r2["chunk_count"]),
                "v1_overlap": int(r1["overlap_chars"]),
                "v2_overlap": int(r2["overlap_chars"]),
            }
        )
    return {
        "rows": rows,
        "avg_speedup": (
            sum(row["speedup_x"] for row in rows) / len(rows) if rows else None
        ),
        "avg_delta_ms": (
            sum(row["delta_ms"] for row in rows) / len(rows) if rows else None
        ),
        "best_row": max(rows, key=lambda row: row["speedup_x"]) if rows else None,
        "worst_row": min(rows, key=lambda row: row["speedup_x"]) if rows else None,
    }


def render_best_overview(replay_records: list[dict[str, Any]]) -> str:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in replay_records:
        grouped[(record["tokenizer_family"], record["language"])].append(record)

    sections = []
    headers = [
        "输入长度",
        "输入 c_in (chars)",
        "输出 n_out (tok)",
        "p (proc)",
        "k (count)",
        "o (chars)",
        "原生 E2E t_native_e2e (ms)",
        "原生 Tokenizer t_native_tok (ms)",
        "LoPT 多进程 t_lopt_mp (ms)",
        "去重 t_dedup (ms)",
        "LoPT E2E t_lopt_e2e (ms)",
        "E2E 加速比 S_e2e (x)",
        "Tokenizer 加速比 S_tok (x)",
        "Tokenizer 降幅 D_tok (%)",
        "精度",
    ]
    for group_key in sorted(
        grouped,
        key=lambda item: (FAMILY_ORDER.get(item[0], 99), LANGUAGE_ORDER.get(item[1], 99)),
    ):
        family, language = group_key
        records = sorted(grouped[group_key], key=lambda record: int(record["input_chars"]))
        rows = []
        for record in records:
            speedup = float(record["e2e_speedup_x"])
            rows.append(
                [
                    escape(record["length_label"]),
                    fmt_int(record["input_chars"]),
                    fmt_int(record["native_output_tokens"]),
                    fmt_int(record["worker_processes"]),
                    fmt_int(record["chunk_count"]),
                    fmt_int(record["overlap_chars"]),
                    fmt_ms(record["native_e2e_time_ms"]),
                    fmt_ms(record["native_tokenizer_time_ms"]),
                    fmt_ms(record["mp_dispatch_process_collect_time_ms"]),
                    fmt_ms(record["chunk_dedup_time_ms"]),
                    fmt_ms(record["lopt_e2e_time_ms"]),
                    f'<span class="metric-badge {badge_class_speedup(speedup)}">{fmt_x(speedup)}</span>',
                    fmt_x(record["tokenizer_speedup_x"]),
                    fmt_pct(record["tokenizer_time_drop_pct"]),
                    "Exact Match" if record["exact_match"] else "Mismatch",
                ]
            )
        sections.append(
            f"""
            <details class="fold" open>
              <summary>{family_language_label(family, language)} 最终 replay 结果表</summary>
              <div class="body">
                {render_table(headers, rows)}
              </div>
            </details>
            """
        )
    return "".join(sections)


def render_svg_line_chart(
    replay_records: list[dict[str, Any]],
    metric_key: str,
    chart_title: str,
    y_label: str,
    formatter: str,
) -> str:
    if not replay_records:
        return ""

    width = 1120
    height = 340
    margin_left = 72
    margin_right = 28
    margin_top = 34
    margin_bottom = 58
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    ordered_lengths = sorted(
        {record["length_label"]: int(record["input_chars"]) for record in replay_records}.items(),
        key=lambda item: item[1],
    )
    length_labels = [label for label, _ in ordered_lengths]
    x_positions = {
        label: (
            margin_left
            + (plot_width * idx / max(1, len(length_labels) - 1))
            if len(length_labels) > 1
            else margin_left + plot_width / 2
        )
        for idx, label in enumerate(length_labels)
    }

    series_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in replay_records:
        series_groups[(record["tokenizer_family"], record["language"])].append(record)

    values = [
        float(record[metric_key])
        for record in replay_records
        if record.get(metric_key) is not None
    ]
    if not values:
        return ""
    y_min = 0.0 if min(values) >= 0 else min(values)
    y_max = max(values)
    if y_max == y_min:
        y_max = y_min + 1.0
    padded_max = y_max * 1.08
    if padded_max == 0:
        padded_max = 1.0

    def y_coord(value: float) -> float:
        return margin_top + plot_height - ((value - y_min) / (padded_max - y_min)) * plot_height

    ticks = 5
    grid_lines = []
    for idx in range(ticks + 1):
        ratio = idx / ticks
        y_value = y_min + (padded_max - y_min) * (1 - ratio)
        y = margin_top + plot_height * ratio
        label = f"{y_value:.2f}" if formatter in {"x", "pct"} else f"{y_value:.0f}"
        if formatter == "pct":
            label = f"{y_value:.0f}%"
        elif formatter == "x":
            label = f"{y_value:.2f}x"
        elif formatter == "ms":
            label = f"{y_value:.0f} ms"
        grid_lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#e3ebf3" stroke-width="1" />'
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11" fill="#6b7c8f">{escape(label)}</text>'
        )

    x_axis_labels = []
    for label in length_labels:
        x = x_positions[label]
        x_axis_labels.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + plot_height}" stroke="#eef3f8" stroke-width="1" />'
            f'<text x="{x:.2f}" y="{height - 22}" text-anchor="middle" font-size="11" fill="#6b7c8f">{escape(label)}</text>'
        )

    line_paths = []
    legend_items = []
    for series_key in sorted(
        series_groups,
        key=lambda item: (FAMILY_ORDER.get(item[0], 99), LANGUAGE_ORDER.get(item[1], 99)),
    ):
        family, language = series_key
        points = sorted(series_groups[series_key], key=lambda record: int(record["input_chars"]))
        color = SERIES_COLORS.get(series_key, "#1f6feb")
        coord_parts = []
        circles = []
        for record in points:
            x = x_positions[record["length_label"]]
            y = y_coord(float(record[metric_key]))
            coord_parts.append(f"{x:.2f},{y:.2f}")
            circles.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{color}" stroke="#ffffff" stroke-width="1.5">'
                f'<title>{escape(family_language_label(family, language))} / {record["length_label"]}: {record[metric_key]}</title>'
                f'</circle>'
            )
        line_paths.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(coord_parts)}" />'
            + "".join(circles)
        )
        legend_items.append(
            f'<div class="chart-legend-item"><span class="chart-swatch" style="background:{color}"></span>{escape(family_language_label(family, language))}</div>'
        )

    return f"""
    <div class="chart-card">
      <div class="chart-head">
        <div>
          <div class="chart-title">{escape(chart_title)}</div>
          <div class="chart-subtitle">{escape(y_label)}</div>
        </div>
        <div class="chart-legend">{''.join(legend_items)}</div>
      </div>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(chart_title)}">
        <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"></rect>
        {''.join(grid_lines)}
        {''.join(x_axis_labels)}
        <line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#b8c8d9" stroke-width="1.2" />
        <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#b8c8d9" stroke-width="1.2" />
        {''.join(line_paths)}
        <text x="{width / 2:.2f}" y="{height - 6}" text-anchor="middle" font-size="12" fill="#5f7286">输入长度</text>
        <text x="18" y="{height / 2:.2f}" text-anchor="middle" font-size="12" fill="#5f7286" transform="rotate(-90 18 {height / 2:.2f})">{escape(y_label)}</text>
      </svg>
    </div>
    """


def render_svg_dual_line_chart(
    replay_records: list[dict[str, Any]],
    primary_metric_key: str,
    secondary_metric_key: str,
    chart_title: str,
    y_label: str,
    primary_label: str = "LoPT E2E",
    secondary_label: str = "原生 E2E",
) -> str:
    if not replay_records:
        return ""

    width = 1120
    height = 340
    margin_left = 72
    margin_right = 28
    margin_top = 34
    margin_bottom = 58
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    ordered_lengths = sorted(
        {record["length_label"]: int(record["input_chars"]) for record in replay_records}.items(),
        key=lambda item: item[1],
    )
    length_labels = [label for label, _ in ordered_lengths]
    x_positions = {
        label: (
            margin_left
            + (plot_width * idx / max(1, len(length_labels) - 1))
            if len(length_labels) > 1
            else margin_left + plot_width / 2
        )
        for idx, label in enumerate(length_labels)
    }

    series_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in replay_records:
        series_groups[(record["tokenizer_family"], record["language"])].append(record)

    values = []
    for record in replay_records:
        if record.get(primary_metric_key) is not None:
            values.append(float(record[primary_metric_key]))
        if record.get(secondary_metric_key) is not None:
            values.append(float(record[secondary_metric_key]))
    if not values:
        return ""
    y_min = 0.0 if min(values) >= 0 else min(values)
    y_max = max(values)
    if y_max == y_min:
        y_max = y_min + 1.0
    padded_max = y_max * 1.08
    if padded_max == 0:
        padded_max = 1.0

    def y_coord(value: float) -> float:
        return margin_top + plot_height - ((value - y_min) / (padded_max - y_min)) * plot_height

    grid_lines = []
    for idx in range(6):
        ratio = idx / 5
        y_value = y_min + (padded_max - y_min) * (1 - ratio)
        y = margin_top + plot_height * ratio
        grid_lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#e3ebf3" stroke-width="1" />'
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11" fill="#6b7c8f">{y_value:.0f} ms</text>'
        )

    x_axis_labels = []
    for label in length_labels:
        x = x_positions[label]
        x_axis_labels.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + plot_height}" stroke="#eef3f8" stroke-width="1" />'
            f'<text x="{x:.2f}" y="{height - 22}" text-anchor="middle" font-size="11" fill="#6b7c8f">{escape(label)}</text>'
        )

    line_paths = []
    legend_items = []
    for series_key in sorted(
        series_groups,
        key=lambda item: (FAMILY_ORDER.get(item[0], 99), LANGUAGE_ORDER.get(item[1], 99)),
    ):
        family, language = series_key
        points = sorted(series_groups[series_key], key=lambda record: int(record["input_chars"]))
        base_color = SERIES_COLORS.get(series_key, "#1f6feb")
        secondary_color = "#91a4b7"
        primary_points = []
        secondary_points = []
        primary_circles = []
        secondary_circles = []
        for record in points:
            x = x_positions[record["length_label"]]
            primary_y = y_coord(float(record[primary_metric_key]))
            secondary_y = y_coord(float(record[secondary_metric_key]))
            primary_points.append(f"{x:.2f},{primary_y:.2f}")
            secondary_points.append(f"{x:.2f},{secondary_y:.2f}")
            primary_circles.append(
                f'<circle cx="{x:.2f}" cy="{primary_y:.2f}" r="4.5" fill="{base_color}" stroke="#ffffff" stroke-width="1.5"></circle>'
            )
            secondary_circles.append(
                f'<circle cx="{x:.2f}" cy="{secondary_y:.2f}" r="3.8" fill="#ffffff" stroke="{base_color}" stroke-width="2"></circle>'
            )
        line_paths.append(
            f'<polyline fill="none" stroke="{secondary_color}" stroke-width="2" stroke-dasharray="6 4" points="{" ".join(primary_points)}" opacity="0.0"></polyline>'
        )
        line_paths.append(
            f'<polyline fill="none" stroke="{base_color}" stroke-width="3" points="{" ".join(primary_points)}" />'
            f'<polyline fill="none" stroke="{base_color}" stroke-width="2" stroke-dasharray="6 4" opacity="0.52" points="{" ".join(secondary_points)}" />'
            + "".join(primary_circles)
            + "".join(secondary_circles)
        )
        legend_items.append(
            f'<div class="chart-legend-item"><span class="chart-swatch" style="background:{base_color}"></span>{escape(family_language_label(family, language))}</div>'
        )

    metric_legend = "".join(
        [
            f'<div class="chart-legend-item"><span class="chart-line solid"></span>{escape(primary_label)}</div>',
            f'<div class="chart-legend-item"><span class="chart-line dashed"></span>{escape(secondary_label)}</div>',
        ]
    )

    return f"""
    <div class="chart-card">
      <div class="chart-head">
        <div>
          <div class="chart-title">{escape(chart_title)}</div>
          <div class="chart-subtitle">{escape(y_label)}</div>
        </div>
        <div>
          <div class="chart-legend">{''.join(legend_items)}</div>
          <div class="chart-legend metric-topology">{metric_legend}</div>
        </div>
      </div>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(chart_title)}">
        <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"></rect>
        {''.join(grid_lines)}
        {''.join(x_axis_labels)}
        <line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#b8c8d9" stroke-width="1.2" />
        <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#b8c8d9" stroke-width="1.2" />
        {''.join(line_paths)}
        <text x="{width / 2:.2f}" y="{height - 6}" text-anchor="middle" font-size="12" fill="#5f7286">输入长度</text>
        <text x="18" y="{height / 2:.2f}" text-anchor="middle" font-size="12" fill="#5f7286" transform="rotate(-90 18 {height / 2:.2f})">{escape(y_label)}</text>
      </svg>
    </div>
    """


def render_svg_stacked_bar_chart(replay_records: list[dict[str, Any]]) -> str:
    if not replay_records:
        return ""

    records = sorted(
        replay_records,
        key=lambda record: (
            FAMILY_ORDER.get(record["tokenizer_family"], 99),
            LANGUAGE_ORDER.get(record["language"], 99),
            int(record["input_chars"]),
        ),
    )

    width = max(1400, 110 + len(records) * 24)
    height = 360
    margin_left = 72
    margin_right = 28
    margin_top = 30
    margin_bottom = 88
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    bar_width = max(10, plot_width / max(1, len(records)) * 0.58)
    gap = plot_width / max(1, len(records))

    max_total = max(float(record["lopt_e2e_time_ms"]) for record in records)
    if max_total <= 0:
        max_total = 1.0
    padded_max = max_total * 1.12

    def bar_height(value: float) -> float:
        return (value / padded_max) * plot_height

    bars = []
    x_labels = []
    for idx, record in enumerate(records):
        x = margin_left + idx * gap + (gap - bar_width) / 2
        chat_h = bar_height(float(record["chat_template_time_ms"]))
        mp_h = bar_height(float(record["mp_dispatch_process_collect_time_ms"]))
        dedup_h = bar_height(float(record["chunk_dedup_time_ms"]))
        base_y = margin_top + plot_height
        y_chat = base_y - chat_h
        y_mp = y_chat - mp_h
        y_dedup = y_mp - dedup_h
        bars.append(
            f'<rect x="{x:.2f}" y="{y_chat:.2f}" width="{bar_width:.2f}" height="{chat_h:.2f}" fill="{BREAKDOWN_COLORS["chat"]}"><title>t_chat={record["chat_template_time_ms"]} ms</title></rect>'
            f'<rect x="{x:.2f}" y="{y_mp:.2f}" width="{bar_width:.2f}" height="{mp_h:.2f}" fill="{BREAKDOWN_COLORS["mp"]}"><title>t_lopt_mp={record["mp_dispatch_process_collect_time_ms"]} ms</title></rect>'
            f'<rect x="{x:.2f}" y="{y_dedup:.2f}" width="{bar_width:.2f}" height="{dedup_h:.2f}" fill="{BREAKDOWN_COLORS["dedup"]}"><title>t_dedup={record["chunk_dedup_time_ms"]} ms</title></rect>'
        )
        x_labels.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{height - 48}" text-anchor="end" font-size="10" fill="#5f7286" transform="rotate(-55 {x + bar_width / 2:.2f} {height - 48})">{escape(record["tokenizer_family"].replace("DeepSeek-V4-Pro", "DSV4-Pro"))}/{escape(record["language"].upper())}/{escape(record["length_label"])}</text>'
        )

    grid_lines = []
    for idx in range(6):
        ratio = idx / 5
        y_value = padded_max * (1 - ratio)
        y = margin_top + plot_height * ratio
        grid_lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#e3ebf3" stroke-width="1" />'
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11" fill="#6b7c8f">{y_value:.0f} ms</text>'
        )

    legend_items = [
        ("t_chat", BREAKDOWN_COLORS["chat"]),
        ("t_lopt_mp", BREAKDOWN_COLORS["mp"]),
        ("t_dedup", BREAKDOWN_COLORS["dedup"]),
    ]
    legend_html = "".join(
        f'<div class="chart-legend-item"><span class="chart-swatch" style="background:{color}"></span>{label}</div>'
        for label, color in legend_items
    )

    return f"""
    <div class="chart-card">
      <div class="chart-head">
        <div>
          <div class="chart-title">LoPT E2E 分段耗时堆叠图</div>
          <div class="chart-subtitle">展示每个最优 replay case 中 t_chat / t_lopt_mp / t_dedup 的组成</div>
        </div>
        <div class="chart-legend">{legend_html}</div>
      </div>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="LoPT E2E breakdown chart">
        <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"></rect>
        {''.join(grid_lines)}
        <line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#b8c8d9" stroke-width="1.2" />
        <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#b8c8d9" stroke-width="1.2" />
        {''.join(bars)}
        {''.join(x_labels)}
        <text x="{width / 2:.2f}" y="{height - 8}" text-anchor="middle" font-size="12" fill="#5f7286">模型 / 语言 / 输入长度</text>
      </svg>
    </div>
    """


def render_svg_dual_bar_chart(
    replay_records: list[dict[str, Any]],
    primary_metric_key: str = "native_e2e_time_ms",
    secondary_metric_key: str = "lopt_e2e_time_ms",
    primary_label: str = "原生 E2E",
    secondary_label: str = "LoPT E2E",
    chart_title: str = "原生 E2E 与 LoPT E2E 对比柱状图",
    chart_subtitle: str = "同一最优配置下，直接对比优化前后总耗时",
) -> str:
    if not replay_records:
        return ""

    records = sorted(
        replay_records,
        key=lambda record: (
            FAMILY_ORDER.get(record["tokenizer_family"], 99),
            LANGUAGE_ORDER.get(record["language"], 99),
            int(record["input_chars"]),
        ),
    )
    width = max(1500, 120 + len(records) * 30)
    height = 380
    margin_left = 72
    margin_right = 28
    margin_top = 30
    margin_bottom = 96
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    pair_gap = plot_width / max(1, len(records))
    bar_width = max(8, min(18, pair_gap * 0.28))
    inner_gap = max(4, bar_width * 0.45)
    max_value = max(
        max(float(record[primary_metric_key]), float(record[secondary_metric_key]))
        for record in records
    )
    if max_value <= 0:
        max_value = 1.0
    padded_max = max_value * 1.12

    def bar_height(value: float) -> float:
        return (value / padded_max) * plot_height

    bars = []
    labels = []
    for idx, record in enumerate(records):
        center_x = margin_left + idx * pair_gap + pair_gap / 2
        native_h = bar_height(float(record[primary_metric_key]))
        lopt_h = bar_height(float(record[secondary_metric_key]))
        base_y = margin_top + plot_height
        native_x = center_x - inner_gap / 2 - bar_width
        lopt_x = center_x + inner_gap / 2
        bars.append(
            f'<rect x="{native_x:.2f}" y="{base_y - native_h:.2f}" width="{bar_width:.2f}" height="{native_h:.2f}" fill="{BREAKDOWN_COLORS["native"]}"><title>{escape(primary_label)}={record[primary_metric_key]} ms</title></rect>'
            f'<rect x="{lopt_x:.2f}" y="{base_y - lopt_h:.2f}" width="{bar_width:.2f}" height="{lopt_h:.2f}" fill="{BREAKDOWN_COLORS["lopt"]}"><title>{escape(secondary_label)}={record[secondary_metric_key]} ms</title></rect>'
        )
        labels.append(
            f'<text x="{center_x:.2f}" y="{height - 52}" text-anchor="end" font-size="10" fill="#5f7286" transform="rotate(-55 {center_x:.2f} {height - 52})">{escape(record["tokenizer_family"].replace("DeepSeek-V4-Pro", "DSV4-Pro"))}/{escape(record["language"].upper())}/{escape(record["length_label"])}</text>'
        )

    grid_lines = []
    for idx in range(6):
        ratio = idx / 5
        y_value = padded_max * (1 - ratio)
        y = margin_top + plot_height * ratio
        grid_lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#e3ebf3" stroke-width="1" />'
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11" fill="#6b7c8f">{y_value:.0f} ms</text>'
        )

    legend_html = "".join(
        f'<div class="chart-legend-item"><span class="chart-swatch" style="background:{color}"></span>{label}</div>'
        for label, color in [
            (primary_label, BREAKDOWN_COLORS["native"]),
            (secondary_label, BREAKDOWN_COLORS["lopt"]),
        ]
    )

    return f"""
    <div class="chart-card">
      <div class="chart-head">
        <div>
          <div class="chart-title">{escape(chart_title)}</div>
          <div class="chart-subtitle">{escape(chart_subtitle)}</div>
        </div>
        <div class="chart-legend">{legend_html}</div>
      </div>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="native versus LoPT E2E chart">
        <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"></rect>
        {''.join(grid_lines)}
        <line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#b8c8d9" stroke-width="1.2" />
        <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#b8c8d9" stroke-width="1.2" />
        {''.join(bars)}
        {''.join(labels)}
        <text x="{width / 2:.2f}" y="{height - 10}" text-anchor="middle" font-size="12" fill="#5f7286">模型 / 语言 / 输入长度</text>
      </svg>
    </div>
    """


def render_family_language_time_small_svg(records: list[dict[str, Any]]) -> str:
    width = 520
    height = 190
    margin_left = 48
    margin_right = 18
    margin_top = 18
    margin_bottom = 42
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    max_value = max(
        max(float(record["native_e2e_time_ms"]), float(record["lopt_e2e_time_ms"]))
        for record in records
    )
    if max_value <= 0:
        max_value = 1.0
    padded_max = max_value * 1.1

    def y_coord(value: float) -> float:
        return margin_top + plot_height - (value / padded_max) * plot_height

    x_positions = {
        record["length_label"]: (
            margin_left
            + (plot_width * idx / max(1, len(records) - 1))
            if len(records) > 1
            else margin_left + plot_width / 2
        )
        for idx, record in enumerate(records)
    }

    native_points = []
    lopt_points = []
    markers = []
    x_labels = []
    for record in records:
        x = x_positions[record["length_label"]]
        native_y = y_coord(float(record["native_e2e_time_ms"]))
        lopt_y = y_coord(float(record["lopt_e2e_time_ms"]))
        native_points.append(f"{x:.2f},{native_y:.2f}")
        lopt_points.append(f"{x:.2f},{lopt_y:.2f}")
        markers.append(
            f'<circle cx="{x:.2f}" cy="{native_y:.2f}" r="3.5" fill="#f59e0b"></circle>'
            f'<circle cx="{x:.2f}" cy="{lopt_y:.2f}" r="3.5" fill="#13835b"></circle>'
        )
        x_labels.append(
            f'<text x="{x:.2f}" y="{height - 12}" text-anchor="middle" font-size="10" fill="#6b7c8f">{escape(record["length_label"])}</text>'
        )

    return f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="small multiple time chart">
      <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"></rect>
      <line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#b8c8d9" stroke-width="1" />
      <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#b8c8d9" stroke-width="1" />
      <polyline fill="none" stroke="#f59e0b" stroke-width="2.5" stroke-dasharray="6 4" points="{" ".join(native_points)}" />
      <polyline fill="none" stroke="#13835b" stroke-width="2.8" points="{" ".join(lopt_points)}" />
      {''.join(markers)}
      {''.join(x_labels)}
      <text x="{margin_left}" y="12" font-size="11" fill="#5f7286">原生/LoPT E2E (ms)</text>
    </svg>
    """


def render_family_language_ratio_small_svg(records: list[dict[str, Any]]) -> str:
    width = 520
    height = 180
    margin_left = 48
    margin_right = 18
    margin_top = 18
    margin_bottom = 44
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    gap = plot_width / max(1, len(records))
    bar_width = max(14, gap * 0.54)
    bars = []
    x_labels = []
    for idx, record in enumerate(records):
        total = max(float(record["lopt_e2e_time_ms"]), 1e-9)
        chat_pct = float(record["chat_template_time_ms"]) / total
        mp_pct = float(record["mp_dispatch_process_collect_time_ms"]) / total
        dedup_pct = float(record["chunk_dedup_time_ms"]) / total
        x = margin_left + idx * gap + (gap - bar_width) / 2
        base_y = margin_top + plot_height
        chat_h = plot_height * chat_pct
        mp_h = plot_height * mp_pct
        dedup_h = plot_height * dedup_pct
        y_chat = base_y - chat_h
        y_mp = y_chat - mp_h
        y_dedup = y_mp - dedup_h
        bars.append(
            f'<rect x="{x:.2f}" y="{y_chat:.2f}" width="{bar_width:.2f}" height="{chat_h:.2f}" fill="{BREAKDOWN_COLORS["chat"]}"></rect>'
            f'<rect x="{x:.2f}" y="{y_mp:.2f}" width="{bar_width:.2f}" height="{mp_h:.2f}" fill="{BREAKDOWN_COLORS["mp"]}"></rect>'
            f'<rect x="{x:.2f}" y="{y_dedup:.2f}" width="{bar_width:.2f}" height="{dedup_h:.2f}" fill="{BREAKDOWN_COLORS["dedup"]}"></rect>'
        )
        x_labels.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{height - 12}" text-anchor="middle" font-size="10" fill="#6b7c8f">{escape(record["length_label"])}</text>'
        )
    return f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="small multiple ratio chart">
      <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"></rect>
      <line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#b8c8d9" stroke-width="1" />
      <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#b8c8d9" stroke-width="1" />
      <text x="{margin_left}" y="12" font-size="11" fill="#5f7286">LoPT 分段占比</text>
      {''.join(bars)}
      {''.join(x_labels)}
      <text x="16" y="{margin_top + 8}" font-size="10" fill="#6b7c8f">100%</text>
      <text x="22" y="{margin_top + plot_height}" font-size="10" fill="#6b7c8f">0%</text>
    </svg>
    """


def render_family_language_small_multiples(replay_records: list[dict[str, Any]]) -> str:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in replay_records:
        grouped[(record["tokenizer_family"], record["language"])].append(record)

    cards = []
    for group_key in sorted(
        grouped,
        key=lambda item: (FAMILY_ORDER.get(item[0], 99), LANGUAGE_ORDER.get(item[1], 99)),
    ):
        family, language = group_key
        records = sorted(grouped[group_key], key=lambda record: int(record["input_chars"]))
        best_speedup = max(float(record["e2e_speedup_x"]) for record in records)
        best_drop = max(float(record["tokenizer_time_drop_pct"]) for record in records)
        all_exact = all(bool(record.get("exact_match")) for record in records)
        cards.append(
            f"""
            <div class="mini-card">
              <div class="mini-card-head">
                <div class="mini-card-title">{escape(family_language_label(family, language))}</div>
                <div class="mini-card-subtitle">原生时间 / LoPT 时间 / LoPT 分段占比</div>
              </div>
              <div class="chip-row compact">
                <span class="chip">case 数: {fmt_int(len(records))}</span>
                <span class="chip">最佳 E2E 加速比: {fmt_x(best_speedup)}</span>
                <span class="chip">最大 Tokenizer 降幅: {fmt_pct(best_drop)}</span>
                <span class="chip">精度: {'全部 Exact Match' if all_exact else '存在 Mismatch'}</span>
              </div>
              <div class="mini-svg-wrap">{render_family_language_time_small_svg(records)}</div>
              <div class="mini-svg-wrap">{render_family_language_ratio_small_svg(records)}</div>
            </div>
            """
        )

    legend = "".join(
        [
            f'<div class="chart-legend-item"><span class="chart-swatch" style="background:{BREAKDOWN_COLORS["native"]}"></span>原生 E2E</div>',
            f'<div class="chart-legend-item"><span class="chart-swatch" style="background:{BREAKDOWN_COLORS["lopt"]}"></span>LoPT E2E</div>',
            f'<div class="chart-legend-item"><span class="chart-swatch" style="background:{BREAKDOWN_COLORS["mp"]}"></span>t_lopt_mp</div>',
            f'<div class="chart-legend-item"><span class="chart-swatch" style="background:{BREAKDOWN_COLORS["dedup"]}"></span>t_dedup</div>',
            f'<div class="chart-legend-item"><span class="chart-swatch" style="background:{BREAKDOWN_COLORS["chat"]}"></span>t_chat</div>',
        ]
    )

    return f"""
    <div class="chart-card">
      <div class="chart-head">
        <div>
          <div class="chart-title">分模型 × 分语言小多图</div>
          <div class="chart-subtitle">每张小图卡片固定一个模型族与一种语言，便于观察不同输入长度下的原生时间、LoPT 时间与 LoPT 分段占比。</div>
        </div>
        <div class="chart-legend">{legend}</div>
      </div>
      <div class="mini-grid">
        {''.join(cards)}
      </div>
    </div>
    """


def render_best_case_summary(replay_records: list[dict[str, Any]]) -> str:
    headers = [
        "模型",
        "语言",
        "输入长度",
        "c_in (chars)",
        "n_out (tok)",
        "最优配置 p / k / o",
        "原生 E2E (ms)",
        "原生 Tokenizer (ms)",
        "LoPT t_chat (ms)",
        "LoPT t_lopt_mp (ms)",
        "LoPT t_dedup (ms)",
        "LoPT E2E (ms)",
        "E2E 加速比 (x)",
        "Tokenizer 加速比 (x)",
        "Tokenizer 降幅 (%)",
        "精度",
    ]
    rows = []
    for record in replay_records:
        rows.append(
            [
                escape(record["tokenizer_family"]),
                escape(language_label(record["language"])),
                escape(record["length_label"]),
                fmt_int(record["input_chars"]),
                fmt_int(record["native_output_tokens"]),
                escape(
                    f"p={fmt_int(record['worker_processes'])} / "
                    f"k={fmt_int(record['chunk_count'])} / "
                    f"o={fmt_int(record['overlap_chars'])}"
                ),
                fmt_ms(record["native_e2e_time_ms"]),
                fmt_ms(record["native_tokenizer_time_ms"]),
                fmt_ms(record["chat_template_time_ms"]),
                fmt_ms(record["mp_dispatch_process_collect_time_ms"]),
                fmt_ms(record["chunk_dedup_time_ms"]),
                fmt_ms(record["lopt_e2e_time_ms"]),
                f'<span class="metric-badge {badge_class_speedup(float(record["e2e_speedup_x"]))}">{fmt_x(record["e2e_speedup_x"])}</span>',
                f'<span class="metric-badge {badge_class_speedup(float(record["tokenizer_speedup_x"]))}">{fmt_x(record["tokenizer_speedup_x"])}</span>',
                fmt_pct(record["tokenizer_time_drop_pct"]),
                "Exact Match" if record["exact_match"] else "Mismatch",
            ]
        )

    speedup_chart = render_svg_line_chart(
        replay_records,
        "e2e_speedup_x",
        "最优配置 E2E 加速比曲线",
        "E2E 加速比 (x)",
        "x",
    )
    native_vs_lopt_chart = render_svg_dual_line_chart(
        replay_records,
        "lopt_e2e_time_ms",
        "native_e2e_time_ms",
        "最优配置 原生 vLLM 与 LoPT E2E 耗时双曲线",
        "原生 E2E 与 LoPT E2E (ms)",
    )
    e2e_bar_chart = render_svg_dual_bar_chart(replay_records)
    tokenizer_drop_chart = render_svg_line_chart(
        replay_records,
        "tokenizer_time_drop_pct",
        "Tokenizer 耗时下降幅度曲线",
        "Tokenizer 降幅 (%)",
        "pct",
    )
    breakdown_chart = render_svg_stacked_bar_chart(replay_records)
    small_multiples = render_family_language_small_multiples(replay_records)

    chips = [
        f"最优 replay case 数: {fmt_int(len(replay_records))}",
        f"全部 exact match: {'是' if all(record.get('exact_match') for record in replay_records) else '否'}",
        f"模型覆盖: {fmt_int(len({record['tokenizer_family'] for record in replay_records}))} 个",
        f"语言覆盖: {fmt_int(len({record['language'] for record in replay_records}))} 种",
    ]

    return f"""
    <section class="section" id="best-case-analysis">
      <div class="section-head">
        <h2>08 · 最佳配置综合对比</h2>
        <p>基于 final replay 的 48 个最优 case，集中展示不同模型 / 中英文 / 不同输入长度下的最佳配置、性能提升、耗时拆分与精度情况。</p>
      </div>
      <div class="chip-row">{''.join(f'<span class="chip">{escape(item)}</span>' for item in chips)}</div>
      <div class="chart-grid">
        {speedup_chart}
        {native_vs_lopt_chart}
      </div>
      <div class="chart-grid single wide-scroll">
        {e2e_bar_chart}
      </div>
      <div class="chart-grid single">
        {tokenizer_drop_chart}
      </div>
      <div class="chart-grid single wide-scroll">
        {breakdown_chart}
      </div>
      <div class="chart-grid single">
        {small_multiples}
      </div>
      <details class="fold" open>
        <summary>最优 replay case 综合表</summary>
        <div class="body">
          <p class="table-note">该表直接展示每个最终最优配置下的原生耗时、LoPT 分段耗时、总加速比和精度结果，适合做横向汇报与复核。</p>
          {render_table(headers, rows, class_name="dense-table")}
        </div>
      </details>
    </section>
    """


def render_version_comparison(ctx: dict[str, Any]) -> str:
    comparison = ctx.get("comparison") or {}
    rows = []
    for row in comparison.get("rows", []):
        rows.append(
            [
                escape(row["tokenizer_family"]),
                escape(language_label(row["language"])),
                escape(row["length_label"]),
                fmt_ms(row["v1_e2e"]),
                fmt_ms(row["v2_e2e"]),
                fmt_x(row["speedup_x"]),
                fmt_ms(row["delta_ms"]),
                "Exact" if row["v1_exact"] and row["v2_exact"] else "Mismatch",
                f"p={fmt_int(row['v1_worker_processes'])}→{fmt_int(row['v2_worker_processes'])}",
                f"k={fmt_int(row['v1_chunk_count'])}→{fmt_int(row['v2_chunk_count'])}",
                f"o={fmt_int(row['v1_overlap'])}→{fmt_int(row['v2_overlap'])}",
            ]
        )
    headers = [
        "模型",
        "语言",
        "长度",
        "v1 E2E (ms)",
        "v2.1 E2E (ms)",
        "加速比 (x)",
        "节省 (ms)",
        "精度",
        "p 变化",
        "k 变化",
        "o 变化",
    ]
    best_row = comparison.get("best_row")
    worst_row = comparison.get("worst_row")
    chips = []
    if comparison.get("avg_speedup") is not None:
        chips.append(f"48 case 平均提速: {fmt_x(comparison['avg_speedup'])}")
    if comparison.get("avg_delta_ms") is not None:
        chips.append(f"平均节省: {fmt_ms(comparison['avg_delta_ms'])} ms")
    if best_row:
        chips.append(
            f"最佳 case: {best_row['tokenizer_family']} / {language_label(best_row['language'])} / {best_row['length_label']} ({fmt_x(best_row['speedup_x'])})"
        )
    if worst_row:
        chips.append(
            f"最弱 case: {worst_row['tokenizer_family']} / {language_label(worst_row['language'])} / {worst_row['length_label']} ({fmt_x(worst_row['speedup_x'])})"
        )
    compare_chart = render_svg_dual_line_chart(
        comparison.get("rows", []),
        "v2_e2e",
        "v1_e2e",
        "v2.1 vs v1 E2E 曲线",
        "E2E 耗时 (ms)",
        primary_label="v2.1 E2E",
        secondary_label="v1 E2E",
    )
    compare_speedup_chart = render_svg_line_chart(
        comparison.get("rows", []),
        "speedup_x",
        "v2.1 相对 v1 加速比曲线",
        "E2E 加速比 (x)",
        "x",
    )
    return f"""
    <section class="section" id="version-compare">
      <div class="section-head">
        <h2>07 · v1 / v2.1 版本对比</h2>
        <p>把同一组 best case 的 v1 与 v2.1 最终 replay 结果并排对照，直观看出新版在哪些长度和语言上更快，以及参数形态是否发生变化。</p>
      </div>
      <div class="chip-row">{''.join(f'<span class="chip">{escape(item)}</span>' for item in chips)}</div>
      <div class="chart-grid">
        {compare_chart}
        {compare_speedup_chart}
      </div>
      <details class="fold" open>
        <summary>v1 vs v2.1 对照表</summary>
        <div class="body">
          <p class="table-note">该表按相同模型 / 语言 / 输入长度对齐 v1 和 v2.1 最优配置，展示最终 E2E 变化与参数变化。</p>
          {render_table(headers, rows, class_name="dense-table")}
        </div>
      </details>
    </section>
    """


def render_worker_best(worker_best_records: list[dict[str, Any]]) -> str:
    grouped: dict[CaseKey, list[dict[str, Any]]] = defaultdict(list)
    for record in worker_best_records:
        grouped[record_case_key(record)].append(record)

    sections = []
    headers = [
        "p (proc)",
        "k (count)",
        "o (chars)",
        "原生 E2E (ms)",
        "原生 Tokenizer (ms)",
        "LoPT 多进程 (ms)",
        "去重 (ms)",
        "LoPT E2E (ms)",
        "E2E 加速比 (x)",
        "Tokenizer 加速比 (x)",
        "Tokenizer 降幅 (%)",
        "精度",
    ]
    for case_key in sorted(grouped, key=sort_case_key):
        records = sorted(grouped[case_key], key=lambda item: int(item["worker_processes"]))
        rows = []
        for record in records:
            rows.append(
                [
                    fmt_int(record["worker_processes"]),
                    fmt_int(record["chunk_count"]),
                    fmt_int(record["overlap_chars"]),
                    fmt_ms(record["native_e2e_time_ms"]),
                    fmt_ms(record["native_tokenizer_time_ms"]),
                    fmt_ms(record["mp_dispatch_process_collect_time_ms"]),
                    fmt_ms(record["chunk_dedup_time_ms"]),
                    fmt_ms(record["lopt_e2e_time_ms"]),
                    fmt_x(record["e2e_speedup_x"]),
                    fmt_x(record["tokenizer_speedup_x"]),
                    fmt_pct(record["tokenizer_time_drop_pct"]),
                    "Exact Match" if record["exact_match"] else "Mismatch",
                ]
            )
        label = f"{case_key.tokenizer_family} / {language_label(case_key.language)} / {case_key.length_label}"
        sections.append(
            f"""
            <details class="subfold">
              <summary>{label}</summary>
              <div class="body">
                {render_table(headers, rows, class_name="dense-table")}
              </div>
            </details>
            """
        )
    return "".join(sections)


def render_candidate_details(
    detail_records: list[dict[str, Any]],
    best_by_case: dict[CaseKey, dict[str, Any]],
    replay_by_case: dict[CaseKey, dict[str, Any]],
) -> str:
    grouped: dict[CaseKey, list[dict[str, Any]]] = defaultdict(list)
    for record in detail_records:
        grouped[record_case_key(record)].append(record)

    family_sections = []
    family_groups: dict[str, dict[str, list[CaseKey]]] = defaultdict(lambda: defaultdict(list))
    for case_key in grouped:
        family_groups[case_key.tokenizer_family][case_key.language].append(case_key)

    candidate_headers = [
        "p (proc)",
        "k (count)",
        "chunk 大小 (chars)",
        "o (chars)",
        "t_chat (ms)",
        "t_lopt_mp (ms)",
        "t_dedup (ms)",
        "t_lopt_e2e (ms)",
        "t_native_e2e (ms)",
        "t_native_tok (ms)",
        "精度",
        "回退",
        "状态",
    ]
    for family in sorted(family_groups, key=lambda item: FAMILY_ORDER.get(item, 99)):
        language_sections = []
        for language in sorted(
            family_groups[family],
            key=lambda item: LANGUAGE_ORDER.get(item, 99),
        ):
            case_sections = []
            for case_key in sorted(family_groups[family][language], key=sort_case_key):
                records = sorted(
                    grouped[case_key],
                    key=lambda item: (
                        int(item["worker_processes"]),
                        int(item["chunk_count"]),
                        int(item["overlap_chars"]),
                    ),
                )
                best_record = best_by_case.get(case_key)
                replay_record = replay_by_case.get(case_key)
                worker_values = sorted({int(item["worker_processes"]) for item in records})
                summary_bits = [
                    f"候选数: {fmt_int(len(records))}",
                    f"测试过的 worker: {', '.join(str(v) for v in worker_values)}",
                    f"回退数: {fmt_int(sum(bool(item.get('fallback_used')) for item in records))}",
                    f"Exact Match 数: {fmt_int(sum(bool(item.get('exact_match')) for item in records))}",
                ]
                best_rows = []
                if best_record is not None:
                    best_rows.append(
                        [
                            "Search 最优",
                            fmt_int(best_record["worker_processes"]),
                            fmt_int(best_record["chunk_count"]),
                            fmt_int(best_record["overlap_chars"]),
                            fmt_ms(best_record["lopt_e2e_time_ms"]),
                            fmt_x(best_record["e2e_speedup_x"]),
                            fmt_x(best_record["tokenizer_speedup_x"]),
                        ]
                    )
                if replay_record is not None:
                    best_rows.append(
                        [
                            "Replay 最终",
                            fmt_int(replay_record["worker_processes"]),
                            fmt_int(replay_record["chunk_count"]),
                            fmt_int(replay_record["overlap_chars"]),
                            fmt_ms(replay_record["lopt_e2e_time_ms"]),
                            fmt_x(replay_record["e2e_speedup_x"]),
                            fmt_x(replay_record["tokenizer_speedup_x"]),
                        ]
                    )
                candidate_rows = []
                for record in records:
                    candidate_rows.append(
                        [
                            fmt_int(record["worker_processes"]),
                            fmt_int(record["chunk_count"]),
                            fmt_int(record["chunk_chars"]),
                            fmt_int(record["overlap_chars"]),
                            fmt_ms(record["chat_template_time_ms"]),
                            fmt_ms(record["mp_dispatch_process_collect_time_ms"]),
                            fmt_ms(record["chunk_dedup_time_ms"]),
                            fmt_ms(record["lopt_e2e_time_ms"]),
                            fmt_ms(record["native_e2e_time_ms"]),
                            fmt_ms(record["native_tokenizer_time_ms"]),
                            "Exact Match" if record["exact_match"] else "Mismatch",
                            "是" if record["fallback_used"] else "否",
                            escape(record["candidate_status"]),
                        ]
                    )
                case_sections.append(
                    f"""
                    <details class="subfold" id="{details_id('case', case_key)}">
                      <summary>{case_key.length_label} / {fmt_int(case_key.input_chars)} chars</summary>
                      <div class="body">
                        <div class="chip-row">{''.join(f'<span class="chip">{escape(bit)}</span>' for bit in summary_bits)}</div>
                        {render_table(
                            ["视图", "p (proc)", "k (count)", "o (chars)", "LoPT E2E (ms)", "E2E 加速比 (x)", "Tokenizer 加速比 (x)"],
                            best_rows or [["-", "-", "-", "-", "-", "-", "-"]],
                            class_name="dense-table",
                        )}
                        {render_table(candidate_headers, candidate_rows, class_name="dense-table")}
                      </div>
                    </details>
                    """
                )
            language_sections.append(
                f"""
                <details class="fold">
                  <summary>{family_language_label(family, language)}</summary>
                  <div class="body">
                    {''.join(case_sections)}
                  </div>
                </details>
                """
            )
        family_sections.append("".join(language_sections))
    return "".join(family_sections)


def render_full_data_explorer(
    detail_records: list[dict[str, Any]],
    replay_records: list[dict[str, Any]],
) -> str:
    explorer_info = [
        f"完整候选行数: {fmt_int(len(detail_records))}",
        f"最终 replay 行数: {fmt_int(len(replay_records))}",
        "所有数据均已嵌入当前 HTML 文件",
        "可使用筛选、关键字搜索和分页浏览",
    ]

    detail_headers = [
        "Idx",
        "模型",
        "语言",
        "长度",
        "c_in (chars)",
        "p (proc)",
        "k (count)",
        "chunk 大小 (chars)",
        "o (chars)",
        "t_chat (ms)",
        "t_lopt_mp (ms)",
        "t_dedup (ms)",
        "t_lopt_e2e (ms)",
        "t_native_e2e (ms)",
        "t_native_tok (ms)",
        "S_e2e (x)",
        "S_tok (x)",
        "D_tok (%)",
        "精度",
        "回退",
        "状态",
        "JSON",
    ]
    replay_headers = [
        "Idx",
        "模型",
        "语言",
        "长度",
        "c_in (chars)",
        "n_out (tok)",
        "p (proc)",
        "k (count)",
        "o (chars)",
        "t_chat (ms)",
        "t_lopt_mp (ms)",
        "t_dedup (ms)",
        "t_lopt_e2e (ms)",
        "t_native_e2e (ms)",
        "t_native_tok (ms)",
        "S_e2e (x)",
        "S_tok (x)",
        "D_tok (%)",
        "精度",
        "JSON",
    ]

    return f"""
    <section class="section" id="data-browser">
      <div class="section-head">
        <h2>10 · 全量数据浏览器</h2>
        <p>在单个 HTML 文件中完整浏览全部 search candidate 与 final replay 数据，适合复核和追溯。</p>
      </div>
      <div class="chip-row">{''.join(f'<span class="chip">{escape(item)}</span>' for item in explorer_info)}</div>

      <details class="fold" open>
        <summary>Search candidate 浏览器 ({fmt_int(len(detail_records))} 行)</summary>
        <div class="body">
          <div class="filter-bar">
            <label>模型
              <select id="detail-family-filter"></select>
            </label>
            <label>语言
              <select id="detail-language-filter"></select>
            </label>
            <label>长度
              <select id="detail-length-filter"></select>
            </label>
            <label>状态
              <select id="detail-status-filter"></select>
            </label>
            <label>搜索
              <input id="detail-search" type="search" placeholder="模型 / 长度 / worker / 状态">
            </label>
            <label>每页行数
              <select id="detail-page-size">
                <option value="25">25</option>
                <option value="50" selected>50</option>
                <option value="100">100</option>
                <option value="200">200</option>
              </select>
            </label>
          </div>
          <p class="table-note">可见列聚焦于 benchmark 关键指标。点击 <code>View</code> 可查看完整嵌入式 JSON，包括 token hash、fallback 耗时、error 文本以及实际 chunk 元数据。</p>
          <div class="pager">
            <button type="button" id="detail-prev">上一页</button>
            <span id="detail-page-info" class="muted"></span>
            <button type="button" id="detail-next">下一页</button>
          </div>
          <div class="table-wrap">
            <table class="dense-table browser-table" id="detail-table">
              <thead><tr>{"".join(f"<th>{header}</th>" for header in detail_headers)}</tr></thead>
              <tbody></tbody>
            </table>
          </div>
        </div>
      </details>

      <details class="fold">
        <summary>Final replay 浏览器 ({fmt_int(len(replay_records))} 行)</summary>
        <div class="body">
          <div class="filter-bar">
            <label>模型
              <select id="replay-family-filter"></select>
            </label>
            <label>语言
              <select id="replay-language-filter"></select>
            </label>
            <label>长度
              <select id="replay-length-filter"></select>
            </label>
            <label>搜索
              <input id="replay-search" type="search" placeholder="模型 / 长度 / worker">
            </label>
            <label>每页行数
              <select id="replay-page-size">
                <option value="12">12</option>
                <option value="24" selected>24</option>
                <option value="48">48</option>
              </select>
            </label>
          </div>
          <p class="table-note">Final replay 行是 full search 后选出的 exact-match 最优配置。JSON 查看器会展示用于精度校验的 hash 和辅助字段。</p>
          <div class="pager">
            <button type="button" id="replay-prev">上一页</button>
            <span id="replay-page-info" class="muted"></span>
            <button type="button" id="replay-next">下一页</button>
          </div>
          <div class="table-wrap">
            <table class="dense-table browser-table" id="replay-table">
              <thead><tr>{"".join(f"<th>{header}</th>" for header in replay_headers)}</tr></thead>
              <tbody></tbody>
            </table>
          </div>
        </div>
      </details>

      <div class="json-viewer" id="json-viewer" hidden>
        <div class="json-viewer-backdrop" data-close-json="1"></div>
        <div class="json-viewer-panel" role="dialog" aria-modal="true" aria-labelledby="json-viewer-title">
          <div class="json-viewer-head">
            <div>
              <div class="json-viewer-kicker">嵌入式记录数据</div>
              <h3 id="json-viewer-title">记录 JSON</h3>
            </div>
            <div class="json-viewer-actions">
              <button type="button" id="json-copy">复制 JSON</button>
              <button type="button" id="json-close" data-close-json="1">关闭</button>
            </div>
          </div>
          <pre id="json-viewer-body"></pre>
        </div>
      </div>
    </section>
    """


def render_v2_artifacts(ctx: dict[str, Any]) -> str:
    v2_summary = summarize_replay_records(ctx["replay_records"])
    best = v2_summary.get("max_speedup")
    worst = v2_summary.get("min_speedup")
    chips = [
        f"case 数: {fmt_int(v2_summary.get('case_count'))}",
        f"平均原生 E2E: {fmt_ms(v2_summary.get('avg_native'))}",
        f"平均 LoPT E2E: {fmt_ms(v2_summary.get('avg_lopt'))}",
        f"平均加速比: {fmt_x(v2_summary.get('avg_speedup'))}",
        f"Exact Match: {fmt_int(v2_summary.get('exact_count'))}/{fmt_int(v2_summary.get('case_count'))}",
    ]
    if best:
        chips.append(
            f"最佳 case: {best['tokenizer_family']} / {language_label(best['language'])} / {best['length_label']} ({fmt_x(best['e2e_speedup_x'])})"
        )
    if worst:
        chips.append(
            f"最弱 case: {worst['tokenizer_family']} / {language_label(worst['language'])} / {worst['length_label']} ({fmt_x(worst['e2e_speedup_x'])})"
        )
    rows = []
    for record in ctx["replay_records"]:
        rows.append(
            [
                escape(record["tokenizer_family"]),
                escape(language_label(record["language"])),
                escape(record["length_label"]),
                fmt_int(record["worker_processes"]),
                fmt_int(record["chunk_count"]),
                fmt_int(record["overlap_chars"]),
                fmt_ms(record.get("dispatch_submit_time_ms")),
                fmt_ms(record.get("dispatch_collect_time_ms")),
                fmt_ms(record.get("worker_encode_time_ms_sum")),
                fmt_ms(record.get("worker_encode_time_ms_max")),
                fmt_ms(record.get("worker_materialize_time_ms_sum")),
                fmt_ms(record.get("worker_materialize_time_ms_max")),
                fmt_ms(record["native_e2e_time_ms"]),
                fmt_ms(record["native_tokenizer_time_ms"]),
                fmt_ms(record["chat_template_time_ms"]),
                fmt_ms(record["mp_dispatch_process_collect_time_ms"]),
                fmt_ms(record["chunk_dedup_time_ms"]),
                fmt_ms(record["lopt_e2e_time_ms"]),
                fmt_x(record["e2e_speedup_x"]),
                fmt_x(record["tokenizer_speedup_x"]),
                fmt_pct(record["tokenizer_time_drop_pct"]),
                "Exact Match" if record["exact_match"] else "Mismatch",
            ]
        )
    return f"""
    <section class="section" id="v2-summary">
      <div class="section-head">
        <h2>06 · v2.1 版本摘要</h2>
        <p>单独补充 v2.1 的完整结果和最优参数，便于和 v1 的旧版报告做直接比较。</p>
      </div>
      <div class="chip-row">{''.join(f'<span class="chip">{escape(item)}</span>' for item in chips)}</div>
      <div class="chart-grid">
        {render_svg_dual_line_chart(ctx["replay_records"], "lopt_e2e_time_ms", "native_e2e_time_ms", "v2.1 最优配置原生 vs LoPT E2E 曲线", "原生 E2E 与 LoPT E2E (ms)")}
        {render_svg_line_chart(ctx["replay_records"], "e2e_speedup_x", "v2.1 最优配置 E2E 加速比曲线", "E2E 加速比 (x)", "x")}
      </div>
      <details class="fold" open>
        <summary>v2.1 最优配置总表</summary>
        <div class="body">
          {render_table(
              [
                  "模型",
                  "语言",
                  "长度",
                  "p (proc)",
                  "k (count)",
                  "o (chars)",
                  "t_submit (ms)",
                  "t_collect (ms)",
                  "worker_encode_sum (ms)",
                  "worker_encode_max (ms)",
                  "worker_mat_sum (ms)",
                  "worker_mat_max (ms)",
                  "原生 E2E (ms)",
                  "原生 Tokenizer (ms)",
                  "LoPT t_chat (ms)",
                  "LoPT t_lopt_mp (ms)",
                  "LoPT t_dedup (ms)",
                  "LoPT E2E (ms)",
                  "E2E 加速比 (x)",
                  "Tokenizer 加速比 (x)",
                  "Tokenizer 降幅 (%)",
                  "精度",
              ],
              rows,
              class_name="dense-table",
          )}
        </div>
      </details>
    </section>
    """


def render_callout(title: str, points: list[str], kind: str = "info") -> str:
    items = "".join(f"<li>{escape(point)}</li>" for point in points)
    return f"""
    <div class="callout {kind}">
      <div class="callout-title">{escape(title)}</div>
      <ul>{items}</ul>
    </div>
    """


def build_report_context(args: argparse.Namespace) -> dict[str, Any]:
    flow_doc = read_text(args.flow_doc)
    detail_records = sorted(
        (
            enrich_detail_record(record)
            for record in load_jsonl(args.search_detail_jsonl)
        ),
        key=lambda record: (
            FAMILY_ORDER.get(record["tokenizer_family"], 99),
            LANGUAGE_ORDER.get(record["language"], 99),
            int(record["input_chars"]),
            int(record["worker_processes"]),
            int(record["chunk_count"]),
            int(record["overlap_chars"]),
        ),
    )
    worker_best_records = sorted(
        load_json(args.worker_best_json),
        key=lambda record: (
            FAMILY_ORDER.get(record["tokenizer_family"], 99),
            LANGUAGE_ORDER.get(record["language"], 99),
            int(record["input_chars"]),
            int(record["worker_processes"]),
        ),
    )
    best_records = sorted(
        load_json(args.best_json),
        key=lambda record: (
            FAMILY_ORDER.get(record["tokenizer_family"], 99),
            LANGUAGE_ORDER.get(record["language"], 99),
            int(record["input_chars"]),
        ),
    )
    v1_best_path = getattr(args, "v1_best_json", None)
    v1_replay_path = getattr(args, "v1_replay_json", None)
    v1_best_records = sorted(
        load_json(v1_best_path),
        key=lambda record: (
            FAMILY_ORDER.get(record["tokenizer_family"], 99),
            LANGUAGE_ORDER.get(record["language"], 99),
            int(record["input_chars"]),
        ),
    ) if v1_best_path and Path(v1_best_path).exists() else []
    v1_replay_records = sorted(
        load_json(v1_replay_path),
        key=lambda record: (
            FAMILY_ORDER.get(record["tokenizer_family"], 99),
            LANGUAGE_ORDER.get(record["language"], 99),
            int(record["input_chars"]),
        ),
    ) if v1_replay_path and Path(v1_replay_path).exists() else []
    replay_records = sorted(
        load_json(args.replay_json),
        key=lambda record: (
            FAMILY_ORDER.get(record["tokenizer_family"], 99),
            LANGUAGE_ORDER.get(record["language"], 99),
            int(record["input_chars"]),
        ),
    )
    search_meta = load_json(args.search_meta_json) if args.search_meta_json else {}
    corpus_sources = load_corpus_sources(args.corpus_meta_dir)
    env_info = load_env_info(args.env_info_json)

    best_by_case = {record_case_key(record): record for record in best_records}
    replay_by_case = {record_case_key(record): record for record in replay_records}
    v1_replay_by_case = {record_case_key(record): record for record in v1_replay_records}

    candidate_counter = len(detail_records)
    fallback_count = sum(bool(record.get("fallback_used")) for record in detail_records)
    exact_count = sum(bool(record.get("exact_match")) for record in detail_records)
    valid_count = sum(record.get("candidate_status") == "valid" for record in detail_records)
    mismatch_count = candidate_counter - exact_count

    max_e2e_record = max(replay_records, key=lambda record: float(record["e2e_speedup_x"]))
    max_tok_record = max(
        replay_records,
        key=lambda record: float(record["tokenizer_speedup_x"]),
    )
    max_drop_record = max(
        replay_records,
        key=lambda record: float(record["tokenizer_time_drop_pct"]),
    )
    comparison = compare_replay_records(v1_replay_records, replay_records)

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "flow_doc": flow_doc,
        "ascii_flow": extract_ascii_flow(flow_doc),
        "source_files": extract_key_source_lines(flow_doc),
        "detail_records": detail_records,
        "worker_best_records": worker_best_records,
        "best_records": best_records,
        "best_by_case": best_by_case,
        "v1_best_records": v1_best_records,
        "v1_replay_records": v1_replay_records,
        "v1_replay_by_case": v1_replay_by_case,
        "replay_records": replay_records,
        "replay_by_case": replay_by_case,
        "comparison": comparison,
        "search_meta": search_meta,
        "corpus_sources": corpus_sources,
        "env_info": env_info,
        "candidate_counter": candidate_counter,
        "valid_count": valid_count,
        "fallback_count": fallback_count,
        "exact_count": exact_count,
        "mismatch_count": mismatch_count,
        "max_e2e_record": max_e2e_record,
        "max_tok_record": max_tok_record,
        "max_drop_record": max_drop_record,
    }


def render_summary_cards(ctx: dict[str, Any], args: argparse.Namespace) -> str:
    case_count = len(ctx["replay_records"])
    exact_rate = 0.0 if ctx["candidate_counter"] == 0 else (ctx["exact_count"] / ctx["candidate_counter"]) * 100.0
    best_e2e = ctx["max_e2e_record"]
    best_tok = ctx["max_tok_record"]
    best_drop = ctx["max_drop_record"]
    comparison = ctx.get("comparison") or {}
    avg_speedup = comparison.get("avg_speedup")
    avg_delta_ms = comparison.get("avg_delta_ms")
    cards = [
        (
            "全局最佳 E2E 加速比",
            fmt_x(best_e2e["e2e_speedup_x"]),
            f"{best_e2e['tokenizer_family']} / {language_label(best_e2e['language'])} / {best_e2e['length_label']}",
        ),
        (
            "全局最佳 Tokenizer 加速比",
            fmt_x(best_tok["tokenizer_speedup_x"]),
            f"{best_tok['tokenizer_family']} / {language_label(best_tok['language'])} / {best_tok['length_label']}",
        ),
        (
            "最大 Tokenizer 降幅",
            fmt_pct(best_drop["tokenizer_time_drop_pct"]),
            f"{best_drop['tokenizer_family']} / {language_label(best_drop['language'])} / {best_drop['length_label']}",
        ),
        (
            "Exact Match 比例",
            fmt_pct(exact_rate),
            f"{fmt_int(ctx['exact_count'])} / {fmt_int(ctx['candidate_counter'])} 个候选配置",
        ),
        (
            "Search 候选总数",
            fmt_int(ctx["candidate_counter"]),
            f"Valid: {fmt_int(ctx['valid_count'])} · Fallback: {fmt_int(ctx['fallback_count'])}",
        ),
        (
            "最终 replay case 数",
            fmt_int(case_count),
            "2 个模型族 · 2 种语言 · 12 个输入长度",
        ),
        (
            "v2.1 对 v1 平均提速",
            fmt_x(avg_speedup),
            f"平均节省 {fmt_ms(avg_delta_ms)} ms / case",
        ),
    ]
    return "".join(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{escape(label)}</div>
          <div class="kpi-value">{escape(value)}</div>
          <div class="kpi-desc">{escape(desc)}</div>
        </div>
        """
        for label, value, desc in cards
    )


def render_setup_table(args: argparse.Namespace, ctx: dict[str, Any]) -> str:
    search_meta = ctx["search_meta"]
    detail_records = ctx["detail_records"]
    env_info = ctx.get("env_info") or {}
    derived_families = sorted(
        {record["tokenizer_family"] for record in detail_records},
        key=lambda item: FAMILY_ORDER.get(item, 99),
    )
    derived_languages = sorted(
        {record["language"] for record in detail_records},
        key=lambda item: LANGUAGE_ORDER.get(item, 99),
    )
    length_map: dict[str, int] = {}
    for record in detail_records:
        length_map.setdefault(record["length_label"], int(record["input_chars"]))
    derived_lengths = [
        label for label, _ in sorted(length_map.items(), key=lambda item: item[1])
    ]
    derived_workers = sorted({int(record["worker_processes"]) for record in detail_records})
    derived_chunks = sorted({int(record["chunk_count"]) for record in detail_records})
    derived_overlaps = sorted({int(record["overlap_chars"]) for record in detail_records})
    cpuset_count = parse_cpuset_count(args.cpu_affinity)
    memory_total_human = env_info.get("memory_total_human") or "-"
    memory_available_human = env_info.get("memory_available_human") or "-"
    hardware_summary = env_info.get("hardware_summary") or {}
    cpu_max_mhz = hardware_summary.get("cpu_max_mhz")
    cpu_max_label = f"{cpu_max_mhz} MHz" if cpu_max_mhz not in (None, "-") else "-"
    env_extra_text = "\n\n".join(
        filter(
            None,
            [
                env_info.get("free_h_raw", "").strip(),
                env_info.get("meminfo_raw", "").strip(),
                env_info.get("numactl_raw", "").strip(),
            ],
        )
    )
    rows = [
        ["本地最新 vLLM 源码", escape(str(args.vllm_src))],
        ["远端 search 输出目录", escape(", ".join(str(path) for path in args.search_input_dirs))],
        ["Benchmark 机器", escape(env_info.get("hostname") or "未提供")],
        ["CPU 架构 / 厂商", escape(f"{hardware_summary.get('architecture', '-')} / {hardware_summary.get('vendor_id', '-')}")],
        ["逻辑 CPU 总数", escape(str(hardware_summary.get("cpu_count", "-")))],
        ["NUMA 节点数", escape(str(hardware_summary.get("numa_nodes", "-")))],
        ["CPU 最高频率", escape(cpu_max_label)],
        ["机器总内存", escape(memory_total_human)],
        ["机器可用内存", escape(memory_available_human)],
        ["CPU 绑核范围", escape(args.cpu_affinity)],
        ["绑核 CPU 数", escape(str(cpuset_count))],
        ["模型族", escape(", ".join(search_meta.get("selected_families") or derived_families))],
        ["语言", escape(", ".join(language_label(lang) for lang in (search_meta.get("languages") or derived_languages)))],
        ["输入长度", escape(", ".join(search_meta.get("lengths") or derived_lengths))],
        ["Worker 搜索空间 p (proc)", escape(", ".join(str(v) for v in (search_meta.get("worker_values") or derived_workers)))],
        ["Chunk 数搜索空间 k (count)", escape(", ".join(str(v) for v in (search_meta.get("chunk_counts") or search_meta.get("chunk_multipliers") or derived_chunks)))],
        ["Overlap 搜索空间 o (chars)", escape(", ".join(str(v) for v in (search_meta.get("overlap_values") or derived_overlaps)))],
        ["固定 min_match_tokens", "2"],
        ["失败处理策略", "直接回退到原生串行 Tokenizer 逻辑（最终有效候选中未触发）"],
        ["语料策略", "使用纯中文与纯英文真实网页语料；当原始抓取不足时，通过循环真实网页文本块扩展到 1024k chars"],
    ]
    env_sections = []
    if env_info.get("lscpu_raw"):
        env_sections.append(
            f"""
            <details class="fold">
              <summary>远端机器环境原始信息</summary>
              <div class="body">
                <p class="table-note">以下内容直接来自实际 benchmark 机器的 <code>lscpu</code>、<code>free -h</code>、<code>/proc/meminfo</code> 和 <code>numactl --hardware</code> 输出。</p>
                <div class="diagram-grid">
                  {render_ascii_block("lscpu", env_info.get("lscpu_raw", ""), accent="blue")}
                  {render_ascii_block("free -h / meminfo / numactl", env_extra_text, accent="teal")}
                </div>
              </div>
            </details>
            """
        )
    return render_table(["配置项", "取值"], rows) + "".join(env_sections)


def render_source_file_list(source_files: list[str]) -> str:
    items = "".join(f"<li><code>{escape(path)}</code></li>" for path in source_files)
    return f"<ul class=\"source-list\">{items}</ul>"


def render_html(args: argparse.Namespace, ctx: dict[str, Any]) -> str:
    native_ascii = ctx["ascii_flow"]
    native_bottleneck_ascii = "\n".join(
        [
            "单个超长 prompt",
            "    |",
            "    v",
            "AsyncMicrobatchTokenizer.encode()",
            "    |",
            "    +--> async queue / same-kwargs microbatch",
            "    |",
            "    v",
            "共享 ThreadPoolExecutor",
            "    |",
            "    v",
            "当前请求最终仍只落到一次底层 tokenizer 调用",
            "",
            "结果：请求间并发不错，",
            "但单个 1M prompt 内部没有被切分并行。",
        ]
    )
    lopt_v1_ascii = """
User text prompt
    |
    v
LoPT v1: split(text, k chunks, overlap=o chars)
    |
    +--> Chunk[0] --------------+
    +--> Chunk[1] --------------+--> ProcessPoolExecutor(p workers)
    +--> ... -------------------+
    +--> Chunk[k-1] ------------+
                |
                v
      per-chunk token IDs + offsets
                |
                v
   parent overlap anchor match + dedup
                |
                v
         merged final token IDs
    """.strip()
    lopt_v2_ascii = """
User text prompt
    |
    v
LoPT v2.1: split(text, k chunks, overlap=o chars)
    |
    +--> Chunk[0] --------------+
    +--> Chunk[1] --------------+--> ProcessPoolExecutor(p workers)
    +--> ... -------------------+
    +--> Chunk[k-1] ------------+
                |
                v
      per-chunk token IDs + offsets
                |
                v
   parent dispatch/collect bookkeeping
        |      |      |
        |      |      +--> t_return_tail / t_lag_max
        |      +--> t_collect_child
        +--> t_lopt_mp
                |
                v
   parent overlap anchor match + dedup
                |
                v
         merged final token IDs
    """.strip()
    lopt_v2_diff_points = [
        "v2.1 仍然保留 v1 的 split / parallel / overlap dedup 主骨架，不改变最终 token IDs 的 exact-match 目标。",
        "v2.1 把父进程侧的调度与收集拆成可量化阶段，显式记录 submit / collect / child compute / return tail / receive lag。",
        "v2.1 引入绝对字符偏移视角，便于定位边界 token 的对齐和去冗余问题。",
        "v2.1 更适合做瓶颈诊断与持续优化；v1 更适合做已知最佳参数的稳定回放基线。",
    ]
    overlap_dedup_ascii = """
Chunk A token IDs + offsets          Chunk B token IDs + offsets
           |                                     |
           +------------------+------------------+
                              v
              父进程只在全局字符区间对齐时
              才比较两个 token
                              |
                              v
                  找到最长有效 overlap anchor
                              |
                              v
                 保留 A-left + overlap + B-right

精度一致条件：
  token ID 相同 且 全局字符区间相同
    """.strip()

    principle_paragraphs = "".join(
        f"<p>{escape(paragraph)}</p>" for paragraph in build_principle_text()
    )
    background_callout = render_callout("业务背景", build_background_points(), kind="info")
    bottleneck_callout = render_callout("原生方案瓶颈分析", build_native_bottleneck_points(), kind="warn")
    report_title = escape(args.report_title)
    report_subtitle = escape(args.report_subtitle)
    source_files_html = render_source_file_list(ctx["source_files"])
    best_case_summary_html = render_best_case_summary(ctx["replay_records"])
    v2_artifacts_html = render_v2_artifacts(ctx)
    version_comparison_html = render_version_comparison(ctx)
    best_overview_html = render_best_overview(ctx["replay_records"])
    worker_best_html = render_worker_best(ctx["worker_best_records"])
    candidate_detail_html = render_candidate_details(
        ctx["detail_records"],
        ctx["best_by_case"],
        ctx["replay_by_case"],
    )
    source_panel_html = render_source_panel(ctx["corpus_sources"])
    data_browser_html = render_full_data_explorer(
        ctx["detail_records"],
        ctx["replay_records"],
    )
    detail_json_payload = json_script_payload(ctx["detail_records"])
    replay_json_payload = json_script_payload(ctx["replay_records"])

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{report_title}</title>
  <style>
    :root {{
      --bg: #ffffff;
      --panel: #fbfdff;
      --line: #d9e2ec;
      --line-strong: #b8c8d9;
      --ink: #102033;
      --muted: #5f7286;
      --faint: #8696a6;
      --blue: #1f6feb;
      --blue-soft: #eaf2ff;
      --teal: #0f9f9a;
      --teal-soft: #e8f7f6;
      --orange: #f59e0b;
      --orange-soft: #fff4df;
      --green: #13835b;
      --green-soft: #ebfaf3;
      --shadow: 0 20px 40px rgba(16, 32, 51, 0.08);
      --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      --sans: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: var(--sans);
      line-height: 1.58;
    }}
    .page {{
      max-width: 1540px;
      margin: 0 auto;
      padding: 42px 42px 80px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: minmax(0, 1.5fr) minmax(320px, 0.85fr);
      gap: 26px;
      margin-bottom: 28px;
    }}
    .hero-main, .hero-side, .section {{
      border: 1px solid var(--line);
      background: linear-gradient(180deg, #fff, var(--panel));
      box-shadow: var(--shadow);
    }}
    .hero-main {{
      padding: 34px 36px;
      position: relative;
      overflow: hidden;
    }}
    .hero-main::after {{
      content: "";
      position: absolute;
      right: -120px;
      top: -120px;
      width: 360px;
      height: 360px;
      border-radius: 50%;
      border: 1px solid rgba(31, 111, 235, 0.14);
      background: radial-gradient(circle, rgba(15, 159, 154, 0.09), transparent 64%);
      pointer-events: none;
    }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--blue);
      font-size: 13px;
      font-weight: 800;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin-bottom: 14px;
    }}
    .eyebrow::before {{
      content: "";
      width: 28px;
      height: 2px;
      background: var(--blue);
    }}
    h1 {{
      margin: 0;
      font-size: clamp(32px, 4vw, 56px);
      line-height: 1.02;
      letter-spacing: 0;
      font-weight: 850;
      max-width: 950px;
    }}
    .subtitle {{
      margin: 18px 0 0;
      color: var(--muted);
      font-size: 17px;
      max-width: 920px;
    }}
    .hero-meta {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 26px;
    }}
    .meta-item {{
      padding: 12px 13px;
      background: #f5f9fc;
      border: 1px solid var(--line);
    }}
    .meta-item .label {{
      display: block;
      color: var(--faint);
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .meta-item .value {{
      display: block;
      margin-top: 4px;
      color: var(--ink);
      font-size: 13px;
      font-family: var(--mono);
      overflow-wrap: anywhere;
    }}
    .hero-side {{
      padding: 24px;
      display: grid;
      gap: 12px;
    }}
    .kpi-card {{
      border: 1px solid var(--line);
      background: #fff;
      padding: 18px;
      min-height: 126px;
    }}
    .kpi-label {{
      color: var(--faint);
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 10px;
    }}
    .kpi-value {{
      color: var(--blue);
      font-size: 34px;
      font-weight: 850;
      line-height: 1;
      letter-spacing: 0;
      margin-bottom: 8px;
    }}
    .kpi-desc {{
      color: var(--muted);
      font-size: 13px;
    }}
    .nav {{
      position: sticky;
      top: 0;
      z-index: 5;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 12px 0 20px;
      background: rgba(255, 255, 255, 0.92);
      backdrop-filter: blur(10px);
    }}
    .nav a, .toolbar button {{
      border: 1px solid var(--line);
      background: #fff;
      color: #37506a;
      padding: 8px 12px;
      font-size: 13px;
      font-weight: 800;
      text-decoration: none;
      cursor: pointer;
    }}
    .nav a:hover, .toolbar button:hover {{
      border-color: var(--blue);
      color: var(--blue);
    }}
    .toolbar {{
      display: flex;
      gap: 8px;
      justify-content: flex-end;
      margin-bottom: 14px;
    }}
    .pager button,
    .table-button,
    .json-viewer-actions button {{
      border: 1px solid var(--line);
      background: #fff;
      color: #37506a;
      padding: 7px 10px;
      font-size: 12px;
      font-weight: 800;
      cursor: pointer;
    }}
    .pager button:hover,
    .table-button:hover,
    .json-viewer-actions button:hover {{
      border-color: var(--blue);
      color: var(--blue);
    }}
    .pager button:disabled {{
      opacity: 0.45;
      cursor: not-allowed;
    }}
    .section {{
      margin-top: 24px;
      padding: 28px;
    }}
    .section-head {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 24px;
      padding-bottom: 16px;
      margin-bottom: 18px;
      border-bottom: 1px solid var(--line);
    }}
    h2 {{
      margin: 0;
      font-size: 24px;
      line-height: 1.18;
      letter-spacing: 0;
    }}
    .section-head p {{
      margin: 0;
      max-width: 720px;
      color: var(--muted);
      font-size: 14px;
      text-align: right;
    }}
    p {{ margin: 0 0 12px; }}
    code {{
      font-family: var(--mono);
      background: #f5f9fc;
      padding: 1px 5px;
      font-size: 12px;
      border-radius: 3px;
    }}
    .callout {{
      border: 1px solid var(--line);
      border-left-width: 4px;
      padding: 16px 18px;
      margin: 12px 0 18px;
      background: #fff;
    }}
    .callout.info {{
      border-left-color: var(--blue);
      background: #f6faff;
    }}
    .callout.warn {{
      border-left-color: var(--orange);
      background: #fffaf1;
    }}
    .callout.ok {{
      border-left-color: var(--green);
      background: #f5fcf9;
    }}
    .callout-title {{
      font-weight: 800;
      margin-bottom: 10px;
    }}
    .callout ul {{
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
    }}
    .diagram-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-top: 16px;
    }}
    .ascii-card {{
      border: 1px solid var(--line);
      background: #fff;
      overflow: hidden;
    }}
    .ascii-card.blue .ascii-title {{
      background: var(--blue-soft);
      color: var(--blue);
    }}
    .ascii-card.teal .ascii-title {{
      background: var(--teal-soft);
      color: var(--teal);
    }}
    .ascii-card .ascii-title {{
      padding: 10px 14px;
      font-size: 13px;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .ascii-card pre {{
      margin: 0;
      padding: 14px;
      background: #f8fbfd;
      border-top: 1px solid var(--line);
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.48;
      overflow: auto;
      white-space: pre-wrap;
    }}
    .chip-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 14px;
    }}
    .chip {{
      display: inline-block;
      border: 1px solid var(--line);
      background: #fff;
      padding: 4px 8px;
      font-size: 12px;
      color: var(--muted);
      font-family: var(--mono);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      margin: 12px 0 16px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 7px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f4f8fb;
      color: var(--ink);
      font-weight: 750;
      white-space: nowrap;
    }}
    tr:hover td {{
      background: #fbfdff;
    }}
    .dense-table th, .dense-table td {{
      padding: 6px 8px;
      font-size: 12px;
    }}
    .filter-bar {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }}
    .filter-bar label {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
    }}
    .filter-bar select,
    .filter-bar input {{
      width: 100%;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      padding: 8px 10px;
      font: inherit;
    }}
    .pager {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      margin: 10px 0 12px;
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--line);
      background: #fff;
    }}
    .table-wrap table {{
      margin: 0;
      min-width: 1600px;
    }}
    .browser-table th {{
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    .table-note {{
      color: var(--muted);
      font-size: 12px;
      margin: 0 0 10px;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin: 14px 0;
    }}
    .chart-grid.single {{
      grid-template-columns: 1fr;
    }}
    .wide-scroll {{
      overflow-x: auto;
    }}
    .chart-card {{
      border: 1px solid var(--line);
      background: #fff;
      padding: 16px;
    }}
    .chart-head {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 14px;
      margin-bottom: 12px;
    }}
    .chart-title {{
      font-size: 15px;
      font-weight: 800;
      color: var(--ink);
      margin-bottom: 4px;
    }}
    .chart-subtitle {{
      font-size: 12px;
      color: var(--muted);
    }}
    .chart-legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
      justify-content: flex-end;
    }}
    .chart-legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
      font-size: 12px;
      color: var(--muted);
      white-space: nowrap;
    }}
    .chart-line {{
      display: inline-block;
      width: 22px;
      height: 0;
      border-top: 3px solid var(--green);
    }}
    .chart-line.dashed {{
      border-top-style: dashed;
      border-top-color: var(--orange);
    }}
    .metric-topology {{
      margin-top: 6px;
      justify-content: flex-end;
    }}
    .chart-swatch {{
      display: inline-block;
      width: 12px;
      height: 12px;
      border: 1px solid rgba(16, 32, 51, 0.14);
    }}
    .mini-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-top: 12px;
    }}
    .mini-card {{
      border: 1px solid var(--line);
      background: #fbfdff;
      padding: 14px;
    }}
    .mini-card-head {{
      margin-bottom: 8px;
    }}
    .mini-card-title {{
      font-size: 14px;
      font-weight: 800;
      color: var(--ink);
      margin-bottom: 4px;
    }}
    .mini-card-subtitle {{
      font-size: 12px;
      color: var(--muted);
    }}
    .chip-row.compact {{
      margin-bottom: 8px;
    }}
    .mini-svg-wrap {{
      overflow-x: auto;
      border-top: 1px solid #edf3f8;
      padding-top: 8px;
      margin-top: 8px;
    }}
    .mono-cell {{
      font-family: var(--mono);
      white-space: nowrap;
    }}
    .soft-wrap {{
      overflow-wrap: anywhere;
    }}
    .legend-table td:first-child,
    .legend-table td:nth-child(2) {{
      white-space: nowrap;
    }}
    .metric-badge {{
      display: inline-block;
      min-width: 68px;
      text-align: center;
      padding: 3px 8px;
      border: 1px solid var(--line);
      font-weight: 800;
      font-family: var(--mono);
    }}
    .metric-badge.good {{
      color: var(--green);
      background: var(--green-soft);
      border-color: #c6ecd8;
    }}
    .metric-badge.ok {{
      color: var(--teal);
      background: var(--teal-soft);
      border-color: #c7ebe8;
    }}
    .metric-badge.warn {{
      color: #915d00;
      background: var(--orange-soft);
      border-color: #ffe0a3;
    }}
    .metric-badge.neutral {{
      color: var(--muted);
      background: #fff;
    }}
    details {{
      border: 1px solid var(--line);
      background: #fff;
      margin: 10px 0;
    }}
    summary {{
      cursor: pointer;
      padding: 14px 16px;
      font-weight: 800;
      list-style: none;
      position: relative;
      color: var(--ink);
      background: linear-gradient(180deg, #fff, #fbfdff);
    }}
    summary::-webkit-details-marker {{
      display: none;
    }}
    summary::before {{
      content: "+";
      position: absolute;
      right: 16px;
      top: 13px;
      width: 22px;
      height: 22px;
      line-height: 20px;
      text-align: center;
      border: 1px solid var(--line-strong);
      color: var(--blue);
      font-family: var(--mono);
      font-weight: 850;
      background: #fff;
    }}
    details[open] summary::before {{
      content: "−";
    }}
    .body {{
      border-top: 1px solid var(--line);
      padding: 16px;
      background: linear-gradient(180deg, #fff, #fbfdff);
    }}
    .subfold {{
      margin-left: 12px;
    }}
    .json-viewer[hidden] {{
      display: none;
    }}
    .json-viewer {{
      position: fixed;
      inset: 0;
      z-index: 40;
    }}
    .json-viewer-backdrop {{
      position: absolute;
      inset: 0;
      background: rgba(16, 32, 51, 0.46);
    }}
    .json-viewer-panel {{
      position: relative;
      max-width: min(1220px, calc(100vw - 40px));
      height: calc(100vh - 40px);
      margin: 20px auto;
      display: grid;
      grid-template-rows: auto 1fr;
      border: 1px solid var(--line-strong);
      background: #fff;
      box-shadow: var(--shadow);
    }}
    .json-viewer-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: #f4f8fb;
    }}
    .json-viewer-kicker {{
      color: var(--blue);
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 4px;
    }}
    .json-viewer-head h3 {{
      margin: 0;
      font-size: 16px;
    }}
    .json-viewer-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
    }}
    .json-viewer pre {{
      margin: 0;
      padding: 16px;
      overflow: auto;
      background: #fbfdff;
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.52;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .muted {{
      color: var(--muted);
    }}
    .source-list {{
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
    }}
    .footer {{
      margin-top: 24px;
      padding-top: 16px;
      border-top: 1px solid var(--line);
      color: var(--faint);
      font-size: 12px;
      display: flex;
      justify-content: space-between;
      gap: 24px;
    }}
    @media (max-width: 1100px) {{
      .page {{
        padding: 28px 22px 52px;
      }}
      .hero, .diagram-grid {{
        grid-template-columns: 1fr;
      }}
      .hero-meta {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .chart-grid {{
        grid-template-columns: 1fr;
      }}
      .mini-grid {{
        grid-template-columns: 1fr;
      }}
      .filter-bar {{
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }}
      .section-head {{
        display: block;
      }}
      .section-head p {{
        text-align: left;
        margin-top: 8px;
      }}
    }}
    @media (max-width: 720px) {{
      .hero-meta {{
        grid-template-columns: 1fr;
      }}
      .filter-bar {{
        grid-template-columns: 1fr;
      }}
      .json-viewer-panel {{
        max-width: calc(100vw - 20px);
        height: calc(100vh - 20px);
        margin: 10px auto;
      }}
      .nav {{
        overflow-x: auto;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div class="hero-main">
        <div class="eyebrow">LoPT Benchmark</div>
        <h1>{report_title}</h1>
        <p class="subtitle">{report_subtitle}</p>
        <div class="hero-meta">
          <div class="meta-item">
            <span class="label">生成时间</span>
            <span class="value">{escape(ctx["generated_at"])}</span>
          </div>
          <div class="meta-item">
            <span class="label">CPU 绑核</span>
            <span class="value">{escape(args.cpu_affinity)}</span>
          </div>
          <div class="meta-item">
            <span class="label">最新 vLLM 源码</span>
            <span class="value">{escape(str(args.vllm_src))}</span>
          </div>
          <div class="meta-item">
            <span class="label">Benchmark 链路</span>
            <span class="value"><code>HfRenderer._tokenize_prompt_async</code> → <code>AsyncMicrobatchTokenizer.encode</code></span>
          </div>
        </div>
      </div>
      <aside class="hero-side">
        {render_summary_cards(ctx, args)}
      </aside>
    </section>

    <nav class="nav">
      <a href="#background">业务背景</a>
      <a href="#native">原生链路</a>
      <a href="#lopt">LoPT</a>
      <a href="#setup">实验设置</a>
      <a href="#legend">指标说明</a>
      <a href="#v2-summary">v2.1 摘要</a>
      <a href="#version-compare">v1/v2.1 对比</a>
      <a href="#best-case-analysis">最佳配置综合对比</a>
      <a href="#results">最终结果</a>
      <a href="#worker-best">Worker 最优</a>
      <a href="#search">全量 Search</a>
      <a href="#data-browser">全量数据</a>
      <a href="#sources">语料来源</a>
      <a href="#artifacts">输出物</a>
    </nav>

    <div class="toolbar">
      <button type="button" onclick="toggleAll(true)">全部展开</button>
      <button type="button" onclick="toggleAll(false)">全部折叠</button>
    </div>

    <section class="section" id="background">
      <div class="section-head">
        <h2>01 · 业务背景</h2>
        <p>解释为什么在当前 Agent + 超长上下文 vLLM 场景中，Tokenizer 延迟会成为主要瓶颈。</p>
      </div>
      {background_callout}
      {bottleneck_callout}
    </section>

    <section class="section" id="native">
      <div class="section-head">
        <h2>02 · 原生 vLLM 执行链路</h2>
        <p>基线测试完全复用本地最新 vLLM 代码链路，而不是远端服务进程。</p>
      </div>
      <p>原生基线严格沿用了 vLLM 在 API Server 侧处理 tokenization 请求时的预处理和 tokenizer 调用链路，只是不通过远端服务进程启动，而是直接对本地最新源码离线执行。按你的要求，热点链路保持为：<code>OpenAIServingTokenization.create_tokenize</code> → <code>OpenAIServingRender.preprocess_completion</code> → <code>HfRenderer.render_cmpl_async</code> → <code>BaseRenderer._tokenize_prompt_async</code> → <code>AsyncMicrobatchTokenizer.encode</code>。</p>
      <div class="diagram-grid">
        {render_ascii_block("原生文本到 token IDs 链路", native_ascii, accent="blue")}
        {render_ascii_block("为什么原生异步 Tokenizer 仍会成为瓶颈", native_bottleneck_ascii, accent="teal")}
      </div>
      <details class="fold" open>
        <summary>本次分析引用的关键 vLLM 源文件</summary>
        <div class="body">
          {source_files_html}
        </div>
      </details>
    </section>

    <section class="section" id="lopt">
      <div class="section-head">
        <h2>03 · LoPT 原理与数据链路</h2>
        <p>同时展示 v1 与 v2.1 的字符链路，并说明两版在调度、收集和诊断上的差异。</p>
      </div>
      {principle_paragraphs}
      <div class="diagram-grid">
        {render_ascii_block("LoPT v1 多进程数据链路", lopt_v1_ascii, accent="teal")}
        {render_ascii_block("LoPT v2.1 多进程数据链路", lopt_v2_ascii, accent="blue")}
      </div>
      {render_callout("v2.1 相对 v1 的差异", lopt_v2_diff_points, kind="ok")}
      <div class="diagram-grid">
        {render_ascii_block("基于位置感知的 overlap 去重逻辑", overlap_dedup_ascii, accent="blue")}
        {render_ascii_block("v2.1 额外可观测的细粒度阶段", """
dispatch_submit
    |
    v
collect_result_receive_lag
    |
    v
collect_child_compute_makespan
    |
    v
collect_result_return_tail
    |
    v
chunk_dedup

这些指标不改变主算法，只是把瓶颈拆得更细，方便后续针对性优化。
        """.strip(), accent="teal")}
      </div>
      <details class="fold">
        <summary>LoPT v1 相对论文方案的工程实现说明</summary>
        <div class="body">
          <ul class="source-list">
            <li>实现了论文中的多进程并行 tokenization 与基于 per-token offset mappings 的位置感知 overlap 合并逻辑。</li>
            <li>通过离线 search 穷举 <code>p</code>、<code>k</code>、<code>o</code>，而不是依赖在线 chunk-length retry。</li>
            <li><code>min_match_tokens</code> 固定为 2；候选配置失败时按要求直接回退到原生串行逻辑。</li>
            <li>最终 search 候选与 replay 最优配置均基于原生 vLLM token IDs 做 exact-match 精度校验。</li>
          </ul>
        </div>
      </details>
    </section>

    <section class="section" id="setup">
      <div class="section-head">
        <h2>04 · 实验设置</h2>
        <p>展示 search 变量、固定变量、语料策略以及机器约束。</p>
      </div>
      {render_setup_table(args, ctx)}
    </section>

    <section class="section" id="legend">
      <div class="section-head">
        <h2>05 · 指标说明</h2>
        <p>所有核心表头均补充了单位、符号含义和计算定义。</p>
      </div>
      {render_metric_legend()}
    </section>

    {v2_artifacts_html}

    <section class="section" id="v1-v2-compare">
      {version_comparison_html}
    </section>

    {best_case_summary_html}

    <section class="section" id="results">
      <div class="section-head">
        <h2>09 · 最终最优结果</h2>
        <p>Replay 表格展示了每个模型族、语言和输入长度下最终选定的最优配置。</p>
      </div>
      {best_overview_html}
    </section>

    <section class="section" id="worker-best">
      <div class="section-head">
        <h2>10 · Worker 维度最优配置</h2>
        <p>展示每个 worker process 数下的最优候选，再从中选择最终最优配置。</p>
      </div>
      {worker_best_html}
    </section>

    <section class="section" id="search">
      <div class="section-head">
        <h2>11 · 全量 Search 细节</h2>
        <p>按模型族 → 语言 → 输入长度分层折叠展示完整候选配置表。</p>
      </div>
      {candidate_detail_html}
    </section>

    {data_browser_html}

    <section class="section" id="sources">
      <div class="section-head">
        <h2>13 · 真实网页语料来源</h2>
        <p>纯中文和纯英文输入均来自真实公开网页的可见文本抽取结果。</p>
      </div>
      {source_panel_html}
    </section>

    <section class="section" id="artifacts">
      <div class="section-head">
        <h2>14 · 输出物路径</h2>
        <p>列出原始候选数据、合并结果、replay 结果以及当前 HTML 报告的落盘路径。</p>
      </div>
      {render_table(
          ["输出物", "路径"],
          [
              ["Merged v2.1 search detail JSONL", escape(str(args.search_detail_jsonl))],
              ["Merged v2.1 worker-best JSON", escape(str(args.worker_best_json))],
              ["Merged v2.1 best JSON", escape(str(args.best_json))],
              ["v2.1 replay JSON", escape(str(args.replay_json))],
              ["v1 replay JSON", escape(str(getattr(args, 'v1_replay_json', '-') or '-'))],
              ["链路分析 Markdown", escape(str(args.flow_doc))],
              ["HTML 报告", escape(str(args.output_html))],
          ],
      )}
    </section>

    <div class="footer">
      <span>本报告由本地结果文件生成，远端 CPU benchmark 数据已同步回当前工作区。</span>
      <span>风格目标：modern technical infographic · white background · sharp lines · blue / teal / orange palette。</span>
    </div>
  </main>
"""

    html += f"""
  <script id="detail-records-json" type="application/json">{detail_json_payload}</script>
  <script id="replay-records-json" type="application/json">{replay_json_payload}</script>
"""
    html += """
  <script>
    (() => {
      const detailRows = parseJsonScript("detail-records-json");
      const replayRows = parseJsonScript("replay-records-json");
      const familyOrder = { "DeepSeek-V4-Pro": 0, "Qwen3.5": 1 };
      const languageOrder = { zh: 0, en: 1 };
      const viewer = document.getElementById("json-viewer");
      const viewerTitle = document.getElementById("json-viewer-title");
      const viewerBody = document.getElementById("json-viewer-body");
      const copyButton = document.getElementById("json-copy");
      let viewerPayload = "";

      window.toggleAll = function(open) {
        document.querySelectorAll("details").forEach((el) => {
          el.open = open;
        });
      };

      function parseJsonScript(id) {
        const node = document.getElementById(id);
        if (!node) {
          return [];
        }
        try {
          return JSON.parse(node.textContent || "[]");
        } catch (error) {
          console.error("Failed to parse embedded JSON for", id, error);
          return [];
        }
      }

      function escapeHtml(value) {
        return String(value)
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/\"/g, "&quot;")
          .replace(/'/g, "&#39;");
      }

      function fmtInt(value) {
        if (value === null || value === undefined || value === "") {
          return "-";
        }
        return Number(value).toLocaleString("en-US");
      }

      function fmtMs(value) {
        if (value === null || value === undefined || value === "") {
          return "-";
        }
        return Number(value).toLocaleString("en-US", {
          minimumFractionDigits: 3,
          maximumFractionDigits: 3,
        });
      }

      function fmtRatio(value) {
        if (value === null || value === undefined || value === "") {
          return "-";
        }
        return `${Number(value).toFixed(3)}x`;
      }

      function fmtPct(value) {
        if (value === null || value === undefined || value === "") {
          return "-";
        }
        return `${Number(value).toFixed(2)}%`;
      }

      function uiLanguageLabel(value) {
        if (value === "zh") {
          return "中文";
        }
        if (value === "en") {
          return "英文";
        }
        return String(value ?? "");
      }

      function uiStatusLabel(value) {
        if (value === "valid") {
          return "有效";
        }
        if (value === "fallback") {
          return "回退";
        }
        if (value === "mismatch") {
          return "不一致";
        }
        if (value === "error") {
          return "错误";
        }
        return value === null || value === undefined || value === "" ? "-" : String(value);
      }

      function speedupClass(value) {
        const numeric = Number(value);
        if (!Number.isFinite(numeric)) {
          return "neutral";
        }
        if (numeric >= 1.2) {
          return "good";
        }
        if (numeric >= 1.0) {
          return "ok";
        }
        return "warn";
      }

      function statusClass(value) {
        if (value === true) {
          return "good";
        }
        if (value === false) {
          return "neutral";
        }
        if (value === "valid") {
          return "good";
        }
        if (value === "fallback" || value === "mismatch" || value === "error") {
          return "warn";
        }
        return "neutral";
      }

      function badge(label, cls) {
        return `<span class="metric-badge ${cls}">${escapeHtml(label)}</span>`;
      }

      function parseLengthOrder(rows) {
        const seen = new Map();
        rows.forEach((row) => {
          if (!seen.has(row.length_label)) {
            seen.set(row.length_label, Number(row.input_chars) || 0);
          }
        });
        return [...seen.entries()]
          .sort((left, right) => left[1] - right[1])
          .map(([label]) => label);
      }

      function sortedUnique(values, sorter) {
        return [...new Set(values)].sort(sorter);
      }

      function naturalCompare(left, right) {
        return String(left).localeCompare(String(right), undefined, { numeric: true });
      }

      function searchBlob(row) {
        return [
          row.tokenizer_family,
          row.model_name,
          row.language,
          row.length_label,
          row.input_chars,
          row.worker_processes,
          row.chunk_count,
          row.chunk_chars,
          row.overlap_chars,
          row.candidate_status,
          row.exact_match,
          row.fallback_used,
          row.error_message,
        ]
          .filter((value) => value !== null && value !== undefined && value !== "")
          .join(" ")
          .toLowerCase();
      }

      function buildJsonTitle(prefix, row, index) {
        return `${prefix} #${index + 1} · ${row.tokenizer_family} / ${uiLanguageLabel(row.language)} / ${row.length_label} / p=${row.worker_processes} / k=${row.chunk_count} / o=${row.overlap_chars}`;
      }

      function openJson(title, payload) {
        viewerTitle.textContent = title;
        viewerPayload = JSON.stringify(payload, null, 2);
        viewerBody.textContent = viewerPayload;
        viewer.hidden = false;
        document.body.style.overflow = "hidden";
      }

      function closeJson() {
        viewer.hidden = true;
        document.body.style.overflow = "";
      }

      async function copyJson() {
        if (!viewerPayload) {
          return;
        }
        try {
          await navigator.clipboard.writeText(viewerPayload);
          copyButton.textContent = "已复制";
          setTimeout(() => {
            copyButton.textContent = "复制 JSON";
          }, 1200);
        } catch (error) {
          console.error("Copy failed", error);
          copyButton.textContent = "复制失败";
          setTimeout(() => {
            copyButton.textContent = "复制 JSON";
          }, 1200);
        }
      }

      function renderNoRows(colspan) {
        return `<tr><td colspan="${colspan}" class="muted">当前筛选条件下没有匹配数据。</td></tr>`;
      }

      function buildBrowser(config) {
        const rows = config.rows;
        const table = document.getElementById(config.tableId);
        if (!table) {
          return;
        }
        const tbody = table.querySelector("tbody");
        const familySelect = document.getElementById(config.familyFilterId);
        const languageSelect = document.getElementById(config.languageFilterId);
        const lengthSelect = document.getElementById(config.lengthFilterId);
        const searchInput = document.getElementById(config.searchId);
        const pageSizeSelect = document.getElementById(config.pageSizeId);
        const pageInfo = document.getElementById(config.pageInfoId);
        const prevButton = document.getElementById(config.prevId);
        const nextButton = document.getElementById(config.nextId);
        const statusSelect = config.statusFilterId
          ? document.getElementById(config.statusFilterId)
          : null;
        const state = {
          page: 1,
          pageSize: Number(pageSizeSelect ? pageSizeSelect.value : config.defaultPageSize || 50),
        };

        function setOptions(select, values, formatter) {
          if (!select) {
            return;
          }
          const display = formatter || ((value) => value);
          select.innerHTML = ['<option value="">全部</option>']
            .concat(values.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(display(value))}</option>`))
            .join("");
        }

        setOptions(
          familySelect,
          sortedUnique(rows.map((row) => row.tokenizer_family), (left, right) => {
            return (familyOrder[left] ?? 99) - (familyOrder[right] ?? 99) || naturalCompare(left, right);
          }),
        );
        setOptions(
          languageSelect,
          sortedUnique(rows.map((row) => row.language), (left, right) => {
            return (languageOrder[left] ?? 99) - (languageOrder[right] ?? 99) || naturalCompare(left, right);
          }),
          uiLanguageLabel,
        );
        setOptions(lengthSelect, parseLengthOrder(rows));
        if (statusSelect) {
          setOptions(
            statusSelect,
            sortedUnique(rows.map((row) => row.candidate_status), naturalCompare),
            uiStatusLabel,
          );
        }

        function filteredRows() {
          const query = String(searchInput ? searchInput.value : "").trim().toLowerCase();
          return rows
            .map((row, index) => ({ row, index }))
            .filter(({ row }) => {
              if (familySelect && familySelect.value && row.tokenizer_family !== familySelect.value) {
                return false;
              }
              if (languageSelect && languageSelect.value && row.language !== languageSelect.value) {
                return false;
              }
              if (lengthSelect && lengthSelect.value && row.length_label !== lengthSelect.value) {
                return false;
              }
              if (statusSelect && statusSelect.value && row.candidate_status !== statusSelect.value) {
                return false;
              }
              if (query && !searchBlob(row).includes(query)) {
                return false;
              }
              return true;
            });
        }

        function render() {
          const filtered = filteredRows();
          const total = filtered.length;
          const pageCount = Math.max(1, Math.ceil(total / state.pageSize));
          state.page = Math.max(1, Math.min(state.page, pageCount));
          const start = total === 0 ? 0 : (state.page - 1) * state.pageSize;
          const end = Math.min(total, start + state.pageSize);
          const slice = filtered.slice(start, end);

          if (!slice.length) {
            tbody.innerHTML = renderNoRows(config.columnCount);
          } else {
            tbody.innerHTML = slice
              .map(({ row, index }) => config.renderRow(row, index))
              .join("");
          }

          pageInfo.textContent = total === 0
            ? "显示 0 / 0 行"
            : `显示 ${start + 1}-${end} / ${total} 行 · 第 ${state.page} / ${pageCount} 页`;
          prevButton.disabled = state.page <= 1;
          nextButton.disabled = state.page >= pageCount;
        }

        [familySelect, languageSelect, lengthSelect, statusSelect, pageSizeSelect].forEach((el) => {
          if (!el) {
            return;
          }
          el.addEventListener("change", () => {
            if (el === pageSizeSelect) {
              state.pageSize = Number(pageSizeSelect.value) || state.pageSize;
            }
            state.page = 1;
            render();
          });
        });

        if (searchInput) {
          searchInput.addEventListener("input", () => {
            state.page = 1;
            render();
          });
        }

        prevButton.addEventListener("click", () => {
          if (state.page > 1) {
            state.page -= 1;
            render();
          }
        });

        nextButton.addEventListener("click", () => {
          state.page += 1;
          render();
        });

        tbody.addEventListener("click", (event) => {
          const button = event.target.closest("[data-json-index]");
          if (!button) {
            return;
          }
          const index = Number(button.getAttribute("data-json-index"));
          if (!Number.isFinite(index) || index < 0 || index >= rows.length) {
            return;
          }
          openJson(buildJsonTitle(config.viewerPrefix, rows[index], index), rows[index]);
        });

        render();
      }

      function detailRowHtml(row, index) {
        return [
          `<tr>`,
          `<td class="mono-cell">${fmtInt(index + 1)}</td>`,
          `<td class="soft-wrap">${escapeHtml(row.tokenizer_family)}</td>`,
          `<td class="mono-cell">${escapeHtml(uiLanguageLabel(row.language))}</td>`,
          `<td class="mono-cell">${escapeHtml(row.length_label)}</td>`,
          `<td class="mono-cell">${fmtInt(row.input_chars)}</td>`,
          `<td class="mono-cell">${fmtInt(row.worker_processes)}</td>`,
          `<td class="mono-cell">${fmtInt(row.chunk_count)}</td>`,
          `<td class="mono-cell">${fmtInt(row.chunk_chars)}</td>`,
          `<td class="mono-cell">${fmtInt(row.overlap_chars)}</td>`,
          `<td class="mono-cell">${fmtMs(row.chat_template_time_ms)}</td>`,
          `<td class="mono-cell">${fmtMs(row.mp_dispatch_process_collect_time_ms)}</td>`,
          `<td class="mono-cell">${fmtMs(row.chunk_dedup_time_ms)}</td>`,
          `<td class="mono-cell">${fmtMs(row.lopt_e2e_time_ms)}</td>`,
          `<td class="mono-cell">${fmtMs(row.native_e2e_time_ms)}</td>`,
          `<td class="mono-cell">${fmtMs(row.native_tokenizer_time_ms)}</td>`,
          `<td>${badge(fmtRatio(row.e2e_speedup_x), speedupClass(row.e2e_speedup_x))}</td>`,
          `<td>${badge(fmtRatio(row.tokenizer_speedup_x), speedupClass(row.tokenizer_speedup_x))}</td>`,
          `<td class="mono-cell">${fmtPct(row.tokenizer_time_drop_pct)}</td>`,
          `<td>${badge(row.exact_match ? "Exact Match" : "Mismatch", statusClass(row.exact_match))}</td>`,
          `<td>${badge(row.fallback_used ? "是" : "否", statusClass(!row.fallback_used))}</td>`,
          `<td>${badge(uiStatusLabel(row.candidate_status), statusClass(row.candidate_status))}</td>`,
          `<td><button type="button" class="table-button" data-json-index="${index}">View</button></td>`,
          `</tr>`,
        ].join("");
      }

      function replayRowHtml(row, index) {
        return [
          `<tr>`,
          `<td class="mono-cell">${fmtInt(index + 1)}</td>`,
          `<td class="soft-wrap">${escapeHtml(row.tokenizer_family)}</td>`,
          `<td class="mono-cell">${escapeHtml(uiLanguageLabel(row.language))}</td>`,
          `<td class="mono-cell">${escapeHtml(row.length_label)}</td>`,
          `<td class="mono-cell">${fmtInt(row.input_chars)}</td>`,
          `<td class="mono-cell">${fmtInt(row.native_output_tokens)}</td>`,
          `<td class="mono-cell">${fmtInt(row.worker_processes)}</td>`,
          `<td class="mono-cell">${fmtInt(row.chunk_count)}</td>`,
          `<td class="mono-cell">${fmtInt(row.overlap_chars)}</td>`,
          `<td class="mono-cell">${fmtMs(row.chat_template_time_ms)}</td>`,
          `<td class="mono-cell">${fmtMs(row.mp_dispatch_process_collect_time_ms)}</td>`,
          `<td class="mono-cell">${fmtMs(row.chunk_dedup_time_ms)}</td>`,
          `<td class="mono-cell">${fmtMs(row.lopt_e2e_time_ms)}</td>`,
          `<td class="mono-cell">${fmtMs(row.native_e2e_time_ms)}</td>`,
          `<td class="mono-cell">${fmtMs(row.native_tokenizer_time_ms)}</td>`,
          `<td>${badge(fmtRatio(row.e2e_speedup_x), speedupClass(row.e2e_speedup_x))}</td>`,
          `<td>${badge(fmtRatio(row.tokenizer_speedup_x), speedupClass(row.tokenizer_speedup_x))}</td>`,
          `<td class="mono-cell">${fmtPct(row.tokenizer_time_drop_pct)}</td>`,
          `<td>${badge(row.exact_match ? "Exact Match" : "Mismatch", statusClass(row.exact_match))}</td>`,
          `<td><button type="button" class="table-button" data-json-index="${index}">View</button></td>`,
          `</tr>`,
        ].join("");
      }

      buildBrowser({
        rows: detailRows,
        tableId: "detail-table",
        familyFilterId: "detail-family-filter",
        languageFilterId: "detail-language-filter",
        lengthFilterId: "detail-length-filter",
        statusFilterId: "detail-status-filter",
        searchId: "detail-search",
        pageSizeId: "detail-page-size",
        pageInfoId: "detail-page-info",
        prevId: "detail-prev",
        nextId: "detail-next",
        columnCount: 22,
        viewerPrefix: "Search 候选",
        renderRow: detailRowHtml,
      });

      buildBrowser({
        rows: replayRows,
        tableId: "replay-table",
        familyFilterId: "replay-family-filter",
        languageFilterId: "replay-language-filter",
        lengthFilterId: "replay-length-filter",
        searchId: "replay-search",
        pageSizeId: "replay-page-size",
        pageInfoId: "replay-page-info",
        prevId: "replay-prev",
        nextId: "replay-next",
        columnCount: 20,
        viewerPrefix: "Replay 记录",
        renderRow: replayRowHtml,
      });

      document.querySelectorAll("[data-close-json]").forEach((node) => {
        node.addEventListener("click", closeJson);
      });
      document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && !viewer.hidden) {
          closeJson();
        }
      });
      copyButton.addEventListener("click", copyJson);
    })();
  </script>
</body>
</html>
"""
    return html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-detail-jsonl", type=Path, required=True)
    parser.add_argument("--worker-best-json", type=Path, required=True)
    parser.add_argument("--best-json", type=Path, required=True)
    parser.add_argument("--replay-json", type=Path, required=True)
    parser.add_argument("--v1-best-json", type=Path, default=None)
    parser.add_argument("--v1-replay-json", type=Path, default=None)
    parser.add_argument("--flow-doc", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--vllm-src", type=Path, required=True)
    parser.add_argument("--cpu-affinity", default="29-31,40-79,155-159")
    parser.add_argument("--report-title", default="vLLM Tokenizer 原生基线与 LoPT 优化 Benchmark 报告")
    parser.add_argument(
        "--report-subtitle",
        default=(
            "基于本地最新 vLLM 源码，比较原生异步线程 Tokenizer 链路与 LoPT 风格多进程并行 Tokenization，在真实中英文长文本输入上的性能与精度表现。"
        ),
    )
    parser.add_argument("--search-meta-json", type=Path, default=None)
    parser.add_argument("--corpus-meta-dir", type=Path, default=None)
    parser.add_argument("--search-input-dirs", type=Path, nargs="*", default=[])
    parser.add_argument("--env-info-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    ctx = build_report_context(args)
    html = render_html(args, ctx)
    args.output_html.write_text(html, encoding="utf-8")
    print(
        json.dumps(
            {
                "output_html": str(args.output_html),
                "candidate_count": ctx["candidate_counter"],
                "replay_case_count": len(ctx["replay_records"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
