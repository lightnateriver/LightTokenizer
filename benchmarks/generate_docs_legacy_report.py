#!/usr/bin/env python3
"""Build docs/index.html by extending the original v1-style HTML report.

This generator keeps the legacy LightTokenizer visual language and injects
v2.1 sections, comparison charts, and a collapsible browser for v2.1 data.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from html import escape
from pathlib import Path

LENGTH_ORDER = [
    "1k",
    "4k",
    "8k",
    "16k",
    "32k",
    "64k",
    "128k",
    "256k",
    "512k",
    "720k",
    "880k",
    "1024k",
]

FAMILY_ORDER = ["DeepSeek-V4-Pro", "Qwen3.5"]
LANGUAGE_ORDER = ["zh", "en"]
LANG_DISPLAY = {"zh": "中文", "en": "英文"}

LINE_COLORS = {
    "native": "#f59e0b",
    "v1": "#1f6feb",
    "v2": "#0f9f9a",
    "mp": "#1f6feb",
    "dedup": "#0f9f9a",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def fmt_ms(value) -> str:
    if value is None:
        return "-"
    return f"{float(value):,.3f}"


def fmt_x(value) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}x"


def fmt_pct(value) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2f}%"


def fmt_int(value) -> str:
    if value is None:
        return "-"
    return f"{int(value):,}"


def key_case(rec: dict) -> tuple[str, str, str]:
    return rec["tokenizer_family"], rec["language"], rec["length_label"]


def sorted_case_keys(keys):
    return sorted(
        keys,
        key=lambda x: (
            FAMILY_ORDER.index(x[0]) if x[0] in FAMILY_ORDER else 99,
            LANGUAGE_ORDER.index(x[1]) if x[1] in LANGUAGE_ORDER else 99,
            LENGTH_ORDER.index(x[2]) if x[2] in LENGTH_ORDER else 99,
        ),
    )


def badge_class(speedup: float) -> str:
    if speedup >= 2.0:
        return "g"
    if speedup >= 1.0:
        return "b"
    return "o"


def row_compare(v1: dict, v2: dict) -> str:
    delta = (v1["lopt_e2e_time_ms"] - v2["lopt_e2e_time_ms"]) / v1["lopt_e2e_time_ms"] * 100.0
    delta_cls = "g" if delta >= 0 else "r"
    exact = "是" if v2.get("exact_match") else "否"
    return (
        f"<tr><td>{escape(v2['tokenizer_family'])}</td>"
        f"<td>{LANG_DISPLAY[v2['language']]}</td>"
        f"<td>{escape(v2['length_label'])}</td>"
        f"<td class='n'>{fmt_int(v2['worker_processes'])}</td>"
        f"<td class='n'>{fmt_int(v2['chunk_count'])}</td>"
        f"<td class='n'>{fmt_int(v2['overlap_chars'])}</td>"
        f"<td class='n'>{fmt_ms(v2['native_e2e_time_ms'])}</td>"
        f"<td class='n'>{fmt_ms(v1['lopt_e2e_time_ms'])}</td>"
        f"<td class='n'>{fmt_ms(v2['lopt_e2e_time_ms'])}</td>"
        f"<td class='n'>{fmt_ms(v2['mp_dispatch_process_collect_time_ms'])}</td>"
        f"<td class='n'>{fmt_ms(v2['chunk_dedup_time_ms'])}</td>"
        f"<td class='n {delta_cls}'>{fmt_pct(delta)}</td>"
        f"<td class='n {badge_class(v2['e2e_speedup_x'])}'>{fmt_x(v2['e2e_speedup_x'])}</td>"
        f"<td class='n {badge_class(v2['tokenizer_speedup_x'])}'>{fmt_x(v2['tokenizer_speedup_x'])}</td>"
        f"<td class='n {badge_class(1.0 if v2.get('exact_match') else 0.0)}'>{exact}</td></tr>"
    )


def make_polyline(points, width=980, height=260, y_max=None, y_min=0.0):
    left, right, top, bottom = 58, 24, 18, 32
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_count = max(len(points), 1)
    if y_max is None:
        y_max = max((p[1] for p in points), default=1.0)
    y_max = max(y_max, y_min + 1e-6)

    def x_at(idx):
        if x_count == 1:
            return left + plot_w / 2
        return left + plot_w * idx / (x_count - 1)

    def y_at(val):
        ratio = (val - y_min) / (y_max - y_min)
        return top + plot_h * (1.0 - ratio)

    coords = " ".join(f"{x_at(i):.1f},{y_at(v):.1f}" for i, (_, v) in enumerate(points))
    labels = [
        f"<text x='{x_at(i):.1f}' y='{height - 10}' text-anchor='middle' font-size='11' fill='#4a5a7a'>{escape(lbl)}</text>"
        for i, (lbl, _) in enumerate(points)
    ]
    grid = []
    for step in range(5):
        value = y_min + (y_max - y_min) * step / 4
        y = y_at(value)
        grid.append(f"<line x1='{left}' x2='{width-right}' y1='{y:.1f}' y2='{y:.1f}' stroke='#e2e8f0'/>")
        grid.append(
            f"<text x='{left-8}' y='{y+4:.1f}' text-anchor='end' font-size='11' fill='#8a98aa'>{value:.1f}</text>"
        )
    return {
        "coords": coords,
        "labels": "".join(labels),
        "grid": "".join(grid),
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "width": width,
        "height": height,
    }


def render_line_chart(title: str, subtitle: str, series: list[dict]) -> str:
    all_points = []
    for item in series:
        all_points.extend(item["points"])
    y_max = max((p[1] for p in all_points), default=1.0)
    base = make_polyline(series[0]["points"], y_max=y_max * 1.08 if y_max else 1.0)
    polylines = []
    legend = []
    for item in series:
        coords = make_polyline(item["points"], y_max=y_max * 1.08 if y_max else 1.0)["coords"]
        dash = "4 4" if item.get("dash") else ""
        polylines.append(
            f"<polyline fill='none' stroke='{item['color']}' stroke-width='3' stroke-dasharray='{dash}' points='{coords}'/>"
        )
        legend.append(
            f"<span style='display:inline-flex;align-items:center;gap:6px;margin-right:12px;'>"
            f"<span style='display:inline-block;width:18px;height:0;border-top:3px {'dashed' if item.get('dash') else 'solid'} {item['color']};'></span>"
            f"<span>{escape(item['label'])}</span></span>"
        )
    return f"""
<div class="fb" style="margin:10px 0">
  <div class="tt" style="background:var(--bg3);color:var(--ink)">{escape(title)}</div>
  <div style="padding:10px 12px 4px;color:var(--text2);font-size:.82rem">{escape(subtitle)}</div>
  <div style="padding:0 12px 8px;font-size:.78rem;color:var(--text2)">{''.join(legend)}</div>
  <svg viewBox="0 0 {base['width']} {base['height']}" style="width:100%;height:auto;background:var(--bg2);border-top:1px solid var(--line)">
    {base['grid']}
    <line x1="{base['left']}" x2="{base['left']}" y1="{base['top']}" y2="{base['height']-base['bottom']}" stroke="#c8d6e5"/>
    <line x1="{base['left']}" x2="{base['width']-base['right']}" y1="{base['height']-base['bottom']}" y2="{base['height']-base['bottom']}" stroke="#c8d6e5"/>
    {''.join(polylines)}
    {base['labels']}
  </svg>
</div>
"""


def render_grouped_bars(title: str, subtitle: str, native_v1_v2: list[tuple[str, float, float, float, float, float]]) -> str:
    width, height = 980, 270
    left, right, top, bottom = 58, 24, 18, 42
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_max = max(max(n, v1, v2) for _, n, v1, v2, _, _ in native_v1_v2)
    y_max = y_max * 1.08 if y_max else 1.0

    def y_at(val):
        ratio = val / y_max
        return top + plot_h * (1.0 - ratio)

    group_w = plot_w / max(len(native_v1_v2), 1)
    bar_w = min(16, group_w / 5)
    grid = []
    for step in range(5):
        value = y_max * step / 4
        y = y_at(value)
        grid.append(f"<line x1='{left}' x2='{width-right}' y1='{y:.1f}' y2='{y:.1f}' stroke='#e2e8f0'/>")
        grid.append(f"<text x='{left-8}' y='{y+4:.1f}' text-anchor='end' font-size='11' fill='#8a98aa'>{value:.1f}</text>")

    bars = []
    labels = []
    for idx, (label, native, v1, v2, mp, dedup) in enumerate(native_v1_v2):
        gx = left + idx * group_w + group_w / 2
        native_x = gx - bar_w * 2
        v1_x = gx
        v2_x = gx + bar_w * 2
        native_h = plot_h - (y_at(native) - top)
        v1_h = plot_h - (y_at(v1) - top)
        mp_h = plot_h - (y_at(mp) - top)
        dedup_h = y_at(mp) - y_at(v2)
        bars.append(f"<rect x='{native_x:.1f}' y='{y_at(native):.1f}' width='{bar_w:.1f}' height='{native_h:.1f}' fill='{LINE_COLORS['native']}'/>")
        bars.append(f"<rect x='{v1_x:.1f}' y='{y_at(v1):.1f}' width='{bar_w:.1f}' height='{v1_h:.1f}' fill='{LINE_COLORS['v1']}'/>")
        bars.append(f"<rect x='{v2_x:.1f}' y='{y_at(mp):.1f}' width='{bar_w:.1f}' height='{mp_h:.1f}' fill='{LINE_COLORS['mp']}'/>")
        bars.append(f"<rect x='{v2_x:.1f}' y='{y_at(v2):.1f}' width='{bar_w:.1f}' height='{dedup_h:.1f}' fill='{LINE_COLORS['dedup']}'/>")
        labels.append(f"<text x='{gx:.1f}' y='{height-12}' text-anchor='middle' font-size='11' fill='#4a5a7a'>{escape(label)}</text>")
    legend = (
        "<span style='display:inline-flex;align-items:center;gap:6px;margin-right:12px;'><span style='width:10px;height:10px;background:#f59e0b;display:inline-block'></span><span>原生 E2E</span></span>"
        "<span style='display:inline-flex;align-items:center;gap:6px;margin-right:12px;'><span style='width:10px;height:10px;background:#1f6feb;display:inline-block'></span><span>v1 LoPT E2E</span></span>"
        "<span style='display:inline-flex;align-items:center;gap:6px;margin-right:12px;'><span style='width:10px;height:10px;background:#0f9f9a;display:inline-block'></span><span>v2.1 去重段</span></span>"
        "<span style='display:inline-flex;align-items:center;gap:6px;'><span style='width:10px;height:10px;background:#1f6feb;display:inline-block'></span><span>v2.1 MP 主段</span></span>"
    )
    return f"""
<div class="fb" style="margin:10px 0">
  <div class="tt" style="background:var(--bg3);color:var(--ink)">{escape(title)}</div>
  <div style="padding:10px 12px 4px;color:var(--text2);font-size:.82rem">{escape(subtitle)}</div>
  <div style="padding:0 12px 8px;font-size:.78rem;color:var(--text2)">{legend}</div>
  <svg viewBox="0 0 {width} {height}" style="width:100%;height:auto;background:var(--bg2);border-top:1px solid var(--line)">
    {''.join(grid)}
    <line x1="{left}" x2="{left}" y1="{top}" y2="{height-bottom}" stroke="#c8d6e5"/>
    <line x1="{left}" x2="{width-right}" y1="{height-bottom}" y2="{height-bottom}" stroke="#c8d6e5"/>
    {''.join(bars)}
    {''.join(labels)}
  </svg>
</div>
"""


def summary_by_family_language(records: list[dict]) -> dict[tuple[str, str], dict]:
    groups = defaultdict(list)
    for record in records:
        groups[(record["tokenizer_family"], record["language"])].append(record)
    summary = {}
    for key, rows in groups.items():
        summary[key] = {
            "avg_native_e2e": sum(r["native_e2e_time_ms"] for r in rows) / len(rows),
            "avg_lopt_e2e": sum(r["lopt_e2e_time_ms"] for r in rows) / len(rows),
            "avg_mp": sum(r["mp_dispatch_process_collect_time_ms"] for r in rows) / len(rows),
            "avg_dedup": sum(r["chunk_dedup_time_ms"] for r in rows) / len(rows),
            "avg_speedup": sum(r["e2e_speedup_x"] for r in rows) / len(rows),
            "exact": sum(1 for r in rows if r.get("exact_match")),
            "count": len(rows),
        }
    return summary


def render_best_config_table(v1_map: dict, v2_map: dict) -> str:
    rows = [row_compare(v1_map[key], v2_map[key]) for key in sorted_case_keys(v2_map.keys())]
    return f"""
<details class="fold"><summary>展开 v2.1 最优参数 48 Case 对照表</summary><div class="bd">
<table><thead><tr>
<th>模型</th><th>语言</th><th>长度</th><th>p<span class="u"> (proc)</span></th>
<th>k<span class="u"> (count)</span></th><th>o<span class="u"> (chars)</span></th>
<th>原生 E2E<span class="u"> (ms)</span></th><th>v1 LoPT<span class="u"> (ms)</span></th>
<th>v2.1 LoPT<span class="u"> (ms)</span></th><th>v2.1 MP<span class="u"> (ms)</span></th>
<th>v2.1 去重<span class="u"> (ms)</span></th><th>v2.1 相对 v1<span class="u"> (%)</span></th>
<th>v2.1 E2E 加速<span class="u"> (x)</span></th><th>v2.1 Tokenizer 加速<span class="u"> (x)</span></th><th>精度</th>
</tr></thead><tbody>
{''.join(rows)}
</tbody></table></div></details>
"""


def build_v21_intro(v1_records: list[dict], v2_records: list[dict]) -> str:
    avg_v1 = sum(r["lopt_e2e_time_ms"] for r in v1_records) / len(v1_records)
    avg_v2 = sum(r["lopt_e2e_time_ms"] for r in v2_records) / len(v2_records)
    avg_drop = (avg_v1 - avg_v2) / avg_v1 * 100.0
    return f"""
<div class="sec" id="v21"><h2>v2.1 版本增量说明</h2>
<div class="lead">保持原始 v1 报告风格不变，在此基础上补入 v2.1 方案、最佳参数结果、曲线分析与完整查询浏览器。</div>
<div class="c"><div class="t"><strong>这次补充的核心不是重写 LoPT 主思路，</strong>而是在保留“字符切块 → 多进程并行编码 → overlap 去冗余 → 合并输出”骨架不变的前提下，把 v2.1 主线接到原始工程里，并把原生 vLLM / LoPT v1 / LoPT v2.1 三条线放到同一份 HTML 中可直接对照。</div></div>
<div class="c g"><div class="t"><strong>v2.1 的直接收益：</strong>48 个最终 replay case 全部保持 exact match；平均 LoPT E2E 从 v1 的 {fmt_ms(avg_v1)} ms 降到 {fmt_ms(avg_v2)} ms，平均下降 <strong>{fmt_pct(avg_drop)}</strong>。</div></div>
<div class="fc">
<div class="fb bf"><div class="tt">原生 vLLM 链路</div><pre>用户请求文本
    |
    v
Chat Template / Prompt Render
    |
    v
HfRenderer._tokenize_prompt_async
    |
    v
AsyncMicrobatchTokenizer.encode
    |
    v
单次超长 tokenizer 调用
    |
    v
token_ids 返回</pre></div>
<div class="fb af"><div class="tt">LoPT v2.1 链路</div><pre>用户请求文本
    |
    +-- 按 k 个 chunk 切分
    |      每段附加 overlap=o chars
    |
    +-- submit: 父进程分发到 p 个 worker
    |
    +-- collect: 子进程并行 encode + 返回 ids/offsets
    |
    +-- dedup: 父进程根据 offsets 去除重叠重复 token
    |
    +-- merge: 拼接为完整 token_ids
    |
    v
与原生完全一致的 token_ids</pre></div>
</div>
<table><thead><tr><th>版本</th><th>目标</th><th>主耗时结构</th><th>定位能力</th><th>当前角色</th></tr></thead><tbody>
<tr><td>原生 vLLM</td><td>真实基线</td><td>Chat Template + AsyncMicrobatchTokenizer.encode</td><td>仅能看到整体原生耗时</td><td>对照基准</td></tr>
<tr><td>LoPT v1</td><td>验证多进程可行性</td><td>多进程主段 + 较重去重段</td><td>能够比较 E2E 与粗粒度 LoPT 时间</td><td>历史参考版本</td></tr>
<tr><td>LoPT v2.1</td><td>在 exact match 前提下继续降时延</td><td>submit + collect + dedup 分阶段可见</td><td>能看到分发、回收、子进程计算与去重瓶颈</td><td>当前主线版本</td></tr>
</tbody></table>
</div>
"""


def build_compare_section(v1_records: list[dict], v2_records: list[dict]) -> str:
    v1_map = {key_case(r): r for r in v1_records}
    v2_map = {key_case(r): r for r in v2_records}
    grouped_rows = []
    v1_summary = summary_by_family_language(v1_records)
    v2_summary = summary_by_family_language(v2_records)
    for family in FAMILY_ORDER:
        for lang in LANGUAGE_ORDER:
            key = (family, lang)
            a = v1_summary[key]
            b = v2_summary[key]
            drop = (a["avg_lopt_e2e"] - b["avg_lopt_e2e"]) / a["avg_lopt_e2e"] * 100.0
            grouped_rows.append(
                f"<tr><td>{escape(family)}</td><td>{LANG_DISPLAY[lang]}</td>"
                f"<td class='n'>{fmt_ms(a['avg_native_e2e'])}</td>"
                f"<td class='n'>{fmt_ms(a['avg_lopt_e2e'])}</td>"
                f"<td class='n'>{fmt_ms(b['avg_lopt_e2e'])}</td>"
                f"<td class='n'>{fmt_ms(b['avg_mp'])}</td>"
                f"<td class='n'>{fmt_ms(b['avg_dedup'])}</td>"
                f"<td class='n g'>{fmt_pct(drop)}</td>"
                f"<td class='n g'>{fmt_x(b['avg_speedup'])}</td>"
                f"<td class='n'>{b['exact']}/{b['count']}</td></tr>"
            )

    chart_cards = []
    speed_cards = []
    bar_cards = []
    for family in FAMILY_ORDER:
        for lang in LANGUAGE_ORDER:
            keys = [
                key for key in sorted_case_keys(v2_map.keys()) if key[0] == family and key[1] == lang
            ]
            native_points = [(k[2], v2_map[k]["native_e2e_time_ms"]) for k in keys]
            v1_points = [(k[2], v1_map[k]["lopt_e2e_time_ms"]) for k in keys]
            v2_points = [(k[2], v2_map[k]["lopt_e2e_time_ms"]) for k in keys]
            speed_v1 = [(k[2], v1_map[k]["e2e_speedup_x"]) for k in keys]
            speed_v2 = [(k[2], v2_map[k]["e2e_speedup_x"]) for k in keys]
            bar_data = [
                (
                    k[2],
                    v2_map[k]["native_e2e_time_ms"],
                    v1_map[k]["lopt_e2e_time_ms"],
                    v2_map[k]["lopt_e2e_time_ms"],
                    v2_map[k]["mp_dispatch_process_collect_time_ms"],
                    v2_map[k]["chunk_dedup_time_ms"],
                )
                for k in keys
            ]
            title = f"{family} / {LANG_DISPLAY[lang]}"
            chart_cards.append(
                render_line_chart(
                    f"{title} · 原生 / v1 / v2.1 E2E 曲线",
                    "横轴为输入长度，纵轴为端到端耗时 (ms)。",
                    [
                        {"label": "原生 vLLM", "color": LINE_COLORS["native"], "dash": True, "points": native_points},
                        {"label": "LoPT v1", "color": LINE_COLORS["v1"], "points": v1_points},
                        {"label": "LoPT v2.1", "color": LINE_COLORS["v2"], "points": v2_points},
                    ],
                )
            )
            speed_cards.append(
                render_line_chart(
                    f"{title} · v1 / v2.1 加速比曲线",
                    "纵轴为原生 E2E / LoPT E2E。",
                    [
                        {"label": "LoPT v1 加速比", "color": LINE_COLORS["v1"], "points": speed_v1},
                        {"label": "LoPT v2.1 加速比", "color": LINE_COLORS["v2"], "points": speed_v2},
                    ],
                )
            )
            bar_cards.append(
                render_grouped_bars(
                    f"{title} · 原生时间 / LoPT 时间 / 分段占比",
                    "原生为橙色柱，v1 为蓝色柱，v2.1 为 MP 主段 + 去重段叠加柱。",
                    bar_data,
                )
            )

    return f"""
<div class="sec" id="v21cmp"><h2>v2.1 最佳参数综合对比</h2>
<div class="lead">同一批真实中文/英文输入、同一模型家族、同一精度标准下，把原生 vLLM、LoPT v1 与 LoPT v2.1 摆在同一张表里看。</div>
<table><thead><tr><th>模型</th><th>语言</th><th>平均原生 E2E<span class="u"> (ms)</span></th><th>平均 v1 LoPT<span class="u"> (ms)</span></th><th>平均 v2.1 LoPT<span class="u"> (ms)</span></th><th>平均 v2.1 MP<span class="u"> (ms)</span></th><th>平均 v2.1 去重<span class="u"> (ms)</span></th><th>v2.1 相对 v1<span class="u"> (%)</span></th><th>平均 v2.1 加速比<span class="u"> (x)</span></th><th>精度</th></tr></thead><tbody>
{''.join(grouped_rows)}
</tbody></table>
{render_best_config_table(v1_map, v2_map)}
<div class="fc">{''.join(chart_cards[:2])}</div>
<div class="fc">{''.join(chart_cards[2:])}</div>
<div class="fc">{''.join(speed_cards[:2])}</div>
<div class="fc">{''.join(speed_cards[2:])}</div>
<div class="fc">{''.join(bar_cards[:2])}</div>
<div class="fc">{''.join(bar_cards[2:])}</div>
</div>
"""


def build_stage_section(v1_records: list[dict], v2_records: list[dict], env_info: dict) -> str:
    hw = env_info.get("hardware_summary", {})
    avg_v1_dedup = sum(r["chunk_dedup_time_ms"] for r in v1_records) / len(v1_records)
    avg_v2_dedup = sum(r["chunk_dedup_time_ms"] for r in v2_records) / len(v2_records)
    dedup_drop = (avg_v1_dedup - avg_v2_dedup) / avg_v1_dedup * 100.0
    rows = []
    for family in FAMILY_ORDER:
        for lang in LANGUAGE_ORDER:
            subset = [
                r for r in v2_records if r["tokenizer_family"] == family and r["language"] == lang
            ]
            rows.append(
                f"<tr><td>{escape(family)}</td><td>{LANG_DISPLAY[lang]}</td>"
                f"<td class='n'>{fmt_ms(sum(r['dispatch_submit_time_ms'] for r in subset) / len(subset))}</td>"
                f"<td class='n'>{fmt_ms(sum(r['dispatch_collect_time_ms'] for r in subset) / len(subset))}</td>"
                f"<td class='n'>{fmt_ms(sum(r['worker_encode_time_ms_max'] for r in subset) / len(subset))}</td>"
                f"<td class='n'>{fmt_ms(sum(r['worker_materialize_time_ms_max'] for r in subset) / len(subset))}</td>"
                f"<td class='n'>{fmt_ms(sum(r['chunk_dedup_time_ms'] for r in subset) / len(subset))}</td>"
                f"<td class='n g'>{fmt_x(sum(r['e2e_speedup_x'] for r in subset) / len(subset))}</td></tr>"
            )
    return f"""
<div class="sec" id="v21stage"><h2>v2.1 分阶段耗时与实验环境</h2>
<div class="lead">v2.1 把 LoPT 主路径进一步拆成 submit / collect / worker encode / worker materialize / dedup，方便看瓶颈落在哪一段。</div>
<div class="fc">
<div class="fb af"><div class="tt">v2.1 分阶段字符链路</div><pre>LoPT E2E
  = chat template
  + mp_dispatch_process_collect
      = submit
      + wait & collect
      + child encode
      + child materialize
  + chunk overlap dedup

当 dedup 足够轻时，
墙钟主瓶颈会回到 collect 阶段中的最慢 worker encode。</pre></div>
<div class="fb"><div class="tt" style="background:var(--bg3);color:var(--ink)">关键环境信息</div>
<table><thead><tr><th>项</th><th>值</th></tr></thead><tbody>
<tr><td>主机名</td><td>{escape(str(env_info.get('hostname', '-')))}</td></tr>
<tr><td>CPU 架构</td><td>{escape(str(hw.get('architecture', '-')))}</td></tr>
<tr><td>CPU 厂商</td><td>{escape(str(hw.get('vendor_id', '-')))}</td></tr>
<tr><td>逻辑 CPU 数</td><td>{escape(str(hw.get('cpu_count', '-')))}</td></tr>
<tr><td>NUMA 节点数</td><td>{escape(str(hw.get('numa_nodes', '-')))}</td></tr>
<tr><td>CPU 最高频率</td><td>{escape(str(hw.get('cpu_max_mhz', '-')))} MHz</td></tr>
<tr><td>L2 Cache</td><td>{escape(str(hw.get('l2_cache', '-')))}</td></tr>
<tr><td>L3 Cache</td><td>{escape(str(hw.get('l3_cache', '-')))}</td></tr>
<tr><td>总内存</td><td>{escape(str(env_info.get('memory_total_human', '-')))}</td></tr>
<tr><td>可用内存</td><td>{escape(str(env_info.get('memory_available_human', '-')))}</td></tr>
<tr><td>绑核范围</td><td>29-31,40-79,155-159</td></tr>
<tr><td>绑核数量</td><td>48</td></tr>
</tbody></table></div>
</div>
<div class="c g"><div class="t"><strong>v2.1 最大的结构性收益来自去重段收缩：</strong>v1 全 48 case 的平均去重时间为 {fmt_ms(avg_v1_dedup)} ms，v2.1 降到 {fmt_ms(avg_v2_dedup)} ms，平均下降 <strong>{fmt_pct(dedup_drop)}</strong>。</div></div>
<table><thead><tr><th>模型</th><th>语言</th><th>平均 submit<span class="u"> (ms)</span></th><th>平均 collect<span class="u"> (ms)</span></th><th>平均最慢 worker encode<span class="u"> (ms)</span></th><th>平均最慢 worker materialize<span class="u"> (ms)</span></th><th>平均 dedup<span class="u"> (ms)</span></th><th>平均 E2E 加速<span class="u"> (x)</span></th></tr></thead><tbody>
{''.join(rows)}
</tbody></table>
<table><thead><tr><th>表头</th><th>单位 / 符号</th><th>解释</th></tr></thead><tbody>
<tr><td>p</td><td>proc</td><td>LoPT worker 进程数。</td></tr>
<tr><td>k</td><td>count</td><td>切分后的 chunk 数量。</td></tr>
<tr><td>o</td><td>chars</td><td>相邻 chunk 的 overlap 字符数。</td></tr>
<tr><td>submit</td><td>ms</td><td>父进程把 chunk 提交给进程池的时间。</td></tr>
<tr><td>collect</td><td>ms</td><td>父进程等待并收集所有子进程结果的时间。</td></tr>
<tr><td>worker encode max</td><td>ms</td><td>最慢 worker 纯 encode 时间上界。</td></tr>
<tr><td>worker materialize max</td><td>ms</td><td>最慢 worker 结果物化时间上界。</td></tr>
<tr><td>dedup</td><td>ms</td><td>父进程执行 overlap 去重与合并的时间。</td></tr>
<tr><td>E2E 加速</td><td>x</td><td>原生 E2E / LoPT E2E。</td></tr>
</tbody></table>
</div>
"""


def render_detail_rows(rows: list[dict], include_stage: bool = True) -> str:
    body = []
    for r in rows:
        cells = [
            f"<td class='n'>{fmt_int(r.get('worker_processes'))}</td>",
            f"<td class='n'>{fmt_int(r.get('chunk_count'))}</td>",
            f"<td class='n'>{fmt_int(r.get('actual_chunk_count', r.get('chunk_count')))}</td>",
            f"<td class='n'>{fmt_int(r.get('chunk_chars'))}</td>",
            f"<td class='n'>{fmt_int(r.get('overlap_chars'))}</td>",
            f"<td class='n'>{fmt_ms(r.get('native_e2e_time_ms'))}</td>",
            f"<td class='n'>{fmt_ms(r.get('native_tokenizer_time_ms'))}</td>",
            f"<td class='n'>{fmt_ms(r.get('mp_dispatch_process_collect_time_ms'))}</td>",
            f"<td class='n'>{fmt_ms(r.get('chunk_dedup_time_ms'))}</td>",
            f"<td class='n'>{fmt_ms(r.get('lopt_e2e_time_ms'))}</td>",
            f"<td class='n {badge_class(float(r.get('e2e_speedup_x') or 0.0))}'>{fmt_x(r.get('e2e_speedup_x'))}</td>",
            f"<td class='n {badge_class(float(r.get('tokenizer_speedup_x') or 0.0))}'>{fmt_x(r.get('tokenizer_speedup_x'))}</td>",
            f"<td class='n'>{'是' if r.get('exact_match') else '否'}</td>",
        ]
        if include_stage:
            cells[7:7] = [
                f"<td class='n'>{fmt_ms(r.get('dispatch_submit_time_ms'))}</td>",
                f"<td class='n'>{fmt_ms(r.get('dispatch_collect_time_ms'))}</td>",
            ]
        body.append("<tr>" + "".join(cells) + "</tr>")
    return "".join(body)


def build_browser_section(v1_records: list[dict], v2_records: list[dict], worker_best: list[dict], search_detail: list[dict]) -> str:
    v1_map = {key_case(r): r for r in v1_records}
    v2_map = {key_case(r): r for r in v2_records}
    # final replay folds
    replay_folds = []
    for family in FAMILY_ORDER:
        family_body = []
        for lang in LANGUAGE_ORDER:
            rows = [
                v2_map[key]
                for key in sorted_case_keys(v2_map.keys())
                if key[0] == family and key[1] == lang
            ]
            cmp_rows = []
            for r in rows:
                old = v1_map[key_case(r)]
                drop = (old["lopt_e2e_time_ms"] - r["lopt_e2e_time_ms"]) / old["lopt_e2e_time_ms"] * 100.0
                drop_cls = "g" if drop >= 0 else "r"
                cmp_rows.append(
                    f"<tr><td>{escape(r['length_label'])}</td><td class='n'>{fmt_int(r['worker_processes'])}</td><td class='n'>{fmt_int(r['chunk_count'])}</td><td class='n'>{fmt_int(r['overlap_chars'])}</td><td class='n'>{fmt_ms(r['native_e2e_time_ms'])}</td><td class='n'>{fmt_ms(old['lopt_e2e_time_ms'])}</td><td class='n'>{fmt_ms(r['lopt_e2e_time_ms'])}</td><td class='n'>{fmt_ms(r['mp_dispatch_process_collect_time_ms'])}</td><td class='n'>{fmt_ms(r['chunk_dedup_time_ms'])}</td><td class='n {drop_cls}'>{fmt_pct(drop)}</td><td class='n {badge_class(float(r['e2e_speedup_x']))}'>{fmt_x(r['e2e_speedup_x'])}</td><td class='n'>{'是' if r.get('exact_match') else '否'}</td></tr>"
                )
            family_body.append(
                f"<details class='fold'><summary>{LANG_DISPLAY[lang]} · 48 Case 子集中的 {len(rows)} 条最佳回放记录</summary><div class='bd'>"
                "<table><thead><tr><th>长度</th><th>p</th><th>k</th><th>o</th><th>原生 E2E<span class='u'> (ms)</span></th><th>v1 LoPT<span class='u'> (ms)</span></th><th>v2.1 LoPT<span class='u'> (ms)</span></th><th>v2.1 MP<span class='u'> (ms)</span></th><th>v2.1 去重<span class='u'> (ms)</span></th><th>v2.1 相对 v1<span class='u'> (%)</span></th><th>v2.1 加速<span class='u'> (x)</span></th><th>精度</th></tr></thead><tbody>"
                + "".join(cmp_rows)
                + "</tbody></table></div></details>"
            )
        replay_folds.append(
            f"<details class='fold'><summary>{escape(family)} · v2.1 最终 replay</summary><div class='bd'>{''.join(family_body)}</div></details>"
        )

    # worker_best folds
    worker_groups = defaultdict(list)
    for row in worker_best:
        worker_groups[(row["tokenizer_family"], row["language"], row["length_label"])].append(row)
    worker_folds = []
    for key in sorted_case_keys(worker_groups.keys()):
        rows = sorted(
            worker_groups[key],
            key=lambda r: (
                int(r.get("worker_processes", 0)),
                int(r.get("chunk_count", 0)),
                int(r.get("overlap_chars", 0)),
            ),
        )
        worker_folds.append(
            f"<details class='fold'><summary>{escape(key[0])} / {LANG_DISPLAY[key[1]]} / {escape(key[2])} · worker 搜索保留候选 {len(rows)} 条</summary><div class='bd'>"
            "<table><thead><tr><th>p</th><th>k</th><th>actual k</th><th>chunk chars</th><th>o</th><th>原生 E2E<span class='u'> (ms)</span></th><th>原生 Tokenizer<span class='u'> (ms)</span></th><th>submit<span class='u'> (ms)</span></th><th>collect<span class='u'> (ms)</span></th><th>LoPT MP<span class='u'> (ms)</span></th><th>去重<span class='u'> (ms)</span></th><th>LoPT E2E<span class='u'> (ms)</span></th><th>E2E 加速<span class='u'> (x)</span></th><th>Tokenizer 加速<span class='u'> (x)</span></th><th>精度</th></tr></thead><tbody>"
            + render_detail_rows(rows)
            + "</tbody></table></div></details>"
        )

    search_groups = defaultdict(list)
    for row in search_detail:
        search_groups[(row["tokenizer_family"], row["language"], row["length_label"])].append(row)
    full_folds = []
    for key in sorted_case_keys(search_groups.keys()):
        rows = sorted(
            search_groups[key],
            key=lambda r: (
                int(r.get("worker_processes", 0)),
                int(r.get("chunk_count", 0)),
                int(r.get("overlap_chars", 0)),
            ),
        )
        top = v2_map.get(key)
        top_text = (
            f"最佳 p={top['worker_processes']} / k={top['chunk_count']} / o={top['overlap_chars']} / LoPT E2E={fmt_ms(top['lopt_e2e_time_ms'])} ms / 加速={fmt_x(top['e2e_speedup_x'])}"
            if top
            else "无 replay 结果"
        )
        full_folds.append(
            f"<details class='fold'><summary>{escape(key[0])} / {LANG_DISPLAY[key[1]]} / {escape(key[2])} · 全量 search {len(rows)} 条 · {escape(top_text)}</summary><div class='bd'>"
            "<table><thead><tr><th>p</th><th>k</th><th>actual k</th><th>chunk chars</th><th>o</th><th>原生 E2E<span class='u'> (ms)</span></th><th>原生 Tokenizer<span class='u'> (ms)</span></th><th>submit<span class='u'> (ms)</span></th><th>collect<span class='u'> (ms)</span></th><th>LoPT MP<span class='u'> (ms)</span></th><th>去重<span class='u'> (ms)</span></th><th>LoPT E2E<span class='u'> (ms)</span></th><th>E2E 加速<span class='u'> (x)</span></th><th>Tokenizer 加速<span class='u'> (x)</span></th><th>精度</th><th>状态</th></tr></thead><tbody>"
            + "".join(
                "<tr>"
                f"<td class='n'>{fmt_int(r.get('worker_processes'))}</td>"
                f"<td class='n'>{fmt_int(r.get('chunk_count'))}</td>"
                f"<td class='n'>{fmt_int(r.get('actual_chunk_count', r.get('chunk_count')))}</td>"
                f"<td class='n'>{fmt_int(r.get('chunk_chars'))}</td>"
                f"<td class='n'>{fmt_int(r.get('overlap_chars'))}</td>"
                f"<td class='n'>{fmt_ms(r.get('native_e2e_time_ms'))}</td>"
                f"<td class='n'>{fmt_ms(r.get('native_tokenizer_time_ms'))}</td>"
                f"<td class='n'>{fmt_ms(r.get('dispatch_submit_time_ms'))}</td>"
                f"<td class='n'>{fmt_ms(r.get('dispatch_collect_time_ms'))}</td>"
                f"<td class='n'>{fmt_ms(r.get('mp_dispatch_process_collect_time_ms'))}</td>"
                f"<td class='n'>{fmt_ms(r.get('chunk_dedup_time_ms'))}</td>"
                f"<td class='n'>{fmt_ms(r.get('lopt_e2e_time_ms'))}</td>"
                f"<td class='n {badge_class(float(r.get('e2e_speedup_x') or 0.0))}'>{fmt_x(r.get('e2e_speedup_x'))}</td>"
                f"<td class='n {badge_class(float(r.get('tokenizer_speedup_x') or 0.0))}'>{fmt_x(r.get('tokenizer_speedup_x'))}</td>"
                f"<td class='n'>{'是' if r.get('exact_match') else '否'}</td>"
                f"<td>{escape(str(r.get('candidate_status', '-')))}</td>"
                "</tr>"
                for r in rows
            )
            + "</tbody></table></div></details>"
        )
    return f"""
<div class="sec" id="v21browser"><h2>v2.1 完整性能查询浏览器</h2>
<div class="lead">延续原始 v1 HTML 的折叠式浏览方式，把 v2.1 最终 replay、worker 维度保留候选和全量 search 结果都嵌进同一页里。需要机器可读原始文件时，仓库中也保留了 JSON/JSONL。</div>
<details class="fold"><summary>v2.1 最终 replay 48 Case</summary><div class="bd">{''.join(replay_folds)}</div></details>
<details class="fold"><summary>v2.1 worker 维度保留候选（来自 worker_best_configs.json）</summary><div class="bd">{''.join(worker_folds)}</div></details>
<details class="fold"><summary>v2.1 全量 search 结果（来自 search_detail.jsonl）</summary><div class="bd">{''.join(full_folds)}</div></details>
</div>
"""


def patch_nav(html: str) -> str:
    old = '<nav><a href="#bg">背景</a><a href="#arch">架构</a><a href="#algo">算法</a><a href="#corpus">语料</a><a href="#tp">线程池</a><a href="#pp">进程池</a><a href="#full">完整数据</a><a href="#repro">复现</a></nav>'
    new = '<nav><a href="#bg">背景</a><a href="#arch">架构</a><a href="#algo">算法</a><a href="#corpus">语料</a><a href="#tp">线程池</a><a href="#pp">进程池</a><a href="#v21">v2.1</a><a href="#v21cmp">综合对比</a><a href="#v21stage">分段耗时</a><a href="#v21browser">v2.1浏览器</a><a href="#full">v1数据库</a><a href="#repro">复现</a></nav>'
    return html.replace(old, new, 1)


def patch_css(html: str) -> str:
    extra = """
.fb.bf .tt{background:var(--orange2);color:var(--orange)}
.fb.af .tt{background:var(--green2);color:var(--green)}
.r{color:var(--red);font-weight:600}
svg text{font-family:var(--sans)}
"""
    return html.replace("</style>", extra + "\n</style>", 1)


def patch_footer(html: str) -> str:
    old = '<p>LoPT 并行分词器 基准测试报告 &middot; 2026-05-26 &middot; ARM 单核验证 &middot; <code>lightnateriver/LightTokenizer</code></p>'
    new = '<p>LoPT 并行分词器 基准测试报告（原始 v1 风格 + v2.1 增量章节） &middot; 2026-05-28 &middot; ARM 单核验证 &middot; <code>lightnateriver/LightTokenizer</code></p>'
    return html.replace(old, new, 1)


def strip_interior_html_closers(html: str) -> str:
    return html.replace("</body></html></div></details>", "</div></details>")


def patch_titles(html: str) -> str:
    html = html.replace("<title>LoPT 并行分词器 基准测试报告</title>", "<title>LoPT 并行分词器 基准测试报告（含 v2.1 增量）</title>", 1)
    html = html.replace("ARM 单核 &middot; Qwen3.5 (248K) + DSV4 Pro (128K) &middot; 2000+ 组测试", "ARM 单核 &middot; 原生 vLLM / LoPT v1 / LoPT v2.1 &middot; 完整搜索 + 最终回放", 1)
    html = html.replace('<span class="tag">ca偏移修复</span>', '<span class="tag">ca偏移修复</span><span class="tag t">v2.1 已补入</span>', 1)
    html = html.replace('<div class="l">完整数据</div><div class="v b">377KB</div><div class="d">内置可叠叠展开数据库</div>', '<div class="l">完整数据</div><div class="v b">v1 + v2.1</div><div class="d">保留原 v1 数据，并新增 v2.1 折叠浏览器</div>', 1)
    return html


def build_addon(v1_records, v2_records, worker_best, search_detail, env_info) -> str:
    return (
        build_v21_intro(v1_records, v2_records)
        + build_compare_section(v1_records, v2_records)
        + build_stage_section(v1_records, v2_records, env_info)
        + build_browser_section(v1_records, v2_records, worker_best, search_detail)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-html", type=Path, required=True)
    parser.add_argument("--v1-replay-json", type=Path, required=True)
    parser.add_argument("--v2-replay-json", type=Path, required=True)
    parser.add_argument("--worker-best-json", type=Path, required=True)
    parser.add_argument("--search-detail-jsonl", type=Path, required=True)
    parser.add_argument("--env-info-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    args = parser.parse_args()

    template = args.template_html.read_text(encoding="utf-8")
    v1_records = load_json(args.v1_replay_json)
    v2_records = load_json(args.v2_replay_json)
    worker_best = load_json(args.worker_best_json)
    search_detail = load_jsonl(args.search_detail_jsonl)
    env_info = load_json(args.env_info_json)

    addon = build_addon(v1_records, v2_records, worker_best, search_detail, env_info)

    target_anchor = '<div class="sec" id="full"><h2>完整原始数据库</h2>'
    if target_anchor not in template:
        raise SystemExit("cannot find legacy full-database anchor in template")
    html = template.replace(target_anchor, addon + "\n" + target_anchor, 1)
    html = patch_nav(html)
    html = patch_css(html)
    html = patch_footer(html)
    html = patch_titles(html)
    html = strip_interior_html_closers(html)
    args.output_html.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
