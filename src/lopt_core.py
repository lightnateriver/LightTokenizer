"""
LoPT 核心实现 — Lossless Parallel Tokenization.

支持 ThreadPool（小 chunk 数）和 ProcessPool spawn（大 chunk 数）自动选择。
提供统一接口 run_lopt()，按需自动选择最佳池类型。

用法:
  from lopt_core import run_lopt
  result = run_lopt(prompt_str, ser_ids, Lc=65536, LO=16, pool="auto")
"""

import bisect, time, multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Tuple, Optional

from tokenizer_utils import _enc, _enc_ids

# ═══════════════════════════════════════════════════════════════
# Anchor 查找
# ═══════════════════════════════════════════════════════════════

def find_anchor(
    off_p: List[Tuple[int, int]],
    off_c: List[Tuple[int, int]],
    ostart: int,
    oend: int,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    在前后 chunk 的重叠区找最长匹配 token 序列（anchor）。

    参数:
        off_p: 前一个 chunk 的 offsets 列表 [(start,end), ...]
        off_c: 后一个 chunk 的 offsets 列表
        ostart: 重叠区起始字符位置
        oend: 重叠区结束字符位置

    返回:
        (pa 中 anchor 起始索引, anchor 长度, ca 中 anchor 起始索引)
        如果没找到返回 (None, None, None)

    注意:
        核心修复：双重循环尝试 ca 偏移 (pv, qv)，
        解决 BPE 在 chunk 边界处偏移 1-2 个 token 的问题。
    """
    n_p, n_c = len(off_p), len(off_c)

    # 找到 pa 中重叠区起始
    l = bisect.bisect_left(off_p, (ostart, 0))
    ps = None
    if l > 0 and off_p[l-1][1] > ostart:
        ps = l - 1
    elif l < n_p and off_p[l][0] < oend:
        ps = l
    if ps is None:
        return None, None, None

    # 找到 ca 中重叠区结束
    r = bisect.bisect_left(off_c, (oend, 0))
    ce = r - 1
    if ce >= 0 and off_c[ce][1] <= ostart:
        while ce >= 0 and off_c[ce][1] <= ostart:
            ce -= 1
    if ce < 0:
        return None, None, None

    pa = off_p[ps:]
    ca = off_c[:ce+1]
    bl, bp, bq = 0, None, None

    # ★ 关键修复：尝试 ca 偏移（qv 循环）
    for pv in range(len(pa)-1, -1, -1):
        for qv in range(len(ca)-1, -1, -1):
            mk = min(len(pa)-pv, len(ca)-qv)
            if mk <= bl:
                continue
            k = 0
            while k < mk and pa[pv+k] == ca[qv+k]:
                k += 1
            if k > bl:
                bl, bp, bq = k, pv, qv

    if bl == 0:
        return None, None, None
    return ps + bp, bl, bq


# ═══════════════════════════════════════════════════════════════
# Merge
# ═══════════════════════════════════════════════════════════════

def merge_chunks(
    res: List[Tuple[List[int], List[Tuple[int, int]]]],
    anchors: List[Tuple[int, int, int]],
) -> List[int]:
    """
    通过 anchor 确定每个 chunk 的贡献段，拼接为完整 ids。

    参数:
        res: [(ids, offsets), ...] 每个 chunk 的 tokenize 结果
        anchors: [(pa_start, length, ca_start), ...] 每对相邻 chunk 的 anchor

    返回:
        完整 token ID 列表，与串行 tokenize 结果一致
    """
    segs = []
    tl = 0

    # 第一个 chunk: 取到第一个 anchor 覆盖的位置
    s = res[0][0][:anchors[0][0] + anchors[0][1]]
    segs.append(s)
    tl += len(s)

    # 中间 chunk: anchor 之间取不重叠的新增部分
    for i in range(1, len(res) - 1):
        st = anchors[i-1][2] + anchors[i-1][1]
        en = anchors[i][0] + anchors[i][1]
        if en > st:
            s = res[i][0][st:en]
            segs.append(s)
            tl += len(s)

    # 最后一个 chunk: 取最后一个 anchor 之后的部分
    s = res[-1][0][anchors[-1][2] + anchors[-1][1]:]
    segs.append(s)
    tl += len(s)

    # 拼接
    m = [0] * tl
    p = 0
    for seg in segs:
        m[p:p+len(seg)] = seg
        p += len(seg)
    return m


# ═══════════════════════════════════════════════════════════════
# 并行 tokenize
# ═══════════════════════════════════════════════════════════════

def _build_chunks(prompt_str: str, Lc: int, LO: int):
    """将文本切分为重叠 chunk。"""
    n = len(prompt_str)
    n_chunks = (n + Lc - 1) // Lc
    return [(prompt_str[i*Lc:min(n, (i+1)*Lc+LO)], i*Lc) for i in range(n_chunks)]


def _tokenize_one(args: Tuple[str, int]) -> Tuple[List[int], List[Tuple[int, int]]]:
    """单个 chunk 的 tokenize 函数（用于 ProcessPool）。"""
    text, offset = args
    ids, offsets = _enc(text)
    return ids, [(x[0]+offset, x[1]+offset) for x in offsets]


def _init_worker():
    """ProcessPool worker 初始化（已在 tokenizer_utils 中全局初始化）。"""
    pass


def run_threadpool(chunks: list, n_threads: int) -> list:
    """ThreadPool 并行 tokenize。"""
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        return list(ex.map(lambda a: (
            lambda ids, off: (ids, [(x[0]+a[1], x[1]+a[1]) for x in off])
        )(*_enc(a[0])), chunks))


def run_processpool(chunks: list, n_workers: int) -> list:
    """ProcessPool(spawn) 并行 tokenize。"""
    with mp.get_context("spawn").Pool(n_workers) as pool:
        return pool.map(_tokenize_one, chunks)


def select_pool(n_chunks: int) -> str:
    """
    根据 chunk 数自动选择最佳池类型。
    ≤10: ThreadPool（零 IPC 开销）
    ≥20: ProcessPool（独立 GIL）
    中间: 都可（实测 ThreadPool 略好）
    """
    if n_chunks <= 10:
        return "thread"
    elif n_chunks >= 20:
        return "process"
    else:
        return "thread"  # 默认 ThreadPool


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def run_lopt(
    prompt_str: str,
    ser_ids: List[int],
    Lc: int = 65536,
    LO: int = 16,
    pool: str = "auto",  # "thread" | "process" | "auto"
    n_workers: int = 16,
) -> dict:
    """
    执行一次 LoPT tokenize 并返回结果和时间统计。

    参数:
        prompt_str: 输入文本字符串
        ser_ids: 串行 tokenize 结果（用于精度验证）
        Lc: chunk 大小（字符数）
        LO: 重叠大小（字符数）
        pool: 池类型 ("thread" | "process" | "auto")
        n_workers: 并行 worker 数

    返回:
        dict with keys:
          ids: token ID 列表（与 ser_ids 一致则正确）
          correct: bool
          has_fallback: bool
          pool_ms: 并行 tokenize 阶段耗时
          merge_ms: anchor 搜索 + 拼接耗时
          total_ms: 总耗时
          n_chunks: chunk 数
          avg_anchor: 平均 anchor 长度
          anc_fails: anchor 失败数
          pool_type: 实际使用的池类型
    """
    n = len(prompt_str)
    n_chunks = (n + Lc - 1) // Lc

    if n_chunks <= 1:
        return {"correct": False, "has_fallback": False,
                "pool_ms": 0, "merge_ms": 0, "total_ms": 0,
                "n_chunks": 1, "avg_anchor": 0, "anc_fails": 0,
                "pool_type": "none", "ids": ser_ids}

    chunks = _build_chunks(prompt_str, Lc, LO)

    # 选择池类型
    if pool == "auto":
        pool_type = select_pool(n_chunks)
    else:
        pool_type = pool

    # 并行 tokenize
    t0 = time.perf_counter()
    if pool_type == "process":
        res = run_processpool(chunks, n_workers)
    else:
        n_t = min(n_workers, n_chunks)
        res = run_threadpool(chunks, n_t)
    pool_ms = (time.perf_counter() - t0) * 1000

    # 找 anchor
    anchors = []
    anc_fails = 0
    anc_tokens = 0
    for i in range(1, len(res)):
        a_start, a_len, q_start = find_anchor(
            res[i-1][1], res[i][1], i*Lc, i*Lc+LO
        )
        if a_len is None or a_len == 0:
            anc_fails += 1
        else:
            anchors.append((a_start, a_len, q_start))
            anc_tokens += a_len

    has_fallback = anc_fails > 0
    if has_fallback:
        ids = _enc_ids(prompt_str)
    else:
        ids = merge_chunks(res, anchors)

    lopt_ms = (time.perf_counter() - t0) * 1000
    merge_ms = lopt_ms - pool_ms
    correct = ids == ser_ids
    avg_anchor = anc_tokens / max(len(anchors), 1)

    return {
        "ids": ids,
        "correct": correct,
        "has_fallback": has_fallback,
        "pool_ms": round(pool_ms, 1),
        "merge_ms": round(merge_ms, 1),
        "total_ms": round(lopt_ms, 1),
        "n_chunks": n_chunks,
        "avg_anchor": round(avg_anchor, 1),
        "anc_fails": anc_fails,
        "pool_type": pool_type,
    }
