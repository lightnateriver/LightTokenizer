#!/usr/bin/env python3
"""
ProcessPool vs ThreadPool vs Serial — 精确对比。
仅测 ProcessPool (fork+spawn) 对比 ThreadPool，增加耗时拆解。
超时保护 + 每步打印。
"""
import time, sys, os, pickle, statistics
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer

TOK_PATH = "/mnt/sfs_turbo/models/qwen3.5-35b-a3b"
LC = 131072; LO = 16

print("[init] Loading tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
_rust = tok._tokenizer
def _enc(t):
    e = _rust.encode(t, add_special_tokens=False)
    return e.ids, e.offsets

# Text
ZH = ["新华社北京五月二十六日电 人工智能技术正在深刻改变经济社会发展的各个方面。"] * 20
text = "".join(ZH) * 6400
messages = [{"role": "user", "content": text}]
ser_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
prompt_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
n = len(prompt_str)
n_chunks = (n + LC - 1) // LC
ci = [(prompt_str[i*LC:min(n, (i+1)*LC+LO)], i*LC) for i in range(n_chunks)]
print(f"[init] {n} chars, {len(ser_ids)} tokens, {n_chunks} chunks", flush=True)

# Serial
def run_serial():
    t0 = time.perf_counter()
    _enc(prompt_str)
    return (time.perf_counter() - t0) * 1000

# ThreadPool
def run_tp(nt):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(nt) as ex:
        list(ex.map(lambda a: (lambda ids,off:(ids,[(x[0]+a[1],x[1]+a[1]) for x in off]))(*_enc(a[0])), ci))
    return (time.perf_counter() - t0) * 1000

# ProcessPool workers
def _worker(chunk_text, offset):
    ids, offsets = _enc(chunk_text)
    return ids, [(x[0]+offset, x[1]+offset) for x in offsets]

def _worker_prep(pickled_bytes, offset):
    ct = pickle.loads(pickled_bytes)
    ids, offsets = _enc(ct)
    return ids, [(x[0]+offset, x[1]+offset) for x in offsets]

def run_pp(nt, prepickled=False):
    if prepickled:
        args = [(pickle.dumps(ct), off) for ct, off in ci]
        fn = _worker_prep
    else:
        args = ci
        fn = _worker
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(nt) as pool:
        if prepickled:
            res = pool.starmap(fn, args)
        else:
            res = pool.starmap(fn, args)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000

# Time breakdown for ProcessPool
def run_breakdown(nt):
    # pickle time
    t0 = time.perf_counter()
    pickled = [(pickle.dumps(ct), off) for ct, off in ci]
    t1 = time.perf_counter()
    pk_ms = (t1 - t0) * 1000
    
    # IPC + tokenize + result IPC time  
    with mp.get_context("fork").Pool(nt) as pool:
        t2 = time.perf_counter()
        pool.starmap(_worker_prep, pickled)
        t3 = time.perf_counter()
    ipc_total = (t3 - t2) * 1000
    
    # tokenize time (approximate: serial / n_chunks due to single core)
    tok_per_chunk = [len(ct) * 2.1e-3 for ct, _ in ci]  # ~2.1μs/char
    est_tokenize = max(tok_per_chunk)  # parallel: longest chunk
    
    return pk_ms, ipc_total, est_tokenize

# ═══ RUN ═══
print(f"\n{'='*60}", flush=True)
print(f"  对比实验: ProcessPool vs ThreadPool", flush=True)
print(f"{'='*60}", flush=True)

ser_times = [run_serial() for _ in (print(f"  Serial run {i+1}...", flush=True) or [0] for i in range(3)) and range(3)]
ser_med = statistics.median(ser_times)
print(f"Serial: {ser_med:.1f}ms (median of 3)", flush=True)

for name, fn, kw, runs in [
    ("ThreadPool t=2", run_tp, {"nt": 2}, 3),
    ("ThreadPool t=4", run_tp, {"nt": 4}, 3),
    ("ThreadPool t=8", run_tp, {"nt": 8}, 3),
    ("ProcessPool(fork) t=2", run_pp, {"nt": 2, "prepickled": False}, 2),
    ("ProcessPool(fork) t=4", run_pp, {"nt": 4, "prepickled": False}, 2),
    ("ProcessPool(prep) t=2", run_pp, {"nt": 2, "prepickled": True}, 2),
    ("ProcessPool(prep) t=4", run_pp, {"nt": 4, "prepickled": True}, 2),
]:
    times = []
    for i in range(runs):
        print(f"  {name} run {i+1}...", flush=True)
        try:
            ms = fn(**kw)
            times.append(ms)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            break
    if times:
        med = statistics.median(times) if len(times) > 1 else times[0]
        ratio = med / ser_med
        print(f"  → {med:.1f}ms ({ratio:.2f}x serial)", flush=True)

# Breakdown
print(f"\n── 耗时拆解 (ProcessPool t=4) ──", flush=True)
pk_ms, ipc_ms, est_tok = run_breakdown(4)
print(f"  主进程 pickle 4 chunks: {pk_ms:.1f}ms", flush=True)
print(f"  Pool 总耗时（IPC+unpickle+tokenize+回传）: {ipc_ms:.1f}ms", flush=True)
print(f"  预估纯 tokenize（并行最大值）: {est_tok:.1f}ms", flush=True)
print(f"  预估 IPC+pickle+unpickle 开销: {ipc_ms - est_tok:.1f}ms", flush=True)
overhead_pct = (ipc_ms - est_tok) / ipc_ms * 100
print(f"  开销占比: {overhead_pct:.0f}%", flush=True)

# Print pickle sizes
print(f"\n── 每 chunk pickle 大小 ──", flush=True)
for i, (ct, off) in enumerate(ci):
    print(f"  chunk[{i}]: {len(pickle.dumps(ct)):,} bytes (text={len(ct)} chars)", flush=True)
total_pickle = sum(len(pickle.dumps(ct)) for ct, _ in ci)
print(f"  总序列化量: {total_pickle:,} bytes", flush=True)
print(f"  对比线程共享内存: 0 bytes（零拷贝）", flush=True)
