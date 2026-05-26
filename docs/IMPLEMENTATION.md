# 实现细节

## 算法流程

```
输入文本 S（长度 N 字符）
│
├── 按 Lc 拆分为 chunk（每个取 Lc+LO 字符）
├── 并行 tokenize（自动选择 ThreadPool / ProcessPool）
│   ├── chunk 数 ≤10  → ThreadPool（零 IPC 开销）
│   └── chunk 数 ≥20  → ProcessPool(spawn)（独立 GIL）
├── 用 offsets 在相邻 chunk 重叠区找 anchor（最长匹配 token 序列）
├── 根据 anchor 确定每个 chunk 的贡献段
└── 拼接为完整 token ID 列表
```

## Anchor 查找修复（关键突破）

**原始问题：** `_find_anchor` 只比较 `pa[pv+k] == ca[k]`，其中 `ca` 始终从索引 0 开始。
但 BPE 在 chunk 边界处可能偏移 1-2 个 token，导致整个序列错位 → 所有 anchor 失败 → fallback 串行。

```
chunk_i  overlap (有上下文):  offsets=[(169,175), (175,178), (178,182), ...]  → pa
chunk_{i+1} overlap (无上下文): offsets=[(170,171), (171,175), (175,178), ...] → ca
    
pa[0]=(169,175) ≠ ca[0]=(170,171) → 找不到 anchor！
但 pa[1]=(175,178) == ca[2]=(175,178) → 实际匹配在 2 个 token 之后
```

**修复：** 增加 ca 偏移双重循环，尝试所有 (pv, qv) 起始对：

```python
for pv in range(len(pa)-1, -1, -1):
    for qv in range(len(ca)-1, -1, -1):   # ← 尝试 ca 偏移
        mk = min(len(pa)-pv, len(ca)-qv)
        if mk <= bl: continue
        k = 0
        while k < mk and pa[pv+k] == ca[qv+k]:
            k += 1
        if k > bl:
            bl, bp, bq = k, pv, qv
```

这个单行修复（加 `qv` 循环）是整个项目的核心突破。在此之前浪费了整个 session 的优化尝试（二分搜索、int 比较、预分配、Rust 后端、动态重试）——都是在解决错误的瓶颈。

## ThreadPool vs ProcessPool

### 实验结果（ARM 单核验证）

**Qwen3.5 中文 1024K**:
| Method | 耗时 | 加速比 |
|:--|:--:|:--:|
| Serial | 2101ms | 1× |
| ThreadPool(16) | 1545ms | 1.35× |
| **ProcessPool(spawn, 16)** | **330ms** | **6.20×** |

**Qwen3.5 英文 1024K**:
| Method | 耗时 | 加速比 |
|:--|:--:|:--:|
| Serial | 3260ms | 1× |
| ThreadPool(16) | 2894ms | 1.13× |
| **ProcessPool(spawn, 16)** | **396ms** | **8.24×** |

**DSV4 Pro 英文 1024K**:
| Method | 耗时 | 加速比 |
|:--|:--:|:--:|
| Serial | 3157ms | 1× |
| ThreadPool(16) | 2790ms | 1.13× |
| **ProcessPool(spawn, 16)** | **387ms** | **8.16×** |

### 为什么 ProcessPool 在单核上也比 ThreadPool 快

1. **独立 GIL** — 每个 spawn 子进程有独立 Python 解释器，Rust tokenizer 不受 GIL 限制
2. **IPC 开销可摊薄** — pickle/pickle 固定开销约 77%，但随文本增大而占比下降
3. **OS 调度** — 即使单核，OS 可以在 tokenize 和 merge 之间时间片交错

### 自动选择规则

| Chunk 数 | 推荐 | 原因 |
|:--|:--|:--|
| ≤10 | ThreadPool | 零 IPC 开销，简单 |
| 10-20 | 都可 | 实测 ThreadPool 略好 |
| ≥20 | **ProcessPool(spawn)** | 独立 GIL 优势显著 |

### Rust 后端线程安全

`transformers.PreTrainedTokenizerFast` 的 Python 层有可变状态，**不是线程安全**的。
必须绕过 Python wrapper，直接调 Rust 底层：

```python
_rust = tok._tokenizer  # thread-safe
def _enc(text):
    e = _rust.encode(text, add_special_tokens=False)
    return e.ids, e.offsets
```

### IPC 开销拆解（37 chunks, 4 workers）

| 阶段 | 耗时 | 占比 |
|:--|:--:|:--:|
| pickle 序列化 | 5ms | 0.4% |
| IPC + unpickle + 结果回传 | 902ms | 77% |
| 纯 tokenize（并行最大值） | 275ms | 23% |
| **Pool 总耗时** | **1177ms** | 100% |

即使 77% 是 IPC 开销，ProcessPool 仍比 ThreadPool（3220ms）快，因为每个 worker 有独立 GIL。

## 已验证结果

### Qwen3.5 (248K vocab) — ThreadPool

| 规模 | Tokens | 串行(ms) | LoPT(ms) | 加速比 |
|:--|:--:|:--:|:--:|:--:|
| 128K 中文 | 127,505 | 214 | 194 | 1.11× |
| 256K 中文 | 255,593 | 451 | 386 | 1.17× |
| 512K 中文 | 511,769 | 993 | 778 | 1.28× |
| 1024K 中文 | 1,023,528 | 2101 | 1560 | **1.35×** |

### Qwen3.5 — ProcessPool(spawn, w=16)

| 规模 | 串行(ms) | LoPT(ms) | 加速比 |
|:--|:--:|:--:|:--:|
| 512K 中文 | 1001 | 190 | **5.27×** |
| 1024K 中文 | 2045 | 330 | **6.20×** |
| 512K 英文 | 1615 | 217 | **7.45×** |
| 1024K 英文 | 3260 | 396 | **8.24×** |

### DSV4 Pro (128K vocab) — ProcessPool(spawn, w=16)

| 规模 | 串行(ms) | LoPT(ms) | 加速比 |
|:--|:--:|:--:|:--:|
| 512K 中文 | 1517 | 189 | **8.21×** |
| 1024K 英文 | 3157 | 387 | **8.16×** |

### 关键发现

- **词汇表大小对 LoPT 效果影响很小** — 128K vs 248K 加速比基本一致
- **ProcessPool 单核可达 8×** — 完全不受 GIL 限制
- **ThreadPool 上限 ~1.35×** — 受 Python GIL 和 merge 阶段限制
- **Lc_opt = chars/128** 通用规则适用于所有 tokenizer
- **中文 LO=16 最优**，英文 LO=32-64
