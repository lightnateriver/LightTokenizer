# 实现细节

## 算法流程

```
输入文本 S（长度 N 字符）
│
├── 按 Lc 拆分为 chunk（每个取 Lc+LO 字符）
├── 每个 chunk 独立 tokenize（ThreadPool 并行，Rust 释放 GIL）
├── 用 offsets 在相邻 chunk 重叠区找 anchor（最长匹配 token 序列）
├── 根据 anchor 确定每个 chunk 的贡献段
└── 拼接为完整 token ID 列表
```

## Anchor 查找修复

**原始问题：** `_find_anchor` 只比较 `pa[pv+k] == ca[k]`，其中 `ca` 始终从索引 0 开始。
但 BPE 在 chunk 边界处可能偏移 1-2 个 token，导致 ca 索引错位 → 所有 anchor 失败 → fallback 串行。

**修复方案：** 增加 ca 偏移循环：

```python
for pv in range(len(pa)-1, -1, -1):
    for qv in range(len(ca)-1, -1, -1):   # ← 尝试 ca 偏移
        mk = min(len(pa)-pv, len(ca)-qv)
        k = 0
        while k < mk and pa[pv+k] == ca[qv+k]:
            k += 1
        if k > bl:
            bl, bp, bq = k, pv, qv
```

## ThreadPool vs ProcessPool

| 方面 | ProcessPool（论文） | ThreadPool（本方案） |
|:--|:--|:--|
| 数据传递 | pickle → IPC → unpickle（每 chunk 4 次拷贝） | 共享内存零拷贝 |
| GIL | 子进程无 GIL 限制 | Rust tokenize 释放 GIL |
| 单核 512K 实测 | 4300ms（4.33×） | 778ms（0.78×） |
| 多核扩展 | IPC 开销被摊薄 | 受 GIL 限制（merge 阶段） |
