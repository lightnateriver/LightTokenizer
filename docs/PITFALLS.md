# 已知陷阱

## 1. Anchor 查找偏移（已修复）

**表现：** 所有 anchor 失败 → fallback 串行 → 比串行更慢。

**原因：** 原始 `_find_anchor` 只比较 `pa[pv+k] == ca[k]`，ca 始终从索引 0 开始。
BPE 在 chunk 边界处可能偏移 1-2 个 token，导致序列错位。

**修复（2026-05-26）：** 增加 ca 偏移双重循环（见 `_find_anchor` 中的 `qv` 循环）。
这是整个项目最关键的单行修复。

## 2. Python 层线程安全问题

**表现：** ThreadPoolExecutor 下，部分 chunk 返回错误的 token ID。

**原因：** `transformers.PreTrainedTokenizerFast` 的 Python 包装层（缓存、`added_tokens_decoder`）
不是线程安全的，多线程并发访问会静默损坏内部状态。

**解决：** 绕过 Python 包装层，直接通过 `tok._tokenizer` 调用 Rust 底层编码器。
Rust 后端（`tokenizers` 库）是纯线程安全的。

**验证：** 每次在 ThreadPool 实验前，用 `assert rust_ids == py_ids` 确认 Rust 结果与 Python 一致。

## 3. Overlayfs 白障（容器环境）

**表现：** `OSError: [Errno 2] No such file or directory`，目录可见但不可写。

**原因：** `pip --force-reinstall` 在 overlayfs 上创建 whiteout 条目，
删除目录后无法重新创建。

**解决：**
- 创建 venv：`python3 -m venv /opt/clean-env`
- 或用 `--ignore-installed` 跳过 uninstall

## 4. ThreadPool 在单核上的收益上限

ARM 单核上 t=2 即饱和，t>2 无收益。
多核机器建议搜索 t=2~16。

## 5. ProcessPool 超出 chunk 减少的合并问题

**表现：** ProcessPool 在只创建 2-4 个 chunk 时反而比 ThreadPool 慢（warmup 开销 > 并行收益）。

**解决：** `select_pool()` 自动选择：≤10 chunk 用 ThreadPool，≥20 用 ProcessPool。

## 6. SSH 输出缓冲

**表现：** 远端运行脚本后，输出长时间不返回。

**原因：** SSH 管道的 stdout 缓冲。

**解决：** 在每个 `print()` 后加 `flush=True`；或在脚本通过 tee 重定向时等待缓冲刷新。

## 7. 短文本（<16K chars）无收益

LoPT 在短文本下加速比 < 1（池开销 > 并行收益）。
break-even 点在 ~100K tokens 以上（ThreadPool）或 ~16K tokens 以上（ProcessPool）。

## 8. 动态重试陷阱

**表现：** anchor 失败后自动加倍 Lc 重试，但 5+ 次重试后总耗时超过串行。

**原因：** 每次重试都要重新创建 pool.map，累积时间 > 串行 baseline。

**解决：** 不设重试，anchor 失败直接 fallback 串行。如果 anchor 经常失败，检查 LO 是否足够（英文至少 32 chars）。
