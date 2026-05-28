# LoPT v2.0 阶段一设计文档

本文档描述 `LoPT v2.0` 的阶段一实现目标、代码结构、验证方案与验收标准。它只面向当前仓库 `/home/light/codes/LoPT` 中已经开始落地的 `stage1` 代码，不讨论未来还未实现的 `stage2/stage3` 优化细节。

---

## 1. 阶段一目标

阶段一不是追求最终最高性能，而是先把下面三件事做好：

1. 把 `v1.0` 中单文件式的 LoPT 原型拆成可维护、可扩展、可定位问题的模块化内核。
2. 显式实现基于**全局绝对字符偏移**的边界对齐逻辑，也就是我们在讨论中提到的 `ca-偏移修复` 思路。
3. 建立一套稳定的 `native vLLM / LoPT v1 / LoPT v2 stage1` 三方对比基线，用 `v1` 已搜索出的最佳参数直接做固定参数回放。

可以把阶段一概括成：

```text
做对
  + 拆清
  + 量准
```

---

## 2. 为什么要做阶段一

`v1.0` 已经证明了 LoPT 风格多进程方案在真实长文本上是可跑、可搜索、可 replay 的，但仍然有几个明显问题：

### 2.1 代码结构偏原型化

`benchmarks/lopt_tokenizer.py` 把：

- 切块
- worker 初始化
- offset 处理
- overlap 匹配
- merge
- 结果统计

都堆在一个文件里。这个版本适合快速验证，但不利于：

- 定位边界问题
- 替换 match 策略
- 补充更细粒度的性能指标
- 后续接入新实验逻辑

### 2.2 偏移逻辑虽然“在用”，但还不是显式设计

`v1.0` 已经在 `collect_overlap_tokens()` 里做了 `result.start + local_offset` 的全局化处理，但它还是隐含在局部函数里的。阶段一要把这件事升级为**明确的数据模型**：

- 每个 token 都拥有绝对字符区间
- 所有 boundary match 和 dedup 都基于绝对字符区间做

这也是阶段一采纳 `ca-偏移修复` 的根本原因。

### 2.3 失败诊断信息不足

`v1.0` 在 overlap 不足时会报错或 fallback，但外部看到的仍然主要是：

- 是否 exact match
- 是否 fallback
- 最终耗时

阶段一希望进一步知道：

- 哪一对 chunk 边界失败
- overlap window 是多少
- 左右 overlap token 数有多少
- matched pairs 数量是多少
- longest run 有多长

这样后续才能有针对性地做 `stage2` 的策略优化。

---

## 3. 阶段一实现边界

### 3.1 阶段一会实现什么

1. 一个模块化的 `LoPT v2` 内核
2. 显式的 absolute char offset 流程
3. 更强的 boundary diagnostics
4. 基于 `v1 replay best config` 的固定参数对比脚本
5. 基础 smoke 验证方案

### 3.2 阶段一不会实现什么

1. 不会直接修改 `/home/light/vllm` 上游源码
2. 不会启动 `vllm serve`
3. 不会修改远端依赖和包
4. 不会在阶段一做完整重新 search
5. 不会在阶段一实现 C++/Rust merge 内核
6. 不会在阶段一实现动态 chunk doubling retry

---

## 4. 代码结构设计

阶段一新增一个独立包：

```text
benchmarks/lopt_v2/
  __init__.py
  config.py
  types.py
  planner.py
  worker.py
  offsets.py
  matcher.py
  merger.py
  validator.py
  fallback.py
  metrics.py
  engine.py

benchmarks/lopt_tokenizer_v2.py
benchmarks/v2_stage1_compare.py
```

### 4.1 各模块职责

#### `config.py`

定义 `LoPTV2Config`：

- `tokenizer_path`
- `processes`
- `overlap_chars`
- `min_match_tokens`
- `initial_chunk_chars`
- `max_retry_rounds`
- `record_boundary_diagnostics`

#### `types.py`

定义阶段一公共数据结构：

- `ChunkTask`
- `AbsoluteTokenSpan`
- `ChunkResult`
- `BoundaryDiagnostics`
- `BoundaryMatch`
- `LoPTV2Result`

这里最关键的是 `AbsoluteTokenSpan`，它把 token 的局部 offset 提升为：

```text
(local_start, local_end) + chunk.start
  =>
(abs_start, abs_end)
```

#### `planner.py`

负责：

- 根据 `chunk_count` 或 `chunk_chars` 计算 chunk 大小
- 生成带 overlap 的 `ChunkTask`

#### `worker.py`

负责：

- 初始化多进程 tokenizer worker
- 子进程编码
- 返回 `input_ids + offset_mapping`

它仍然沿用阶段一的保守策略：

- `ProcessPoolExecutor`
- `spawn`
- 每个 worker 持有独立 tokenizer 实例

#### `offsets.py`

负责把 `ChunkResult` 中的局部 offsets 标准化为绝对偏移列表：

```text
ChunkResult.offsets
  ->
ChunkResult.absolute_spans
```

这一步是阶段一的核心改动之一。

#### `matcher.py`

负责相邻 chunk 的边界匹配：

1. 先计算 overlap window
2. 提取 overlap window 内的 `absolute_spans`
3. 以 `(abs_start, abs_end, token_id)` 为 key 做两边对齐
4. 找最长连续 run
5. 输出 `BoundaryMatch`
6. 同时记录 `BoundaryDiagnostics`

#### `merger.py`

负责：

- 校验所有 boundary match 是否合法
- 根据 match 结果做 token 去冗余合并

#### `validator.py`

负责：

- token hash
- 首个 mismatch 位置说明

#### `engine.py`

定义 `LoPTV2ParallelTokenizer`，是阶段一主入口。

执行链路为：

```text
text
  -> planner.build_chunk_plan
  -> worker.dispatch
  -> offsets.materialize_absolute_offsets
  -> matcher.build_boundary_match
  -> merger.merge_chunk_results
  -> build_inputs_with_special_tokens
  -> LoPTV2Result
```

#### `lopt_tokenizer_v2.py`

对外兼容 façade，便于后续脚本像使用 `v1.0` 一样接入 `v2`。

#### `v2_stage1_compare.py`

阶段一新增对比脚本，核心目的：

- 直接读取 `v1.0 final replay` 的最佳配置
- 固定 `worker_processes / chunk_count / overlap_chars`
- 对比：
  - native vLLM
  - LoPT v1
  - LoPT v2 stage1

---

## 5. 阶段一核心逻辑

### 5.1 `ca-偏移修复` 的采纳方式

阶段一采纳该思路，但不是简单照搬别的实现，而是把它固定成仓库内部的正式规则：

#### 原则

每个 token 的有效表示，不再只是：

```text
token_id + local offset
```

而是：

```text
token_id + absolute char offset
```

#### 原因

chunk overlap 区里，真正需要比较的是：

- 这个 token 是什么
- 它在原始整段文本中的字符区间在哪里

只有两者都一致，才能说两边的 token 是“同一个边界 token”。

#### 阶段一收益

1. 逻辑更显式
2. 调试信息更丰富
3. 后续做动态 chunk 调整时，更容易保正确性

### 5.2 boundary match 规则

阶段一 match 规则如下：

1. 仅比较 overlap window 内的 token
2. 使用 `(abs_start, abs_end, token_id)` 做精确配对
3. 配对成功后，在 token index 序列上找最长连续 run
4. 若最长 run `< min_match_tokens`，判定为边界不足

### 5.3 merge 规则

阶段一仍然保持父进程 deterministic merge：

- 第一块保留到第一处 match 末尾
- 中间块保留上一个 match 末尾到下一个 match 末尾之间的部分
- 最后一块保留上一个 match 末尾后的部分

### 5.4 special tokens 规则

阶段一仍然沿用 `v1.0` 的保守规则：

- chunk 编码时 `add_special_tokens=False`
- 完成整段 raw token merge 后
- 最后统一调用 `build_inputs_with_special_tokens()`

这样可以避免 chunk 内 special token 干扰边界匹配。

---

## 6. 验证方案

阶段一验证分两层：

### 6.1 结构验证

本地先做：

- `py_compile`
- 模块导入检查

目的：

- 避免明显语法错误
- 保证回放脚本、v2 内核能被加载

### 6.2 远端 smoke 验证

由于本地工作区当前没有 `transformers/tokenizers` 依赖，也没有可直接读取的 tokenizer 目录，阶段一实际运行验证放在远端已有环境中做。

远端约束保持不变：

- 不改包
- 不起 vLLM 服务
- 复用已有 tokenizer 目录
- 复用已有真实语料

### 6.3 精度验证口径

阶段一依然采用最严格标准：

1. token 数量一致
2. token IDs 完全逐项一致
3. token hash 一致

只要有一项不满足，就不算通过。

### 6.4 性能验证口径

阶段一不是重新 search，而是固定使用 `v1 replay` 的最佳参数做对比。

参数来源：

```text
results/replay_merged_20260526/final_replay_results.json
```

对每个 `(model, language, length)` case，固定使用其中记录的：

- `worker_processes`
- `chunk_count`
- `overlap_chars`

然后分别跑：

1. native vLLM baseline
2. LoPT v1
3. LoPT v2 stage1

记录指标：

- `native_e2e_time_ms`
- `native_tokenizer_time_ms`
- `v1_e2e_time_ms`
- `v1_mp_time_ms`
- `v1_dedup_time_ms`
- `v2_e2e_time_ms`
- `v2_mp_time_ms`
- `v2_dedup_time_ms`
- `v2_vs_v1_e2e_speedup_x`
- `v2_vs_v1_e2e_delta_ms`
- `v2_vs_v1_dedup_delta_ms`
- `v1_exact_match`
- `v2_exact_match`
- `v2_status`
- `v2_first_boundary_issue`

---

## 7. 阶段一 smoke 建议执行方式

建议先跑代表性子集，而不是一上来全量 48 case：

```text
模型: DeepSeek-V4-Pro / Qwen3.5
语言: en / zh
长度: 1k / 16k / 128k / 512k / 1024k
```

如果 smoke 全通过，再做完整 best-config 回放。

这样可以更快回答三个问题：

1. `v2` 是否稳定 exact match
2. `v2` 是否比 `v1` 更容易解释边界行为
3. `v2` 性能是否没有系统性退化

---

## 8. 阶段一验收标准

阶段一完成的标准定义如下：

1. `benchmarks/lopt_v2/` 模块化内核可正常导入
2. `LoPTV2ParallelTokenizer` 可按 `v1` 风格被脚本调用
3. `v2_stage1_compare.py` 可直接复用 `v1 best config`
4. smoke case 中 `v2` 对 native 保持 exact match
5. 若失败，能通过 `boundary diagnostics` 定位到具体 chunk 边界
6. 不修改远端依赖，不启动远端 vLLM 服务

---

## 9. 阶段一产物

当前阶段一新增产物应包括：

- `benchmarks/lopt_v2/*`
- `benchmarks/lopt_tokenizer_v2.py`
- `benchmarks/v2_stage1_compare.py`
- `docs/v2_stage1_design.md`

后续如果 smoke 和固定参数回放结果落盘，还会新增：

- `results/v2_stage1_compare_*/`

---

## 10. 后续阶段衔接

阶段一完成后，阶段二再做：

1. 基于 `v2` 重新完整 search
2. 评估 `v2` 最优参数是否已偏离 `v1`
3. 决定是否实现动态 chunk retry
4. 决定 dedup 是否需要更低层内核化

也就是说，阶段一的目标不是“直接毕业”，而是把后续所有优化工作的地基打稳。
