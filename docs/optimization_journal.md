# Tokenizer Optimization Journal

这份文档用于持续记录 `LightTokenizer` 工程里每一轮优化的目标、实现、测量口径、实验结果、问题分析与下一步动作。

它的定位不是一次性的设计稿，而是一份可以持续续写的“优化演进日志”。

后续每当我们做下面任一类工作时，都应该往本文追加一节：

- 新增打点
- 新增优化策略
- 修改 merge / dedup / boundary match 逻辑
- 调整 search 空间
- 发现新的瓶颈
- 完成一次阶段性复盘

建议把它当作“项目优化主线文档”来维护。

---

## 1. 文档使用原则

### 1.1 每次追加记录都应包含什么

每一轮优化至少记录以下内容：

1. 背景
2. 本轮想解决的问题
3. 代码改动位置
4. 打点或测量口径
5. 实验输入范围
6. 精度结果
7. 性能结果
8. 结论
9. 下一步建议

### 1.2 推荐的追加模板

后续新增记录时，建议直接复用下面模板：

```text
## X. 某次优化标题

### 背景

### 目标

### 代码改动

### 测量口径

### 实验范围

### 精度结果

### 性能结果

### 结论

### 下一步
```

---

## 2. 项目优化主线概览

当前项目的优化主线可以概括为：

```text
先对齐原生链路
  -> 建立 v1 多进程 LoPT 原型
  -> 完整 search / replay / report
  -> 进入 v2 模块化重构
  -> 补充更细粒度打点
  -> 基于瓶颈继续做定向优化
```

更具体一点：

```text
原生 vLLM Tokenizer
  -> 证明 Tokenizer 是长文本 CPU 瓶颈
  -> LoPT v1: 多进程切块并行 + overlap 去冗余
  -> LoPT v2 stage1: 模块化 + 绝对字符偏移 + diagnostics
  -> 细粒度 collect 分解: child compute / result return / dedup
  -> 下一阶段: 降低 collect 尾部阻塞与 merge 开销
```

---

## 3. 阶段 0：原生链路对齐与基线建立

### 3.1 目标

在不启动远端 `vllm serve`、不修改远端依赖的前提下，直接复用本地最新 `vLLM` 源码里的 API Server Tokenizer 路径，建立可信基线。

### 3.2 对齐的原生路径

原生基线路径对齐到：

```text
OpenAIServingTokenization.create_tokenize
  -> OpenAIServingRender.preprocess_completion
  -> HfRenderer.render_cmpl_async
  -> BaseRenderer._tokenize_prompt_async
  -> AsyncMicrobatchTokenizer.encode
```

核心关注点是：

- `HfRenderer._tokenize_prompt_async`
- `AsyncMicrobatchTokenizer.encode`

### 3.3 为什么这一步重要

如果基线不是沿着真实 API Server Tokenizer 链路测出来的，而只是“手写一个 HF tokenizer benchmark”，那么后续 LoPT 优化即使看起来有收益，也很难说明对 vLLM 真正的服务路径有价值。

### 3.4 当前基线指标

当前基线统计：

- `native_e2e_time_ms`
- `native_tokenizer_time_ms`

---

## 4. 阶段 1：LoPT v1 多进程原型

### 4.1 目标

证明在真实中文/英文网页长文本上，采用 LoPT 风格的多进程并行方案，可以在保持 Token IDs 完全一致的前提下显著降低耗时。

### 4.2 方案摘要

`LoPT v1` 的基本链路：

```text
超长文本
  -> 按字符切分为带 overlap 的 chunk
  -> 多进程并行 tokenizer encode
  -> 子进程返回 input_ids + offset_mapping
  -> 父进程基于 overlap 去重拼接
  -> 输出最终 token ids
```

### 4.3 v1 的价值

这一步解决的是“方案可跑通、可搜索、可 replay、可报告”的问题。

也就是说，`v1` 的贡献主要是：

- 跑通完整实验链路
- 建立真实语料与多模型对比
- 建立搜索空间
- 建立最终 replay 与 HTML 报告

### 4.4 v1 的已知局限

`v1` 更偏原型，存在几个明显问题：

- 代码集中在单文件里，可维护性一般
- 失败诊断信息不够细
- `collect / merge / dedup` 的子阶段打点不够细
- 很难精确判断瓶颈到底在 encode、回传还是 merge

---

## 5. 阶段 2：LoPT v2 stage1 模块化重构

### 5.1 目标

`v2 stage1` 的目标不是再次 full search，而是把系统从“能跑的原型”推进到“能持续优化的内核”。

### 5.2 核心改动

当前已经完成的核心改动包括：

1. 把 `v1` 拆成模块化 `benchmarks/lopt_v2/`
2. 引入基于绝对字符偏移的边界匹配模型
3. 提供更可诊断的 `BoundaryDiagnostics`
4. 新增 `native / v1 / v2` 的固定参数对比脚本

相关代码入口：

- [benchmarks/lopt_v2/types.py](/home/light/codes/LoPT/benchmarks/lopt_v2/types.py)
- [benchmarks/lopt_v2/worker.py](/home/light/codes/LoPT/benchmarks/lopt_v2/worker.py)
- [benchmarks/lopt_v2/engine.py](/home/light/codes/LoPT/benchmarks/lopt_v2/engine.py)
- [benchmarks/v2_stage1_compare.py](/home/light/codes/LoPT/benchmarks/v2_stage1_compare.py)

### 5.3 当前验证结果

在固定最优参数回放对比中，`v2` 已经验证：

- 精度可与 native 完全对齐
- 相比 `v1` 在多数 case 上更快
- dedup 阶段已有明显下降

---

## 6. 阶段 3：细粒度链路打点

### 6.1 为什么要补这轮打点

此前我们只知道：

```text
LoPT E2E
  = chat template
  + mp dispatch/process/collect
  + chunk dedup
```

这个口径足以说明“LoPT 比原生快不快”，但还不够回答下面几个更关键的问题：

1. `mp` 时间里到底是子进程 tokenizer 计算慢，还是结果回传/父进程 gather 慢？
2. `merge/dedup` 在长文本里已经涨到什么程度？
3. 下一阶段应该优先优化进程收集方式，还是直接去做 native merge / C++ merge？

### 6.2 本轮新增打点字段

`v2` 当前已经新增以下细粒度指标：

- `collect_child_compute_makespan_s`
- `collect_result_return_tail_s`
- `collect_result_receive_lag_max_s`
- `collect_result_receive_lag_avg_s`
- `worker_encode_time_s_sum`
- `worker_encode_time_s_max`
- `worker_materialize_time_s_sum`
- `worker_materialize_time_s_max`

定义位置：

- [benchmarks/lopt_v2/types.py](/home/light/codes/LoPT/benchmarks/lopt_v2/types.py:98)

子进程时间采集位置：

- [benchmarks/lopt_v2/worker.py](/home/light/codes/LoPT/benchmarks/lopt_v2/worker.py:22)

汇总计算位置：

- [benchmarks/lopt_v2/engine.py](/home/light/codes/LoPT/benchmarks/lopt_v2/engine.py:115)

导出到 compare JSON 的位置：

- [benchmarks/v2_stage1_compare.py](/home/light/codes/LoPT/benchmarks/v2_stage1_compare.py:257)

### 6.2.1 对应的字符链路图

这轮打点对应的执行链路可以表示为：

```text
用户文本
  |
  v
chunk plan / 切块
  |
  +--> submit 到 ProcessPoolExecutor
  |      指标: dispatch_submit_time
  |
  +--> 子进程 tokenizer 计算
  |      |
  |      +--> HF tokenizer encode
  |      |      指标:
  |      |      - worker_encode_time_sum
  |      |      - worker_encode_time_max
  |      |
  |      +--> materialize input_ids / offsets
  |             指标:
  |             - worker_materialize_time_sum
  |             - worker_materialize_time_max
  |
  |      子进程整体 wall-clock
  |      指标:
  |      - collect_child_compute_makespan
  |
  +--> 结果回传 + 父进程 gather
  |      指标:
  |      - collect_result_return_tail
  |      - collect_result_receive_lag_max
  |      - collect_result_receive_lag_avg
  |      - dispatch_collect_time
  |
  +--> overlap 去冗余 / merge
         指标:
         - chunk_dedup_time
  |
  v
final token ids

E2E
  = chat_template_time
  + mp_dispatch_process_collect_time
  + chunk_dedup_time
```

### 6.3 当前打点定义

#### `collect_child_compute_makespan`

定义为：

```text
最后一个 worker 完成时间
  - 第一个 worker 开始时间
```

它表示“这一批 chunk 在子进程侧真正跑 tokenizer 计算的总 wall-clock 覆盖时间”。

#### `collect_result_return_tail`

定义为：

```text
最后一个结果被父进程接收的时间
  - 最后一个 worker 完成时间
```

它描述的是“最后一段子进程计算结束后，到父进程真正把最后一个结果拿回来的尾巴时间”。

#### `collect_result_receive_lag_max / avg`

定义为每个 chunk：

```text
父进程收到该结果的时间
  - 该 worker 完成计算的时间
```

这个指标用于观察 IPC 回传和父进程收集延迟。

### 6.4 重要说明

`collect_child_compute_makespan` 与 `dispatch_collect_time` 不是严格可加关系。

原因是：

- `dispatch_collect_time` 是父进程开始 `future.result()` 到全部收集完成的时间窗
- `collect_child_compute_makespan` 是 worker 侧从最早开始到最晚结束的时间窗

两者只是不同视角的时间定义，不能机械相加。

另外，`worker_encode_time_sum` 也不是 wall-clock 耗时，而是所有子进程 encode 时间的求和，因此它更像“总 CPU 工作量”指标；`worker_encode_time_max` 则更接近单个最慢 chunk 的计算上界。

---

## 7. 最近一次 probe 的结果记录

### 7.1 实验范围

为了快速定位瓶颈，本轮只挑了 `16` 个 probe case：

- 模型：
  - `DeepSeek-V4-Pro`
  - `Qwen3.5`
- 语言：
  - `en`
  - `zh`
- 输入长度：
  - `1k`
  - `32k`
  - `256k`
  - `1024k`

实验约束：

- 远端 CPU 机器执行
- 不启动 `vllm serve`
- CPU 绑定：
  - `29-31,40-79,155-159`

结果文件：

- `/root/LoPT/results/v2_mp_breakdown_20260528_b/breakdown_probe_b.json`

### 7.2 精度结果

本轮结果：

- `v1 exact match = 16 / 16`
- `v2 exact match = 16 / 16`

说明当前 `v2` 打点没有破坏精度。

### 7.3 v1 / v2 对比结果

本轮结果：

- `v2 better cases = 14 / 16`
- `v2 average speedup vs v1 = 1.306x`

### 7.4 按长度聚合后的关键结果

单位：`ms`

| length | submit | collect | child compute | return tail | dedup | e2e |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1k | 0.029 | 1.411 | 1.111 | 0.185 | 0.040 | 1.528 |
| 32k | 0.133 | 11.941 | 12.118 | 0.934 | 2.513 | 16.337 |
| 256k | 0.141 | 70.191 | 59.326 | 11.107 | 12.659 | 81.984 |
| 1024k | 7.383 | 244.142 | 262.014 | 1.995 | 66.267 | 336.180 |

### 7.5 占比观察

按 `E2E` 看：

- `child compute` 大致占 `72% ~ 78%`
- `dedup` 已经占到 `2.6% ~ 19.7%`
- `return tail` 在 `256k` 达到 `13.5% E2E`

按 `collect` 看：

- `1k`: `return tail / collect = 13.1%`
- `32k`: `return tail / collect = 7.8%`
- `256k`: `return tail / collect = 15.8%`
- `1024k`: `return tail / collect = 0.8%`

### 7.5.1 这轮 probe 的细粒度理解

可以把这轮结果概括成：

```text
1k / 32k:
  collect 主要还是子进程计算

256k:
  子进程计算仍然最重
  但结果回传 / gather 已经明显抬头

1024k:
  collect 仍由子进程计算主导
  return tail 很小
  但 dedup/merge 已经成为新的大块开销
```

这也是为什么下一阶段应该优先尝试：

1. 优化 `collect` 收集方式
2. 然后继续压 `dedup/merge`

### 7.6 当前结论

这一轮打点得到的主要结论是：

1. 总体主瓶颈仍然是子进程里的 tokenizer encode 计算
2. `worker_materialize` 很小，不是当前重点
3. `256k` 档位里，结果回传 / 父进程 gather 已经是可见的次级瓶颈
4. `1024k` 档位里，`dedup/merge` 已经很重，应该进入下一阶段重点优化范围

---

## 8. 当前推荐的优化优先级

基于现有数据，当前建议的优化顺序如下。

### 8.1 优先级一：优化 collect 收集方式

当前父进程按提交顺序调用：

```text
for future in futures:
    result = future.result()
```

这容易产生 head-of-line blocking。

建议下一步优先尝试：

- 改成 `as_completed()` 收集
- 收集后按 `chunk.index` 恢复顺序

预期收益：

- 降低 `receive_lag`
- 降低 `return_tail`
- 对 `256k` 档位最可能有直接收益

### 8.2 优先级二：优化结果回传载荷

当前子进程返回：

- `list[int]`
- `list[tuple[int, int]]`

长文本下回传对象比较重。

可尝试方向：

- 更紧凑的连续 buffer 表示
- 共享内存 / mmap
- 只回传 merge 必需字段

### 8.3 优先级三：优化 dedup / merge

`1024k` 时 `dedup` 已接近 `20% E2E`。

这意味着即使继续压缩 `collect`，长文本尾部也会越来越被 merge 侧拖住。

值得继续探索：

- 更轻量的数据结构
- 减少 Python 对象数量
- native merge 内核
- C++ / Rust merge path

### 8.4 优先级四：继续追 encode 内核

从本轮数据看：

- `materialize` 很小
- 真正重的是 tokenizer `encode`

因此如果后续要做更大级别优化，最值得投资的方向仍然是：

- tokenizer encode 内核
- offsets 生成路径
- 更贴近底层实现的并行 tokenizer 策略

---

## 9. 后续记录规范

从现在开始，建议每次优化都按下面方式往本文追加：

1. 记录这次改了什么
2. 记录改动对应的代码路径
3. 记录测量口径有没有变化
4. 记录至少一个结果文件路径
5. 记录“是否保持 exact match”
6. 记录本轮新增判断
7. 记录下一步动作

如果某次优化只是一个小 patch，也建议至少写一小节，不要只留在对话里。

---

## 10. 下一条准备追加的记录

下一条建议记录的主题是：

```text
LoPT v2 collect 阶段 as_completed 收集优化
```

重点关注：

- `collect_result_return_tail`
- `collect_result_receive_lag_max`
- `collect_result_receive_lag_avg`
- `256k` 与 `1024k` 的收益差异
- 是否继续影响 dedup 阶段

---

## 11. LoPT v2 collect 阶段 as_completed 收集优化

### 11.1 背景

在上一轮 `16` 个 probe case 的细粒度打点里，我们已经确认：

- 总体主瓶颈仍然是子进程 tokenizer 计算
- `256k` 档位中，结果回传 / 父进程 gather 是明显的次级瓶颈
- 当前父进程按提交顺序 `future.result()`，存在 head-of-line blocking 风险

因此，这一轮的目标不是大改算法，而是非常克制地优化 collect 收集方式。

### 11.2 本轮目标

把父进程的结果收集从：

```text
for future in futures:
    result = future.result()
```

改成：

```text
for future in as_completed(futures):
    result = future.result()
```

然后在父进程按 `chunk.index` 排回原顺序，再进入后续 merge。

这意味着：

- 收集路径更及时
- merge 顺序仍保持 deterministic
- 精度逻辑不依赖“先返回谁就先 merge 谁”

### 11.3 代码改动

本轮只改了收集逻辑，没有改 merge 规则。

核心改动文件：

- [benchmarks/lopt_v2/worker.py](/home/light/codes/LoPT/benchmarks/lopt_v2/worker.py:65)

关键变化：

1. 引入 `as_completed`
2. 按完成顺序先把结果收回父进程
3. 在返回前按 `result.index` 排序，恢复 chunk 原始顺序

### 11.4 为什么这个改法是安全的

本轮优化只影响“父进程何时收到结果”，不改变：

- 子进程 tokenizer encode 行为
- `offset_mapping` 内容
- boundary match 规则
- merge 顺序

因此理论上不应破坏和 native 的精度对齐；真正需要验证的是：

- 是否仍然 exact match
- `return_tail / lag` 是否改善
- 对总 `collect` 和 `E2E` 是否有帮助

### 11.5 实验范围

为了快速验证，本轮没有重跑全部 case，而是选了对 collect 更敏感的 `8` 个长文本 case：

- 模型：
  - `DeepSeek-V4-Pro`
  - `Qwen3.5`
- 语言：
  - `en`
  - `zh`
- 长度：
  - `256k`
  - `1024k`

约束保持不变：

- 远端 CPU 机器
- 不启动 `vllm serve`
- CPU 绑定：
  - `29-31,40-79,155-159`

结果文件：

- `/root/LoPT/results/v2_as_completed_probe_20260528/as_completed_probe.json`

### 11.6 精度结果

本轮结果：

- `case_count = 8`
- `v1 exact match = 8 / 8`
- `v2 exact match = 8 / 8`

结论：

`as_completed` 收集版在这 `8` 个 case 上仍然与 native 原始链路完全对齐，没有精度损失。

### 11.7 相对 v1 的结果

本轮结果：

- `v2 better cases = 8 / 8`
- `average v2 vs v1 speedup = 1.359x`

按长度聚合后的中位数结果如下，单位 `ms`：

| length | v1 e2e | v2 e2e | v2 collect | v2 child | v2 tail | v2 lag avg | v2 dedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256k | 108.397 | 78.956 | 66.547 | 54.624 | 13.231 | 10.908 | 12.197 |
| 1024k | 415.382 | 300.555 | 227.392 | 230.056 | 1.891 | 2.192 | 64.111 |

### 11.8 相对上一轮旧版 v2 collect 的净变化

为了看清 `as_completed` 的净效果，我们又把它和上一轮旧版 `v2` probe 做了同 case 对比。

按长度聚合后的平均变化，单位 `ms`：

| length | e2e delta | collect delta | child delta | tail delta | lag avg delta | dedup delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256k | -7.042 | -6.474 | -8.492 | 2.570 | -1.321 | -0.569 |
| 1024k | -32.404 | -13.851 | -29.224 | -0.148 | -1.140 | -1.975 |

解释方式：

- 负值表示新版本更快
- 正值表示新版本更慢

### 11.9 如何理解这轮结果

#### 256k

`256k` 上有几个值得注意的现象：

1. `E2E` 和 `collect` 都变快了
2. `lag avg` 下降了，说明结果接收平均延迟有所改善
3. 但 `tail delta` 没有统一下降，部分 case 甚至略有上升

这说明：

- `as_completed` 确实改善了整体收集平滑度
- 但 `256k` 的次级瓶颈不只是“等待最慢 future”
- 结果回传对象本身的成本仍然存在

#### 1024k

`1024k` 上结果更干净：

1. `collect` 有下降
2. `child compute` 有明显下降
3. `tail` 基本持平但更低
4. `lag avg` 略有下降
5. `E2E` 有较明显改善

不过这里也要注意：

`1024k` 的收益并不能全部归因于 `as_completed`，因为大文本多次回放本身存在一定运行波动，所以这轮更适合解释为：

```text
as_completed 没有带来负收益
并且在长文本下总体趋势是正向的
```

### 11.10 当前结论

这轮优化的结论可以概括成：

1. `as_completed` 收集方式是安全的，精度与 native 完全对齐
2. 它对长文本 case 是正收益，本轮 `8 / 8` 都优于 `v1`
3. 相对旧版 `v2`，它能进一步降低一部分 `collect` 与 `E2E`
4. 但它不是解决 `256k` 回传成本的最终答案
5. `1024k` 的长期主线仍然是继续压 `dedup/merge`

### 11.11 下一步建议

基于这轮结果，建议下一步按这个顺序继续：

1. 保留 `as_completed` 作为默认 collect 实现
2. 继续优化结果回传载荷
3. 把 `dedup/merge` 作为长文本重点优化项
4. 后续再决定是否要引入共享内存或 native merge 内核

---

## 12. 主线切回 as_completed，准备优化 dedup / merge

### 12.1 当前主线说明

从这一轮开始，主线统一以 `as_completed` 版本为准。

原因很直接：

- 它已经证明精度稳定
- 它对 `1k / 32k / 256k / 1024k` 的整体收益是正向的
- 它比原始 `v2` 更接近当前最优可用版本

因此后续所有新的性能优化，都以 `as_completed` 为对照基线。

### 12.2 为什么现在转向 dedup / merge

当前打点已经给出比较明确的信号：

- `collect` 已经被压了一轮
- `1024k` 上 `chunk_dedup_time` 已经非常可观
- `compact boundary` 路线虽然精度成立，但当前实现还没有把性能打正

所以更稳妥的主线是：

```text
as_completed
  -> 优化 dedup / merge
  -> 再看长文本收益
```

### 12.3 下一轮优化目标

本轮准备重点优化：

- `merge_chunk_results`
- `validate_matches`
- 边界匹配到 merge 的数据流

优先思路不是一上来就改算法，而是先压 Python 对象和重复切片：

1. 尽量少建中间 list
2. 尽量少做重复边界检查
3. 尽量让 merge 路径更扁平
4. 保持 exact match 不变

### 12.4 这一轮的验收标准

新的 dedup / merge 优化只有在以下条件都满足时才算通过：

- 与 `as_completed` 基线相比有可测收益
- 性能浮动用 `%` 表示
- `1k / 32k / 256k / 1024k` 都要测
- 与 native 原始链路 exact match 一致

---

## 13. LoPT v2 dedup / merge 优化（merge_opt）

### 13.1 背景

在 `as_completed` 成为稳定主线之后，新的瓶颈判断已经比较清楚：

- `collect` 的 head-of-line blocking 已经被压下去
- 长文本下 `chunk_dedup_time` 仍然非常显眼
- 继续从 `collect` 榨收益的边际开始变小

因此，这一轮把主攻方向切到 `dedup / merge`，目标是让长文本路径继续降耗，同时不破坏和 native 的 token 级精度对齐。

### 13.2 本轮优化结果

结果文件：

- `/root/LoPT/results/v2_merge_opt_20260528/merge_opt_summary.json`
- `/root/LoPT/results/v2_merge_opt_20260528/merge_opt.json`

本轮统计：

- `case_count = 16`
- `v1 exact match = 16 / 16`
- `v2 exact match = 16 / 16`
- `v2 better cases = 14 / 16`
- `average v2 vs v1 speedup = 1.442x`

这里的 `v1` 仍然是原始 native 链路，说明本轮优化没有引入精度损失。

### 13.3 相对 as_completed 基线的净收益

本轮我们又把 `merge_opt` 和 `as_completed` 基线做了同 case 对比。下面的 `%` 都表示：

```text
(merge_opt - as_completed) / as_completed
```

整体平均值如下，单位 `ms`：

| 指标 | as_completed | merge_opt | 变化 |
| --- | ---: | ---: | ---: |
| E2E | 112.206 | 79.627 | -29.04% |
| dispatch_collect | 85.084 | 75.730 | -10.99% |
| child_compute_makespan | 87.025 | 75.048 | -13.76% |
| result_return_tail | 3.328 | 1.327 | -60.12% |
| result_receive_lag_avg | 4.215 | 1.349 | -67.99% |
| chunk_dedup | 20.135 | 1.804 | -91.04% |
| dispatch_submit | 2.392 | 1.917 | -19.85% |
| mp_time | 92.071 | 77.823 | -15.47% |

### 13.4 按长度聚合后的变化

单位：`ms`，括号内为相对 `as_completed` 的变化百分比。

| length | E2E | collect | child | return tail | lag avg | dedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1k | 1.619 -> 1.532 (-5.31%) | 1.549 -> 1.485 (-4.12%) | 1.191 -> 1.044 (-12.36%) | 0.196 -> 0.229 (+16.96%) | 0.198 -> 0.232 (+17.28%) | 0.043 -> 0.024 (-43.93%) |
| 32k | 15.305 -> 13.176 (-13.91%) | 11.161 -> 12.700 (+13.79%) | 11.430 -> 12.129 (+6.12%) | 0.872 -> 0.337 (-61.28%) | 1.664 -> 0.403 (-75.80%) | 2.314 -> 0.295 (-87.25%) |
| 256k | 88.104 -> 63.019 (-28.47%) | 75.579 -> 62.052 (-17.90%) | 64.538 -> 57.980 (-10.16%) | 10.182 -> 3.725 (-63.41%) | 11.646 -> 3.551 (-69.51%) | 12.382 -> 0.873 (-92.95%) |
| 1024k | 343.796 -> 240.779 (-29.96%) | 252.047 -> 226.683 (-10.06%) | 270.940 -> 229.037 (-15.47%) | 2.061 -> 1.016 (-50.69%) | 3.351 -> 1.210 (-63.88%) | 65.801 -> 6.023 (-90.85%) |

### 13.5 这轮结果怎么理解

这轮的信号非常明确：

1. `dedup` 被压得最狠，说明 merge fast path 的方向是对的
2. `return tail` 和 `lag avg` 也明显下降，说明父进程收集后的尾部代价被削掉了
3. `E2E` 在 `256k / 1024k` 上都有稳定收益，已经开始真正影响长文本主链路
4. 小文本 `1k` 和部分 `32k` case 仍然有波动，说明这类 case 的收益更容易被调度噪声掩盖

需要特别说明的是：

- `32k` 上 `dispatch_collect` 与 `child_compute_makespan` 有少量回升，不应简单理解为 merge 优化变慢
- 这更像是运行波动和不同长度下调度行为带来的差异
- 真正稳定的改进点还是 `chunk_dedup_time` 和 `return_tail`

### 13.6 当前结论

这一轮可以得出一个比较稳的判断：

- `merge_opt` 没有破坏精度，`16 / 16` 与 native 完全对齐
- 相比 `as_completed`，整体 E2E 继续下降，平均降幅约 `29%`
- 长文本收益最稳定，尤其是 `256k` 和 `1024k`
- 下一步如果继续优化，最值得继续压的是 `dedup / merge` 的 Python 对象开销和边界处理开销

### 13.7 后续建议

建议后续主线继续保持：

```text
as_completed baseline
  -> dedup / merge fast path
  -> 更轻的数据结构
  -> 进一步减少 Python 切片与中间对象
```

下一轮如果要继续追，优先看两件事：

1. `merge_chunk_results` 是否还能再少建中间 list
2. `validate_matches` / boundary 诊断路径是否能进一步瘦身

---

## 14. 第一优先级 IPC 瘦身实验（未超过 v2.1）

### 14.1 背景

在 `v2.1` 主线已经稳定以后，我们按优先级一尝试了“worker 回传数据瘦身”：

- `input_ids` 改成 packed `array('I')`
- `left_boundary_entries / right_boundary_entries` 改成 packed 形式
- matcher 直接读取 packed boundary 数据

目标是继续压 `IPC / Python 对象展开` 的成本。

### 14.2 实验结果

本轮使用 `v2.1` 的 16 case 作为 reference 重放，结论如下：

- `case_count = 16`
- `v2_exact_match_cases = 16`
- `v2_matches_reference_hash_cases = 16`
- `v2_better_than_reference_cases = 3`
- `average_v2_vs_reference_speedup_x = 0.919`

也就是说：

- 精度是对齐的
- 但平均性能没有超过 `v2.1`

### 14.3 当前判断

这次改动更像是一次“方向正确但收益不足”的实验。

可能的原因包括：

1. packed 表示减少了部分对象开销，但引入了额外 unpack / 重组成本
2. 对长文本来说，真正重的仍是 tokenizer encode 和 merge 主路径
3. 短文本里 packed 结构的收益不足以抵消额外处理成本

### 14.4 结论

本轮 IPC 瘦身**不替换主线**，保留为实验分支记录。

当前主线仍然是：

```text
v2.1
  -> 继续优化 dedup / merge 或更高层次的回传结构
```

下一步如果还要继续压 IPC，建议优先考虑：

1. 更端到端的紧凑 buffer 协议
2. 减少 boundary 采样路径里的重组
3. 再评估是否值得直接下沉到更低层实现

---

## 15. v2.1 + shm input_ids 实验分支

### 15.1 目标

在保持 `v2.1` 主线不变的前提下，引入一个**默认关闭**的 shm 加速分支，用来验证：

- 是否能把 `input_ids` 的大对象回传从普通 IPC 挪到共享内存
- 是否能在长文本上压低 `collect / return tail / lag`
- 是否能在出现 shm 异常时，整请求回退到 `v2.1 inline` 路径

### 15.2 当前实现方式

当前这版 shm 不是主链路，而是一个 opt-in 实验路径：

- 默认 `enable_shm_input_ids = False`
- 开启后，仅 `input_ids` 尝试走 shm
- `boundary_entries / meta / timing` 仍走普通 IPC
- shm 路径异常时，整请求会重跑一次 `disable_shm=True` 的 `v2.1` 路径

这意味着：

```text
正确性主路径 = v2.1
shm = 机会型加速分支
```

### 15.3 代码改动

本轮主要改动：

- [benchmarks/lopt_v2/config.py](/home/light/codes/LoPT/benchmarks/lopt_v2/config.py)
  - 新增 shm 开关与预算参数
- [benchmarks/lopt_v2/worker.py](/home/light/codes/LoPT/benchmarks/lopt_v2/worker.py)
  - worker 写 shm / attach / cleanup / permit budget / fallback
- [benchmarks/lopt_v2/engine.py](/home/light/codes/LoPT/benchmarks/lopt_v2/engine.py)
  - 新增 shm 异常后的整请求 inline fallback
- [benchmarks/compare_v2_against_reference.py](/home/light/codes/LoPT/benchmarks/compare_v2_against_reference.py)
  - 新增 shm 实验参数与 reference compare
- [benchmarks/v2_stage1_compare.py](/home/light/codes/LoPT/benchmarks/v2_stage1_compare.py)
  - 新增 shm 参数透传

### 15.4 安全策略

这一版实现刻意保守，核心策略有三条：

1. **默认关闭**
   - 不影响现有 `v2.1` 主线
2. **预算受限**
   - 用 permit budget 控制 shm inflight 体量
3. **失败回退**
   - shm transport 失败后，整请求回退为 inline `v2.1`

### 15.5 smoke 实验范围

为了先验证链路和正确性，本轮只跑了 `8` 个 reference compare case：

- 模型：
  - `DeepSeek-V4-Pro`
  - `Qwen3.5`
- 语言：
  - `en`
  - `zh`
- 长度：
  - `1k`
  - `256k`

结果目录：

- `/root/LoPT/results/v2_shm_smoke_20260528/v2_shm_smoke.json`

### 15.6 精度结果

本轮结果：

- `case_count = 8`
- `v2_exact_match_cases = 8 / 8`
- `v2_matches_reference_hash_cases = 8 / 8`

说明：

当前 shm 分支在 smoke 范围内**没有破坏精度**，并且回退策略没有让输出偏离 `v2.1 / native` reference。

### 15.7 性能结果

本轮结果：

- `v2_better_than_reference_cases = 2 / 8`
- `average_v2_vs_reference_speedup_x = 0.749`

也就是：

- 平均没有超过 `v2.1`
- 整体趋势明显偏慢

按长度聚合后的平均结果：

| length | v2.1 E2E | shm 版 E2E | 变化 |
| --- | ---: | ---: | ---: |
| 1k | 1.532 ms | 1.546 ms | +0.86% |
| 256k | 63.019 ms | 147.075 ms | +133.38% |

### 15.8 当前判断

这一版 shm 分支在正确性上成立，但在当前实现下，性能表现不理想，尤其是 `256k` 档位明显变慢。

这说明：

1. 简单的“每个 chunk 独立 SharedMemory + attach/cleanup”开销很重
2. 当前实现还没有把 shm 的理论优势兑现出来
3. 共享内存 transport 本身，已经成为比原始 IPC 更贵的新负担

### 15.9 当前结论

本轮 shm 分支的结论很明确：

- **保留实现**
  - 因为它已经具备正确性与回退机制
- **不切主线**
  - 当前主线仍然是 `v2.1`
- **后续若继续做**
  - 需要重点优化 shm 生命周期和 attach/cleanup 成本

### 15.10 下一步建议

如果继续沿 shm 方向走，建议按这个顺序收敛：

1. 先补全 `worker_shm_used_chunk_count / failure_reason / shm write time` 的观测
2. 判断当前慢在哪里，是 `create/unlink`，还是 `attach/view`
3. 若要继续优化，再考虑：
   - pooled shm slots
   - 更轻的 descriptor
   - 更激进的长度门控

在没有新的正收益证据前，`v2.1` 继续作为默认主线。

---

## 16. shm 分支归档与主线确认

### 16.1 当前状态

在完成 shm smoke 验证后，当前代码已经回滚到**纯 `v2.1` 主线**：

- `shm input_ids` 相关代码已从可运行主链路移除
- 配置项、worker 附加逻辑、fallback compare 分支都已清理
- 保留了这次 shm 实验的完整日志，供后续复盘展示

### 16.2 归档结论

这次 shm 试验的结论保持不变：

- 正确性上可以对齐 native / v2.1
- 性能上当前实现没有形成正收益
- 因此不进入主线，不影响后续全量参数搜索

### 16.3 后续主线

当前主线仍然是：

```text
v2.1
  -> full search
  -> 保留完整搜索结构与结果目录
  -> 后续展示时直接沿用搜索输出
```

### 16.4 回归 smoke

为了确认代码已经真正回到纯 `v2.1`，在远端机器上重新执行了一轮 fixed-config smoke：

- 模型：`DeepSeek-V4-Pro` / `Qwen3.5`
- 语言：`en` / `zh`
- 长度：`1k / 32k / 256k / 1024k`
- 绑定 CPU：`29-31,40-79,155-159`
- 输出目录：`/root/LoPT/results/v2_1_revert_smoke_20260528/`

结果：

- `case_count = 16`
- `v1_exact_match_cases = 16`
- `v2_exact_match_cases = 16`
- `v2_better_cases = 14`
- `average_v2_vs_v1_speedup_x = 2.238`

这说明当前代码：

- 精度已恢复到与 native 精确对齐
- 性能表现与既有 `v2.1` 主线一致
- 可以继续做纯 `v2.1` 的 full search

### 16.5 full search 启动

为了保持既有结果结构，full search 继续沿用“分 family 搜索，再 merge”的目录布局：

- `/root/LoPT/results/search_v2_1_deepseek_20260528/`
- `/root/LoPT/results/search_v2_1_qwen_20260528/`
- `/root/LoPT/results/search_merged_v2_1_20260528/`

搜索参数保持全量：

- `worker_values = 1,2,4,8,16,32,64`
- `chunk_multipliers = 0.5,1,2,4`
- `overlap_values = 32,64,128,256,512,1024,2048,4096,8192`
- `languages = en, zh`
- `lengths = 1k,4k,8k,16k,32k,64k,128k,256k,512k,720k,880k,1024k`
- `min_match_tokens = 2`
- `repeats = 1`
- `lopt_version = v2`

当前状态：

- `DeepSeek-V4-Pro` full search 已启动
- `Qwen3.5` full search 等待前者完成后继续

### 16.6 full search 完成情况

本轮 pure `v2.1` full search 已完成，结果目录如下：

- 远端：
  - `/root/LoPT/results/search_v2_1_deepseek_20260528/`
  - `/root/LoPT/results/search_v2_1_qwen_20260528/`
  - `/root/LoPT/results/search_merged_v2_1_20260528/`
- 本地：
  - `results/search_v2_1_deepseek_20260528/`
  - `results/search_v2_1_qwen_20260528/`
  - `results/search_merged_v2_1_20260528/`

merge 结果：

- `detail_rows = 7263`
- `valid_rows = 7263`
- `worker_best_count = 210`
- `best_count = 48`
- `case_failure_count = 0`

这说明本轮完整搜索里：

- 所有保留下来的候选都保持精度正确
- `48` 个模型/语言/长度 case 都找到了 best config
- 没有出现需要 fallback 才能成立的失败 case

### 16.7 队列器小坑

为了让 `DeepSeek -> Qwen -> merge` 自动串起来，曾经尝试过一个简单的 bash 等待器。

问题在于：

- 等待条件里用了 `pgrep -af "benchmarks.search_lopt_configs.*search_v2_1_deepseek_20260528"`
- 这个字符串会匹配到等待器**自己的命令行**
- 结果就是等待器会“自匹配自等待”，一直空转

最终处理：

- 直接停掉这个等待器
- 在 `DeepSeek` 完成后，手动启动 `Qwen` full search
- `Qwen` 完成后再执行 merge

结论：

- 这个问题不影响最终搜索结果
- 但如果之后还要做后台串行编排，等待条件必须避开自匹配
