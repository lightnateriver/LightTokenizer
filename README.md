# LightTokenizer — LoPT: Lossless Parallel Tokenization

> **论文**: LoPT — Lossless Parallel Tokenization for Long Context Inference  
> (Huawei, arXiv:2511.04952, 2025)  
> **验证环境**: 华为云 ModelArts ARM 单核服务器  
> **Tokenizers**: Qwen3.5-35B-A3B (248K vocab) + DeepSeek-V4-Pro (128K vocab)

LoPT 将长文本拆分为多个 chunk 并行 tokenize，再将结果无损拼接。
ProcessPool(spawn) 模式在 ARM 单核上达到 **6-8× 加速**。

---

## 关键结果

### ProcessPool(spawn, w=16) — 1024K 对比

| Tokenizer | 语言 | 1024K 串行 | LoPT(ms) | **加速比** |
|:--|:--|:--:|:--:|:--:|
| Qwen3.5 (248K) | 中文 | 2045ms | 330ms | **6.20×** |
| Qwen3.5 (248K) | 英文 | 3260ms | 396ms | **8.24×** |
| DeepSeek-V4-Pro (128K) | 中文 | 2574ms¹ | — | **~5.66×²** |
| DeepSeek-V4-Pro (128K) | 英文 | 3157ms | 387ms | **8.16×** |

¹ DSV4 Pro 中文 1024K 串行耗时来自 ThreadPool 穷举搜索基准（该数据点的 ProcessPool 未运行）。  
² DSV4 Pro 中文 ProcessPool 实测仅 512K w=16 达 5.66×；1024K 未测试，从趋势估计相近。

所有结果 **100% token 精确匹配**（逐位 == 验证）。

### ThreadPool — 中文穷举搜索

| 规模 | Tokens | 串行(ms) | LoPT(ms) | 加速比 |
|:--|:--:|:--:|:--:|:--:|
| 128K | 127,505 | 214.1 | 193.5 | 1.11× |
| 256K | 255,593 | 450.9 | 386.3 | 1.17× |
| 512K | 511,769 | 993.1 | 778.0 | 1.28× |
| 1024K | 1,023,528 | 2101.4 | 1560.1 | **1.35×** |

### DSV4 Pro 英文 ProcessPool 详细

| 规模 | 串行(ms) | ProcessPool(w=16) | 加速比 |
|:--|:--:|:--:|:--:|
| 128K | 347 | 128 | 2.72× |
| 512K | 1551 | 189 | **8.21×** |
| 1024K | 3157 | 387 | **8.16×** |

---

## 快速开始

```bash
pip install transformers
# 准备 tokenizer 目录（从 HuggingFace 下载或复制已有）

# 快速验证（中文 16K tokens）
python src/lopt_sweep.py --lang zh --sizes 16K,64K --threads 2,4 --md results/quick.md

# ThreadPool vs ProcessPool 对比
python src/lopt_bench_process.py --lang zh --sizes 128K,512K --workers 4,8,16
```

## 目录结构

```
LightTokenizer/
├── src/
│   ├── lopt_core.py            # 核心算法 + ProcessPool 自动选择
│   ├── lopt_sweep.py           # 穷举参数搜索（ThreadPool）
│   ├── lopt_bench_process.py   # ProcessPool vs ThreadPool 对比
│   ├── corpora.py              # 中英文 20 段测试语料
│   └── tokenizer_utils.py      # tokenizer 加载 + Rust 后端
├── results/
│   └── lopt_zh_results.md      # 中文穷举搜索数据（1700 组测试）
├── docs/
│   ├── index.html              # HTML 分析报告
│   ├── IMPLEMENTATION.md       # 算法实现细节 + 实验记录
│   ├── SWEEP_GUIDE.md          # 换机重搜指南
│   └── PITFALLS.md             # 已知陷阱
├── scripts/
│   ├── run_quick_test.sh       # 快速验证
│   └── run_sweep.sh            # 一键搜索
├── tests/
│   ├── test_anchor.py          # anchor 查找正确性
│   └── test_merge.py           # 合并精度
├── requirements.txt
└── README.md
```

## 核心发现

1. **Anchor 偏移修复至关重要**: ca-offset 双重循环（`pa[pv+k]==ca[qv+k]` 而非 `==ca[k]`）
   将 BPE 边界效应从"永远找不到 anchor"变为"稳定 2×+"。
2. **ProcessPool 远优于 ThreadPool**: 单核上 ProcessPool(spawn) 达 6-8×，ThreadPool 上限 ~1.35×。
3. **词汇表大小影响极小**: 128K vs 248K vocab 加速比基本一致。
4. **Lc_opt = chars/128**: 通用规则，适用于所有 tokenizer 和文本规模。
5. **中文 LO=16 最优**, 英文 LO=32-64。

## 多核扩展路线图

| 方案 | 单核 | 4核 | 16核 | 状态 |
|:--|:--:|:--:|:--:|:--:|
| ThreadPool | 1.35× | ~2.0× | ~2.3× | ✅ 已验证 |
| ProcessPool(spawn) | **8.2×** | ~12× | ~15×+ | ✅ 已验证（单核） |
| ProcessPool + shm | — | ~14× | ~18×+ | 🔜 计划中 |

## 链接

- [HTML 报告](docs/index.html)
- [论文 PDF](https://arxiv.org/abs/2511.04952)
- [vLLM](https://github.com/vllm-project/vllm)
