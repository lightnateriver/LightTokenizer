# LightTokenizer — LoPT: Lossless Parallel Tokenization

> **论文**: LoPT — Lossless Parallel Tokenization for Long Context Inference  
> (Huawei, arXiv:2511.04952, 2025)  
> **基于 Qwen3.5-35B-A3B (248K vocab) 在 ARM 单核服务器验证**

LoPT 将长文本拆分为多个 chunk 并行 tokenize，再将结果无损拼接。
中文 1024K tokens 达 **1.35× 加速**，英文预期 **2-2.5×**。

---

## 快速开始

```bash
pip install -r requirements.txt
python src/lopt_sweep.py --lang zh --md results/my_results.md
```

## 目录结构

```
LightTokenizer/
├── src/               # 核心实现 + 搜索脚本
│   ├── lopt_core.py       # 算法核心：_find_anchor, merge_chunks, run_lopt
│   ├── lopt_sweep.py      # 穷举参数搜索脚本（8 size × 8 Lc × 5 LO × 6 thread）
│   ├── corpora.py         # 中英文测试语料
│   └── tokenizer_utils.py # tokenizer 加载工具
├── results/           # 已有搜索结果
│   └── lopt_zh_results.md # 中文全量穷搜数据（1680 组测试）
├── docs/              # 文档
│   ├── html/              # HTML 报告 → 用于 GitHub Pages
│   │   └── index.html
│   ├── IMPLEMENTATION.md  # 算法实现细节 + anchor 修复记录
│   ├── SWEEP_GUIDE.md     # 换机重搜指南
│   └── PITFALLS.md        # 已知陷阱
├── scripts/           # 快捷脚本
│   ├── run_quick_test.sh  # 快速验证
│   └── run_sweep.sh       # 一键搜索
├── tests/             # 测试
│   ├── test_anchor.py     # anchor 查找正确性
│   └── test_merge.py      # 合并精度
├── requirements.txt
└── README.md
```

## 已验证结果

| 规模 | Token | 串行(ms) | LoPT(ms) | 加速比 |
|:--|:--|:--|:--|:--|
| 128K | 127,505 | 214.1 | 193.5 | 1.11× |
| 256K | 255,593 | 450.9 | 386.3 | 1.17× |
| 512K | 511,769 | 993.1 | 778.0 | 1.28× |
| 1024K | 1,023,528 | 2101.4 | 1560.1 | 1.35× |

## 多核扩展路线图

| 方案 | 单核 | 4核 | 16核 | 状态 |
|:--|:--|:--|:--|:--|
| ThreadPool | 1.35× | ~2.0× | ~2.3× | ✅ 已实现 |
| ProcessPool (mmap) | — | ~2.0× | ~4.2× | 🔜 计划中 |
| Rust 原生 (rayon) | — | ~7.0× | ~10×+ | 🔮 远期 |

## 链接

- [HTML 报告](https://lightnateriver.github.io/LightTokenizer/)
- [论文 PDF](https://arxiv.org/abs/2511.04952)
- [vLLM](https://github.com/vllm-project/vllm)
