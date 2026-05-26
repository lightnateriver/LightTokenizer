# 换机重搜指南

在不同硬件上重新搜索 LoPT 最佳参数。

## 一键搜索

```bash
# 中文穷举（8 规模 × 8 Lc × 5 LO × 6 线程 = ~1700 组，约 1-2 小时）
python src/lopt_sweep.py --lang zh --md results/my_zh_results.md

# 英文贫举
python src/lopt_sweep.py --lang en \
  --sizes 128K,256K,512K,1024K \
  --threads 1,2,4,8,16 \
  --md results/my_en_results.md

# 自定义 tokenizer
python src/lopt_sweep.py --lang zh \
  --tokenizer /path/to/local/tokenizer \
  --md results/custom.md
```

## ProcessPool 对比

穷举搜索只用 ThreadPool。拿到最佳 Lc 后，用以下脚本测 ProcessPool：

```bash
# ThreadPool vs ProcessPool 全量对比
python src/lopt_bench_process.py --lang zh \
  --sizes 128K,512K,1024K \
  --workers 2,4,8,16,32

# 英文 + 自定义 tokenizer
python src/lopt_bench_process.py --lang en \
  --tokenizer /path/to/tokenizer \
  --sizes 512K,1024K --workers 4,8,16,32
```

ProcessPool(spawn) 在 chunk 数 ≥20 时显著优于 ThreadPool。

## 变量 vs 硬件依赖

| 变量 | 依赖 | 换机后需重搜？ |
|:--|:--|:--|
| Lc（chunk 大小） | CPU cache、tokenizer 速度、文本规模 | 建议重搜 |
| LO（重叠） | 语言（chars/token 比） | 按语言固定（zh=16, en=32-64） |
| N_workers | 物理核数 × chunk 数 | 建议重搜 |
| Pool 类型 | chunk 数：≤10 用 ThreadPool，≥20 用 ProcessPool | 自动选择 |
| Tokenizer | 词表大小 | 必须重搜 |

## 远端机器快速复制

如果测试机没有网络或需要特定硬件：

```bash
# 1. 克隆仓库（在远端）
git clone https://github.com/lightnateriver/LightTokenizer.git

# 2. 安装依赖
pip install transformers

# 3. 准备 tokenizer 目录
# 从 HuggingFace 下载 tokenizer 到本地目录
# 或复制已有目录

# 4. 运行搜索
python src/lopt_sweep.py --lang zh \
  --tokenizer /path/to/tokenizer \
  --md results/remote_results.md
```

## 预估时间

| 机器 | 单语言全量搜索 | ProcessPool 对比 |
|:--|:--|:--|
| ARM 单核 | ~1.5 小时 | ~10 分钟 |
| x86 4 核 | ~20-30 分钟 | ~3 分钟 |
| x86 16 核 | ~5-10 分钟 | ~1 分钟 |
