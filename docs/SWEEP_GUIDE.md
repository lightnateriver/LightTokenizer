# 换机重搜指南

不同机器配置下 LoPT 的最佳参数不同。通过 `lopt_sweep.py` 重新搜索。

## 变量 vs 硬件依赖

| 变量 | 依赖 | 换机后需重搜？ |
|:--|:--|:--|
| Lc（chunk 大小） | CPU cache、tokenizer 速度、文本规模 | 建议重搜 |
| LO（重叠） | 语言（chars/token 比） | 按语言固定（zh=16, en=256） |
| N_threads | 物理核数 | 建议重搜 |
| Tokenizer | 词表大小 | 必须重搜 |

## 运行

```bash
# 中文全部搜索（8 种规模，约 1-2 小时）
python src/lopt_sweep.py --lang zh --md results/my_results.md

# 只搜特定规模和线程数
python src/lopt_sweep.py --lang en \
  --sizes 128K,512K,1024K \
  --threads 1,2,4,8,16 \
  --md results/en_results.md

# 自定义 tokenizer
python src/lopt_sweep.py --lang zh \
  --tokenizer /path/to/tokenizer \
  --md results/custom.md
```

## 预估时间

| 机器 | 单语言全量搜索 |
|:--|:--|
| ARM 单核 | ~1.5 小时 |
| x86 4 核 | ~20-30 分钟 |
| x86 16 核 | ~5-10 分钟 |
