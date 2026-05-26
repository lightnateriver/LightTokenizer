# 已知陷阱

## 1. Anchor 查找偏移

**表现：** 所有 anchor 失败 → fallback 串行 → 比串行更慢。

**原因：** 原始 `_find_anchor` 只比较 `pa[pv+k] == ca[k]`，ca 始终从索引 0 开始。
BPE 在 chunk 边界处可能偏移 1-2 个 token，导致序列错位。

**修复：** 增加 ca 偏移双重循环（见 IMPLEMENTATION.md）。

## 2. Overlayfs 白障（容器环境）

**表现：** `OSError: [Errno 2] No such file or directory`，目录可见但不可写。

**原因：** `pip --force-reinstall` 在 overlayfs 上创建 whiteout 条目，
删除目录后无法重新创建。

**解决：**
- 创建 venv：`python3 -m venv /opt/clean-env`
- 或用 `--ignore-installed` 跳过 uninstall

## 3. ThreadPool 在单核上的收益上限

ARM 单核上 t=2 即饱和，t>2 无收益。
多核机器建议搜索 t=2~16。

## 4. vLLM 0.18 + torch 2.10 不兼容

**表现：** EngineCore 初始化报 `FakeTensorMode` AttributeError。

**解决：** 启动加 `--enforce-eager`。
