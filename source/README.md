# 研究源码阅读副本

`scripts/` 包含截至 2026 年 11 月 1 日的 169 个研究脚本，`upstream/CAT_MT_phase7_v0.7/` 包含 9 个历史依赖脚本。

最新入口见 [三组规则判别实验](../docs/LATEST_EXPERIMENT.md)。从低阶状态预测三组响应，再用通道阻断与只保留干预检验输出传递。早期中文注释和必要的技术说明已保留。

本目录相当于原研究项目根目录。完整实验的数据、模型、冻结文件和大量运行结果未包含，请先读 [复现范围](../docs/REPRODUCTION.md)。完整源码索引见 [SCRIPT_INDEX.md](../docs/SCRIPT_INDEX.md)。

发布文件由根目录 `MANIFEST.json` 校验，源码来源由 `PROVENANCE.json` 记录。原实验的冻结档案独立保存。
