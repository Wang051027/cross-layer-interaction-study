# 研究源码

`scripts/` 包含截至 2026 年 11 月 1 日的 169 个研究脚本，`upstream/CAT_MT_phase7_v0.7/` 包含 9 个历史依赖脚本。

最新入口见 [三组规则判别实验](../docs/LATEST_EXPERIMENT.md)。从低阶状态预测三组响应，再用通道阻断与只保留干预检验输出传递。

本目录对应原研究项目根目录。[最新实验入口](../reproduce_latest.py) 会将复跑所需的源码、模型、图像和冻结记录放入新的工作目录。更早实验的完整训练和运行档案另存。[复现范围](../docs/REPRODUCTION.md) · [完整源码索引](../docs/SCRIPT_INDEX.md)

发布文件由根目录 `MANIFEST.json` 校验，源码来源由 `PROVENANCE.json` 记录。
