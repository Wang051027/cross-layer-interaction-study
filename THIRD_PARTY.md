# 数据与历史代码来源

## dSprites

- 原作者：Loic Matthey、Irina Higgins、Demis Hassabis、Alexander Lerchner。
- 官方仓库：https://github.com/google-deepmind/dsprites-dataset
- 使用的固定提交：`fa310c66517cfc1939d77fe17c725154efc97127`。
- 原始 NPZ SHA-256：`857dc001287e8efa226e0025fac6b29e5745c621b380b0d3363e4e33f4e28fa2`。
- 官方 [LICENSE](third_party/dsprites/LICENSE) 与 [README](third_party/dsprites/README.upstream.md) 随本仓库保留。它们适用于相应第三方材料，不是原创代码的许可证。

公开复跑面板从上述固定版本中选取了 36 张 64×64 原始二值图像。面板保留官方行号、标签和因素，另附研究使用的配对编号与面积。[面板及核验记录](reproduction/held_condition/runs/dsprites_shape_donor_20261010/panel)。仓库也包含实验的缓存输出和研究图。图中的严格右移版本由研究代码派生。完整数据集可由 `prepare_dsprites.py` 从上述固定提交下载。

## Fashion-MNIST

官方来源：https://github.com/zalandoresearch/fashion-mnist 。本仓库的小示例包含冻结模型的缓存响应与内部状态，不包含原始服装图像。下载及来源核验逻辑见 `source/scripts/prepare_fashion_mnist.py`。

## CAT-MT v0.7 研究前身

`source/upstream/CAT_MT_phase7_v0.7/` 是本研究已有历史研究包的源码依赖，沿用其原目录名称。`PROVENANCE.json` 记录其源文件与公开注释副本的哈希。这里的“upstream”表示研究历史来源，不表示与 dSprites 作者、论文作者或外部项目存在合作关系。

历史研究包没有随此次发布提供可据以为整个项目重新授权的许可证，因此本次未自行选择 MIT、Apache 等整体许可证。
