# 中文阅读说明｜用途：按原分组算法划分 Fashion-MNIST，并固定方法训练、验证的上下文索引。
# 数据流：输入官方训练 IDX；输出图像/标签 NPY、划分 NPZ 和审计 JSON。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Use the upstream fixed grouping algorithm and seed; write outside upstream."""
from __future__ import annotations
import json
import hashlib
from datetime import datetime, timezone
from research_runtime import BASE, UPSTREAM, np
from prepare_external import load_idx, hashes, grouped_split


# 用途：按原分组算法划分 Fashion-MNIST，并固定方法训练、验证的上下文索引。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def main():
    raw = BASE / "data/fashion_mnist/raw"
    output = BASE / "data/fashion_mnist/prepared"
    output.mkdir(parents=True, exist_ok=False)
    x = load_idx(raw, "train-images-idx3-ubyte.gz")
    y = load_idx(raw, "train-labels-idx1-ubyte.gz")
    assert x.shape == (60000,28,28) and y.shape == (60000,)
    h = hashes(x)
    indices = grouped_split(y,h,2026090717)
    assert [len(indices[k]) for k in ("train","validation","calibration")] == [50000,5000,5000]
    assert len(np.unique(np.concatenate(list(indices.values())))) == 60000
    assert all(len(set(h[indices[a]]) & set(h[indices[b]])) == 0
               for a,b in (("train","validation"),("train","calibration"),("validation","calibration")))
    # 保存具名数值数组，便于后来按同一索引回读和独立复算。
    np.savez_compressed(output / "split_indices.npz", **indices)
    np.save(output / "official_train_images.npy",x)
    np.save(output / "official_train_labels.npy",y)
    np.savez_compressed(output / "context_indices.npz", train=indices["train"][:512],
                        validation=indices["validation"][:128], calibration_reserved=indices["calibration"][:256],
                        test_reserved=np.arange(256,dtype=np.int64))
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),
                split_seed=2026090717,algorithm="unchanged upstream code/prepare_external.py::grouped_split",
                source_sha256=hashlib.sha256((UPSTREAM / "code/prepare_external.py").read_bytes()).hexdigest(),
                split_sizes={key:len(value) for key,value in indices.items()},
                class_counts={key:np.bincount(y[value],minlength=10).tolist() for key,value in indices.items()},
                context_rule="first 512/128/256 entries in fixed randomized train/validation/calibration split order; official test IDs 0..255 reserved",
                official_test_model_access=False,calibration_model_access=False,
                normalization="float32 / 255, no augmentation or fitted normalization",
                split_indices_sha256=hashlib.sha256((output / "split_indices.npz").read_bytes()).hexdigest(),
                context_indices_sha256=hashlib.sha256((output / "context_indices.npz").read_bytes()).hexdigest())
    # 把结构化配置或结果落盘，供后续阶段核验和报告读取。
    (output / "split_audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__ == "__main__":
    main()
