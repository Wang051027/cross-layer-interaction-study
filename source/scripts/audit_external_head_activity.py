# 中文阅读说明｜用途：检查已冻结 CNN 的头部激活频率，解释低预算表现是否受不活跃单元影响。
# 数据流：读取教师拟合状态；输出逐教师、逐层活动统计，属于结果后的描述性检查。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Post-test descriptive diagnostic from already frozen training fits only.

This does not fit or change a predictor, access new inputs, or conduct a test.
"""
from pathlib import Path
import csv
import json
import numpy as np

BASE = Path(__file__).resolve().parents[1]
RUN = BASE / "runs/fashion_external_20260911"


# 用途：检查已冻结 CNN 的头部激活频率，解释低预算表现是否受不活跃单元影响。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def main():
    records = []
    for seed in range(202609071, 202609077):
        for layer in (1, 2):
            with np.load(RUN / f"seed_{seed}/fit_L{layer}.npz", allow_pickle=False) as fit:
                gate = fit["gate"]
                mixed = (gate > 0) & (gate < 1)
                records.append(dict(seed=seed, layer=layer, training_contexts=512,
                    positive_on_some_training_inputs=int((gate > 0).sum()),
                    never_positive_on_training_inputs=int((gate == 0).sum()),
                    always_positive_on_training_inputs=int((gate == 1).sum()),
                    mixed_gate_units=int(mixed.sum()),
                    mixed_units_omitted_by_greedy32=int(mixed[fit["order"][32:]].sum()),
                    mixed_units_omitted_by_weight32=int(mixed[fit["weight_order"][32:]].sum()),
                    max_absolute_training_gain_after32=float(np.abs(fit["train_gain"][32:]).max())))
    target = BASE / "audit/head_activity_20260911.csv"
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    # 把结构化配置或结果落盘，供后续阶段核验和报告读取。
    (BASE / "audit/head_activity_20260911.json").write_text(json.dumps(dict(
        type="post-test descriptive diagnostic; no confirmatory inference",
        source="frozen training fit gate probabilities and selection orders",
        new_model_evaluations=0, training_inputs=512,
        limitations="sample inactivity is not proof of global inactivity or inactivity under unseen interventions",
        records=records), ensure_ascii=False, indent=2), encoding="utf-8")
    print(records)


# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__ == "__main__":
    main()
