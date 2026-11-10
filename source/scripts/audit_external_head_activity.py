"""Summarize head activations from the stored training fits."""
from pathlib import Path
import csv
import json
import numpy as np

BASE = Path(__file__).resolve().parents[1]
RUN = BASE / "runs/fashion_external_20260911"


# 检查已冻结 CNN 的头部激活频率，解释低预算表现是否受不活跃单元影响。
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
    (BASE / "audit/head_activity_20260911.json").write_text(json.dumps(dict(
        type="post-test descriptive diagnostic; no confirmatory inference",
        source="frozen training fit gate probabilities and selection orders",
        new_model_evaluations=0, training_inputs=512,
        limitations="sample inactivity is not proof of global inactivity or inactivity under unseen interventions",
        records=records), ensure_ascii=False, indent=2), encoding="utf-8")
    print(records)


if __name__ == "__main__":
    main()
