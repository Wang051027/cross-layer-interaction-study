# 中文阅读说明｜用途：用四张固定图像的历史响应演示单独/成对预测、误差评分与全组合诊断。
# 数据流：输入 example/ 中的数组和已保存指标；默认打印结果，指定 --output-dir 才另存文件。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""中文阅读入口：用四张图像的已保存响应，演示单独/成对组合预测。

运行 python -B run_demo.py。只读示例，无需 GPU、模型权重或网络连接。
它复用原始数学实现；这是历史数据示例，不生成新的网络实验结果。
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "source/scripts"
sys.path.insert(0, str(CORE))
from joint_response import (cube, origin_models, contrast, per_input_metrics,
                            walsh_analysis, independent_state_response)


# 用途：递归把 NumPy 数值转为可序列化对象，并将不可定义的非有限数记为 null。
# 参数：value；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：字典（字段见 return） / [clean(v) for v in value] / float(value) if np.isfinite(value) else None；另有分支返回，见函数体。
def clean(value):
    """把不可定义的相对误差保存为 null，而不是伪装成零。"""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


# 用途：用四张固定图像的历史响应演示单独/成对预测、误差评分与全组合诊断。
# 参数：output_dir；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：result。
def main(output_dir=None):
    # 1. 数据来自历史真实网络输出：4 张图像、256 种组合、10 个类别。
    with np.load(ROOT / "example/input.npz", allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    expected = json.loads((ROOT / "example/expected_metrics.json").read_text(encoding="utf-8"))
    expected = {(r["alpha"], r["input_id"]): r for r in expected}
    q = data["h"].shape[1]
    bits = cube(q)
    order = bits.sum(axis=1).astype(int)
    held = order > 2  # 拟合只看空、单、双组合；评分看三个以上同时干预。
    assert q == 8 and len(bits) == 256 and held.sum() == 219
    rows, predictions = [], {}
    max_metric_difference = 0.
    max_state_difference = 0.
    for alpha in (1., .5):
        reference = data[f"responses_{alpha}"]
        assert reference.shape == (4, 256, 10)

        # 2. 仅使用 9/37 个响应计算预测公式，无神经网络训练。
        additive, quadratic, _ = origin_models(reference, bits)
        target = contrast(reference, data["labels"])
        score = {}
        for name, prediction in (("single", additive), ("pair", quadratic)):
            score[name, "scalar"] = per_input_metrics(contrast(prediction, data["labels"]), target, held)
            score[name, "vector"] = per_input_metrics(prediction, reference, held)

        # 3. 全表最佳低阶投影访问全部 256 个答案，只作为诊断。
        coefficients, _, inverse_error, parseval_error = walsh_analysis(reference, order)
        scalar_coefficients = contrast(coefficients, data["labels"])
        high_order_rmse = np.sqrt(np.square(scalar_coefficients[:, order > 2]).sum(axis=1))
        full_effect_rms = np.sqrt(np.square(target).mean(axis=1))

        # 4. 使用保存的内部状态独立重建响应，检查与原网络输出是否一致。
        reconstructed, _, _ = independent_state_response(
            data["h"], data["z"], data["W"], data["U"], bits, alpha, data["labels"])
        state_difference = float(np.abs(reconstructed - reference).max())
        max_state_difference = max(max_state_difference, state_difference)
        assert max(state_difference, inverse_error, parseval_error) < 1e-9

        for i, input_id in enumerate(data["ids"]):
            row = {"input_id": int(input_id), "label": int(data["labels"][i]), "alpha": alpha}
            for (method, target_name), metrics in score.items():
                for key, values in metrics.items():
                    row[f"{method}_{target_name}_{key}"] = float(values[i])
            row["scalar_oracle_degree2_rmse"] = float(high_order_rmse[i])
            row["scalar_oracle_degree2_nrmse"] = (float(high_order_rmse[i] / full_effect_rms[i])
                if full_effect_rms[i] > 1e-9 else np.nan)
            old = expected[alpha, int(input_id)]
            for key, value in row.items():
                if key in ("input_id", "label", "alpha"):
                    assert value == old[key]
                else:
                    if old[key] is None:
                        assert np.isnan(value), key
                    else:
                        error = abs(value - old[key])
                        assert error < 1e-9, (key, error)
                        max_metric_difference = max(max_metric_difference, error)
            rows.append(clean(row))
        predictions[f"single_{alpha}"] = additive
        predictions[f"pair_{alpha}"] = quadratic

    result = dict(status="PASS",purpose="Historical four-input teaching example; not new validation",
        inputs=4,source_units=q,coalitions=256,scored_coalitions=219,strengths=[1., .5],
        single_queries=9,pair_queries=37,new_network_forward_runs=0,
        maximum_difference_from_saved_metrics=max_metric_difference,
        maximum_state_reconstruction_difference=max_state_difference,rows=rows)
    # 默认只打印；显式指定目录时才另存结果，而且拒绝覆盖已有目录。
    if output_dir is not None:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
        # 把结构化配置或结果落盘，供后续阶段核验和报告读取。
        (output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        # 保存具名数值数组，便于后来按同一索引回读和独立复算。
        np.savez_compressed(output_dir / "predictions.npz", **predictions)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__ == "__main__":
    # 解析运行阶段/路径等命令行参数；具体允许的阶段由 choices 或分派逻辑限定。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", help="可选：将结果写入一个尚不存在的目录")
    args = parser.parse_args()
    main(args.output_dir)
