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


def clean(value):
    """NumPy 标量转为 JSON 标量，非有限浮点值写作 null。"""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def main(output_dir=None):
    # 响应数组的三个轴依次是 4 张图像、256 种组合和 10 个类别。
    with np.load(ROOT / "example/input.npz", allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    expected = json.loads((ROOT / "example/expected_metrics.json").read_text(encoding="utf-8"))
    expected = {(r["alpha"], r["input_id"]): r for r in expected}
    q = data["h"].shape[1]
    bits = cube(q)
    order = bits.sum(axis=1).astype(int)
    held = order > 2  # 空组、单组和双组用于拟合，三组及以上用于评分。
    assert q == 8 and len(bits) == 256 and held.sum() == 219
    rows, predictions = [], {}
    max_metric_difference = 0.
    max_state_difference = 0.
    for alpha in (1., .5):
        reference = data[f"responses_{alpha}"]
        assert reference.shape == (4, 256, 10)

        # 加性模型使用 9 个响应，成对模型使用 37 个。
        additive, quadratic, _ = origin_models(reference, bits)
        target = contrast(reference, data["labels"])
        score = {}
        for name, prediction in (("single", additive), ("pair", quadratic)):
            score[name, "scalar"] = per_input_metrics(contrast(prediction, data["labels"]), target, held)
            score[name, "vector"] = per_input_metrics(prediction, reference, held)

        # 在全部 256 个响应上计算低阶投影。
        coefficients, _, inverse_error, parseval_error = walsh_analysis(reference, order)
        scalar_coefficients = contrast(coefficients, data["labels"])
        high_order_rmse = np.sqrt(np.square(scalar_coefficients[:, order > 2]).sum(axis=1))
        full_effect_rms = np.sqrt(np.square(target).mean(axis=1))

        # 用缓存的内部状态重建输出。
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

    result = dict(status="PASS",purpose="Coalition responses for four Fashion-MNIST images",
        inputs=4,source_units=q,coalitions=256,scored_coalitions=219,strengths=[1., .5],
        single_queries=9,pair_queries=37,new_network_forward_runs=0,
        maximum_difference_from_saved_metrics=max_metric_difference,
        maximum_state_reconstruction_difference=max_state_difference,rows=rows)
    if output_dir is not None:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        np.savez_compressed(output_dir / "predictions.npz", **predictions)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", help="保存 metrics.json 和 predictions.npz 的新目录")
    args = parser.parse_args()
    main(args.output_dir)
