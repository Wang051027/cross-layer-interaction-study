"""Fit additive and pairwise responses from empty, singleton, and pair masks. A set bit scales its activation by 1 − alpha. The Walsh transform uses the full cube."""
from __future__ import annotations
import itertools
import numpy as np


# 枚举所有二元干预组合。
# q 是源坐标数；整数索引的第 i 位对应坐标 i。
# float64 数组 [2**q,q]；1 表示置零/衰减，0 表示保持原值。
def cube(q):
    return ((np.arange(1 << q)[:, None] >> np.arange(q)) & 1).astype(np.float64)


# values[图像,组合,类别]；bits[组合,q]，组合按二进制整数升序。
# 返回一阶预测、二阶预测和系数字典；拟合不读取三阶及以上组合。
def origin_models(values, bits):
    """Fit at empty/single/pair masks only; retain a separate input batch axis."""
    q = bits.shape[1]
    pairs = list(itertools.combinations(range(q), 2))
    # 整数 1<<i 对应只干预第 i 个源坐标，因此可直接索引响应表。
    singleton = 1 << np.arange(q)
    baseline = values[:, 0, :]
    first = values[:, singleton, :] - baseline[:, None, :]
    pair_ids = np.array([(1 << i) | (1 << j) for i, j in pairs], dtype=int)
    # 二阶有限差分扣除两个单独效应，并加回一次共同基线。
    second = np.stack([
        values[:, (1 << i) | (1 << j), :] - values[:, 1 << i, :]
        - values[:, 1 << j, :] + baseline for i, j in pairs
    ], axis=1)
    # einsum 的 m/i/c/d 分别对应组合、源坐标、图像与输出类别。
    linear = baseline[:, None, :] + np.einsum('mi,cid->cmd', bits, first)
    # 只有某一对源坐标同时被干预时，才启用该成对修正项。
    pair_features = np.stack([bits[:, i] * bits[:, j] for i, j in pairs], axis=1)
    quadratic = linear + np.einsum('mp,cpd->cmd', pair_features, second)
    return linear, quadratic, dict(baseline=baseline, first=first, second=second,
        singleton_ids=singleton, pair_ids=pair_ids)


# values 为完整响应数组，axis 指组合轴；该轴长度必须是 2 的幂。
# 返回同形数组；连续变换两次得到组合数乘以原数组。
def fwht(values, axis=1):
    """Unnormalised Walsh-Hadamard transform; H(H(v)) = number_of_masks * v."""
    moved = np.moveaxis(np.asarray(values, dtype=np.float64), axis, 0).copy()
    n = len(moved)
    if n == 0 or n & (n - 1):
        raise ValueError('Walsh transform requires a complete power-of-two cube')
    # 每轮把相邻长度为 step 的区块相加/相减；总复杂度为 O(M log M)。
    step = 1
    while step < n:
        view = moved.reshape((-1, 2 * step) + moved.shape[1:])
        left, right = view[:, :step].copy(), view[:, step:].copy()
        view[:, :step] = left + right
        view[:, step:] = left - right
        step *= 2
    return np.moveaxis(moved, 0, axis)


# 按 Walsh 系数的阶数分解响应能量，诊断最佳低阶近似的遗漏。
# 完整响应 values 和每个基函数索引的置位数 orders。
# 返回系数、各阶能量、逆变换误差与 Parseval 能量差；使用了全表答案。
def walsh_analysis(values, orders):
    # 正交展开系数在正变换时除以组合数。
    coefficients = fwht(values) / len(orders)
    per_order = np.stack([
        np.square(coefficients[:, orders == degree, :]).sum(axis=1)
        for degree in range(int(orders.max()) + 1)
    ], axis=1)
    recon = fwht(coefficients)
    # 全表平均平方应等于全部系数平方和，用于检查变换尺度。
    parseval = float(np.max(np.abs(np.square(values).mean(axis=1) - per_order.sum(axis=1))))
    return coefficients, per_order, float(np.max(np.abs(recon - values))), parseval


# values[图像,组合,10]；labels[图像]，标签不随组合变化。
# [图像,组合] 标量：真实类变化减去其他九类变化的均值。
def contrast(values, labels):
    """True-class logit minus mean of the nine other logits; mask-independent."""
    selected = np.take_along_axis(values, np.asarray(labels)[:, None, None], axis=2)[..., 0]
    return (10 * selected - values.sum(axis=2)) / 9


# 在同一留出组合集合上计算每张输入的绝对和相对预测误差。
# prediction/reference 形状一致；mask 指评分组合；effect_floor 为分母下限。
# 返回每图像 RMSE、效应 RMS、NRMSE、MAE、95 分位及最大绝对误差。
def per_input_metrics(prediction, reference, mask, effect_floor=1e-9):
    diff = prediction[:, mask] - reference[:, mask]
    ref = reference[:, mask]
    axes = tuple(range(1, diff.ndim))
    rms = np.sqrt(np.square(diff).mean(axis=axes))
    effect = np.sqrt(np.square(ref).mean(axis=axes))
    ratio = np.full_like(rms, np.nan)
    # 效应接近零时保留 NaN，避免把不可定义的相对误差写成零。
    np.divide(rms, effect, out=ratio, where=effect > effect_floor)
    return dict(rmse=rms, effect_rms=effect, nrmse=ratio,
                mae=np.abs(diff).mean(axis=axes),
                p95_abs=np.quantile(np.abs(diff).reshape(len(diff), -1), .95, axis=1),
                max_abs=np.abs(diff).reshape(len(diff), -1).max(axis=1))


# h[n,q]、z[n,J]、W[J,q]、U[10,J]；bits[M,q]；alpha 为衰减强度。
# 输出[n,M,10]、非线性余项和状态统计；这是结构恒等式核验。
def independent_state_response(h, z, W, U, bits, alpha, labels):
    """Reconstruct outputs from saved states and count threshold crossings on the complete binary cube. Keep near-zero states in the candidate set."""
    outputs, nonlinear, rows = [], [], []
    for hi, zi, label in zip(h, z, labels):
        # 每列是一处源坐标被干预时对全部下游状态的增量。
        changes = -alpha * W * hi[None]
        # 独立选择各坐标时，所有负增量之和/正增量之和给出范围端点。
        low = changes.clip(max=0).sum(axis=1)
        high = changes.clip(min=0).sum(axis=1)
        # 为接近零的状态加上 float64 舍入余量。
        guard = 64 * np.finfo(np.float64).eps * (np.abs(zi) + high - low + 1)
        candidates = ~((zi + low > guard) | (zi + high < -guard))
        strict = (zi + low < -guard) & (zi + high > guard)
        offsets = bits @ changes.T
        responses = np.maximum(zi[None] + offsets, 0) - np.maximum(zi, 0)[None]
        outputs.append(responses @ U.T)
        # 从精确 ReLU 变化中扣掉原符号下的线性变化，剩下跨界造成的余项。
        rem = responses - offsets * (zi > 0)[None]
        omitted = rem[:, ~candidates]
        nonlinear.append(rem[:, candidates] @ U[:, candidates].T)
        endpoint_rem = np.maximum(
            np.maximum(zi + low, 0) - np.maximum(zi, 0) - low * (zi > 0),
            np.maximum(zi + high, 0) - np.maximum(zi, 0) - high * (zi > 0))
        # 先把十类读出权重投影到固定目标，再构造该目标的仿射余项上界。
        target_weights = (10 * U[int(label)] - U.sum(axis=0)) / 9
        rows.append(dict(candidate_states=int(candidates.sum()),
            strict_crossing_states=int(strict.sum()),affected_states=len(zi),
            candidate_fraction=float(candidates.mean()),
            source_l2=float(np.linalg.norm(hi)),
            target_affine_remainder_bound=float(np.abs(target_weights) @ endpoint_rem),
            max_omitted_remainder=float(np.abs(omitted).max()) if omitted.size else 0.))
    return np.stack(outputs), np.stack(nonlinear), rows
