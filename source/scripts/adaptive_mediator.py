"""第二轮：在相同候选空间内比较固定/逐输入选择与梯度/有限干预评分。

复用第一轮的模型计算和中介修补；本文件只增加选择规则。
逐输入选择只看半强度响应。完整强度组的结果不得回流到选择中。
"""
from __future__ import annotations
import numpy as np
import torch
from cross_layer_mechanism import pair_difference, target, tail_from_middle, patch_outputs

BUDGETS = (8, 16, 32)
METHODS = ('fixed_gradient', 'input_gradient', 'fixed_finite', 'input_finite',
           'fixed_single', 'input_single', 'input_gradient_full', 'legacy_fixed',
           'random_0', 'random_1', 'random_2', 'random_3')


def ranked_nodes(scores, nodes):
    """按分数降序排列候选，完全相等时保留候选的原顺序；不看评价结果。"""
    nodes = np.asarray(nodes, dtype=int)
    return nodes[np.argsort(-np.asarray(scores)[nodes], kind='stable')]


def input_features(model, middle, label, stencil):
    """计算该图的交互梯度分数、单独效应分数和激活大小。

    与第一轮的分数定义一致，区别是这里保留逐图结果，且只使用 α=0.5。
    梯度取无干预基线；它可能遗漏有限干预引起的门控变化。
    """
    baseline = middle[:1].detach().clone().requires_grad_(True)
    grad = torch.autograd.grad(target(tail_from_middle(model, baseline), label).sum(), baseline)[0]
    with torch.no_grad():
        interaction = pair_difference(middle, stencil)
        single = middle[[1, 2, 4, 8]] - middle[:1]
        values = [(interaction * grad).abs().mean(0), (single * grad).abs().mean(0), middle[0].abs()]
        return np.stack([v.flatten().cpu().numpy() for v in values])


def make_shortlist(features, pool, size=64):
    """训练均值的交互前32与单独效应前32取并集，再按交互排序补齐64。

    主比较的四个方法始终使用同一名单，避免把候选范围变化误当方法提升。
    返回的坐标按编号排序，使并列分数的处理可复现。
    """
    inter = ranked_nodes(features[0], pool)
    single = ranked_nodes(features[1], pool)
    chosen = set(map(int, inter[:size//2])) | set(map(int, single[:size//2]))
    for node in inter:
        if len(chosen) == size:
            break
        chosen.add(int(node))
    assert len(chosen) == size
    return np.array(sorted(chosen), dtype=int)


def scalar_output(logits, label):
    """NumPy 实现真实类与其余类平均的 logit 差，保留前面的任意轴。"""
    return (10 * logits[..., int(label)] - logits.sum(-1)) / 9


def contrast_last(values, stencil):
    """沿末尾的 16 个组合计算 24 个条件成对差分。"""
    return (values[..., stencil[:, 0]] - values[..., stencil[:, 1]]
            - values[..., stencil[:, 2]] + values[..., stencil[:, 3]])


def finite_scores(singleton_logits, natural_logits, label, stencil):
    """按实际单坐标修补的联合平方误差改善评分，可正可负。

    原交互 I；恢复交互 R_j；移植交互 T_j。
    s_j = mean(I²) - [mean(R_j²)+mean((T_j-I)²)]/2。
    分数来自完整有限干预，不是梯度。但单坐标排序不保证组的最优性，
    可能遗漏只有多个坐标一起修补才出现的协作。这里不按小分母归一化。
    """
    original = contrast_last(scalar_output(natural_logits, label), stencil)
    patched = contrast_last(scalar_output(singleton_logits, label), stencil)
    return np.mean(original**2) - .5 * (np.mean(patched[:, 0]**2, axis=-1)
                                      + np.mean((patched[:, 1]-original)**2, axis=-1))


def score_to_order(scores, shortlist):
    """有限评分的轴直接对应 shortlist；不把坐标编号误当作分数下标。"""
    return np.asarray(shortlist)[np.argsort(-np.asarray(scores), kind='stable')]


@torch.no_grad()
def singleton_outputs(model, middle, shortlist):
    """实际评价64个单坐标，返回[64,恢复/移植,16组合,10类别]原始 logits。"""
    return patch_outputs(model, middle, [[int(n)] for n in shortlist]).cpu().numpy()


def fixed_orders(mean_features, mean_finite, shortlist, seed):
    """只用训练数据冻结固定排序；随机排序也固定于教师并供所有输入复用。"""
    result = dict(fixed_gradient=ranked_nodes(mean_features[0], shortlist).tolist(),
                  fixed_single=ranked_nodes(mean_features[1], shortlist).tolist(),
                  fixed_finite=score_to_order(mean_finite, shortlist).tolist())
    for repeat in range(4):
        rng = np.random.default_rng(seed + 2909202600 + repeat)
        result[f'random_{repeat}'] = rng.permutation(shortlist).tolist()
    return result


def selection_specs(pool_size):
    """每预算12种选择；另设整个64候选和整个结构可达集合的容量参照。"""
    result = [dict(method=method, budget=k) for k in BUDGETS for method in METHODS]
    return result + [dict(method='all_shortlist', budget=64),
                     dict(method='all_reachable', budget=int(pool_size))]


def select_for_input(features, finite, shortlist, pool, fixed, legacy):
    """在 α=0.5 上一次决定所有预算的集合；同一集合用于之后各强度评价。

    input_gradient_full 允许使用全部结构可达点，是候选范围诊断，
    不能混入主2×2比较。legacy_fixed 保留上一轮原选择，用于同批次参照。
    """
    orders = dict(fixed)
    orders.update(input_gradient=ranked_nodes(features[0], shortlist),
                  input_single=ranked_nodes(features[1], shortlist),
                  input_finite=score_to_order(finite, shortlist),
                  input_gradient_full=ranked_nodes(features[0], pool))
    chosen = []
    for spec in selection_specs(len(pool)):
        method, k = spec['method'], spec['budget']
        if method == 'legacy_fixed':
            nodes = next(s['nodes'] for s in legacy if s['method']=='interaction' and s['budget']==k)
        elif method == 'all_shortlist':
            nodes = shortlist
        elif method == 'all_reachable':
            nodes = pool
        else:
            nodes = orders[method][:k]
        nodes = np.asarray(nodes, dtype=int)
        assert len(nodes) == k and len(np.unique(nodes)) == k
        assert set(nodes).issubset(set(pool))
        chosen.append(nodes)
    return chosen


def encode_selections(selections, width):
    """用-1填充可变长度名单，原始预算由 specs 决定，避免 object 数组。"""
    encoded = np.full((len(selections), width), -1, dtype=np.int64)
    for i, nodes in enumerate(selections):
        encoded[i, :len(nodes)] = nodes
    return encoded


def metric_arrays(natural, patched, labels, stencil):
    """从原始输出计算每输入、每方案的绝对均方量，再由调用方做教师内汇总。

    natural=[N,16,10]；patched=[N,方法,2,16,10]。
    保存绝对量可避免只报告近零分母下不稳定的比率。
    """
    ni = np.stack([contrast_last(scalar_output(n, y), stencil) for n, y in zip(natural, labels)])
    pi = np.stack([contrast_last(scalar_output(p, y), stencil) for p, y in zip(patched, labels)])
    natural_mse = np.mean(ni**2, axis=-1)
    restore_mse = np.mean(pi[:, :, 0]**2, axis=-1)
    transplant_mse = np.mean((pi[:, :, 1]-ni[:, None])**2, axis=-1)
    return natural_mse, restore_mse, transplant_mse
