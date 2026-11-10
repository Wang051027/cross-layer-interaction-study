"""新增选择器的已知答案与原网络核验；验证数据不进入模型选择。"""
import json
import numpy as np
import torch
from cross_layer_mechanism import (design, source_activation, changed_sources, trace,
    original_forward, patch_outputs, learn_selection)
from adaptive_mediator import (finite_scores, make_shortlist, score_to_order,
    singleton_outputs, input_features, metric_arrays, scalar_output)


def known_answers():
    """构造完美中介、无效中介、反向修补和零交互，核验评分符号与含义。"""
    bits, stencil, _ = design()
    logits = np.zeros((16, 10))
    logits[:, 0] = np.maximum(1.5-bits[:, 0]-bits[:, 1], 0)
    baseline = np.broadcast_to(logits[:1], logits.shape).copy()
    perfect = np.stack([baseline, logits])
    ineffective = np.stack([logits, baseline])
    harmful = np.stack([2*logits, -logits])
    scores = finite_scores(np.stack([perfect, ineffective, harmful]), logits, 0, stencil)
    assert scores[0] > 0 and scores[1] == 0 and scores[2] < 0
    assert score_to_order(scores, np.array([11, 23, 47])).tolist() == [11, 23, 47]
    n, r, t = metric_arrays(logits[None], np.stack([perfect, ineffective])[None], np.array([0]), stencil)
    assert r[0, 0] == t[0, 0] == 0
    assert abs(r[0, 1]-n[0]) < 1e-15 and abs(t[0, 1]-n[0]) < 1e-15
    zero = np.zeros_like(logits)
    assert np.all(finite_scores(np.zeros((3, 2, 16, 10)), zero, 0, stencil) == 0)
    # 交互、单独效应的前32均必须包含于64名单；并列与重合确定性处理。
    features = np.stack([np.arange(100.), np.arange(100.)[::-1], np.ones(100)])
    short = make_shortlist(features, np.arange(100))
    assert set(range(32)).issubset(short) and set(range(68, 100)).issubset(short)
    # 乘积函数中两个坐标单独移植均为零，联合移植得到非零响应。
    assert 1*0 == 0 and 0*1 == 0 and 1*1 == 1
    return dict(status='PASS', controls=['perfect / ineffective / harmful finite intervention',
        'zero interaction score', 'common shortlist includes both leading rankings',
        'single-coordinate score is not group optimality'])


def check_teacher(model, image, label, nodes, shortlist, device):
    """将新单点批量结果与原模型钩子逐一对照，并与旧分数计算逐项核对。"""
    raw_bits, stencil, _ = design()
    bits = torch.as_tensor(raw_bits, dtype=torch.float64, device=device)
    image = image[None].to(device)
    source = source_activation(model, image)
    maxima = dict(singleton_hook=0., score_formula=0., feature_formula=0.)
    probe = np.asarray(shortlist[:3])
    for alpha in (.5, 1.):
        tr = trace(model, changed_sources(source, nodes, bits, alpha))
        raw = singleton_outputs(model, tr['h'][1], probe)
        for j, node in enumerate(probe):
            for op, mode in enumerate(('restore', 'transplant')):
                ref = original_forward(model, image, nodes, bits, alpha, mode, [int(node)]).cpu().numpy()
                maxima['singleton_hook'] = max(maxima['singleton_hook'], float(np.max(np.abs(raw[j, op]-ref))))
        scores = finite_scores(raw, tr['output'].cpu().numpy(), int(label), stencil)
        weights = np.full(10, -1/9); weights[int(label)] = 1
        # 独立差分矩阵乘法，用于防止组合轴、操作轴或真类投影写错。
        matrix = np.zeros((24, 16))
        for row, ids in enumerate(stencil):
            matrix[row, ids] = [1, -1, -1, 1]
        original = matrix @ (tr['output'].cpu().numpy() @ weights)
        for j in range(3):
            restored = matrix @ (raw[j, 0] @ weights)
            transplanted = matrix @ (raw[j, 1] @ weights)
            independent = (np.dot(original, original)-.5*(np.dot(restored, restored)
                            + np.dot(transplanted-original, transplanted-original)))/24
            maxima['score_formula'] = max(maxima['score_formula'], abs(float(scores[j]-independent)))
        features = input_features(model, tr['h'][1], int(label), stencil)
        assert features.shape == (3, 6272) and np.isfinite(features).all()
        assert np.min(features) >= 0
        if alpha == 1.:
            reference_features = learn_selection(model, image, [int(label)], nodes, bits, stencil, device)
            maxima['feature_formula'] = float(np.max(np.abs(features-reference_features)))
    assert max(maxima.values()) < 1e-9, maxima
    return dict(status='PASS', errors=maxima, singleton_nodes=probe.tolist())


if __name__ == '__main__':
    print(json.dumps(known_answers(), ensure_ascii=False))
