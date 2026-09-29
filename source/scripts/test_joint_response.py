# 中文阅读说明｜用途：用已知一阶、二阶、纯三阶函数及 ReLU 例子检查核心数值实现。
# 数据流：固定随机种子生成控制数据；错误超过容差即停止，不使用真实模型训练。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Meaningful numerical controls for the finite-game analysis."""
import json
import numpy as np
from scipy.linalg import hadamard
from joint_response import cube, origin_models, fwht, walsh_analysis, contrast, independent_state_response


# 用途：用规模较小且有可比较答案的情形检查实现，数值不一致时立即停止。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：字典（字段见 return）。
def smoke():
    checks = {}
    rng = np.random.default_rng(20260925)
    for q in (4, 8):
        bits = cube(q)
        values = rng.normal(size=(2, len(bits), 3))
        direct = np.einsum('mn,cnd->cmd', hadamard(len(bits)), values)
        checks[f'fwht_direct_q{q}'] = float(np.max(np.abs(fwht(values) - direct)))
        c, energy, inv, parseval = walsh_analysis(values, bits.sum(axis=1).astype(int))
        checks[f'inverse_q{q}'] = inv
        checks[f'parseval_q{q}'] = parseval
        linear = 2 + bits @ rng.normal(size=(q, 3))
        quadratic = linear + (bits[:, 0] * bits[:, 2])[:, None] * np.array([2., -1., .5])
        pred1, pred2, _ = origin_models(quadratic[None], bits)
        assert np.max(np.abs(pred1 - quadratic)) > .1
        checks[f'exact_known_quadratic_q{q}'] = float(np.max(np.abs(pred2 - quadratic)))
        coef, _, _, _ = walsh_analysis(quadratic[None], bits.sum(axis=1).astype(int))
        # 评分只包含三个及以上源坐标同时被干预的组合，避免混入二阶拟合点。
        checks[f'known_quadratic_high_order_q{q}'] = float(np.max(np.abs(coef[:, bits.sum(axis=1)>2])))
    bits = cube(4)
    cubic = (bits[:, 0] * bits[:, 1] * bits[:, 2])[None, :, None]
    p1, p2, _ = origin_models(cubic, bits)
    assert np.all(p2 == 0) and np.max(np.abs(cubic - p2)) == 1
    coef, energy, _, _ = walsh_analysis(cubic, bits.sum(axis=1).astype(int))
    checks['known_cubic_degree3_energy'] = abs(float(energy[:, 3].sum()) - 1/64)
    h = np.array([[.3, .7, 1., .2]])
    W = rng.normal(size=(7, 4)); U = rng.normal(size=(10, 7))
    z = np.full((1, 7), 100.)
    vals, remainder, states = independent_state_response(h,z,W,U,bits,1.,[2])
    p1, p2, _ = origin_models(vals,bits)
    checks['stable_gate_additive'] = float(np.max(np.abs(p1-vals)))
    assert states[0]['candidate_states'] == 0
    z = rng.normal(size=(1, 7))
    vals, remainder, states = independent_state_response(h,z,W,U,bits,1.,[2])
    coef, _, _, _ = walsh_analysis(vals,bits.sum(axis=1).astype(int))
    rc = fwht(remainder)/len(bits)
    checks['state_remainder_high_orders'] = float(np.max(np.abs((coef-rc)[:,bits.sum(axis=1)>1])))
    labels = np.array([2, 5]); values = rng.normal(size=(2,16,10))
    weights = -np.ones((2,10))/9
    weights[np.arange(2),labels] = 1
    checks['fixed_contrast'] = float(np.max(np.abs(contrast(values,labels)-np.einsum('cmd,cd->cm',values,weights))))
    for name, value in checks.items():
        assert value < 1e-10,(name,value)
    assert [int((cube(q).sum(axis=1)>2).sum()) for q in (4,8,12)] == [5,219,4017]
    return dict(status='passed',checks=checks,controls=['arbitrary Walsh vs dense Hadamard',
        'known quadratic','pure cubic invisible to pair-at-origin queries',
        'stable ReLU additive','crossing ReLU nonlinear remainder','fixed target contrast'])


# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__ == '__main__':
    print(json.dumps(smoke(),ensure_ascii=False))
