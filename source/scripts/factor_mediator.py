"""在开发集上拟合类别、大小和角度与单点收益的关系，从 64 个候选中选取前 16 个进行联合恢复和移植。"""
from __future__ import annotations

import numpy as np

LAMBDAS = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
RULES = ('global_fixed', 'class_fixed', 'class_scale', 'class_angle_ridge',
         'class_factor_ridge', 'class_factor_knn')


def basis(factors, physical, rule):
    """大小用固定索引缩放，角度用官方物理弧度；不能误用40等分角度。

因素规则包含大小二次项、角度的三阶傅里叶项以及二者的乘积。返回矩阵
不含常数列，截距单独拟合。这样的基函数是待检验的平滑近似，不是理论定理。
    """
    theta = np.asarray(physical)[:, 3]
    angular = np.column_stack([f(h * theta) for h in (1, 2, 3)
                               for f in (np.sin, np.cos)])
    if rule == 'class_angle_ridge':
        return angular
    assert rule == 'class_factor_ridge'
    u = (np.asarray(factors)[:, 2] - 2.5) / 2.5
    size = np.column_stack((np.ones(len(u)), u, u ** 2))
    angle = np.column_stack((np.ones(len(u)), angular))
    return (size[:, :, None] * angle[:, None, :]).reshape(len(u), -1)[:, 1:]


def fit_ridge(x, y, penalty):
    """仅用拟合行标准化；截距不受惩罚，返回可审计的数值参数。

目标函数为残差平方和加 penalty×系数平方和。常数特征标准差置1，避免
除零；系数由正定线性方程求解，不需要或读取评价分数。
    """
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-12, scale, 1.)
    z = (x - mean) / scale
    intercept = y.mean(0)
    coef = np.linalg.solve(z.T @ z + penalty * np.eye(z.shape[1]),
                           z.T @ (y - intercept))
    return dict(mean=mean.tolist(), scale=scale.tolist(), intercept=intercept.tolist(),
                coef=coef.tolist(), penalty=float(penalty))


def predict_ridge(model, x):
    """冻结的变换和系数可在新角度上直接给出64个候选的预测收益。"""
    return ((x - np.asarray(model['mean'])) / np.asarray(model['scale'])) @ np.asarray(
        model['coef']) + np.asarray(model['intercept'])


def tune_ridge(factors, physical, scores, rule):
    """整组留出一个开发角度：同时留出三类和六种大小，避免角度泄漏。

一个教师、一个输出目标共用一个惩罚系数，三类的回归分别拟合。按全部
留出行及64候选的原始分数MSE选系数；不按新面板的修补效果选择。
    """
    x, labels = basis(factors, physical, rule), factors[:, 1]
    angles = np.unique(factors[:, 3])
    cv = np.empty((len(LAMBDAS), len(scores), scores.shape[1]))
    for li, penalty in enumerate(LAMBDAS):
        for angle in angles:
            hold = factors[:, 3] == angle
            for label in range(3):
                train = (labels == label) & ~hold
                test = (labels == label) & hold
                cv[li, test] = predict_ridge(fit_ridge(x[train], scores[train], penalty), x[test])
    mse = np.mean((cv - scores[None]) ** 2, axis=(1, 2))
    best = int(np.argmin(mse))  # 相等时保留预先列出的较小惩罚系数。
    models = [fit_ridge(x[labels == c], scores[labels == c], LAMBDAS[best]) for c in range(3)]
    return models, dict(lambdas=list(LAMBDAS), mse=mse.tolist(), chosen=LAMBDAS[best]), cv


def train_rules(factors, physical, scores):
    """全局、按类、按类与大小的均值表，加两个平滑模型和固定四邻居参照。"""
    labels = factors[:, 1]
    model = dict(global_fixed=scores.mean(0).tolist(),
                 class_fixed=[scores[labels == c].mean(0).tolist() for c in range(3)],
                 class_scale=[[scores[(labels == c) & (factors[:, 2] == s)].mean(0).tolist()
                               for s in range(6)] for c in range(3)],
                 class_factor_knn=dict(factors=factors.tolist(), physical=physical.tolist(),
                                       scores=scores.tolist(), neighbors=4))
    diagnostics, predictions = {}, {}
    for rule in ('class_angle_ridge', 'class_factor_ridge'):
        model[rule], diagnostics[rule], predictions[rule] = tune_ridge(factors, physical, scores, rule)
    return model, diagnostics, predictions


def predict_rule(model, factors, physical, rule):
    """按真实类别匹配规则；角度距离作周期处理，邻居数和大小尺度预先固定。"""
    labels = factors[:, 1]
    if rule == 'global_fixed':
        return np.repeat(np.asarray(model[rule])[None], len(factors), axis=0)
    if rule == 'class_fixed':
        return np.asarray(model[rule])[labels]
    if rule == 'class_scale':
        return np.asarray(model[rule])[labels, factors[:, 2]]
    if rule in ('class_angle_ridge', 'class_factor_ridge'):
        x = basis(factors, physical, rule)
        return np.stack([predict_ridge(model[rule][int(c)], row[None])[0]
                         for c, row in zip(labels, x)])
    assert rule == 'class_factor_knn'
    d = model[rule]
    previous, actual, score = map(np.asarray, (d['factors'], d['physical'], d['scores']))
    prediction = []
    for f, p in zip(factors, physical):
        candidates = np.flatnonzero(previous[:, 1] == f[1])
        distance_angle = np.angle(np.exp(1j * (actual[candidates, 3] - p[3]))) / np.pi
        distance_size = (previous[candidates, 2] - f[2]) / 5.
        order = np.argsort(distance_size ** 2 + distance_angle ** 2, kind='stable')
        prediction.append(score[candidates[order[:d['neighbors']]]].mean(0))
    return np.stack(prediction)


def top_coordinates(scores, pool, shortlist, budget=16):
    """分数轴是64候选；先映射到支持集索引，再映射到真实扁平层坐标。"""
    order = np.argsort(-scores, axis=-1, kind='stable')[..., :budget]
    return np.asarray(pool)[np.asarray(shortlist)[order]]


def known_answers():
    """验证可知线性解、截距、周期性、稳定并列与两级坐标映射。"""
    x = np.arange(21, dtype=float)[:, None] / 10
    y = np.column_stack((3 + 2 * x[:, 0], np.ones(len(x)) * 7))
    model = fit_ridge(x, y, 1e-10)
    assert np.max(np.abs(predict_ridge(model, x) - y)) < 1e-9
    constant = fit_ridge(np.ones((10, 2)), np.ones((10, 3)) * 4, .1)
    assert np.max(np.abs(predict_ridge(constant, np.ones((5, 2))) - 4)) < 1e-12
    factors = np.array([[0, 0, 2, 3, 15, 15], [0, 0, 4, 8, 15, 15]])
    physical = factors.astype(float); physical[:, 3] = [.4, 1.1]
    moved = physical.copy(); moved[:, 3] += 2 * np.pi
    for rule in ('class_angle_ridge', 'class_factor_ridge'):
        assert np.max(np.abs(basis(factors, physical, rule) - basis(factors, moved, rule))) < 1e-12
    assert np.array_equal(top_coordinates(np.zeros((1, 3)), np.array([100, 500, 900, 1200]),
                                          np.array([3, 0, 2]), 2), [[1200, 100]])
    return dict(status='PASS',controls=['known linear fit', 'unpenalized intercept',
                 'constant features', '2pi periodicity', 'stable ties and coordinate mapping'])
