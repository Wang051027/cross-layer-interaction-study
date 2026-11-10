"""由原始响应、差分矩阵和增广最小二乘重算因素迁移结果，核对划分中的重复图像。"""
from __future__ import annotations

import dsprites_teacher as teacher
import itertools
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
import run_dsprites_factor_transfer as run

ERRORS = {}
COUNTS = {}


def compare(name, actual, expected, atol=1e-10):
    """记录各项核验的数值个数和最大差异。"""
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all(), name
    difference = float(np.max(np.abs(a - b))) if a.size else 0.
    ERRORS[name] = max(ERRORS.get(name, 0.), difference)
    COUNTS[name] = COUNTS.get(name, 0) + int(a.size)
    assert np.allclose(a, b, atol=atol, rtol=1e-8), (name, difference)


def contrast_matrix():
    """将24个条件差分写成稀疏系数的稠密矩阵，独立于四列索引实现。"""
    rows = []
    for a, b in itertools.combinations(range(4), 2):
        for background in range(16):
            if background & (2 ** a + 2 ** b): continue
            row = np.zeros(16)
            row[background] = 1
            row[background + 2 ** a] = -1
            row[background + 2 ** b] = -1
            row[background + 2 ** a + 2 ** b] = 1
            rows.append(row)
    return np.stack(rows)


def interaction(raw, labels, mode, matrix):
    """先矩阵求差分，再投影类别方向；支持多种方法与两种修补操作。"""
    value = np.einsum('km,n...mc->n...kc', matrix, raw)
    classes = raw.shape[-1]
    if mode == 'vector':
        return value @ (np.eye(classes) - np.ones((classes, classes)) / classes)
    weights = -np.ones((len(labels), classes)) / (classes - 1)
    weights[np.arange(len(labels)), labels] = 1.
    return np.einsum('n...kc,nc->n...k', value, weights)[..., None]


def singleton_scores(natural, singleton, labels, matrix):
    """用单点修补的 logit 重算两种评价目标的收益。"""
    scores = []
    for mode in ('margin', 'vector'):
        original = interaction(natural, labels, mode, matrix)
        patch = interaction(singleton, labels, mode, matrix)
        scores.append(np.mean(original ** 2, axis=(-2, -1))[:, None] - .5 * (
            np.mean(patch[:, :, 0] ** 2, axis=(-2, -1)) +
            np.mean((patch[:, :, 1] - original[:, None]) ** 2, axis=(-2, -1))))
    return np.stack(scores, axis=1)


def independent_basis(factors, physical, rule):
    """用复指数与显式幂次构造因素基函数。"""
    theta = physical[:, 3]
    exponentials = np.exp(1j * theta[:, None] * np.array([1, 2, 3])[None])
    angular = np.stack((exponentials.imag, exponentials.real), axis=-1).reshape(len(theta), 6)
    if rule == 'class_angle_ridge': return angular
    u = (factors[:, 2] - 2.5) / 2.5
    values = np.column_stack((np.ones(len(theta)), angular))
    return np.column_stack([u ** p * values[:, j] for p in range(3) for j in range(7)][1:])


def independent_ridge(x, y, test, penalty):
    """用增广设计矩阵的最小二乘解核对正则回归，截距显式且不惩罚。"""
    mean, std = x.mean(0), x.std(0)
    std = np.where(std > 1e-12, std, 1.)
    z = np.column_stack((np.ones(len(x)), (x - mean) / std))
    regularization = np.diag(np.r_[0., np.repeat(np.sqrt(penalty), x.shape[1])])
    coef = np.linalg.lstsq(np.vstack((z, regularization)),
                         np.vstack((y, np.zeros((x.shape[1] + 1, y.shape[1])))), rcond=None)[0]
    return np.column_stack((np.ones(len(test)), (test - mean) / std)) @ coef


def audit_panel():
    """重新检查所有历史像素、官方因素对应和输出无关的方向覆盖选择。"""
    with np.load(run.RUN / 'panel/data.npz', allow_pickle=False) as p:
        ids, pixels, factors, physical = (p[k].copy() for k in ('official_ids', 'images', 'factors', 'physical'))
    full_images = np.load(run.FULL / 'imgs.npy', mmap_mode='r', allow_pickle=False)
    full_factors = np.load(run.FULL / 'latents_classes.npy', mmap_mode='r', allow_pickle=False)
    full_values = np.load(run.FULL / 'latents_values.npy', mmap_mode='r', allow_pickle=False)
    assert np.array_equal(pixels, full_images[ids]) and np.array_equal(factors, full_factors[ids])
    assert np.array_equal(physical, full_values[ids]) and set(np.unique(pixels)) == {0, 1}
    hashes = {run.digest(x) for x in pixels}; assert len(hashes) == 72
    previous = set()
    for split in ('train', 'validation', 'evaluation', 'translated'):
        previous.update(run.digest(x) for x in np.load(run.DATA / f'{split}_images.npy', mmap_mode='r', allow_pickle=False))
    assert not hashes & previous
    with np.load(run.DATA / 'indices.npz', allow_pickle=False) as a:
        used = set(map(int, np.concatenate([a[s] for s in ('train', 'validation', 'evaluation')])))
    assert not set(map(int, ids)) & used
    eligible = []
    for scale in range(6):
        candidate = []
        for angle in range(40):
            group = [(((c * 6 + scale) * 40 + angle) * 32 + 15) * 32 + 15 for c in range(3)]
            if not set(group) & used and not {run.digest(full_images[i]) for i in group} & previous:
                candidate.append(angle)
        eligible.append(set(candidate))
    common = sorted(set.intersection(*eligible)); candidates = []
    for angles in itertools.combinations(common, 4):
        group = [(((c * 6 + s) * 40 + a) * 32 + 15) * 32 + 15 for s in range(6) for a in angles for c in range(3)]
        if len({run.digest(full_images[i]) for i in group}) != 72: continue
        theta = full_values[np.array([((a * 32 + 15) * 32 + 15) for a in angles]), 3]
        differences = (theta[:, None] - np.arange(4)[None] * np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
        row, column = linear_sum_assignment(differences ** 2)
        candidates.append((round(float(np.sum(differences[row, column] ** 2)), 12), angles))
    assert tuple(np.unique(factors[:, 3])) == min(candidates)[1]
    assert all(int(np.sum((factors[:, 1] == c) & (factors[:, 2] == s))) == 4 for c in range(3) for s in range(6))
    return dict(status='PASS', images=72, duplicate_pixels=0, balanced_size_class_cells=18,
                chosen_angles=list(min(candidates)[1]), candidate_panels=len(candidates))


def audit_fitted(seed, matrix):
    """开发分数、所有交叉验证折、选定惩罚系数和最终新因素预测分别核验。"""
    folder = run.RUN / f'fitted/seed_{seed}'
    dev = np.load(run.OLD / f'development/seed_{seed}/discovery.npz', allow_pickle=False)
    score = singleton_scores(dev['natural'], dev['singleton_logits'], dev['labels'], matrix)
    compare('development_finite_scores', dev['finite_scores'], score)
    factors = np.load(run.DATA / 'development_factors.npy', allow_pickle=False)
    with np.load(run.RUN / 'panel/data.npz', allow_pickle=False) as p:
        physical, new_factors, new_values = p['development_physical'].copy(), p['factors'].copy(), p['physical'].copy()
    models, diagnostics = teacher.read(folder / 'models.json'), teacher.read(folder / 'cv.json')
    predictions = np.load(folder / 'predictions.npz', allow_pickle=False)
    cv_arrays = np.load(folder / 'cv_predictions.npz', allow_pickle=False)
    labels, new_labels = factors[:, 1], new_factors[:, 1]
    rules = ('global_fixed', 'class_fixed', 'class_scale', 'class_angle_ridge', 'class_factor_ridge', 'class_factor_knn')
    for ti, target in enumerate(('margin', 'vector')):
        y = dev['finite_scores'][:, ti]  # 回归的实际训练输入；上面已核对其原始来源。
        for ri, rule in enumerate(rules):
            predicted = np.empty((72, 64))
            if rule == 'global_fixed': predicted[:] = y.mean(0)
            elif rule == 'class_fixed':
                for c in range(3): predicted[new_labels == c] = y[labels == c].mean(0)
            elif rule == 'class_scale':
                for c in range(3):
                    for s in range(6):
                        predicted[(new_labels == c) & (new_factors[:, 2] == s)] = y[(labels == c) & (factors[:, 2] == s)].mean(0)
            elif rule == 'class_factor_knn':
                for i, (f, p) in enumerate(zip(new_factors, new_values)):
                    candidates = np.flatnonzero(labels == f[1])
                    angle = (physical[candidates, 3] - p[3] + np.pi) % (2 * np.pi) - np.pi
                    distance = ((factors[candidates, 2] - f[2]) / 5) ** 2 + (angle / np.pi) ** 2
                    nearest = candidates[np.argsort(distance, kind='stable')[:4]]
                    predicted[i] = y[nearest].mean(0)
            else:
                x = independent_basis(factors, physical, rule)
                new_x = independent_basis(new_factors, new_values, rule)
                candidates = diagnostics[target][rule]['lambdas']
                cv = np.empty((len(candidates), 144, 64))
                for li, penalty in enumerate(candidates):
                    for angle in np.unique(factors[:, 3]):
                        for c in range(3):
                            train = (labels == c) & (factors[:, 3] != angle)
                            test = (labels == c) & (factors[:, 3] == angle)
                            cv[li, test] = independent_ridge(x[train], y[train], x[test], penalty)
                compare('cv_predictions', cv_arrays[f'{target}_{rule}'], cv)
                mse = np.mean((cv - y[None]) ** 2, axis=(1, 2))
                compare('cv_mse', diagnostics[target][rule]['mse'], mse)
                chosen = candidates[int(np.argmin(mse))]
                assert chosen == diagnostics[target][rule]['chosen']
                for c in range(3):
                    predicted[new_labels == c] = independent_ridge(x[labels == c], y[labels == c], new_x[new_labels == c], chosen)
                    m = models[target][rule][c]
                    stored = ((new_x[new_labels == c] - m['mean']) / m['scale']) @ np.asarray(m['coef']) + m['intercept']
                    compare('stored_parameters', stored, predicted[new_labels == c])
            compare('factor_predictions', predictions['scores'][:, ti * 6 + ri], predicted)
            chosen = dev['pool'][dev['shortlist'][np.argsort(-predictions['scores'][:, ti * 6 + ri], axis=-1, kind='stable')[:, :16]]]
            assert np.array_equal(chosen, predictions['selections'][:, ti * 6 + ri])


def audit():
    """只有冻结先后、全部原始输出和历史交付同时通过才写最终审计记录。"""
    assert not (run.RUN / 'independent_audit.json').exists()
    started = time.perf_counter()
    for name in ('panel_freeze', 'fixed_freeze', 'adaptive_freeze'): run.verify_freeze(name)
    assert teacher.read(run.RUN / 'fixed_freeze.json')['utc'] < teacher.read(run.RUN / 'selection_opened.json')['utc']
    assert teacher.read(run.RUN / 'adaptive_freeze.json')['utc'] < teacher.read(run.RUN / 'evaluation_opened.json')['utc']
    panel = audit_panel(); matrix = contrast_matrix()
    summary = teacher.read(run.RUN / 'summary.json')
    rows = summary['methods']; checks = 0
    for seed in run.SEEDS:
        audit_fitted(seed, matrix)
        input_folder = run.RUN / f'input_selection/seed_{seed}'; run.verify_complete(input_folder)
        selection = np.load(input_folder / 'selection.npz', allow_pickle=False)
        actual_scores = singleton_scores(selection['natural'], selection['singleton_logits'], selection['labels'], matrix)
        compare('new_singleton_scores', selection['finite_scores'], actual_scores)
        specifications = teacher.read(input_folder / 'specs.json')
        predetermined = np.load(run.RUN / f'fitted/seed_{seed}/predictions.npz', allow_pickle=False)
        for i in range(72):
            for j, spec in enumerate(specifications):
                choice = selection['selections'][i, j, :spec['budget']]
                assert len(set(map(int, choice))) == spec['budget'] and set(choice) <= set(selection['pool'])
                if j < 12: expected = predetermined['selections'][i, j]
                elif j < 14:
                    order = np.argsort(-selection['finite_scores'][i, j - 12], kind='stable')[:16]
                    expected = selection['pool'][selection['shortlist'][order]]
                elif j < 18: expected = predetermined['selections'][i, j - 2]
                elif j == 18: expected = selection['pool'][selection['shortlist']]
                else: expected = selection['pool']
                assert np.array_equal(choice, expected)
        for alpha in run.STRENGTHS:
            folder = run.RUN / f'evaluation/seed_{seed}/alpha_{alpha}'; run.verify_complete(folder)
            raw = np.load(folder / 'responses.npz', allow_pickle=False)
            labels = raw['labels']
            compare('decomposition', raw['contributions'].sum(1), np.einsum('km,nmc->nkc', matrix, raw['natural']))
            compare('all_restore', raw['patched'][:, -1, 0], np.repeat(raw['natural'][:, :1], 16, axis=1))
            compare('all_transplant', raw['patched'][:, -1, 1], raw['natural'])
            if alpha == .5: compare('half_replay', raw['natural'], selection['natural'])
            for target in ('margin', 'vector'):
                original = interaction(raw['natural'], labels, target, matrix)
                patched = interaction(raw['patched'], labels, target, matrix)
                n = np.mean(original ** 2, axis=(-2, -1))
                r = np.mean(patched[:, :, 0] ** 2, axis=(-2, -1))
                t = np.mean((patched[:, :, 1] - original[:, None]) ** 2, axis=(-2, -1))
                metrics = np.load(folder / f'metrics_{target}.npz', allow_pickle=False)
                for key, expected in (('natural_mse', n), ('restore_mse', r), ('transplant_error_mse', t)):
                    compare('metrics_' + key, metrics[key], expected); checks += int(expected.size)
                factors = metrics['factors']
                for row in (x for x in rows if x['seed'] == seed and x['alpha'] == alpha and x['target'] == target):
                    mask = np.ones(72, dtype=bool) if row['stratum'] == 'all' else labels == row['value'] if row['stratum'] == 'class' else factors[:, 2 if row['stratum'] == 'scale' else 3] == row['value']
                    j = next(j for j, spec in enumerate(specifications) if spec['method'] == row['method'])
                    denominator = float(np.sqrt(n[mask].mean()))
                    if denominator > 1e-9:
                        compare('summary_ratios', [row['restore_ratio'], row['transplant_error_ratio']],
                                [np.sqrt(r[mask, j].mean()) / denominator, np.sqrt(t[mask, j].mean()) / denominator])
                    else: assert row['restore_ratio'] is None and row['transplant_error_ratio'] is None
        print(f'audit seed {seed} PASS', flush=True)
    old_manifest = teacher.read(run.OLD / 'delivery_manifest.json')
    for entry in old_manifest['files']:
        assert teacher.sha(run.BASE / entry['path']) == entry['sha256'], entry['path']
    value = dict(status='PASS', utc=teacher.now(), panel=panel, teachers=6, cases=24,
                 numeric_errors=ERRORS, checked_values=COUNTS, metric_values_checked=checks,
                 old_delivery_files_unchanged=len(old_manifest['files']), old_manifest_sha256=teacher.sha(run.OLD / 'delivery_manifest.json'),
                 freeze_order_checked=True, cv_folds_recomputed=True, seconds=time.perf_counter() - started)
    teacher.dump(run.RUN / 'independent_audit.json', value)
    print(value, flush=True)


if __name__ == '__main__':
    teacher.configure(); audit()
