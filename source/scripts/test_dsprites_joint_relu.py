"""选择前的已知答案和原网络核验；只用上一轮已指定的第一个案例。"""
from __future__ import annotations
import itertools
import json
import numpy as np
import torch
import dsprites_teacher as teacher
import joint_relu_mechanism as core
from residual_route_mechanism import trace
from run_dsprites_joint_relu import BASE, load
from test_dsprites_residual_routes import original_holds


def known():
    # 六个通道共同关闭，任何五个都不行；再用全部64子集核验排序界。
    changes = np.zeros(32); changes[:6] = -1.1
    answer = core.minimum_order([6.3], changes[None], [-1])
    assert answer['k'][0] == 6
    assert all(6.3+sum(changes[list(ids)]) > 0 for k in range(6) for ids in itertools.combinations(range(6), k))
    assert 6.3+changes[:6].sum() < 0
    opening = core.minimum_order([-6.3], -changes[None], [1]); assert opening['k'][0] == 6
    one = changes.copy(); one[6] = -7
    assert core.minimum_order([6.3], one[None], [-1])['k'][0] == 1
    # 第一个通道正好落在零点，严格数值隔离带不假称最小基数为2。
    edge = np.zeros(32); edge[:2] = [-1., -.2]
    assert not core.minimum_order([1.], edge[None], [-1])['valid'][0]
    return dict(minimum_six=6, opening_and_closing=True, exact_boundary_excluded=True)


def preflight():
    path = BASE/'reports/dsprites_joint_relu_preflight_20261017.json'; assert not path.exists()
    device = teacher.configure(); rows = []
    for seed in teacher.SEEDS:
        model = teacher.load_teacher(seed, device)
        for p in model.parameters(): p.requires_grad_(False)
        error = 0.
        # 旧交付仅保存第一教师案例场；其他教师仍用同两幅旧图重建前缀。
        panel = load(BASE/'runs/dsprites_shape_donor_20261010/panel/data.npz')
        from layer_transfer_mechanism import design, lower_fields
        from shape_donor_mechanism import replace_groups
        d = design()
        with torch.no_grad():
            x = torch.as_tensor(panel['images'][:2, None], device=device, dtype=torch.float64)
            src = model.block1(model.stem(x))
            h = model.block2(replace_groups(src[:1], src[1:], torch.as_tensor(d['bits'], device=device), .5))
            a = lower_fields(h, d)[1][0]
        w = np.array([-1., 1., 0.])
        for mask in core.MASKS:
            t = trace(model, a[mask:mask+1], h[mask:mask+1], w); pop = core.census(model, t)
            chosen = core.select(t, pop, seed+mask); out = core.measure(model, a[mask:mask+1], h[mask:mask+1], w, pop, chosen)
            lists = chosen['lists']; groups = [x[:1] for x in lists] + list(lists) + [np.flatnonzero(chosen['pool']), np.arange(32768)]
            for j, ids in enumerate(groups):
                ids = ids[ids >= 0]
                ref = original_holds(model, h[mask:mask+1], t['ga'], t['gb'], ids, None).detach().cpu().numpy()[0]
                error = max(error, float(np.max(abs(out['held'][j]-ref))))
            assert error < 1e-9
        rows.append(dict(seed=seed, error=error)); print(json.dumps(rows[-1]), flush=True)
    teacher.dump(path, dict(status='PASS', utc=teacher.now(), known=known(), teachers=rows))


if __name__ == '__main__': preflight()
