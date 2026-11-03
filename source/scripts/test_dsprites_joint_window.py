"""冻结协议前验证联合必要性逻辑、固定参照不变性和局部/原整图一致。"""
from __future__ import annotations
import itertools
import json
import numpy as np
import torch
import dsprites_teacher as teacher
import joint_window_mechanism as core
from run_dsprites_joint_window import BASE, PRIOR, SHAPE, load, sources_for, two_traces


def known():
    """三通道反例说明“每个单独不足”不等于原组每个成员必要。"""
    values = np.array([.5, .5, .5]); b = -.9
    assert max(b+values) < 0 and b+values.sum() > 0
    assert max(b+values.sum()-values) > 0
    for b, d in ((-.6, np.array([.4, .5])), (-.9, values), (-.7, np.array([.6, .3, -.1]))):
        all_proper = [b+d[list(ids)].sum() for k in range(len(d)) for ids in itertools.combinations(range(len(d)), k)]
        certificate = min(-b, b+d.sum(), -max(b+d.sum()-d)) > 0
        assert certificate == (b < 0 and b+d.sum() > 0 and max(all_proper) < 0)
    h = torch.arange(7*32*3*3, dtype=torch.float64).reshape(1, 7, 32, 3, 3)/10000
    a, _ = core.rebuild(h, 3); c = dict(mask=3, C=np.array([0, 1]), w=torch.ones((32, 3, 3), dtype=torch.float64), bias=torch.tensor(-1.), a1=a[0], direction=1)
    before = core.geometry(c, h); changed = h.clone(); changed[:, 1, 0, 1, 1] += .2
    after = core.geometry(c, changed); assert float(abs(before[:, 1:]-after[:, 1:]).max()) == 0.
    return dict(single_channel_insufficiency_not_exactnecessity=True, fixedreference_invariant_to_anchor_only_change=True)


@torch.no_grad()
def preflight():
    """复用case0，不做新搜索；边界点仅用于数值实现测试而非机制定位。"""
    path = BASE/'reports/dsprites_joint_window_preflight_20261018.json'; assert not path.exists()
    device = teacher.configure(); panel = load(SHAPE/'panel/data.npz'); rows = []
    for seed in teacher.SEEDS:
        model = teacher.load_teacher(seed, device)
        for p in model.parameters(): p.requires_grad_(False)
        source = sources_for(model, panel, device); recipient, donor = panel['pairs'][0]
        half, full = two_traces(model, source, recipient, donor)
        old = load(PRIOR/f'discovery/seed_{seed}/selection.npz'); frozen = load(PRIOR/f'discovery/seed_{seed}/frozen_prior.npz'); maximum = 0.
        for m, mask in enumerate((3, 15)):
            selection = {k: v[0, m] for k, v in old.items()}; fixed = {k: v[0, m] for k, v in frozen.items()}
            pp = core.paths(selection); c = core.cache(model, half, full, int(fixed['lists'][0, 0]), mask, fixed)
            for family in range(4):
                ids = pp[family*12]['ids']; amounts = np.array([0., .375, 1.])
                actual, _ = core.full_fields(model, half, full, [(family, ids, float(v)) for v in amounts])
                expected = core.crop(actual, c['y'], c['x'], 1)
                maximum = max(maximum, float(abs(core.local_fields(c, family, ids, amounts)-expected).max()))
            for point in (0, 32767):
                edge = core.cache(model, half, full, point, mask, fixed); ids = pp[0]['ids']
                actual, _ = core.full_fields(model, half, full, [(0, ids, .5)])
                maximum = max(maximum, float(abs(core.local_fields(edge, 0, ids, [.5])-core.crop(actual, edge['y'], edge['x'], 1)).max()))
        assert maximum < 1e-9; rows.append(dict(seed=seed, local_original_maximum_error=maximum)); print(json.dumps(rows[-1]), flush=True)
    teacher.dump(path, dict(status='PASS', utc=teacher.now(), known=known(), teachers=rows))


if __name__ == '__main__': preflight()
