"""定位前检查：原网络锚点、单开关精确分数、联动端点及两层恒等式。"""
from __future__ import annotations
import json
import numpy as np
import torch
import dsprites_teacher as teacher
import upstream_rescue_mechanism as core
from upstream_rescue_measure import measure
from run_dsprites_upstream_rescue import BASE, PRIOR, JOINT, SHAPE, load, sources_for, two_traces


@torch.no_grad()
def preflight():
    """使用每模型既有case0；例子和最终排名不由该检查结果决定。"""
    path = BASE/'reports/dsprites_upstream_rescue_preflight_20261017.json'; assert not path.exists()
    device = teacher.configure(); panel = load(SHAPE/'panel/data.npz'); records = []
    for seed in teacher.SEEDS:
        model = teacher.load_teacher(seed, device)
        for p in model.parameters(): p.requires_grad_(False)
        source = sources_for(model, panel, device); recipient, donor = panel['pairs'][0]
        half, full = two_traces(model, source, recipient, donor); prior = load(PRIOR/f'evaluation/seed_{seed}/data.npz')
        frozen = load(JOINT/f'discovery/seed_{seed}/selection.npz'); maximum = 0.
        assert float(abs(model.block2(half['x'])-half['h']).max()) < 1e-12
        assert float(abs(model.block2(full['x'])-full['h']).max()) < 1e-12
        for m, mask in enumerate(core.MASKS):
            old = {k: v[0, m] for k, v in frozen.items()}; chosen = core.discover(model, half, full, mask, old, seed+20261017000+mask)
            wj = core.target_weight(model, int(old['lists'][0, 0])); rows = core.anchors(mask); s = int(old['half_direction'])
            for stage in range(2):
                key = f'gate{stage+1}'; ids = chosen[key+'_ids']; score = chosen[key+'_scores']; pool = chosen[key+'_pool']
                eligible = np.flatnonzero(pool); check_ids = eligible[:3].tolist()
                if len(eligible): check_ids += [int(eligible[np.argmax(score[eligible])])]
                for i in check_ids:
                    point = ids[i]; b = model.block2
                    if stage == 0:
                        u = full['u'][rows].clone(); u.flatten(1)[:, point] = ((half['z1'][rows] > 0)*full['z1'][rows]).flatten(1)[:, point]
                        h = b.finish(u, full['skip'][rows])
                    else:
                        h = full['h'][rows].clone(); h.flatten(1)[:, point] = ((half['z2'][rows] > 0)*full['z2'][rows]).flatten(1)[:, point]
                    actual = s*float(((h-full['h'][rows])*wj).sum()); maximum = max(maximum, abs(actual-score[i]))
            a, r = core.intervene(model, half, full, mask, chosen); _, r1 = core.reconstruct(full['h'], mask)
            weight = np.zeros(3); weight[panel['labels'][donor]] = 1.; weight[panel['labels'][recipient]] = -1.
            out = measure(model, a, r, r1, weight, old)
            maximum = max(maximum, float(abs(out['natural'][0, 0]-prior['natural'][0, m, :, 3]).max()))
            for n in (5, 10):
                maximum = max(maximum, float(abs(out['natural'][n, 0]-prior['natural'][0, m, :, 1]).max()),
                              float(abs(out['natural'][n, 1]-prior['natural'][0, m, :, 0]).max()))
        assert maximum < 1e-9; records.append(dict(seed=seed, maximum_error=maximum)); print(json.dumps(records[-1]), flush=True)
    teacher.dump(path, dict(status='PASS', utc=teacher.now(), teachers=records, new_images=0, original_endpoints=True))


if __name__ == '__main__': preflight()
