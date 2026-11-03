"""协议前的已知四格答案和原模型核验；不寻找新的图像或坐标。"""
from __future__ import annotations
import json
import numpy as np
import torch
import dsprites_teacher as teacher
import ar_factorial_mechanism as core
from run_dsprites_ar_factorial import BASE, PRIOR, SHAPE, load, middle_fields
from test_dsprites_residual_routes import original_holds


def known():
    """第一预激活无AR交互，但ReLU后四格可以出现交互。"""
    values = np.array([.5, 2., 0., 1.])
    terms = core.factorial(values)
    assert np.max(abs(terms-[-.5, 1.5, -.5])) < 1e-12 and abs(terms.sum()-(values[3]-values[0])) < 1e-12
    z = np.array([.2, .05, .05, -.1]); assert abs(core.factorial(z)[2]) < 1e-12
    assert np.all(z[:3] > 0) and z[3] < 0
    return dict(factorial_terms=terms.tolist(), affine_no_interaction_but_joint_failure=True)


def preflight():
    path = BASE/'reports/dsprites_ar_factorial_preflight_20261017.json'; assert not path.exists()
    device = teacher.configure(); panel = load(SHAPE/'panel/data.npz'); results = []
    for seed in teacher.SEEDS:
        model = teacher.load_teacher(seed, device)
        for p in model.parameters(): p.requires_grad_(False)
        with torch.no_grad():
            images = torch.as_tensor(panel['images'][:, None], device=device, dtype=torch.float64)
            sources = torch.cat([model.block1(model.stem(v)) for v in images.split(16)])
        r, donor = map(int, panel['pairs'][0]); a, residual = middle_fields(model, sources, r, donor, device)
        selection = load(PRIOR/f'discovery/seed_{seed}/selection.npz'); error = 0.
        w = np.zeros(3); w[panel['labels'][donor]] = 1.; w[panel['labels'][r]] = -1.
        for m in range(2):
            chosen = {k: v[0, m] for k, v in selection.items()}; out = core.compute(model, a[:, m], residual[:, m], w, chosen)
            groups = core.gate_groups(chosen); C = chosen['channels']; C = C[C >= 0]
            for context in range(2):
                for cell, (i, j) in enumerate(core.CELLS):
                    x = a[i:i+1, m].clone()
                    if context == 0: x[:, C] += residual[j:j+1, m][:, C]
                    else: x += residual[j:j+1, m]
                    with torch.no_grad():
                        ga = model.block3.bn1(model.block3.conv1(a[i:i+1, m])) > 0
                        for k, ids in enumerate(groups):
                            original = original_holds(model, x, ga, ga, ids, None).cpu().numpy()[0]
                            error = max(error, float(np.max(abs(original-out['held'][context, cell, k]))))
        assert error < 1e-9; results.append(dict(seed=seed, original_hold_error=error)); print(json.dumps(results[-1]), flush=True)
    teacher.dump(path, dict(status='PASS', utc=teacher.now(), known=known(), teachers=results))


if __name__ == '__main__': preflight()
