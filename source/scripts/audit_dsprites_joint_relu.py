"""独立审计：原网络32个单通道forward、原mid/finish开关干预、自动微分。

不调用新核心的阈值、选点、展开卷积或干预函数。全部案例都用真实
单通道卷积重建仿射项，避免两个汇总共享同一辅助函数却误称独立验证。
"""
from __future__ import annotations
import json
import numpy as np
import torch
import dsprites_teacher as teacher
from run_dsprites_joint_relu import BASE, RUN, SHAPE, load, verify_seal, verify_folder
from test_dsprites_residual_routes import original_holds

ERRORS = {}
EPS = 1e-8
MASKS = (3, 15)


def compare(name, actual, expected):
    actual = np.asarray(actual); expected = np.asarray(expected)
    assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
    error = float(np.max(abs(actual.astype(float)-expected.astype(float)))) if actual.size else 0.
    assert error < 1e-9, (name, error)
    ERRORS[name] = max(ERRORS.get(name, 0.), error)


def minimum(base, changes, directions):
    """独立的排序上界；输入分项来自32个真实forward，而非卷积展开。"""
    base = np.atleast_1d(base); changes = np.asarray(changes).reshape(-1, 32)
    directions = np.atleast_1d(directions)
    sorted_values = np.sort(changes*directions[:, None], axis=1)[:, ::-1]
    margins = directions[:, None]*base[:, None] + np.cumsum(sorted_values, axis=1)
    result = np.zeros(len(base), dtype=int)
    before = -directions*base; after = directions*base
    for i in range(len(base)):
        if directions[i]*base[i] >= -EPS: continue
        crossed = np.flatnonzero(margins[i] > EPS)
        if not len(crossed): continue
        j = int(crossed[0]); previous = directions[i]*base[i] if j == 0 else margins[i, j-1]
        before[i] = -previous; after[i] = margins[i, j]
        if previous < -EPS and directions[i]*base[i] < -EPS: result[i] = j+1
    return result, before, after


@torch.no_grad()
def independent_census(model, a, h):
    b = model.block3
    za = b.bn1(b.conv1(a)); zh = b.bn1(b.conv1(h))
    fields = a.expand(32, -1, -1, -1).clone()
    for c in range(32): fields[c, c] = h[0, c]
    solos = b.bn1(b.conv1(fields)).flatten(1).cpu().numpy().T
    aa = za.flatten().cpu().numpy(); hh = zh.flatten().cpu().numpy()
    changes = solos - aa[:, None]
    compare('affine_original_solo_sum', aa+changes.sum(1), hh)
    flips = (aa > 0) != (hh > 0)
    robust = flips & (abs(aa) > EPS) & (abs(hh) > EPS)
    ids = np.flatnonzero(robust); directions = np.where(hh[ids] > 0, 1, -1)
    k, _, _ = minimum(aa[ids], changes[ids], directions)
    kmap = np.zeros(32768, dtype=int); kmap[ids] = k
    pool = kmap >= 2
    counts = np.array([32768, flips.sum(), robust.sum(), (k == 0).sum(),
                       (kmap == 1).sum(), pool.sum(), (pool & (hh > 0)).sum(), (pool & (hh < 0)).sum()])
    return za, zh, solos, changes, kmap, counts


def audit():
    assert not (RUN/'independent_audit.json').exists(); protocol = verify_seal()
    device = teacher.configure(); panel = load(SHAPE/'panel/data.npz'); endpoint_count = 0; summaries = []
    for seed in teacher.SEEDS:
        model = teacher.load_teacher(seed, device)
        for p in model.parameters(): p.requires_grad_(False)
        with torch.no_grad():
            images = torch.as_tensor(panel['images'][:, None], dtype=torch.float64, device=device)
            sources = torch.cat([model.block1(model.stem(v)) for v in images.split(16)])
        half = load(RUN/f'discovery/seed_{seed}/selection.npz')
        for alpha in (.5, 1.):
            folder = RUN/f'evaluation/seed_{seed}/alpha_{alpha}'; verify_folder(folder)
            raw = load(folder/'data.npz'); sel = load(folder/'selection.npz')
            for key in sel: assert np.array_equal(sel[key], half[key]), key
            for case, (recipient, donor) in enumerate(panel['pairs']):
                with torch.no_grad():
                    # 独立按四个源组构建16状态；A由原空/单组场直接相加。
                    inputs = []
                    for mask in range(16):
                        value = sources[recipient:recipient+1].clone()
                        for g in range(4):
                            if mask & (1 << g):
                                sl = slice(g*4, (g+1)*4)
                                value[:, sl] += alpha*(sources[donor:donor+1, sl]-sources[recipient:recipient+1, sl])
                        inputs.append(value)
                    middle = model.block2(torch.cat(inputs))
                weight = np.zeros(3); weight[panel['labels'][donor]] = 1.; weight[panel['labels'][recipient]] = -1.
                w = torch.as_tensor(weight, device=device, dtype=torch.float64)
                b = model.block3
                for m, mask in enumerate(MASKS):
                    h = middle[mask:mask+1]
                    a = middle[:1].clone()
                    for g in range(4):
                        if mask & (1 << g): a += middle[(1 << g):(1 << g)+1]-middle[:1]
                    za, zh, solos, changes, kmap, counts = independent_census(model, a, h)
                    compare('census_counts', raw['counts'][case, m], counts)
                    compare('census_histogram', raw['histogram'][case, m], np.bincount(kmap[kmap > 0], minlength=33)[1:33])
                    ga = za > 0; ua = b.mid(a); va = b.bn2(b.conv2(ua))+a; gb = va > 0
                    uh = b.mid(h); vh = b.bn2(b.conv2(uh))+h
                    with torch.no_grad():
                        natural = model.readout(b(h).mean((2, 3)))[0].cpu().numpy()
                        baseline = model.readout(b(a).mean((2, 3)))[0].cpu().numpy()
                    compare('natural_original', raw['natural'][case, m], natural)
                    compare('baseline_original', raw['baseline'][case, m], baseline)
                    # 自动微分直接得到第一激活到输出的导数，与新核心的转置卷积不同。
                    with torch.enable_grad():
                        leaf = ua.detach().clone().requires_grad_(True)
                        f = model.readout(b.finish(leaf, a).mean((2, 3))) @ w
                        p = torch.autograd.grad(f.sum(), leaf)[0].detach()
                    first = (torch.relu(zh)-ga*zh)*p
                    if alpha == .5:
                        pool = np.flatnonzero(kmap >= 2); score = abs(first.flatten().cpu().numpy())
                        score[score < 1e-12] = 0.
                        ordered = pool[np.lexsort((pool, -score[pool]))]; budget = min(32, len(pool))
                        rng = np.random.default_rng(seed+20261017000+case*97+mask)
                        lists = [ordered[:budget], rng.permutation(pool)[:budget], rng.permutation(pool)[:budget]]
                        lists = np.array([np.pad(ids, (0, 32-len(ids)), constant_values=-1) for ids in lists])
                        compare('half_pool', half['pool'][case, m], kmap >= 2)
                        compare('half_gate_ranking', half['lists'][case, m], lists)
                        point = int(lists[0, 0]); channels = np.full(32, -1, dtype=int)
                        k = 0; direction = 0
                        if point >= 0:
                            k = int(kmap[point]); direction = 1 if float(zh.flatten()[point]) > 0 else -1
                            channel_order = np.lexsort((np.arange(32), -direction*changes[point]))
                            channels[:k] = channel_order[:k]
                        compare('half_channel_ranking', half['channels'][case, m], channels)
                        compare('half_minimum', half['half_k'][case, m], np.array(k))
                        compare('half_direction', half['half_direction'][case, m], np.array(direction))
                        compare('half_budget', half['budget'][case, m], np.array(budget))
                    lists = sel['lists'][case, m]; point = int(lists[0, 0])
                    pool = np.flatnonzero(sel['pool'][case, m])
                    compare('pool_recurrence', raw['pool_stats'][case, m],
                            np.array([len(pool), (kmap[pool] > 0).sum(), (kmap[pool] >= 2).sum()]))
                    groups = [ids[:1] for ids in lists]+list(lists)+[pool, np.arange(32768)]
                    q = (w @ model.readout.weight)[None, :, None, None]/1024
                    for j, ids in enumerate(groups):
                        ids = ids[ids >= 0]
                        reference = original_holds(model, h, ga, gb, ids, None).detach().cpu().numpy()[0]
                        compare('held_original_forward', raw['held'][case, m, j], reference)
                        compare('direct_autograd', raw['direct'][case, m, j], np.array(float(first.flatten()[ids].sum())))
                        with torch.no_grad():
                            changed_u = uh.clone(); changed_u.flatten(1)[:, ids] = torch.where(ga.flatten(1)[:, ids], zh.flatten(1)[:, ids], 0.)
                            changed_v = b.bn2(b.conv2(changed_u))+h
                            response = ((torch.relu(vh)-gb*vh-torch.relu(changed_v)+gb*changed_v)*q).sum()
                        compare('response_original', raw['response'][case, m, j], np.array(float(response)))
                        compare('net_direct_response', np.array((natural-reference)@weight),
                                np.array(float(first.flatten()[ids].sum()+response)))
                    for j, ids in enumerate(lists):
                        answer = np.zeros(32, dtype=int); valid = ids >= 0; answer[valid] = kmap[ids[valid]]
                        compare('frozen_list_minimum', raw['list_k'][case, m, j], answer)
                    with torch.no_grad():
                        channels = sel['channels'][case, m]; channels = channels[channels >= 0]
                        fields = torch.cat([a.clone() for _ in range(4)])
                        if len(channels):
                            fields[1, channels[:-1]] = h[0, channels[:-1]]
                            fields[2, channels] = h[0, channels]
                        fields[3] = h[0]
                        logits = model.readout(b(fields).mean((2, 3))).cpu().numpy()
                        held = logits.copy(); zz = np.zeros(4); solo = np.zeros(32); changes_point = np.zeros(32); meta = np.zeros(11)
                        if point >= 0:
                            held = original_holds(model, fields, ga, gb, np.array([point]), None).cpu().numpy()
                            zz = b.bn1(b.conv1(fields)).flatten(1)[:, point].cpu().numpy()
                            solo = solos[point]; changes_point = changes[point]
                            aa = float(za.flatten()[point]); hh = float(zh.flatten()[point]); direction = int(sel['half_direction'][case, m])
                            fixed = -direction*aa > EPS and direction*zz[2] > EPS
                            endpoint = -direction*aa > EPS and direction*hh > EPS
                            _, before, after = minimum([aa], changes_point[None], [direction])
                            meta = np.array([aa, hh, direction, kmap[point], fixed,
                                             endpoint and kmap[point] >= 2, fixed and np.max(direction*solo) < -EPS,
                                             before[0], after[0], len(channels), 0.])
                    compare('patch_original_forward', raw['patch_logits'][case, m], logits)
                    compare('patch_original_hold', raw['patch_held'][case, m], held)
                    compare('patch_original_preactivation', raw['patch_z'][case, m], zz)
                    compare('solo_original_preactivation', raw['solo_z'][case, m], solo)
                    compare('channel_affine_terms', raw['point_change'][case, m], changes_point)
                    compare('point_stability_meta', raw['point_meta'][case, m], meta)
                    endpoint_count += 1
                if (case+1) % 24 == 0: print(json.dumps(dict(seed=seed, alpha=alpha, cases=case+1)), flush=True)
            summaries.append(dict(seed=seed, alpha=alpha, endpoints=144))
    count = 0
    for name, row in teacher.read(RUN/'protocol.json')['preservation'].items():
        path = BASE/name; assert teacher.sha(path) == row['sha256'] == path.with_suffix('.sha256').read_text().strip()
        records = teacher.read(path)['files']; assert len(records) == row['files']
        for item in records:
            p = BASE/item['path']; assert p.stat().st_size == item['bytes'] and teacher.sha(p) == item['sha256'], item['path']; count += 1
    assert count == 1286; teacher.verify_old()
    result = dict(status='PASS', utc=teacher.now(), scope='all12frames;all1728endpoints;original32solo firstfields andall8holds,all4patches,halfselections',
                  endpoints=endpoint_count, original_solo_fields=endpoint_count*32,
                  held_interventions=endpoint_count*8, patch_conditions=endpoint_count*4,
                  maximum_errors=ERRORS, frames=summaries, old_delivery_files=count)
    teacher.dump(RUN/'independent_audit.json', result)
    print(json.dumps(dict(status='PASS', maximum_error=max(ERRORS.values()), endpoints=endpoint_count, old_entries=count)), flush=True)


if __name__ == '__main__': audit()
