"""独立检查上游修复：原forward注入、裁剪卷积候选分数及自动微分。

不调用新核心的源状态、重建、排名、权重、干预或输出分解函数。
全部既有案例和22方法均检查；第一开关单候选另用5×5原卷积计算，
区别于生产的边权展开。审计失败时不生成完成记录。
"""
from __future__ import annotations
import json
import numpy as np
import torch
from torch.nn import functional as F
import dsprites_teacher as teacher
from run_dsprites_upstream_rescue import BASE, RUN, PRIOR, JOINT, SHAPE, load, verify_protocol, verify_selection, verify_folder

ERRORS = {}; EPS = 1e-8
NAMES = ('natural', 'src_selected4', 'src_random0', 'src_random1', 'src_selected1', 'src_all',
         'mid_selected4', 'mid_random0', 'mid_random1', 'mid_selected1', 'mid_all',
         'g1_selected16', 'g1_random0', 'g1_random1', 'g1_selected1', 'g1_all',
         'g2_selected16', 'g2_random0', 'g2_random1', 'g2_selected1', 'g2_all', 'g12_all')


def check(name, x, ref, tolerance=1e-9):
    """保存最大误差；对布尔量和形状同样严格检查。"""
    if torch.is_tensor(x): x = x.detach().cpu().numpy()
    if torch.is_tensor(ref): ref = ref.detach().cpu().numpy()
    x = np.asarray(x); ref = np.asarray(ref); assert x.shape == ref.shape, (name, x.shape, ref.shape)
    error = float(abs(x.astype(float)-ref.astype(float)).max()) if x.size else 0.
    assert error < tolerance, (name, error)
    ERRORS[name] = max(ERRORS.get(name, 0.), error)


def rebuild(h, mask):
    """直接按定义相加单组锚点，不使用生产的索引和重建函数。"""
    a = h[..., 0, :, :, :].clone()
    for g in range(4):
        if mask & (1 << g): a += h[..., g+1, :, :, :]-h[..., 0, :, :, :]
    target = 5 if mask == 3 else 6
    return a, h[..., target, :, :, :]-a


def own_ids(selection, method):
    """按方法名解析，独立于生产method_spec的整数算式。"""
    if method == 'natural': return '', np.array([], dtype=int)
    if method == 'g12_all': return 'both', np.arange(32768)
    prefix, variant = method.split('_')
    if variant == 'all':
        return prefix, np.flatnonzero(selection['source_pool']) if prefix == 'src' else np.arange(32 if prefix == 'mid' else 32768)
    lists = selection['source_lists'] if prefix == 'src' else selection['middle_lists'] if prefix == 'mid' else selection['gate_lists'][int(prefix[1])-1]
    index = {'selected4': 0, 'selected16': 0, 'selected1': 0, 'random0': 1, 'random1': 2}[variant]
    ids = lists[index]; ids = ids[:1] if variant == 'selected1' else ids
    return prefix, ids[ids >= 0]


@torch.no_grad()
def capture_original(block, values, point=None, reference=None):
    """从原forward的BN钩子读取预激活，避免为了审计重复运行卷积。

可在原mid后注入同一个下游点；finish始终是原模型的真实运算。
钩子只保存张量，不改变BN的输出，finally恢复全部临时方法。
"""
    result = {}; original_mid, original_finish = block.mid, block.finish
    def save1(module, args, output): result['z1'] = output
    def save2(module, args, output): result['branch2'] = output
    hooks = [block.bn1.register_forward_hook(save1), block.bn2.register_forward_hook(save2)]
    def mid(x):
        value = original_mid(x)
        if point is not None:
            value = value.clone(); value.flatten(1)[:, point] = torch.where(reference.flatten(1)[:, point], result['z1'].flatten(1)[:, point], 0.)
        result['u'] = value; return value
    def finish(u, skip):
        value = original_finish(u, skip); result['z2'] = result['branch2']+skip; return value
    try:
        block.mid, block.finish = mid, finish; result['h'] = block(values)
    finally:
        block.mid, block.finish = original_mid, original_finish
        for hook in hooks: hook.remove()
    return result


@torch.no_grad()
def original_upstream(model, x0, x1, h0, z10, z20, selection):
    """原block2.forward在mid/finish处注入；用finally恢复原方法。"""
    b = model.block2; original_mid, original_finish = b.mid, b.finish; captured = {}
    def save1(module, args, output): captured['z1'] = output
    def save2(module, args, output): captured['z2'] = output
    hooks = [b.bn1.register_forward_hook(save1), b.bn2.register_forward_hook(save2)]
    x = x1.repeat(22, 1, 1, 1)
    specs = [own_ids(selection, name) for name in NAMES]
    for n, (kind, ids) in enumerate(specs):
        if kind == 'src':
            view = x[n*7:(n+1)*7]; view[:, ids] = x0[:, ids]
    def mid(value):
        out = original_mid(value); z = captured['z1']
        for n, (kind, ids) in enumerate(specs):
            if kind in ('g1', 'both'):
                out[n*7:(n+1)*7].flatten(1)[:, ids] = torch.where((z10 > 0).flatten(1)[:, ids], z[n*7:(n+1)*7].flatten(1)[:, ids], 0.)
        return out
    def finish(value, skip):
        out = original_finish(value, skip); z = captured['z2']+skip
        for n, (kind, ids) in enumerate(specs):
            view = out[n*7:(n+1)*7]
            if kind in ('g2', 'both'): view.flatten(1)[:, ids] = torch.where((z20 > 0).flatten(1)[:, ids], z[n*7:(n+1)*7].flatten(1)[:, ids], 0.)
            if kind == 'mid': view[:, ids] = h0[:, ids]
        return out
    try:
        b.mid, b.finish = mid, finish
        result = b(x).reshape(22, 7, 32, 32, 32)
    finally:
        b.mid, b.finish = original_mid, original_finish
        for hook in hooks: hook.remove()
    return result


@torch.no_grad()
def candidate_audit(model, x0, x1, h0, h1, z10, z11, z20, z21, selection, mask, point, w, direction, random_seed):
    """验证全部有效候选和确定性排名；不读取输出或救回结果。"""
    rows = [g+1 for g in range(4) if mask & (1 << g)]; b = model.block2
    _, y, x = np.unravel_index(point, (32, 32, 32)); source_scores = np.zeros(16)
    source_ids = np.flatnonzero(selection['source_pool']); source_fields = []
    for c in source_ids:
        g = c//4+1; value = x1[g:g+1].clone(); value[:, c] = x0[g:g+1, c]; source_fields.append(value)
    source_out = b(torch.cat(source_fields))
    source_scores[source_ids] = (direction*((source_out-h1[source_ids//4+1])*w).sum((1, 2, 3))).cpu().numpy()
    a0, _ = rebuild(h0, mask); a1, _ = rebuild(h1, mask)
    middle_scores = (direction*(a0-a1)*w[0]).sum((1, 2)).cpu().numpy()
    check('all_source_scores', selection['source_scores'], source_scores); check('all_middle_scores', selection['middle_scores'], middle_scores)
    u1 = b.mid(x1); rem1 = u1-(z10 > 0)*z11; rem2 = h1-(z20 > 0)*z21
    anchor_rem1 = rem1[rows].flatten(1)
    expected = [(np.arange(16), source_scores, selection['source_pool'], 4, 'source_lists'),
                (np.arange(32), middle_scores, np.ones(32, bool), 4, 'middle_lists')]
    for stage, remainder in enumerate((rem1, rem2)):
        radius = 2 if stage == 0 else 1
        ids = np.array([c*1024+yy*32+xx for c in range(32) for yy in range(max(0, y-radius), min(32, y+radius+1))
                        for xx in range(max(0, x-radius), min(32, x+radius+1))])
        pool = abs(remainder[rows].flatten(1)[:, ids]).max(0).values.cpu().numpy() > EPS
        values = np.zeros(len(ids))
        if stage == 0:
            # 原conv2+BN2在5×5裁剪上无padding得到3×3；图外输出W严格为零。
            crop = F.pad(u1[rows], (2, 2, 2, 2))[:, :, y:y+5, x:x+5]
            skip = F.pad(b.shortcut(x1)[rows], (1, 1, 1, 1))[:, :, y:y+3, x:x+3]
            weight = F.pad(w, (1, 1, 1, 1))[:, :, y:y+3, x:x+3]
            original = F.relu(b.bn2(F.conv2d(crop, b.conv2.weight))+skip)
            for start in range(0, len(ids), 256):
                current = ids[start:start+256]; p = len(current); n = len(rows)
                patched = crop[None].expand(p, n, 32, 5, 5).clone()
                coordinate = torch.as_tensor(current, device=x1.device)
                ci, yy, xx = coordinate//1024, coordinate % 1024//32, coordinate % 32
                pi = torch.arange(p, device=x1.device)[:, None]; ai = torch.arange(n, device=x1.device)[None]
                patched[pi, ai, ci[:, None], (yy-y+2)[:, None], (xx-x+2)[:, None]] -= anchor_rem1[:, coordinate].T
                result = F.relu(b.bn2(F.conv2d(patched.reshape(p*n, 32, 5, 5), b.conv2.weight))+skip.repeat(p, 1, 1, 1)).reshape(p, n, 32, 3, 3)
                values[start:start+p] = (direction*((result-original[None])*weight[None]).sum((1, 2, 3, 4))).cpu().numpy()
        else:
            # 第二开关的实际激活替换；W来自独立自动微分。
            values = (-direction*rem2[rows].flatten(1)[:, ids].sum(0)*w.flatten()[ids]).cpu().numpy()
        key = f'gate{stage+1}'; size = 800 if stage == 0 else 288
        check(key+'_ids', selection[key+'_ids'], np.pad(ids, (0, size-len(ids)), constant_values=-1))
        check(key+'_pool', selection[key+'_pool'], np.pad(pool, (0, size-len(ids))))
        check(key+'_all_scores', selection[key+'_scores'], np.pad(values, (0, size-len(ids))))
        expected.append((ids, values, pool, 16, stage))
    rng = np.random.default_rng(random_seed); counts = []
    for ids, scores, pool, budget, key in expected:
        good = pool & (scores > EPS); selected = ids[good][np.lexsort((ids[good], -scores[good]))][:budget]
        width = 16 if key == 'source_lists' or isinstance(key, int) else 32
        n = len(selected); controls = [selected, rng.permutation(ids[pool])[:n], rng.permutation(ids[pool])[:n]]
        result = np.array([np.pad(v, (0, width-len(v)), constant_values=-1) for v in controls])
        ref = selection['gate_lists'][key] if isinstance(key, int) else selection[key]
        check('baseline_only_ranking_'+str(key), ref, result)
        counts.append([n, int(pool.sum()), int(good.sum())])
    check('selection_counts', selection['selection_counts'], counts)
    return sum(int(x[2].sum()) for x in expected[2:])


def audit():
    """端点、源层分解、名单和下游输出分别使用独立路径检查。"""
    assert not (RUN/'independent_audit.json').exists(); protocol = verify_protocol(); verify_selection()
    device = teacher.configure(); panel = load(SHAPE/'panel/data.npz'); count = 0; candidate_count = 0
    for seed in teacher.SEEDS:
        model = teacher.load_teacher(seed, device)
        for p in model.parameters(): p.requires_grad_(False)
        with torch.no_grad():
            images = torch.as_tensor(panel['images'][:, None], device=device, dtype=torch.float64)
            source = torch.cat([model.block1(model.stem(v)) for v in images.split(16)])
        df = RUN/f'discovery/seed_{seed}'; ef = RUN/f'evaluation/seed_{seed}'; verify_folder(df); verify_folder(ef)
        sel = load(df/'selection.npz'); frozen = load(df/'frozen_prior.npz'); raw = load(ef/'data.npz')
        old = load(JOINT/f'discovery/seed_{seed}/selection.npz'); assert all(np.array_equal(v, old[k]) for k, v in frozen.items())
        for case, (recipient, donor) in enumerate(panel['pairs']):
            fields = []
            with torch.no_grad():
                for alpha in (.5, 1.):
                    values = []
                    for mask in (0, 1, 2, 4, 8, 3, 15):
                        value = source[recipient:recipient+1].clone()
                        for g in range(4):
                            if mask & (1 << g): value[:, g*4:g*4+4] += alpha*(source[donor:donor+1, g*4:g*4+4]-source[recipient:recipient+1, g*4:g*4+4])
                        values.append(value)
                    fields.append(torch.cat(values))
                x0, x1 = fields; b = model.block2; h0, h1 = b(x0), b(x1)
                z10, z11 = b.bn1(b.conv1(x0)), b.bn1(b.conv1(x1))
                z20 = b.bn2(b.conv2(b.mid(x0)))+b.shortcut(x0); z21 = b.bn2(b.conv2(b.mid(x1)))+b.shortcut(x1)
            for m, mask in enumerate((3, 15)):
                chosen = {k: v[case, m] for k, v in sel.items()}; fixed = {k: v[case, m] for k, v in frozen.items()}
                point = int(fixed['lists'][0, 0]); direction = int(fixed['half_direction']); rows = [g+1 for g in range(4) if mask & (1 << g)]
                a0, _ = rebuild(h0, mask); a1, r1 = rebuild(h1, mask)
                # W从block3原第一预激活求导，JA/p从block2原forward求导。
                with torch.enable_grad():
                    leaf = a0[None].detach().requires_grad_(True)
                    scalar = model.block3.bn1(model.block3.conv1(leaf)).flatten()[point]
                    w = torch.autograd.grad(scalar, leaf)[0].detach()
                    up = x0[1:5].detach().requires_grad_(True); mid = b.mid(up); out = b.finish(mid, b.shortcut(up))
                    ja, pa = torch.autograd.grad((out*w).sum(), (up, mid))
                linear = (ja*(x1[1:5]-x0[1:5])).sum((2, 3))
                first = ((b.mid(x1)[1:5]-(z10[1:5] > 0)*z11[1:5])*pa).sum((2, 3))
                second = ((h1[1:5]-(z20[1:5] > 0)*z21[1:5])*w).sum((2, 3))
                delta = ((h1[1:5]-h0[1:5])*w).sum((1, 2, 3))
                check('autograd_source_linear', chosen['source_linear'], linear); check('autograd_gate_parts', chosen['gate_parts'], torch.stack((first, second)))
                check('anchor_delta', chosen['anchor_delta'], delta); check('anchor_identity', delta, linear.sum(1)+first.sum(1)+second.sum(1))
                fl = []
                for before, after in ((z10[1:5], z11[1:5]), (z20[1:5], z21[1:5])):
                    fl.append(torch.stack(((~(before > 0) & (after > 0)).sum((1, 2, 3)), ((before > 0) & ~(after > 0)).sum((1, 2, 3))), 1))
                check('upstream_flip_counts', chosen['upstream_flips'], torch.stack(fl))
                candidate_count += candidate_audit(model, x0, x1, h0, h1, z10, z11, z20, z21, chosen, mask, point, w, direction, seed+20261017000+case*97+mask)
                if seed == 202609291 and case == 1 and mask == 3:
                    example = dict(seed=seed, case=case, mask=mask, point=point, C=fixed['channels'][fixed['channels'] >= 0].tolist(), gates=[])
                    for stage, (before, after) in enumerate(((z10, z11), (z20, z21))):
                        for coordinate in chosen['gate_lists'][stage, 0]:
                            if coordinate < 0: continue
                            records = []
                            for row in rows:
                                z0 = float(before[row].flatten()[coordinate]); z1 = float(after[row].flatten()[coordinate])
                                rem = max(z1, 0.)-(z0 > 0)*z1
                                multiplier = float(pa[row-1].flatten()[coordinate]) if stage == 0 else float(w.flatten()[coordinate])
                                records.append(dict(source_group=row-1, half_z=z0, full_z=z1, half_open=z0 > 0, full_open=z1 > 0,
                                                    remainder=rem, propagated_contribution=rem*multiplier))
                            target_row = 5 if mask == 3 else 6
                            tz0 = float(before[target_row].flatten()[coordinate]); tz1 = float(after[target_row].flatten()[coordinate])
                            target = dict(target_half_z=tz0, target_full_z=tz1, target_half_open=tz0 > 0, target_full_open=tz1 > 0,
                                          target_remainder=max(tz1, 0.)-(tz0 > 0)*tz1)
                            example['gates'].append(dict(stage=stage+1, id=int(coordinate), coordinate=list(map(int, np.unravel_index(coordinate, (32, 32, 32)))), anchors=records, target=target))
                    teacher.dump(RUN/'planned_example_upstream.json', example)
                with torch.no_grad():
                    changed = original_upstream(model, x0, x1, h0, z10, z20, chosen); ai, ri = rebuild(changed, mask)
                    base_trace = capture_original(model.block3, ai)
                    baselog = model.readout(base_trace['h'].mean((2, 3))); za = base_trace['z1']
                    check('original_baseline', raw['baseline'][case, m], baselog); check('original_base_z', raw['base_z'][case, m], za.flatten(1)[:, point])
                    check('discovery_base_z', chosen['base_z'], model.block3.bn1(model.block3.conv1(torch.stack((a0, a1)))).flatten(1)[:, point])
                    C = fixed['channels']; C = C[C >= 0]
                    changes = []; bases = []
                    for n in range(22):
                        for residue in (r1, ri[n]):
                            restricted = torch.zeros_like(residue); restricted[C] = residue[C]
                            changes.extend((ai[n]+restricted, ai[n]+residue)); bases.extend((n, n))
                    x = torch.stack(changes); base_index = torch.tensor(bases, device=device); a = ai[base_index]
                    b3 = model.block3; natural_trace = capture_original(b3, x)
                    e = natural_trace['h']; z = natural_trace['z1']; u = natural_trace['u']; v = natural_trace['z2']
                    natural = model.readout(e.mean((2, 3)))
                    zbase = za[base_index]; ga = zbase > 0
                    vbase = base_trace['z2'][base_index]; gb = vbase > 0; ebase = base_trace['h'][base_index]
                    held_trace = capture_original(b3, x, point, ga); held = model.readout(held_trace['h'].mean((2, 3)))
                check('original_natural', raw['natural'][case, m], natural.reshape(22, 2, 2, 3))
                check('original_point_hold', raw['held'][case, m], held.reshape(22, 2, 2, 3))
                weight = torch.zeros(3, device=device, dtype=torch.float64); weight[panel['labels'][donor]] = 1.; weight[panel['labels'][recipient]] = -1.
                with torch.enable_grad():
                    leaf = ai.detach().requires_grad_(True); mid = b3.mid(leaf); task = model.readout(b3.finish(mid, leaf).mean((2, 3)))@weight
                    grad, p = torch.autograd.grad(task.sum(), (leaf, mid))
                q = (weight@model.readout.weight)[None, :, None, None]/1024
                direct_field = (u-ga*z)*p[base_index]; second_field = (e-gb*v)*q
                parts = torch.stack(((grad[base_index]*(x-a)).sum((2, 3)), direct_field.sum((2, 3)), second_field.sum((2, 3))), 1)
                newv = held_trace['z2']; newe = held_trace['h']
                direct = direct_field.flatten(1)[:, point]; response = (second_field-(newe-gb*newv)*q).sum((1, 2, 3))
                check('downstream_autograd_parts', raw['parts'][case, m], parts.reshape(22, 2, 2, 3, 32))
                check('point_direct', raw['direct'][case, m], direct.reshape(22, 2, 2)); check('point_response', raw['response'][case, m], response.reshape(22, 2, 2))
                check('point_net_identity', (natural-held)@weight, direct+response)
                check('point_channel_net', raw['channel_net'][case, m], ((e-newe)*q).sum((2, 3)).reshape(22, 2, 2, 32))
                check('output_channel_d', raw['d_channel'][case, m], ((e-ebase)*q).sum((2, 3)).reshape(22, 2, 2, 32))
                flips = torch.stack((((z > 0) != ga).sum((1, 2, 3)), ((v > 0) != gb).sum((1, 2, 3))), 1)
                flips2 = torch.stack((((v > 0) & ~(newv > 0)).sum((1, 2, 3)), (~(v > 0) & (newv > 0)).sum((1, 2, 3))), 1)
                check('downstream_flips', raw['flips'][case, m], flips.reshape(22, 2, 2, 2)); check('hold_second_flips', raw['second_flips'][case, m], flips2.reshape(22, 2, 2, 2))
                check('point_remainder', raw['point_remainder'][case, m], (u-ga*z).flatten(1)[:, point].reshape(22, 2, 2))
                incoming = torch.stack([torch.stack((((r1)*w[0]).sum((1, 2)), (ri[n]*w[0]).sum((1, 2)))) for n in range(22)])
                base = za.flatten(1)[:, point]; zc = base[:, None]+incoming[:, :, C].sum(2); zall = base[:, None]+incoming.sum(2)
                check('incoming_autograd_weight', raw['incoming'][case, m], incoming)
                check('group_z_actual', raw['group_z'][case, m], z.flatten(1)[:, point].reshape(22, 2, 2)[:, :, 0])
                check('all_z_actual', raw['all_z'][case, m], z.flatten(1)[:, point].reshape(22, 2, 2)[:, :, 1])
                same = -direction*base > EPS; cross = same[:, None] & (direction*zc > EPS)
                safe = (direction*(base[:, None, None]+incoming)).max(2).values < -EPS
                flags = torch.stack((same[:, None].expand(22, 2), cross, safe, cross & safe, same[:, None] & (direction*zall > EPS)), 2)
                check('threshold_flags', raw['threshold'][case, m], flags)
                count += 88
            if (case+1) % 24 == 0: print(json.dumps(dict(seed=seed, cases=case+1, fields=count, maximum_error=max(ERRORS.values()))), flush=True)
    old_entries = 0
    for name, record in protocol['preservation'].items():
        p = BASE/name; assert teacher.sha(p) == record['sha256'] == p.with_suffix('.sha256').read_text().strip()
        rows = teacher.read(p)['files']; assert len(rows) == record['files']
        for r in rows:
            p = BASE/r['path']; assert teacher.sha(p) == r['sha256'] and p.stat().st_size == r['bytes']; old_entries += 1
    assert old_entries == 1467; teacher.verify_old(); assert count == 76032
    result = dict(status='PASS', utc=teacher.now(), endpoint_fields=count, point_holds=count, baseline_fields=19008,
                  candidate_gate_pool_entries=candidate_count, source_score_entries=13824, middle_score_entries=27648,
                  case_states=864, old_delivery_files=old_entries, maximum_errors=ERRORS)
    teacher.dump(RUN/'independent_audit.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__': audit()
