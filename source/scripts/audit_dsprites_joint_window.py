"""独立审计有限搜索与真实恢复：原卷积/原forward、所有真子集及共同点控制。

不调用新核心的局部缓存、重建、几何、搜索、名单或完整干预函数。
生产用11×11线性缓存，本审计用13×13原conv/BN直接重算；输出再用
原block2/3.forward核验。任何失败都不写PASS完成记录。
"""
from __future__ import annotations
import json
import numpy as np
import torch
from torch.nn import functional as F
import dsprites_teacher as teacher
from run_dsprites_joint_window import BASE, RUN, PRIOR, SHAPE, load, verify_protocol, verify_selection, verify_folder

ERRORS = {}
GRID = np.linspace(0., 1., 33)
GUARD = 1e-6


def check(name, x, y, tolerance=1e-9):
    """布尔标志、轴和数值都比较；记录误差而非只比较汇总数。"""
    if torch.is_tensor(x): x = x.detach().cpu().numpy()
    if torch.is_tensor(y): y = y.detach().cpu().numpy()
    x, y = np.asarray(x), np.asarray(y); assert x.shape == y.shape, (name, x.shape, y.shape)
    error = float(abs(x.astype(float)-y.astype(float)).max()) if x.size else 0.
    assert error < tolerance, (name, error); ERRORS[name] = max(ERRORS.get(name, 0.), error)


def crop(value, y, x, radius):
    """独立裁剪只用于原模块输入和输出的支持；边界按模型零填充。"""
    return F.pad(value, (radius, radius, radius, radius))[..., y:y+2*radius+1, x:x+2*radius+1]


def rebuild(h, mask):
    """按A定义逐单组累加，未用核心的高级索引重建。"""
    a = h[..., 0, :, :, :].clone()
    for g in range(4):
        if mask & (1 << g): a += h[..., g+1, :, :, :]-h[..., 0, :, :, :]
    return a, h[..., 5 if mask == 3 else 6, :, :, :]-a


def own_paths(units):
    """按继承的物理名单重建48条前缀并独立处理重复实际预算。"""
    result = []
    for family, lists in enumerate((units['source_lists'], units['middle_lists'], units['gate_lists'][0], units['gate_lists'][1])):
        for variant in range(3):
            ids = lists[variant]; ids = ids[ids >= 0]; used = []
            for wanted in ((1, 2, 3, 4) if family < 2 else (1, 4, 8, 16)):
                count = min(len(ids), wanted); good = count > 0 and count not in used; used.append(count)
                result.append((family, variant, ids[:count], good))
    return result


@torch.no_grad()
def native_trace(block, x, point=None, reference=None):
    """原forward旁路读取BN；指定j时只在原mid后注入，后续原finish不变。"""
    result = {}; old_mid, old_finish = block.mid, block.finish
    def first(module, args, output): result['z1'] = output
    def second(module, args, output): result['branch2'] = output
    hooks = [block.bn1.register_forward_hook(first), block.bn2.register_forward_hook(second)]
    def mid(value):
        out = old_mid(value)
        if point is not None:
            out = out.clone(); out.flatten(1)[:, point] = torch.where(reference, result['z1'].flatten(1)[:, point], 0.)
        result['u'] = out; return out
    def finish(value, skip):
        out = old_finish(value, skip); result['z2'] = result['branch2']+skip; return out
    try: block.mid, block.finish = mid, finish; result['h'] = block(x)
    finally:
        block.mid, block.finish = old_mid, old_finish
        for hook in hooks: hook.remove()
    return result


@torch.no_grad()
def independent_local(model, xx, traces, y, x, family, ids, amounts):
    """13×13原conv1、带padding原conv2再裁剪；不使用生产的仿射差缓存。"""
    b = model.block2; half, full = traces; n = len(amounts)
    lam = torch.as_tensor(amounts, dtype=xx[0].dtype, device=xx[0].device)[:, None, None, None, None]
    u1, u0 = crop(full['u'], y, x, 2), crop(half['u'], y, x, 2)
    h1, h0 = crop(full['h'], y, x, 1), crop(half['h'], y, x, 1)
    yi, xi = torch.meshgrid(torch.arange(y-2, y+3, device=lam.device), torch.arange(x-2, x+3, device=lam.device), indexing='ij')
    valid = ((yi >= 0) & (yi < 32) & (xi >= 0) & (xi < 32))[None, None]
    if family == 0:
        x1, x0 = crop(xx[1], 2*y, 2*x, 6), crop(xx[0], 2*y, 2*x, 6)
        source = x1[None].expand(n, 7, 16, 13, 13).clone(); source[:, :, ids] += lam*(x0[:, ids]-x1[:, ids])[None]
        flat = source.reshape(n*7, 16, 13, 13)
        u = b.mid(flat)[:, :, 1:6, 1:6].reshape(n, 7, 32, 5, 5)*valid
        skip = b.shortcut(flat[:, :, 4:9, 4:9]).reshape(n, 7, 32, 3, 3)
    elif family == 2:
        u = u1[None].expand(n, 7, 32, 5, 5).clone()
        for coordinate in ids:
            ch, yy, xxx = np.unravel_index(int(coordinate), (32, 32, 32)); iy, ix = yy-y+2, xxx-x+2
            u[:, :, ch, iy, ix] += lam.flatten()[:, None]*(u0[:, ch, iy, ix]-u1[:, ch, iy, ix])[None]
        skip = crop(b.shortcut(xx[1]), y, x, 1)[None].expand(n, -1, -1, -1, -1)
    else:
        h = h1[None].expand(n, 7, 32, 3, 3).clone()
        if family == 1: h[:, :, ids] += lam*(h0[:, ids]-h1[:, ids])[None]
        else:
            for coordinate in ids:
                ch, yy, xxx = np.unravel_index(int(coordinate), (32, 32, 32)); iy, ix = yy-y+1, xxx-x+1
                h[:, :, ch, iy, ix] += lam.flatten()[:, None]*(h0[:, ch, iy, ix]-h1[:, ch, iy, ix])[None]
        return h
    branch = b.bn2(b.conv2(u.reshape(n*7, 32, 5, 5)))[:, :, 1:4, 1:4].reshape(n, 7, 32, 3, 3)
    return F.relu(branch+skip)*valid[..., 1:4, 1:4]


def context_fields(seven, a1, C, mask):
    """显式三个基准和全C场，独立于生产的逐通道预激活贡献。"""
    a, _ = rebuild(seven, mask); target = seven[:, 5 if mask == 3 else 6]
    fixed = a1.expand_as(a).clone(); actualbase = target.clone(); actualbase[:, C] = a1[C]
    bases = torch.stack((a, fixed, actualbase), 1)
    full = bases.clone(); full[:, :, C] = target[:, None, C]
    return bases, full


@torch.no_grad()
def direct_metrics(model, seven, a1, C, mask, point, direction, enumerate_subsets=False):
    """原block3卷积核实际计算基准/全C/每个删一；可枚举全部真子集。"""
    co = point//1024; b = model.block3; bases, full = context_fields(seven, a1, C, mask); n = len(seven)
    def z(fields):
        shape = fields.shape[:-3]
        value = F.conv2d(fields.reshape(-1, 32, 3, 3), b.conv1.weight[co:co+1])
        result = (value-b.bn1.running_mean[co])*b.bn1.weight[co]/torch.sqrt(b.bn1.running_var[co]+b.bn1.eps)+b.bn1.bias[co]
        return result.reshape(shape)
    base_z, full_z = z(bases), z(full)
    deleted = full[:, :, None].expand(n, 3, len(C), 32, 3, 3).clone()
    for k, ch in enumerate(C): deleted[:, :, k, ch] = bases[:, :, ch]
    drop_z = z(deleted)
    margins = torch.stack((-direction*base_z, direction*full_z, -(direction*drop_z).max(-1).values), -1)
    count = 0
    if enumerate_subsets:
        fields = []
        for bits in range((1 << len(C))-1):
            value = bases.clone()
            for k, ch in enumerate(C):
                if bits & (1 << k): value[:, :, ch] = full[:, :, ch]
            fields.append(value)
        all_z = z(torch.stack(fields, 2)); max_proper = (direction*all_z).max(-1).values
        explicit = (direction*full_z >= GUARD) & (max_proper <= -GUARD)
        certificate = (margins >= GUARD).all(-1)
        check('all_proper_subsets_vs_certificate', explicit, certificate)
        count = n*3*len(fields)
    return margins.cpu().numpy(), count


def objective(margins, strategy):
    """独立使用三参照最低余量，不调用新核心objectives。"""
    q = np.asarray(margins).min(-1)
    return q[..., 0] if strategy == 0 else q[..., 1:].min(-1) if strategy == 1 else q.min(-1)


def choose(records):
    """用Python元组排序独立复核预算/安全余量/强度的选择次序。"""
    records = [r for r in records if r[3] > 0.]
    good = [r for r in records if r[1] >= GUARD]
    return min(good, key=lambda r: (r[2], -r[1], r[3])) if good else min(records, key=lambda r: (-r[1], r[2], r[3]))


@torch.no_grad()
def original_fields(model, xx, traces, specs):
    """每个端点由原block2.forward在原mid后注入或原输出处替换，finally恢复。"""
    b = model.block2; half, full = traces; result = []
    for family, ids, amount in specs:
        if family < 0 or amount == 0.: value = full['h'].clone()
        elif family == 0:
            source = xx[1].clone(); source[:, ids] = (1-amount)*source[:, ids]+amount*xx[0][:, ids]; value = b(source)
        elif family == 2:
            old_mid = b.mid
            def mid(x):
                out = old_mid(x).clone(); out.flatten(1)[:, ids] = (1-amount)*out.flatten(1)[:, ids]+amount*half['u'].flatten(1)[:, ids]; return out
            try: b.mid = mid; value = b(xx[1])
            finally: b.mid = old_mid
        else:
            value = b(xx[1])
            if family == 1: value[:, ids] = (1-amount)*value[:, ids]+amount*half['h'][:, ids]
            else: value.flatten(1)[:, ids] = (1-amount)*value.flatten(1)[:, ids]+amount*half['h'].flatten(1)[:, ids]
        assert float(value.min()) >= -1e-12; result.append(value)
    return torch.stack(result)


def audit():
    """全部网格/细化/邻点及整图端点独立重算；最终再核对全部历史交付。"""
    assert not (RUN/'independent_audit.json').exists(); protocol = verify_protocol(); verify_selection()
    device = teacher.configure(); panel = load(SHAPE/'panel/data.npz')
    field_count = candidate_count = subset_count = drops_count = 0
    for seed in teacher.SEEDS:
        model = teacher.load_teacher(seed, device)
        for p in model.parameters(): p.requires_grad_(False)
        with torch.no_grad():
            images = torch.as_tensor(panel['images'][:, None], dtype=torch.float64, device=device)
            source = torch.cat([model.block1(model.stem(v)) for v in images.split(16)])
        sf = RUN/f'search/seed_{seed}'; ef = RUN/f'evaluation/seed_{seed}'
        verify_folder(sf); verify_folder(ef)
        search = load(sf/'data.npz'); units = load(sf/'units.npz'); frozen = load(sf/'frozen.npz'); raw = load(ef/'data.npz')
        for case, (recipient, donor) in enumerate(panel['pairs']):
            xx = []
            with torch.no_grad():
                for alpha in (.5, 1.):
                    values = []
                    for bits in (0, 1, 2, 4, 8, 3, 15):
                        x = source[recipient:recipient+1].clone()
                        for g in range(4):
                            if bits & (1 << g): x[:, g*4:g*4+4] += alpha*(source[donor:donor+1, g*4:g*4+4]-source[recipient:recipient+1, g*4:g*4+4])
                        values.append(x)
                    xx.append(torch.cat(values))
                traces = [native_trace(model.block2, value) for value in xx]
            weight = torch.zeros(3, device=device, dtype=torch.float64); weight[panel['labels'][donor]] = 1.; weight[panel['labels'][recipient]] = -1.
            for m, mask in enumerate((3, 15)):
                selected = {k: v[case, m] for k, v in units.items()}; fixed = {k: v[case, m] for k, v in frozen.items()}
                answer = {k: v[case, m] for k, v in search.items()}; pp = own_paths(selected)
                point = int(fixed['lists'][0, 0]); _, y, x = np.unravel_index(point, (32, 32, 32))
                C = fixed['channels']; C = C[C >= 0]; direction = int(fixed['half_direction'])
                a1, _ = rebuild(traces[1]['h'], mask); local_a1 = crop(a1, y, x, 1)
                for index, (family, variant, ids, valid) in enumerate(pp):
                    h = independent_local(model, xx, traces, y, x, family, ids, GRID)
                    margins, _ = direct_metrics(model, h, local_a1, C, mask, point, direction)
                    check('all_coarse_original_modules', answer['coarse_margins'][index], margins); candidate_count += len(GRID)
                specs = [(-1, np.array([], dtype=int), 0.)]
                for strategy in range(3):
                    for family in range(4):
                        for variant in range(3):
                            row = strategy*12+family*3+variant
                            eligible = [i for i, p in enumerate(pp) if p[0] == family and p[1] == variant and p[3]]
                            if eligible:
                                records = [(i, float(q), len(pp[i][2]), float(lam)) for i in eligible
                                           for q, lam in zip(objective(answer['coarse_margins'][i], strategy), GRID)]
                                pindex, _, budget, center = choose(records)
                                amounts = np.linspace(max(0., center-1/32), min(1., center+1/32), 17)
                                ids = pp[pindex][2]
                                h = independent_local(model, xx, traces, y, x, family, ids, amounts)
                                margins, _ = direct_metrics(model, h, local_a1, C, mask, point, direction)
                                check('all_refinement_original_modules', answer['refine_margins'][row], margins)
                                combined = np.concatenate((answer['coarse_margins'][pindex], answer['refine_margins'][row]))
                                all_amounts = np.concatenate((GRID, amounts)); scores = objective(combined, strategy)
                                index, _, _, amount = choose([(i, float(q), budget, float(lam)) for i, (q, lam) in enumerate(zip(scores, all_amounts))])
                                check('independent_selection', answer['choices'][row], [pindex, amount, True])
                                check('selected_geometry', answer['selected_margins'][row], combined[index])
                                specs.append((family, ids, amount))
                            else:
                                amount = 0.; ids = np.array([], dtype=int); check('empty_selection', answer['choices'][row], [-1, 0., False])
                                specs.append((-1, ids, amount)); amounts = np.zeros(17)
                                h = independent_local(model, xx, traces, y, x, family, ids, amounts)
                                margins, _ = direct_metrics(model, h, local_a1, C, mask, point, direction)
                                check('empty_refine', answer['refine_margins'][row], margins)
                            neighbors = np.clip([amount-1/256, amount, amount+1/256], 0., 1.)
                            h = independent_local(model, xx, traces, y, x, family, ids, neighbors)
                            margins, _ = direct_metrics(model, h, local_a1, C, mask, point, direction)
                            check('all_neighbor_checks', answer['neighbor_margins'][row], margins); candidate_count += 20
                specs += [(0, np.arange(16), 1.), (1, np.arange(32), 1.)]
                seven = original_fields(model, xx, traces, specs)
                a, r = rebuild(seven, mask); target = seven[:, 5 if mask == 3 else 6]
                margins, subsets = direct_metrics(model, crop(seven, y, x, 1), local_a1, C, mask, point, direction, True)
                check('selected_full_forward_geometry', raw['margins'][case, m], margins); subset_count += subsets
                check('search_vs_full_geometry', answer['selected_margins'], margins[1:37])
                moving = a.clone(); moving[:, C] = target[:, C]
                common = a1.expand_as(a).clone(); common[:, C] = target[:, C]
                noC = target.clone(); noC[:, C] = a1[C]
                fields = torch.stack((moving, common, target, noC), 1).reshape(-1, 32, 32, 32)
                natural = native_trace(model.block3, fields)
                with torch.no_grad():
                    baseline = model.readout(model.block3(a).mean((2, 3)))
                    actual0 = traces[1]['h'][5 if mask == 3 else 6]
                    task = model.readout(natural['h'].mean((2, 3)))
                    ref_z = model.block3.bn1(model.block3.conv1(a1[None])); reference = ref_z.flatten()[point] > 0
                    held = native_trace(model.block3, fields, point, reference)
                    held_task = model.readout(held['h'].mean((2, 3)))
                check('original_task_logits', raw['natural'][case, m], task.reshape(39, 4, 3))
                check('original_common_point_hold', raw['held'][case, m], held_task.reshape(39, 4, 3))
                check('original_baseline', raw['baseline'][case, m], baseline)
                check('point_z_original', raw['point_z'][case, m], natural['z1'].flatten(1)[:, point].reshape(39, 4))
                with torch.enable_grad():
                    leaf = a1[None].detach().requires_grad_(True); mid = model.block3.mid(leaf)
                    base_task = model.readout(model.block3.finish(mid, leaf).mean((2, 3)))@weight
                    pa = torch.autograd.grad(base_task.sum(), mid)[0].detach()
                q = (weight@model.readout.weight)[None, :, None, None]/1024
                with torch.no_grad():
                    ref_mid = model.block3.mid(a1[None]); ref_v = model.block3.bn2(model.block3.conv2(ref_mid))+a1[None]
                    g2 = ref_v > 0
                    direct = pa.flatten()[point]*(natural['u'].flatten(1)[:, point]-reference*natural['z1'].flatten(1)[:, point])
                    response = (((natural['h']-g2*natural['z2'])-(held['h']-g2*held['z2']))*q).sum((1, 2, 3))
                    channel_net = ((natural['h']-held['h'])*q).sum((2, 3))
                    actual_e = natural['h'].reshape(39, 4, 32, 32, 32)[:, 2]
                    delta_channel = ((actual_e-actual_e[:1])*q).sum((2, 3))
                    delta_h, delta_a = target-actual0, a-a1
                    _, r0 = rebuild(traces[1]['h'], mask); delta_r = r-r0
                    shifts = torch.stack((abs(delta_h).flatten(1).max(1).values, delta_h.square().mean((1, 2, 3)).sqrt(),
                                          abs(delta_a).flatten(1).max(1).values, abs(delta_r).flatten(1).max(1).values), 1)
                    check('actual_H_A_R_changes', raw['shifts'][case, m], shifts)
                    check('AR_identity_max', raw['identity'][case, m], abs(delta_h-delta_a-delta_r).flatten(1).max(1).values)
                    check('point_autograd_direct', raw['direct'][case, m], direct.reshape(39, 4))
                    check('point_second_response', raw['response'][case, m], response.reshape(39, 4))
                    check('point_channel_net', raw['point_channel_net'][case, m], channel_net.reshape(39, 4, 32))
                    check('final_channel_changes', raw['channel_change'][case, m], delta_channel)
                    check('point_identity', (task-held_task)@weight, direct+response)
                    zactual = natural['z1'].reshape(39, 4, 32, 32, 32)[:, 2]; vactual = natural['z2'].reshape(39, 4, 32, 32, 32)[:, 2]
                    flips = torch.stack((((zactual > 0) != (zactual[:1] > 0)).sum((1, 2, 3)), ((vactual > 0) != (vactual[:1] > 0)).sum((1, 2, 3))), 1)
                    check('actual_downstream_flips', raw['actual_flips'][case, m], flips)
                    check('actual_j_flip', raw['actual_point_flip'][case, m], (zactual.flatten(1)[:, point] > 0) != (zactual[:1].flatten(1)[:, point] > 0))
                    for eindex, endpoint in enumerate((0, 25, 28, 31, 34)):
                        drop = target[endpoint:endpoint+1].expand(len(C), -1, -1, -1).clone()
                        for j, ch in enumerate(C): drop[j, ch] = a1[ch]
                        logits = model.readout(model.block3(drop).mean((2, 3)))
                        dz = model.block3.bn1(model.block3.conv1(drop)).flatten(1)[:, point]
                        check('each_C_member_logits', raw['drop_logits'][case, m, eindex, :len(C)], logits)
                        check('each_C_member_preactivation', raw['drop_z'][case, m, eindex, :len(C)], dz)
                        drops_count += len(C)
                field_count += 156
            if (case+1) % 24 == 0: print(json.dumps(dict(seed=seed, cases=case+1, fields=field_count, maximum_error=max(ERRORS.values()))), flush=True)
    old_count = 0
    for name, record in protocol['preservation'].items():
        path = BASE/name; assert teacher.sha(path) == record['sha256'] == path.with_suffix('.sha256').read_text().strip()
        rows = teacher.read(path)['files']; assert len(rows) == record['files']
        for row in rows: assert teacher.sha(BASE/row['path']) == row['sha256'], row['path']; old_count += 1
    assert old_count == 1558 and field_count == 134784; teacher.verify_old()
    result = dict(status='PASS', utc=teacher.now(), endpoint_fields=field_count, point_holds=field_count,
        searched_geometry_fields=candidate_count, enumerated_proper_subset_fields=subset_count, original_C_member_drop_fields=drops_count,
        old_delivery_entries=old_count, maximum_errors=ERRORS)
    teacher.dump(RUN/'independent_audit.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__': audit()
