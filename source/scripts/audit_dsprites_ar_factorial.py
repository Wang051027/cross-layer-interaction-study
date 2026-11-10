"""用原模型前向运算重建 A/R 四个状态，以 32 个单通道卷积和自动微分核对阈值及传递量。"""
from __future__ import annotations
import json
import numpy as np
import torch
import dsprites_teacher as teacher
from run_dsprites_ar_factorial import BASE, RUN, PRIOR, SHAPE, load, verify_protocol, verify_folder
from test_dsprites_residual_routes import original_holds

ERRORS = {}
EPS = 1e-8


def check(name, value, reference, tolerance=1e-9):
    """记录各项数值检查的最大误差。"""
    value = np.asarray(value); reference = np.asarray(reference)
    assert value.shape == reference.shape, (name, value.shape, reference.shape)
    error = float(np.max(abs(value.astype(float)-reference.astype(float)))) if value.size else 0.
    assert error < tolerance, (name, error)
    ERRORS[name] = max(ERRORS.get(name, 0.), error)


def minimum(base, changes, direction):
    """从原单通道forward得到的分项独立检查最小基数及严格余量。"""
    gap = -direction*base
    before, after, k = gap, -gap, 0
    ordered = np.sort(direction*changes)[::-1]
    margins = direction*base + np.cumsum(ordered)
    crossed = np.flatnonzero(margins > EPS)
    if gap > EPS and len(crossed):
        pos = int(crossed[0]); previous = direction*base if pos == 0 else margins[pos-1]
        before, after = -previous, margins[pos]
        if previous < -EPS: k = pos+1
    return k, before, after


def audit():
    """每次只保留一个案例的场；全部6912个反事实端点均实际重算。"""
    assert not (RUN/'independent_audit.json').exists(); protocol = verify_protocol()
    device = teacher.configure(); panel = load(SHAPE/'panel/data.npz'); completed = 0
    a_index = torch.as_tensor([0, 0, 1, 1, 0, 0, 1, 1], device=device)
    for seed in teacher.SEEDS:
        model = teacher.load_teacher(seed, device)
        for p in model.parameters(): p.requires_grad_(False)
        folder = RUN/f'evaluation/seed_{seed}'; verify_folder(folder); raw = load(folder/'data.npz')
        selection = load(folder/'selection.npz'); old_selection = load(PRIOR/f'discovery/seed_{seed}/selection.npz')
        assert all(np.array_equal(v, old_selection[k]) for k, v in selection.items())
        with torch.no_grad():
            images = torch.as_tensor(panel['images'][:, None], device=device, dtype=torch.float64)
            sources = torch.cat([model.block1(model.stem(v)) for v in images.split(16)])
        for case, (recipient, donor) in enumerate(panel['pairs']):
            fields = []
            with torch.no_grad():
                for alpha in (.5, 1.):
                    joined = []
                    for mask in range(16):
                        value = sources[recipient:recipient+1].clone()
                        for group in range(4):
                            if mask & (1 << group):
                                sl = slice(group*4, (group+1)*4)
                                value[:, sl] += alpha*(sources[donor:donor+1, sl]-sources[recipient:recipient+1, sl])
                        joined.append(value)
                    fields.append(model.block2(torch.cat(joined)))
            w = torch.zeros(3, device=device, dtype=torch.float64)
            w[int(panel['labels'][donor])] = 1.; w[int(panel['labels'][recipient])] = -1.
            weight = w.cpu().numpy(); b = model.block3
            for m, mask in enumerate((3, 15)):
                bases = []; residuals = []
                for middle in fields:
                    a = middle[:1].clone()
                    for g in range(4):
                        if mask & (1 << g): a += middle[(1 << g):(1 << g)+1]-middle[:1]
                    bases.append(a); residuals.append(middle[mask:mask+1]-a)
                a = torch.cat(bases); r = torch.cat(residuals)
                point = int(selection['lists'][case, m, 0, 0]); C = selection['channels'][case, m]; C = C[C >= 0]
                ids3 = selection['lists'][case, m]
                groups = [ids[ids >= 0] for ids in [v[:1] for v in ids3]+list(ids3)]
                groups += [np.flatnonzero(selection['pool'][case, m]), np.arange(32768)]
                direction = int(selection['half_direction'][case, m])
                with torch.no_grad():
                    x = []
                    for restricted in (True, False):
                        for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)):
                            value = a[i:i+1].clone()
                            if restricted: value[:, C] += r[j:j+1, C]
                            else: value += r[j:j+1]
                            x.append(value)
                    x = torch.cat(x)
                    baseline_logits = model.readout(b(a).mean((2, 3)))
                    natural = model.readout(b(x).mean((2, 3)))
                    za = b.bn1(b.conv1(a)); ga = za > 0; ua = b.mid(a)
                    va = b.bn2(b.conv2(ua))+a; gb = va > 0; ea = b.finish(ua, a)
                    z = b.bn1(b.conv1(x)); u = b.mid(x); v = b.bn2(b.conv2(u))+x; e = b.finish(u, x)
                    q = (w @ model.readout.weight)[None, :, None, None]/1024
                # 同时对Ai和其第一激活自动微分，独立得到JA和pA。
                with torch.enable_grad():
                    leaf = a.detach().clone().requires_grad_(True); first_leaf = b.mid(leaf)
                    task = model.readout(b.finish(first_leaf, leaf).mean((2, 3))) @ w
                    ja, pa = torch.autograd.grad(task.sum(), (leaf, first_leaf))
                first_term = (u-ga[a_index]*z)*pa.detach()[a_index]
                second_term = (e-gb[a_index]*v)*q
                linear = ja.detach()[a_index]*(x-a[a_index])
                parts = torch.stack([value.sum((2, 3)) for value in (linear, first_term, second_term)], 1)
                channel_d = ((e-ea[a_index])*q).sum((2, 3))
                check('original_baseline', raw['baseline'][case, m], baseline_logits.cpu().numpy())
                check('original_natural', raw['natural'][case, m], natural.cpu().numpy().reshape(2, 4, 3))
                check('autograd_parts', raw['parts'][case, m], parts.detach().cpu().numpy().reshape(2, 4, 3, 32))
                check('final_channel_d', raw['d_channel'][case, m], channel_d.cpu().numpy().reshape(2, 4, 32))
                flip = torch.stack([((z > 0) != ga[a_index]).sum((1, 2, 3)), ((v > 0) != gb[a_index]).sum((1, 2, 3))], 1)
                check('natural_flip_counts', raw['flips'][case, m], flip.cpu().numpy().reshape(2, 4, 2))
                held_logits = []
                with torch.no_grad():
                    for g, ids in enumerate(groups):
                        reference = original_holds(model, x, ga[a_index], gb[a_index], ids, None)
                        held_logits.append(reference)
                        ref = reference.cpu().numpy().reshape(2, 4, 3)
                        check('original_local_hold', raw['held'][case, m, :, :, g], ref)
                        changed_u = u.clone(); changed_u.flatten(1)[:, ids] = torch.where(ga[a_index].flatten(1)[:, ids], z.flatten(1)[:, ids], 0.)
                        changed_v = b.bn2(b.conv2(changed_u))+x; changed_e = b.finish(changed_u, x)
                        direct = first_term.flatten(1)[:, ids].sum(1)
                        response = ((e-gb[a_index]*v-torch.relu(changed_v)+gb[a_index]*changed_v)*q).sum((1, 2, 3))
                        channel_net = ((e-changed_e)*q).sum((2, 3))
                        closed = ((v > 0) & ~(changed_v > 0)).sum((1, 2, 3))
                        opened = (~(v > 0) & (changed_v > 0)).sum((1, 2, 3))
                        check('direct_autograd', raw['direct'][case, m, :, :, g], direct.cpu().numpy().reshape(2, 4))
                        check('response_original', raw['response'][case, m, :, :, g], response.cpu().numpy().reshape(2, 4))
                        check('local_net_identity', ((natural-reference)@w).cpu().numpy(), (direct+response).cpu().numpy())
                        check('net_output_channels', raw['channel_net'][case, m, :, :, g], channel_net.cpu().numpy().reshape(2, 4, 32))
                        check('held_second_flip_counts', raw['second_flips'][case, m, :, :, g], torch.stack((closed, opened), 1).cpu().numpy().reshape(2, 4, 2))
                        common = original_holds(model, x, ga[:1], gb[:1], ids, None)
                        common_base = original_holds(model, a, ga[:1], gb[:1], ids, None)
                        check('common_reference_original', raw['common_held'][case, m, :, :, g], common.cpu().numpy().reshape(2, 4, 3))
                        check('common_baseline_original', raw['common_baseline_held'][case, m, :, g], common_base.cpu().numpy())
                    probe_u = u.clone(); probe_u.flatten(1)[:, point] += 1.
                    probe_logits = model.readout(b.finish(probe_u, x).mean((2, 3)))
                    check('matched_amplitude_probe', raw['probe'][case, m], probe_logits.cpu().numpy().reshape(2, 4, 3))
                    remainder = (u-ga[a_index]*z).flatten(1)[:, point]
                    effect = ((natural-held_logits[0])@w)
                    valid = remainder > EPS
                    gain = torch.where(valid, effect/remainder.clamp_min(EPS), 0.)
                    check('point_remainder', raw['point_remainder'][case, m], remainder.cpu().numpy().reshape(2, 4))
                    check('point_secant_gain', raw['point_gain'][case, m], gain.cpu().numpy().reshape(2, 4))
                    check('point_gain_validity', raw['gain_valid'][case, m], valid.cpu().numpy().reshape(2, 4))
                    # 用A0为基准的64个真实单通道forward重建两套残差分项。
                    solos = a[:1].expand(64, -1, -1, -1).clone()
                    for j in range(2):
                        for c in range(32): solos[j*32+c, c] += r[j, c]
                    solo_z = b.bn1(b.conv1(solos)).flatten(1)[:, point].cpu().numpy().reshape(2, 32)
                    base_z = za.flatten(1)[:, point].cpu().numpy(); incoming = solo_z-base_z[0]
                    check('actual_solo_incoming', raw['incoming'][case, m], incoming)
                    check('base_preactivation', raw['base_z'][case, m], base_z)
                    zc = z[:4].flatten(1)[:, point].cpu().numpy(); zz = z[4:].flatten(1)[:, point].cpu().numpy()
                    check('group_preactivation', raw['group_z'][case, m], zc)
                    check('all_preactivation', raw['all_z'][case, m], zz)
                truth = []
                for cell, (i, j) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                    same = -direction*base_z[i] > EPS; cross = same and direction*zc[cell] > EPS
                    safe = max(direction*(base_z[i]+incoming[j])) < -EPS
                    k, before, after = minimum(base_z[i], incoming[j], direction)
                    truth.append([same, cross, safe, cross and safe, k, before, after, same and direction*zz[cell] > EPS])
                    check('affine_group_prediction', np.array(zc[cell]), np.array(base_z[i]+incoming[j, C].sum()))
                check('threshold_states', raw['threshold'][case, m], np.asarray(truth))
                completed += 8
            if (case+1) % 24 == 0: print(json.dumps(dict(seed=seed, cases=case+1)), flush=True)
    count = 0
    for name, item in protocol['preservation'].items():
        p = BASE/name; assert teacher.sha(p) == item['sha256'] == p.with_suffix('.sha256').read_text().strip()
        records = teacher.read(p)['files']; assert len(records) == item['files']
        for row in records:
            file = BASE/row['path']; assert file.stat().st_size == row['bytes'] and teacher.sha(file) == row['sha256'], row['path']; count += 1
    assert count == 1394; teacher.verify_old()
    result = dict(status='PASS', utc=teacher.now(), cases=864, endpoint_fields=completed,
                  local_holds=55296, common_holds=55296, common_baseline_holds=13824,
                  original_solo_fields=55296, matched_probes=6912,
                  maximum_errors=ERRORS, old_delivery_files=count)
    teacher.dump(RUN/'independent_audit.json', result)
    print(json.dumps(dict(status='PASS', maximum_error=max(ERRORS.values()), endpoints=completed, old_entries=count)), flush=True)


if __name__ == '__main__': audit()
