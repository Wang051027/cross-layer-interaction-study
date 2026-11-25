"""在同一组源干预上分开改变中间通道预算、尾部读出和组外门控。"""
import numpy as np
import torch
import dsprites_triple_rule as rule

BUDGETS = (0, 4, 8, 16, 32)
MASKS = tuple(range(15))  # 所有低阶角和四个三组角，不需要四组全换。


def third(x):
    """轴 0 为掩码编号；对完整的八角逐项求三阶差分。"""
    rows = []
    for m in rule.TRIPLES:
        z, a, b, c, ab, ac, bc, abc = rule.corners(m)
        rows.append(x[abc]-x[ab]-x[ac]-x[bc]+x[a]+x[b]+x[c]-x[z])
    return torch.stack(rows)


def task(logits, r, d):
    """始终使用供体减受体 logit，不对不同输出目标挑选结果。"""
    return logits[..., d] - logits[..., r]


def tail(model, h, r, d):
    return task(rule.tail_from_middle(model, h), r, d)


def values(t):
    return t.detach().cpu().numpy()


def project(jac, field):
    return (field * jac).sum((1, 2, 3))


def mask_from(field, ids):
    mask = field.new_zeros(1, field.shape[1], 1, 1)
    if ids:
        mask[:, ids] = 1
    return mask


def forecast(model, x, donor, r, d, strength, fit, order):
    """只有 11 个不同低阶状态可见，逐个三组只使用其七个子集。"""
    first = fit['first']
    low = rule.low_states(model, x, donor, strength, first)
    drive = rule.propagated(model, low.first_residual, first, list(range(32)))
    full = (low.quad_v + drive).relu() - low.quad_h
    j3 = rule.empty_jacobian(model, low.base_h, list(range(32)))
    jac = j3[d] - j3[r]
    qf = task(low.quad_logits, r, d)
    stats = {'class_mean':task(full.new_tensor(fit['class_logits'][r]),r,d)}
    for b in BUDGETS:
        corr = full * mask_from(full, order[:b])
        endpoint = low.quad_h + corr
        stats[f'fixed_{b}'] = project(jac, corr)
        stats[f'live_{b}'] = tail(model, endpoint, r, d) - qf
        stats[f'clip_{b}'] = tail(model, endpoint.clamp_min(0), r, d) - qf
        stats[f'negative_{b}'] = (endpoint < -1e-9).double().mean((1, 2, 3))
    return stats, (low, full, jac)


def gated(model, x, r, d, first=None, second=None):
    """仅在被指定的 ReLU 上把 max(z,0) 替换为 D_ref*z；旁路不变。"""
    b = model.block3
    u = b.bn1(b.conv1(x))
    a = u.relu() if first is None else u * first
    v = b.bn2(b.conv2(a)) + b.shortcut(x)
    h = v.relu() if second is None else v * second
    return task(model.readout(h.mean((2,3))),r,d), (u > 0, v > 0)


def tangent(model, delta, gates, r, d):
    """固定两层开关后的精确方向导数，BN 平移和读出偏置在差分中消去。"""
    b = model.block3
    assert isinstance(b.shortcut, torch.nn.Identity)
    def slope(bn):
        return (bn.weight / (bn.running_var + bn.eps).sqrt())[None, :, None, None]
    u = b.conv1(delta) * slope(b.bn1) * gates[0]
    v = (b.conv2(u) * slope(b.bn2) + delta) * gates[1]
    return (v.mean((2,3)) * (model.readout.weight[d]-model.readout.weight[r])).sum(1)


def measure(model, x, donor, r, d, strength, fit, order, ctx, pred):
    """只有本函数读取真实三组状态。所有门控作用都在同一批角上比较。"""
    low, full, jac = ctx
    h = model.block2(rule.mixed_sources(x, donor, MASKS, strength))
    base = h[:1]
    c = (h-base)*mask_from(h,fit['middle'])
    o = h-base-c
    f0, g0 = gated(model,base,r,d)
    fc, gc = gated(model,base+c,r,d)
    fo, go = gated(model,base+o,r,d)
    f, _ = gated(model,h,r,d)
    gh, y = third(h), third(f)
    keep, outside, removed = third(fc), third(fo), third(f-fo)
    gap = removed-keep
    jc, jo = project(jac,third(c)), project(jac,third(o))
    stats = {'y':y,'keep':keep,'removed':removed,'outside':outside,'gap':gap,
             'true_norm_sq':gh.square().sum((1,2,3)),
             'error_est_C':jc-pred['fixed_4'],'error_omit_O':jo,
             'error_tail_C':keep-jc,'error_tail_O':outside-jo,'error_mixed':gap}
    for b in BUDGETS:
        mask = mask_from(h,order[:b])
        stats[f'middle_error_sq_{b}'] = (gh-full*mask).square().sum((1,2,3))
        stats[f'oracle_fixed_{b}'] = project(jac,gh*mask)
        stats[f'oracle_live_{b}'] = tail(model,low.quad_h+gh*mask,r,d)-task(low.quad_logits,r,d)
    # 仅改变组外触发的门控：CO 的参照是 C，O 的参照是 00。
    # 两种参照不同，因此即使两层都移植，剩余量也未被代数强制为零。
    for key, use1, use2 in [('first',1,0),('second',0,1),('both',1,1)]:
        on,_=gated(model,h,r,d,gc[0] if use1 else None,gc[1] if use2 else None)
        off,_=gated(model,base+o,r,d,g0[0] if use1 else None,g0[1] if use2 else None)
        stats['gap_lock_'+key]=third(on-off)-keep
    # 单独让组外变化，测它怎样改变同一 C 位移的局部读出斜率。
    t0 = tangent(model,c,g0,r,d)
    to = tangent(model,c,go,r,d)
    stats['slope_empty'],stats['slope_outside'] = third(t0),third(to)
    stats['slope_shift'] = third(to-t0)
    for k in range(2):
        stats[f'flip{k+1}_fraction'] = (go[k][list(rule.TRIPLES)] != g0[k]).double().mean((1,2,3))
    # 独立对照：原生 block3、自动微分、移植相同掩码的有限前向。
    a=gated(model,h,r,d,*g0)[0]
    b=gated(model,base+o,r,d,*g0)[0]
    cc=gated(model,base+c,r,d,*g0)[0]
    z=gated(model,base,r,d,*g0)[0]
    ao=gated(model,h,r,d,*go)[0]
    bo=gated(model,base+o,r,d,*go)[0]
    err=sum(stats[k] for k in ('error_est_C','error_omit_O','error_tail_C','error_tail_O','error_mixed'))
    checks={'native_tail':f-tail(model,h,r,d),'empty_field':base-low.base_h,
            'true_field':gh-(h[list(rule.TRIPLES)]-low.quad_h),
            'own_gates':fo-bo,'constant_gate_mix':a-b-cc+z,
            'autograd_empty_tangent':t0-project(jac,c),'outside_tangent':to-(ao-bo),
            'oracle_native_32':stats['oracle_live_32']-y,'error_identity':err-(y-pred['fixed_4'])}
    return stats,{name:float(t.abs().max()) for name,t in checks.items()}
