"""梯度变化接续实验：init → select → seal → evaluate → summarize。

六个冻结教师、四个旧源点、旧64候选不变。新36图只由像素去重及角度
覆盖选择；半强度选点封存后，才测其他强度。旧交付保持只读。
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import time
from pathlib import Path

import dsprites_teacher as teacher
import numpy as np
import torch
import dsprites_mechanism as mechanism
import gradient_mechanism as gradient

BASE, OLD, DATA = teacher.BASE, teacher.RUN, teacher.DATA
PRIOR = BASE / 'runs/dsprites_factor_transfer_20261007'
RUN = BASE / 'runs/dsprites_gradient_20261009'
FULL = BASE / 'data/dsprites'
SEEDS = teacher.SEEDS
STRENGTHS = (1., .25, .5, .75)
METHODS = ('class_fixed', 'input_gradient', 'fixed_linear', 'state_linear', 'input_finite',
           'random_0', 'random_1', 'random_2', 'random_3', 'all_shortlist', 'all_reachable')
CODE = ('scripts/gradient_mechanism.py', 'scripts/run_dsprites_gradient.py', 'scripts/test_dsprites_gradient.py')


def event(kind, **details):
    """将阶段事件追加到运行日志。"""
    value = dict(utc=teacher.now(), event=kind, **details)
    with (RUN / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')
    print(json.dumps(value, ensure_ascii=False), flush=True)


def write(path, value):
    """JSON强制有限数值，所有证据使用UTF-8。"""
    teacher.dump(path, value)


def complete(folder, started):
    """阶段标记在数值文件落盘后写入，不覆盖完成或未完成的原件。"""
    files = {p.name: teacher.sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
    write(folder / 'completion.json', dict(utc=teacher.now(), seconds=time.perf_counter()-started, files=files))


def verify_folder(folder):
    """只允许跳过哈希一致且完整结束的阶段。"""
    for name, expected in teacher.read(folder / 'completion.json')['files'].items():
        assert teacher.sha(folder / name) == expected, str(folder / name)


def verify_protocol():
    """输入及数值代码一旦冻结，后续阶段不得悄悄改动。"""
    protocol = teacher.read(RUN / 'protocol.json')
    assert teacher.sha(RUN / 'protocol.json') == (RUN / 'protocol.sha256').read_text().strip()
    for name, expected in protocol['inputs'].items():
        assert teacher.sha(BASE / name) == expected, name
    for name, expected in protocol['sources'].items():
        assert teacher.sha(BASE / name) == expected, name
        assert teacher.sha(RUN / 'source' / name) == expected, name
    verify_folder(RUN / 'panel')
    return protocol


def digest(pixel):
    """像素身份高于因素标签身份，防止旋转对称图像进入新面板。"""
    return hashlib.sha256(np.ascontiguousarray(pixel).tobytes()).hexdigest()


def prepare_panel():
    """不访问模型，选6大小×2共同角度×3形状；与全部先前像素去重。

    两个角度按到0和π的周期距离覆盖代价选取。先保留各大小共同合格
    角度，再枚举二角度面板；代价舍入12位，索引字典序打破并列。
    """
    folder = RUN / 'panel'; folder.mkdir(); started = time.perf_counter()
    previous = set()
    for split in ('train', 'validation', 'evaluation', 'translated'):
        pixels = np.load(DATA / f'{split}_images.npy', mmap_mode='r', allow_pickle=False)
        previous.update(digest(x) for x in pixels)
    with np.load(PRIOR / 'panel/data.npz', allow_pickle=False) as prior:
        previous.update(digest(x) for x in prior['images'])
    images = np.load(FULL / 'imgs.npy', mmap_mode='r', allow_pickle=False)
    latent = np.load(FULL / 'latents_classes.npy', mmap_mode='r', allow_pickle=False)
    physical = np.load(FULL / 'latents_values.npy', mmap_mode='r', allow_pickle=False)
    def ids(scale, angle):
        """官方因素笛卡尔顺序，固定中心位置索引15、15。"""
        return (((np.arange(3) * 6 + scale) * 40 + angle) * 32 + 15) * 32 + 15
    eligible = []
    for scale in range(6):
        eligible.append([a for a in range(40) if not {digest(images[i]) for i in ids(scale, a)} & previous])
    common = sorted(set.intersection(*(set(a) for a in eligible)))
    candidates = []
    for angles in itertools.combinations(common, 2):
        index = np.concatenate([ids(s, a) for s in range(6) for a in angles])
        if len({digest(images[i]) for i in index}) != 36:
            continue
        theta = np.asarray([physical[ids(0, a)[0], 3] for a in angles])
        cost = min(float(np.square(np.angle(np.exp(1j * (theta[list(order)] - [0., np.pi])))).sum())
                   for order in itertools.permutations(range(2)))
        candidates.append((round(cost, 12), angles))
    assert candidates, 'No new balanced panel'
    cost, angles = min(candidates)
    index = np.concatenate([ids(s, a) for s in range(6) for a in angles])
    hashes = [digest(images[i]) for i in index]
    assert len(set(hashes)) == 36 and not set(hashes) & previous
    np.savez_compressed(folder / 'data.npz', images=np.asarray(images[index]), labels=latent[index, 1],
                        factors=np.asarray(latent[index]), physical=np.asarray(physical[index]), official_ids=index)
    write(folder / 'audit.json', dict(status='PASS', chosen_angles=list(angles), common_angles=common,
          eligible_by_scale=eligible, candidate_panels=len(candidates), coverage_cost=cost,
          images=36, unique_pixels=36, hashes=hashes, previous_pixel_overlap=0,
          scope='new pixels and center factor combinations; not globally unseen orientations'))
    complete(folder, started)
    event('panel_prepared', images=36, angles=list(angles))


def initialize(device):
    """冻结数值规则与主要比较；显式记录研究方向受旧结果启发。"""
    assert not RUN.exists(), 'Preserve existing run'
    teacher.verify_protocol()
    preservation = {}
    for path in (OLD / 'delivery_manifest.json', PRIOR / 'delivery_manifest.json'):
        delivery = teacher.read(path)
        for item in delivery['files']:
            assert teacher.sha(BASE / item['path']) == item['sha256'], item['path']
        preservation[str(path.relative_to(BASE)).replace('\\', '/')] = dict(
            sha256=teacher.sha(path), files=len(delivery['files']))
    old_manifests = teacher.verify_old()
    inputs = {}
    dependency_paths = [DATA / 'manifest.json', DATA / 'indices.npz', PRIOR / 'panel/data.npz',
                        FULL / 'latents_classes.npy', FULL / 'latents_values.npy']
    for seed in SEEDS:
        dependency_paths += [OLD / f'seed_{seed}/teacher.pt',
                             OLD / f'development/seed_{seed}/discovery.npz',
                             OLD / f'development/seed_{seed}/fixed_orders.json']
    for name in ('dsprites_teacher.py', 'dsprites_mechanism.py', 'cross_layer_mechanism.py',
                 'residual_teacher.py', 'external_experiment.py', 'research_runtime.py'):
        dependency_paths.append(BASE / 'scripts' / name)
    inputs = {str(p.relative_to(BASE)).replace('\\', '/'): teacher.sha(p) for p in dependency_paths}
    RUN.mkdir()
    for name in CODE:
        destination = RUN / 'source' / name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASE / name, destination)
    protocol = dict(utc=teacher.now(), version='frozen-state-gradient-v1', seeds=SEEDS,
        prior_knowledge='Previous factor-transfer results read; this is a new follow-up, not an original preregistration',
        gradient='logit derivatives w.r.t source and middle activations; frozen weights, eval BN, float64',
        selection_strength=.5, evaluation_strengths=STRENGTHS, candidate_count=64, sparse_budget=16,
        primary=dict(target='margin', strength=1., comparison='state_linear vs fixed_linear',
                     metrics=['restore_RMS/natural_RMS', 'transplant_error_RMS/natural_RMS'],
                     aggregation='ratios per teacher first; average all six, report every paired result'),
        diagnostic=['gradient norm and direction drift over nonempty coalitions',
                    'source directional derivative of conditional interaction',
                    'middle baseline-J interaction plus exact nonlinear remainder',
                    'frozen four-gate affine control', 'predicted versus actual half-strength singleton scores'],
        singleton_prediction='restore f(S)-J(S)*delta_j or f(S)-J(0)*delta_j; transplant f(0)+J(0)*delta_j',
        methods=METHODS, vector_secondary='same margin-selected lists; centered-vector output metric',
        tuning='none; no fitting on new outputs; fixed 0.5 lists sealed before other strengths',
        zero_denominator='raw natural MSE stored; do not interpret undefined zero-signal ratios',
        sources={n: teacher.sha(BASE/n) for n in CODE}, inputs=inputs,
        preservation=preservation, legacy_manifests=old_manifests,
        environment=dict(torch=torch.__version__, numpy=np.__version__, device=device,
                         gpu=torch.cuda.get_device_name(0) if device=='cuda' else None))
    write(RUN / 'protocol.json', protocol)
    (RUN / 'protocol.sha256').write_text(teacher.sha(RUN / 'protocol.json')+'\n', encoding='ascii')
    prepare_panel()
    write(RUN / 'known_answers.json', gradient.known_answers())
    event('protocol_locked', sha256=teacher.sha(RUN / 'protocol.json'))


def old_coordinates(seed):
    """复制冻结坐标；shortlist为pool内索引，不是全层实际坐标。"""
    with np.load(OLD / f'development/seed_{seed}/discovery.npz', allow_pickle=False) as old:
        return [old[k].copy() for k in ('source_nodes', 'pool', 'shortlist')]


def load_panel(device):
    """官方二值像素直接转换float64，不除以255。"""
    with np.load(RUN / 'panel/data.npz', allow_pickle=False) as p:
        arrays = {k: p[k].copy() for k in p.files}
    return arrays, torch.as_tensor(arrays['images'][:, None], dtype=torch.float64, device=device)


def select(device):
    """在半强度下按梯度预测或实测单点收益选择中介。前九种方法共用候选名单和 16 个位置的预算。"""
    verify_protocol(); assert not (RUN / 'selection_freeze.json').exists()
    panel, images = load_panel(device); bits, stencil, metadata = mechanism.design()
    tensor_bits = torch.as_tensor(bits, dtype=torch.float64, device=device)
    (RUN / 'selection').mkdir(exist_ok=True)
    for seed in SEEDS:
        folder = RUN / 'selection' / f'seed_{seed}'
        if (folder / 'completion.json').exists(): verify_folder(folder); continue
        assert not folder.exists(), 'Preserve incomplete selection'; folder.mkdir(); started = time.perf_counter()
        model = teacher.load_teacher(seed, device)
        for parameter in model.parameters(): parameter.requires_grad_(False)
        nodes, pool, shortlist = old_coordinates(seed)
        fixed = teacher.read(OLD / f'development/seed_{seed}/fixed_orders.json')
        collected = {k: [] for k in ('natural', 'delta', 'middle_jacobian', 'source_jacobian',
                                     'finite_scores', 'predicted_scores', 'selections', 'singleton_logits')}
        for i, (image, label) in enumerate(zip(images, panel['labels'])):
            source = mechanism.source_activation(model, image[None])
            altered = mechanism.changed_sources(source, nodes, tensor_bits, .5)
            values = gradient.state_gradients(model, altered, pool, nodes)
            natural = values['logits'].cpu().numpy()
            middle = values['middle']; delta = (middle-middle[:1]).flatten(1)[:, pool].cpu().numpy()
            jac = values['middle_jacobian'].cpu().numpy()
            predicted = gradient.gradient_score_predictions(natural, delta, jac, shortlist, int(label), stencil)
            singleton = mechanism.patch(model, middle, [pool[[j]] for j in shortlist]).cpu().numpy()
            actual = mechanism.finite_scores(singleton, natural, int(label), stencil)
            old_feature = np.mean(np.abs(mechanism.pair_difference(delta, stencil) * gradient.margin_jacobian(jac[0], int(label))), 0)
            orders = [np.asarray(fixed['margin_class_finite'][int(label)]),
                      shortlist[np.argsort(-old_feature[shortlist], kind='stable')],
                      shortlist[np.argsort(-predicted[0, 0], kind='stable')],
                      shortlist[np.argsort(-predicted[1, 0], kind='stable')],
                      shortlist[np.argsort(-actual[0], kind='stable')]]
            orders += [np.asarray(fixed[f'random_{r}']) for r in range(4)]
            selections = [pool[x[:16]] for x in orders] + [pool[shortlist], pool]
            for k in range(9): assert len(np.unique(selections[k])) == 16
            payload = dict(natural=natural, delta=delta, middle_jacobian=jac,
                source_jacobian=values['source_jacobian'].cpu().numpy(), finite_scores=actual,
                predicted_scores=predicted, selections=mechanism.encode(selections, len(pool)), singleton_logits=singleton)
            for k, value in payload.items(): collected[k].append(value)
            if (i+1) % 6 == 0: event('selection_progress', seed=seed, images=i+1, total=len(images))
        np.savez_compressed(folder / 'data.npz', **{k: np.asarray(v) for k,v in collected.items()},
                            labels=panel['labels'], source_nodes=nodes, pool=pool, shortlist=shortlist,
                            stencil=stencil, pair_metadata=metadata)
        write(folder / 'cost.json', dict(images=len(images), state_gradient_forward_batches=len(images),
              class_backward_passes=len(images)*3, gradient_batch_size=16,
              finite_reference_tail_vectors=len(images)*64*2*16,
              note='Backward passes and forward vectors have different compute costs; report actual wall time'))
        complete(folder, started); event('selection_complete', seed=seed, seconds=time.perf_counter()-started)
        del model


def seal():
    """一次冻结全部教师的半强度名单，再开放其他强度评价。"""
    verify_protocol(); assert not (RUN / 'selection_freeze.json').exists()
    files = {}
    for seed in SEEDS:
        folder = RUN / 'selection' / f'seed_{seed}'; verify_folder(folder)
        files.update({str(p.relative_to(RUN)).replace('\\','/'):teacher.sha(p) for p in folder.iterdir()})
    write(RUN / 'selection_freeze.json', dict(utc=teacher.now(), protocol_sha256=teacher.sha(RUN/'protocol.json'), files=files))
    (RUN / 'selection_freeze.sha256').write_text(teacher.sha(RUN/'selection_freeze.json')+'\n', encoding='ascii')
    event('selection_sealed', files=len(files))


def verify_seal():
    """评价仅能使用已经封存的选择结果。"""
    verify_protocol()
    assert teacher.sha(RUN/'selection_freeze.json') == (RUN/'selection_freeze.sha256').read_text().strip()
    for name, expected in teacher.read(RUN/'selection_freeze.json')['files'].items():
        assert teacher.sha(RUN/name)==expected, name


def evaluate(device):
    """自然梯度变化、精确交互分解和实际联合修补；不再改名单。

    源方向导数用链式法则计算：d f_S/dα = -Σ_j bit_Sj*u_j*J_source,Sj。
    中介分解是精确恒等式 I=J0*ΔΔh + ΔΔ(f-f0-J0*Δh)，无需积分近似。
    """
    verify_seal(); panel, images = load_panel(device); bits, stencil, metadata = mechanism.design()
    tensor_bits = torch.as_tensor(bits, dtype=torch.float64, device=device)
    (RUN / 'evaluation').mkdir(exist_ok=True)
    for seed in SEEDS:
        model = teacher.load_teacher(seed, device)
        for parameter in model.parameters(): parameter.requires_grad_(False)
        nodes, pool, shortlist = old_coordinates(seed)
        with np.load(RUN/f'selection/seed_{seed}/data.npz', allow_pickle=False) as saved:
            selection = {k:saved[k].copy() for k in saved.files}
        for alpha in STRENGTHS:
            folder = RUN / 'evaluation' / f'seed_{seed}' / f'alpha_{alpha}'
            if (folder/'completion.json').exists(): verify_folder(folder); continue
            assert not folder.exists(), 'Preserve incomplete evaluation'; folder.mkdir(parents=True); started=time.perf_counter()
            collected={k:[] for k in ('natural','patched','delta','middle_jacobian','source_jacobian',
                       'middle_drift','source_drift','base_gradient_norm','directional_derivative',
                       'linear_middle_interaction','nonlinear_tail_interaction','frozen_gate_logits','gate_flips')}
            maximum=dict(forward=0., outside_pool=0., decomposition=0., fixed_gate_interaction=0., fixed_gate_baseline=0.)
            for i,(image,label) in enumerate(zip(images,panel['labels'])):
                source=mechanism.source_activation(model,image[None])
                altered=mechanism.changed_sources(source,nodes,tensor_bits,alpha)
                traced=mechanism.trace(model,altered)
                if alpha==.5:
                    values={k:torch.as_tensor(selection[k][i],device=device) for k in ('natural','middle_jacobian','source_jacobian')}
                    values['logits']=values.pop('natural');values['middle']=traced['h'][1]
                else: values=gradient.state_gradients(model,altered,pool,nodes)
                natural=values['logits'].cpu().numpy();middle=values['middle']
                full_delta=(middle-middle[:1]).flatten(1)
                delta=full_delta[:,pool].cpu().numpy()
                sj=values['source_jacobian'].cpu().numpy();mj=values['middle_jacobian'].cpu().numpy()
                md,mnorm=gradient.drift_statistics(mj,int(label));sd,snorm=gradient.drift_statistics(sj,int(label))
                direction=-source.flatten()[nodes].cpu().numpy()[None]*bits
                directional=np.einsum('scj,sj->sc',sj,direction)
                linear_values=np.einsum('cj,sj->sc',mj[0],delta)
                linear_interaction=mechanism.pair_difference(linear_values,stencil)
                remainder=natural-natural[:1]-linear_values
                nonlinear=mechanism.pair_difference(remainder,stencil)
                frozen=gradient.frozen_gate_output(model,altered,traced).cpu().numpy()
                selections=[row[row>=0] for row in selection['selections'][i]]
                patched=mechanism.patch(model,middle,selections).cpu().numpy()
                gate_flips=np.stack([((z>0)!=(z[:1]>0)).flatten(1).sum(1).cpu().numpy() for z in traced['z']],-1)
                outside=full_delta.clone();outside[:,pool]=0
                maximum['outside_pool']=max(maximum['outside_pool'],float(outside.abs().max()))
                maximum['forward']=max(maximum['forward'],float(np.max(np.abs(natural-traced['output'].cpu().numpy()))))
                maximum['decomposition']=max(maximum['decomposition'],float(np.max(np.abs(
                    linear_interaction+nonlinear-mechanism.pair_difference(natural,stencil)))))
                maximum['fixed_gate_interaction']=max(maximum['fixed_gate_interaction'],float(np.max(np.abs(mechanism.pair_difference(frozen,stencil)))))
                maximum['fixed_gate_baseline']=max(maximum['fixed_gate_baseline'],float(np.max(np.abs(frozen[0]-natural[0]))))
                payload=dict(natural=natural,patched=patched,delta=delta,middle_jacobian=mj,source_jacobian=sj,
                    middle_drift=md,source_drift=sd,base_gradient_norm=[mnorm,snorm],directional_derivative=directional,
                    linear_middle_interaction=linear_interaction,nonlinear_tail_interaction=nonlinear,
                    frozen_gate_logits=frozen,gate_flips=gate_flips)
                for k,value in payload.items():collected[k].append(value)
                if (i+1)%12==0:event('evaluation_progress',seed=seed,alpha=alpha,images=i+1,total=len(images))
            raw={k:np.asarray(v) for k,v in collected.items()}
            np.savez_compressed(folder/'data.npz',**raw,labels=panel['labels'],source_nodes=nodes,pool=pool,
                                shortlist=shortlist,stencil=stencil,pair_metadata=metadata)
            for mode in mechanism.TARGETS:
                n,r,t=mechanism.metric_arrays(raw['natural'],raw['patched'],panel['labels'],stencil,mode)
                np.savez_compressed(folder/f'metrics_{mode}.npz',natural_mse=n,restore_mse=r,transplant_error_mse=t)
            assert max(maximum.values())<1e-9, maximum
            write(folder/'checks.json',dict(status='PASS',maximum=maximum))
            complete(folder,started);event('evaluation_complete',seed=seed,alpha=alpha,seconds=time.perf_counter()-started,checks=maximum)
        del model


def summarize():
    """先按教师求平方均值之比的平方根，保持旧研究统计单位。"""
    verify_seal(); assert not (RUN/'summary.json').exists()
    rows=[]
    for seed in SEEDS:
        for alpha in STRENGTHS:
            folder=RUN/'evaluation'/f'seed_{seed}'/f'alpha_{alpha}';verify_folder(folder)
            for target in mechanism.TARGETS:
                with np.load(folder/f'metrics_{target}.npz',allow_pickle=False) as m:
                    denominator=float(m['natural_mse'].mean())
                    assert denominator>1e-20,'No measurable natural interaction'
                    for j,method in enumerate(METHODS):
                        rows.append(dict(seed=seed,alpha=alpha,target=target,method=method,
                            natural_rms=float(np.sqrt(denominator)),
                            restore_ratio=float(np.sqrt(m['restore_mse'][:,j].mean()/denominator)),
                            transplant_error_ratio=float(np.sqrt(m['transplant_error_mse'][:,j].mean()/denominator))))
    write(RUN/'summary.json',dict(utc=teacher.now(),rows=rows))
    event('summary_complete',rows=len(rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['init','select','seal','evaluate','summarize'],required=True)
    args=parser.parse_args();device=teacher.configure()
    if args.stage=='init':initialize(device)
    elif args.stage=='select':select(device)
    elif args.stage=='seal':seal()
    elif args.stage=='evaluate':evaluate(device)
    else:summarize()
