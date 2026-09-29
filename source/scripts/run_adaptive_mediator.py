"""第二轮的显式阶段入口：开发、锁定规则、半强度选择、锁定名单、完整强度评价。

复用旧项目的文件读取、模型加载、计算与数值核验，绝不改写上一轮目录。
所有写入都位于 runs/adaptive_mediator_20260928，报告由单独脚本生成。
"""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for thread_key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[thread_key] = '1'
import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
import run_cross_layer_mechanism as previous
from run_cross_layer_mechanism import (BASE, OLD, SEEDS, data, configure, sha, read,
    dump, now, clean, write_csv, load_teacher, verify_completion, ratio)
from cross_layer_mechanism import (design, source_activation, changed_sources, trace,
    decomposition, patch_outputs)
from adaptive_mediator import (BUDGETS, METHODS, input_features, make_shortlist, finite_scores,
    singleton_outputs, fixed_orders, selection_specs, select_for_input, encode_selections,
    metric_arrays)
from test_adaptive_mediator import known_answers, check_teacher

RUN = BASE / 'runs/adaptive_mediator_20260928'
PREVIOUS = BASE / 'runs/cross_layer_20260928'
ALPHAS = (1., .5)
CODE = ('scripts/adaptive_mediator.py', 'scripts/test_adaptive_mediator.py',
        'scripts/run_adaptive_mediator.py', 'reports/自适应中介_实验协议_2026-09-28.md')


def event(kind, **values):
    """记录阶段与进度；事件日志不参与选择评分。"""
    value = dict(utc=now(), event=kind, **values)
    with (RUN/'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(clean(value), ensure_ascii=False)+'\n')
    print(json.dumps(clean(value), ensure_ascii=False), flush=True)


def verify_prior():
    """原数值协议和两轮交付文件均保持原哈希，防止复用时改变历史。"""
    previous.verify_protocol()
    preserved = read(PREVIOUS/'delivery_manifest.json')['files']
    for entry in preserved:
        assert sha(BASE/entry['path']) == entry['sha256'], entry['path']
    older = read(BASE/'audit/joint_response_manifest_20260925.json')['entries']
    for entry in older:
        assert sha(BASE/entry['path']) == entry['sha256'], entry['path']
    return len(preserved), len(older)


def legacy_setup(seed):
    """读取第一轮已冻结源坐标、结构可达池和原始固定中介名单。"""
    folder = PREVIOUS/'development'/f'seed_{seed}'
    with np.load(folder/'selection_features.npz', allow_pickle=False) as saved:
        nodes, pool = saved['nodes'].copy(), saved['pool'].copy()
    return nodes, pool, read(folder/'selections.json')['selections']


def initialize(device):
    """新建独立目录、锁定代码/协议/数据编号；已有目录一律拒绝覆盖。"""
    assert not RUN.exists(), 'Preserve existing run'
    prior_counts = verify_prior()
    with np.load(PREVIOUS/'input_ids.npz', allow_pickle=False) as ids:
        discovery, validation = ids['discovery'].copy(), ids['validation'][:1].copy()
        old_evaluation = ids['evaluation'].copy()
    evaluation = np.arange(1088, 1152)
    assert not set(evaluation) & set(old_evaluation)
    RUN.mkdir()
    np.savez_compressed(RUN/'input_ids.npz', discovery=discovery, validation=validation, evaluation=evaluation)
    for name in CODE:
        dest = RUN/'source'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASE/name, dest)
    protocol = dict(version='adaptive-mediator-v1', utc=now(), seeds=SEEDS, q=4,
        selection_alpha=.5, evaluation_alphas=ALPHAS, primary=dict(alpha=1., budget=16),
        budgets=BUDGETS, methods=METHODS, shortlist_size=64, discovery_inputs=64,
        evaluation_ids=[1088,1151], evaluation_images=64, conditional_selection_uses_label=True,
        conditional_selection='same input half-dose only; fixed rule; no full-dose access before all selections sealed',
        statistics='descriptive teacher-first; no confirmatory p-values; half-dose is reused selection condition',
        source_and_pool='unchanged from cross_layer_20260928', calibration_model_access=False,
        training_runs=0, sources={n:sha(BASE/n) for n in CODE}, ids_sha256=sha(RUN/'input_ids.npz'),
        previous_manifest_sha256=sha(PREVIOUS/'delivery_manifest.json'), preserved_prior_files=prior_counts,
        environment=dict(device=device, torch=torch.__version__, numpy=np.__version__, dtype='float64'))
    dump(RUN/'protocol.json', protocol)
    (RUN/'protocol.sha256').write_text(sha(RUN/'protocol.json')+'\n', encoding='ascii')
    dump(RUN/'known_answer_controls.json', known_answers())
    event('protocol_locked', sha256=sha(RUN/'protocol.json'))


def verify_protocol():
    """各阶段重新核对新旧代码、输入和数据；不自动修复哈希不一致。"""
    p = read(RUN/'protocol.json')
    assert sha(RUN/'protocol.json') == (RUN/'protocol.sha256').read_text().strip()
    assert sha(RUN/'input_ids.npz') == p['ids_sha256']
    for name, expected in p['sources'].items():
        assert sha(BASE/name) == expected and sha(RUN/'source'/name) == expected, name
    assert sha(PREVIOUS/'delivery_manifest.json') == p['previous_manifest_sha256']
    verify_prior()
    return p


def complete(folder, filenames, **info):
    """阶段只在全部输出写好后提交完成标志，保留原始数组与耗时。"""
    dump(folder/'completion.json', dict(utc=now(), files={n:sha(folder/n) for n in filenames}, **info))


def prepare_trace(model, image, nodes, bits, alpha, device):
    """保持与第一轮相同的前缀、源修改和逐层网络计算。"""
    source = source_activation(model, image[None].to(device))
    return trace(model, changed_sources(source, nodes, bits, alpha))


def develop(device):
    """仅用训练输入获得共用候选、固定梯度排序和固定实际干预排序。"""
    verify_protocol()
    assert not (RUN/'fixed_freeze.json').exists()
    with np.load(RUN/'input_ids.npz') as ids:
        train_ids, val_ids = ids['discovery'].copy(), ids['validation'].copy()
    images, labels = data('train', train_ids)
    validation, validation_labels = data('train', val_ids)
    raw_bits, stencil, metadata = design()
    bits = torch.as_tensor(raw_bits, dtype=torch.float64, device=device)
    for seed in SEEDS:
        folder = RUN/'development'/f'seed_{seed}'
        if folder.exists():
            verify_completion(folder); continue
        folder.mkdir(parents=True)
        start = time.perf_counter()
        model = load_teacher(OLD/f'seed_{seed}', device=device)
        nodes, pool, _ = legacy_setup(seed)
        features = []
        for image, label in zip(images, labels):
            traced = prepare_trace(model, image, nodes, bits, .5, device)
            features.append(input_features(model, traced['h'][1], int(label), stencil))
        features = np.stack(features)
        shortlist = make_shortlist(features.mean(0), pool)
        checks = check_teacher(model, validation[0], int(validation_labels[0]), nodes, shortlist, device)
        singletons, natural, scores = [], [], []
        for i, (image, label) in enumerate(zip(images, labels)):
            traced = prepare_trace(model, image, nodes, bits, .5, device)
            raw = singleton_outputs(model, traced['h'][1], shortlist)
            original = traced['output'].cpu().numpy()
            singletons.append(raw); natural.append(original)
            scores.append(finite_scores(raw, original, int(label), stencil))
            if (i+1) % 16 == 0:
                event('development_progress', seed=seed, inputs=i+1, total=len(images))
        scores = np.stack(scores)
        orders = fixed_orders(features.mean(0), scores.mean(0), shortlist, seed)
        np.savez_compressed(folder/'discovery.npz', features=features, finite_scores=scores,
            singleton_logits=np.stack(singletons), natural=np.stack(natural), shortlist=shortlist,
            pool=pool, source_nodes=nodes, ids=train_ids, labels=labels, stencil=stencil, pair_metadata=metadata)
        dump(folder/'fixed_orders.json', orders)
        dump(folder/'numerical_checks.json', checks)
        complete(folder, ['discovery.npz','fixed_orders.json','numerical_checks.json'],
                 seconds=time.perf_counter()-start, singleton_output_vectors=len(images)*64*2*16)
        event('development_complete', seed=seed, seconds=time.perf_counter()-start,
              max_numeric_error=max(checks['errors'].values()))
        del model


def freeze_files(name, folders):
    """把所有教师的指定阶段统一封存，禁止先评价部分教师再调其余规则。"""
    verify_protocol()
    assert not (RUN/f'{name}.json').exists()
    files = {}
    for folder in folders:
        done = verify_completion(folder)
        for file in [*done['files'], 'completion.json']:
            files[(folder/file).relative_to(RUN).as_posix()] = sha(folder/file)
    dump(RUN/f'{name}.json', dict(utc=now(), protocol_sha256=sha(RUN/'protocol.json'), files=files))
    (RUN/f'{name}.sha256').write_text(sha(RUN/f'{name}.json')+'\n', encoding='ascii')
    event(name, files=len(files))


def verify_freeze(name):
    """检查指定冻结点及其全部阶段文件，包含源协议绑定关系。"""
    verify_protocol()
    assert sha(RUN/f'{name}.json') == (RUN/f'{name}.sha256').read_text().strip()
    frozen = read(RUN/f'{name}.json')
    assert frozen['protocol_sha256'] == sha(RUN/'protocol.json')
    for file, expected in frozen['files'].items():
        assert sha(RUN/file) == expected, file


def select_inputs(device):
    """只运行新输入的半强度干预以选出中介，不读取完整强度的任何输出。"""
    verify_freeze('fixed_freeze')
    assert not (RUN/'evaluation_opened.json').exists()
    assert not (RUN/'adaptive_freeze.json').exists()
    if not (RUN/'selection_opened.json').exists():
        dump(RUN/'selection_opened.json', dict(utc=now(), fixed_freeze_sha256=sha(RUN/'fixed_freeze.json')))
    with np.load(RUN/'input_ids.npz') as saved:
        ids = saved['evaluation'].copy()
    images, labels = data('test', ids)
    raw_bits, stencil, metadata = design()
    bits = torch.as_tensor(raw_bits, dtype=torch.float64, device=device)
    for seed in SEEDS:
        folder = RUN/'input_selection'/f'seed_{seed}'
        if folder.exists():
            verify_completion(folder); continue
        folder.mkdir(parents=True)
        start = time.perf_counter()
        dev = RUN/'development'/f'seed_{seed}'
        with np.load(dev/'discovery.npz') as saved:
            shortlist, pool, nodes = saved['shortlist'].copy(), saved['pool'].copy(), saved['source_nodes'].copy()
        fixed = read(dev/'fixed_orders.json')
        _, _, legacy = legacy_setup(seed)
        model = load_teacher(OLD/f'seed_{seed}', device=device)
        natural, features, singles, scores, selections = [], [], [], [], []
        for i, (image, label) in enumerate(zip(images, labels)):
            traced = prepare_trace(model, image, nodes, bits, .5, device)
            original = traced['output'].cpu().numpy()
            feature = input_features(model, traced['h'][1], int(label), stencil)
            singleton = singleton_outputs(model, traced['h'][1], shortlist)
            score = finite_scores(singleton, original, int(label), stencil)
            chosen = select_for_input(feature, score, shortlist, pool, fixed, legacy)
            selections.append(encode_selections(chosen, len(pool)))
            natural.append(original); features.append(feature); singles.append(singleton); scores.append(score)
            if (i+1) % 16 == 0:
                event('half_dose_selection_progress', seed=seed, inputs=i+1, total=len(images))
        np.savez_compressed(folder/'selection.npz', natural=np.stack(natural), features=np.stack(features),
            singleton_logits=np.stack(singles), finite_scores=np.stack(scores), selections=np.stack(selections),
            shortlist=shortlist, pool=pool, source_nodes=nodes, ids=ids, labels=labels,
            stencil=stencil, pair_metadata=metadata)
        dump(folder/'specs.json', selection_specs(len(pool)))
        complete(folder, ['selection.npz','specs.json'], seconds=time.perf_counter()-start,
                 singleton_output_vectors=len(images)*64*2*16, alpha_accessed=.5)
        event('input_selection_complete', seed=seed, seconds=time.perf_counter()-start)
        del model


def evaluate(device):
    """全部半强度名单锁定后才运行完整强度；不允许评价中更改选点。"""
    verify_freeze('fixed_freeze'); verify_freeze('adaptive_freeze')
    if not (RUN/'evaluation_opened.json').exists():
        dump(RUN/'evaluation_opened.json', dict(utc=now(), adaptive_freeze_sha256=sha(RUN/'adaptive_freeze.json')))
    with np.load(RUN/'input_ids.npz') as saved:
        ids = saved['evaluation'].copy()
    images, labels = data('test', ids)
    raw_bits, stencil, metadata = design()
    bits = torch.as_tensor(raw_bits, dtype=torch.float64, device=device)
    for seed in SEEDS:
        selected_folder = RUN/'input_selection'/f'seed_{seed}'
        with np.load(selected_folder/'selection.npz') as saved:
            selections, nodes = saved['selections'].copy(), saved['source_nodes'].copy()
            original_half = saved['natural'].copy()
        specs = read(selected_folder/'specs.json')
        model = load_teacher(OLD/f'seed_{seed}', device=device)
        for alpha in ALPHAS:
            folder = RUN/'evaluation'/f'seed_{seed}'/f'alpha_{alpha}'
            if folder.exists():
                verify_completion(folder); continue
            folder.mkdir(parents=True)
            start = time.perf_counter()
            natural, patched = [], []
            maxima = {}
            for i, image in enumerate(images):
                traced = prepare_trace(model, image, nodes, bits, alpha, device)
                _, _, errors = decomposition(model, traced, stencil)
                chosen = [selections[i, j, :s['budget']] for j, s in enumerate(specs)]
                output = patch_outputs(model, traced['h'][1], chosen).cpu().numpy()
                original = traced['output'].cpu().numpy()
                # 可达池之外的坐标不应受源干预影响，因此全可达修补等价全切口。
                errors['all_reachable_restore'] = float(np.max(np.abs(output[-1, 0]-original[:1])))
                errors['all_reachable_transplant'] = float(np.max(np.abs(output[-1, 1]-original)))
                if alpha == .5:
                    errors['selection_natural_replay'] = float(np.max(np.abs(original-original_half[i])))
                assert max(errors.values()) < 1e-9, errors
                for key, value in errors.items():
                    maxima[key] = max(maxima.get(key, 0.), value)
                natural.append(original); patched.append(output)
                if (i+1) % 16 == 0:
                    event('evaluation_progress', seed=seed, alpha=alpha, inputs=i+1, total=len(images))
            np.savez_compressed(folder/'responses.npz', natural=np.stack(natural), patched=np.stack(patched),
                ids=ids, labels=labels, stencil=stencil, pair_metadata=metadata, bits=raw_bits)
            dump(folder/'checks.json', dict(status='PASS', errors=maxima))
            complete(folder, ['responses.npz','checks.json'], seconds=time.perf_counter()-start,
                     natural_output_vectors=len(images)*16, patched_output_vectors=len(images)*len(specs)*2*16)
            event('evaluation_complete', seed=seed, alpha=alpha, seconds=time.perf_counter()-start,
                  max_numeric_error=max(maxima.values()))
        del model


def summarize():
    """按教师聚合平方量再转RMS；另存每输入绝对量，不做事后主终点筛选。"""
    verify_freeze('fixed_freeze'); verify_freeze('adaptive_freeze')
    assert not (RUN/'summary.json').exists()
    baseline, methods, per_input, overlaps = [], [], [], []
    for seed in SEEDS:
        selection_folder = RUN/'input_selection'/f'seed_{seed}'
        specs = read(selection_folder/'specs.json')
        with np.load(selection_folder/'selection.npz') as saved:
            selections, ids = saved['selections'].copy(), saved['ids'].copy()
        # 同一评分下固定/动态的重叠及动态输入之间的重叠只是选择变化描述。
        for method, fixed_method in [('input_gradient','fixed_gradient'),('input_finite','fixed_finite')]:
            a = next(i for i,s in enumerate(specs) if s==dict(method=method,budget=16))
            b = next(i for i,s in enumerate(specs) if s==dict(method=fixed_method,budget=16))
            sets = [set(row[a,:16]) for row in selections]
            fixed_set = set(selections[0,b,:16])
            overlap = [len(x & fixed_set)/len(x | fixed_set) for x in sets]
            pair_overlap = [len(sets[i]&sets[j])/len(sets[i]|sets[j]) for i in range(len(sets)) for j in range(i)]
            overlaps.append(dict(seed=seed,method=method,budget=16,
                mean_jaccard_to_fixed=float(np.mean(overlap)), mean_jaccard_between_inputs=float(np.mean(pair_overlap))))
        for alpha in ALPHAS:
            folder = RUN/'evaluation'/f'seed_{seed}'/f'alpha_{alpha}'
            verify_completion(folder)
            with np.load(folder/'responses.npz') as saved:
                natural, patched = saved['natural'], saved['patched']
                labels, stencil = saved['labels'], saved['stencil']
                n, r, t = metric_arrays(natural, patched, labels, stencil)
            denominator = float(np.sqrt(n.mean()))
            baseline.append(dict(seed=seed,alpha=alpha,interaction_rms=denominator,inputs=len(ids)))
            for j, spec in enumerate(specs):
                rr, tt = float(np.sqrt(r[:,j].mean())), float(np.sqrt(t[:,j].mean()))
                restore, transplant = ratio(rr,denominator), ratio(tt,denominator)
                methods.append(dict(seed=seed,alpha=alpha,**spec,interaction_rms=denominator,
                    restored_rms=rr,transplant_error_rms=tt,restore_ratio=restore,
                    transplant_error_ratio=transplant,joint_loss=.5*(restore**2+transplant**2)))
                for i, input_id in enumerate(ids):
                    den, res, err = np.sqrt(n[i]), np.sqrt(r[i,j]), np.sqrt(t[i,j])
                    per_input.append(dict(seed=seed,alpha=alpha,input_id=int(input_id),label=int(labels[i]),
                        **spec,interaction_rms=float(den),restored_rms=float(res),transplant_error_rms=float(err),
                        restore_ratio=ratio(res,den),transplant_error_ratio=ratio(err,den)))
    for name, rows in [('baseline',baseline),('methods',methods),('per_input',per_input),('overlaps',overlaps)]:
        write_csv(RUN/f'{name}.csv', rows)
    result = dict(status='PASS',utc=now(),baseline=baseline,methods=methods,overlaps=overlaps,
        teachers=6,distinct_images=64,primary=dict(alpha=1.,budget=16),selection_alpha=.5,
        natural_evaluation_vectors=6*64*2*16,group_patch_evaluation_vectors=6*64*2*38*2*16,
        singleton_selection_vectors=2*6*64*64*2*16,calibration_model_access=False,new_training_runs=0)
    dump(RUN/'summary.json',result)
    event('summary_complete',method_rows=len(methods),input_rows=len(per_input))


def verify_all():
    """完成后的只读完整性和阶段先后检查；无需再次运行任何模型。"""
    verify_freeze('fixed_freeze'); verify_freeze('adaptive_freeze')
    assert read(RUN/'fixed_freeze.json')['utc'] < read(RUN/'selection_opened.json')['utc']
    assert read(RUN/'adaptive_freeze.json')['utc'] < read(RUN/'evaluation_opened.json')['utc']
    for seed in SEEDS:
        for stage in ('development','input_selection'):
            verify_completion(RUN/stage/f'seed_{seed}')
        for alpha in ALPHAS:
            verify_completion(RUN/'evaluation'/f'seed_{seed}'/f'alpha_{alpha}')
    summary=read(RUN/'summary.json')
    assert len(summary['methods']) == 456
    print(json.dumps(dict(status='PASS',method_rows=456,old_delivery_preserved=[90,136],
                         calibration_model_access=False)),flush=True)


def smoke(device):
    """正式冻结前仅用既有验证输入核查新旧公式和钩子，不创建实验结果。"""
    print(json.dumps(known_answers(),ensure_ascii=False),flush=True)
    with np.load(PREVIOUS/'input_ids.npz') as saved:
        ids = saved['validation'][:1]
    images, labels = data('train',ids)
    seed=SEEDS[0]; nodes,pool,_=legacy_setup(seed)
    model=load_teacher(OLD/f'seed_{seed}',device=device)
    print(json.dumps(check_teacher(model,images[0],int(labels[0]),nodes,pool[:64],device),ensure_ascii=False),flush=True)


def main():
    """一个命令只执行一个阶段；评价前必须有六模型的两次完整冻结。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['smoke','init','develop','freeze','select','seal','evaluate','summarize','verify'])
    args=parser.parse_args(); device=configure()
    if args.stage=='smoke': smoke(device)
    elif args.stage=='init': initialize(device)
    elif args.stage=='develop': develop(device)
    elif args.stage=='freeze':
        assert not (RUN/'selection_opened.json').exists()
        freeze_files('fixed_freeze',[RUN/'development'/f'seed_{s}' for s in SEEDS])
    elif args.stage=='select': select_inputs(device)
    elif args.stage=='seal':
        verify_freeze('fixed_freeze')
        assert not (RUN/'evaluation_opened.json').exists()
        freeze_files('adaptive_freeze',[RUN/'input_selection'/f'seed_{s}' for s in SEEDS])
    elif args.stage=='evaluate': evaluate(device)
    elif args.stage=='summarize': summarize()
    else: verify_all()


if __name__=='__main__': main()
