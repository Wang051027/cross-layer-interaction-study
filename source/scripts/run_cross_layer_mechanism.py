"""跨层机制第一轮的阶段入口：先开发冻结，再评价和只读核验。

运行顺序：init → develop → freeze → evaluate → summarize → verify。
所有新文件位于独立 runs/cross_layer_20260928；不改写旧模型或旧报告。
"""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for thread_key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[thread_key] = '1'
import argparse, csv, gzip, hashlib, json, platform, shutil
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
from residual_teacher import load_teacher
from cross_layer_mechanism import (STAGES, design, source_activation, changed_sources,
    trace, decomposition, patch_outputs, reachable_middle, select_source_nodes,
    learn_selection, selections_from_features)
from test_cross_layer_mechanism import known_answers, check_teacher

BASE = Path(__file__).resolve().parents[1]
RUN = BASE/'runs/cross_layer_20260928'
OLD = BASE/'runs/fashion_residual_20260913'
DATA = BASE/'data/fashion_mnist/prepared'
SEEDS = list(range(202609071, 202609077))
ALPHAS = (1., .5)
TOL = 1e-9
CODE = ['scripts/cross_layer_mechanism.py', 'scripts/test_cross_layer_mechanism.py',
        'scripts/run_cross_layer_mechanism.py', 'reports/跨层交互_实验协议_2026-09-28.md']


def sha(path):
    """按文件字节计算摘要；用于来源、阶段冻结和完成后回读。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    """读取本阶段 JSON 配置或记录。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def clean(value):
    """转成 JSON 类型，近零分母产生的 NaN 明确保留为 null。"""
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    return value


def dump(path, value):
    """写可审阅的 UTF-8 JSON，不允许非标准 NaN 字面量。"""
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def now():
    """记录 UTC 时间，区分开发、冻结和评价开始。"""
    return datetime.now(timezone.utc).isoformat()


def event(kind, **values):
    """即时打印进度，并在新运行目录保存同一条事件。"""
    record = dict(utc=now(), event=kind, **values)
    with (RUN/'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(clean(record), ensure_ascii=False)+'\n')
    print(json.dumps(clean(record), ensure_ascii=False), flush=True)


def configure():
    """固定线程、确定性和精度，优先用本机 CUDA；不创建训练任务。"""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def initialize(device):
    """核对旧来源，保存协议和源码快照；拒绝覆盖已存在的研究目录。"""
    assert not RUN.exists(), 'Keep existing run; never overwrite protocol'
    inputs = read(OLD/'protocol.json')['data_hashes'].copy()
    for seed in SEEDS:
        folder = OLD/f'seed_{seed}'
        expected = read(folder/'teacher_summary.json')['checkpoint_sha256']
        assert sha(folder/'teacher.pt') == expected
        inputs[(folder/'teacher.pt').relative_to(BASE).as_posix()] = expected
    # 已完成研究清单中的原件全部保全；新文件不会写入这些历史位置。
    preserved = read(BASE/'audit/joint_response_manifest_20260925.json')['entries']
    for entry in preserved: assert sha(BASE/entry['path']) == entry['sha256'], entry['path']
    for relative, expected in inputs.items(): assert sha(BASE/relative) == expected, relative
    RUN.mkdir()
    with np.load(DATA/'context_indices.npz', allow_pickle=False) as archive:
        selection_ids = archive['train'].copy()
        discovery_ids = selection_ids[:64].copy()
        validation_ids = archive['validation'][:4].copy()
    evaluation_ids = np.arange(1024, 1088)
    with np.load(DATA/'split_indices.npz', allow_pickle=False) as archive:
        assert set(selection_ids).issubset(set(archive['train']))
        assert set(validation_ids).issubset(set(archive['validation']))
        assert not set(selection_ids) & set(archive['calibration'])
        assert not set(validation_ids) & set(archive['calibration'])
    np.savez_compressed(RUN/'input_ids.npz', source_selection=selection_ids, discovery=discovery_ids,
                        validation=validation_ids, evaluation=evaluation_ids)
    for name in CODE:
        dest = RUN/'source'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASE/name, dest)
    original_sources = ['scripts/residual_teacher.py', 'scripts/external_experiment.py',
        'scripts/research_runtime.py', 'upstream/CAT_MT_phase7_v0.7/code/context_module.py',
        'upstream/CAT_MT_phase7_v0.7/vendor/local_modules.py']
    protocol = dict(version='cross-layer-mechanism-v1', utc=now(), seeds=SEEDS,
        q=4, strengths=ALPHAS, budgets=[8, 16, 32], primary=dict(alpha=1., budget=16),
        source='block1 output; top4 training mean in center3x3 across16channels',
        stages=STAGES, mediator='block2 full output; selected scalar coordinates',
        discovery_inputs=64, source_selection_inputs=512, validation_inputs=4, evaluation_inputs=64,
        evaluation_ids=[1024,1087], pair_backgrounds='all24; origin6 supplementary',
        main_target='true class logit minus mean9other logits',
        selection='development mean abs(baseline target gradient * conditional pair interaction)',
        controls=['single_effect', 'activation', '4 matched repeats'],
        calibration_model_access=False, new_training_runs=0,
        prior_access='old full test classification known; new early-source interventions on1024..1087; same dataset follow-up',
        statistics='descriptive teacher-first mean/SD; no confirmatory p-values or universal mediation claims',
        tolerance=TOL, sources={name:sha(BASE/name) for name in CODE},
        old_sources={name:sha(BASE/name) for name in original_sources}, input_hashes=inputs,
        ids_sha256=sha(RUN/'input_ids.npz'), preserved_old_delivery=preserved,
        environment=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
            device=device, gpu=torch.cuda.get_device_name(0) if device=='cuda' else None, precision='float64', threads=1))
    dump(RUN/'protocol.json', protocol)
    (RUN/'protocol.sha256').write_text(sha(RUN/'protocol.json')+'\n', encoding='ascii')
    dump(RUN/'known_answer_controls.json', known_answers())
    event('protocol_locked', sha256=sha(RUN/'protocol.json'))


def verify_protocol():
    """核对当前算法、快照、旧模型和输入；任何变化均停止而不绕过冻结。"""
    p = read(RUN/'protocol.json')
    assert sha(RUN/'protocol.json') == (RUN/'protocol.sha256').read_text().strip()
    assert sha(RUN/'input_ids.npz') == p['ids_sha256']
    for name, expected in p['sources'].items():
        assert sha(BASE/name) == expected and sha(RUN/'source'/name) == expected, name
    for name, expected in {**p['old_sources'], **p['input_hashes']}.items():
        assert sha(BASE/name) == expected, name
    return p


def data(pool, ids):
    """按明确的训练/官方测试来源取图像；像素预处理与原教师保持一致。"""
    if pool == 'train':
        images = np.load(DATA/'official_train_images.npy', mmap_mode='r', allow_pickle=False)[ids]
        labels = np.load(DATA/'official_train_labels.npy', allow_pickle=False)[ids]
    elif pool == 'test':
        raw = gzip.decompress((BASE/'data/fashion_mnist/raw/t10k-images-idx3-ubyte.gz').read_bytes())
        assert tuple(int.from_bytes(raw[i:i+4], 'big') for i in range(0, 16, 4)) == (2051,10000,28,28)
        images = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(10000,28,28)[ids]
        labels_raw = gzip.decompress((BASE/'data/fashion_mnist/raw/t10k-labels-idx1-ubyte.gz').read_bytes())
        assert int.from_bytes(labels_raw[:4], 'big') == 2049
        labels = np.frombuffer(labels_raw, dtype=np.uint8, offset=8)[ids]
    else: raise ValueError(pool)
    normalized = images[:,None].astype(np.float32) / np.float32(255)
    return torch.from_numpy(normalized.astype(np.float64)), labels.astype(int)


def verify_completion(folder):
    """完整设置可跳过重跑；未完成目录保持原样，便于检查中断。"""
    done = read(folder/'completion.json')
    for filename, expected in done['files'].items(): assert sha(folder/filename) == expected
    return done


def develop(device):
    """只访问训练与验证输入，生成六教师候选集合并完成独立数值检查。"""
    verify_protocol()
    assert not (RUN/'evaluation_opened.json').exists()
    with np.load(RUN/'input_ids.npz') as ids:
        select_images, _ = data('train', ids['source_selection'])
        images, labels = data('train', ids['discovery'])
        validation, _ = data('train', ids['validation'])
    raw_bits, stencil, _ = design()
    bits = torch.as_tensor(raw_bits, dtype=torch.float64, device=device)
    for seed in SEEDS:
        folder = RUN/'development'/f'seed_{seed}'
        if folder.exists(): verify_completion(folder); continue
        folder.mkdir(parents=True)
        model = load_teacher(OLD/f'seed_{seed}', device=device)
        nodes = select_source_nodes(model, select_images, device)
        pool = reachable_middle(model, nodes, device)
        features = learn_selection(model, images, labels, nodes, bits, stencil, device)
        selections = selections_from_features(features, pool, seed)
        checks = check_teacher(model, validation.to(device), nodes)
        np.savez_compressed(folder/'selection_features.npz', nodes=nodes, pool=pool, features=features)
        dump(folder/'selections.json', dict(seed=seed, nodes=nodes, pool_size=len(pool), selections=selections))
        dump(folder/'numerical_checks.json', checks)
        dump(folder/'completion.json', dict(utc=now(), files={name:sha(folder/name) for name in
            ('selection_features.npz','selections.json','numerical_checks.json')}))
        event('development_complete', seed=seed, source_nodes=nodes, pool_size=len(pool),
              max_numeric_error=max(checks['errors'].values()))
        del model


def freeze():
    """六教师全部完成后一次冻结全部选择；评价不能重新挑坐标或预算。"""
    verify_protocol()
    assert not (RUN/'evaluation_opened.json').exists()
    assert not (RUN/'freeze.json').exists()
    files = {}
    for seed in SEEDS:
        folder = RUN/'development'/f'seed_{seed}'
        completion = verify_completion(folder)
        for name in [*completion['files'], 'completion.json']:
            files[(folder/name).relative_to(RUN).as_posix()] = sha(folder/name)
    dump(RUN/'freeze.json', dict(utc=now(), protocol_sha256=sha(RUN/'protocol.json'), files=files))
    (RUN/'freeze.sha256').write_text(sha(RUN/'freeze.json')+'\n', encoding='ascii')
    event('all_selections_frozen', files=len(files), sha256=sha(RUN/'freeze.json'))


def verify_freeze():
    """每次进入评价或汇总均重新检查原冻结选择。"""
    verify_protocol()
    assert sha(RUN/'freeze.json') == (RUN/'freeze.sha256').read_text().strip()
    f = read(RUN/'freeze.json')
    assert f['protocol_sha256'] == sha(RUN/'protocol.json')
    for name, expected in f['files'].items(): assert sha(RUN/name) == expected, name


def evaluate(device):
    """运行固定新批次的自然干预和恢复/移植，保存全部 logits 供回读。"""
    verify_freeze()
    if not (RUN/'evaluation_opened.json').exists():
        dump(RUN/'evaluation_opened.json', dict(utc=now(), freeze_sha256=sha(RUN/'freeze.json')))
    with np.load(RUN/'input_ids.npz') as saved: ids = saved['evaluation'].copy()
    images, labels = data('test', ids)
    raw_bits, stencil, metadata = design()
    bits = torch.as_tensor(raw_bits, dtype=torch.float64, device=device)
    for seed in SEEDS:
        selection = read(RUN/'development'/f'seed_{seed}'/'selections.json')
        sets = [s['nodes'] for s in selection['selections']]
        model = load_teacher(OLD/f'seed_{seed}', device=device)
        for alpha in ALPHAS:
            folder = RUN/'evaluation'/f'seed_{seed}'/f'alpha_{alpha}'
            if folder.exists(): verify_completion(folder); continue
            folder.mkdir(parents=True)
            natural, patched, parts, stages = [], [], [], []
            maxima = {}
            for index, image in enumerate(images):
                source = source_activation(model, image[None].to(device))
                traced = trace(model, changed_sources(source, selection['nodes'], bits, alpha))
                contribution, layer, checks = decomposition(model, traced, stencil)
                assert max(checks.values()) < TOL, checks
                for name, value in checks.items(): maxima[name] = max(maxima.get(name,0.), value)
                intervention = patch_outputs(model, traced['h'][1], sets)
                natural.append(traced['output'].cpu().numpy())
                patched.append(intervention.cpu().numpy())
                parts.append(contribution.cpu().numpy()); stages.append(layer)
                if (index+1) % 16 == 0:
                    event('evaluation_progress', seed=seed, alpha=alpha, inputs=index+1, total=len(images))
            np.savez_compressed(folder/'responses.npz', natural=np.stack(natural),
                patched=np.stack(patched), contributions=np.stack(parts), stage_statistics=np.stack(stages),
                ids=ids, labels=labels, bits=raw_bits, stencil=stencil, pair_metadata=metadata)
            dump(folder/'checks.json', dict(status='PASS', errors=maxima, input_count=len(images),
                natural_output_vectors=len(images)*16, patched_output_vectors=len(images)*16*2*len(sets)))
            dump(folder/'completion.json', dict(utc=now(), files={name:sha(folder/name) for name in
                ('responses.npz','checks.json')}))
            event('evaluation_case_complete', seed=seed, alpha=alpha, max_numeric_error=max(maxima.values()))
        del model
    event('evaluation_complete', seeds=len(SEEDS), inputs_per_teacher=len(images))


def project(values, labels):
    """对保存数组作固定标量投影，保留组合、阶段等中间维度。"""
    label_shape = (len(labels),) + (1,)*(values.ndim-1)
    selected = np.take_along_axis(values, labels.reshape(label_shape), axis=-1)[...,0]
    return (10*selected-values.sum(-1))/9


def differences(values, stencil):
    """独立 NumPy 回读：组合轴为 1，与执行模块的组合轴 0 实现分开。"""
    return values[:,stencil[:,0]]-values[:,stencil[:,1]]-values[:,stencil[:,2]]+values[:,stencil[:,3]]


def rms(value):
    """报告绝对效应/残差；不同层的原始 RMS 不直接作为跨层强弱排序。"""
    return float(np.sqrt(np.square(value).mean()))


def ratio(numerator, denominator):
    """不裁剪比率；零分母记 NaN，序列化时写为 null。"""
    return numerator/denominator if denominator > TOL else np.nan


def write_csv(path, rows):
    """按固定字段顺序保存 UTF-8 CSV。"""
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(clean(rows))


def summarize():
    """从已保存 logits 独立计算指标，先教师内汇总，再跨教师描述。"""
    verify_freeze()
    summary_path = RUN/'summary.json'
    assert not summary_path.exists(), 'Preserve completed summary; use verify for rereading'
    baseline_rows, method_rows, input_rows, generation_rows, stage_rows = [], [], [], [], []
    numerical = 0.
    for seed in SEEDS:
        selections = read(RUN/'development'/f'seed_{seed}'/'selections.json')['selections']
        for alpha in ALPHAS:
            folder = RUN/'evaluation'/f'seed_{seed}'/f'alpha_{alpha}'
            verify_completion(folder)
            with np.load(folder/'responses.npz', allow_pickle=False) as f:
                natural=f['natural']; patched=f['patched']; labels=f['labels']; ids=f['ids']
                stencil=f['stencil']; metadata=f['pair_metadata']; contributions=f['contributions']
                stage=f['stage_statistics']
            vector_i=differences(natural,stencil)
            numerical=max(numerical,float(np.abs(contributions.sum(1)-vector_i).max()))
            scalar=project(natural,labels); scalar_i=differences(scalar,stencil)
            source_effect=scalar[:,1:]-scalar[:,:1]
            inter_rms=rms(scalar_i); effect_rms=rms(source_effect)
            origin=metadata[:,2]==0
            baseline_rows.append(dict(seed=seed,alpha=alpha,interaction_rms=inter_rms,
                effect_rms=effect_rms,interaction_to_effect=ratio(inter_rms,effect_rms),
                origin_interaction_rms=rms(scalar_i[:,origin]),vector_interaction_rms=rms(vector_i),
                active_interaction_inputs=int((np.sqrt(np.square(scalar_i).mean(1))>TOL).sum()),
                input_count=len(ids)))
            projected_parts=project(contributions,labels)
            for g in range(4):
                generation_rows.append(dict(seed=seed,alpha=alpha,stage=STAGES[g],
                    output_component_rms=rms(projected_parts[:,g]),
                    component_to_total=ratio(rms(projected_parts[:,g]),inter_rms)))
                stage_rows.append(dict(seed=seed,alpha=alpha,stage=STAGES[g],
                    interaction_state_rms=float(np.sqrt(np.square(stage[:,g,0]).mean())),
                    fixed_gate_term_rms=float(np.sqrt(np.square(stage[:,g,1]).mean())),
                    gate_remainder_rms=float(np.sqrt(np.square(stage[:,g,2]).mean())),
                    changed_gate_count=float(stage[:,g,3].mean()),state_count=int(stage[0,g,4])))
            for index,spec in enumerate(selections):
                restored=patched[:,index,0]; transplanted=patched[:,index,1]
                ri=differences(project(restored,labels),stencil)
                ti=differences(project(transplanted,labels),stencil)
                row=dict(seed=seed,alpha=alpha,method=spec['method'],budget=spec['budget'],
                    natural_interaction_rms=inter_rms,restored_interaction_rms=rms(ri),
                    transplant_interaction_error_rms=rms(ti-scalar_i),
                    restore_ratio=ratio(rms(ri),inter_rms),transplant_error_ratio=ratio(rms(ti-scalar_i),inter_rms),
                    vector_restore_ratio=ratio(rms(differences(restored,stencil)),rms(vector_i)),
                    vector_transplant_error_ratio=ratio(rms(differences(transplanted,stencil)-vector_i),rms(vector_i)),
                    restored_effect_rms=rms(project(restored,labels)[:,1:]-project(restored,labels)[:,:1]),
                    transplant_response_error_rms=rms((project(transplanted,labels)[:,1:]-project(transplanted,labels)[:,:1])-source_effect))
                method_rows.append(row)
                for n,input_id in enumerate(ids):
                    denominator=rms(scalar_i[n]); eff=rms(source_effect[n])
                    input_rows.append(dict(seed=seed,alpha=alpha,input_id=int(input_id),label=int(labels[n]),
                        method=spec['method'],budget=spec['budget'],interaction_rms=denominator,effect_rms=eff,
                        interaction_to_effect=ratio(denominator,eff),restored_interaction_rms=rms(ri[n]),
                        transplant_error_rms=rms(ti[n]-scalar_i[n]),restore_ratio=ratio(rms(ri[n]),denominator),
                        transplant_error_ratio=ratio(rms(ti[n]-scalar_i[n]),denominator)))
    assert numerical<TOL
    for filename,rows in [('baseline.csv',baseline_rows),('methods.csv',method_rows),
            ('per_input.csv',input_rows),('generation.csv',generation_rows),('stages.csv',stage_rows)]:
        write_csv(RUN/filename,rows)
    summary=dict(status='PASS',utc=now(),baseline=baseline_rows,methods=method_rows,
        generation=generation_rows,stage_statistics=stage_rows,independent_output_decomposition_error=numerical,
        natural_output_vectors=6*64*2*16,patched_output_vectors=6*64*2*16*21*2,
        teachers=6,distinct_evaluation_images=64,calibration_model_access=False,new_teacher_training_runs=0)
    dump(summary_path,summary)
    event('summary_complete',input_metric_rows=len(input_rows),method_rows=len(method_rows),numeric_error=numerical)


def verify_all():
    """只读回查所有完成文件和旧研究原件，检查冻结时间及汇总记录完整。"""
    verify_freeze()
    assert read(RUN/'freeze.json')['utc'] < read(RUN/'evaluation_opened.json')['utc']
    summary=read(RUN/'summary.json')
    assert len(summary['baseline'])==12 and len(summary['methods'])==252
    for seed in SEEDS:
        verify_completion(RUN/'development'/f'seed_{seed}')
        for alpha in ALPHAS: verify_completion(RUN/'evaluation'/f'seed_{seed}'/f'alpha_{alpha}')
    preserved=read(RUN/'protocol.json')['preserved_old_delivery']
    for entry in preserved: assert sha(BASE/entry['path'])==entry['sha256'],entry['path']
    print(json.dumps(dict(status='PASS',old_delivery_files_preserved=len(preserved),
        completed_evaluation_cases=12,summary_method_rows=252,calibration_model_access=False)),flush=True)


def main():
    """显式选择一个阶段；不会自动重训或隐式打开评价数据。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['init','develop','freeze','evaluate','summarize','verify'])
    args=parser.parse_args();device=configure()
    if args.stage=='init': initialize(device)
    elif args.stage=='develop': develop(device)
    elif args.stage=='freeze': freeze()
    elif args.stage=='evaluate': evaluate(device)
    elif args.stage=='summarize': summarize()
    else: verify_all()


if __name__=='__main__': main()
