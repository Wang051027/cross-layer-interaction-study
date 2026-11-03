"""开发数据拟合因素规则，冻结后在未用过的dSprites像素上评价。

阶段顺序：init → fit → freeze → select → seal → evaluate → summarize。
旧运行只读；新的日志、模型参数、原始输出和汇总均写入独立目录。
运行示例：python -B scripts/run_dsprites_factor_transfer.py --stage init
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import time
from pathlib import Path

import dsprites_teacher as teacher  # 先配置确定性环境，再导入torch。
import numpy as np
import torch
import dsprites_mechanism as mechanism
import factor_mediator as factors_model
from test_dsprites_mechanism import check_model, known_answers as mechanism_answers

BASE, OLD, DATA = teacher.BASE, teacher.RUN, teacher.DATA
RUN = BASE / 'runs/dsprites_factor_transfer_20261007'
FULL = BASE / 'data/dsprites'
SEEDS = teacher.SEEDS
STRENGTHS = (1., .25, .5, .75)  # 主评价先运行全强度，三种补充强度复用同一名单。
CODE = ('scripts/factor_mediator.py', 'scripts/run_dsprites_factor_transfer.py')
DEPENDENCIES = ('scripts/dsprites_teacher.py', 'scripts/dsprites_mechanism.py',
                'scripts/cross_layer_mechanism.py', 'scripts/test_dsprites_mechanism.py',
                'scripts/residual_teacher.py', 'scripts/external_experiment.py',
                'scripts/research_runtime.py')


def write(path, value):
    """只写UTF-8数值JSON；不允许隐藏NaN或无穷值。"""
    teacher.dump(path, value)


def event(kind, **details):
    """新的运行拥有独立日志，不能调用旧模块中绑定旧目录的日志函数。"""
    value = dict(utc=teacher.now(), event=kind, **details)
    with (RUN / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')
    print(json.dumps(value, ensure_ascii=False), flush=True)


def verify_protocol():
    """每个阶段核对源码、教师、开发响应和旧运行的承诺文件。"""
    p = teacher.read(RUN / 'protocol.json')
    assert teacher.sha(RUN / 'protocol.json') == (RUN / 'protocol.sha256').read_text().strip()
    for name, expected in p['sources'].items():
        assert teacher.sha(BASE / name) == expected, name
        assert teacher.sha(RUN / 'source' / name) == expected, name
    for name, expected in p['inputs'].items():
        assert teacher.sha(BASE / name) == expected, name
    return p


def finish(folder, names, seconds):
    """完成标记记录文件哈希，中断后的残留目录不会被自动覆盖。"""
    write(folder / 'completion.json', dict(utc=teacher.now(), seconds=seconds,
          files={name: teacher.sha(folder / name) for name in names}))


def verify_complete(folder):
    """断点继续只接受已经完成且哈希一致的阶段。"""
    for name, expected in teacher.read(folder / 'completion.json')['files'].items():
        assert teacher.sha(folder / name) == expected, str(folder / name)


def freeze(name, stage):
    """全部六教师的名单一起封存，再开启下一阶段，阻止按新结果回调规则。"""
    verify_protocol()
    assert not (RUN / f'{name}.json').exists()
    files = {}
    folders = [RUN / stage] if stage == 'panel' else [RUN / stage / f'seed_{s}' for s in SEEDS]
    for folder in folders:
        verify_complete(folder)
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                files[str(path.relative_to(RUN)).replace('\\', '/')] = teacher.sha(path)
    write(RUN / f'{name}.json', dict(utc=teacher.now(), protocol_sha256=teacher.sha(RUN / 'protocol.json'), files=files))
    (RUN / f'{name}.sha256').write_text(teacher.sha(RUN / f'{name}.json') + '\n', encoding='ascii')
    event(name, files=len(files))


def verify_freeze(name):
    """封存后的参数和选择文件每次用前只读核验。"""
    verify_protocol()
    p = teacher.read(RUN / f'{name}.json')
    assert teacher.sha(RUN / f'{name}.json') == (RUN / f'{name}.sha256').read_text().strip()
    for name, expected in p['files'].items():
        assert teacher.sha(RUN / name) == expected, name


def official_ids(scale, angle):
    """按官方因素笛卡尔顺序，在中心位置取三类为一组。"""
    return (((np.arange(3) * 6 + scale) * 40 + angle) * 32 + 15) * 32 + 15


def digest(pixel):
    """排除像素完全相同的不同因素标签，包括形状旋转对称造成的重复。"""
    return hashlib.sha256(np.ascontiguousarray(pixel).tobytes()).hexdigest()


def prepare_panel():
    """不查询模型：从像素不重叠的共同角度中选择覆盖四个方向的四组。

覆盖代价按物理弧度计算周期距离；浮点代价先舍入12位，再按角度索引
字典序打破并列。所有大小使用同一组角度，使同类的大小比较可以配对。
    """
    folder = RUN / 'panel'; assert not folder.exists(); folder.mkdir()
    started = time.perf_counter()
    with np.load(DATA / 'indices.npz', allow_pickle=False) as arrays:
        used = set(map(int, np.concatenate([arrays[n] for n in ('train', 'validation', 'evaluation')])))
        development_ids = arrays['development'].copy()
    previous = set()
    split_counts = {}
    for split in ('train', 'validation', 'evaluation', 'translated'):
        pixels = np.load(DATA / f'{split}_images.npy', mmap_mode='r', allow_pickle=False)
        hashes = {digest(x) for x in pixels}; previous.update(hashes)
        split_counts[split] = dict(rows=len(pixels), unique_pixels=len(hashes))
    images = np.load(FULL / 'imgs.npy', mmap_mode='r', allow_pickle=False)
    physical = np.load(FULL / 'latents_values.npy', mmap_mode='r', allow_pickle=False)
    latent = np.load(FULL / 'latents_classes.npy', mmap_mode='r', allow_pickle=False)
    eligible = []
    for scale in range(6):
        keep = []
        for angle in range(40):
            ids = official_ids(scale, angle)
            if not set(map(int, ids)) & used and not {digest(images[i]) for i in ids} & previous:
                keep.append(angle)
        eligible.append(keep)
    common = sorted(set.intersection(*(set(x) for x in eligible)))
    anchors = np.arange(4) * np.pi / 2
    candidates = []
    for angles in itertools.combinations(common, 4):
        ids = np.concatenate([official_ids(s, a) for s in range(6) for a in angles])
        hashes = [digest(images[i]) for i in ids]
        if len(set(hashes)) != len(ids):
            continue
        theta = np.asarray([physical[official_ids(0, a)[0], 3] for a in angles])
        cost = min(float(np.square(np.angle(np.exp(1j * (theta[list(order)] - anchors)))).sum())
                   for order in itertools.permutations(range(4)))
        candidates.append((round(cost, 12), angles))
    assert candidates, 'No balanced pixel-disjoint panel'
    cost, angles = min(candidates)
    ids = np.concatenate([official_ids(s, a) for s in range(6) for a in angles])
    hashes = [digest(images[i]) for i in ids]
    assert len(ids) == 72 and len(set(hashes)) == 72 and not set(hashes) & previous
    assert not set(map(int, ids)) & used
    np.savez_compressed(folder / 'data.npz', images=np.asarray(images[ids]), labels=latent[ids, 1],
        factors=np.asarray(latent[ids]), physical=np.asarray(physical[ids]), official_ids=ids,
        development_physical=np.asarray(physical[development_ids]))
    write(folder / 'audit.json', dict(status='PASS', chosen_angles=list(angles), common_angles=common,
        eligible_by_scale=eligible, coverage_cost=cost, candidate_panels=len(candidates), inputs=72,
        unique_pixels=72, previous_splits=split_counts, duplicate_pixels_with_previous=0,
        official_id_overlap=0, hashes=hashes,
        scope='new central pixels and factor combinations; not angles globally excluded from training'))
    finish(folder, ['data.npz', 'audit.json'], time.perf_counter() - started)
    event('panel_prepared', inputs=72, angles=list(angles))


def initialize(device):
    """先写协议，再生成面板；已看过旧评价结果，明确记录本轮为接续检验。"""
    assert not RUN.exists(), 'Preserve existing run'
    teacher.verify_protocol()
    for manifest in ('model_freeze', 'fixed_freeze'):
        for name, expected in teacher.read(OLD / f'{manifest}.json')[('checkpoints' if manifest == 'model_freeze' else 'files')].items():
            assert teacher.sha(OLD / name) == expected, name
    inputs = {}
    paths = [DATA / 'manifest.json', DATA / 'indices.npz', DATA / 'development_factors.npy',
             OLD / 'delivery_manifest.json', OLD / 'model_freeze.json', OLD / 'fixed_freeze.json']
    paths += [DATA / f'{split}_images.npy' for split in ('train', 'validation', 'evaluation', 'translated')]
    paths += [OLD / f'development/seed_{s}/{name}' for s in SEEDS for name in ('discovery.npz', 'fixed_orders.json')]
    paths += [OLD / f'seed_{s}/teacher.pt' for s in SEEDS]
    paths += [FULL / name for name in ('imgs.npy', 'latents_values.npy', 'latents_classes.npy')]
    inputs = {str(p.relative_to(BASE)).replace('\\', '/'): teacher.sha(p) for p in paths}
    RUN.mkdir()
    sources = {p: teacher.sha(BASE / p) for p in (*CODE, *DEPENDENCIES)}
    for name in sources:
        destination = RUN / 'source' / name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASE / name, destination)
    write(RUN / 'protocol.json', dict(utc=teacher.now(), version='factor-transfer-v1', seeds=SEEDS,
        previous_evaluation_inspected=True, development_only_fitting=True, no_retraining=True,
        inputs=inputs, sources=sources, primary=dict(alpha=1., budget=16, target='margin',
            comparison='class_factor_ridge versus class_fixed', outcomes=['restore_ratio', 'transplant_error_ratio']),
        strengths=list(STRENGTHS), targets=list(mechanism.TARGETS), rules=list(factors_model.RULES),
        ridge_lambda_grid=list(factors_model.LAMBDAS), cv='leave one whole development angle out; pooled raw singleton score MSE',
        budget=16, selection_alpha=.5, neighbors=4, known_factors=True,
        factors='true shape class, size index, official physical orientation radians',
        panel='6 sizes x 4 shared orientations x 3 shapes at (15,15); exclude old train/validation/evaluation/translation pixels',
        panel_choice='min cyclic squared cost to quarter-circle anchors; round cost to12 decimals then lexicographic angle tuple',
        random_references=4, shortlist=64, reuse_old_sources_and_candidate_pools=True,
        input_reference='64 singleton restore/transplant queries at alpha .5; same lists at all strengths',
        half_strength_is_selection_condition=True, ratio_floor=1e-9,
        statistics='aggregate interaction MSE within teacher; six paired teacher results; all candidates reported',
        test_scope='same architecture and trained teachers; new pixels, not rotation OOD',
        environment=dict(torch=torch.__version__, numpy=np.__version__, device=device,
                         gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None)))
    (RUN / 'protocol.sha256').write_text(teacher.sha(RUN / 'protocol.json') + '\n', encoding='ascii')
    event('protocol_locked', sha256=teacher.sha(RUN / 'protocol.json'))
    prepare_panel(); freeze('panel_freeze', 'panel')


def specs(pool_size):
    """两个目标各六个无需新模型查询的规则，另加逐图参照和数值端点。"""
    result = [dict(method=f'{target}_{rule}', budget=16) for target in mechanism.TARGETS for rule in factors_model.RULES]
    result += [dict(method=f'{target}_input_finite', budget=16) for target in mechanism.TARGETS]
    result += [dict(method=f'random_{i}', budget=16) for i in range(4)]
    return result + [dict(method='all_shortlist', budget=64), dict(method='all_reachable', budget=int(pool_size))]


def fit(device):
    """拟合、调参和预测新面板名单只读取旧开发输出及新面板的因素标签。"""
    verify_freeze('panel_freeze'); assert not (RUN / 'selection_opened.json').exists()
    with np.load(RUN / 'panel/data.npz', allow_pickle=False) as a:
        new_factors, physical, dev_physical = (a[k].copy() for k in ('factors', 'physical', 'development_physical'))
    dev_factors = np.load(DATA / 'development_factors.npy', allow_pickle=False)
    validation = np.load(DATA / 'validation_images.npy', mmap_mode='r', allow_pickle=False)
    val_labels = np.load(DATA / 'validation_labels.npy', allow_pickle=False)
    for seed in SEEDS:
        folder = RUN / 'fitted' / f'seed_{seed}'
        if folder.exists(): verify_complete(folder); continue
        folder.mkdir(parents=True); started = time.perf_counter()
        with np.load(OLD / f'development/seed_{seed}/discovery.npz', allow_pickle=False) as a:
            score, pool, shortlist, nodes = (a[k].copy() for k in ('finite_scores', 'pool', 'shortlist', 'source_nodes'))
        fixed = teacher.read(OLD / f'development/seed_{seed}/fixed_orders.json')
        models, diagnostics, cv, predictions, selections = {}, {}, {}, [], []
        for ti, target in enumerate(mechanism.TARGETS):
            models[target], diagnostics[target], target_cv = factors_model.train_rules(dev_factors, dev_physical, score[:, ti])
            for rule, values in target_cv.items(): cv[f'{target}_{rule}'] = values
            for rule in factors_model.RULES:
                pred = factors_model.predict_rule(models[target], new_factors, physical, rule)
                chosen = factors_model.top_coordinates(pred, pool, shortlist)
                predictions.append(pred); selections.append(chosen)
                if rule in ('global_fixed', 'class_fixed'):
                    order = fixed[f"{target}_{'global' if rule == 'global_fixed' else 'class'}_finite"]
                    expected = [pool[np.asarray(order if rule == 'global_fixed' else order[int(c)])[:16]] for c in new_factors[:, 1]]
                    assert np.array_equal(chosen, expected), 'Changed original baseline'
        for i in range(4):
            selections.append(np.repeat(pool[np.asarray(fixed[f'random_{i}'])[:16]][None], 72, axis=0))
        # 这里的名单顺序先是12个因素规则，再4个随机规则；逐图参照下一阶段插入。
        model = teacher.load_teacher(seed, device)
        checks = check_model(model, torch.as_tensor(np.array(validation[0])[None], dtype=torch.float64),
                             int(val_labels[0]), nodes, device)
        write(folder / 'models.json', models); write(folder / 'cv.json', diagnostics)
        write(folder / 'checks.json', dict(factor_controls=factors_model.known_answers(),
                                         mechanism_controls=mechanism_answers(), model_hooks=checks))
        np.savez_compressed(folder / 'predictions.npz', scores=np.stack(predictions, axis=1),
            selections=np.stack(selections, axis=1), source_nodes=nodes, pool=pool, shortlist=shortlist,
            factors=new_factors, physical=physical)
        np.savez_compressed(folder / 'cv_predictions.npz', **cv)
        finish(folder, ['models.json', 'cv.json', 'checks.json', 'predictions.npz', 'cv_predictions.npz'], time.perf_counter() - started)
        event('factor_rules_fitted', seed=seed, seconds=time.perf_counter() - started,
              margin_factor_lambda=diagnostics['margin']['class_factor_ridge']['chosen'])
        del model


def select(device):
    """逐图参照只在半强度查询64个单点，然后封存；不回调因素模型。"""
    verify_freeze('fixed_freeze'); assert not (RUN / 'evaluation_opened.json').exists()
    if not (RUN / 'selection_opened.json').exists():
        write(RUN / 'selection_opened.json', dict(utc=teacher.now(), alpha=.5))
    with np.load(RUN / 'panel/data.npz', allow_pickle=False) as a:
        images, labels = a['images'].copy(), a['labels'].copy()
    bits, stencil, meta = mechanism.design(); bits = torch.as_tensor(bits, dtype=torch.float64, device=device)
    for seed in SEEDS:
        folder = RUN / 'input_selection' / f'seed_{seed}'
        if folder.exists(): verify_complete(folder); continue
        folder.mkdir(parents=True); started = time.perf_counter()
        with np.load(RUN / f'fitted/seed_{seed}/predictions.npz', allow_pickle=False) as a:
            predetermined, nodes, pool, shortlist = (a[k].copy() for k in ('selections', 'source_nodes', 'pool', 'shortlist'))
        model = teacher.load_teacher(seed, device)
        natural, singleton, finite, choices = [], [], [], []
        for i, (image, label) in enumerate(zip(images, labels)):
            x = torch.as_tensor(image[None, None], dtype=torch.float64, device=device)
            source = mechanism.source_activation(model, x)
            tr = mechanism.trace(model, mechanism.changed_sources(source, nodes, bits, .5))
            raw = mechanism.patch(model, tr['h'][1], [[int(x)] for x in pool[shortlist]]).cpu().numpy()
            original = tr['output'].cpu().numpy()
            scores = mechanism.finite_scores(raw, original, int(label), stencil)
            selected = list(predetermined[i, :12])
            selected += [factors_model.top_coordinates(scores[ti], pool, shortlist) for ti in range(2)]
            selected += list(predetermined[i, 12:]) + [pool[shortlist], pool]
            choices.append(mechanism.encode(selected, len(pool)))
            natural.append(original); singleton.append(raw); finite.append(scores)
            if (i + 1) % 12 == 0: event('selection_progress', seed=seed, inputs=i + 1, total=72)
        np.savez_compressed(folder / 'selection.npz', natural=np.stack(natural), singleton_logits=np.stack(singleton),
            finite_scores=np.stack(finite), selections=np.stack(choices), labels=labels,
            source_nodes=nodes, pool=pool, shortlist=shortlist, stencil=stencil, pair_metadata=meta)
        write(folder / 'specs.json', specs(len(pool)))
        write(folder / 'cost.json', dict(selection_prefix_inputs=72, selection_natural_tail_outputs=72 * 16,
            selection_singleton_tail_outputs=72 * 64 * 2 * 16, singleton_outputs_per_image=2048,
            factor_rules_new_model_queries=0, old_development_outputs_reused=True,
            seconds=time.perf_counter() - started, timing='total selection stage; not a comparative speed benchmark'))
        finish(folder, ['selection.npz', 'specs.json', 'cost.json'], time.perf_counter() - started)
        event('selection_complete', seed=seed, seconds=time.perf_counter() - started)
        del model


def evaluate(device):
    """冻结名单在四种强度下双向修补，保留原始logits及逐层恒等式检查。"""
    verify_freeze('fixed_freeze'); verify_freeze('adaptive_freeze')
    if not (RUN / 'evaluation_opened.json').exists():
        write(RUN / 'evaluation_opened.json', dict(utc=teacher.now(), primary_alpha=1.))
    with np.load(RUN / 'panel/data.npz', allow_pickle=False) as a:
        images, labels = a['images'].copy(), a['labels'].copy()
    bits, stencil, meta = mechanism.design(); bits = torch.as_tensor(bits, dtype=torch.float64, device=device)
    for seed in SEEDS:
        with np.load(RUN / f'input_selection/seed_{seed}/selection.npz', allow_pickle=False) as a:
            selected, nodes, half = (a[k].copy() for k in ('selections', 'source_nodes', 'natural'))
        current_specs = teacher.read(RUN / f'input_selection/seed_{seed}/specs.json')
        model = teacher.load_teacher(seed, device)
        for alpha in STRENGTHS:
            folder = RUN / 'evaluation' / f'seed_{seed}' / f'alpha_{alpha}'
            if folder.exists(): verify_complete(folder); continue
            folder.mkdir(parents=True); started = time.perf_counter()
            natural, patched, parts, stages, maxima = [], [], [], [], {}
            for i, image in enumerate(images):
                x = torch.as_tensor(image[None, None], dtype=torch.float64, device=device)
                source = mechanism.source_activation(model, x)
                tr = mechanism.trace(model, mechanism.changed_sources(source, nodes, bits, alpha))
                selections = [selected[i, j, :s['budget']] for j, s in enumerate(current_specs)]
                original = tr['output'].cpu().numpy()
                output = mechanism.patch(model, tr['h'][1], selections).cpu().numpy()
                contribution, layer, errors = mechanism.decomposition(model, tr, stencil)
                errors['all_restore'] = float(np.max(np.abs(output[-1, 0] - original[:1])))
                errors['all_transplant'] = float(np.max(np.abs(output[-1, 1] - original)))
                if alpha == .5: errors['half_replay'] = float(np.max(np.abs(original - half[i])))
                assert max(errors.values()) < 1e-9, errors
                for key, value in errors.items(): maxima[key] = max(maxima.get(key, 0.), float(value))
                natural.append(original); patched.append(output); parts.append(contribution.cpu().numpy()); stages.append(layer)
                if (i + 1) % 12 == 0: event('evaluation_progress', seed=seed, alpha=alpha, inputs=i + 1, total=72)
            np.savez_compressed(folder / 'responses.npz', natural=np.stack(natural), patched=np.stack(patched),
                contributions=np.stack(parts), stage_statistics=np.stack(stages), labels=labels,
                source_nodes=nodes, stencil=stencil, pair_metadata=meta)
            write(folder / 'checks.json', dict(status='PASS', errors=maxima))
            finish(folder, ['responses.npz', 'checks.json'], time.perf_counter() - started)
            event('evaluation_complete', seed=seed, alpha=alpha, seconds=time.perf_counter() - started,
                  max_error=max(maxima.values()))
        del model


def summarize():
    """先在模型内汇总MSE，再报告六个教师；不把24个差分当独立样本。"""
    verify_freeze('adaptive_freeze'); assert not (RUN / 'summary.json').exists()
    with np.load(RUN / 'panel/data.npz', allow_pickle=False) as a:
        factors = a['factors'].copy()
    rows, baselines = [], []
    for seed in SEEDS:
        specifications = teacher.read(RUN / f'input_selection/seed_{seed}/specs.json')
        for alpha in STRENGTHS:
            folder = RUN / f'evaluation/seed_{seed}/alpha_{alpha}'; verify_complete(folder)
            with np.load(folder / 'responses.npz', allow_pickle=False) as a:
                natural, patched, labels, stencil = (a[k].copy() for k in ('natural', 'patched', 'labels', 'stencil'))
            for target in mechanism.TARGETS:
                n, r, t = mechanism.metric_arrays(natural, patched, labels, stencil, target)
                np.savez_compressed(folder / f'metrics_{target}.npz', natural_mse=n, restore_mse=r,
                    transplant_error_mse=t, labels=labels, factors=factors)
                strata = [('all', -1, np.ones(len(labels), dtype=bool))]
                strata += [('class', c, labels == c) for c in range(3)]
                strata += [('scale', s, factors[:, 2] == s) for s in range(6)]
                strata += [('angle', int(a), factors[:, 3] == a) for a in np.unique(factors[:, 3])]
                for stratum, value, mask in strata:
                    denominator = float(np.sqrt(n[mask].mean()))
                    baselines.append(dict(seed=seed, alpha=alpha, target=target, stratum=stratum, value=value,
                        interaction_rms=denominator, inputs=int(mask.sum()),
                        accuracy=float((natural[mask, 0].argmax(1) == labels[mask]).mean()),
                        source_class_changes=int((natural[mask].argmax(-1) != natural[mask, 0].argmax(-1)[:, None]).sum())))
                    for j, spec in enumerate(specifications):
                        restore, error = float(np.sqrt(r[mask, j].mean())), float(np.sqrt(t[mask, j].mean()))
                        rr, tt = (restore / denominator, error / denominator) if denominator > 1e-9 else (None, None)
                        rows.append(dict(seed=seed, alpha=alpha, target=target, stratum=stratum, value=value, **spec,
                            interaction_rms=denominator, restore_rms=restore, transplant_error_rms=error,
                            restore_ratio=rr, transplant_error_ratio=tt,
                            joint_loss=None if rr is None else .5 * (rr ** 2 + tt ** 2)))
    write(RUN / 'summary.json', dict(status='PASS', utc=teacher.now(), teachers=6, inputs=72,
        strengths=list(STRENGTHS), methods=rows, baseline=baselines,
        metadata_is_oracle=True, half_strength_is_selection_condition=True))
    event('summary_complete', method_rows=len(rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['init', 'fit', 'freeze', 'select', 'seal', 'evaluate', 'summarize'])
    stage = parser.parse_args().stage
    device = teacher.configure()
    if stage == 'init': initialize(device)
    elif stage == 'fit': fit(device)
    elif stage == 'freeze':
        assert not (RUN / 'selection_opened.json').exists(); freeze('fixed_freeze', 'fitted')
    elif stage == 'select': select(device)
    elif stage == 'seal':
        assert not (RUN / 'evaluation_opened.json').exists(); verify_freeze('fixed_freeze'); freeze('adaptive_freeze', 'input_selection')
    elif stage == 'evaluate': evaluate(device)
    else: summarize()
