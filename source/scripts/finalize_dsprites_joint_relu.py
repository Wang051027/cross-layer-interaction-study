"""封存本轮独立交付；仅CURRENT_RESEARCH.md的日期快照进入清单。"""
from __future__ import annotations
import json
import shutil
import dsprites_teacher as teacher
from run_dsprites_joint_relu import BASE, RUN, verify_seal, verify_folder

REPORT = BASE/'reports/dSprites_多通道联合ReLU的稳定性与输出作用_2026-10-17.md'
CODE = ('joint_relu_mechanism.py', 'run_dsprites_joint_relu.py', 'test_dsprites_joint_relu.py',
        'audit_dsprites_joint_relu.py', 'report_dsprites_joint_relu.py',
        'finalize_dsprites_joint_relu.py', 'verify_dsprites_joint_relu_delivery.py')


def finalize():
    assert not (RUN/'delivery_manifest.json').exists(); verify_seal()
    analysis = teacher.read(RUN/'analysis.json'); audit = teacher.read(RUN/'independent_audit.json')
    assert analysis['status'] == audit['status'] == 'PASS' and audit['endpoints'] == 1728
    assert len(analysis['overview_effects']) == 48 and len(analysis['figures']) == 8
    assert REPORT.exists() and (RUN/'README.md').exists()
    for seed in teacher.SEEDS:
        for alpha in (.5, 1.): verify_folder(RUN/f'evaluation/seed_{seed}/alpha_{alpha}')
    copies = [(BASE/'scripts'/name, RUN/'source/scripts'/name) for name in CODE]
    copies += [(REPORT, RUN/'source/reports'/REPORT.name), (BASE/'CURRENT_RESEARCH.md', RUN/'source/CURRENT_RESEARCH.md')]
    copies += [(BASE/name, RUN/'source/reports/figures'/name.split('/')[-1]) for name in analysis['figures']]
    for origin, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists(): assert teacher.sha(origin) == teacher.sha(destination), destination
        else: shutil.copy2(origin, destination)
    paths = [p for p in RUN.rglob('*') if p.is_file() and '.plot_cache' not in p.parts]
    paths += [BASE/'scripts'/name for name in CODE] + [REPORT] + [BASE/name for name in analysis['figures']]
    paths += [BASE/'reports/dsprites_joint_relu_preflight_20261017.json']
    records = []
    for path in sorted(set(paths)):
        assert path.resolve().is_relative_to(BASE.resolve())
        records.append(dict(path=str(path.relative_to(BASE)).replace('\\', '/'), bytes=path.stat().st_size, sha256=teacher.sha(path)))
    teacher.dump(RUN/'delivery_manifest.json', dict(status='PASS', version='joint-relu-stability-delivery-v1', utc=teacher.now(),
                 files=records, old_delivery_files=audit['old_delivery_files'],
                 mutable_progress='live CURRENT_RESEARCH.md excluded;dated source snapshot included'))
    digest = teacher.sha(RUN/'delivery_manifest.json')
    (RUN/'delivery_manifest.sha256').write_text(digest+'\n', encoding='ascii')
    return dict(status='PASS', files=len(records), bytes=sum(r['bytes'] for r in records), manifest_sha256=digest)


if __name__ == '__main__': print(json.dumps(finalize()))
