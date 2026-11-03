"""封存A/R四格实验，不修改九份历史交付。"""
from __future__ import annotations
import json
import shutil
import dsprites_teacher as teacher
from run_dsprites_ar_factorial import BASE, RUN, verify_protocol, verify_folder

REPORT = BASE/'reports/dSprites_基准与组合残差的分离控制_2026-10-17.md'
CODE = ('ar_factorial_mechanism.py', 'run_dsprites_ar_factorial.py', 'test_dsprites_ar_factorial.py',
        'audit_dsprites_ar_factorial.py', 'report_dsprites_ar_factorial.py',
        'finalize_dsprites_ar_factorial.py', 'verify_dsprites_ar_factorial_delivery.py')


def finalize():
    assert not (RUN/'delivery_manifest.json').exists(); verify_protocol()
    analysis = teacher.read(RUN/'analysis.json'); audit = teacher.read(RUN/'independent_audit.json')
    assert analysis['status'] == audit['status'] == 'PASS' and audit['endpoint_fields'] == 6912
    assert teacher.read(RUN/'claim_checks.json')['status'] == 'PASS'
    assert REPORT.exists() and (RUN/'README.md').exists() and len(analysis['figures']) == 8
    for seed in teacher.SEEDS: verify_folder(RUN/f'evaluation/seed_{seed}')
    copies = [(BASE/'scripts'/n, RUN/'source/scripts'/n) for n in CODE]
    copies += [(REPORT, RUN/'source/reports'/REPORT.name), (BASE/'CURRENT_RESEARCH.md', RUN/'source/CURRENT_RESEARCH.md')]
    copies += [(BASE/name, RUN/'source/reports/figures'/name.split('/')[-1]) for name in analysis['figures']]
    for origin, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists(): assert teacher.sha(origin) == teacher.sha(destination)
        else: shutil.copy2(origin, destination)
    paths = [p for p in RUN.rglob('*') if p.is_file() and '.plot_cache' not in p.parts]
    paths += [BASE/'scripts'/n for n in CODE]+[REPORT]+[BASE/name for name in analysis['figures']]
    paths += [BASE/'reports/dsprites_ar_factorial_preflight_20261017.json']
    records = []
    for p in sorted(set(paths)):
        assert p.resolve().is_relative_to(BASE.resolve())
        records.append(dict(path=str(p.relative_to(BASE)).replace('\\', '/'), bytes=p.stat().st_size, sha256=teacher.sha(p)))
    teacher.dump(RUN/'delivery_manifest.json', dict(status='PASS', utc=teacher.now(), version='ar-factorial-delivery-v1',
                 files=records, old_delivery_files=audit['old_delivery_files'],
                 mutable_progress='live CURRENT_RESEARCH.md excluded;dated snapshot included'))
    digest = teacher.sha(RUN/'delivery_manifest.json')
    (RUN/'delivery_manifest.sha256').write_text(digest+'\n', encoding='ascii')
    return dict(status='PASS', files=len(records), bytes=sum(r['bytes'] for r in records), manifest_sha256=digest)


if __name__ == '__main__': print(json.dumps(finalize()))
