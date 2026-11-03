"""封存联合必要性恢复；不改十一份历史交付及旧根文件。"""
from __future__ import annotations
import json
import shutil
import dsprites_teacher as teacher
from run_dsprites_joint_window import BASE, RUN, verify_protocol, verify_selection, verify_folder
from report_dsprites_joint_window import REPORT

CODE = ('joint_window_mechanism.py', 'joint_window_measure.py', 'run_dsprites_joint_window.py', 'test_dsprites_joint_window.py',
        'audit_dsprites_joint_window.py', 'report_dsprites_joint_window.py', 'finalize_dsprites_joint_window.py', 'verify_dsprites_joint_window_delivery.py')


def finalize():
    """只在数值、逐行交付检查和视觉检查完成后封存。"""
    assert not (RUN/'delivery_manifest.json').exists(); verify_protocol(); verify_selection()
    analysis = teacher.read(RUN/'analysis.json'); audit = teacher.read(RUN/'independent_audit.json')
    assert analysis['status'] == audit['status'] == teacher.read(RUN/'claim_checks.json')['status'] == 'PASS'
    assert audit['endpoint_fields'] == 134784 and audit['old_delivery_entries'] == 1558
    review = teacher.read(RUN/'visual_review.json'); assert review['status'] == 'PASS' and len(review['files']) == 8
    assert [r['path'] for r in review['files']] == analysis['figures']
    for r in review['files']: assert teacher.sha(BASE/r['path']) == r['sha256']
    assert REPORT.exists() and (RUN/'README.md').exists() and not (RUN/'.plot_cache').exists()
    for seed in teacher.SEEDS: verify_folder(RUN/f'search/seed_{seed}'); verify_folder(RUN/f'evaluation/seed_{seed}')
    copies = [(BASE/'scripts'/name, RUN/'source/scripts'/name) for name in CODE]
    copies += [(REPORT, RUN/'source/reports'/REPORT.name), (BASE/'CURRENT_RESEARCH.md', RUN/'source/CURRENT_RESEARCH.md')]
    copies += [(BASE/name, RUN/'source/reports/figures'/name.split('/')[-1]) for name in analysis['figures']]
    for origin, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists(): assert teacher.sha(origin) == teacher.sha(target)
        else: shutil.copy2(origin, target)
    paths = [p for p in RUN.rglob('*') if p.is_file()]
    paths += [BASE/'scripts'/n for n in CODE]+[REPORT]+[BASE/n for n in analysis['figures']]+[BASE/'reports/dsprites_joint_window_preflight_20261018.json']
    records = []
    for p in sorted(set(paths)):
        assert p.resolve().is_relative_to(BASE.resolve())
        records.append(dict(path=str(p.relative_to(BASE)).replace('\\', '/'), bytes=p.stat().st_size, sha256=teacher.sha(p)))
    teacher.dump(RUN/'delivery_manifest.json', dict(status='PASS', utc=teacher.now(), files=records, old_delivery_entries=1558,
        mutable_progress='live CURRENT_RESEARCH.md excluded;dated snapshotincluded'))
    digest = teacher.sha(RUN/'delivery_manifest.json'); (RUN/'delivery_manifest.sha256').write_text(digest+'\n', encoding='ascii')
    return dict(status='PASS', files=len(records), bytes=sum(r['bytes'] for r in records), manifest_sha256=digest)


if __name__ == '__main__': print(json.dumps(finalize()))
