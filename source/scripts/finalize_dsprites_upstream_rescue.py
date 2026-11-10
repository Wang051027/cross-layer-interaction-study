"""汇总上游基准定位与恢复实验的文件哈希。"""
from __future__ import annotations
import json
import shutil
import dsprites_teacher as teacher
from run_dsprites_upstream_rescue import BASE, RUN, verify_protocol, verify_selection, verify_folder
from report_dsprites_upstream_rescue import REPORT

CODE = ('upstream_rescue_mechanism.py', 'upstream_rescue_measure.py', 'run_dsprites_upstream_rescue.py',
        'test_dsprites_upstream_rescue.py', 'audit_dsprites_upstream_rescue.py', 'report_dsprites_upstream_rescue.py',
        'finalize_dsprites_upstream_rescue.py', 'verify_dsprites_upstream_rescue_delivery.py')


def finalize():
    """把原始数据、排名、图表、代码和日期进度快照一起计算清单。"""
    assert not (RUN/'delivery_manifest.json').exists(); verify_protocol(); verify_selection()
    analysis = teacher.read(RUN/'analysis.json'); audit = teacher.read(RUN/'independent_audit.json')
    assert analysis['status'] == audit['status'] == 'PASS' and audit['endpoint_fields'] == 76032
    assert teacher.read(RUN/'claim_checks.json')['status'] == 'PASS'
    visual = teacher.read(RUN/'visual_review.json'); assert visual['status'] == 'PASS'
    assert [r['path'] for r in visual['files']] == analysis['figures']
    for row in visual['files']: assert teacher.sha(BASE/row['path']) == row['sha256']
    assert REPORT.exists() and (RUN/'README.md').exists() and len(analysis['figures']) == 8
    assert not (RUN/'.plot_cache').exists()
    for seed in teacher.SEEDS:
        verify_folder(RUN/f'discovery/seed_{seed}'); verify_folder(RUN/f'evaluation/seed_{seed}')
    copies = [(BASE/'scripts'/n, RUN/'source/scripts'/n) for n in CODE]
    copies += [(REPORT, RUN/'source/reports'/REPORT.name), (BASE/'CURRENT_RESEARCH.md', RUN/'source/CURRENT_RESEARCH.md')]
    copies += [(BASE/n, RUN/'source/reports/figures'/n.split('/')[-1]) for n in analysis['figures']]
    for origin, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists(): assert teacher.sha(origin) == teacher.sha(destination)
        else: shutil.copy2(origin, destination)
    paths = [p for p in RUN.rglob('*') if p.is_file()]
    paths += [BASE/'scripts'/n for n in CODE]+[REPORT]+[BASE/n for n in analysis['figures']]
    paths += [BASE/'reports/dsprites_upstream_rescue_preflight_20261017.json']
    records = []
    for p in sorted(set(paths)):
        assert p.resolve().is_relative_to(BASE.resolve())
        records.append(dict(path=str(p.relative_to(BASE)).replace('\\', '/'), bytes=p.stat().st_size, sha256=teacher.sha(p)))
    teacher.dump(RUN/'delivery_manifest.json', dict(status='PASS', utc=teacher.now(), version='upstream-rescue-delivery-v1',
                 files=records, old_delivery_files=audit['old_delivery_files'],
                 mutable_progress='live CURRENT_RESEARCH.md excluded;dated snapshot included'))
    digest = teacher.sha(RUN/'delivery_manifest.json'); (RUN/'delivery_manifest.sha256').write_text(digest+'\n', encoding='ascii')
    return dict(status='PASS', files=len(records), bytes=sum(r['bytes'] for r in records), manifest_sha256=digest)


if __name__ == '__main__': print(json.dumps(finalize()))
