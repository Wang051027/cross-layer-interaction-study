"""核对因素迁移实验的清单和代码快照。--deep-inputs 额外检查原始输入和源码哈希。"""
from pathlib import Path
import argparse
import hashlib
import json

BASE = Path(__file__).resolve().parents[1]
RUN = BASE / 'runs/dsprites_factor_transfer_20261007'


def sha(path):
    """分块读取文件并计算 SHA-256。"""
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    """读取JSON数据，清单内容不作为可执行代码。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify(deep=False):
    """清单、独立审计、三个冻结阶段与可选原始依赖都必须一致。"""
    manifest = RUN / 'delivery_manifest.json'
    assert sha(manifest) == (RUN / 'delivery_manifest.sha256').read_text().strip()
    value = read(manifest); assert value['status'] == 'PASS'
    for entry in value['files']:
        assert sha(BASE / entry['path']) == entry['sha256'], entry['path']
    assert read(RUN / 'independent_audit.json')['status'] == 'PASS'
    assert read(RUN / 'fixed_freeze.json')['utc'] < read(RUN / 'selection_opened.json')['utc']
    assert read(RUN / 'adaptive_freeze.json')['utc'] < read(RUN / 'evaluation_opened.json')['utc']
    for name in ('panel_freeze', 'fixed_freeze', 'adaptive_freeze'):
        path = RUN / f'{name}.json'
        assert sha(path) == (RUN / f'{name}.sha256').read_text().strip()
        for relative, expected in read(path)['files'].items():
            assert sha(RUN / relative) == expected, relative
    if deep:
        protocol = read(RUN / 'protocol.json')
        for relative, expected in {**protocol['inputs'], **protocol['sources']}.items():
            assert sha(BASE / relative) == expected, relative
        preservation = read(RUN / 'legacy_preservation.json')
        for relative, expected in preservation['manifests'].items():
            path = BASE / relative; assert sha(path) == expected, relative
            previous = read(path)
            # 早期清单的files是计数，文件列表在entries；后期清单用files列表。
            entries = previous.get('entries')
            if not isinstance(entries, list): entries = previous.get('files')
            assert isinstance(entries, list), relative
            for entry in entries:
                assert sha(BASE / entry['path']) == entry['sha256'], entry['path']
    print(json.dumps(dict(status='PASS', files=len(value['files']), deep_inputs=deep), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deep-inputs', action='store_true')
    verify(parser.parse_args().deep_inputs)
