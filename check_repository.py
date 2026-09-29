"""只用标准库检查本次发布的文件哈希与 Python 语法，不运行历史实验。"""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def main():
    """清单自身不自引用；Git 提交另外记录清单文件的版本。"""
    manifest = json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        path = (ROOT/entry['path']).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Manifest path escapes repository')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != entry['sha256']:
            raise ValueError(f'File changed: {entry["path"]}')
    files = list((ROOT/'source').rglob('*.py')) + list(ROOT.glob('*.py'))
    for path in files:
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=path.name)
    result = dict(status='PASS', checked_files=len(manifest['files']), python_files=len(files),
                  scope='Release checksums and syntax, not historical experiment reproduction')
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == '__main__':
    main()
