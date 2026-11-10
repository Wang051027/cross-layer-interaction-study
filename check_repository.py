from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def main():
    """按清单验证文件哈希并解析 Python 源码。"""
    manifest = json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
    python_files = 0
    for entry in manifest['files']:
        path = (ROOT/entry['path']).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Manifest path escapes repository')
        contents = path.read_bytes()
        actual = hashlib.sha256(contents).hexdigest()
        if actual != entry['sha256']:
            raise ValueError(f'File changed: {entry["path"]}')
        if path.suffix == '.py':
            ast.parse(contents.decode('utf-8-sig'), filename=entry['path'])
            python_files += 1
    result = dict(status='PASS', checked_files=len(manifest['files']), python_files=python_files,
                  scope='Release checksums and Python syntax')
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == '__main__':
    main()
