"""Package the completed research without changing existing research artifacts."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json
import time
import zipfile

BASE = Path(__file__).resolve().parents[1]
DEST = BASE.parent.parent
DATE = "2026-09-12"
TOP = f"CAT_MT_Research_{DATE}"
VISUAL = Path("visualizations/local-diamond.html")
CHUNK = 8 * 1024 * 1024
PRECOMPRESSED = {".npz", ".gz", ".zip", ".png", ".pdf", ".docx"}
READING_TYPES = {".md", ".json", ".jsonl", ".csv", ".sha256", ".py", ".txt", ".png", ".pdf", ".docx", ".log", ".html"}

# 下面的字符串是旧打包器生成的独立校验程序；保留原文字节，不在字符串内部插入注释。
VERIFY_SCRIPT = '''"""Verify this extracted package using only the Python standard library."""
from pathlib import Path
import hashlib
import json

base = Path(__file__).resolve().parent
manifest = json.loads((base / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
failed = []
for item in manifest["files"]:
    path = (base / item["path"]).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file() or path.is_symlink():
        failed.append(item["path"])
        continue
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != item["sha256"] or path.stat().st_size != item["bytes"]:
        failed.append(item["path"])
print(json.dumps({"status": "failed" if failed else "passed", "files": len(manifest["files"]), "failed": failed}, ensure_ascii=True))
raise SystemExit(1 if failed else 0)
'''


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def event(**value):
    print(json.dumps(value, ensure_ascii=True), flush=True)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


# 根据交付包类型生成配套阅读说明，解释文件范围和复现限制。
def guide(kind):
    opening = ("这是完整研究归档，包含当前研究目录的全部文件，以及教学交互图的 HTML 源码。"
               if kind == "full" else
               "这是轻量阅读版，包含讲解、报告、图表、代码、结果表与审计记录。原始数据、模型权重、大型预测／参考数组及原始 ZIP 不在本阅读版中；完整数值复核请使用完整研究包。")
    return f"""# CAT-MT 研究资料：先读这里

打包日期：{DATE}。研究结果截至 2026-09-11；此次打包没有新增实验或改变研究结论。

{opening}

## 建议阅读顺序

1. [从原始设想到当前实验](cat_mt_followup/reports/从原始设想到当前实验_2026-09-11.md)：从决策树、局部组合和棱形构想到当前背景选择实验的完整讲解。
2. [最新结构对照报告](cat_mt_followup/reports/激活基线与线性读出对照_2026-09-11.md)：激活相关基线、六个 GAP 模型、结构边界与限制。
3. [原六种子正式测试](cat_mt_followup/reports/六种子外部测试与推进建议_2026-09-11.md)：原主比较及完整成本。
4. [项目入口](cat_mt_followup/README.md) 与 [复现运行说明](cat_mt_followup/RUNBOOK.md)。

## 当前完成情况

- 原 v0.7 研究包已保全并复核，271 项原始清单一致。
- Fashion-MNIST 原六个 CNN 教师及后续六个 GAP 教师已完成。
- 原正式干预测试 12 个模型—层设置，后续结构控制 30 个设置。
- 最新一轮执行 3,932,160 个参考输出向量，660 组保存指标回读一致。
- 当前结论针对所规定的内部组合干预；尚未完成全网络语义解释或普遍架构验证。
- 官方测试两个干预批次为 0–255 与 256–511；5000 张校准数据仍未用于模型评价。

## 目录说明

- `cat_mt_followup/reports/`：完整中文讲解、研究报告、科研图。
- `cat_mt_followup/scripts/`：实验、分析、验证与本次打包脚本。
- `cat_mt_followup/runs/`：协议、代码快照、模型与结果；完整包含所有二进制数组。
- `cat_mt_followup/data/`：数据来源、划分及审计；完整包含实际研究数据。
- `cat_mt_followup/upstream/`、`incoming/`：原研究资料；完整包含原始 ZIP。
- `supplements/local-diamond.html`：教学交互图的 HTML 片段源码，依赖外部页面样式，不是实验数据。
- `PACKAGE_MANIFEST.json` 与 `MANIFEST.sha256`：本次归档包含文件的字节数和 SHA-256。

## 完整性与运行环境

将 ZIP 完整解压后，可以在本目录使用 Python 3.11 或更高版本运行：

```text
python -B verify_package.py
```

该检查只需要 Python 标准库。它核对当前包的全部有效载荷；清单文件本身由 ZIP 旁的整体 SHA-256 保护。

研究脚本所需的 Python / PyTorch 等版本见 RUNBOOK。本机虚拟环境不在压缩包中，换电脑需另行准备；RUNBOOK 内的绝对解释器路径记录的是本机执行环境。历史交付检查脚本中的教学图路径也是本机路径；本包的跨机器文件核验使用根目录的 `verify_package.py`。

历史方案、失败后保留的中间运行以及旧状态记录都予以保留。以最新结构对照报告和 `RESEARCH_STATUS.json` 判断当前进度，不覆盖旧冻结记录。
"""


# 收集打包前需保全文件的摘要，打包后再次核对是否改变。
def preserved_hashes():
    expected = {}
    latest = read(BASE / "audit/structure_manifest_20260911.json")
    for item in latest["entries"]:
        expected[item["path"]] = item["sha256"]
    previous = read(BASE / "audit/followup_manifest_20260911.json")
    for item in previous["entries"]:
        if item["path"].startswith("runs/fashion_external_20260911/"):
            expected[item["path"]] = item["sha256"]
    upstream = BASE / "upstream/CAT_MT_phase7_v0.7"
    for line in (upstream / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        expected["upstream/CAT_MT_phase7_v0.7/" + name] = digest
    expected["incoming/CAT_MT_background_selection_v0.7.zip"] = "8fe0094d2260e84e98905e53f2fdec211fa0644b1e0ecfa194ab2633ad6f0aae"
    return expected


# 向 ZIP 写入内存中的说明或清单文本，并记录内容摘要。
def add_bytes(archive, name, content):
    archive.writestr(TOP + "/" + name, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    return dict(path=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest())


# 把磁盘文件加入 ZIP，并更新文件长度与校验信息。
def add_file(archive, path, name):
    assert path.is_file() and not path.is_symlink(), str(path)
    before = path.stat()
    info = zipfile.ZipInfo.from_file(path, arcname=TOP + "/" + name)
    info.compress_type = zipfile.ZIP_STORED if path.suffix.lower() in PRECOMPRESSED else zipfile.ZIP_DEFLATED
    info._compresslevel = 6
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as source, archive.open(info, "w", force_zip64=before.st_size > 2_000_000_000) as target:
        while chunk := source.read(CHUNK):
            digest.update(chunk)
            target.write(chunk)
            count += len(chunk)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), f"Source changed: {path}"
    assert count == before.st_size
    return dict(path=name, bytes=count, sha256=digest.hexdigest())


# 按清单逐项核对 ZIP 成员和校验值。
def verify_zip(path, manifest, manifest_bytes, checksums):
    expected = {TOP + "/" + item["path"]: item for item in manifest["files"]}
    with zipfile.ZipFile(path) as archive:
        all_names = archive.namelist()
        meta_names = {TOP + "/PACKAGE_MANIFEST.json", TOP + "/MANIFEST.sha256"}
        assert len(all_names) == len(set(all_names))
        assert set(all_names) == set(expected) | meta_names
        assert archive.read(TOP + "/PACKAGE_MANIFEST.json") == manifest_bytes
        assert archive.read(TOP + "/MANIFEST.sha256") == checksums
        last = time.monotonic()
        for index, (name, item) in enumerate(expected.items(), 1):
            with archive.open(name) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            assert digest == item["sha256"], name
            assert archive.getinfo(name).file_size == item["bytes"], name
            if time.monotonic() - last > 15:
                event(stage="verifying", files=index, total=len(expected))
                last = time.monotonic()
    return len(expected) + 2


# 根据明确文件列表构建研究快照 ZIP，并在完成前检查交付完整性。
def build(kind, destination, files, expected, project_count):
    partial = destination.with_suffix(destination.suffix + ".partial")
    assert not destination.exists() and not partial.exists()
    assert destination.parent.resolve() == DEST.resolve()
    records = []
    guarded = 0
    written = 0
    last = time.monotonic()
    with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        records.append(add_bytes(archive, "00_先读我.md", guide(kind).encode("utf-8")))
        records.append(add_bytes(archive, "verify_package.py", VERIFY_SCRIPT.encode("utf-8")))
        for index, (path, name) in enumerate(files, 1):
            item = add_file(archive, path, name)
            records.append(item)
            written += item["bytes"]
            if name.startswith("cat_mt_followup/"):
                relative = name.removeprefix("cat_mt_followup/")
                if relative in expected:
                    assert item["sha256"] == expected[relative], f"Preservation mismatch: {relative}"
                    guarded += 1
            if time.monotonic() - last > 15:
                event(stage="packing", edition=kind, files=index, total=len(files), bytes=written)
                last = time.monotonic()
        if kind == "full":
            assert guarded == len(expected), (guarded, len(expected))
        manifest = dict(utc=datetime.now(timezone.utc).isoformat(), edition=kind,
            research_cutoff="2026-09-11", package_date=DATE,
            project_files_at_packaging=project_count,
            preserved_historical_hashes_checked=guarded,
            excludes=[] if kind == "full" else ["original archives", "raw data", "model weights", "prediction/reference and fitted arrays"],
            files=records)
        manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        checksums = "".join(item["sha256"] + "  " + item["path"] + "\n" for item in records).encode("utf-8")
        archive.writestr(TOP + "/PACKAGE_MANIFEST.json", manifest_bytes)
        archive.writestr(TOP + "/MANIFEST.sha256", checksums)
    event(stage="archive_written", edition=kind, bytes=partial.stat().st_size)
    entries = verify_zip(partial, manifest, manifest_bytes, checksums)
    digest = sha(partial)
    partial.rename(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(digest + "  " + destination.name + "\n", encoding="utf-8")
    result = dict(edition=kind,path=str(destination),bytes=destination.stat().st_size,
        sha256=digest,zip_entries=entries,payload_files=len(records),
        payload_bytes=sum(item["bytes"] for item in records),
        preserved_historical_hashes_checked=guarded,crc_and_sha256_verified=True)
    event(stage="completed", **result)
    return result


# 按历史交付规则收集项目文件，生成 ZIP、清单和包内校验程序。
def main():
    expected = preserved_hashes()
    project_files = sorted(p for p in BASE.rglob("*") if p.is_file())
    assert all(not p.is_symlink() and p.resolve().is_relative_to(BASE.resolve()) for p in project_files)
    actual_names = {p.relative_to(BASE).as_posix() for p in project_files}
    assert set(expected) <= actual_names
    files = [(p, "cat_mt_followup/" + p.relative_to(BASE).as_posix()) for p in project_files]
    files.append((VISUAL, "supplements/local-diamond.html"))
    full_path = DEST / f"CAT_MT_完整研究包_{DATE}.zip"
    light_path = DEST / f"CAT_MT_轻量阅读版_{DATE}.zip"
    receipt = DEST / f"CAT_MT_打包核验_{DATE}.json"
    assert not receipt.exists()
    event(stage="inventory", project_files=len(project_files), bytes=sum(p.stat().st_size for p in project_files),
          historical_hashes=len(expected))
    full = build("full", full_path, files, expected, len(project_files))
    reading_files = [(p,n) for p,n in files if p.suffix.lower() in READING_TYPES]
    light = build("reading", light_path, reading_files, expected, len(project_files))
    compile(VERIFY_SCRIPT, "verify_package.py", "exec")
    result = dict(utc=datetime.now(timezone.utc).isoformat(),status="passed",project=str(BASE),
        existing_research_files_modified=False,packages=[full,light])
    receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    event(stage="delivery_verified",receipt=str(receipt))


if __name__ == "__main__":
    main()
