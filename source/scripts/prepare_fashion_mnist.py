# 中文阅读说明｜用途：下载并核验官方 Fashion-MNIST 文件，检查格式、校验值和重复图像。
# 数据流：仅使用标准库处理下载与审计；输出 raw/ 数据和质量记录，不执行模型。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Download official Fashion-MNIST bytes and audit them without running models.

Standard-library only. Existing data with a different checksum is not overwritten.
No training, split selection, intervention, or test-performance evaluation occurs.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import struct
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "fashion_mnist" / "raw"
AUDIT = ROOT / "data" / "fashion_mnist" / "audit"
SOURCE = "https://github.com/zalandoresearch/fashion-mnist"
EXPECTED = {
    "train-images-idx3-ubyte.gz": ("8d4fb7e6c68d591d4c3dfef9ec88bf0d", 2051, 60000),
    "train-labels-idx1-ubyte.gz": ("25c81989df183df01b3e8a0aad5dffbe", 2049, 60000),
    "t10k-images-idx3-ubyte.gz": ("bef4ecab320f06d8554ea6380940ec79", 2051, 10000),
    "t10k-labels-idx1-ubyte.gz": ("bb300cfdad3c16e7a12a480ee83cd310", 2049, 10000),
}


# 用途：通过 HTTP 请求读取原始数据字节；这里只获取文件，不解析图像。
# 参数：url；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：response.read()。
def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "CAT-MT-data-audit/1.0"})
    with urllib.request.urlopen(req, timeout=45) as response:
        return response.read()


# 用途：按指定算法计算一段字节的摘要，用于下载数据校验。
# 参数：data、algorithm；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：hashlib.new(algorithm, data).hexdigest()。
def digest(data: bytes, algorithm: str = "sha256") -> str:
    return hashlib.new(algorithm, data).hexdigest()


# 用途：把数据审计结果写成便于回读的 JSON 文件。
# 参数：path、value；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def save_json(path: Path, value: object) -> None:
    # 把结构化配置或结果落盘，供后续阶段核验和报告读取。
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# 用途：下载并核验官方 Fashion-MNIST 文件，检查格式、校验值和重复图像。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    revision = json.loads(fetch("https://api.github.com/repos/zalandoresearch/fashion-mnist/commits/master"))["sha"]
    source_base = f"https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/{revision}"
    readme = fetch(source_base + "/README.md")
    license_text = fetch(source_base + "/LICENSE")
    (AUDIT / "official_README.md").write_bytes(readme)
    (AUDIT / "official_LICENSE.txt").write_bytes(license_text)
    files = []
    parsed = {}
    for name, (expected_md5, magic, count) in EXPECTED.items():
        if expected_md5.encode() not in readme:
            raise ValueError(f"Official README does not confirm expected checksum for {name}")
        path = RAW / name
        url = source_base + "/data/fashion/" + name
        downloaded = not path.exists()
        compressed = fetch(url) if downloaded else path.read_bytes()
        if digest(compressed, "md5") != expected_md5:
            raise ValueError(f"Checksum mismatch: {name}; no replacement written")
        content = gzip.decompress(compressed)
        actual_magic, actual_count = struct.unpack(">II", content[:8])
        if (actual_magic, actual_count) != (magic, count):
            raise ValueError(f"Unexpected IDX header: {name}")
        if magic == 2051:
            rows, cols = struct.unpack(">II", content[8:16])
            if (rows, cols) != (28, 28) or len(content) != 16 + count * rows * cols:
                raise ValueError(f"Unexpected image shape/length: {name}")
            body = content[16:]
            item = {"shape": [count, rows, cols], "dtype": "uint8"}
        else:
            body = content[8:]
            if len(body) != count or any(v > 9 for v in body):
                raise ValueError(f"Unexpected label values/length: {name}")
            item = {"shape": [count], "dtype": "uint8", "class_counts": dict(sorted(Counter(body).items()))}
        if downloaded:
            path.write_bytes(compressed)
        parsed[name] = body
        files.append({"name": name, "url": url, "bytes": len(compressed), "md5": expected_md5,
                      "sha256": digest(compressed), "idx_sha256": digest(content), "header_valid": True, **item})
        print(f"VERIFIED {name}: {len(compressed):,} bytes", flush=True)

    groups = {}
    for split in ("train", "t10k"):
        images = parsed[f"{split}-images-idx3-ubyte.gz"]
        labels = parsed[f"{split}-labels-idx1-ubyte.gz"]
        by_hash = defaultdict(list)
        for i, label in enumerate(labels):
            image_hash = digest(images[i * 784 : (i + 1) * 784])
            by_hash[image_hash].append((i, int(label)))
        groups[split] = by_hash

    duplicates = {}
    duplicate_detail = {}
    for split, by_hash in groups.items():
        repeated = {h: items for h, items in by_hash.items() if len(items) > 1}
        duplicates[split] = {
            "unique_images": len(by_hash),
            "duplicate_groups": len(repeated),
            "records_in_duplicate_groups": sum(len(items) for items in repeated.values()),
            "excess_duplicate_records": sum(len(items) - 1 for items in repeated.values()),
            "conflicting_label_groups": sum(len({v[1] for v in items}) > 1 for items in repeated.values()),
        }
        duplicate_detail[split] = repeated
    overlap = sorted(groups["train"].keys() & groups["t10k"].keys())
    duplicate_detail["official_train_test_overlap"] = [
        {"sha256": h, "train": groups["train"][h], "test": groups["t10k"][h]} for h in overlap
    ]
    duplicates["official_train_test_overlap"] = {
        "unique_images": len(overlap),
        "train_records": sum(len(groups["train"][h]) for h in overlap),
        "test_records": sum(len(groups["t10k"][h]) for h in overlap),
    }
    save_json(AUDIT / "exact_duplicate_groups.json", duplicate_detail)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": SOURCE,
        "source_commit": revision,
        "official_readme_sha256": digest(readme),
        "official_license_sha256": digest(license_text),
        "audit_script_sha256": digest(Path(__file__).read_bytes()),
        "files": files,
        "exact_pixel_duplicates": duplicates,
        "near_duplicate_audit": "not performed",
        "development_split": "not generated; reconcile with original frozen protocol before training",
        "test_access": "IDX integrity, label validity/counts, and exact-duplicate audit only; no model predictions or intervention outcomes",
        "research_results_reproduced": False,
        "model_training_runs": 0,
    }
    save_json(AUDIT / "data_audit.json", report)
    print(json.dumps({"status": "all_four_files_verified", "source_commit": revision,
                      "exact_pixel_duplicates": duplicates, "model_training_runs": 0}, indent=2), flush=True)


# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__ == "__main__":
    main()
