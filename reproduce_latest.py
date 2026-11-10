"""从仓库中的冻结模型和输入完整复跑 dSprites 三组规则判别实验。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4
from zipfile import ZipFile

from check_repository import main as check_release


ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "reproduction/held_condition"
REFERENCE = BUNDLE / "reference"
PUBLISHED = ROOT / "results/rule_discrimination"
EXPERIMENT = "dsprites_rule_discrimination_20261031"
RTOL, ATOL = 1e-8, 1e-9


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_value(actual, expected, name):
    """标识、计数和判定精确匹配，浮点值允许不同后端的舍入误差。"""
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"{name}: keys differ")
        for key, value in expected.items():
            check_value(actual[key], value, f"{name}.{key}")
    elif isinstance(expected, list):
        for i, (value, reference) in enumerate(zip(actual, expected, strict=True)):
            check_value(value, reference, f"{name}[{i}]")
    elif isinstance(expected, float):
        if not math.isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL):
            raise ValueError(f"{name}: {actual!r} vs {expected!r}")
    elif actual != expected:
        raise ValueError(f"{name}: {actual!r} vs {expected!r}")


def prepare(work):
    # 保持原项目的相对路径。绘图脚本的缓存也会落在本次工作目录中。
    base = work / "experiment/source"
    base.mkdir(parents=True)
    for name in ("scripts", "upstream"):
        shutil.copytree(ROOT / "source" / name, base / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    protocol = read(REFERENCE / "protocol.json")
    with ZipFile(BUNDLE / "protocol_sources.zip") as frozen:
        if sorted(frozen.namelist()) != sorted(protocol["source"]):
            raise ValueError("Source archive differs from the protocol file list")
        for name in protocol["source"]:
            if name != f"scripts/{Path(name).name}" or not name.endswith(".py"):
                raise ValueError(f"Invalid source path in protocol: {name}")
            (base / name).write_bytes(frozen.read(name))
    shutil.copytree(BUNDLE / "runs", base / "runs")
    (base / "reports").mkdir()
    shutil.copyfile(REFERENCE / "source/CURRENT_RESEARCH.md", base / "CURRENT_RESEARCH.md")
    run = base / "runs" / EXPERIMENT
    run.mkdir()
    shutil.copyfile(REFERENCE / "design.json", run / "design.json")
    for path, expected in {**protocol["source"], **protocol["inputs"]}.items():
        if digest(base / path) != expected:
            raise ValueError(f"Original protocol hash differs: {path}")
    return base, run


def compare_table(path, original, identifiers):
    with path.open(encoding="utf-8-sig", newline="") as a, original.open(
            encoding="utf-8-sig", newline="") as b:
        actual, expected = csv.DictReader(a), csv.DictReader(b)
        if actual.fieldnames != expected.fieldnames:
            raise ValueError(f"{path.name}: columns differ")
        count = 0
        for row, reference in zip(actual, expected, strict=True):
            count += 1
            for column, value in reference.items():
                observed = row[column]
                if observed == value:
                    continue
                label = f"{path.name} row {count}, {column}"
                if column in identifiers or not value or not observed:
                    raise ValueError(f"{label}: {observed!r} vs {value!r}")
                check_value(float(observed), float(value), label)
        return count


def compare_run(run):
    historical = read(REFERENCE / "protocol.json")
    protocol = read(run / "protocol.json")
    for key in ("design", "source", "inputs", "split", "seeds", "strengths"):
        check_value(protocol[key], historical[key], key)
    for seed in protocol["seeds"]:
        name = f"fit/{seed}.json"
        check_value(read(run / name), read(REFERENCE / name), name)
    analysis = read(run / "analysis.json")
    archived = read(PUBLISHED / "analysis.json")
    check_value({k: v for k, v in analysis.items() if k != "utc"},
                {k: v for k, v in archived.items() if k != "utc"}, "analysis")
    audit = read(run / "independent_audit.json")
    check_value(audit["status"], "PASS", "audit")
    check_value(audit["counts"], read(PUBLISHED / "independent_audit.json")["counts"], "audit.counts")
    posthoc = read(run / "posthoc.json")
    for key in ("inputs", "stats", "used_for_selection"):
        check_value(posthoc[key], read(REFERENCE / "posthoc.json")[key], f"posthoc.{key}")
    identifiers = {"seed", "strength", "case", "size", "angle", "recipient", "donor", "mask",
                   "predicted_stratum", "patch", "channels", "target", "method", "sign_hits", "sign_eligible"}
    counts = {}
    for name in ("cases", "patches", "teacher_scores"):
        path = Path("tables") / f"{name}.csv"
        original = (PUBLISHED if name == "teacher_scores" else REFERENCE) / path
        counts[name] = compare_table(run / path, original, identifiers)
    # 压缩文件的逐字节匹配另行报告。跨平台比较以实测结果和独立核验为准。
    exact = {}
    for stage in ("fit", "prediction", "measurement"):
        archived_hashes = read(REFERENCE / f"{stage}_freeze.json")["files"]
        replay_hashes = read(run / f"{stage}_freeze.json")["files"]
        if replay_hashes.keys() != archived_hashes.keys():
            raise ValueError(f"{stage}: file list differs")
        exact[stage] = sum(replay_hashes[k] == value for k, value in archived_hashes.items())
    return {"status": "PASS", "decisions": analysis["decisions"], "compared_rows": counts,
            "independent_audit": audit["counts"], "archive_byte_matches": exact,
            "rtol": RTOL, "atol": ATOL}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new directory (default: a fresh local_output subdirectory)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if not __debug__:
        parser.error("Run without -O so the original numerical checks are enabled.")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONOPTIMIZE", None)
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable with this PyTorch installation.")
    device = "cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    check_release()
    if args.output is None:
        parent = ROOT / "local_output"
        parent.mkdir(exist_ok=True)
        work = parent / f"held-condition-{uuid4().hex[:12]}"
        work.mkdir()
    else:
        work = args.output.resolve()
        work.mkdir(parents=True, exist_ok=False)
    base, run = prepare(work)
    print(f"Device: {device}. Run directory: {run}", flush=True)
    stages = [
        ("run_dsprites_triple_rule.py", "predict"),
        ("run_dsprites_triple_rule.py", "measure"),
        ("audit_dsprites_triple_rule.py", None),
        ("analyze_dsprites_triple_rule.py", None),
        ("report_dsprites_triple_rule.py", "report"),
        ("report_dsprites_triple_rule.py", "seal"),
        ("report_dsprites_triple_rule.py", "verify"),
    ]
    for script, stage in stages:
        command = [sys.executable, "-B", "-X", "utf8", str(base / "scripts" / script),
                   "--run-dir", str(run)]
        if stage:
            command.extend(["--stage", stage])
        print(f"Running {script} {stage or ''}", flush=True)
        subprocess.run(command, cwd=base, env=env, check=True)
    comparison = compare_run(run)
    comparison["device"] = device
    comparison["torch"] = str(torch.__version__)
    comparison["report"] = (base / "reports/dSprites_留出图像条件的三组规则判别实验_2026-11-01.md").relative_to(work).as_posix()
    output = work / "comparison.json"
    output.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"PASS: {output}", flush=True)


if __name__ == "__main__":
    main()
