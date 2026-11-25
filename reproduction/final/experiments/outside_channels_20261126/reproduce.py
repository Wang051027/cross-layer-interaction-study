"""从冻结权重重选组外通道，在新目录重跑单独和联合阻断。"""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parents[1]
RUN_NAME = "dsprites_outside_channels_20261126"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--assets", type=Path, help="冻结输入所在仓库的根目录")
    parser.add_argument("--out", type=Path, help="新建的输出目录")
    args = parser.parse_args()
    root = next((p for p in HERE.parents if (p / "reproduction/held_condition").is_dir()), BUNDLE)
    assets = args.assets.resolve() if args.assets else root
    created = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out = args.out.resolve() if args.out else root / "local_output" / ("outside-channels-" + created)
    if out.exists():
        raise FileExistsError(out)

    def run(program, *arguments):
        subprocess.run([sys.executable, "-B", "-X", "utf8", str(HERE / program), *map(str, arguments)], check=True)

    run("run.py", "all", "--assets", assets, "--out", out, "--device", args.device)
    run("score.py", out)
    run("verify_replay.py", BUNDLE / "runs" / RUN_NAME, out, "--out", out / "replay_comparison.json")
    run("figure.py", out)
    print(out)


if __name__ == "__main__":
    main()
