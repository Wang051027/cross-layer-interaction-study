"""用同一组冻结输入重做剩余三阶作用的路径测量。"""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parents[1]
RUN_NAME = "dsprites_remaining_routes_20261126"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--assets", type=Path, help="包含冻结权重与上一轮位置名单的仓库根目录")
    parser.add_argument("--out", type=Path, help="本次复跑要新建的目录")
    args = parser.parse_args()
    root = next((p for p in HERE.parents if (p / "reproduction/held_condition").is_dir()), BUNDLE)
    assets = args.assets.resolve() if args.assets else root
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out = args.out.resolve() if args.out else root / "local_output" / ("remaining-routes-" + timestamp)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)

    def run(program, *arguments):
        subprocess.run([sys.executable, "-B", "-X", "utf8", str(HERE / program),
                        *map(str, arguments)], check=True)

    # CPU 逐个替换通道并重新前向，CUDA 用分组卷积并行计算各个输入通道的改变量。
    engine = "direct" if args.device == "cpu" else "grouped"
    run("experiment.py", "all", "--assets", assets, "--out", out, "--device", args.device, "--engine", engine)
    run("summarize.py", out)
    run("audit_links.py", "--assets", assets, "--run", out)
    run("verify.py", BUNDLE / "runs" / RUN_NAME, out, "--out", out / "replay_comparison.json")
    run("figure.py", out)
    print(out)


if __name__ == "__main__":
    main()
