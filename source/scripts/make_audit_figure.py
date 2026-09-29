# 中文阅读说明｜用途：把首轮复核和成本检查中的已有数值绘成静态图。
# 数据流：读取保存表格，输出图像；绘图过程不训练模型或改变预测。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Standalone scientific figure of local replay accuracy and matched CPU timing."""
from __future__ import annotations
import os
import csv
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"]=str(BASE / "runs/figure_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 用途：读取本函数指定的记录文件，返回解析后的对象；JSON/CSV 形式见函数体。
# 参数：path；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：list(csv.DictReader(f))。
def read(path):
    with path.open(encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))


# 用途：把首轮复核和成本检查中的已有数值绘成静态图。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def main():
    accuracy=read(BASE / "runs/upstream_replay_20260910_v2/quarter_summary.csv")
    timing=read(BASE / "runs/pca_cost_audit_20260910_v2/timing_summary.csv")
    output=BASE / "reports"
    output.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.2))
    styles={"greedy":("#137c66","o","Selected head"),
            "pca_fused":("#8b4fc0","s","Fused PCA"),
            "exact":("#43546a","^","Exact local")}
    for ax,(dataset,layer) in zip(axes.flat,(("digits",1),("digits",2),("synthetic",1),("synthetic",2))):
        for method,(color,marker,label) in styles.items():
            t=next(r for r in timing if r["dataset"]==dataset and int(r["layer"])==layer
                   and r["method"]==method and r["queries"]=="256" and r["workload"]=="from_image")
            if method=="exact":
                mean,sd=0.,0.
            else:
                a=next(r for r in accuracy if r["dataset"]==dataset and int(r["layer"])==layer and r["method"]==method)
                mean,sd=100*float(a["nrmse_mean"]),100*float(a["nrmse_sd"])
            ax.errorbar(float(t["ms_mean"]),mean,xerr=float(t["ms_seed_sd"]),yerr=sd,
                        color=color,marker=marker,markersize=7,capsize=3,lw=1.2,label=label)
        ax.set_title(f"{dataset.title()} / layer {layer}",loc="left",fontsize=12,fontweight="bold")
        ax.set_xlabel("CPU time from image (ms)")
        ax.set_ylabel("Intervention NRMSE (%)")
        ax.set_ylim(bottom=-1)
        ax.grid(alpha=.18)
        ax.spines[["top","right"]].set_visible(False)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.955),ncol=3,frameon=False)
    fig.suptitle("Accuracy and complete online cost at 25% head context",fontsize=15,y=.998)
    fig.text(.5,.015,"Selection/PCA: k = J/4; exact baseline: full head.\n"
             "Q = 256; error scored on 219 high-order coalitions. Points: six-teacher means; bars: SD.\n"
             "CPU float64, one thread. Historical development data; timing from four inputs per teacher/layer.",
             ha="center",fontsize=9,color="#43546a")
    fig.tight_layout(rect=(0,.08,1,.91))
    fig.savefig(output / "accuracy_cost_audit.png",dpi=170)
    fig.savefig(output / "accuracy_cost_audit.pdf")
    plt.close(fig)


# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__=="__main__":
    main()
