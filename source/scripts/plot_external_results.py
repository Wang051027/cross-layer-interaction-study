# 中文阅读说明｜用途：绘制原六教师实验的误差和成本对比。
# 数据流：读取原实验汇总和计时文件；输出报告使用的图像。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Standalone figure; primary accuracy and descriptive matched-budget cost."""
from __future__ import annotations
import os
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"]=str(BASE / "runs/external_figure_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv


# 用途：读取本函数指定的记录文件，返回解析后的对象；JSON/CSV 形式见函数体。
# 参数：path；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：list(csv.DictReader(f))。
def read(path):
    with path.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))


# 用途：绘制原六教师实验的误差和成本对比。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def main():
    run=BASE / "runs/fashion_external_20260911"
    rows=read(run / "test/all_results.csv")
    costs=read(run / "analysis/cost_and_accuracy_seeds.csv")
    out=BASE / "reports"
    methods=[("greedy","Selected head","#137c66"),("weight","Weight norm","#cc7935"),("pca_fused","Fused PCA","#8354be")]
    fig,axes=plt.subplots(2,2,figsize=(11.2,8.0))
    for col,layer in enumerate((1,2)):
        ax=axes[0,col]
        for offset,(method,label,color) in zip((-.22,0,.22),methods):
            for pos,k in enumerate((16,32)):
                values=np.array([100*float(r["nrmse"]) for r in rows if r["layer"]==str(layer) and r["alpha"]=="1.0" and r["method"]==method and r["k"]==str(k)])
                assert len(values)==6
                x=pos+offset
                ax.bar(x,values.mean(),width=.2,color=color,alpha=.78,label=label if pos==0 else None)
                ax.errorbar(x,values.mean(),yerr=values.std(ddof=1),color=color,capsize=3,lw=1.2)
                ax.scatter(x+np.linspace(-.045,.045,6),values,s=15,color=color,edgecolor="white",linewidth=.4,zorder=5)
        ax.axhline(10,color="#ad3434",lw=1.1,ls="--")
        ax.set_xticks((0,1),("16 / 64 states","32 / 64 states"))
        ax.set_ylabel("Test intervention NRMSE (%)")
        ax.set_title(f"Layer {layer}: error on fixed test inputs",loc="left",fontsize=12,fontweight="bold")
        ax=axes[1,col]
        for i,(method,label,color) in enumerate(methods):
            for shift,scope,marker in ((-.11,"from_image","o"),(.11,"cached","s")):
                values=np.array([float(r["speedup"]) for r in costs if r["layer"]==str(layer) and r["method"]==method and r["rule"]=="fixed32" and r["queries"]=="256" and r["workload"]==scope])
                assert len(values)==6
                ax.errorbar(i+shift,values.mean(),yerr=values.std(ddof=1),color=color,marker=marker,markersize=6,capsize=3,lw=1.1)
                ax.scatter(i+shift+np.linspace(-.04,.04,6),values,s=13,marker=marker,color=color,alpha=.5)
        ax.axhline(1,color="#43546a",lw=1.1,ls="--")
        ax.set_xticks(range(3),[m[1] for m in methods])
        ax.set_ylabel("Paired speedup vs exact local")
        ax.set_title(f"Layer {layer}: cost at k = 32, Q = 256",loc="left",fontsize=12,fontweight="bold")
    for ax in axes.flat:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.16)
        ax.set_axisbelow(True)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.947),ncol=3,frameon=False)
    fig.suptitle("Fashion-MNIST: six frozen teachers",fontsize=16,y=.995)
    fig.text(.5,.015,"Dots: individual teachers; bars: mean +/- SD. Error: 256 official test images, 219 high-order coalitions.\n"
        "Cost: CPU float64, one thread, four validation images/teacher. Circles: from image; squares: cached.\n"
        "Primary contrast: selected vs weight norm, layer 1, k = 16. Other comparisons are descriptive.",
        ha="center",fontsize=8.8,color="#43546a")
    fig.tight_layout(rect=(0,.095,1,.91))
    fig.savefig(out / "six_seed_external_results.png",dpi=170)
    fig.savefig(out / "six_seed_external_results.pdf")
    plt.close(fig)


# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__=="__main__":main()
