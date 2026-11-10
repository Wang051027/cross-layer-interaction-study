"""Plot the completed follow-up and the quality limitation of the GAP control."""
from pathlib import Path
import os
BASE=Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"]=str(BASE / "runs/structure_figure_cache")
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    with path.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))


# 绘制激活选择和 GAP 结构对照，帮助解释背景需求来自哪里。
def main():
    run=BASE / "runs/fashion_structure_20260911"
    data=read(run / "analysis/seed_nrmse.csv")
    classes=[read(BASE / "runs/fashion_external_20260911/test/teacher_classification.csv"),
             read(run / "test/gap_classification.csv")]
    fig,axes=plt.subplots(2,2,figsize=(11.3,8.1))
    methods=[("greedy","Residual selection","#147a68"),("weight","Weight norm","#ce803d"),
             ("gate_weight","Gate-weight score","#705ab0"),("active_random_mean","Active random (4-run mean)","#488eb5")]
    for column,layer in enumerate((1,2)):
        ax=axes[0,column]
        for offset,(method,label,color) in zip(np.linspace(-.27,.27,4),methods):
            for position,k in enumerate((16,32)):
                values=np.array([100*float(r["nrmse"]) for r in data if r["architecture"]=="cnn" and r["layer"]==str(layer) and r["k"]==str(k) and r["alpha"]=="1.0" and r["method"]==method])
                assert len(values)==6
                ax.errorbar(position+offset,values.mean(),yerr=values.std(ddof=1),fmt="o",color=color,capsize=3,
                    label=label if position==0 else None,ms=5,lw=1.1)
                ax.scatter(position+offset+np.linspace(-.028,.028,6),values,s=13,color=color,alpha=.55)
        ax.set_xticks([0,1],["16 / 64 states","32 / 64 states"])
        ax.axhline(10,color="#9f3f3f",ls="--",lw=1)
        ax.set_title(f"Original CNN, layer {layer}: activity controls",loc="left",fontsize=11.6,fontweight="bold")
        ax.set_ylabel("Intervention NRMSE (%)")
        ax.set_xlim(-.45,1.45)
    ax=axes[1,0]
    for offset,method,label,color in [(-.12,"mean_boundary","Training mean boundary","#ce803d"),
                                    (.12,"original_region","Original-region formula","#705ab0")]:
        for position,layer in enumerate((1,2,3)):
            values=np.array([100*float(r["nrmse"]) for r in data if r["architecture"]=="gap" and r["layer"]==str(layer) and r["alpha"]=="1.0" and r["method"]==method])
            assert len(values)==6
            ax.errorbar(position+offset,values.mean(),yerr=values.std(ddof=1),fmt="o",color=color,capsize=3,
                label=label if position==0 else None,ms=5)
            ax.scatter(position+offset+np.linspace(-.025,.025,6),values,s=12,color=color,alpha=.55)
    ax.axhline(0,color="#526273",lw=1,ls="--",label="Exact local: numerical precision")
    ax.set_xticks([0,1,2],["After ReLU1","After ReLU2","After MaxPool"])
    ax.set_ylabel("GAP model intervention NRMSE (%)")
    ax.set_title("Linear readout removes head, not earlier boundaries",loc="left",fontsize=11.2,fontweight="bold")
    ax.legend(fontsize=8.1,frameon=False,loc="best")
    ax=axes[1,1]
    for position,(group,color) in enumerate(zip(classes,["#147a68","#647c99"])):
        values=np.array([100*float(r["accuracy"]) for r in group])
        ax.bar(position,values.mean(),width=.43,color=color,alpha=.75)
        ax.errorbar(position,values.mean(),yerr=values.std(ddof=1),color=color,capsize=4)
        ax.scatter(position+np.linspace(-.10,.10,6),values,s=20,color=color,edgecolor="white",linewidth=.4)
        ax.text(position,values.max()+2,f"{values.mean():.2f}%",ha="center",fontsize=10)
    ax.set_xticks([0,1],["Original nonlinear head","GAP + linear head"])
    ax.set_ylim(0,102)
    ax.set_ylabel("Classification accuracy (%)")
    ax.set_title("GAP control has lower task accuracy",loc="left",fontsize=11.6,fontweight="bold")
    for ax in axes.flat:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.16);ax.set_axisbelow(True)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.954),ncol=2,frameon=False,fontsize=9)
    fig.suptitle("Follow-up: where does the background benefit come from?",fontsize=15,y=.995)
    fig.text(.5,.012,"Interventions: fixed official IDs 256-511; same dataset, additional batch. Six seeds; mean +/- SD.\n"
        "Classification: 10,000 official test images. Descriptive follow-up; no new confirmatory p-values.",
        ha="center",fontsize=8.5,color="#526273")
    fig.tight_layout(rect=(0,.065,1,.88),h_pad=2.2)
    fig.savefig(BASE / "reports/structure_controls.png",dpi=170)
    fig.savefig(BASE / "reports/structure_controls.pdf")
    plt.close(fig)


if __name__=="__main__":main()
