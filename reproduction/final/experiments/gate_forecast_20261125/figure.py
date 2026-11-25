"""绘制全强度留出条件下的逐模型对照。"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from run import read_csv


def main(run):
    available = {font.name for font in font_manager.fontManager.ttflist}
    chinese = next((f for f in ("Microsoft YaHei", "Noto Sans CJK SC", "PingFang SC") if f in available), None)
    if chinese:
        plt.rcParams["font.family"] = chinese
    plt.rcParams.update({"axes.unicode_minus":False, "font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    def label(cn, en):
        return cn if chinese else en
    def full(path):
        return [r for r in read_csv(path) if float(r["strength"])==1.]
    out=full(run/"teacher_scores.csv")
    gates=full(run/"gate_scores.csv")
    endpoint=full(run/"intervention_scores.csv")
    all_corners=full(run/"eight_corners/teacher_scores.csv")
    fig, axes=plt.subplots(1,3,figsize=(13.6,4.5),layout="constrained",gridspec_kw={"width_ratios":[1,1.2,1.05]})
    colors=("#667581","#227363","#825829","#426b94")
    def paired(ax, positions, series, color):
        data=np.asarray(series)
        for points in data.T:
            ax.plot(positions,points,color=color,alpha=.32,linewidth=.9)
        ax.plot(positions,data.mean(1),color=color,linewidth=2)
        for x,values in zip(positions,data):
            ax.scatter(np.full(len(values),x),values,s=14,color=color,alpha=.8)
            ax.scatter(x,values.mean(),marker="D",s=42,edgecolor="white",linewidth=.7,color=color,zorder=3)
    ax=axes[0]
    for start,layer,col in [(0,"first",colors[1]),(2.5,"second",colors[3])]:
        scores=[[float(r["f1"]) for r in gates if r["layer"]==layer and r["method"]==method and r["direction"]=="flip"]
                for method in ("majority","predicted")]
        paired(ax,[start,start+1],scores,col)
    ax.set_xticks([0,1,2.5,3.5],[label("多数票","Vote"),label("外推","Forecast")]*2)
    ax.set_xlabel(label("第一处 ReLU                    第二处 ReLU","First ReLU                       Second ReLU"))
    ax.set_ylim(0,1.03)
    ax.set_ylabel(label("翻转位置 F1","Flip F1"))
    ax.set_title(label("哪些位置会翻转？","Which sites flip?"),loc="left")
    ax=axes[1]
    methods=["fixed_4","class_mean","predicted_4","live_4"]
    names=[label("固定梯度","Fixed gradient"),label("类别均值","Class mean"),label("预测开关","Forecast gates"),label("重算下游","Natural readout")]
    for i,(key,col) in enumerate(zip(methods, (colors[0],colors[2],colors[1],colors[3]))):
        values=np.array([float(row[key]) for row in out])
        ax.scatter(values,i+np.linspace(-.105,.105,len(values)),s=18,color=col,alpha=.65)
        ax.scatter(values.mean(),i,s=55,marker="D",edgecolor="white",color=col,zorder=3)
    ax.set_yticks(range(4),names)
    ax.set_ylim(3.5,-.5)
    ax.axvline(1,color="#a8adb1",linestyle=":",linewidth=1.2)
    ax.set_xlim(0,1.47)
    ax.set_xlabel(label("三阶输出误差比","Third-order output error ratio"))
    ax.set_title(label("四通道的输出预测","Four-channel forecast"),loc="left")
    ax=axes[2]
    for i,(arm,col) in enumerate(zip(("target","random1","random2"),(colors[1],colors[0],colors[3]))):
        series=[[float(r["actual_rms"]) for r in data if r["arm"]==arm and r["layer"]=="both"]
                for data in (endpoint,all_corners)]
        shift=(i-1)*.12
        paired(ax,[0+shift,1+shift],series,col)
        ax.plot([],[],color=col,label=[label("定向","Targeted"),label("随机一","Random 1"),label("随机二","Random 2")][i])
    ax.set_xticks([0,1],[label("三组端点","Triple endpoint"),label("八状态三阶差分","Eight-state difference")])
    ax.set_xlim(-.35,1.35)
    ax.set_yscale("log")
    ax.set_ylabel(label("阻断改变量 RMS（logit，对数刻度）","RMS patch change (logit, log scale)"))
    ax.legend(frameon=False,fontsize=8,loc="upper right")
    ax.set_title(label("同一份位置名单的作用","Intervening on the same sites"),loc="left")
    for ax in axes:
        ax.grid(axis="y",color="#d9dddf",linewidth=.6,alpha=.7)
        ax.set_axisbelow(True)
    fig.savefig(run/"experiment_results.png",dpi=170,bbox_inches="tight",facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    main(parser.parse_args().run)
