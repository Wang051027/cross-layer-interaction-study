"""从模型级结果画主强度比较。虚线标出基线，保留逐模型离散程度。"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from run_experiment import js,read_csv


def make(run):
    # 优先选有中文字符的系统字体；缺少时回退到英文标签。
    fonts={f.name for f in font_manager.fontManager.ttflist}
    choice=next((n for n in ('Microsoft YaHei','Noto Sans CJK SC','SimHei','PingFang SC') if n in fonts),None)
    if choice:plt.rcParams['font.family']=choice
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.facecolor':'white','figure.facecolor':'white'})
    def name(zh,en):return zh if choice else en
    s=js(run/'summary.json')['summary']['1.0']
    t=[r for r in read_csv(run/'teacher_scores.csv') if float(r['strength'])==1]
    fig,ax=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    left,right=ax;budgets=[0,4,8,16,32];x=np.arange(5)
    for key,color,marker,label in [('fixed','#9a4c25','s',name('固定空组梯度','Fixed empty-state gradient')),
                                   ('live','#007d73','o',name('重算下游 ReLU','Recomputed downstream ReLUs'))]:
        y=[s[f'{key}_{b}']['mean'] for b in budgets]
        left.plot(x,y,marker=marker,lw=2.4,color=color,label=label,zorder=3)
        for i in (1,4):
            left.annotate(f'{y[i]:.3f}',(i,y[i]),xytext=(0,8 if key=='fixed' else -14),
                          textcoords='offset points',ha='center',color=color,fontsize=9)
    left.axhline(1,color='#676e75',lw=1,ls='--')
    left.axhline(s['class_mean']['mean'],color='#597da3',ls=':',lw=1.6,
                 label=name('开发格形状转换均值','Development class/triple mean'))
    left.set(xticks=x,xticklabels=budgets,xlim=(-.25,4.3),ylim=(0,1.21),
             xlabel=name('接受三阶修正的中间通道数','Middle channels receiving a correction'),
             ylabel=name('输出三阶误差比（越低越好）','Normalized output error (lower is better)'),
             title=name('A　同一中间状态，两种输出读法','A   Same middle field, different readout'))
    left.legend(loc='lower left',fontsize=9,frameon=False)
    labels=[name('仅第一处','First'),name('仅第二处','Second'),name('两处同时','Both')]
    positions=np.arange(3);jitter=np.linspace(-.12,.12,len(t))
    right.axhline(1,color='#676e75',lw=1,ls='--',label=name('自然交互 = 1','Native interaction = 1'))
    for i,key in enumerate(('first','second','both')):
        ys=[float(r['gap_left_'+key]) for r in t];avg=np.mean(ys)
        right.scatter(i+jitter,ys,s=30,color='#90a6ba',alpha=.85,zorder=2,
                      label=name('每个点：一个模型','One point per model') if i==0 else None)
        right.scatter([i],[avg],marker='D',s=54,color='#1b425d',zorder=3,
                      label=name('六模型均值','Six-model mean') if i==0 else None)
        right.annotate(f'{avg:.3f}',(i,avg),xytext=(11,0),textcoords='offset points',
                       va='center',color='#1b425d',fontsize=9)
    right.set(xticks=positions,xticklabels=labels,xlim=(-.45,2.55),ylim=(0,1.75),
              xlabel=name('对哪一处 ReLU 使用组外不变时的开关','ReLUs given matched outside-off masks'),
              ylabel=name('组内外交互剩余 RMS 比','RMS of remaining interaction / native RMS'),
              title=name('B　阻止组外改变开关，交互未必缩小','B   Preventing outside flips can increase the gap'))
    right.legend(loc='lower left',fontsize=9,frameon=False)
    for a in ax:a.grid(axis='y',alpha=.15);a.set_axisbelow(True)
    fig.savefig(run/'result_overview.png',dpi=175,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path);make(ap.parse_args().run)
