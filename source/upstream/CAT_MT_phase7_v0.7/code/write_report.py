from pathlib import Path
import json,hashlib
import pandas as pd
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
R=Path(__file__).resolve().parents[1]

# 创建 Word 文档并配置字体、页边距和标题等基础样式。
def base(title):
 d=Document();s=d.sections[0];s.page_width=Inches(8.27);s.page_height=Inches(11.69)
 s.top_margin=Inches(.62);s.bottom_margin=Inches(.62);s.left_margin=Inches(.72);s.right_margin=Inches(.72)
 st=d.styles['Normal'];st.font.name='Calibri';st.font.size=Pt(10.5);st.element.rPr.rFonts.set(qn('w:eastAsia'),'Noto Sans CJK SC')
 st.paragraph_format.space_after=Pt(5);st.paragraph_format.line_spacing=1.14
 for name,size in [('Title',21),('Heading 1',15),('Heading 2',12)]:
  st=d.styles[name];st.font.name='Calibri';st.font.size=Pt(size);st.font.color.rgb=RGBColor.from_string('17344D');st.element.rPr.rFonts.set(qn('w:eastAsia'),'Noto Sans CJK SC')
 h=s.header.paragraphs[0];h.text=title+'  |  CAT-MT v0.7';h.style='Caption'
 f=s.footer.paragraphs[0];f.alignment=2;f.add_run('开发实验 · 2026-09-07  |  ')
 field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');f._p.append(field)
 d.core_properties.title=title;d.core_properties.author='';return d

# 向 Word 文档加入段落，可选择加粗。
def p(d,text,bold=False):
 pp=d.add_paragraph();rr=pp.add_run(text);rr.bold=bold;return pp

# 把表头和数据行组织为报告表格；Word/Markdown 的输出形式由本文件决定。
def table(d,headers,rows,widths=None):
 t=d.add_table(rows=1,cols=len(headers));t.style='Light Shading Accent 1';t.autofit=False
 if widths:
  for j,w in enumerate(widths):t.columns[j].width=Inches(w)
 for i,h in enumerate(headers):t.rows[0].cells[i].text=h
 trPr=t.rows[0]._tr.get_or_add_trPr();rep=OxmlElement('w:tblHeader');trPr.append(rep)
 for vals in rows:
  cells=t.add_row().cells
  for c,v in zip(cells,vals):c.text=str(v)
 for row in t.rows:
  pr=row._tr.get_or_add_trPr();cs=OxmlElement('w:cantSplit');pr.append(cs)
  for j,c in enumerate(row.cells):
   if widths:c.width=Inches(widths[j])
   for pp in c.paragraphs:
    pp.paragraph_format.space_after=Pt(3);pp.paragraph_format.line_spacing=1.05
    for rr in pp.runs:rr.font.size=Pt(9.2)
 return t

# 插入分页并建立下一节标题。
def page(d,title):d.add_page_break();d.add_heading(title,0)
# 向 Word 文档插入指定图像，并按参数设置宽度及说明。
def pic(d,path,width=5.9,alt=''):
 pp=d.add_paragraph();pp.alignment=1;run=pp.add_run();sh=run.add_picture(str(path),width=Inches(width));sh._inline.docPr.set('descr',alt)

# 原 v0.7 的 Word 研究报告生成器。
def main():
 v=json.loads((R/'results/verification.json').read_text());ac=json.loads((R/'logs/fashion_acquisition.json').read_text())
 sums=pd.read_csv(R/'tables/background_summary.csv');quarter=pd.read_csv(R/'tables/quarter_budget_summary.csv');vs=pd.read_csv(R/'tables/validation_selection_summary.csv');bt=pd.read_csv(R/'tables/backend_timing_summary.csv')
 d=base('最小背景条件实验执行报告');d.add_heading('最小背景条件下的局部干预预测',0)
 p(d,'Phase VII 执行报告｜背景消融、受约束选择模块与外部数据准入',True)
 p(d,'当前状态：开发阶段已完成；Fashion-MNIST 外部数据准入未通过，外部训练和测试均未执行。',True)
 d.add_heading('1. 执行范围与阻塞',1)
 p(d,'原 v0.6 归档保持不变，本轮校验其218项文件清单，复制必要模型、数据和代码至新目录。沿用12个历史冻结教师模型，分别分析第一、第二卷积层后的干预。没有新增教师，也没有把旧测试集重新命名为外部验证。')
 table(d,['环节','状态与证据'],[
 ('旧版本冻结与核验','完成；218项源文件无哈希不符'),('Fashion-MNIST 获取','失败；四个文件各尝试两个官方地址，均未取得有效二进制'),('背景必要性消融','完成；24个模型—层运行、两种干预强度'),('一个受约束模块','完成；只学习保留的下游状态集合与平均门控，固定其余模型参数'),('独立外部测试','未执行；必须先通过官方校验和与IDX解析门槛'),('外部协议与数据脚本','已锁定取样和评价协议')],[1.8,5.0])
 d.add_heading('2. 实验规模与评价含义',1)
 p(d,'每个模型—层使用192个训练情境、64个验证情境和128个历史测试情境；每次选择8个固定源激活。训练只使用37个空集／单个／两两组合；测试参考执行256个组合，误差仅统计未用于低阶构建的219个高阶组合。')
 p(d,f"实际执行测试尾网络组合输出 {v['test_full_tail_output_evaluations']:,} 个，验证输出 {v['validation_full_tail_output_evaluations']:,} 个；保存 {v['result_rows']:,} 行配置结果。这些查询共享输入和模型，不是独立模型复现。统计以每个数据集的6个教师为单位。")
 p(d,'主表的百分比是完整logit变化的效应归一化RMSE，越低越好，不是分类准确率。NRMSE=100%对应始终预测零变化；近零真实效应可能使该比例放大，因此同时保存绝对误差。')
 p(d,'主开发设计在运行前本地记录代码哈希，但使用的是已多次探索的数据，不属于独立确认性研究。后文的Torch计时是看到初版NumPy开销后，单独记录的实现诊断。')

 page(d,'背景到底在哪里不可省略？')
 d.add_heading('3. 四层背景消融',1)
 p(d,'A仅用源激活和训练均值背景；B准确更新相邻层与池化，但不读取新的下游头部状态；C在B上保留训练选定的少量头部预激活；D保留完整背景，精确执行。A→B会恢复原网络的局部计算，因此不能把全部增益都归因于学习。')
 sr=[]
 for ds,lbl in [('synthetic','Synthetic'),('digits','Digits')]:
  for layer in [1,2]:
   part=sums[(sums.dataset==ds)&(sums.layer==layer)&(sums.alpha==1)].set_index('method')
   q=quarter[(quarter.dataset==ds)&(quarter.layer==layer)&(quarter.method=='greedy')].iloc[0]
   sr.append([lbl+' / L'+str(layer),f"{100*part.loc['local_only','nrmse_mean']:.2f}%",f"{100*part.loc['boundary_only','nrmse_mean']:.2f}%",f"{100*q.nrmse_mean:.2f}%",'数值精确'])
 table(d,['数据／位置','A 局部','B 邻层','C 25%头部','D 完整'],sr,[1.8,1.1,1.1,1.35,1.4])
 d.add_heading('4. 受约束模块如何计算',1)
 p(d,'设 d 是由精确局部卷积／池化计算出的变化，z是当前输入的头部预激活，W、U是原教师权重。完整响应为：')
 p(d,'ΔF = U [ReLU(z + Wd) − ReLU(z)]',True)
 p(d,'训练阶段估计每个头部单元的平均开门频率 g，选择少量单元S保留当前z值。未保留单元并非删除或置零，而是用固定的平均门控线性响应替代：')
 p(d,'ΔF近似 = U_S[ReLU(z_S + W_Sd) − ReLU(z_S)] + M_未选 d',True)
 p(d,'M_未选 = U_未选 diag(g_未选) W_未选，可以预先合并。保留集合仅由训练干预中的输出残差选择；贪心评分计入不同单元的输出抵消。测试真实干预不参与选择。')
 p(d,'筛选按局部干预响应的重建误差排序。贪心重构、结构化剪枝和贡献分解的方法见已有研究。[2–3]')
 p(d,'部署实现确实不计算被删去的原始头部预激活；将所有未选z值替换成10000后，已提交预测保持完全不变。但源层之前计算、邻层背景和池化计算仍保留。')

 page(d,'25%头部预算：效果与代价')
 d.add_heading('5. 同头部维数的控制比较',1)
 qr=[]
 for ds,lbl in [('synthetic','Synthetic'),('digits','Digits')]:
  for layer in [1,2]:
   part=quarter[(quarter.dataset==ds)&(quarter.layer==layer)].set_index('method')
   qr.append([lbl+' / L'+str(layer)]+[f"{100*part.loc[m,'nrmse_mean']:.2f}%" for m in ['greedy','random','weight','pca']])
 table(d,['数据／位置','训练选择','随机选择','权重排序','PCA*'],qr,[1.8,1.25,1.25,1.25,1.25])
 p(d,'* PCA同样只输出k维编码，但必须先读取全部头部状态才能投影，并仍执行完整头部非线性。Digits上PCA误差更低，必须保留这一结果；它不是相同背景获取成本的竞争项。不得据此声称训练选择全面优于PCA。')
 p(d,'Synthetic头部为48个单元，本表保留12个；Digits头部为64个，本表保留16个。邻层原始窗口信息没有减少。')
 pic(d,R/'figures/head_vs_total_background.png',5.8,'保留四分之一头部时，头部状态减少75%，但包括邻层背景后的总体减少在早层仅约8%，后层约52%至54%。')
 p(d,'“头部状态减少75%”不等于模型缩小75%，也不等于总背景减少75%。L1的总背景状态仅减少Synthetic 7.50%、Digits 8.71%；L2约减少51.64%和54.27%。这些是状态标量计数，不是总驻留内存；运行对象仍持有教师模型。',True)

 page(d,'误差门槛与完整成本')
 d.add_heading('6. 验证集选择的10%门槛没有自动迁移',1)
 vr=[]
 for row in vs.itertuples():
  vr.append([row.dataset+' / L'+str(row.layer),f'{row.k_mean:.2f} / {row.J}',f'{100*row.nrmse_mean:.2f}%',f'{int(row.seeds_meet_10pct)}/6'])
 table(d,['数据／位置','平均保留头部','测试NRMSE','满足≤10%的模型'],vr,[2.0,1.65,1.3,1.85])
 p(d,'保留数量只用验证集选择：取满足NRMSE≤10%的最小候选k。测试时不调整。表中部分模型仍越过10%，所以该规则是经验选参，不是逐输入可靠性、条件接受或OOD保证。固定25%预算与这一自适应预算是不同实验，不混写。')
 d.add_heading('7. 后验实现诊断：不能只挑有利计时',1)
 p(d,'初版NumPy实现的L1部分背景流程约比最快精确实现慢3倍。随后单独记录协议，用等价Torch实现控制后端开销；没有重新选择掩码或调整误差结果。下面是该诊断，不是预先设定的性能证明。')
 br=[]
 for row in bt[(bt.k==bt.J/4)&(bt.queries==256)].itertuples():
  br.append([row.dataset+' / L'+str(row.layer),f'{row.partial_ms:.3f}',f'{row.exact_ms:.3f}',f'{row.paired_speedup_mean:.2f}×'])
 table(d,['数据／位置','部分背景 ms','较强精确基线 ms','平均配对比'],br,[2.,1.5,1.75,1.45])
 p(d,'一次请求包含256个干预。计时包括从源层状态获取边界和所需头部背景；共同的源层之前前向计算均不计入。基线取多个精确实现的较快者。CPU单线程、float64、小架构；不推断GPU或大型CNN收益。')
 p(d,'因此，目前的速度改善约为1.09至1.21倍，而不是由“减少75%头部状态”推导出4倍加速。离线拟合和教师存储未被计成免费，相关记录单独保存；执行数组字节也不等于实际总内存。')

 page(d,'核验、外部准入与下一步决定')
 d.add_heading('8. 已完成的实现检查',1)
 table(d,['核查','结果'],[
 ('72组已保存主预测重新评分',f"最大差 {v['score_max_difference']:.3g}"),('24组序列化模块重载预测','最大差 0'),('24个代表情境完整尾网络重算','最大差 0'),('未保留头部状态污染检查','输出差 0'),('完整背景公式对独立尾网络',f"最大误差 {v['maximum_full_exact_error']:.3g}"),('原始测试预测与参考文件顺序','主预测文件先写入；模型／参数已冻结')],[3.6,3.2])
 p(d,'时间戳、哈希和同环境重跑说明文件与计算一致，不证明全过程被独立监督，也不是第三方科学复现。')
 d.add_heading('9. 外部实验仍然不能开跑',1)
 p(d,'已核验官方Fashion-MNIST说明与四项MD5。[1] 八次实际下载尝试均未获得二进制文件。acquire_fashion.py及prepare_external.py会在缺失、校验不符或IDX格式异常时拒绝继续。已实际检查它们的缺文件失败路径，成功获取／解析完整数据的路径尚未在本环境执行。')
 p(d,'外部协议已写为独立文件：首先获得官方文件，记录重复与固定分组划分，再训练新教师、训练背景选择模块、锁定后进行官方测试。当前外部教师训练次数=0，外部干预实验次数=0，残差架构实验次数=0。')
 d.add_heading('10. 研究决定',1)
 p(d,'继续进行外部可行性验证有依据：在这些旧模型上，训练选择的小部分头部状态能够明显优于同预算随机选择和权重排序。但不能宣布关键机制迁移已普遍解决；局部卷积／池化仍精确执行，PCA在Digits更准确，总背景减少和速度收益也有限。')
 p(d,'本轮不再另写“一区级”论文版本。先把这个单一受约束候选带到真正的新任务；若外部结果失败，保留失败，不通过增加词典或改指标来追认成功。',True)
 d.add_heading('参考与阅读范围',1)
 for txt in [
 '[1] Zalando Research. Fashion-MNIST官方仓库及校验值：https://github.com/zalandoresearch/fashion-mnist',
 '[2] Luo, Wu & Lin (2017). ThiNet. ICCV官方摘要：结构化选择与下一层重构已有基础；本轮未复现该方法。',
 '[3] Melander et al. (2026). CODEC. ICLR官方记录：贡献分解是直接近邻；本轮未运行其官方实现。',
 '本轮是定向来源核验和开发实验，不是新的系统文献综述。完整URL与阅读层次见literature/implementation_scope.md。']:
  pp=p(d,txt)
  for rr in pp.runs:rr.font.size=Pt(8.5)
 d.save(R/'reports/background_execution_report_zh.docx')

 # One-page external specification. No result cells are invented.
 x=base('外部验证冻结协议');x.styles['Normal'].font.size=Pt(9.5);x.styles['Normal'].paragraph_format.line_spacing=1.05;x.styles['Normal'].paragraph_format.space_after=Pt(4)
 x.add_heading('外部验证：一页冻结协议',0)
 p(x,'状态：设计已锁定，未执行。数据准入失败时不得继续。',True)
 table(x,['条目','预先指定内容'],[
 ('主问题','训练选定25%下游头部状态，能否在未见图像的高阶内部干预上，优于同预算权重排序，并维持实际成本优势？'),
 ('数据准入','官方Fashion-MNIST四个gzip文件；MD5、IDX、shape、标签与像素哈希全部通过。'),
 ('划分','名义50,000训练／5,000验证／5,000校准；重复像素组同划分，实际数量记录；官方10,000测试不改动，另报无训练重合子集。'),
 ('教师','新的两卷积ReLU CNN，8/16通道、2×2 MaxPool、64单元头部；6个固定新种子，AdamW、最多15轮、验证损失早停。不是残差架构验证。'),
 ('模块','与本轮同一训练残差贪心选择；主预算16/64。训练仅用512个情境的37个低阶响应；验证128情境；校准256情境预留。'),
 ('测试对象','256个固定分层抽样测试输入；q=8，主位置为第一卷积后，alpha=1；第二层和alpha=0.5为次要分析。目标是完整logit变化。'),
 ('隔离','冻结掩码、预算与所有参数后保存预测，再执行测试真实干预；不允许测试结果选模块或重拟合。'),
 ('比较','同预算随机／权重排序、平均门控、池化Ridge、同编码维数但读取全背景的PCA、完整精确局部执行。'),
 ('主要统计','一个预设主对比：第一层alpha=1的种子级NRMSE（贪心－权重排序）。其余描述性；6种子并非经过充分功效证明。'),
 ('实际成功条件','同时报告NRMSE与绝对误差；≤10%为聚合容忍度，不是逐样本保证。计入全部背景、计算、存储与回退；不从查询数推断速度。'),
 ('不允许的结论','已解决黑箱、语义机制发现、任意OOD保证、全面超越CODEC／Pathwise、一区接收承诺。')],[1.15,5.65])
 p(x,'执行入口：先运行 code/acquire_fashion.py --offline，再运行 code/prepare_external.py。这些只完成数据准入；教师训练和外部干预需在数据通过后另起日志执行。')
 p(x,'冻结机器可读协议：protocol/external_protocol.json；模块源代码SHA-256和数据种子一并记录。局部哈希锁定不是外部预注册。')
 x.save(R/'reports/external_protocol_onepage_zh.docx')
if __name__=='__main__':main()
