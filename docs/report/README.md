# 正式英文报告草稿

[报告PDF](SafeInvest_report_draft.pdf) · [LaTeX主文件](main.tex) · [来源及生成资产校验](evidence.json) · [本次验收记录](validation.json)

本稿依据迁出仓库保留的最终发布决定，采用冻结的校准Plan A：A/D人工、B/C Conservative、E Balanced。它不是神经网络实验报告，也没有把后续未通过完整采用条件的研究写成最终方案。正文为英文，9页A4；正文、表格、图注和图中文字至少11pt，公式上下标使用正常数学排版。首页队名为Fen8FenQian，成员按用户提供的顺序署名：Huang Sinan、Cai Xunzhi、Lin Zihan。仍需团队审阅全文及实际AI使用披露。

## 内容与题目要求

| 任务 | 报告位置 |
| --- | --- |
| 数据、开发/最终分区和最终选择 | 第1页 |
| 真实购买力、费用时序、增长和损失口径 | 第2页 |
| 三产品优化、凸性范围、求解及嵌套选择 | 第3页 |
| 各期限权重、五客户筛选与拒绝自动服务 | 第4页 |
| 保持/再平衡/改执行目标/人工，成本门槛与20词解释 | 第5页 |
| 等权与同锚参照、增长/损失/交易率和配对区间 | 第6页 |
| 客户实际期限、波动/终值亏损、严重路径155 | 第7页 |
| 成本敏感性、保存核验与假设局限 | 第8页 |
| 结论、来源许可、参考文献与末尾AI披露 | 第9页 |

两张图分别来自最终保存的`summary.csv`：风险收益对比、费用敏感性；均在正文中引用。严重案例仅使用保存的峰值、谷值、交易数和人工复核时点，不插值或虚构缺失的年度财富曲线。正文明确区分P90、普通回撤CVaR和优化用的对数回撤ES，也不把OOF上界当作重拟合策略的严格覆盖保证。

## 修改和重建

在根目录的[报告配置](../../configs/report.json)中维护`members`、`team_name`和实际`ai_tools`；当前成员与队名已填写。正文放在本目录`sections/`，参考文献放在`references.tex`。生成的`generated/*.tex`不是手工数值来源，修改配置后重新生成。

Windows PowerShell，在项目根目录执行：

```powershell
.venv/Scripts/python.exe scripts/build_formal_report.py
Push-Location docs/report
New-Item -ItemType Directory -Force build | Out-Null
pdflatex --disable-installer --interaction=nonstopmode --halt-on-error -output-directory=build -jobname=SafeInvest_report_draft main.tex
pdflatex --disable-installer --interaction=nonstopmode --halt-on-error -output-directory=build -jobname=SafeInvest_report_draft main.tex
pdfinfo build/SafeInvest_report_draft.pdf
pdftoppm -r 100 -png build/SafeInvest_report_draft.pdf build/page
# 检查所有页面和编译日志后，再更新交付PDF。
Copy-Item -LiteralPath build/SafeInvest_report_draft.pdf -Destination SafeInvest_report_draft.pdf
Pop-Location
```

LaTeX由`5writing/templates/en/mcm-latex`适配，按本题要求去掉通用COMAP封面/目录，不能误套其他比赛的页数规则。主文件兼容XeLaTeX；本机XeLaTeX初始化遇到用户路径撇号问题，本次使用现有pdfLaTeX构建。没有安装或修改项目的金融依赖。Python绘图使用现有NumPy、Pandas、Matplotlib环境，本次只复现保存数值的表格和图形，不声称重建了历史金融训练环境。

`build_formal_report.py`只读取配置列出的保存JSON/CSV及题目PDF字节（用于来源hash），不调用市场读取、模型拟合或最终评价入口。它核对发布策略、冻结身份、最终锁、源码身份、60个最终场景、五客户映射及保存核验，并确认来源文件生成前后hash一致。`evidence.json`记录来源与生成资产hash。正文中的常量和定性规则另按`METHODOLOGY.md`及现有实现核对。

## 当前验收与剩余事项

2026-10-19再次核对第6、7节：按审计实现纠正`3.64e-12`的量纲为期末持仓金额误差（名义财富单位），明确780项标量参照仅针对等权策略；补充成本变化后建议是否改变、客户筛选仅覆盖基础和翻倍费率、五档费率共用60条最终路径。结论补齐三产品定义、选择性调仓建议以及人工触发和等待行为；本轮修改第8、9页，第1至7页文字保持一致。

- PDF共9页A4；两遍编译完成，无未定义引用和溢出警告；全部页面完成视觉检查。
- 表格、关键正文数值及冻结分配核验通过；原金融源码、冻结模型、最终锁、历史结果和实验登记未修改。
- 本次不是新开发实验，不增加实验登记。没有重跑`make test`或金融评估；本次证据检查和排版验证见`validation.json`，不将历史103项测试冒充本次执行。
- 成员与队名已补齐，PDF作者元数据同步使用三位成员姓名；AI披露只列本次明确使用的工具，团队应按完整实际使用情况修改，最终责任由成员承担。
- 本地旧源码ZIP及其收据不变；本报告没有自动写入旧归档，也未向比赛平台提交。若改动正文、署名或图表，应重新编译、检查页数与逐页渲染，并刷新本次验收记录。
