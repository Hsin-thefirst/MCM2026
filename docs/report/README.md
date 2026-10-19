# 正式英文报告草稿

[报告PDF](SafeInvest_report_draft.pdf) · [Part 4–5审阅说明](REVIEW_PARTS_4_5.md) · [LaTeX主文件](main.tex) · [来源及生成资产校验](evidence.json) · [本次验收记录](validation.json)

本稿依据迁出仓库保留的最终发布决定，采用冻结的校准Plan A：A/D人工、B/C Conservative、E Balanced。它不是神经网络实验报告，也没有把后续未通过完整采用条件的研究写成最终方案。正文为英文，9页A4；正文、表格、图注和图中文字至少11pt，公式上下标使用正常数学排版。首页队名为Fen8FenQian，成员按用户提供的顺序署名：Huang Sinan、Cai Xunzhi、Lin Zihan。仍需团队审阅全文及实际AI使用披露。

## 内容与题目要求

| 任务 | 报告位置 |
| --- | --- |
| 数据、开发/最终分区和最终选择 | 第1页 |
| 真实购买力、费用时序、增长和损失口径 | 第2页 |
| 三产品优化、凸性范围、求解及嵌套选择 | 第3页 |
| 各期限权重、五客户筛选与拒绝自动服务 | 第4页 |
| 保持/再平衡/改执行目标/人工，成本门槛、动作规则表与20词解释 | 第5页 |
| 等权与同锚参照、风险收益图、增长/损失/交易率和配对区间表 | 第6页 |
| 客户实际期限、波动/终值亏损、严重路径155的保存财富轨迹 | 第7页 |
| 成本敏感性、保存核验与假设局限 | 第8页 |
| 结论、来源许可、参考文献与末尾AI披露 | 第9页 |

全文8张表、3张图。风险收益与费用敏感性图来自最终保存的`summary.csv`；严重案例图来自保存的`decisions.csv.gz`，严格筛选原案例155的30条年度财富记录，不读取原始市场或重新执行策略，不推测年内价格。三图均在正文引用，提供PDF/SVG和PNG预览。正文明确区分P90、普通回撤CVaR和优化用的对数回撤ES，也不把OOF上界当作重拟合策略的严格覆盖保证。

正文第4、5节目前同在`sections/4_solution.tex`，文件名不等于正文节号。本轮审阅修正了成本门槛的候选处理、动作分类和严重案例交易时序，保留全部客户指标与不利结果；细节见审阅说明。

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

LaTeX由`5writing/templates/en/mcm-latex`适配，按本题要求去掉通用COMAP封面/目录，不能误套其他比赛的页数规则。主文件兼容pdfLaTeX与XeLaTeX；此前使用MiKTeX pdfLaTeX，本轮使用临时Tectonic 0.17.0的XeTeX引擎。XeLaTeX分支改用TeX Gyre字体文件名查找，避免依赖系统字体注册；pdfLaTeX分支不变。没有安装或修改项目的金融依赖。Python绘图使用现有NumPy、Pandas、Matplotlib环境，本次只复现保存数值的表格和图形，不声称重建了历史金融训练环境。

Tectonic已安装时，在`docs/report/`运行`tectonic --keep-logs --outdir build main.tex`（先建立`build/`目录）。它会自动完成必要的交叉引用重编译；首次运行可能需要联网取得TeX资源。检查完成后把`build/main.pdf`复制为本目录交付PDF。

## Nature图形检查

本轮安装并使用[nature-skills](https://github.com/Yuan1z0825/nature-skills)中的`nature-figure`、`nature-polishing`及共享依赖。绘图遵循其可编辑文字、克制配色、来源数据和导出后检查规范，字号仍按竞赛要求设为11pt。常规构建不依赖技能安装；以下为安装后可选的Python检查流程，在项目根目录执行：

```python
import sys
from pathlib import Path

skill_scripts = Path.home() / ".codex/skills/nature-figure/scripts"
sys.path.insert(0, str(skill_scripts))
sys.path.insert(0, "scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment
from build_formal_report import build

build(require_matplotlib_panel_alignment)
```

随后对三份最终图形PDF分别运行技能的`audit_pdf_text.py <PDF> --min-pt 11 --json`和`audit_figure_collisions.py <PDF> --json-out <记录路径>`，并运行`validate_figure.py scripts/build_formal_report.py --json`。图形几何、字体、碰撞和静态检查记录放在`figure_qa/`，最终仍须逐图及逐页检查。重新生成图片或正文后，旧`validation.json`不会自动成为新版本的验收证明，须重新核对并更新。

`build_formal_report.py`只读取配置列出的保存JSON/CSV及题目PDF字节（用于来源hash），不调用市场读取、模型拟合或最终评价入口。它核对发布策略、冻结身份、最终锁、源码身份、60个最终场景、五客户映射及保存核验，并确认来源文件生成前后hash一致。`evidence.json`记录来源与生成资产hash。正文中的常量和定性规则另按`METHODOLOGY.md`及现有实现核对。

## 当前验收与剩余事项

- PDF共9页A4；两遍编译完成，无未定义引用和溢出警告；全部页面完成视觉检查。
- 表格、关键正文数值及冻结分配核验通过；原金融源码、冻结模型、最终锁、历史结果和实验登记未修改。
- 本次不是新开发实验，不增加实验登记。没有重跑`make test`或金融评估；本次证据检查和排版验证见`validation.json`，不将历史103项测试冒充本次执行。
- 成员与队名已补齐，PDF作者元数据同步使用三位成员姓名；AI披露只列本次明确使用的工具，团队应按完整实际使用情况修改，最终责任由成员承担。
- 本地旧源码ZIP及其收据不变；本报告没有自动写入旧归档，也未向比赛平台提交。若改动正文、署名或图表，应重新编译、检查页数与逐页渲染，并刷新本次验收记录。
