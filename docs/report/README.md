# 正式英文报告草稿

[报告PDF](SafeInvest_report_draft.pdf) · [LaTeX主文件](main.tex) · [来源及生成资产校验](evidence.json) · [本次验收记录](validation.json)

本稿依据迁出仓库保留的最终发布决定，采用冻结的校准Plan A：A/D人工、B/C Conservative、E Balanced。它不是神经网络实验报告，也没有把后续未通过完整采用条件的研究写成最终方案。正文为英文，9页A4；正文、表格、图注和图中文字至少11pt，公式上下标使用正常数学排版。首页队名为Fen8FenQian，成员按用户提供的顺序署名：Huang Sinan、Cai Xunzhi、Lin Zihan。仍需团队审阅全文及实际AI使用披露。

**当前状态：前4页（Abstract、第1–3节）已完成精修，PDF尚未用当前源码重建**（本机MiKTeX半途更新后所有TeX可执行文件报DLL缺失，见下文）。最终编译与逐页目检需在另一台设备完成，交付PDF仍为上一版，其SHA256记录在`validation.json`。

## 内容与题目要求

| 任务 | 报告位置 |
| --- | --- |
| 数据、开发/最终分区和最终选择 | 第1页 |
| 真实购买力、费用时序、增长和损失口径；日常波动与异常大损失的分离 | 第2页 |
| 三产品优化、凸性范围、求解及嵌套选择 | 第3页 |
| 各期限权重、440家族Bonferroni筛选、五客户逐产品筛选表与拒绝自动服务 | 第4页 |
| 保持/再平衡/改执行目标/人工，成本门槛与20词解释 | 第5页 |
| 等权与同锚参照、增长/损失/交易率和配对区间 | 第6页 |
| 客户实际期限、波动/终值亏损、严重路径155 | 第7页 |
| 成本敏感性、保存核验与假设局限 | 第8页 |
| 结论、来源许可、参考文献与末尾AI披露 | 第9页 |

两张图分别来自最终保存的`summary.csv`：风险收益对比、费用敏感性；均在正文中引用。严重案例仅使用保存的峰值、谷值、交易数和人工复核时点，不插值或虚构缺失的年度财富曲线。正文明确区分P90、普通回撤CVaR和优化用的对数回撤ES，也不把OOF上界当作重拟合策略的严格覆盖保证。

第4页的筛选表（`generated/suitability.tex`）由保存的`selected_suitability_risk.csv`生成，脚本内断言“转人工者不得有合格对”且“推荐产品必须等于合格对中OOF中位增长最高者”，因此表格、入学规则与冻结分配不可能悄悄不一致。

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

LaTeX由`5writing/templates/en/mcm-latex`适配，按本题要求去掉通用COMAP封面/目录，不能误套其他比赛的页数规则。主文件兼容XeLaTeX：XeTeX分支按**字体文件名**选择TeX Gyre Termes/Heros，因此在TeX Live、MiKTeX与Tectonic下都能找到字体；pdfLaTeX分支不受影响。没有安装或修改项目的金融依赖。Python绘图使用现有NumPy、Pandas、Matplotlib环境，本次只复现保存数值的表格和图形，不声称重建了历史金融训练环境。

### 本机编译环境故障（2026-10-19）

本机MiKTeX处于半更新状态：25.12的可执行文件依赖`MiKTeX251200-fmt/log4cxx/lzma2/bzip2/curl/expat/libressl`，而这些已被26.05版替换，26.05的三方库又依赖未安装的`MiKTeX260500-core`等模块。结果是所有TeX可执行文件（含`pdflatex`、`mpm`、`miktex-console`）均以`0xC0000135`（DLL缺失）退出，无法自修复。因此：

- 本目录下的交付PDF仍为上一版，**尚未用当前源码重建**；其SHA256见`validation.json`。
- 本次仅用自包含引擎Tectonic 0.17.0（XeTeX）做排版与分页预检：9页A4、第4节从第5页开始、无overfull/underfull、无未定义引用。预览产物不入库。
- **最终PDF请在另一台设备用上面的pdfLaTeX命令重建**，并逐页目检。第2页几乎满页（版心下界上方仅约2pt），若pdfLaTeX下有差异，删减第2节末句即可回收一行；第4页约有42pt余量。

若另一台设备也只有Tectonic，等价命令为`tectonic -X compile main.tex --outdir build`（字体已改为文件名选取）。

`build_formal_report.py`只读取配置列出的保存JSON/CSV及题目PDF字节（用于来源hash），不调用市场读取、模型拟合或最终评价入口。它核对发布策略、冻结身份、最终锁、源码身份、60个最终场景、五客户映射及保存核验，并确认来源文件生成前后hash一致。`evidence.json`记录来源与生成资产hash。正文中的常量和定性规则另按`METHODOLOGY.md`及现有实现核对。

## 当前验收与剩余事项

- **范围**：本次只精修第1–4页（Abstract、第1–3节）。第4–7节与参考文献未变动，哈希见`validation.json`。
- **数值**：正文与表格数字全部来自宏和`generated/*.tex`，未手工改动；`numbers.tex`等生成文件的数值内容与上一版一致（仅图片二进制因本机Matplotlib版本重新编码）。
- **合规补齐**：第2页显式分离日常波动（σ）与异常大损失（P90 D、CVaR90）并声明均为实际购买力口径；显式写出`t−1`因果信息集、单向计费与20单位→0.05算例；第3页给出R_ref定义顺序、q与验证预算关系、四项内层可行性口径与单位、440家族构成；第4页新增逐客户×逐产品筛选表（k/120与单侧上界，加粗=过全部规则）。
- **未验证**：本机无法用pdfLaTeX编译，PDF未重建；表格/页数仅为Tectonic预检结果。没有重跑`make test`或金融评估，本次证据检查见`validation.json`，不将历史103项测试冒充本次执行。
- **交付PDF**：仍为上一版，原始PDF已备份到`D:/MCM_report_backup_20261019/SafeInvest_report_draft_original.pdf`；重建后请刷新`validation.json`中的`pdf_sha256`与页数/字号测量。
- 成员与队名已补齐，PDF作者元数据同步使用三位成员姓名；AI披露只列本次明确使用的工具，团队应按完整实际使用情况修改，最终责任由成员承担。
- 本地旧源码ZIP及其收据不变；本报告没有自动写入旧归档，也未向比赛平台提交。
