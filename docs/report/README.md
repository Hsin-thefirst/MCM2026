# 正式英文报告草稿

[报告PDF](SafeInvest_report_draft.pdf) · [LaTeX主文件](main.tex) · [来源及资产校验](evidence.json) · [合并验收记录](validation.json)

本稿已合并两位组员对前5节的精修及本地第6、7节修订，采用最终冻结的校准Plan A：A/D人工、B/C Conservative、E Balanced。报告为9页A4英文正文，正文、表格及图中文字至少11pt；公式上下标使用正常数学排版。队名为Fen8FenQian，署名依次为Huang Sinan、Cai Xunzhi、Lin Zihan。

本次已用可用的MiKTeX pdfLaTeX从合并源码重新编译PDF并检查全部页面。此前其他设备的DLL故障和未重建状态保留在根LOG历史记录中，不再代表本次交付状态。

## 内容与位置

| 内容 | 位置 |
| --- | --- |
| 摘要、数据与最终选择 | 第1页 |
| 真实购买力、单向费用、波动与异常损失口径 | 第2页 |
| 三产品优化、凸性范围与嵌套选择 | 第3页 |
| 各期限权重、440家族筛选及逐客户逐产品表 | 第4页 |
| 四种动作规则、成本门槛及20词客户解释 | 第5页 |
| 等权/同锚比较及配对区间 | 第6页 |
| 实际客户期限、终值亏损与严重场景155 | 第7页 |
| 成本敏感性、核算验收与局限 | 第8页 |
| 产品建议、人工保障、参考文献及AI披露 | 第9页 |

报告共8表3图。筛选表由保存的开发结果生成，检查推荐产品等于合格产品中OOF中位增长最高者。风险收益图、费用图来自保存汇总；严重案例曲线来自第4、5节作者已提取的30条年度记录，仅连接年末观测。三图PDF/SVG/PNG与该作者的图形QA版本统一。详细前5节审阅说明见[REVIEW_PARTS_4_5.md](REVIEW_PARTS_4_5.md)。

第6、7节保留本地修订：`3.64e-12`是期末持仓金额重建误差（名义财富单位），780项标量参照仅针对等权；五档费率共用60条独立路径；明确客户筛选费率范围、选择性调仓建议和人工复核等待行为。

## 编译已有表格与图形

在项目根目录使用PowerShell执行以下命令，无需读取原始市场数据或重新评价策略：

```powershell
Push-Location docs/report
New-Item -ItemType Directory -Force build | Out-Null
1..2 | ForEach-Object {
    pdflatex --disable-installer --interaction=nonstopmode --halt-on-error -output-directory=build -jobname=SafeInvest_report_draft main.tex
    if ($LASTEXITCODE -ne 0) { throw "LaTeX compilation failed" }
}
pdfinfo build/SafeInvest_report_draft.pdf
pdftoppm -r 100 -png build/SafeInvest_report_draft.pdf build/page
# 检查页数、编译日志及全部页面后更新交付件。
Copy-Item -LiteralPath build/SafeInvest_report_draft.pdf -Destination SafeInvest_report_draft.pdf
Pop-Location
```

主文件保留XeLaTeX兼容分支，字体按文件名选取。模板由`5writing/templates/en/mcm-latex`按本题10页上限适配，去掉通用COMAP独立封面和目录。本次实际验收使用pdfLaTeX。

## 重建数据资产

配置位于[configs/report.json](../../configs/report.json)，正文位于`sections/`，参考文献位于`references.tex`。`generated/`中的宏和表格应由生成器维护。合并后的生成器同时保留前半稿的客户筛选表与后半稿的严重曲线、SVG导出和可选图形对齐检查接口。

```powershell
# 先从团队可信交付包恢复完整保存证据，参见根README。
.venv/Scripts/python.exe scripts/build_formal_report.py
```

完整资产重建需要配置列出的`benchmark/outputs/final_submission_v1/decisions.csv.gz`。Git迁出仓库不包含该大账本；已有图表仍可直接编译，`generated/severe_path.csv`保留30条紧凑提取记录。本次合并从已提交记录与图形检查峰谷、终值、复核时点和六次初始配置后交易，没有重新取得或解析最终市场。账本历史hash来自成员提交，当前缺失状态在验收JSON中明确记录；不冒称本次重新核验了完整账本。

重新生成图形后，需要重新检查字体、碰撞和对齐，不能沿用旧`figure_qa/`记录。`build(require_matplotlib_panel_alignment=None)`可接受成员原有的外部检查函数；普通构建不要求安装该技能。本次保留该成员的同源图形及QA，当前PDF另行检查版面和字号。

## 当前验收与剩余事项

- 已核对冻结方案、成员映射、表格/图形数字和来源；金融实现、冻结包、最终锁、历史manifest及依赖文件未改。
- 新增3项生成数据回归检查，覆盖严重路径筛选、记录完整性和复核后禁止自动交易，均通过且不读取真实最终集。隔离工作树的统一`make test`实际运行106项，105项通过；唯一失败是旧版也能复现的Windows路径分隔符断言（`scripts\\__pycache__`与`scripts/__pycache__`）。临时PYTHON启动适配器只修正Windows导入路径，不改变测试发现或断言；详情见`validation.json`。
- 9页A4，8表3图；交叉引用和逐页版面检查见`validation.json`。本次只编译已有证据，没有金融重训或最终评价。
- 团队仍需核实实际AI使用披露并审阅全文；本地旧源码ZIP未覆盖，也未向比赛平台提交。
