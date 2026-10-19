# 提交、打包与组员交接

唯一日常入口为根[README](../README.md)，交付参数统一在[configs/delivery.json](../configs/delivery.json)。本次整理没有修改金融策略、客户分配或一次性最终结果。

## 当前交付版本

- 日常提交包：`dist/SafeInvest_main_source.zip`，包含本次整理后的源码、最新讲解、原冻结程序、默认复现证据和后续研究紧凑结果。
- 包外收据：`dist/SafeInvest_main_source.verification.json`记录实际大小、SHA256与文件数；`dist/SafeInvest_main_source.reproduction.json`记录独立解压测试和重打包。
- 最终金融程序：`plan_a_balanced`，全120重拟合参数risk_ratio=0.95、tracking_gamma=4；A/D人工、B/C保守、E均衡。
- 最新组员讲解：[SOLUTION_GUIDE.md](SOLUTION_GUIDE.md)。它包含数据解释和写report知识，不是正式英文论文。
- 历史包：`SafeInvest_source.zip`、`SafeInvest_final_source.zip`及`SafeInvest_submission.zip`仍在dist保留原hash；早期v1/v2草稿移到`dist/archive/`。日常交付使用上面的唯一新包，历史包原74/95项测试结论保持原义。

整理后的源码与新包测试数量以本次独立复现收据为准；讲解中的98项属于最后客户风险研发快照，不要和历史包或本轮测试数量混用。

## 组员只看结果

先读根README，再读SOLUTION_GUIDE第7–10节的Benchmark/客户/严重案例/费用，最后读第12–15节的新研究、风险边界和report表述。若需公式与假设，以METHODOLOGY为准；若需解释为何未采用新版本，以FINDINGS及保存的decision为准。

最终60情景的代表数值可直接看[最终汇总](../benchmark/outputs/final_submission_v1/summary.csv)及[五客户](../benchmark/outputs/final_submission_v1/suitability.csv)。完整逐路径指标/账本位于本地或ZIP；恢复后再打开讲解末尾相应大文件链接。

## 安装与检查

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt -e '.[analysis]'
make check
make test
```

`make check`核对冻结包完整性、金融源码、精确依赖、最终锁、发布决定及必要文档，只读已有材料。Git checkout若缺大证据会明确说明；不重新评价最终集。

`make test`是统一测试入口。`PYTHON=/path/to/python`可以覆盖解释器；Windows可用兼容shell，或直接使用`python scripts/project_tools.py <action>`。

## 从源码生成比赛代码ZIP

在拥有完整保存证据的工作区执行：

```bash
make check
make test
make package
make verify-package
```

默认输出`dist/SafeInvest_main_source.zip`。已有包不会被覆盖；新快照使用`make package PACKAGE=dist/SafeInvest_<新版本>.zip`。每次包中的README/LOG/代码都是构建时快照，不会随之后的Git提交自动改变。

包内包括：公共/金融/研究源码，配置，测试，中文文档，依赖锁，题目PDF及原数据的原样副本，冻结模型及最终锁，完整原最终账本，默认复现所需数值及历史参照，以及后续研究汇总/失败约束/核验结果。后续研究的大训练缓存、开发年度大账本、虚拟环境、Git历史、缓存和旧ZIP不入包。

原始数据只按字节复制和校验，不用于新最终分析。包必须小于50,000,000字节；所有文件SHA256和ZIP CRC均核验。`.gitignore`、`.gitattributes`也随包保存，便于组员解压后继续协作。

独立解压后在兼容精确环境运行`make test`、`make check`、`make guide`，再按同一配置重打包到不同文件名，比较全部字节。测试读取解压目录自身模块，不借用原仓库代码。记录见包外reproduction JSON。

## Git clone后恢复大证据

Git只带关键紧凑结果，不传完整实验缓存。拿到团队验证的提交ZIP后：

```bash
make verify-package PACKAGE=/path/to/SafeInvest_main_source.zip
make restore PACKAGE=/path/to/SafeInvest_main_source.zip
make check
make guide
```

恢复前验证ZIP安全路径、成员清单、全部SHA256及旁边已有的verification收据；再核对包内冻结模型/最终锁与当前仓库相同。只恢复`benchmark/outputs/`：不会覆盖源码、配置、文档、原数据或锁。任何目标证据不同均在写入前停止；已有相同文件保留。

SHA256自洽并不代替来源可信；通过团队认可的渠道获得ZIP和收据。Git push不包含ZIP，组员通过上述交付包补齐大材料。

## 本地清理

```bash
make clean
```

只清除项目内Python、测试和绘图缓存；不进入`.venv/`、`.git/`、`benchmark/outputs/`或`dist/`，不跟随符号链接，Git跟踪文件受到保护。历史实验约1.9GB保留用于追溯，不参与日常Git和代码ZIP。不要使用`git clean -fdx`代替此命令，它会删掉被忽略的本地证据和虚拟环境。

## 默认金融开发的复现

如确需复现实验，先在实验登记追加复现记录，再用不存在的新目录：

```bash
.venv/bin/python benchmark/run.py development --config configs/submission.json --out benchmark/outputs/development_submission_reproduction
.venv/bin/python scripts/verify_benchmark.py benchmark/outputs/development_submission_reproduction
.venv/bin/python scripts/verify_plan_a.py benchmark/outputs/development_submission_reproduction
.venv/bin/python scripts/verify_submission.py benchmark/outputs/development_submission_reproduction
```

后续客户/融合研究需要工作区的对应训练模型和大缓存，ZIP只带紧凑研究证据；不声称ZIP可独立重跑所有研究。详细入口见INTERFACES和SOLUTION_GUIDE。

## 提交前由团队完成

正式英文主report草稿已生成：[9页PDF](report/SafeInvest_report_draft.pdf)，[修改与重建说明](report/README.md)。按题目要求≤10页A4、字号≥11pt；队名Fen8FenQian和成员Huang Sinan、Cai Xunzhi、Lin Zihan已署名，仍需核实来源与AI使用披露并审阅全文。代码ZIP≤50MB，旧源码包没有随此次文稿自动更新。由队长在2026-10-20 23:59北京时间前提交，具体题目与平台要求见[题目PDF](../2026_MCM_SafeInvest.pdf)。

真实最终场景121–180已经一次性使用。不得删除最终锁、重跑final入口或根据已见最终表现改选金融模型。论文起草、整理和打包不代表已经向比赛平台提交。
