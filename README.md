# SafeInvest

2026 MCM Senior Group A题：可解释、考虑购买力风险和交易费用的自动投资顾问。

**当前提交金融程序：冻结的校准A；客户A/D人工、B/C保守、E均衡。** 后续研究没有通过完整晋级，详见讲解中的风险与收益取舍。真实最终集已使用一次，不能重跑或用来改选模型。

## 从这里开始

| 你要做什么 | 入口 |
| --- | --- |
| 组员理解方法、数据、五客户和写report要点 | [解题讲解与数据使用指南](docs/SOLUTION_GUIDE.md) |
| 阅读正式英文报告草稿、修改署名与重建 | [报告PDF](docs/report/SafeInvest_report_draft.pdf) · [编写与核验说明](docs/report/README.md) |
| 打包、交付、恢复大证据、核对版本 | [提交与复现](docs/SUBMISSION.md) |
| 核对最新研究是否被采用 | [实验结论](docs/FINDINGS.md) |
| 查数学定义、接口、实验规则 | [方法](docs/METHODOLOGY.md) · [接口](docs/INTERFACES.md) · [协议](docs/PROTOCOL.md) |
| 看历史过程或剩余工作 | [工作日志](LOG.md) · [待办](docs/ISSUES.md) |
| 看原始要求和数据来源 | [题目PDF](2026_MCM_SafeInvest.pdf) · [数据说明](docs/DATA.md) |

正式英文report已起草为9页A4，正文、表格及图中文字≥11pt；队名Fen8FenQian及三位成员署名已补齐。2026-10-19已精修前4页（Abstract、第1–3节），但交付PDF尚未用当前源码重建：本机MiKTeX半途更新后无法编译，需在另一台设备用pdfLaTeX重建并逐页目检，说明见[编写与核验说明](docs/report/README.md)。仍需团队审阅AI披露与全文。代码ZIP须≤50MB，历史源码包未随报告自动更新。截止2026-10-20 23:59北京时间，以题目PDF为准。

## 安装和常用命令

Python 3.10+；精确验证环境使用Python 3.13。所有命令在项目根目录执行：

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt -e '.[analysis]'
make help
make check
make test
```

`pyproject.toml`维护依赖范围，`requirements.lock.txt`维护精确验证环境；不另外维护requirements范围列表。

```bash
make package          # 当前源码、讲解、冻结模型和证据 → dist/SafeInvest_main_source.zip
make verify-package   # 校验ZIP内全部文件hash和外部收据
make clean            # 只删Python/绘图缓存，保留环境、全部实验与交付包
make guide            # 从保存数值重建讲解，需要完整原保存证据
```

`make package`不会覆盖已有ZIP；需另建版本时使用`make package PACKAGE=dist/<新文件名>.zip`。打包只复制保存材料，不运行金融策略或最终评价。

## Git仓库与本地大文件

Git包含源码、配置、题目/数据、讲解、冻结模型、最终锁及关键紧凑数值。虚拟环境、完整实验账本/缓存和ZIP均被忽略；无需Git LFS，不要直接压缩整个工作目录。

刚clone后可以阅读文档、运行测试及`make check`。完整原始指标和账本不随Git传输；打包、重建讲解或重放账本前，从团队交付的可信ZIP恢复：

```bash
make verify-package PACKAGE=/path/to/SafeInvest_main_source.zip
make restore PACKAGE=/path/to/SafeInvest_main_source.zip
make check
```

恢复工具只写`benchmark/outputs/`，核对冻结模型与最终锁，遇到不同的已有证据会停止，不覆盖代码。新ZIP解压目录已包含默认提交程序的完整证据，可直接测试/重建讲解；全部后续研究的大训练缓存仍只在原工作区。

本版经最终复核后合入`main`。组员在各自feature分支修改；提交代码、配置、测试和文档，勿将`.venv/`、整个`benchmark/outputs/`或`dist/*.zip`强行加入Git。历史研发分支与工作记录见LOG和FINDINGS。

## 目录职责

```text
README.md / LOG.md       项目入口 / 唯一工作日志
src/safeinvest/          公共类型与数据接口
benchmark/safebench/     冻结金融实现及原Benchmark
benchmark/outputs/       紧凑证据已入Git；大结果/训练缓存被忽略
benchmark/*.json        冻结模型、最终锁、实验登记
configs/                输入与实验/交付配置
scripts/                开发、核验、讲解生成与打包工具
tests/                  统一测试，使用生成数据或开发分区
docs/                   团队讲解、方法、接口、协议、结论与提交说明
data/                   题目提供的原始数据
dist/                   本地提交ZIP与校验；archive/存放早期草稿
```

## 需要继续开发时

```bash
# 每个开发实验必须登记，并使用不存在的新目录
make benchmark RUN=benchmark/outputs/<新实验名>
make audit RUN=benchmark/outputs/<实验名>
make replay RUN=benchmark/outputs/<实验名>
```

默认benchmark为原25策略；复现实际提交版使用`configs/submission.json`，具体命令见[提交说明](docs/SUBMISSION.md)。默认不生成报告；`make report RUN=...`仅在明确需要图表展示时使用。

开发仅使用场景1–120。121–180已于2026-10-17按冻结方案评价一次；保留`benchmark/FINAL_EVALUATION_LOCK.json`和历史manifest，不修改其hash。`make test/check/package/restore/clean/guide`均不重新运行真实最终评价。
