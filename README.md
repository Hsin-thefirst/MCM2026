# SafeInvest

2026 MCM A 题：构建可解释、考虑风险和交易成本的自动投资顾问。

[题目](2026_MCM_SafeInvest.pdf) · [数据说明](docs/DATA.md) · [基准报告](benchmark/outputs/development_consolidated/report.html) · [实验结论](docs/FINDINGS.md) · [工作记录](LOG.md)

已实现 17 种基准策略、年度回测引擎、嵌套情景交叉验证、客户期限评价、配对区间、成本敏感性及受控最终评价。开发结果中，成本感知策略增长更高，但回撤也更大；尚无门控候选同时满足预设增长与下行风险晋级条件。

**情景 1–120 用于开发，121–180 只供冻结后的最终评价。日常测试使用生成数据或开发分区。**

## 安装与运行

Python 3.10+；本次精确环境为 Python 3.13。所有命令在项目根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[analysis]'
make test
```

依赖范围只在 [pyproject.toml](pyproject.toml) 维护；[requirements.lock.txt](requirements.lock.txt) 保存验证过的精确运行版本，职责不同。精确复现时使用兼容的 Python，再执行 `python -m pip install -r requirements.lock.txt -e '.[analysis]'`。

```bash
# 每个新实验使用新目录，已有运行不会被覆盖
make benchmark RUN=benchmark/outputs/my_run
make report RUN=benchmark/outputs/my_run
make audit RUN=benchmark/outputs/my_run
make replay RUN=benchmark/outputs/my_run
```

打开输出目录中的 `report.html` 查看中文报告；英文图表标签便于比赛提交引用。`make report/audit/replay` 默认指向 `development_consolidated`。可通过 `PYTHON=/path/to/python` 指定解释器，也可直接执行 `python benchmark/run.py --help`。

配置统一在 [configs/benchmark.json](configs/benchmark.json)，客户问卷在 [configs/profiles.json](configs/profiles.json)。从其他目录调用时使用脚本和输出的绝对路径，默认输入按项目位置解析。真实最终评价不属于普通测试/实验命令，必须按 [实验协议](docs/PROTOCOL.md) 冻结后执行。

## 项目结构

```text
MCM2026/
├── README.md                  唯一项目入口
├── LOG.md                     唯一工作日志
├── AGENTS.md                  英文 agent 指令
├── docs/                      中文方法、接口、协议、结论与任务
├── configs/                   实验配置与客户问卷
├── tests/                     全部共享与基准测试
├── scripts/                   独立市场重放等工具
├── src/safeinvest/             共享类型与兼容加载器
├── benchmark/
│   ├── safebench/             基准实现
│   ├── outputs/               历次实验快照和图表（Git 忽略）
│   ├── experiment_registry.json
│   └── run.py                 实验入口
├── data/                      原始数据
├── Makefile                   统一执行命令
├── pyproject.toml             项目元数据与依赖范围
└── requirements.lock.txt      精确环境版本
```

历史运行是实验快照，保留其配置和 manifest，不改写旧代码 hash。新 checkout 的结果目录为空，需运行实验生成报告。原失败/修正记录及目录整合验证见 [实验登记](benchmark/experiment_registry.json)。

## 文档与协作

| 文档 | 唯一维护的内容 |
| --- | --- |
| [建模与指标](docs/METHODOLOGY.md) | 资产核算、指标、策略、适配假设及文献 |
| [代码接口](docs/INTERFACES.md) | 模块职责、扩展方法、结果格式 |
| [实验协议](docs/PROTOCOL.md) | 调参、统计解释、晋级、冻结及最终评价 |
| [实验结论](docs/FINDINGS.md) | 比较结果、严重案例、客户结果及局限 |
| [数据说明](docs/DATA.md) | 数据字段、分区、来源与许可 |
| [任务进展](docs/ISSUES.md) | 已完成工作与后续待办 |

开始前阅读根日志和接口；完成后在根日志追加执行者、改动、文件、验证、下一步。面向人的文档和日志用中文，agent 指令、标识符及开发注释用英文。公共接口变更同步代码与接口文档；PR 说明改动、验证和关联任务，并请另一位成员审阅。

提交材料还需团队姓名、来源、AI 使用披露及独立最终结果。正式英文报告≤10 页 A4、字号≥11pt，代码 ZIP≤50 MB；提交截止为 2026-10-20 23:59，北京时间。
