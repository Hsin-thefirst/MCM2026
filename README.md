# SafeInvest

2026 MCM A 题：构建可解释、考虑风险和交易成本的自动投资顾问。

[题目](2026_MCM_SafeInvest.pdf) · [数据说明](docs/DATA.md) · [最新开发指标](benchmark/outputs/development_calibrated_v1/summary.csv) · [候选选择依据](benchmark/outputs/development_calibrated_v1/selection.json) · [工作记录](LOG.md)

当前配置评价35种策略，包含17种原参照、专家/神经/RL、路径风险预算、多期情景规划、细网格风险偏好与按同风险标准校准的简单门控。已修复奖励缺少历史最大回撤状态的问题，客户推荐关联实际策略，费用/账本/完整网格和严格匹配对照可核验。

本轮完成5批新开发研究，历史配置、失败结果及hash均保留。当前主候选`frontier_mpc`的30年中位实际CAGR为4.4068%、P10为1.6018%、P90最大回撤40.5324%，交易率60.43%；原Balanced年度分别为4.4965%、1.4726%、44.0277%、100%。简单风险校准门控为4.2821%/40.3133%，交易率41.95%。主候选具有可用的增长/风险取舍，但仍未通过严格多候选区间晋级，不能宣称普遍胜出或得到官方综合分。64项统一测试通过；最新35策略全部844,200次年度变化完成独立市场重放。

[冻结开发候选](benchmark/frozen_development_candidate.json)保存全部120条开发情景重拟合后的模型、参数、实际客户策略和比较列表。客户C在开发诊断中可选择该Balanced规则，A/D转人工，B/E为Conservative；学习策略的折外风险界不是冻结模型的覆盖保证。真实最终情景仍封存。按用户要求，当前不准备报告/论文，`generate_report=false`关闭实验自动生成HTML。

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
make model-audit RUN=benchmark/outputs/my_run
```

`make audit/replay/model-audit`默认指向`development_calibrated_v1`；`model-audit`独立重算每个训练期限的风险预算。已保存指标和决策可直接检查；报告生成须明确调用`make report`，本轮未为最新实验生成HTML。直接CLI调用应显式指定`--out`，避免使用历史默认目录。可通过`PYTHON=/path/to/python`指定解释器。

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
