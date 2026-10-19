# SafeInvest

2026 MCM A 题：构建可解释、考虑风险和交易成本的自动投资顾问。

[题目](2026_MCM_SafeInvest.pdf) · [数据说明](docs/DATA.md) · [基准报告](benchmark/outputs/development_consolidated/report.html) · [实验结论](docs/FINDINGS.md) · [工作记录](LOG.md)

已实现 17 种基准策略、年度回测引擎、嵌套情景交叉验证、客户期限评价、配对区间、成本敏感性及受控最终评价。开发结果中，成本感知策略增长更高，但回撤也更大；尚无门控候选同时满足预设增长与下行风险晋级条件。

上一轮[路径风险约束选择研究](benchmark/outputs/development_path_risk_utf8/comparison/report.html)中，回撤降低伴随增长下降，20个策略族/外层折中18个无可行参数。

上一轮[固定目标与情景效用效率研究](benchmark/outputs/development_efficiency_verified/efficiency_report.html)已执行用户确认的180次预算。一年候选中位实际增长4.46%、P90回撤42.14%、交易率14.34%；年度Balanced参照为4.50%、44.03%、100%。交易与费用改善通过预声明区间，增长和回撤非劣性尚未确立，两候选都不满足整体采用条件。

上一轮[反事实离线强化学习研究](benchmark/outputs/development_counterfactual_fqi_v2/learning_report.html)完成90次候选/内折评价和80个模型训练/重拟合。学生树中位实际增长4.87%、P90回撤50.37%、交易率30.14%；年度Balanced为4.50%、44.03%、100%。三报告策略各0/5内层选择可行，条件配对区间显示回撤恶化；增长伴随更高权益敞口，未通过风险路线与客户适配。

最新[小型神经网络路径约束研究](benchmark/outputs/development_neural_path/neural_report.html)完成120个网络训练、240个成员检查点和90次候选/内折评价。完整MLP中位实际增长4.42%、P90回撤41.91%、交易率12.44%；同约束线性策略为4.41%、40.93%、11.09%。三类候选均为0/5内层选择可行，训练约束及18项调整区间的采用条件也未通过；回撤点估计改善尚未确立，不能作为Balanced升级。独立网络/约束/部署重放及八张PDF检查通过，完整依据见[实验结论](docs/FINDINGS.md)。

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

# 同预算比较效用选择与路径风险约束选择，只使用开发情景
make study RUN=benchmark/outputs/development_path_risk_study

# 已确认的固定目标、漂移带与情景效用效率研究
make efficiency RUN=benchmark/outputs/development_efficiency

# 已登记的反事实离线强化学习与学生树开发研究
make learning RUN=benchmark/outputs/development_counterfactual_fqi_v2

# 小型神经网络、路径风险约束和同约束消融
make neural RUN=benchmark/outputs/development_neural_path
```

Windows 可指定本地解释器，如 `make test PYTHON=C:/Python313/python.exe`；测试入口在 Python 内设置源码路径。`make study` 使用 `configs/path_risk.json`，保存双运行、逐情景调参记录及配对研究报告；已有目录会拒绝覆盖。

`make efficiency` 使用 `configs/efficiency.json`，只在开发集比较一年/三年情景估值；主报告为运行目录中的 `efficiency_report.html`，完整参照为 `report.html`。保存全部内层训练片段、180次候选/内折评价依据、效率区间和客户期限诊断。该运行的 `make replay` 还会独立重建训练片段并复算效益/费用门槛。

打开输出目录中的 `report.html` 查看中文报告；英文图表标签便于比赛提交引用。`make report/audit/replay` 默认指向 `development_consolidated`。可通过 `PYTHON=/path/to/python` 指定解释器，也可直接执行 `python benchmark/run.py --help`。

`make learning` 使用 `configs/learning.json`，按客户期限分别模拟反事实拟合Q迭代教师、深度2学生及去除滞后市场特征的消融。主报告为 `learning_report.html`；保留内外折反事实数据、树模型、全部内层选择和逐期限账本。`make replay`同时独立复核名义金额后继、Bellman/蒸馏与部署动作。该候选只开放开发研究，真实冻结/最终评价入口拒绝执行。

`make neural`使用`configs/neural.json`，比较同约束线性logits、16×16完整MLP及持仓特征MLP；保存所有训练种子、检查点和路径约束状态。训练需要`python -m pip install -e '.[analysis,neural]'`；CPU精确版本及官方wheel索引统一记录在根lock。Windows项目解释器为`.venv/Scripts/python.exe`，Linux为`.venv/bin/python`；Makefile按系统选择默认路径。主报告为`neural_report.html`，`make replay`追加独立NumPy网络/约束/动作核验。该研究只使用开发分区，不开放真实冻结/最终评价。

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
