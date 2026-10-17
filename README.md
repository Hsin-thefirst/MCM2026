# SafeInvest

2026 MCM A 题：构建可解释、考虑风险和交易成本的自动投资顾问。

[题目](2026_MCM_SafeInvest.pdf) · [数据说明](docs/DATA.md) · [基准数值](benchmark/outputs/development_consolidated/scenario_metrics.csv) · [实验结论](docs/FINDINGS.md) · [工作记录](LOG.md)

已实现17种基准策略、计划A的8种产品/对照、B的10种信号挑战及C的4种规划对照；提供年度回测引擎、嵌套情景交叉验证、客户期限评价、配对区间、成本敏感性及受控最终评价。计划A增加连续凸路径风险配置、成本门控部分再平衡与实际策略的客户适配；目前仍无候选通过预设晋级条件。

2026-10-16：[解题路线与研发计划](docs/STRATEGY_PLANS.md)补充了固定配置、部分再平衡、弱预测信号及回撤反馈的小规模验证；用于选择下一步研究方向，不替代正式Benchmark晋级。当前仅解题，暂停报告与论文制作。

2026-10-17：计划A首轮开发完成，结果见[实验记录](docs/FINDINGS.md#计划a首轮开发2026-10-17)。H30均衡版中位实际增长4.6206%、P90回撤42.5352%、后续交易率54.48%；风险预算校准尚未通过。当前配置不自动生成报告；最终源码对应运行`development_plan_a_v2`，年度顺序与约束诊断位于`plan_a_diagnostics_v1`。

同日后续：[风险预算校准](docs/FINDINGS.md#计划a风险校准2026-10-17)分开固定验证目标与训练安全余量，同一风险线下内层合格产品/折从7/15增至15/15。均衡版增长4.3401%、P90回撤42.2199%，降低部分风险并付出收益代价，仍未通过统计晋级。该轮校准源码和配置对应`development_plan_a_calibrated_v1`；`plan_a_calibrated_diagnostics_v1`保存执行分解与顺序压力，`plan_a_calibrated_comparison_v1`保存同风险线对照。

计划B/C实现位于独立分支`feat/plan-bc-evaluation`。B以固定Ridge股票相对bills信号挑战A，分开窄/宽动作与无信号对照；C以相同情景和动作比较一步/三步规划，并追踪历史最大实际回撤。配置及新接口见[接口约定](docs/INTERFACES.md#计划bc接口2026-10-17)，两批首测与独立核验已完成；提交版本65项统一测试通过；最终运行`development_plan_b_v3`及`development_plan_c_screen_v3`已完成独立核验，与v1/v2全部金融指标精确一致。默认配置仍运行原25策略。H30所选B窄/宽增长4.3504%/4.3383%，C一步/三步4.2918%/4.2644%，均未建立晋级证据；C按预声明停止全期限扩展，详见[本轮结论](docs/FINDINGS.md#计划b与c开发验证2026-10-17)。

**情景1–120用于开发。121–180已于2026-10-17按冻结方案评价一次，最终锁已保存；不得重跑最终集或据其结果调参。日常测试仍只使用生成数据或开发分区。**

提交前[有限优化计划](docs/STRATEGY_PLANS.md#2026-10-17-面向提交的最后一轮优化计划)已在`feat/submission-policy`执行：八条执行规则与一个内层选择程序，保留原25策略，共33策略、51,480条指标及789,840条年度记录。74项统一测试、独立市场/优化证书/内层选择/限幅核验全部通过。S8的开发H30增长4.3953%、P90回撤41.8682%，未通过晋级，按预声明保留校准A；客户E改为均衡，其余A/D人工、B/C保守。全120重拟合及实际客户目标核验后冻结，最终60情景只评价一次：A的H30中位实际增长4.1548%、P10增长1.6443%、P90回撤48.0160%、后续交易率35.34%；等权年度为3.3761%/1.4987%/39.5260%/100%。增长和风险都更高，未据最终结果改选。详见[本轮结果](docs/FINDINGS.md#提交候选有限优化2026-10-17)与[最终评价](docs/FINDINGS.md#冻结与一次性最终评价2026-10-17)。报告与论文继续暂停。

复现此版使用`configs/submission.json`，开发运行`benchmark/outputs/development_submission_v1`；默认benchmark仍只评价原25策略。冻结、数值核验和源码包入口见[提交与复现](docs/SUBMISSION.md)。

正式[源码ZIP](dist/SafeInvest_source.zip)为45,514,426字节（45.5MB），含117文件；[交付校验](dist/SafeInvest_source.verification.json)记录CRC/SHA256、解压后的74项统一测试及字节完全相同的独立重打包。归档保留冻结包和最终锁，论文继续暂停。

用户追加授权的跨分支研究在`feat/strategy-fusion-research`完成：参考Cai与MPC代码，固定三条融合规则，开发H30回撤融合为4.3244%增长/40.4352% P90回撤，相对A增长略降、回撤点估计下降，但费用增加51.6%，未通过原增长/风险筛选。78项测试及77,760条年度核算通过。运行`development_fusion_v1`属于最终集已暴露后的探索，原提交分支、源码ZIP、冻结包、客户分配及最终锁保留；没有新增最终评价。详见[融合结论](docs/FINDINGS.md#跨分支融合开发研究2026-10-17)。

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

当前`generate_report=false`，实验只保存数值证据；显式执行`make report`才生成HTML及图表。`make report/audit/replay`默认指向历史`development_consolidated`，新实验请传`RUN`。可通过`PYTHON=/path/to/python`指定解释器，也可直接执行`python benchmark/run.py --help`。

计划A的独立优化核验：`.venv/bin/python scripts/verify_plan_a.py benchmark/outputs/development_plan_a_calibrated_v1`。固定已选参数的诊断复现：`.venv/bin/python scripts/diagnose_plan_a.py benchmark/outputs/development_plan_a_calibrated_v1 --out benchmark/outputs/my_plan_a_diagnostic`。

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
| [提交与复现](docs/SUBMISSION.md) | 冻结版本、结果核验与源码ZIP入口 |

开始前阅读根日志和接口；完成后在根日志追加执行者、改动、文件、验证、下一步。面向人的文档和日志用中文，agent 指令、标识符及开发注释用英文。公共接口变更同步代码与接口文档；PR 说明改动、验证和关联任务，并请另一位成员审阅。

提交材料还需团队姓名、来源、AI 使用披露及独立最终结果。正式英文报告≤10 页 A4、字号≥11pt，代码 ZIP≤50 MB；提交截止为 2026-10-20 23:59，北京时间。
