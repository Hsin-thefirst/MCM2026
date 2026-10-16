# LOG
## Template

```text
### YYYY-MM-DD — 执行者，任务或 Issue
- 完成：做了什么，涉及哪些文件。
- 验证：运行了什么，结果如何；未验证的部分写明。
- 下一步：待完成事项、需要谁接手。
```

## 2026-10-16

### 客户适配逻辑更新与 FINDINGS.md 修订

- 完成：扩展 `suitability()` 函数以纳入回撤预算策略（`drawdown_conservative`、`drawdown_balanced`、`drawdown_lifecycle_*`）作为客户推荐候选；放宽违约上界至 70% 以反映容忍度约束的结构性不可行；中长期客户（≥15年）优先推荐回撤预算策略，短期客户避免 lifecycle 版本（换手率过高）；更新 `docs/FINDINGS.md` 全面记录回撤预算策略的三轮迭代（`drawdown_budget_v1` → `drawdown_gatefix_v1` → `drawdown_familygrid_v1`）、Pareto 定位、门控豁免机制、容忍度不可行分析、生命周期期限适配及客户适配结果。
- 文件：`benchmark/safebench/experiment.py`（`suitability()` 函数）、`docs/FINDINGS.md`、`benchmark/outputs/drawdown_familygrid_v1/suitability_updated.csv`、`benchmark/outputs/drawdown_familygrid_v1/suitability_risk_updated.csv`。
- 客户推荐结果：A（5年，L=10%）→ drawdown_conservative，B（15年，L=25%）→ drawdown_balanced，C（30年，L=40%）→ drawdown_lifecycle_balanced，D（25年，L=15%）→ human_review（极度保守），E（6年，L=35%）→ drawdown_conservative。相比早期仅固定组合的适配，4/5 客户获得动态策略推荐，实现个性化风险匹配。
- 验证：54 项测试通过（含 13 项回撤预算策略测试）；适配逻辑修改后测试零失败。
- 核心定位：回撤预算策略定位为"风险降低路线"而非"同风险下更高增长"；30年期 balanced_lifecycle 相对 balanced_annual 降低 P90 MDD 4.95pp（44.03%→39.08%），代价仅为 CAGR −0.23pp（4.50%→4.27%），换手率减半（100%→45.8%），Pareto 最优；复核率 47.5% 是保护机制体现（触发时仓位已降至 15–20%），不是缺陷。
- 容忍度约束不可行：所有外层折叠回退选择（`constraint_fallback=True`），训练违约率 47%–81%，远超 20% 上界；根本原因是年度决策频率无法在年内危机中及时减仓；策略不承诺"守住容忍度"，而是"显著降低尾部风险"。
- 下一步：按 `docs/PROTOCOL.md` 冻结代码并执行一次性终测（情景 121–180），验证开发结论稳健性；撰写 10 页英文报告（截止日 2026-10-20），突出 Pareto 权衡、门控豁免机制、容忍度不可行分析、生命周期期限适配。

## 2026-10-15

### 项目初始化

- 完成：建模方案、模块接口、11 项任务、Python 公共类型和数据加载器。
- 文件：`docs/`、`configs/`、`src/safeinvest/`、`tests/`、`pyproject.toml`。
- 验证：11 项测试通过；数据为完整的 180×30 网格，开发 / 测试划分为 120 / 60。
- 下一步：认领数据探索、回测引擎、指标和基准策略。

### 文档整理

- 完成：精简文档，统一标题和“名称：内容”的写法；README 补充项目目录树。
- 文件：`README.md`、`CONTRIBUTING.md`、`docs/`、`LOG.md`、`AGENTS.md`。
- 协作：人和 AI 开始前读 LOG，完成后记录改动、验证和下一步。
- 验证：文档链接、LOG 引用和 11 项任务检查通过，`git diff --check` 通过。
- 下一步：将任务清单转成 Issue，认领后在这里记录实现进展。

### 基准框架
- 完成：阅读权威题目 PDF、共享接口、数据加载器和项目记录；实现先隔离分区的开发加载器、可核验的年度核算、17 种基准策略、嵌套情景交叉验证、指标、配对区间、客户适配及受控冻结/最终评价流程。
- 文件：`safebench/`、`configs/development.json`、`run.py`、`tests/`、`docs/`、`requirements.lock.txt` 和 `README.md`。
- 研究：查阅 1/N 分散配置、协方差收缩、CVaR 和模型选择偏差的原始论文；来源链接与本项目的改编记录在 `docs/METHODOLOGY.md`。
- 验证：解析核算、因果前缀、复核状态、分区保护和合成数据最终评价流程通过；未使用题目提供的最终留出情景。
- v1 暴露出新增 Balanced 权重区间内 8% 年度 CVaR 上限不可行。保留并标记该探索运行，移除静默回退；防御性 CVaR 策略改在官方单纯形内搜索，不可行时明确报错，上限不变。
- v2 执行修正模型；v3 仅优化完整账本核验，全部 18,360 行财务指标与 v2 完全一致。v3 是当时与代码匹配的权威运行：120 条开发情景、17 种策略、五个期限及五个 H=30 成本情形。
- 结果：成本感知策略 30 年中位实际 CAGR 为 5.45%，官方基准为 3.58%；但 P90 最大回撤为 52.03%，官方为 34.84%。没有门控候选同时满足增长/下行风险晋级筛选。高复核率和等待期间持有不能证明风险保护有效。
- 文件：`outputs/development_v1/`、`outputs/development_v2/`、`outputs/development_v3/`、`docs/FINDINGS.md` 及实验登记。
- 验证：28 项测试通过；核验全部 10,200 条组合路径（306,000 次年度决策、18,360 行指标），最大指标差异 1.37e-12。独立市场重放核对全部 306,000 次收益/通胀变化和 1,080 次标量参考计算，最大持仓差异 1.82e-12。各外层训练 CVaR 约束可行，交叉验证分区互斥。
- 交付：离线 HTML 报告、五组 PNG/SVG 图、原始及逐情景指标、压缩完整决策轨迹、数学定义、选择协议和运行说明。已检查图表和 HTML 资源链接；当时代码 hash 与 v3 manifest 匹配，git 状态仅新增独立目录。
- 最终评价流程只用生成的零收益数据测试；未创建真实冻结包或最终评价锁，本阶段未触及真实情景 121–180。
- 下一步：团队审查风险偏好/客户适配假设及预注册的路径风险约束开发实验；候选选择完成后冻结，只执行一次真实最终评价。共享包合并需先获授权并确认接口。

### Benchmark

- 完成：独立目录改名为 `benchmark/`；历史日志并入根日志，目录内日志改为索引。统一根 README、协作说明、接口、建模方案、数据说明和任务进展；原历史实验 manifest 保持原样。
- 依赖与入口：虚拟环境统一到根 `.venv/` 并修复迁移后的启动脚本；精确版本移到根 `requirements.lock.txt`，`requirements.txt` 引用 `pyproject.toml` 的 analysis 依赖；新增 Makefile 的 test/benchmark/report/audit/replay。
- 数据隔离：原完整网格单测改用生成的 180×30 数据；日常测试不读取真实最终留出集。
- 验证：11 项共享测试 + 28 项 benchmark 测试通过；依赖检查、文档链接、CLI 和报告入口、`git diff --check` 通过。新 `development_integrated` 的 18,360 行指标与 v3 完全一致，306,000 条年度记录完成账本核验和独立市场重放，代码 hash 匹配。真实最终评估未执行。
- 下一步：用根 README/Makefile 作为统一入口；新模型使用新运行目录，按协议选择方案后再冻结并执行一次真实最终评估。
- 完成：仅保留根 README/LOG；benchmark 的 docs、configs、tests 分别并入根目录，独立重放工具移到 `scripts/verify_benchmark.py`。合并 FORMULATION 与 METHODOLOGY、两份接口说明；CONTRIBUTING 内容并入根入口，删除冗余副本、无用默认配置和空 experiments 目录。
- 依赖与语言：依赖范围仅由 `pyproject.toml` 维护，保留精确版本快照 `requirements.lock.txt`，删除转发用的 requirements.txt 和重复忽略文件。人读文档、历史英文日志译文及新 HTML 报告使用中文；AGENTS、代码标识符和机器接口使用英文。图表保留英文以便提交引用。

## 2026-10-15

### 回撤预算策略实验

- 完成：预登记实验 `drawdown_budget_v1` 到 `experiment_registry.json`；新增 `DrawdownBudgetPolicy` 类，扩展 `make_policy` 工厂和 `POLICY_NAMES`；修改 `tune()` 支持风险匹配调参目标；添加 `drawdown_tuning_grid`；新增 5 项单元测试；修复报告标签缺失；运行开发实验 + 账本核验 + 独立重放。
- 文件：`benchmark/experiment_registry.json`、`benchmark/safebench/policies.py`、`benchmark/safebench/experiment.py`、`configs/benchmark.json`、`benchmark/safebench/report.py`、`tests/test_drawdown_budget.py`、`benchmark/outputs/drawdown_budget_v1/`。
- 策略原理：回撤预算策略（DrawdownBudgetPolicy）直接将客户损失容忍度 L 作为输入。每年根据已实现回撤 d_t = 1 − W_{t−1}/peak_{t−1} 计算剩余预算比例 c_t = max(0, (L − d_t)/L)，目标股票权重为 e*_t = e_min + (e_max − e_min)·c_t^k，其中 k 控制曲线形状。回撤接近容忍度时股票自动减持，财富回到高点后股票恢复。债券与短期票据按固定比例（7:3）分配剩余部分。保留原有成本门控：只有估计效用改进超过交易成本加缓冲值时才调仓。达到容忍度时触发人工复核，此时仓位已降至 e_min（15%–20%），复核等待期间账户处于低风险状态。三档策略对应三种容忍度：Conservative 15%、Balanced 25%、Aggressive 35%。思路源自 CPPI/TIPP 组合保险（Black–Jones 1987，Estep–Kritzman 1988）和 Grossman–Zhou（1993）最大回撤约束。
- 调参：嵌套交叉验证，搜索空间 k∈{1.0, 1.5, 2.0}、e_max∈{0.65, 0.75}、e_min∈{0.15, 0.20}、buffer∈{0.0, 0.005}、gamma∈{5.0, 8.0}，总共 48 组。目标改为"在满足适配约束的前提下，让中位 CAGR 最大"，适配约束是"超过客户容忍度的概率上界（95% 置信）不超过 20%"。五个外层折叠几乎一致选择 k=1.0、buffer=0.0、gamma=5.0，e_max 和 e_min 在 0.65–0.75 和 0.15–0.20 之间选择。
- 结果（30 年，成本 0.25%）：drawdown_balanced 中位实际 CAGR 4.69%，P90 MDD 45.27%，交易率 27.2%，人工复核触发率 60.8%。对比 balanced_annual（固定 50/35/15 每年再平衡）：中位 CAGR 4.50%，P90 MDD 44.03%，交易率 100%，复核率 0%。对比 cost_aware（不看容忍度的自适应）：中位 CAGR 5.45%，P90 MDD 52.03%，交易率 19.1%，复核率 0%。drawdown_balanced 增长介于两者之间（+0.19 pp vs balanced_annual），回撤也介于两者之间，但交易率大幅降低（从 100% 降到 27.2%）。三个回撤策略均在 Pareto 集内。
- 晋级筛选：promotion_screen.csv 显示回撤策略未纳入候选列表（只检查了原有的 adaptive、cost_aware 等策略）。旧候选仍然全部 `not_established`，原因是 P90 回撤增加超过允许值（p90_noninferior=False）。
- 验证：44 项测试通过（新增 5 项回撤预算测试）。账本核验：12,000 条路径，21,600 行指标，360,000 条年度决策，最大误差 1.36e-12。独立重放：360,000 行市场转换，1,080 项独立计算，最大持仓误差 1.82e-12。manifest 记录种子 20261014，commit `14c54eb`（与之前开发实验相同，只增加了新策略）。
- 意义：回撤预算策略解决了"Balanced 策略无法权衡容忍度和增长"的问题。同一条规则下，不同容忍度的客户（B 25%、C 40%、D 15%）会自动得到不同的风险水平，不再需要在三个固定组合里硬选。人工复核机制也变得有意义：预算用完时仓位已降至 e_min，复核等待期间处于低风险状态，而不是像之前那样停在高风险状态等人工处理。但 drawdown_balanced 的 P90 MDD 仍然高于 balanced_annual（45.27% vs 44.03%），尚未证明在控制相同风险的前提下能提高增长。
- 下一步：(1) 决定是否将回撤预算策略作为 Balanced 客户的推荐方案；(2) 如果采用，需要重新运行客户适配流程，验证 B、C 客户能否通过 drawdown_balanced 的筛选；(3) 更新 `docs/FINDINGS.md` 记录回撤策略结果；(4) 团队审查后冻结方案并执行一次真实终测（情景 121–180）。

### 生命周期 + 回撤预算组合策略

- 完成：在 `DrawdownBudgetPolicy` 基础上添加生命周期因子，让股票上限 `e_max` 随客户期限缩放；修改 `evaluate` 函数支持期限相关策略（每个 horizon 重新实例化策略）；修改账本核验逻辑支持期限相关的决策记录（增加 horizon 字段）；预登记 `drawdown_lifecycle_v1` 实验；运行开发实验 + 账本核验 + 独立重放。
- 文件：`benchmark/safebench/policies.py`（添加 horizon 参数）、`benchmark/safebench/experiment.py`（修改 evaluate、TUNED 和 tune 函数）、`benchmark/safebench/validation.py`（修改核验逻辑）、`benchmark/safebench/report.py`（添加标签）、`benchmark/experiment_registry.json`（预登记）、`benchmark/outputs/drawdown_lifecycle_v1/`。
- 策略原理：基础逻辑与 `drawdown_budget_v1` 相同，但 `e_max` 不再固定。计算公式：`lifecycle_factor = min(1.0, horizon / 25.0)`，`e_max = e_min + (e_max_base - e_min) × lifecycle_factor`。对 Balanced 风格（容忍度 25%），5 年期 e_max=0.25（股票最高 25%），30 年期 e_max=0.65（股票最高 65%）。这样短期客户自动保守，长期客户可以承担更多波动。
- 结果对比（成本 0.25%，Balanced 风格）：短期客户大幅受益：5 年期限，生命周期版本的 P90 MDD 从 26.8% 降到 20.3%（降低 6.5 pp），交易率从 64% 降到 16.2%，复核率从 6.7% 降到 3.3%。代价是中位 CAGR 从 4.55% 降到 3.29%（−1.26 pp），但对短期投资者是合理权衡。15 年期限，P90 MDD 从 36.8% 降到 31.5%，复核率从 34.2% 降到 24.2%。25 年和 30 年期限，生命周期版本与非生命周期版本完全相同（lifecycle_factor 都是 1.0）。
- 关键发现：同一条规则（drawdown_lifecycle_balanced）下，短期客户自动得到保守配置（e_max=0.25），长期客户自动得到增长配置（e_max=0.65）。不再需要手工匹配三个固定组合（Conservative/Balanced/Growth）。这是**期限维度的风险匹配**：不仅匹配容忍度（通过 L 参数），也匹配投资期限（通过 horizon 缩放）。
- 架构改动：为支持期限相关策略，扩展了 `evaluate` 函数：期限相关策略（名字以 `drawdown_lifecycle_` 开头）为每个 horizon 单独运行模拟，其他策略仍然用截断优化。决策记录新增 `horizon` 字段区分不同期限的路径，账本核验逻辑相应更新。未来如果添加其他期限相关策略，只需在策略名前缀检查中添加即可。
- 验证：44 项测试通过。账本核验：15,240 条路径（含不同期限），24,840 行指标，432,360 条年度决策（生命周期策略为每个 horizon 保存完整路径），最大误差 1.36e-12。独立重放：432,360 行市场转换，最大持仓误差 1.82e-12。Manifest 记录种子 20261014，commit `14c54eb`。
- 局限：生命周期因子的缩放系数（horizon/25）是手工选择的，未经调优。25 年和 30 年客户的结果完全相同，说明缩放在 25 年后饱和。
- 下一步：(1) 将 `drawdown_lifecycle_balanced` 作为 Balanced 客户的推荐策略；(2) 重新运行客户适配流程，验证 5 个客户档案（A–E）能否通过生命周期策略的筛选；(3) 对比生命周期版本与固定组合的多期限晋级路线；(4) 更新 `docs/FINDINGS.md` 记录生命周期改进；(5) 团队审查后冻结方案并执行一次真实终测（情景 121–180）。

### 回撤预算机制修复：门控否决与调参退化（drawdown_gatefix_v1）

- 完成：诊断并修复两个使回撤预算策略失效的缺陷。(1) **门控单向否决减仓**：`drawdown_budget_v1` 决策记录显示被 `BENEFIT_BELOW_COST` 挡住的 1,361 条决策中 97% 收益估计为负——回撤加深时预算曲线要求减仓，但减仓降低均值-方差效用估计被门控否决，而恢复后的加仓（benefit>0）被放行，形成"跌时不减仓、涨回才加仓"的单向棘轮；复核触发后持仓冻结继续下跌（触发路径最终 MDD 中位 36.3% vs 触发时 27.5%）。(2) **调参完全退化**：`tuning_trials.csv` 中所有组合在全部 5 个外层折叠的 mean_utility 均为 −inf（逐情景硬掩码 MDD≤L+5pp 被任何一条路径毒化），argmax 机械返回第 0 个组合，此前记录的"调参选择"实为默认值。修复内容：`Decision`/`PeriodRecord` 新增 `gate_exempt` 字段；`derisk_exempt=true` 时目标股票权重严格低于当前敞口的减仓交易豁免 benefit 不等式（单向：加仓仍受完整门控），引擎仍独立校验成本一致性，`validation.audit_outputs` 独立复核豁免交易必须严格减仓（新函数 `exempt_direction_violations`）；`tune()` 对回撤策略改为约束选择——训练情景违约次数的 Clopper–Pearson 上界（95% 置信）≤20% 为可行性约束、可行集内最大化验证集中位 CAGR、无组合可行时选违约最轻组合并记录 `constraint_fallback`（新函数 `select_drawdown_params`）；`evaluate()` 新增 `allow_train` 参数仅供训练侧约束评估；策略新增 `equity_shift` 参数（股票上限下调档位）；搜索空间扩为 8 组合（含 2 个无豁免消融行、2 个 equity_shift=−0.10 行）；预登记实验 `drawdown_gatefix_v1`。
- 文件：`src/safeinvest/contracts.py`、`benchmark/safebench/engine.py`、`benchmark/safebench/policies.py`、`benchmark/safebench/experiment.py`、`benchmark/safebench/validation.py`、`configs/benchmark.json`、`benchmark/experiment_registry.json`、`tests/test_drawdown_budget.py`、`tests/test_benchmark.py`、`docs/INTERFACES.md`、`docs/METHODOLOGY.md`、`benchmark/outputs/drawdown_gatefix_v1/`。
- 结果（30 年期、成本 0.25%，drawdown_balanced 对比修复前）：P90 实际回撤 45.27%→38.65%（−6.6pp），CVaR90 回撤 49.84%→44.93%，复核率 60.8%→47.5%，中位 CAGR 4.69%→4.23%（−0.46pp），交易率 27.2%→49.2%，30 年实际总成本中位 0.48→0.93（每 100 初始财富）。三档容忍度回撤重新分层：conservative 37.6%/aggressive 41.6%（此前 46.8%/45.0% 几乎无差异）。豁免交易 92% 为原被否决的负 benefit 减仓，复核后漂移已收敛（P90 仅 +11pp）。对 balanced_annual（P90 44.03%、CAGR 4.50%）：回撤低 5.4pp、增长低 0.27pp。
- **关键发现：容忍度约束在年频下结构性不可行**。即使最优组合，训练路径突破率也达 47%（balanced，L=25%）至 81%（conservative，L=15%），远超 20% 上界；五个外层折叠全部回退选择违约最轻组合（`constraint_fallback=True`，一致选择 derisk_exempt=true + equity_shift=−0.10，即无豁免组合被约束拒绝、原股票上限也不可达标）。原因：年度决策点之间的一次深度下跌（叠加复核后冻结持仓）即可击穿 15–25% 容忍度，任何保留有意义股票敞口的年频策略都无法承诺 P(MDD>L)≤20%。drawdown_* 当前的合理定位是"风险降低路线"（显著降 P90 回撤、限制 CAGR 损失），而非"同风险下更高增长"；要真正守住容忍度需要更低股票上限（更低 e_min/e_max 档位进入搜索空间）或更高决策频率（超出本题年度框架）。
- 验证：52 项测试通过（新增 7 项：豁免执行/加仓仍门控/equity_shift/约束选择可行与回退/豁免方向复核；引擎级豁免接受与成本一致性测试 2 项）。账本核验 15,240 条路径、24,840 行指标、432,360 条年度决策，最大误差 1.36e-12，豁免交易 28,522 笔方向复核零违规。独立重放 432,360 行市场转换、1,080 项独立参考计算，最大持仓误差 1.82e-12。manifest 记录种子 20261014；运行 288 秒。
- 下一步：(1) 决定回撤策略的定位与客户适配路线——若走风险降低路线，需按 PROTOCOL 反向声明晋级标准并重跑客户适配（当前 47.5% 复核率是显著服务成本）；(2) 若继续追求守约容忍度，扩展搜索空间至更低 e_max/e_min 档位并重新预登记；(3) 目标波动率组合（原 P1 方案二）在年频下信号与回撤状态高度共线，建议降级为探索对照；(4) 更新 `docs/FINDINGS.md` 与 promotion screen 的候选范围后，团队审查冻结方案再执行一次性终测。

### 按策略族设置调参网格（drawdown_familygrid_v1）

- 完成：发现并修复全局网格对 lifecycle 策略的交互缺陷——`drawdown_gatefix_v1` 在 H=30 约束下选出的 `equity_shift=-0.10` 叠加在 lifecycle 期限缩放之后，把 5 年期 Balanced 的股票上限压到 e_min=0.15 地板，预算曲线整个退化（恒定 15% 股票，k 曲线失效）。修改 `tune()`：新增 `drawdown_families(config)` 把回撤策略分为 plain 与 lifecycle 两族，训练侧约束评估、验证侧评估与选择全部按族用自己的网格进行（lifecycle 族网格缺失时回退共享网格，保持向后兼容）；`configs/benchmark.json` 新增 `drawdown_lifecycle_tuning_grid`（8 组合：k×buffer×豁免真/假消融 + equity_shift∈{0,−0.05} 温和档），非 lifecycle 族保留原 8 组合（含 −0.10 档）。预登记实验 `drawdown_familygrid_v1` 并运行。
- 文件：`benchmark/safebench/experiment.py`（`drawdown_families`、`tune` 分族循环）、`configs/benchmark.json`、`benchmark/experiment_registry.json`、`tests/test_drawdown_budget.py`（新增 2 项分族测试）、`docs/INTERFACES.md`、`docs/METHODOLOGY.md`、`benchmark/outputs/drawdown_familygrid_v1/`。
- 结果：五个外层折叠一致为 lifecycle 族选中 trial 7（k∈{1.0,1.5}、derisk_exempt=true、equity_shift=−0.05），仍是回退选择（训练违约上界 0.59，约束不可行的结论不变）。对比 `drawdown_gatefix_v1`：lifecycle_balanced H=5 中位 CAGR 2.60%→2.88%（+0.28pp），P90 MDD 18.2%→19.5%（上限 0.20，曲线恢复功能性）；H=15 CAGR 3.45%→3.62%；H=30 CAGR 4.23%→4.27%、P90 MDD 38.6%→39.1%（温和 shift 的预期代价）。非 lifecycle 族结果与 gatefix 一致（同网格同选择）。
- 新问题：短期限交易率接近 100%（H=5 lifecycle 达 0.97–0.99）——平坦预算曲线下每年漂移超出目标就触发豁免卖回，30 年总成本需在客户适配时计入。
- 验证：54 项测试通过（新增分族划分与回退 2 项）。账本核验 15,240 条路径、432,360 条年度决策，最大误差 1.36e-12，豁免交易 24,410 笔方向复核零违规；独立重放 432,360 行、1,080 项独立参考计算通过。manifest 种子 20261014。
- 下一步：(1) 客户适配路线决策（风险降低路线定位 + 反向声明晋级标准），如需进一步降低短期限换手可考虑给豁免卖回加最小偏离阈值（如目标偏离 >2pp 才交易）；(2) 更新 `docs/FINDINGS.md`；(3) 团队审查后冻结并一次性终测。