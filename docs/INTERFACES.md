# 代码接口与结果格式

本文件合并公共接口与原 benchmark 扩展说明。数学口径仅在 [METHODOLOGY.md](METHODOLOGY.md) 维护，实验程序仅在 [PROTOCOL.md](PROTOCOL.md) 维护。

## 模块职责

| 位置 | 职责 |
| --- | --- |
| `src/safeinvest/contracts.py` | v1 公共 dataclass、Enum、Policy 协议和校验 |
| `src/safeinvest/data.py` | 完整 180×30 网格校验/排序；兼容接口，日常测试仅用生成数据 |
| `benchmark/safebench/data.py` | 开发分区读取、情景折划分、实际收益转换 |
| `benchmark/safebench/engine.py` | 交易核算、因果上下文和年度模拟 |
| `benchmark/safebench/policies.py` | 17 种策略、训练估计及策略工厂 |
| `benchmark/safebench/experts.py` | 三专家、两层神经门控、监督初始化、REINFORCE 及三种子集成 |
| `benchmark/safebench/frontier.py` | 路径风险预算、情景价值、线性/神经近似及严格匹配参照 |
| `benchmark/safebench/cushion.py` | 细网格、平滑回撤风险偏好、逆优化先验与同风险标准校准门控 |
| `benchmark/safebench/metrics.py` | 路径指标、聚合、配对区间和 Pareto |
| `benchmark/safebench/experiment.py` | 嵌套 CV、多期限/成本评价、适配、冻结/最终流程 |
| `benchmark/safebench/validation.py` | 独立账本核验及晋级检查 |
| `benchmark/safebench/report.py` | 中文 HTML 展示与可用于英文提交的图表 |
| `scripts/verify_benchmark.py` | 独立市场重放和标量等权计算 |
| `configs/benchmark.json` / `configs/profiles.json` | 唯一实验配置 / 客户问卷 |

两种加载器职责不同：完整网格加载器保留既有公共 API，分区加载器为开发隔离而只解析所选数据。不要先全量读取真实 CSV 再划分来做开发。benchmark 代码复用公共类型；没有第二套 contracts。

## 调用约定

资产顺序固定为 `equity,bond,bill`，持仓为名义金额，权重非负且和为 1，所有率使用小数。导入设置为 `PYTHONPATH=src:benchmark`；CLI 已处理源码路径。

```python
load_partition(path, phase='development', authorized=False) -> (tuple[Scenario, ...], partition_hash)
execute_trade(holdings, target, cost_rate) -> TradeResult
simulate(scenario, policy, initial_target, config) -> SimulationResult
compute_metrics(result) -> dict
summarize(metrics_dataframe) -> pandas.DataFrame
fit_model(training_scenarios, config) -> serializable_model_dict
make_policy(name, model, params, config, horizon=None) -> policy  # exposes initial_target
fit_expert_bundle(training_scenarios, base_model, config, penalties=None) -> serializable_bundle
```

坏数据、非法动作和优化不可行明确报错；不能静默删除情景。业务不适配或模型预警可返回人工复核动作。Python API 是研究接口，不是安全沙箱；CLI 将输出/bundle 限制在 `benchmark/`。

## 策略接口与时序

`PolicyContext` 包含决策年 t、1..t−1 的历史、交易前持仓、当前目标、截至 t−1 的累计通胀、实际财富峰值、费用率和复核状态。它不暴露完整评价情景。每个情景创建新的策略实例，拟合参数固定。

| 动作 | target_weights | 行为 |
| --- | --- | --- |
| keep | None | 保持资产，权重自然漂移 |
| rebalance | 当前目标 | 恢复当前目标；引擎拒绝借此修改目标 |
| change_target | 新目标 | 交易并替换目标 |
| human_review | None | 持仓不变，进入持续复核状态 |

决定附原因码和解释。门控策略暴露 `requires_benefit_gate=True` 与 `buffer`；交易必须提供收益/成本估计，满足定义过的门槛，引擎独立验证成本。无条件诊断策略不声称满足实际建议策略的收益门槛。

初始配置由引擎在第 1 年收益前完成并收费，后续才调用策略。复核待处理期间不再调用策略，继续持有和估值，不切换到 bills 或虚构顾问干预。初始交易字段同时保存在第一条年度记录中，费用不可重复累加。

## 添加未来模型

训练工厂仅接收训练 IDs；归一化、特征选择、专家门控、早停和强化学习训练均在内层训练边界完成。保存参数、种子、训练 IDs；`evaluate` 拒绝训练/评价 ID 重叠。

上下文兼容v1位置参数，新增回撤字段见1.1扩展；期限由工厂的 `horizon` 参数传入。17 种历史策略兼容省略该参数；新增专家策略必须显式提供期限。`evaluate` 现在对所有策略逐期限重新实例化和模拟，不再取30年路径前缀。年度记录增加 `simulation_horizon`，唯一键为策略×情景×费用×模拟期限×年度；旧结果仍可按原规则核验与重画。

`configs/benchmark.json` 的 `expert_research.enabled` 控制五个新增候选：`expert_defensive`、`expert_greedy`、`moe_supervised`、`moe_rl`、`moe_rl_static`。`POLICY_NAMES` 保留17种传统策略；默认评价在启用研究配置时追加专家候选。监督/RL模型存于 `model['experts']`，包含训练IDs、特征名、训练内标准化、固定三种子网络、参数和训练轨迹。`params[name]['drawdown_penalty']` 指定内层选中的拟合；临时 `moe_rl_seed0/1/2` 仅用于种子诊断，不参与赢家选择。

模型指标增加 `simulation_seconds`、专家 `policy_inference_seconds`、`policy_decision_calls`、`fallback_count`。训练异常显式中止；执行估计无效则转人工并按持续持有参与评价。原因码记录所选专家及三专家门控概率，客户解释保持英文且不超过50词。

## 指标与输出映射

公共 `ScenarioMetrics` / `EvaluationSummary` 结构仍为 v1。benchmark 为容纳额外指标返回 dict/DataFrame，`policy_name` 对应 CSV 的 `policy`；`real_cagr`、`real_max_drawdown`、`real_return_volatility`、`terminal_real_wealth`、`trade_rate`、`total_nominal_cost` 保留对应含义。

MDD 包含每次扣费后的财富标记，替代初始文档仅明确初次费用的口径。H=1 波动及后续交易率在解析测试中为 0，正式期限至少 5 年。聚合键是策略、期限、成本，情景 ID 必须唯一。

每次实验位于 `benchmark/outputs/<run_id>/`：

| 文件 | 内容 |
| --- | --- |
| config.json、manifest.json | 有效配置、分区/代码 hash、种子、环境和阶段 |
| scenario_metrics.csv | 每个策略×情景×期限×成本的原始指标 |
| decisions.csv.gz | 每年动作、解释、估计、费用、目标及交易前后/年末持仓 |
| summary.csv、comparison_table.csv | 原始小数汇总 / 供人阅读的中文展示表 |
| paired_comparisons.csv、matched_comparisons.csv | 官方等权 / 同初始 Balanced 的配对比较 |
| promotion_screen.csv | 研究晋级筛选，不是官方评分 |
| suitability.csv、suitability_risk.csv | 推荐与候选风险依据 |
| splits.json、tuning_trials.csv、fitted_models.json | 所有划分、搜索和拟合参数 |
| validation.json、independent_replay.json | 核验与独立重放证据 |
| risk_promotion_screen.csv | 预声明风险下降路线的探索性晋级筛选，不是官方得分 |
| seed_metrics.csv、seed_summary.csv、training_trace.csv | RL固定三种子的折外诊断及外层训练轨迹 |
| expert_customer_risk.csv | 实际专家策略在客户期限上的风险诊断，不自动修改入门分配 |
| report.html、figures/ | 本地报告、PNG/SVG 图表 |

CSV 使用 UTF-8、无索引；三维权重/持仓拆成资产列，原因码为 JSON 数组，缺失估计留空。JSON 中缺失值为 null。新运行按模拟期限分别保存年度记录；`allocation_exposure.csv` 同样按 `simulation_horizon` 分组，避免短期/长期重复加权。

冻结 artifact 保存训练状态、参数、代码与依赖版本、客户分配及最终比较列表；最终独占锁在读取真实最终收益前创建。目录重构后保留旧 manifest，不以新 hash 伪装旧实验；当前代码只从版本匹配的开发运行冻结。

## 提交版路径风险扩展（2026-10-16）

公共上下文升级为1.1：`PolicyContext.real_max_drawdown`新增默认值0，表示截至决策前、含扣费时点的历史最大实际回撤。引擎逐期维护；旧位置参数调用仍兼容，学习模型需重新拟合，历史模型不应加载到新特征结构。

新增`frontier.py`：`fit_frontier(training_scenarios, config)`只接收开发训练路径，保存期限配置、风险预算可行性、训练路径的资产累计实际收益，以及线性/三种子神经价值近似。`model['frontier']`与基础模型训练IDs必须一致。新策略为`frontier_annual/hold/gate/mpc/linear/neural`：前两者是相同训练配置的诊断参照，其余执行精确费用门槛。神经模型为多期情景价值的监督近似，不称为强化学习。

`suitability(metrics, config, balanced_policy=None)`将Balanced候选映射到实际执行策略，输出新增`policy/evidence`。冻结时按最终选中策略重新计算开发客户分配；最终评价只执行冻结分配，按`policy`取结果。固定策略风险界和共享训练折的学习策略诊断界明确区分。新版manifest保存`evaluated_policies`；审计检查全部策略×情景×期限×成本组合、指标/账本路径对应和完整收益/成本buffer。

第二批增加`frontier_matched_annual/hold/gate`，开发评价参数直接引用同折`frontier_mpc`；冻结时引用实际选中的frontier候选，避免独立调参混淆消融。价值函数新增可选`risk_barrier/risk_slack`，惩罚超过同状态战略动作P90回撤的部分；历史配置省略时仍为0。近似模型的费用输入固定为训练费用，敏感性只施加声明的解析费用修正，不激活训练中从未变化的输入维度。`generate_report=false`关闭开发/最终自动HTML生成；原显式report命令保留。

`scripts/check_matched_controls.py RUN`仅在运行与当前代码/开发分区hash一致时执行，引用各候选内层已选参数，不重拟合、不选择参数。它在`RUN/matched_controls/`保存H30、官方及翻倍费用下的完整参照账本、审计及来源hash，并生成`strict_matched_comparisons.csv`。该表只报告逐点区间，不替代预声明多候选晋级。`scripts/diagnose_market.py --out ...`以固定ridge=1检验四项因果历史特征，输出按整情景划分的折外误差，不能将其等同收益增益。

平滑风险偏好扩展：`cushion.py`提供`cushion_static/cushion_gate`，都以原Balanced初始配置开始，使用同训练均值/协方差和2.5%细网格，独立收益/成本门槛。动态版本按回撤增加风险厌恶，回撤达到预算前2个百分点时参数饱和，防止超过预算后风险厌恶反向下降。`tune_cushion`只拟合该族的内层训练矩估计。`scripts/run_cushion_experiment.py --source ... --out ...`验证历史源代码快照及配置差异，仅复用外层训练模型；重新评价全部旧候选并逐指标核对财务结果一致，新增策略在同内层划分独立选择。源运行、模型文件hash及增量核验单独保存，旧manifest不改写。

平滑规则v2新增`strategic_prior_weight/risk_response_strength`，省略时分别为0/1，保留首版含义。逆优化先验为`gamma_at_peak × covariance × Balanced + common_shift`，共同平移保持Balanced训练预期收益；该先验使峰值下Balanced为摩擦前最优配置。风险反馈强度0.5将动态风险厌恶增量减半，固定风险对照不受影响。增量实验对比源配置时只允许cushion配置变化；同名cushion策略是新版，旧31个候选必须重算一致，旧cushion调参行不混入新版本试验表。

`calibrated_fixed/calibrated_gate`复用细网格成本门控，但明确采用经验均值及固定风险系数，不启用战略先验或回撤反馈。前者gamma=4，后者由`tune_calibrated`在内层按原风险路线从4/4.5/5选择，保留不可行标记。增量脚本根据`calibrated_research.enabled`选择仅拟合该新族；源运行33个旧候选均需完整重算一致，配置只允许相应研究段变化。未来完整开发/冻结入口也支持同样内层选择。
