# 代码接口与结果格式

本文件合并公共接口与原 benchmark 扩展说明。数学口径仅在 [METHODOLOGY.md](METHODOLOGY.md) 维护，实验程序仅在 [PROTOCOL.md](PROTOCOL.md) 维护。

## 模块职责

| 位置 | 职责 |
| --- | --- |
| `src/safeinvest/contracts.py` | v1 公共 dataclass、Enum、Policy 协议和校验 |
| `src/safeinvest/data.py` | 完整 180×30 网格校验/排序；兼容接口，日常测试仅用生成数据 |
| `benchmark/safebench/data.py` | 开发分区读取、情景折划分、实际收益转换 |
| `benchmark/safebench/engine.py` | 交易核算、因果上下文和年度模拟 |
| `benchmark/safebench/policies.py` | 原17种策略、训练估计及可配置策略工厂 |
| `benchmark/safebench/plan_a.py` | 计划A路径风险凸优化、三产品及5种执行消融 |
| `benchmark/safebench/metrics.py` | 路径指标、聚合、配对区间和 Pareto |
| `benchmark/safebench/experiment.py` | 嵌套 CV、多期限/成本评价、适配、冻结/最终流程 |
| `benchmark/safebench/validation.py` | 独立账本核验及晋级检查 |
| `benchmark/safebench/report.py` | 中文 HTML 展示与可用于英文提交的图表 |
| `scripts/verify_benchmark.py` | 独立市场重放和标量等权计算 |
| `scripts/verify_plan_a.py` | 独立穷举回撤区间、LP原始/对偶证书及旧参照一致性 |
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
policy_names(config) -> list[str]  # 17 legacy names; 25 when Plan A is enabled
make_policy(name, model, params, config, horizon=None) -> policy  # exposes initial_target
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

上下文没有 horizon；计划A通过工厂显式接收 H，并逐期限重新实例化/模拟。省略 H 明确报错。原17种期限无关策略仍可模拟30年再取前缀。未来期限相关策略须遵守相同规则。计划A保存决策次数和推理耗时；模型失效转人工并按真实动作参与评价，优化失败则终止并保留失败记录。

`fit_model` 在 `plan_a.enabled=true` 时增加 `model['plan_a']`：训练IDs、训练耗时及以`产品:H:风险比例`为键的配置。每个配置保存权重、无费用风险及预算、含费用年度诊断、复核阈值、网格差、凸目标上下界、收敛间隙与LP约束/对偶证书。`solve_allocation(returns, budget, alpha=.9, bill_floor=0., options=None, cached_grid=None)`接收情景×年度×3的实际收益数组；预算不可行或未收敛均报错。完整执行策略的样本外风险另行评价。

## 指标与输出映射

公共 `ScenarioMetrics` / `EvaluationSummary` 结构仍为 v1。benchmark 为容纳额外指标返回 dict/DataFrame，`policy_name` 对应 CSV 的 `policy`；`real_cagr`、`real_max_drawdown`、`real_return_volatility`、`terminal_real_wealth`、`trade_rate`、`total_nominal_cost` 保留对应含义。

MDD 包含每次扣费后的财富标记，替代初始文档仅明确初次费用的口径。H=1 波动及后续交易率在解析测试中为 0，正式期限至少 5 年。聚合键是策略、期限、成本，情景 ID 必须唯一。

每次实验位于 `benchmark/outputs/<run_id>/`：

| 文件 | 内容 |
| --- | --- |
| config.json、manifest.json | 有效配置、分区/代码 hash、种子、环境和阶段 |
| scenario_metrics.csv | 每个策略×情景×期限×成本的原始指标 |
| decisions.csv.gz | 每年动作、解释、估计、费用、目标及交易前后/年末持仓 |
| summary.csv、comparison_table.csv | 原始小数汇总 / 显式生成报告时的中文展示表 |
| paired_comparisons.csv、matched_comparisons.csv | 官方等权 / 同初始 Balanced 的配对比较 |
| promotion_screen.csv、risk_promotion_screen.csv | 增长路线 / 计划A风险路线的研究筛选，不是官方评分 |
| suitability.csv、suitability_risk.csv | 推荐与候选风险依据 |
| splits.json、tuning_trials.csv、fitted_models.json | 所有划分、搜索和拟合参数 |
| validation.json、independent_replay.json | 核验与独立重放证据 |
| allocation_certificates.csv、independent_plan_a.json | 计划A独立优化核验及旧17策略一致性 |
| report.html、figures/ | 显式请求生成时的本地报告、PNG/SVG图表 |

CSV 使用 UTF-8、无索引；三维权重/持仓拆成资产列，原因码为 JSON 数组，缺失估计留空。JSON 中缺失值为 null。新增`simulation_horizon`区分独立模拟：决策唯一键为`policy,scenario_id,cost_rate,simulation_horizon,period`；指标仍以`policy,scenario_id,cost_rate,horizon`为唯一键。原策略的短期限指标映射30年轨迹；计划A映射相同期限的独立轨迹。旧文件缺少新字段时，审计按每条轨迹最大年度推断。决策额外保存`required_buffer`和`requires_benefit_gate`，支持逐交易核验。manifest声明完整策略集合，审计拒绝缺失策略或指标网格。

`suitability`在计划A开启且三产品结果齐备时使用三个实际执行策略，并保存`policy`和证据类型。冻结沿用明确的策略ID，最终评价不改客户分配。OOF二项上界是筛选诊断，不能解释为重拟合策略的覆盖保证。

当前配置`generate_report=false`，开发实验不生成报告或论文。优化核验命令：`.venv/bin/python scripts/verify_plan_a.py benchmark/outputs/development_plan_a_v2`。`scripts/diagnose_plan_a.py <run> --out <new_directory>`复算已选参数的内层约束并保持模型不变做年度顺序压力测试；保存`selected_constraints.csv`、`order_stress_metrics.csv`、`order_stress_summary.csv`及原指标重现核验，输出只能新建，不再调参。

冻结 artifact 保存训练状态、参数、代码与依赖版本、客户分配及最终比较列表；最终独占锁在读取真实最终收益前创建。目录重构后保留旧 manifest，不以新 hash 伪装旧实验；当前代码只从版本匹配的开发运行冻结。

## 解题方案诊断工具（2026-10-16）

以下独立研究工具不改变原17策略、模拟接口或冻结候选；只读取开发分区或已保存结果，不自动生成报告。方案与解释见 [研发计划](STRATEGY_PLANS.md)。

```bash
.venv/bin/python scripts/probe_predictability.py --out benchmark/outputs/strategy_planning_20261016/predictability
.venv/bin/python scripts/probe_strategy_routes.py --out benchmark/outputs/strategy_planning_20261016/routes
.venv/bin/python scripts/check_strategy_probes.py --run benchmark/outputs/strategy_planning_20261016/routes --out benchmark/outputs/strategy_planning_20261016/routes/verified_checks
```

输出只能新建，复现时使用新的目录；核验工具要求预测结果在策略目录同级的`predictability/`，历史参照从`development_consolidated/scenario_metrics.csv`读取。`--self-check-only`可仅执行预测器的合成因果前缀检查。策略探针使用`configs/strategy_planning.json`，预测器为本批固定ridge10、5年窗口、seed20261014的诊断，manifest记录其精确设置。

固定网格数组保存为`static_grid_paths.npz`，索引为权重×情景；按每个训练折选择权重，正式费用选定后不随敏感性费率重选。带宽规则仅为诊断参照。门控候选使用原Balanced初始配置、官方单纯形与固定风险参数；年度轨迹只保存至30年，因为本批执行规则不依赖期限。所有逐点区间均为探索性诊断，不替代原晋级协议。
