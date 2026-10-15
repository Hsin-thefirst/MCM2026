# 代码接口与结果格式

本文件合并公共接口与原 benchmark 扩展说明。数学口径仅在 [METHODOLOGY.md](METHODOLOGY.md) 维护，实验程序仅在 [PROTOCOL.md](PROTOCOL.md) 维护。

## 模块职责

| 位置 | 职责 |
| --- | --- |
| `src/safeinvest/contracts.py` | v1 公共 dataclass、Enum、Policy 协议和校验 |
| `src/safeinvest/data.py` | 完整 180×30 网格校验/排序；兼容接口，日常测试仅用生成数据 |
| `benchmark/safebench/data.py` | 开发分区读取、情景折划分、实际收益转换 |
| `benchmark/safebench/engine.py` | 交易核算、因果上下文和年度模拟 |
| `benchmark/safebench/policies.py` | 28种策略/配对消融、训练估计及策略工厂 |
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
make_policy(name, model, params, config) -> policy  # exposes initial_target
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

**保护性减仓豁免**：`Decision.gate_exempt=True` 标记保护性减仓交易（原因码 `DRAWDOWN_RISK_REDUCTION`）。引擎对所有门控策略交易独立重算并校验成本一致性；对豁免交易跳过"收益 > 成本+buffer"不等式（减仓的价值是损失保护，均值-方差收益估计无法表达），但豁免必须单向：豁免交易（第 1 年之后）必须严格降低股票敞口，`validation.audit_outputs` 独立复核方向，违规即核验失败。加仓与中性再平衡仍受完整门槛约束。

初始配置由引擎在第 1 年收益前完成并收费，后续才调用策略。复核待处理期间不再调用策略，继续持有和估值，不切换到 bills 或虚构顾问干预。初始交易字段同时保存在第一条年度记录中，费用不可重复累加。

## 添加未来模型

训练工厂仅接收训练 IDs；归一化、特征选择、专家门控、早停和强化学习训练均在内层训练边界完成。保存参数、种子、训练 IDs；`evaluate` 拒绝训练/评价 ID 重叠。

上下文没有 horizon，期限通过 `make_policy(..., horizon=H)` 传入工厂。`drawdown_lifecycle_*` 与 `budget_lifecycle_*` 对每个 H 重新实例化并模拟，其他期限无关策略可模拟最长路径再取前缀。当前生命周期只按初始期限设置股票上限，不是逐年剩余期限滑道。期限相关 MPC/RL 也不能继承前缀优化。增加昂贵模型时记录推理耗时、失败次数、显式回退和解释；生产回退若转人工，应按真实动作参与评价。

方案 A 的 `budget_lifecycle_balanced` 使用 `params['by_horizon'][str(H)]` 保存期限独立参数；`_zero_band`、`_static`、`_annual`、`_hold` 四个配对对照复用同一折同一期限的参数和初始配置，不独立搜索。`fit_model` 的 `risk_rows` 仅保存该层训练情景的实际年度收益，用于一年期超容忍度损失代理。新候选没有保护性豁免，所有后续交易均须通过收益/费用门槛。年度再平衡和持有对照仅用于诊断。

年度输出另含 `requires_benefit_gate`、`buffer`，供独立核验准确重算普通门控交易。核验逐一覆盖每个指标键：期限无关策略使用完整账本前缀，期限相关策略使用对应 H 的独立账本；缺少覆盖或重复覆盖均失败。历史无门控元数据的产物不能冒充具有完整的门控核验。

`budget_value_tuning_grid` 是新族唯一搜索网格（4组）。按各 H 的内层折外路径检验风险约束并最大化中位 CAGR；无可行项明确记录研究回退。旧回撤选择也统一为中位数排序，不再混用均值的一标准误规则。缺失调参诊断使用 JSON null，冻结摘要拒绝未清洗的非有限值。客户适配新增确切 `policy` 标识；冻结及最终评价保留开发分配，不依据最终数据重选。

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
| report.html、figures/ | 本地报告、PNG/SVG 图表 |
| incremental_comparisons.csv、incremental_screen.csv | 方案 A 的同初始配置消融区间及预声明风险降低筛选 |
| band_ablation.csv、incremental_protocol.json | 交易带消融与多重比较范围/解释边界 |

CSV 使用 UTF-8、无索引；三维权重/持仓拆成资产列，原因码为 JSON 数组，缺失估计留空。JSON 中缺失值为 null。年度记录存到 30 年，短期限结果使用相应前缀。`decisions.csv.gz` 含 `gate_exempt` 列标记豁免交易；`tuning_trials.csv` 中回撤策略行含约束诊断列（`breach_count`、`train_paths`、`breach_upper_bound`、`constraint_satisfied`、`constraint_fallback`、`median_cagr_valid`）。回撤策略按族调参：`drawdown_*` 使用 `drawdown_tuning_grid`，`drawdown_lifecycle_*` 使用独立的 `drawdown_lifecycle_tuning_grid`（缺失时回退共享网格）；网格参数项支持 `derisk_exempt` 与 `equity_shift`。

冻结 artifact 保存训练状态、参数、代码与依赖版本、客户分配及最终比较列表；最终独占锁在读取真实最终收益前创建。目录重构后保留旧 manifest，不以新 hash 伪装旧实验；当前代码只从版本匹配的开发运行冻结。
