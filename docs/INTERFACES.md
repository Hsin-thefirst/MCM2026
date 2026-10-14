# 统一接口

公共类型：[contracts.py](../src/safeinvest/contracts.py)。版本：v1.0。数据加载已实现，其余模块按以下接口开发。

## 基础约定

- Python 3.10+；模块间传递 dataclass 和 tuple，内部可用 numpy / pandas。
- 资产顺序：`equity, bond, bill`。权重：非负、和为 1。持仓：同顺序的名义金额。
- 收益、通胀、风险和成本：小数，`0.05` 表示 5%。时间：年。
- `Scenario`：一条完整情景；`MarketObservation`：一个年度的名义收益与通胀。
- 无效数据：抛出异常；客户不适配或模型预警：返回人工复核动作及理由。

## 模块接口

| 模块 | 责任 | 主要类型 |
| --- | --- | --- |
| `data.py` | 加载、验证和划分情景 | `Scenario` |
| `engine.py` | 执行动作、计费和推进年度 | `TradeResult`, `SimulationResult` |
| `metrics.py` | 计算路径指标和情景汇总 | `ScenarioMetrics`, `EvaluationSummary` |
| `suitability.py` | 问卷适配及解释 | `CustomerProfile`, `SuitabilityResult` |
| `policies/` | 根据已知信息选择动作 | `PolicyContext`, `Decision` |
| `evaluation.py` | 批量评估及保存结果 | 上述类型 |

```python
# data.py（已实现）
load_scenarios(path: str | Path) -> tuple[Scenario, ...]
split_scenarios(scenarios: tuple[Scenario, ...]) -> tuple[
    tuple[Scenario, ...], tuple[Scenario, ...]
]

# engine.py
execute_trade(holdings: Holdings, target: Weights, cost_rate: float) -> TradeResult
simulate(scenario: Scenario, policy: Policy,
         initial_target: Weights, config: SimulationConfig) -> SimulationResult

# metrics.py
compute_metrics(result: SimulationResult) -> ScenarioMetrics
summarize(metrics: tuple[ScenarioMetrics, ...]) -> EvaluationSummary

# suitability.py
class TransparentSuitabilityRule:
    def recommend(self, profile: CustomerProfile) -> SuitabilityResult: ...

# policies/
class Policy(Protocol):
    name: str
    def decide(self, context: PolicyContext) -> Decision: ...

# evaluation.py
evaluate(scenarios: tuple[Scenario, ...], policy_factory: Callable[[], Policy],
         initial_target: Weights, config: SimulationConfig
) -> tuple[tuple[SimulationResult, ...], tuple[ScenarioMetrics, ...], EvaluationSummary]
```

加载器校验列名、完整的 180×30 网格、唯一主键及有限数值，收益和通胀均须大于 −1。返回按情景 / 年度排序的对象；划分结果为开发集、最终测试集。

引擎模拟前 `config.horizon` 年；短路径测试可直接构造 `Scenario`。适配规则构造时接收组合、阈值和按组合 / 期限索引的开发风险表；`portfolio=None` 表示人工复核。

## 策略输入与动作

`PolicyContext` 包含：年度 t、1..t−1 的市场历史、交易前持仓、当前目标、截至 t−1 的累计通胀和实际财富峰值、复核状态、成本率。每条情景创建一个新的策略实例，训练参数保持固定。

| action | target_weights | 行为 |
| --- | --- | --- |
| `keep` | `None` | 保持实际持仓，权重自然漂移 |
| `rebalance` | 当前目标 | 恢复当前目标 |
| `change_target` | 新目标 | 交易并更新目标 |
| `human_review` | `None` | 保持持仓，进入复核状态 |

每个 `Decision` 附 `reason_codes` 和 `explanation`。建议原因码：`BENEFIT_BELOW_COST`、`REBALANCE_BENEFICIAL`、`TARGET_CHANGE_BENEFICIAL`、`DRAWDOWN_ALERT`、`REVIEW_PENDING`。

Balanced 交易决策还需提供 `estimated_benefit` 和 `estimated_cost`，均为交易前财富的比例，并满足 `benefit > cost + buffer`。等权基准每年恢复目标，估计字段可为 `None`。上下文相关检查（如再平衡目标是否正确）由引擎验证。

## 年度时序与交易核算

第 1 年：全 bills → 初始目标 → 扣费 → 应用收益和通胀，记录 `rebalance / INITIAL_ALLOCATION`。第 2..H 年：建立 context → 策略决策 → 执行动作 → 应用当年收益和通胀。

转人工当期保持持仓；后续年度由引擎记录 `keep / REVIEW_PENDING`，复核状态持续到模拟结束。每个结果包含 H 条年度记录。

```text
V = Σholdings
Q = V × 0.5 × Σ|target − holdings/V|
F = cost_rate × Q
post_trade_holdings = (V − F) × target
```

Q 按收费前目标计算；费用在收益前从目标持仓按比例支付。保持 / 复核时 Q=F=0，持仓保持原值。例：100 单位全 bills 转 20 到 equities，Q=20、F=0.05，扣费后持仓为 `(19.99,0,79.96)`；三等分建仓费用为 `1/6`。

`SimulationResult.initial_*` 与第一条记录的建仓信息相同。总费用只对年度记录求和一次。实际交易判定：`moved_amount > 1e-10 × initial_wealth`。

## 指标

W 为实际财富，H 为期限，V₀ 为初始财富。

| 字段 | 定义 |
| --- | --- |
| `real_cagr` | `(W_H/V₀)^(1/H)−1`，包含全部费用 |
| `real_max_drawdown` | 期初、首次扣费后及年末财富的最大相对峰谷损失 |
| `real_return_volatility` | `W_t/W_{t−1}−1` 的样本标准差，ddof=1，W₀=V₀ |
| `terminal_real_wealth` | 第 H 年末实际财富 |
| `trade_rate` | 第 2..H 年实际交易次数 / `(H−1)`；H=1 时为 0 |
| `total_nominal_cost` | 年度记录费用之和，包含建仓 |

`compute_metrics` 要求 H≥2。汇总输入须为同策略、同期限、情景 ID 互异的非空结果。分位数采用线性插值：排序后索引 `(n−1)p`，与 numpy `method="linear"` 一致。汇总报告 CAGR 中位数 / P10、MDD 中位数 / P90、波动中位数和交易率均值。

## 输出

运行目录：`outputs/<run_id>/`。报告图表：`report/figures/`。

| 文件 | 内容 |
| --- | --- |
| `config.json` | 执行配置、权重、估计器参数、策略门槛和冻结标记 |
| `manifest.json` | 接口版本、phase、情景 IDs、期限、数据 / 配置 hash、代码版本及未提交状态、依赖版本、种子 |
| `scenario_metrics.csv` | `run_id` + `ScenarioMetrics` 字段 |
| `summary.csv` | `run_id` + `EvaluationSummary` 字段 |
| `decisions.csv` | `run_id,scenario_id,policy_name` + `PeriodRecord` 字段 |
| `suitability.csv` | 客户字段 + `portfolio,reason_codes,explanation` |
| `figures/` | 比较图、严重案例、成本敏感性 |

CSV：UTF-8、无 index、百分比保留小数。三元素 tuple 展开为 `<字段名>_equity/_bond/_bill`；`reason_codes`、`scenario_ids` 为 JSON 数组。缺失估计值留空，人工复核的 portfolio 导出为 `human_review`。JSON 中 tuple 为数组、Enum 为字符串、缺失值为 null。

配置中的 `[1,120]` 等表示闭区间端点；manifest 保存完整实际 ID 列表。phase 取 `development / final / sensitivity`。
