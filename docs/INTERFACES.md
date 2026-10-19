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
| `benchmark/safebench/metrics.py` | 路径指标、聚合、配对区间和 Pareto |
| `benchmark/safebench/experiment.py` | 嵌套 CV、多期限/成本评价、适配、冻结/最终流程 |
| `benchmark/safebench/selection.py` | 原效用选择及可选的内层路径风险约束、不可行诊断 |
| `benchmark/safebench/research.py` | 从两个已保存开发运行做配对选择规则比较，不读取市场数据 |
| `benchmark/safebench/scenario_utility.py` | 训练片段估值、CRRA效益门槛和固定目标策略 |
| `benchmark/safebench/efficiency.py` | 效率路线内层选择、20项配对区间、期限诊断与报告 |
| `benchmark/safebench/learning_core.py` | 反事实状态/动作、有限期限拟合Q迭代、确定性回归树与学生策略 |
| `benchmark/safebench/learning_study.py` | 学习策略逐期限模拟、内层选择、18项风险区间和中文报告 |
| `benchmark/safebench/neural_core.py` | 可移植NumPy网络推理、训练市场张量、因果状态及明确费用门槛 |
| `benchmark/safebench/neural_training.py` | CPU双精度可微路径模拟、原始/对偶训练和双种子检查点 |
| `benchmark/safebench/neural_study.py` | 神经网络内层检查点选择、路径约束诊断、风险区间及中文报告 |
| `benchmark/safebench/validation.py` | 独立账本核验及晋级检查 |
| `benchmark/safebench/report.py` | 中文 HTML 展示与可用于英文提交的图表 |
| `scripts/verify_benchmark.py` | 独立市场重放和标量等权计算 |
| `scripts/verify_efficiency.py` | 独立重建内外层训练片段，复算效益门槛和人工复核状态 |
| `scripts/verify_learning.py` | 独立名义金额递推、反事实标签、Bellman/树节点均值及逐期限动作复算 |
| `scripts/verify_neural.py` | 独立NumPy复算网络、训练路径约束、逐期限账本及部署门槛 |
| `scripts/run_tests.py` / `scripts/run_path_risk_study.py` | 跨平台统一测试 / 同预算双运行研究入口 |
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
tune(scenarios, config, seed, trace_path=None) -> (params, trial_rows, split_audits)
select_trials(scores, config, reference=None) -> (params, trial_rows)
compare_selection_runs(control, constrained, out) -> pandas.DataFrame
fit_utility_blocks(scenarios, horizons=(1,3)) -> serializable_blocks_dict
log_certainty_equivalent(gross, weights, gamma, probabilities) -> float
select_efficiency_trials(scores, reference, config) -> (params, trial_rows)
efficiency_intervals(metrics, config) -> pandas.DataFrame
counterfactual_data(training_scenarios, config) -> dict[str, numpy.ndarray]
fit_fqi(counterfactual_arrays, penalty, feature_set, config) -> serializable_value_trees
learning_intervals(metrics, config, policy_names=LEARNING_NAMES, prefix='learning') -> pandas.DataFrame
neural_data(training_scenarios, config) -> dict[str, numpy.ndarray]
fit_neural_models(training_scenarios, base_model, config) -> serializable_neural_models
rollout(networks, policy_name, tensor_data, mean, scale, mu, cov, config, horizon, reference=False) -> tensor_metrics
```

`study_kind=scenario_efficiency` 才在 `fit_model` 中增加 `ce_blocks`，按估值期限保存实际资产增长因子、概率、情景IDs与起始年；各训练情景总概率相同。`tune` 转入六组band/buffer的效率选择，并保存 `tuning_models/` 全部内层训练状态。`evaluation_policy_names` 决定该实验明确评价的策略，缺省仍为旧17种；既有策略的动作公式不变。

新增 `utility_band_1y/utility_band_3y` 只保持或恢复原50/35/15目标，复核优先。`estimated_benefit=1−CE(hold)/CE(target)` 为代理下可接受的当期财富收费比例，与引擎重算的费用比例比较；带内不计算交易估值。`balanced_annual_review` 是同复核规则的无条件诊断。效率筛选失败或任何外层内选不可行的CE候选禁止冻结；允许的冻结策略须出现在配置的评价列表中。旧17策略的缺省列表与共享v1类型保留。

`selection_rule` 缺省为 `one_se_utility`，保持既有数值行为；`path_risk` 只按内层折外路径与同初始 Balanced 参照比较，约束参数见配置。`trial_rows` 新增 `selected`、P90/P10、参照值、`risk_feasible` 和 `selection_status`；没有可行候选时状态为 `infeasible_diagnostic`，仍评价全部外层情景，但禁止冻结。返回参数保留 gamma/buffer/window，并在新规则下附选择状态与 trial ID。

`trace_path` 可选且只允许新建，保存各内层情景的候选指标（参照 trial=-1）；开发运行默认输出 `tuning_metrics/outer_<fold>.csv.gz`。选择比较要求两运行的开发分区、代码、环境、内外层划分和全部配置相同，只有 `selection_rule` 可不同。比较输出含源文件 hash、配对区间、汇总及图表；不覆盖历史运行。

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

共享上下文保持v1，不含horizon。原有策略仍按30年模拟并取前缀；`counterfactual_fqi`研究通过工厂的`horizon`参数传入H，所有策略按每个H重新实例化/模拟，账本新增`evaluation_horizon`，主键包含该列；核验按对应期限的独立路径分组。学习工厂不接受省略H或超过已拟合期限的请求。增加昂贵模型时记录推理耗时、失败次数、显式回退和解释；生产回退若转人工，应按真实动作参与评价。

`study_kind=counterfactual_fqi`增加模型键`fqi`，按`full/portfolio`特征集、风险惩罚、剩余期限及`teacher/student`保存树节点、训练来源、数据摘要与Bellman拟合/蒸馏统计。`fqi_student/fqi_teacher/fqi_portfolio`的参数键为`penalty`；教师沿用主学生选参。Q树输出编码为保持动作价值及其余三动作相对优势乘固定缩放因子。回归目标不含直接即期费用；部署和Bellman延续均用`Q_gross+log(1-fee)`并通过`1-exp(-(Q_gross[a]-Q_gross[keep])) > fee+buffer`门槛。后继状态含实际扣费对回撤/持仓的影响，费用不重复计入直接增长。新研究冻结入口明确拒绝，尚未声称具备正式final接口。

`study_kind=neural_path_control`增加模型键`neural`，按三策略和检查点保存所有成员的线性层权重/偏置、特征索引、标准化、目标凸包、种子、训练约束与优化轨迹。`neural_linear/neural_mlp/neural_portfolio`参数键为`checkpoint`；通过现有工厂传入H，每个期限独立模拟，账本使用`evaluation_horizon`。推理只需NumPy，训练延迟导入PyTorch；其他研究不强制加载Torch。新研究manifest额外记录torch版本，真实freeze/final拒绝执行。

神经策略输入现有12维状态加`remaining/30`，完整模型取13维，持仓消融只取索引0/1/2/3/10/11/12。标准化的历史市场均值/标准差仅由训练情景求得，其余尺度为固定物理尺度；标准化后裁剪到[-5,5]。成员输出softmax三权重，再线性组合三个合法目标；部署对所有预定种子的目标取算术平均。费用门槛复用既有均值方差效用的财富比例估计，gamma固定；约束训练不会把该代理升级为收益保证。

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
| tuning_metrics/ | 逐内层验证情景、策略、参数 trial 的原始指标及参照 |
| tuning_models/ | 效率研究每个内层训练估值器及片段来源 |
| efficiency_intervals.csv、efficiency_screen.csv | 20项调整区间及原/双倍成本效率条件 |
| candidate_suitability.csv、review_matched_comparisons.csv | CE候选的客户期限风险诊断 / 同复核参照比较 |
| efficiency_report.html、utility_replay.json | 效率报告 / 独立CE门槛与训练片段重放证据 |
| validation.json、independent_replay.json | 核验与独立重放证据 |
| reproduction_verification.json | 确定性复现的逐文件摘要、代码/配置匹配及报告验收证据（有复现时生成） |
| training_data/*.npz | 学习研究内外折的反事实状态、后继、费用、增长、首次越界及来源ID/时期 |
| learning_intervals.csv、learning_screen.csv | 三学习策略原/双倍费用下18项风险路线区间与整体状态 |
| learning_suitability.csv、learning_review_comparisons.csv | 学习策略客户对应期限诊断 / 同复核年度参照比较 |
| learning_training.csv、learning_rules.json、learning_report.html | 教师/学生拟合与压缩摘要、树叶规则示例及中文研究报告 |
| learning_replay.json | 反事实训练、Bellman/蒸馏与部署决策的独立复算证据 |
| learning_acceptance.json | 学习研究的实现验收、财务采用状态、核验计数及产物SHA256；实现通过不等于模型通过 |
| neural_intervals.csv、neural_screen.csv | 三神经策略原/双倍费用18项风险区间及内选/训练/区间通过状态 |
| neural_constraints.csv、neural_training.csv | 每折/策略/检查点/期限/费用的训练约束，以及各种子的梯度/对偶轨迹 |
| neural_suitability.csv、neural_review_comparisons.csv | 神经策略客户期限风险与相同复核年度参照诊断 |
| neural_report.html、neural_replay.json | 神经网络研究报告及独立网络/约束/动作核验 |
| neural_acceptance.json | 神经研究的实现验收与采用状态、训练/敞口诊断、环境匹配及产物SHA256；实现通过不等于采用通过 |
| report.html、figures/ | 本地报告、PNG/SVG 图表及无标题的论文用矢量 PDF |

CSV 使用 UTF-8、无索引；三维权重/持仓拆成资产列，原因码为 JSON 数组，缺失估计留空。JSON 中缺失值为 null。原有年度记录存到30年并按前缀评价；学习研究账本按`evaluation_horizon`独立保存每个期限。确定性树动作复算读取CSV时须使用`float_precision='round_trip'`，还原保存的双精度值，避免默认解析误差跨越分割阈值。

神经研究的`training_data/*.npz`保存训练情景IDs、名义收益、通胀和逐期因果历史特征，不保存任何外层验证未来信息。每个检查点的`training_constraints`分别记录实际MDD-CVaR90、弱情景年化对数增长尾均值、相应参照界限和可行标志。两者均为训练经验约束，和外层报告的P90 MDD/P10 CAGR区分。

冻结 artifact 保存训练状态、参数、代码与依赖版本、客户分配及最终比较列表；最终独占锁在读取真实最终收益前创建。目录重构后保留旧 manifest，不以新 hash 伪装旧实验；当前代码只从版本匹配的开发运行冻结。
