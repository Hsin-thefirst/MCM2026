# 代码接口与结果格式

本文件合并公共接口与原 benchmark 扩展说明。数学口径仅在 [METHODOLOGY.md](METHODOLOGY.md) 维护，实验程序仅在 [PROTOCOL.md](PROTOCOL.md) 维护。

## 项目维护与交付入口（2026-10-18）

交付配置唯一来源为`configs/delivery.json`，金融接口和hash未变。`scripts/project_tools.py`提供`check/package/verify/restore/clean`，分别由`make check/package/verify-package/restore/clean`调用；`PACKAGE`覆盖ZIP路径，`PYTHON`覆盖解释器。`make guide`调用已有保存数据讲解生成器。

main合并前复核：`make guide`改为调用`project_tools.py guide`并读取交付配置中的`research_guide_run`；默认ZIP路径也只从交付配置读取，Makefile仅在显式传`PACKAGE`时覆盖。`check`通过`validate_release(release, bundle)`要求ready决定的策略、冻结包hash及金融源码hash一致，客户核验成功且没有重用最终集/新最终评价标记。新增校验只检查保存元数据，不更改金融规则。

`check`核对冻结源码/环境/最终锁/发布决定，报告Git checkout是否缺完整保存证据；不读取市场数据。`package`复用原打包器并加入最新客户研究紧凑证据及Git配置，拒绝覆盖已有ZIP；`verify`核对安全成员路径、准确清单、每文件SHA256和存在的外部收据。`restore`先验证同一冻结包/锁及所有目标冲突，仅恢复`benchmark/outputs/`；不覆盖不同证据、源码或配置。`clean`仅删受控缓存名，不进入环境、Git、实验结果或dist，不跟随符号链接且拒删tracked文件。

ZIP仍严格小于50MB、保留最终锁并只包含已保存的最终数值。后续研究大缓存不入Git/ZIP；新clone可阅读紧凑结果和测试，打包/重建讲解前可从可信交付包恢复原默认证据。历史manifest、金融程序及旧ZIP不因整理而重写。

## 客户风险最后复核（2026-10-18）

`scripts/customer_risk.py`提供独立研发接口，不修改冻结金融核心：

```python
CustomerRiskPolicy(name, model, params, config, profile, fraction,
                   early_review=False, early_review_fraction=.8, mode="partial")
select_fraction(inner_execution_frame, profile, settings) -> selection_dict
unavoidable_drawdown(real_returns) -> numpy.ndarray
```

仅接受开发训练IDs。初始目标为原A与同训练集最小方差目标的凸组合；比例1精确恢复原A，后续继续原A的净收益/成本门槛。提前复核只改变触发阈值，待处理仍持有原资产。内层完整含费路径在五个固定比例内选择满足客户风险余量、风险改善和流动性条件的最高中位收益者；无可行者完整保留最小违约诊断，`feasible=false`不得标为自动推荐。

配置为`configs/customer_risk_review.json`。`run_customer_risk_review.py --out benchmark/outputs/<new_run>`使用已有两套划分的训练模型及各产品原参数，输出内层指标、选择证据、外层路径/账本和全资产共同下跌导致的回撤下界。`analyze_customer_risk.py <run>`仅读保存结果，输出客户比较、配对区间和未替换冻结提交的决定。此研究发生在原最终集暴露之后，没有独立最终成绩；OOF二项上界只是选择诊断。

`verify_customer_risk.py <run>`仅读开发市场和保存研究输出，独立标量核算全部外层记录并从内层CSV重算全部选择；不重跑策略。`build_solution_guide.py --research-run <run> --out docs/SOLUTION_GUIDE.md`只读已保存数值和冻结模型，交叉核对汇总、客户越界与证据hash后生成团队讲解Markdown，不读取市场CSV、不创建正式论文。新研究依赖原工作区的训练缓存，旧ZIP的完整默认金融复现范围保持原义。

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
| `scripts/verify_plan_a.py` | 独立穷举回撤、核验参照风险与可行/不可行证书、调参排名及旧参照一致性 |
| `scripts/compare_plan_a_runs.py` | 从保存结果按相同风险线比较历史与当前版本，不重新拟合 |
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

2026-10-17校准扩展：可选`plan_a.validation_risk_ratio`固定验证目标，条目保存`validation_budget`，指标新增`plan_a_validation_budget`；`risk_budget`/`plan_a_risk_budget`仍表示训练战略预算。缺少新配置时保持旧版按候选训练预算验收的行为。当前验证比例1.0、训练比例0.75/0.85/0.95，分别留25%/15%/5%余量；这些都是研究参数。

连续求解的已证明不可行以`AllocationInfeasible(RuntimeError)`携带LP松弛证据；`fit_plan_a`将该候选保存为`status=infeasible`、`failure_certificate`且不提供权重。其余求解错误仍终止。工厂拒绝执行不可行条目；调参排除任一内层训练或全训练重拟合不可行的完整候选，不删验证情景。`tune(scenarios, config, seed, refit_model=None)`可复用同一训练总体的模型，仅按训练可行性过滤，不接收外层验证结果；无可执行候选时报错。

`selection_constraints(block, config)`返回总违约量和各期限诊断；所有候选共享验证风险线，候选训练安全余量不影响该线。`tuning_trials.csv`新增`allocation_feasible`、`refit_feasible`、`inner_infeasible_folds`和JSON格式`horizon_constraints`；`splits.json`保留各内层的不可行配置证据，`fitted_models.json`保留外层对应证据。独立优化核验同时统计可行和不可行证书。

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

当前配置`generate_report=false`，开发实验不生成报告或论文。优化核验命令：`.venv/bin/python scripts/verify_plan_a.py benchmark/outputs/development_plan_a_calibrated_v1`。`scripts/diagnose_plan_a.py <run> --out <new_directory>`复算已选参数的内层约束并保持模型不变做年度顺序压力测试；保存`selected_constraints.csv`、`order_stress_metrics.csv`、`order_stress_summary.csv`及原指标重现核验。新增`risk_decomposition.csv`对H30各产品/外折分开记录训练无费配置风险、含费年度风险、含费执行风险与验证风险；训练路径只作诊断，不参与外层评分。输出只能新建，不再调参。

诊断另存`training_risk_floors.csv`，以231点网格检查各训练分区的风险可行域，仅用于诊断，不代表连续最小风险证明。`compare_plan_a_runs.py --previous <旧运行> --previous-diagnostics <旧诊断> --current <新运行> --current-diagnostics <新诊断> --out <新目录>`以新版本固定验证比例重新检验两版已有的内层/外层结果，先核对分区、划分配置与相同风险线；保存`common_inner_risk_screen.csv`、`common_outer_risk.csv`、`financial_comparison.csv`及核验JSON。不把换验收线造成的通过率变化当成金融改善，输出同样只能新建。

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


## 计划B/C接口（2026-10-17）

契约版本1.1：`PolicyContext.historical_max_drawdown`为决策前历史最大实际回撤（默认0，兼容旧构造），引擎同时纳入费用时点和年度终点；当前回撤与历史最大回撤分别保留。`PeriodRecord.policy_diagnostics`默认None，CSV按JSON保存B信号/回退及C规划净价值/费用分解；初始配置与待人工复核期无策略诊断。

`signals.py`统一维护原12因果特征与Ridge10实现；探针继续导出原函数。`model.signals`冻结训练ID、三预测器的标准化/系数/同目标期均值。B固定及内层所选动作族共10策略，均继承同一A均衡锚、协方差、gamma、复核与安全边际。宽动作零信号单独作为参照。训练不足/异常不静默重拟合，短历史或标准化超阈值回退A并记录。

`model.scenario_control`冻结训练ID、联合实际收益路径及每期限防御配置。C四策略共享A锚及后续执行；一步/三步、风险系数1/0固定。C先比较H30，`plan_c.evaluation_horizons`仅限制C评价输出，不改变A五期限训练/调参或客户诊断。C筛选不全的配置禁止冻结C候选；真实最终评价仍走原冻结流程。

配置可用`extends`相对当前文件继承，递归合并对象、替换数组，检测循环；运行目录`config.json`保存完全展开的唯一有效配置。`configs/plan_b.json`及`configs/plan_c_screen.json`可分别传给`benchmark/run.py development --config ... --out <新目录>`，默认benchmark仍只跑原25策略。`extension_comparisons.csv`与`extension_screen.csv`保存新增匹配比较、既定增长/风险筛选及全部检查布尔值；这些为反复使用开发总体的探索性证据。


提交前补充：B诊断用`warmup`明确区分短历史与`fallback`异常；C当前及未来A估计无效同样进入吸收复核。冻结B/C时客户证据仍标为OOF诊断，冻结比较列表包含该配置完整完成开发评价的B/C策略族；C仅筛选H30时无条件禁止冻结C。预测独立重建与B内层选择可分别运行`scripts/verify_extensions.py <run>`、`scripts/verify_plan_b_selection.py <B_run>`；`compare_extension_runs.py <旧运行> <新运行>`核对接口修复前后全部金融指标，均只允许开发分区且验证JSON只新建。


B信号倾斜及C前瞻防御动作有各自的英文交易说明，不沿用A“向锚配置靠拢”的说明，因为新动作可能离开A锚；短历史/异常回退及窄动作eta0继续原样复用A决策。该修正只改变解释文本，须以新源码hash另存运行并核对金融指标一致。

## 面向提交的执行扩展（2026-10-17）

`configs/submission.json`继承统一输入，开启`submission.enabled`，保留原25策略并新增8个策略ID，共33个。`submission.py`的`BASE_RULES`包含原A与7条执行挑战，另有`submission_selected`在内层选择；复杂度顺序固定。A参数先按原程序选择，全执行族共享，不增加笛卡尔参数搜索。`tuning_trials.csv`记录8条规则的全部约束/分数/一标准误，`splits.json`的每个内层条目保存`submission_validation_metrics`供独立重建。无合格者回退A并标记`execution_selection_feasible=false`。

`make_submission_policy(name,model,params,config,horizon)`构造期限相关策略。S1保留原五点并加入`continuous_fraction(c,q,f)`，对每个候选先筛费用门槛；S2固定0.05偏离带并对到边界动作执行同一门槛。限幅C一步在`rollout`规划动作阶段以`move_cap`限制搬动比例，当前费用/风险/续接均用限制后的目标重新计算；后续A续接不改动，初次建仓不受限。B长组采用原窄动作Ridge，H<=6零倾斜，其他正式期限分别0.25/0.5。所有新策略保留英文客户说明及原人工复核机制。

`evaluation_schedule(config)`返回费用及对应期限。新增`customer_stress_costs=[0.005]`使双倍费用也覆盖全部五个客户期限，默认旧配置行为不变；开发、最终及审计使用同一日程。新增指标`execution_rule`为实际执行规则，`strategic_bill_min`为公开战略目标集的最低bills比例（C同时含A锚及防御目标），不是路径实际持仓下限。

客户主规则读取`submission.balanced_policy`。高流动性采用公开战略目标最低bills比例0.4作为代理，不能声称满足未知提款或保证实际持仓下限。增长产品H>=15限制不变。实际执行P90风险次序反转时转人工；推荐同时检查官方与双倍费用的容忍度越界上界。家族大小440事前固定，覆盖5客户×11种唯一产品/程序×4种流动性口径×2种费用，不在选择后缩减；这些OOF二项上界仍仅为诊断。`customer_rule_sensitivity.csv`和`customer_candidate_risk.csv`保存原标签限制及0.4/0.2/0.3目标比例的全部诊断，不据其收益选择阈值。

`submission_comparisons.csv`、`submission_screen.csv`保存全部匹配比较与原增长/风险晋级条件；`submission_decision.json`按预声明规则只在内层选择程序与校准A之间确定拟提交版本，不从外层逐个挑战者中挑赢家。冻结沿用该实际策略ID，保留全部已完成开发的扩展参照，最终客户分配不重选。报告生成继续关闭。

`scripts/verify_submission.py <development_run>`独立重建全部8规则×5期限×5外折的选择约束、一标准误、旧A/C1金融结果及限幅/净价值/回撤记忆，并复拟合训练信号。`scripts/verify_saved_final.py <final_run>`仅从保存账本复核最终核算和冻结客户分配，不重读市场数据。`scripts/package_submission.py --development-run ... --bundle ... --final-run ... --out ...`在核验通过后打包源码和数值证据，检查50MB上限及逐文件hash；详见[提交与复现](SUBMISSION.md)。

开发目录中的`suitability.csv`为配置默认研究候选的映射；依据预声明采用决定另外保存`selected_suitability.csv`、`selected_suitability_risk.csv`作为实际提交候选诊断。正式冻结仍用实际策略ID重算，冻结包的`frozen_suitability`是最终客户分配依据；最终输出只追加评价，不重新推荐。

## 跨分支融合研究（2026-10-17）

`scripts/run_fusion_research.py --out benchmark/outputs/<新目录>`读取`configs/fusion_research.json`，复用已保存的五折A模型和内层选参，仅评价开发1–120。`scripts/fusion_policy.py`提供三个研究策略：A锚加持有延续MPC、追加Cai回撤动作、追加Cai期限缩放回撤动作。初始配置、复核规则、收益成本门槛与原A一致；所有后续候选动作先限搬动比例10%，再计算训练情景同年度后缀的含费对数增长、尾部回撤和风险屏障。期限缩放使用客户原始H/25，不是剩余年限；它只改变新增动作，不改初始配置。

研究脚本独立于正式策略工厂，不能据此直接冻结/最终评价。它保存完整年度账本、五期限两费用指标、配对区间和独立标量核验；复现A共同1200条指标。`manifest.json`记录训练来源hash、脚本hash、设置以及最终集已暴露的事实，不能称为新的独立确认。客户映射不改变，也不移植Cai分支的成本豁免或风险筛选放宽。原冻结包/最终锁和核心源码hash保持不变。

`scripts/summarize_fusion.py <run>`只读取已保存指标/账本，生成`screen.csv`与`action_counts.csv`，拒绝覆盖；沿用既有增长/风险筛选阈值，保存逐条件布尔值、折间方向与双倍费用差。筛选不包含新客户采用认证，不构成提交自动选择。

### 融合执行效率迭代（2026-10-18）

`scripts/run_fusion_iteration.py --config configs/fusion_efficiency.json --out benchmark/outputs/<新目录>`复用同样的五折A模型，运行配置中固定的执行规则。`EfficientFusion`保留独立A建议器，新增安全边际只控制最终动作，不改变A候选建议；可配置`reverse_wait=1`在交易下一年拒绝权益方向相反的候选，人工复核仍优先。`efficient_base`复现旧回撤融合1200条指标。

每条实际交易均按该规则的真实buffer核验，保存完整`decisions.jsonl.gz`、指标、比较及标量核算。`scripts/analyze_fusion_iteration.py <run>`生成不覆盖的效率筛选：30%费用下降、增长损失≤0.1个百分点、P90恶化≤1个百分点、至少4/5折费用下降，双倍费用同样满足；另报告区间和旧金融筛选，不把费用改善冒充增长显著提高。旧实验manifest和源码hash不重写，核心冻结流程不变。

第二轮配置`fusion_efficiency_round2.json`增加2%/4%边际、1%加反向等待及`remaining_scale=true`的剩余期限价值归一化对照。第三轮`fusion_efficiency_round3.json`固定主研究候选`efficient_horizon`：H≤6直接使用原A的建议与原buffer，较长期采用2%门槛融合；短期不施加融合10%限幅。另保留长期1%反向等待及`annual_buffer=0.0005`转换为`1-exp(-annual_buffer*H)`的平滑对照。主候选在第三轮输出及新划分运行前固定，不按各期限最优结果拼接。

`scripts/refit_fusion_stability.py --out benchmark/outputs/<新目录>`使用预登记seed20261019重新做开发五外折/三内折和原A参数选择，重新拟合路径模型，再评价固定主候选与五个参照。保存完整内层审计、参数、模型、年度账本和原始指标；所有来源均限1–120。该重复划分是训练/划分敏感性检查，不能作为新的独立验证；原最终包、锁和提交ZIP保留。

## 长期融合执行校准研究（2026-10-18）

`scripts/run_fusion_calibration.py --out benchmark/outputs/<新目录>`读取`configs/fusion_calibration.json`。固定复用seed20261014/20261019的已存外层A模型和完整划分；各内折重新拟合战略模型及未来路径库，仅用该内折训练情景。原五期限选参A作为短期固定程序和当前低换手基线；K1按长期原A路径共选原六参数，K2按长期融合实际路径共选，K2为预指定主挑战者。所有期限独立模拟；同参数年度参照包含真实费用，复核保持行为与原实现一致。金融核心工厂和冻结入口不变。

`scripts/fusion_calibration.py`提供`inner_scores(...)`和`choose_group(...)`：前者拒绝训练/内层验证重叠，返回原A及融合相对同锚年度的实际路径指标；后者仅在指定期限组检查原固定约束，保留全部候选和不可行记录，完整性检查拒绝缺失/重复情景期限，沿用原A排序及最小违约失败标志。外层验证结果不进入参数选择。

运行根目录保存不可覆盖manifest、内层原始指标及总核验；两个seed子目录各保存完整内外层模型/划分、参数、调参JSON、指标、汇总、JSONL压缩年度账本。同锚A/年度/持有参照分别对应K1/K2实际选中配置。`scripts/verify_calibration_ledger.py <seed子目录>`使用独立标量核算全部账本、收益门槛、限幅、复核后保持，并比较K1/K2短期逐年记录的精确hash；核验工具自己的源码hash另存于审计JSON。原`verify_saved_iteration.py`仍可重放相同账本格式。`scripts/analyze_fusion_calibration.py <运行根目录>`从保存的内层指标独立重算约束和排名，输出选择核验、费用保护、金融筛选及`decision.json`；不拟合、不评价新路径。后续扩展只有两套划分全部K2内层组通过、短期重现且所有长期双费用均不超过基线110%时才启动。两个新程序的条件区间按每指标/参照/期限/费用/划分校正，仍是开发探索，不能替代新独立数据。

`scripts/run_fusion_value_extensions.py --out benchmark/outputs/<新目录>`由`configs/fusion_value_extensions.json`定义两个独立规则，启动时强制检查上述K2扩展门槛。复用已审计的内折模型而非外折模型，分别按实际路径重新选择六参数。`fusion_a_continuation`保留首步候选，预测后续按原含费A决策、复核后保持的路径；`fusion_uncertainty`在固定200次完整训练情景bootstrap中重算价值差，减一个标准差后排名和门控，不重拟合锚，不作为统计置信保证。加权计数实现与显式重采样的ES/线性分位数精确等价；训练bootstrap种子20261021固定。

扩展运行每套划分包含16策略/参照、19,200条外层指标，其中9,600条为复制已审计的参照，9,600条为新评价；账本完整保存。短期两个新程序与当前A逐年完全相同。分析工具可通过配置`families`、`baseline`、`cost_baseline`、`references`和`research_priority`处理该固定扩展：主比较K2，费用保护及额外金融比较仍对原低换手K0，按原优先顺序检查两个程序且完成后停止。新程序的`k2_anchor_*`对应含费A延续，`k1_anchor_*`对应不确定性规则。历史运行的manifest与原hash不修改，可从相应Git提交恢复当时工具。

## 最终收敛的预算惩罚研究（2026-10-18）

`scripts/run_final_budget_iteration.py --out benchmark/outputs/<新目录>`读取`configs/final_budget_iteration.json`，固定比较`budget_uncertainty`和`budget_mean`。沿用2%门槛、10%限幅及短期原A，对预测对数回撤ES超出同训练验证预算的部分施加`max(ES_log-budget,0)/H`价值惩罚；前者在包含该惩罚的200次训练bootstrap中减一个标准差，后者使用均值。系数固定1；实际路径仍按原六参数、原风险/增长/P10条件校准，不提供风险上限保证。

每个内折仅使用对应已存训练模型，六参数任务通过四个独立Python子进程并行，子进程只读开发分区，返回按trial顺序合并的CSV。并行数是运行设置，不是金融参数。每套划分保存17个程序/参照、20,400条外层路径，其中10,800条复制已审计参照；同锚对照分别保留。分析工具可选`required_release_references`，本轮额外要求相对原A通过原金融路线；不根据结果更改标准。两规则完成后作最终采用决定并打包；没有通过者保留原冻结A，最终集不再评价。

打包工具`package_submission.py`增加可重复的`--research-run <开发目录>`与可选`--release-decision <JSON>`。默认四个原参数的行为保持；追加研究目录只纳入白名单中的配置、汇总、约束/选择和核验证据，不包含大账本/模型缓存。发布决定必须为`status=ready`、`holdout_reused=false`且所选金融策略与原冻结包一致，否则拒绝。原金融程序、历史manifest不改写；新源码包与旧包使用不同文件名，独立解压测试并重打包核验。

`scripts/finalize_release.py <最后开发目录> --tests-passed <已通过数量>`在所有新候选未通过采用条件时核对原冻结A、最终锁、保存最终核验及原五客户映射/实际目标，生成不可覆盖的`release_decision.json`和`customer_verification.json`，不打开市场数据或模拟策略。若有新候选通过则拒绝跳过其全开发重拟合和客户检查；不能借此工具静默回退或替换新候选。发布决定中的路径核验数量包含该批明确声明的复用参照。

`scripts/verify_saved_iteration.py <run>`只重放保存的JSONL账本与开发市场，不重新运行策略；独立核验全部年度参照及研究路径的扣费、收益、通胀、回撤、增长、费用和交易率，并保存不可覆盖的`independent_replay.json`。
