"""Build the Chinese team explainer from saved evidence, without market-data access."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "benchmark/outputs"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def pct(value):
    return f"{value * 100:.2f}%"


def validate_summary(frame, summary, first_id, last_id):
    assert not frame.duplicated(["policy", "horizon", "cost_rate", "scenario_id"]).any()
    assert frame.scenario_id.between(first_id, last_id).all()
    assert frame.groupby(["policy", "horizon", "cost_rate"]).size().eq(last_id - first_id + 1).all()
    for r in summary.itertuples():
        g = frame[(frame.policy == r.policy) & (frame.horizon == r.horizon) & (frame.cost_rate == r.cost_rate)]
        checks = [(r.median_growth, g.real_cagr.median()), (r.p10_growth, g.real_cagr.quantile(.1)),
                  (r.median_loss, g.real_max_drawdown.median()), (r.p90_loss, g.real_max_drawdown.quantile(.9)),
                  (r.mean_real_cost, g.total_real_cost.mean()), (r.mean_trade_rate, g.trade_rate.mean()),
                  (r.terminal_loss_probability, g.terminal_loss.mean())]
        assert all(abs(a - b) < 1e-12 for a, b in checks)


def build(research):
    research = Path(research).resolve()
    dev = OUTPUTS / "development_submission_v1"
    final = OUTPUTS / "final_submission_v1"
    ds, fs = [pd.read_csv(p / "summary.csv") for p in [dev, final]]
    dm, fm = [pd.read_csv(p / "scenario_metrics.csv") for p in [dev, final]]
    validate_summary(dm, ds, 1, 120)
    validate_summary(fm, fs, 121, 180)
    bundle = json.loads((ROOT / "benchmark/submission_frozen_v1.json").read_text())
    profiles = json.loads((ROOT / "configs/profiles.json").read_text())
    assignments = pd.read_csv(final / "suitability.csv").set_index("profile_id")
    risks = pd.read_csv(dev / "selected_suitability_risk.csv")
    new = pd.read_csv(research / "customer_comparison.csv")
    decision = json.loads((research / "decision.json").read_text())
    audit = json.loads((research / "independent_replay.json").read_text())
    assert audit["passed"] and not decision["replace_frozen_submission"]
    blocks = []

    def add(text):
        blocks.append(text.strip())

    add(r"""
# SafeInvest 解题讲解与数据使用指南

更新日期：2026-10-18。用途：供团队理解模型、核对数据、讨论结论和准备英文report；本文不是正式论文，不代替题目或原始数值证据。

**当前结论：最终金融程序保留冻结的校准A，客户A/D人工、B/C保守、E均衡。最后客户风险优化确实降低了部分开发回撤，但收益代价较大、内层并非全部可行，未通过原晋级条件。不能写成“所有问题已经解决”。**

本文中的数值是保存结果的说明快照。公式、接口、协议与历史结论的权威入口依次为[METHODOLOGY](METHODOLOGY.md)、[INTERFACES](INTERFACES.md)、[PROTOCOL](PROTOCOL.md)、[FINDINGS](FINDINGS.md)；后续修改应先改对应原始来源，再用文末命令重建本文。

## 1. 题目到底要求回答什么

题目的核心是可信、可解释的自动投顾，不是单纯追求最高收益。它没有给出可以据此计算官方总分的加权公式。

| 题目任务 | 我们需要交代的内容 | 当前证据 |
| --- | --- | --- |
| 客户适配 | 三产品配置规则；期限、购买力损失容忍度、流动性如何进入推荐；A–E及人工理由 | 冻结分配、开发适配表、五客户最终结果 |
| 自动投资 | 何时保持、调仓、换目标、转人工；为何估计收益足以覆盖成本 | 因果接口、成本门槛、逐年度账本 |
| 对比基准 | 必需等权年度调仓；中位收益/大损失、P10收益/P90大损失；交易比例 | 保存的开发与最终汇总、配对比较 |
| 严重案例 | 展示一次损失如何发生，安全措施做到了什么 | 最终场景155的保存账本 |
| 可靠性 | 费用或重要假设敏感性；一项局限；何时人工及等待期间行为 | 双费用、费用敏感性、失败候选、人工诊断 |

重点：客户容忍度是问卷回答，不是题目要求保证的硬损失上限；普通波动与严重损失必须分开；原始数据是合成独立情景，不是连续历史回测或未来预测。

## 2. 数据与实验：先分清三层结果

原始字段为`scenario_id, period, equity_return, bond_return, bill_proxy_return, inflation`，资产顺序固定为股票、政府债券、短期bill。180个情景各30年；开发120个、最终60个。详细来源与许可见[DATA](DATA.md)。

| 层次 | 使用数据 | 能支持的结论 | 不能支持的结论 |
| --- | --- | --- | --- |
| 开发交叉验证 | 1–120，外5折/内3折 | 比较、选参、发现失败和取舍 | 反复研发后的完全无偏成绩 |
| 原冻结最终评价 | 121–180，只执行一次 | 原冻结A及预声明参照的留出表现 | 用最终表现改选候选再声称独立验证 |
| 后续研究 | 仍只用1–120，原划分及新划分 | 研发稳定性、局部改进、结构性诊断 | 新的独立最终成绩 |

统计抽样单位是情景。120×30=3,600个年度观测不等于3,600个独立样本；同情景多个期限、两套划分也不独立。

嵌套交叉验证的顺序是：外层暂存24条路径；在其余96条内部再分训练/验证；模型及参数只由该96条决定；最后对24条外层路径评价。全开发集重拟合后才冻结。验证情景的当年与未来收益不会进入策略上下文。

真实最终集已在2026-10-17使用。本轮没有再读最终市场CSV或运行新最终策略。最后新版本也没有最终成绩；不能把原A的最终成绩移接给它。

## 3. 核算为何必须先于模型

每条路径从100单位bills开始，无后续存取款；只做多、满仓；年度检查；不计税、其他费用和终值清算费。初始建仓也收费。

对交易前财富V、当前权重p、新目标w、费率c：

```text
转移金额 Q = V × ||w-p||₁ / 2
交易费用 F = cQ
扣费后持仓 h_i⁺ = (V-F)w_i
年末持仓 h_i,end = h_i⁺(1+r_i,t)
实际财富 W_t = Σh_i,end / ∏(1+inflation_t)
```

100单位bills转20单位到股票，Q=20而不是40，费用0.05；扣费后股票19.99、bills79.96，总财富99.95。费用立即从财富扣除，净CAGR上不能再扣一遍。

第t年交易只用截至t−1年的信息。费用标记和年末财富都进入回撤计算，避免忽略交易当时的损失。数据是年度频率，无法观察实际年内最低点；本文的回撤不能称为日内或逐日回撤。

## 4. 指标：哪些数能比较，哪些不能

| 指标 | 数学/计算口径 | 常见误读 |
| --- | --- | --- |
| 实际CAGR | `(W_H/100)^(1/H)-1`，先逐路径算 | 不是平均单年收益；不是名义收益 |
| 收益中位数 | 跨路径CAGR的中位数 | 不等于每位客户确定会获得的收益 |
| P10收益 | 跨路径CAGR第10百分位，线性插值 | 不是单年最差收益 |
| 年度波动 | 每条路径年度实际收益样本标准差，再跨路径取中位 | 不能代替严重损失指标 |
| 最大回撤MDD | 从既往实际财富峰值到后续标记的最大比例下降 | 和期末是否亏损不同 |
| P90回撤 | 路径MDD第90百分位 | 不是最大可能损失或90%置信上限 |
| 回撤CVaR90 | 最坏10%路径MDD的平均，分数边界质量 | 不是P90；也不是对数回撤CVaR |
| 期末亏损率 | `W_H<100`路径数/路径总数 | 中途大跌后恢复也可能不计期末亏损 |
| 交易频率 | 第2至H年有交易的次数/(H−1)，再跨路径平均 | 不是换手率；另一列含初始建仓 |
| 累计实际费用 | 各期费用除以当时交易前累计通胀后求和，再取路径均值 | 初始100的财富单位，不是年费率或复利终值损失 |
| 客户越界率 | `MDD>客户容忍度`的路径比例 | 是样本观察，不是损失保证 |
| 复核率 | 至少进入过一次人工复核的路径比例 | 不是年度人工动作占比，也不是救助成功率 |

所有表格除另标外采用费率0.25%。比较收益要匹配期限、费用、情景及资产约束。不同客户的期末财富跨15/30/6年，不能据此排名；权重不同的策略差异也不能都归因于调仓时机。

## 5. 计划A：战略配置与实际执行分开理解

### 5.1 战略层在优化什么

对训练路径的固定年度权重w，令最大对数回撤为：

```text
Y_s(w) = max_{0≤u≤v≤H} -Σ[t=u+1..v] log(1+r_real(s,t)ᵀw)
minimize -mean[s,t] log(1+r_real(s,t)ᵀw)
subject to ES90(Y_s(w)) ≤ risk_budget, w≥0, Σw=1
```

负对数增长、区间最大值、经验尾部均值保留凸性，因此这是无费用、固定年度配置的凸战略子问题。切平面LP提供目标下界，可行配置提供上界；保存原始/对偶证书和收敛间隙。它不证明整个动态投顾系统全局最优，也不保证样本外损失。

风险预算来自同期限的20/30/50、50/35/15、75/20/5参照组合，再乘训练安全比例。当前校准候选是0.75/0.85/0.95，固定验证风险比例1.0；gamma候选4/12。保守产品有战略bills下限40%。这些是研究设计，不是题目规定。

普通MDD与对数回撤满足`D=1-exp(-Y)`，但非线性变换不能直接移到CVaR之外。战略约束也不能自动传递到有费用、自然漂移和人工复核的实际路径，所以还要做含费执行校准。

### 5.2 执行层如何决定交易

把战略锚w*视为长期偏好点，惩罚偏离锚的风险；候选是保持或向锚移动0/25/50/75/100%。对候选价值A和本次费用比例f，比较`(1-f)A`，且严格满足：

```text
estimated_benefit > actual_fee_fraction + 0.0001
```

这里0.0001是财富比例、即1个基点的模型安全余量。价值是战略风险偏好代理，不能写成预测收益一定会改善。引擎独立核验费用与门槛。初次配置按入门规则建仓收费；后续才使用交易门槛。

四种动作：保持现有资产；恢复当前目标；改变目标并交易；转人工。部分调整后的当前目标可能改变，但战略锚保持固定；这不是每年重新预测出一套战略权重。

### 5.3 人工不是止损交易

原A在已观察回撤超过训练阈值时转人工，阈值来自训练含费年度策略P90的1.5倍，并截到10%–75%。等待期间继续持有现有资产、估值并停止自动调仓。系统没有假定顾问即时响应、卖成bills或挽回损失。

因此，“提前转人工”与“提前减仓”是两件事。本轮诊断会验证前者的真实效果，不能直接当成风险已被控制。

## 6. 三个产品究竟是什么

固定保守20/30/50、均衡50/35/15、成长75/20/5是预算参照和诊断基准；最终A产品的实际配置由相应期限的凸优化给出，不是始终固定这些比例。完整产品定义是“期限相关的配置规则+实际执行规则”，不能只列一个30年向量代表所有期限。

下表直接读取原冻结模型，不重新拟合；比例顺序为股票/债券/bills。
""")
    rows = []
    for horizon in [5, 6, 15, 25, 30]:
        weights = []
        for product in ["conservative", "balanced", "growth"]:
            ratio = bundle["params"]["plan_a_" + product]["risk_ratio"]
            entry = bundle["model"]["plan_a"]["entries"][f"{product}:{horizon}:{ratio:g}"]
            weights.append(" / ".join(pct(w) for w in entry["weights"]))
        rows.append([f"{horizon}年", *weights])
    add(table(["期限", "A保守", "A均衡", "A成长"], rows))
    add("""
产品名称不等于风险保证。入门筛选还要看实际执行路径的风险次序、期限、流动性及越界诊断；不能因为叫Conservative就忽略其30年回撤。

## 7. Benchmark：开发与最终结果

### 7.1 原17种基准在开发集30年的完整概览

这是120条开发路径的交叉验证结果。完整原始策略说明见[METHODOLOGY](METHODOLOGY.md#17-种基准策略)，不可据此事后挑最高收益作为提交程序。
""")
    original_names = ["equal_annual", "equal_hold", "bills_hold", "conservative_annual", "balanced_annual",
                      "growth_annual", "balanced_hold", "balanced_periodic", "balanced_threshold",
                      "minimum_variance", "inverse_volatility", "mean_variance", "cvar_constrained",
                      "balanced_gate", "cost_aware", "adaptive", "adaptive_review"]
    rows = []
    for name in original_names:
        r = ds[(ds.policy == name) & (ds.horizon == 30) & (ds.cost_rate == .0025)].iloc[0]
        rows.append([name, pct(r.median_growth), pct(r.p10_growth), pct(r.median_loss),
                     pct(r.p90_loss), f"{r.mean_real_cost:.4f}", pct(r.mean_trade_rate)])
    add(table(["策略", "收益中位", "P10收益", "回撤中位", "P90回撤", "实际费用", "交易率"], rows))
    add("""
### 7.2 原冻结最终评价：30年代表性比较

每行60条相同最终路径。最终共21个预声明策略，下面列出10个代表；完整表在[final summary](../benchmark/outputs/final_submission_v1/summary.csv)。B/C候选的最终数值是原预声明参照，不能根据这些数值改选提交赢家。
""")
    names = ["equal_annual", "conservative_annual", "balanced_annual", "balanced_hold", "growth_annual",
             "plan_a_annual", "plan_a_balanced", "plan_a_conservative", "submission_b_long05", "submission_c_cap10"]
    rows = []
    for name in names:
        r = fs[(fs.policy == name) & (fs.horizon == 30) & (fs.cost_rate == .0025)].iloc[0]
        rows.append([name, pct(r.median_growth), pct(r.p10_growth), pct(r.median_loss),
                     pct(r.p90_loss), f"{r.mean_real_cost:.4f}", pct(r.mean_trade_rate)])
    add(table(["策略", "收益中位", "P10收益", "回撤中位", "P90回撤", "实际费用", "交易率"], rows))
    add("""
A相对等权：收益中位数高0.7787个百分点/年，P90回撤也高8.4900个百分点。相对同锚年度：收益中位数高0.1325个百分点、P90回撤低0.1031个百分点、费用低26.83%。同锚比较更有助于判断执行规则；不能把不同配置的收益差全部写成自动择时价值。

保存的配对bootstrap中，A相对等权的收益中位数差95%区间约为+0.325至+1.184个百分点；调整后的家族区间约+0.098至+1.372。风险同时显著增加，不能宣称风险收益全面占优。这些区间也不覆盖最终暴露后新增策略的表现。

### 7.3 同一战略锚下，自动执行有没有价值

这些策略初始配置一致，有助于分离配置与执行。完全门控/无复核对照不启用人工；比较时不能把复核差异忽略。
""")
    rows = []
    for name in ["plan_a_balanced", "plan_a_annual", "plan_a_hold", "plan_a_band_edge", "plan_a_full_gate", "plan_a_no_review"]:
        r = fs[(fs.policy == name) & (fs.horizon == 30) & (fs.cost_rate == .0025)].iloc[0]
        rows.append([name, pct(r.median_growth), pct(r.p10_growth), pct(r.p90_loss),
                     f"{r.mean_real_cost:.4f}", pct(r.mean_trade_rate), pct(r.review_rate)])
    add(table(["同锚执行", "收益中位", "P10收益", "P90回撤", "实际费用", "交易率", "曾复核路径比例"], rows))
    add("""
相对同锚持有，A的P90回撤低约11.13个百分点，收益中位低约0.11个百分点；相对同锚年度，费用更低且收益点估计稍高。这支持“门控执行存在风险/成本取舍价值”，不支持“自动权重更新普遍提高收益”。本表是点估计比较，不是所有消融的独立显著性证明。

推荐保留A有限、可解释的成本门控执行，不采用没有建立增益的额外预测倾斜/复杂规划。原A本身也不是所有基准上的统计赢家。题目问自动权重更新是否有价值，应回答具体改善了什么、付出了什么，而不是只写有/没有。

最终30年A的后续交易率为35.34%，若把初次建仓也计入30个年度，则为37.50%；正式表格需说明分母。

### 7.4 均衡A的期限结果

这是统一均衡产品，不是五客户实际分配。B、C使用保守；A、D人工。
""")
    rows = []
    for r in fs[(fs.policy == "plan_a_balanced") & (fs.cost_rate == .0025)].itertuples():
        rows.append([r.horizon, pct(r.median_growth), pct(r.p10_growth), pct(r.median_loss),
                     pct(r.p90_loss), pct(r.terminal_loss_probability), f"{r.mean_real_cost:.4f}", pct(r.mean_trade_rate)])
    add(table(["年", "收益中位", "P10收益", "回撤中位", "P90回撤", "期末亏损率", "实际费用", "交易率"], rows))
    add("""
## 8. 五客户：推荐、配置、最终表现

原最终分配在最终暴露之前已冻结。A/D人工不是“收益零”或“风险零”，而是没有自动投资绩效可填写。
""")
    rows, outcomes, extra = [], [], []
    for profile in profiles:
        pid = profile["profile_id"]
        assignment = assignments.loc[pid]
        policy = assignment.policy
        allocation = "—"
        if policy != "human_review":
            product = assignment.portfolio
            ratio = bundle["params"][policy]["risk_ratio"]
            weights = bundle["model"]["plan_a"]["entries"][f"{product}:{profile['horizon']}:{ratio:g}"]["weights"]
            allocation = " / ".join(pct(w) for w in weights)
        rows.append([pid, profile["horizon"], pct(profile["loss_tolerance"]), profile["liquidity"],
                     "人工" if policy == "human_review" else assignment.portfolio, allocation])
        if policy == "human_review":
            outcomes.append([pid, *["—"] * 6])
            continue
        g = fm[(fm.policy == policy) & (fm.horizon == profile["horizon"]) & (fm.cost_rate == .0025)]
        breaches = int((g.real_max_drawdown > profile["loss_tolerance"]).sum())
        assert abs(assignment.final_breach_probability - breaches / 60) < 1e-12
        outcomes.append([pid, pct(g.real_cagr.median()), pct(g.real_cagr.quantile(.1)), pct(g.real_max_drawdown.median()),
                         pct(g.real_max_drawdown.quantile(.9)), f"{breaches}/60 ({pct(breaches/60)})",
                         f"{int(g.terminal_loss.sum())}/60 ({pct(g.terminal_loss.mean())})"])
        extra.append([pid, f"{g.terminal_real_wealth.median():.2f}", f"{g.terminal_real_wealth.quantile(.1):.2f}",
                      pct(g.real_max_drawdown.max()), pct(g.real_return_volatility.median()),
                      f"{g.total_real_cost.mean():.4f}", pct(g.trade_rate.mean()), f"{int(g.review.sum())}/60"])
    add(table(["客户", "年", "容忍度", "流动性", "分配", "初始股票/债券/bills"], rows))
    add(table(["客户", "收益中位", "P10收益", "回撤中位", "P90回撤", "容忍度越界", "期末亏损"], outcomes))
    add(table(["客户", "财富中位", "财富P10", "最坏回撤", "波动率中位", "实际费用", "交易率", "曾人工复核"], extra))
    add("""
财富初始100。B：P90回撤25.38%略超容忍度25%，最终7/60条越界；C：虽P90低于40%，仍有3条超过40%；E：6年P10年化收益为负、期末亏损9/60条。不要写成“所有客户都受到损失上限保护”。

下面比较原开发OOF与原冻结最终；两者不是同一模型的重复测量。
""")
    rows = []
    for p in profiles:
        pid = p["profile_id"]
        if pid not in ["B", "C", "E"]:
            continue
        policy = assignments.loc[pid].policy
        groups = [f[(f.policy == policy) & (f.horizon == p["horizon"]) & (f.cost_rate == .0025)] for f in [dm, fm]]
        rows.append([pid, " → ".join(pct(g.real_cagr.median()) for g in groups),
                     " → ".join(pct(g.real_cagr.quantile(.1)) for g in groups),
                     " → ".join(pct(g.real_max_drawdown.quantile(.9)) for g in groups),
                     " → ".join(f"{int((g.real_max_drawdown > p['loss_tolerance']).sum())}/{len(g)}" for g in groups)])
    add(table(["客户", "收益中位：开发→最终", "P10收益", "P90回撤", "越界数"], rows))
    add("""
B/C最终P90回撤均比开发更高，B的越界率从2.50%升到11.67%；这正是不能把开发适配当作未来保证的实例。

### 8.1 为什么不是给A、D也分一个保守产品

当前入门条件包含：客户期限；高流动性目标bills至少40%；成长产品至少15年；正常/双费用越界上界均不高于20%；三产品实际风险次序。20%是服务设计假设，40%也不是提款满足率。

原客户家族大小440，对5%总错误概率做Bonferroni分配，每次单侧二项界使用`1-0.05/440`置信水平。对OOF结果，模型随折改变且训练集重叠，二项界只是筛选诊断，不能直接当作固定重拟合模型的严格覆盖率保证。60条最终样本也不能认证罕见风险。
""")
    rows = []
    for pid in ["A", "B", "C", "D", "E"]:
        product = "balanced" if pid == "E" else "conservative"
        r = risks[(risks.profile == pid) & (risks.portfolio == product)].iloc[0]
        rows.append([pid, product, f"{r.breaches}/120", pct(r.breach_probability), pct(r.upper_bound),
                     pct(r.stress_upper_bound), "通过" if r.eligible else "不通过"])
    add(table(["客户", "检查产品", "越界数", "经验越界率", "诊断上界", "双费用最坏上界", "开发适配"], rows))
    add("""
A/D在当前候选与规则下不合格，人工是透明的拒绝自动化，不是模型失败后删掉难样本。C的均衡候选即使经验越界率14.17%低于20%，其调整上界28.93%仍高于20%，所以原分配选保守。

## 9. 严重损失案例：最终场景155

仅解释保存结果，未据此调整模型。
""")
    severe = json.loads((final / "severe_case.json").read_text())
    add(table(["项目", "数据"], [["实际财富峰值", f"第6年末，{severe['peak']:.4f}"],
        ["实际财富低点/终值", f"第30年末，{severe['terminal_real_wealth']:.4f}"],
        ["最大回撤", pct(severe["max_drawdown"])], ["实际CAGR", pct(severe["real_cagr"])],
        ["首次人工复核", "第23年"], ["后续交易次数", severe["trades"]]]))
    add("""
30年A的回撤CVaR90为59.31%，最大回撤65.19%，不能只展示P90的48.02%。复核发生在已观察到严重损失之后，等待期间保持资产，后续损失没有被神奇地消除。这是必须解释的局限。

## 10. 费用敏感性

原冻结A，最终60条、30年。费用变化会改变交易动作，所以收益不必随费用单调下降。
""")
    rows = []
    for r in fs[(fs.policy == "plan_a_balanced") & (fs.horizon == 30)].sort_values("cost_rate").itertuples():
        rows.append([f"{r.cost_rate * 100:.3f}%", pct(r.median_growth), pct(r.p10_growth), pct(r.p90_loss),
                     f"{r.mean_real_cost:.4f}", pct(r.mean_trade_rate)])
    add(table(["费率", "收益中位", "P10收益", "P90回撤", "实际费用", "交易率"], rows))
    add("""
0.25%翻倍到0.50%时，收益中位数4.15%变4.13%、交易率35.34%变28.68%。这支持费用适应性，不能因此说交易费不重要；实际累计费用从0.5834升到1.0668。

## 11. B/C与跨分支融合：为什么实现了却没有提交

| 路线 | 关键想法 | 我们学到什么 |
| --- | --- | --- |
| B | 过去最多5年的信号，固定Ridge预测股票相对bills收益，倾斜A的动作 | 弱预测信号没有建立稳健增益；必须与同动作无信号对照比较 |
| C | 相同训练未来库与动作族，比较一步/三步规划 | 多步规划更复杂，并没有自动带来更好的样本外结果 |
| 跨分支融合 | 组合A、防御和回撤反馈建议，限幅并统一估值 | 早期回撤降低却交易过多，费用抵消收益 |
| K0低换手 | 短期沿用A、长期提高净改善门槛 | 相对旧融合大幅降费，但金融非劣区间仍未通过 |
| K1/K2执行校准 | 按实际部署的长期规则重新选参 | 改善内层可行性；不能只校准一个策略再执行另一个 |
| 不确定性/预算 | 对动作价值差惩罚训练估计不确定性和超预算风险 | 回撤与费用有改善，但完整收益/风险晋级仍未通过 |

下一表均为开发30年；每格是原划分/第二套划分。两套同样本不独立。
""")
    rows = []
    old = OUTPUTS / "development_final_budget_v1"
    for name in ["plan_a_balanced", "efficient_horizon", "calibration_group_fusion", "fusion_uncertainty",
                 "budget_uncertainty", "budget_mean"]:
        rr = []
        for seed in [20261014, 20261019]:
            s = pd.read_csv(old / str(seed) / "summary.csv")
            rr.append(s[(s.policy == name) & (s.horizon == 30) & (s.cost_rate == .0025)].iloc[0])
        rows.append([name, " / ".join(pct(r.median_growth) for r in rr),
                     " / ".join(pct(r.p90_loss) for r in rr), " / ".join(f"{r.mean_real_cost:.4f}" for r in rr),
                     " / ".join(pct(r.mean_trade_rate) for r in rr)])
    add(table(["版本", "收益中位", "P90回撤", "实际费用", "交易率"], rows))
    add("""
这些是金融取舍，不是官方分数。没有通过完整条件，就保留冻结程序；不能以某个期限或某套划分最好为理由改选。

## 12. 本次最后优化：直接按客户风险收紧配置

### 12.1 固定规则与验证顺序

新增配置为`configs/customer_risk_review.json`，程序在`scripts/customer_risk.py`。原金融核心没有改动。

1. 保留对应产品原A参数，用相同训练集的最小方差配置作为防御锚。
2. 新战略目标为`w_new = lambda*w_A + (1-lambda)*w_minvar`，仅测试预声明`lambda={0,.25,.5,.75,1}`；这是建仓目标混合比例，与A每次交易的部分移动比例是不同变量。
3. 原外层训练96条内，复用三个内折的训练模型，完整模拟该客户期限、正常/双费用的实际门控路径。
4. 要求P90不超过客户容忍度80%，并且比原A降低至少3个百分点；越界诊断上界不超过20%，高流动性目标bills至少40%。在合格比例中选择收益中位数最高者。
5. 五个比例都失败时保留最小违约诊断，明确`feasible=false`；相应外层数值是反事实诊断，不可当作已合格自动服务表现。
6. 独立诊断：同一选定配置把人工复核线提前到客户容忍度80%，不据此另选赢家。另保留原A、同新锚年度、bills和等权对照。

本轮选择家族500=5客户×5比例×2费用×2划分×5外折，单侧二项置信水平0.9999；仍是OOF选择诊断。内层没有额外要求收益只下降0.5个百分点，因此“客户风险条件合格”和“原金融晋级”必须分别检查。

A/D比较用原保守产品作为反事实参照，并非改变原人工分配。B/C各用原保守参数、E用原均衡参数，未把所有人都套用均衡参数。

### 12.2 两套划分的内层可行性
""")
    counts = decision["feasible_folds"]
    rows = []
    for pid in ["A", "B", "C", "D", "E"]:
        rr = [next(r for r in counts if r["seed"] == seed and r["profile_id"] == pid)["feasible_folds"]
              for seed in [20261014, 20261019]]
        rows.append([pid, f"{rr[0]}/5", f"{rr[1]}/5", "两套全部通过" if min(rr) == 5 else "存在失败，不可声称全部校准成功"])
    add(table(["客户", "原划分", "第二套划分", "解释"], rows))
    add("""
### 12.3 外层风险改善和收益代价

下表均为开发120路径、正常费率，展示原A与新预算规则。B原划分包含两个内层失败折，必须作为完整反事实诊断阅读；不能删除那两折来美化结果。
""")
    rows = []
    for seed in [20261014, 20261019]:
        for pid in ["B", "C", "E"]:
            a = new[(new.seed == seed) & (new.profile_id == pid) & (new.policy == "original")].iloc[0]
            b = new[(new.seed == seed) & (new.profile_id == pid) & (new.policy == "customer_budget")].iloc[0]
            rows.append([str(seed), pid, f"{pct(a.median_growth)} → {pct(b.median_growth)}",
                         f"{pct(a.p90_loss)} → {pct(b.p90_loss)}", f"{a.breaches}/120 → {b.breaches}/120",
                         f"{a.mean_real_cost:.4f} → {b.mean_real_cost:.4f}"])
    add(table(["划分", "客户", "收益中位", "P90回撤", "越界数", "实际费用"], rows))
    add("""
C：P90回撤约27%降到22%，年化中位收益损失约0.94–1.05个百分点；E：P90约26%降到21%，收益损失约0.67–0.77个百分点。原风险晋级只允许中位增长损失不超过0.50个百分点，还需满足区间等条件；不能为采用新模型而放宽标准。

**本轮全部客户/规则都未通过完整原风险晋级。** 风险下降是有效点估计，代价也真实存在。新规则不是原A的全面升级，未替换冻结提交，没有新最终成绩。

### 12.4 提前人工有没有解决尾部损失
""")
    rows = []
    for pid in ["A", "B", "C", "D", "E"]:
        a = new[(new.seed == 20261014) & (new.profile_id == pid) & (new.policy == "customer_budget")].iloc[0]
        b = new[(new.seed == 20261014) & (new.profile_id == pid) & (new.policy == "customer_early_review")].iloc[0]
        rows.append([pid, f"{pct(a.p90_loss)} → {pct(b.p90_loss)}", f"{pct(a.review_rate)} → {pct(b.review_rate)}",
                     f"{a.breaches}/120 → {b.breaches}/120"])
    add(table(["客户，原划分", "P90回撤：预算→提前复核", "曾复核路径比例", "越界数"], rows))
    add("""
提前复核主要增加了人工路径数量，没有改善上述主要P90指标；D原划分越界还从39条变40条。原因不能归结为所有人工服务无效：本模拟没有顾问干预效果，待处理只保持资产，所以它检验的是“停止自动调仓并等待”，不是实际顾问服务或止损。

## 13. 一个可证明的边界：为什么不能承诺零越界

以下是本轮新增的数学诊断，不依赖某个拟合策略。令第t年三资产最大的实际总收益倍数为：

```text
g_max,t = max_i [(1+r_i,t)/(1+inflation_t)]
任意允许策略：W_t/W_(t-1) ≤ g_max,t
任意区间：W_v/W_u ≤ ∏[t=u+1..v] g_max,t
所以 MDD ≥ max(0, max_(u<v) [1-∏ g_max,t])
```

证明：只做多、满仓的单期实际总收益是三个资产实际总收益的凸组合；费用只会再降低财富。把每期上界相乘即可。历史峰值至少等于任一既往W_u，所以区间损失给出最大回撤下界。即使事先知道每年的最佳资产，也不能突破共同下跌区间的这个界。

注意：下界低不证明某个因果策略能达到它；下界越界证明该条具体路径上所有允许策略都会越界。它既不是概率估计，也不证明任何服务越界概率标准整体不可达。
""")
    bounds = pd.read_csv(research / "unavoidable_drawdown.csv")
    rows = []
    for p in profiles:
        g = bounds[bounds.profile_id == p["profile_id"]]
        count = int(g.unavoidable_breach.sum())
        rows.append([p["profile_id"], pct(p["loss_tolerance"]), f"{count}/120", pct(g.lower_bound.max())])
    add(table(["客户期限", "容忍度", "所有允许策略都无法守住的开发路径", "最大路径下界"], rows))
    add("""
A至少3条、D至少6条开发路径无法保持其问卷损失线。这说明零越界承诺不现实；并没有证明A/D在所有可能模型和20%服务概率标准下必然无法自动化。现有A/D人工决定仍基于实际候选筛选，而不是夸大这个下界。

## 14. 原采用标准与本轮解决情况

增长路线：中位增长至少+0.10个百分点/年，调整区间下界>0；P10损失不超过0.10个百分点、P90增加不超过1个百分点；至少4/5外折方向一致，费用翻倍仍增益，复核与客户适配通过。

风险路线：P90至少降低3个百分点且调整区间上界<0；中位增长损失不超过0.50个百分点、P10损失不超过0.20个百分点；至少4/5外折风险改善，双费用保持风险改善且增长损失不超过0.50个百分点，复核率不超过10%。采用还需相关同锚与均衡参照条件。以[PROTOCOL](PROTOCOL.md)为准。

| 问题 | 本轮做了什么 | 当前状态 |
| --- | --- | --- |
| 客户风险与统一产品预算脱节 | 新增客户期限/容忍度驱动的防御混合选择 | C/E内层全部通过；B仍一套有失败；A/D未适配 |
| 尾部回撤较高 | 开发C/E明显降低P90，保留费用及收益对照 | 有改善，但增长代价超原金融标准 |
| 认为提前人工就能止损 | 实际模拟提前复核且等待继续持有 | 否定此推断，不能作为自动止损宣传 |
| bills被当成购买力无风险 | 增加全资产共同损失的路径下界及证明 | 解释边界已补齐；市场风险没有消失 |
| 保存数据混用/误标 | 区分原冻结、开发研究、反事实、不可行折 | 文档与数值来源明确 |
| 独立验证不足 | 保护原冻结/最终锁，不新跑或改选最终集 | 无新独立数据，仍未解决，不能伪造证据 |

## 15. 写report时可以怎么表述

| 有证据的表述 | 需要避免的表述 |
| --- | --- |
| A在最终集比等权提高收益，同时承担更高回撤 | 我们的策略在收益和风险上全面优于等权 |
| 同锚年度对照显示节费和交易减少，收益差较小 | 全部超额收益来自智能择时 |
| 三位自动配置、两位人工；解释未适配原因 | 五位客户均获得安全自动投资 |
| 客户容忍度用于适配筛选，存在越界路径 | 客户损失被硬性限制在问卷值以内 |
| 新防御版本开发风险下降，但收益代价较大 | 最新版本已在最终集取得更高分 |
| OOF上界、条件bootstrap用于诊断 | 所有调参和人工搜索后的95%总体保证 |
| 复核期间持有现有资产，未假设顾问效果 | 触发人工后风险自动消失 |

### 15.1 参数、假设与题目条件不要混淆

- 题目条件：三资产、只做多满仓、初始100 bills、年度信息因果、0.25%费用、无后续现金流、按客户期限评价。
- 建模选择：CAGR/波动/MDD、CVaR90、嵌套划分、固定协方差收缩、三产品风险参照、费用支付分摊约定。
- 服务选择：越界概率阈值20%、高流动性bills目标40%、成长至少15年、待人工期间持有。
- 研发超参数：A训练比例及gamma、融合2%门槛与10%限幅、本轮五混合比例/80%客户余量/3个百分点改善。

不能把最后两类写成题目强制规定，也不能把合成情景的统计频率当成真实市场事件概率。

### 15.2 建议优先放进≤10页英文report的内容

这只是团队的信息组织建议，不是论文正文：

1. 开头明确结论、三产品和五客户分配。
2. 说明购买力核算、波动/MDD区别及唯一交易费用公式。
3. 一张流程图说明配置→适配→年度动作→费用门槛→人工。
4. 写清战略凸问题与实际门控，避免堆砌所有研发模型。
5. 放最终等权、A、同锚年度及必要持有对照；包含题目要求的四个分位/中位指标和交易比例。
6. 放五客户表与严重案例155；保留越界和人工情况。
7. 费用敏感性、局限、待人工行为；说明最后防御研究的取舍即可。
8. 收尾回答是否采用自动权重更新，并完整写来源、团队成员和AI使用披露。

如果篇幅有限，优先原冻结最终证据和题目必答项；后续研究不宜挤占核心解释。无须把所有实验都写成贡献，也不要用开发最佳值替代最终结果。

### 15.3 需要准备的图，以及如何避免误导

- 最终收益–P90回撤散点：明确更高收益伴随更高风险；每点同30年、同费率。
- 三种同锚执行的费用/交易率比较：帮助分离配置与执行作用。
- 五客户风险表：A/D使用“人工/不适用”，不能画成零风险。
- 场景155的实际财富曲线：标出第6年峰值、第23年复核、第30年终值；不能抹去复核后的亏损。
- 费用敏感性曲线：同时给收益与交易频率，解释为何非单调。

需要画正式图时直接从对应保存CSV/账本生成，注明样本数、期限、费用和分区；当前文件不生成图表或论文。

### 15.4 一条不超过50词的客户解释

以下为代码已有交易解释，20词：

> This adjustment moves your investments toward the strategic allocation. Its estimated risk-adjusted benefit exceeds trading costs. Investment losses remain possible.

它解释动作与费用依据，并保留亏损可能。不要把“估计价值超过费用”翻译成“保证赚回费用”。

### 15.5 团队还需要自行补充的真实信息

团队成员姓名；实际提交人；实际使用的AI工具与用途；真实人工核查情况。不要杜撰顾问服务响应、实盘表现、独立保管者、额外样本或未做过的文献复核。

题目要求英文主report≤10页A4、字号≥11pt、代码ZIP≤50MB，截止2026-10-20 23:59北京时间；具体要求以本地[题目](../2026_MCM_SafeInvest.pdf)为准。

## 16. 验证与代码位置
""")
    verification = json.loads((research / "verification.json").read_text())
    add(table(["核验", "完成情况"], [
        ["统一测试", "98项通过；生成数据覆盖比例1恢复A、因果前缀、复核后持有、失败保留和回撤下界"],
        ["本轮内层完整执行", f"{verification['inner_paths']:,}条路径"],
        ["本轮外层完整执行", f"{audit['paths']:,}条路径、{audit['annual_records']:,}年度记录"],
        ["独立内层选择重算", f"{audit['selection_checks']}项"],
        ["独立收益/费用/回撤核算", f"最大数值差{audit['maximum_error']:.3g}"],
        ["独立交易门槛检查", f"{audit['gate_checks']:,}次"],
        ["历史原A指标重现", f"{verification['original_reproduction_paths']:,}条；第二套保守无旧保存指标，不冒称已历史重现"],
        ["保护对象", "金融核心、冻结模型、最终锁、两版旧ZIP的hash保持一致"],
        ["最终市场/新最终评价", "未读取/未运行"]]))
    add("""
测试通过保证的是实现和核算一致性，不是金融收益保证。新客户研究没有全120重拟合部署或新最终评价，不能当成新提交金融版本。

| 文件/目录 | 查什么 |
| --- | --- |
| `src/safeinvest/contracts.py` | 输入、动作、上下文、持仓约束 |
| `benchmark/safebench/engine.py` | 费用、信息时点、财富与人工状态 |
| `benchmark/safebench/plan_a.py` | 凸战略配置、部分再平衡、复核 |
| `benchmark/safebench/experiment.py` | 嵌套CV、客户适配、冻结/最终流程 |
| `benchmark/submission_frozen_v1.json` | 最终金融程序、固定参数、模型与客户映射 |
| `scripts/customer_risk.py` | 本轮客户防御混合、内层选择、不可避免回撤下界 |
| `scripts/run_customer_risk_review.py` | 固定有限开发执行，完整保留失败 |
| `scripts/verify_customer_risk.py` | 独立标量账本与选择核算 |
| `scripts/analyze_customer_risk.py` | 新研究比较与区间 |
| `scripts/build_solution_guide.py` | 本说明的保存数据生成器，不读市场CSV |

## 17. 复现命令与提交边界

所有命令在项目根目录执行，环境安装与依赖范围见根README和`requirements.lock.txt`。

```bash
# 日常统一测试，只使用生成数据或开发分区
make test

# 从已有保存数据重建本指南，不重新运行金融策略
.venv/bin/python scripts/build_solution_guide.py \
  --research-run benchmark/outputs/development_customer_risk_review_v1 \
  --out docs/SOLUTION_GUIDE.md

# 若需复现本轮开发，先登记复现记录，再用不存在的新输出目录
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/run_customer_risk_review.py \
  --out benchmark/outputs/development_customer_risk_review_reproduction
.venv/bin/python scripts/analyze_customer_risk.py \
  benchmark/outputs/development_customer_risk_review_reproduction
.venv/bin/python scripts/verify_customer_risk.py \
  benchmark/outputs/development_customer_risk_review_reproduction
```

本轮客户实验依赖工作区保留的训练模型/内层缓存，不声称45.8MB旧ZIP能独立重跑全部新研究。旧提交包是固定快照；本文和本轮新研究代码在新分支。金融默认程序与一次性最终证据仍为原冻结A。

最终包`dist/SafeInvest_final_source.zip`为45,828,045字节、207文件，原独立解压95项测试和字节相同重打包已通过；“95项”属于该包，而“98项”属于本次新分支，不能互换。旧包不包含本轮后来新增的讲解文档和客户研究代码。

当前没有新的独立最终数据。若未来继续改金融模型，应由真正未参与研发的新数据确认；不能删除最终锁、重跑121–180，或把新划分叫新holdout。

## 18. 数值来源与校验收据

本文生成前，从开发51,480条/429组、最终16,380条/273组逐场景指标重算主要汇总；检查主键唯一、每组分别120/60条、分区正确。五客户最终越界次数与冻结适配表交叉核对。本轮研究依赖独立账本核验通过后才生成。

下列文件是本说明的数值来源；哈希用于识别保存证据，不表示统计显著性。
""")
    sources = [dev / "summary.csv", dev / "selected_suitability_risk.csv", final / "summary.csv",
               final / "scenario_metrics.csv", final / "suitability.csv", final / "paired_comparisons.csv",
               final / "severe_case.json", ROOT / "benchmark/submission_frozen_v1.json",
               research / "customer_comparison.csv", research / "decision.json", research / "independent_replay.json",
               research / "unavoidable_drawdown.csv"]
    rows = []
    for path in sources:
        label = str(path.relative_to(ROOT))
        rows.append([f"[{label}](../{label})", hashlib.sha256(path.read_bytes()).hexdigest()])
    add(table(["来源", "SHA256"], rows))
    return "\n\n".join(blocks) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从保存证据生成团队中文解题讲解Markdown，不读取市场数据")
    parser.add_argument("--research-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    destination = args.out.resolve()
    destination.relative_to(ROOT / "docs")
    destination.write_text(build(args.research_run))
    print(destination)
