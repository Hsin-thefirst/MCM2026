"""Read saved customer research outputs; never select a replacement from final results."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "benchmark")]
import numpy as np
import pandas as pd
from safebench.experiment import dump


def analyze(run):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    settings = manifest["settings"]
    selections = json.loads((run / "selections.json").read_text())
    profiles = json.loads((ROOT / "configs/profiles.json").read_text())
    comparisons, intervals = [], []
    rng = np.random.default_rng(settings["analysis_seed"])
    gates = []
    for seed in settings["seeds"]:
        frame = pd.read_csv(run / str(seed) / "scenario_metrics.csv")
        assert not frame.duplicated(["policy", "horizon", "cost_rate", "scenario_id"]).any()
        for profile in profiles:
            pid = profile["profile_id"]
            normal = frame[(frame.profile_id == pid) & (frame.cost_rate == settings["costs"][0])]
            indices = rng.integers(0, 120, (settings["bootstrap"], 120))
            for name in sorted(normal.policy.unique()):
                g = normal[normal.policy == name]
                assert len(g) == 120
                feasible = all(r["feasible"] for r in selections if r["seed"] == seed and r["profile_id"] == pid)
                comparisons.append(dict(seed=seed, profile_id=pid, policy=name, horizon=profile["horizon"],
                    median_growth=float(g.real_cagr.median()), p10_growth=float(g.real_cagr.quantile(.1)),
                    p90_loss=float(g.real_max_drawdown.quantile(.9)), median_loss=float(g.real_max_drawdown.median()),
                    breaches=int((g.real_max_drawdown > profile["loss_tolerance"]).sum()),
                    terminal_loss=int(g.terminal_loss.sum()), mean_real_cost=float(g.total_real_cost.mean()),
                    trade_rate=float(g.trade_rate.mean()), review_rate=float(g.review.mean()),
                    selection_feasible=feasible if name.startswith("customer_") else None,
                    automated_result=bool(pid in ["B", "C", "E"] and
                                          (not name.startswith("customer_") or feasible))))
            for name in ["customer_budget", "customer_early_review"]:
                checks = {}
                for reference in ["original", "same_anchor_annual"]:
                    a = normal[normal.policy == name].sort_values("scenario_id")
                    b = normal[normal.policy == reference].sort_values("scenario_id")
                    assert a.scenario_id.tolist() == b.scenario_id.tolist()
                    values = {}
                    for col, quantile, label in [("real_cagr", .5, "growth"), ("real_cagr", .1, "p10"),
                                                  ("real_max_drawdown", .9, "p90")]:
                        x, y = a[col].to_numpy(), b[col].to_numpy()
                        boot = np.quantile(x[indices], quantile, axis=1) - np.quantile(y[indices], quantile, axis=1)
                        row = dict(seed=seed, profile_id=pid, policy=name, reference=reference, metric=label,
                                   difference=float(np.quantile(x, quantile) - np.quantile(y, quantile)),
                                   lower95=float(np.quantile(boot, .025)), upper95=float(np.quantile(boot, .975)),
                                   family_lower=float(np.quantile(boot, .0125)), family_upper=float(np.quantile(boot, .9875)))
                        intervals.append(row)
                        values[label] = row
                    stress = frame[(frame.profile_id == pid) & (frame.cost_rate == settings["costs"][1])]
                    x, y = stress[stress.policy == name], stress[stress.policy == reference]
                    doubled_growth = float(x.real_cagr.median() - y.real_cagr.median())
                    doubled_p90 = float(x.real_max_drawdown.quantile(.9) - y.real_max_drawdown.quantile(.9))
                    risk_folds = sum(a[a.fold == k].real_max_drawdown.quantile(.9) <
                                     b[b.fold == k].real_max_drawdown.quantile(.9) for k in range(5))
                    checks[reference] = bool(values["p90"]["difference"] <= -.03 and
                        values["p90"]["family_upper"] < 0 and values["growth"]["family_lower"] >= -.005 and
                        values["p10"]["family_lower"] >= -.002 and risk_folds >= 4 and
                        doubled_p90 < 0 and doubled_growth >= -.005 and a.review.mean() <= .1)
                calibrated = all(r["feasible"] for r in selections if r["seed"] == seed and r["profile_id"] == pid)
                gates.append(dict(seed=seed, profile_id=pid, policy=name, all_folds_feasible=calibrated,
                                  risk_route_vs_original=checks["original"],
                                  risk_route_vs_same_anchor=checks["same_anchor_annual"]))
    pd.DataFrame(comparisons).to_csv(run / "customer_comparison.csv", index=False)
    pd.DataFrame(intervals).to_csv(run / "paired_intervals.csv", index=False)
    counts = [{"seed": seed, "profile_id": p["profile_id"],
               "feasible_folds": sum(r["feasible"] for r in selections if r["seed"] == seed and r["profile_id"] == p["profile_id"])}
              for seed in settings["seeds"] for p in profiles]
    dump(run / "decision.json", dict(replace_frozen_submission=False, independent_confirmation=False,
         final_evaluation_rerun=False, selected_policy="plan_a_balanced", gates=gates,
         feasible_folds=counts, stopping_rule="五个固定比例、一次主规则和提前复核诊断完成后停止；不追加参数。",
         interpretation="风险优先研发，未通过项及不可行路径完整保存；新策略没有独立最终成绩。"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从保存数据比较客户风险、收益代价与提前复核")
    parser.add_argument("run", type=Path)
    analyze(parser.parse_args().run)
