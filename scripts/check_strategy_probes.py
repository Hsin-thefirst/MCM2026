"""Inspect saved planning probes without selecting new candidates or rereading markets."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_strategy_routes import summarize


def main():
    parser = argparse.ArgumentParser(description="核验已保存的解题诊断与严格同权重对照")
    parser.add_argument("--run", type=Path, default=ROOT / "benchmark/outputs/strategy_planning_20261016/routes")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    out = (args.out or run / "checks").resolve()
    out.relative_to((ROOT / "benchmark/outputs").resolve())
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((run / "manifest.json").read_text())
    for name, digest in manifest["code_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    config = manifest["config"]
    models = json.loads((run / "models.json").read_text())
    selections = json.loads((run / "static_selection.json").read_text())
    with np.load(run / "static_grid_paths.npz") as saved:
        grid = {key: saved[key] for key in saved.files}
    rows = []
    for selection in selections:
        if selection["mode"] != "band_edge" or not selection["feasible"]:
            continue
        fold, h, cost = selection["fold"], selection["horizon"], selection["cost_rate"]
        index = selection["weight_index"]
        for mode in config["static_modes"]:
            for sid in models[fold]["validation_ids"]:
                rows.append(dict(policy=mode, budget=selection["budget"], horizon=h, cost_rate=cost,
                    scenario_id=sid, fold=fold,
                    **{k: float(grid[f"{mode}_{cost}_{h}_{k}"][index, sid - 1])
                       for k in ["real_cagr", "real_max_drawdown", "trade_rate", "total_real_cost"]}))
    matched = pd.DataFrame(rows)
    matched.to_csv(out / "matched_static_metrics.csv", index=False)
    summary = summarize(matched, ["policy", "budget", "horizon", "cost_rate"])
    summary.to_csv(out / "matched_static_summary.csv", index=False)
    comparisons = []
    rng = np.random.default_rng(config["seed"])
    indices = rng.integers(0, 120, (config["bootstrap"], 120))
    for budget in config["risk_budgets"]:
        block = matched[(matched.horizon == 30) & (matched.cost_rate == .0025) & (matched.budget == budget)]
        candidate = block[block.policy == "band_edge"].sort_values("scenario_id")
        for reference in ["annual", "band_full", "hold"]:
            control = block[block.policy == reference].sort_values("scenario_id")
            for metric, quantile in [("real_cagr", .5), ("real_cagr", .1), ("real_max_drawdown", .9)]:
                a, b = candidate[metric].to_numpy(), control[metric].to_numpy()
                distribution = np.quantile(a[indices], quantile, axis=1) - np.quantile(b[indices], quantile, axis=1)
                comparisons.append(dict(budget=budget, reference=reference, metric=metric, quantile=quantile,
                    difference=float(np.quantile(a, quantile) - np.quantile(b, quantile)),
                    lower95=float(np.quantile(distribution, .025)), upper95=float(np.quantile(distribution, .975))))
    pd.DataFrame(comparisons).to_csv(out / "matched_static_pointwise.csv", index=False)
    gated = pd.read_csv(run / "gate_oof_metrics.csv")
    assert len(gated) == 12 * 120 * 2 * 5
    assert not gated.duplicated(["policy", "scenario_id", "cost_rate", "horizon"]).any()
    for _, block in gated.groupby(["policy", "cost_rate", "horizon"]):
        assert sorted(block.scenario_id) == list(range(1, 121))
    for model in models:
        assert not set(model["train_ids"]) & set(model["validation_ids"])
        assert sorted(model["train_ids"] + model["validation_ids"]) == list(range(1, 121))
    decisions = pd.read_csv(run / "gate_decisions.csv.gz")
    assert len(decisions) == 12 * 120 * 2 * 30
    trades = decisions[(decisions.period > 1) & decisions.policy.str.startswith("gate_") &
                       decisions.action.isin(["rebalance", "change_target"])]
    assert (trades.estimated_benefit > trades.estimated_cost + config["buffer"]).all()
    assert np.isfinite(gated.select_dtypes(include="number").to_numpy()).all()
    comparison = []
    base = pd.read_csv(ROOT / "benchmark/outputs/development_consolidated/scenario_metrics.csv")
    keys = ["policy", "scenario_id", "horizon", "cost_rate"]
    check = gated[gated.policy.isin(["equal_annual", "balanced_annual", "balanced_hold"])].merge(
        base, on=keys, suffixes=("_new", "_old"), validate="one_to_one")
    assert len(check) == 3 * 120 * 2 * 5 - 3 * 120 * 4
    # Historical sensitivity ran H30 only; compare only the available intersections.
    for name in ["real_cagr", "real_max_drawdown", "trade_rate", "total_real_cost"]:
        error = float((check[name + "_new"] - check[name + "_old"]).abs().max())
        assert error < 1e-10
        comparison.append({"metric": name, "max_error": error})
    prediction = pd.read_csv(run.parent / "predictability/oof_predictions.csv")
    assert len(prediction) == 3360 and not prediction.duplicated(["scenario_id", "period"]).any()
    audit = dict(gate_paths=len(gated), gate_decisions=len(decisions),
        benefit_gate_checks=len(trades), historical_reference_rows=len(check),
        historical_reference_errors=comparison, predictor_rows=len(prediction),
        matched_anchor="band_edge selected training-fold weight at each risk budget and horizon",
        notes="复用已保存完整网格，不新调参；同权重对照不具有相同事后风险。",
        all_passed=True, checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(summary[(summary.horizon == 30) & (summary.budget == .35) & (summary.cost_rate == .0025)].to_string(index=False))


if __name__ == "__main__":
    main()
