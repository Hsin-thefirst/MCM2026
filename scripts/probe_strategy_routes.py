"""Development-only planning probes; retain every declared candidate and failure."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "benchmark"), str(ROOT / "scripts")]
from safeinvest.contracts import Action, Decision, MarketObservation, Scenario, SimulationConfig
from safebench.data import folds, load_partition, real_rows
from safebench.engine import simulate
from safebench.metrics import compute_metrics
from safebench.policies import PORTFOLIOS, EQUAL, grid, moments, StaticPolicy
from probe_predictability import causal_features, design, fit_ridge, predict


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def static_target(current, anchor, mode, band):
    if mode == "annual":
        return np.broadcast_to(anchor, current.shape)
    if mode == "hold":
        return current
    gap = np.max(np.abs(current - anchor), axis=-1, keepdims=True)
    if mode == "band_full":
        return np.where(gap > band, anchor, current)
    if mode == "band_edge":
        ratio = np.minimum(1.0, band / np.maximum(gap, 1e-30))
        return anchor + (current - anchor) * ratio
    raise ValueError(mode)


def static_paths(gross, weights, cost, mode, band, horizons):
    """Vectorize exact fee accounting in real currency, preserving fee-time marks."""
    n, years, _ = gross.shape
    holdings = np.zeros((len(weights), n, 3))
    holdings[..., 2] = 100.0
    anchor = weights[:, None, :]
    peak = np.full((len(weights), n), 100.0)
    loss = np.zeros_like(peak)
    trades = np.zeros_like(peak)
    fees = np.zeros_like(peak)
    output = {}
    for t in range(years):
        wealth = holdings.sum(axis=-1)
        current = holdings / wealth[..., None]
        target = np.broadcast_to(anchor, holdings.shape) if t == 0 else static_target(current, anchor, mode, band)
        moved = .5 * np.abs(wealth[..., None] * target - holdings).sum(axis=-1)
        fee = cost * moved
        fees += fee
        if t > 0:
            trades += moved > 1e-8
        holdings = (wealth - fee)[..., None] * target
        loss = np.maximum(loss, 1 - holdings.sum(axis=-1) / peak)
        holdings *= gross[None, :, t, :]
        wealth = holdings.sum(axis=-1)
        peak = np.maximum(peak, wealth)
        loss = np.maximum(loss, 1 - wealth / peak)
        if t + 1 in horizons:
            output[t + 1] = {
                "real_cagr": (wealth / 100) ** (1 / (t + 1)) - 1,
                "real_max_drawdown": loss.copy(),
                "trade_rate": trades / max(1, t),
                "total_real_cost": fees.copy(),
            }
    return output


class BandPolicy:
    def __init__(self, weights, mode, band):
        self.name, self.initial_target, self.mode, self.band = mode, tuple(weights), mode, band

    def decide(self, c):
        current = np.asarray(c.holdings) / sum(c.holdings)
        target = static_target(current, np.asarray(self.initial_target), self.mode, self.band)
        if np.max(np.abs(target - current)) < 1e-12:
            return Decision(Action.KEEP, ("DIAGNOSTIC_HOLD",), "Hold the existing assets.")
        action = Action.REBALANCE if np.allclose(target, c.target_weights, atol=1e-10, rtol=0) else Action.CHANGE_TARGET
        return Decision(action, ("DIAGNOSTIC_BAND",), "Diagnostic band adjustment.", tuple(target))


class ProbeGate:
    requires_benefit_gate = True

    def __init__(self, name, mu, cov, predictor, weights, gamma, mode, config):
        self.name, self.mu, self.cov, self.predictor = name, mu, cov, predictor
        self.weights, self.gamma, self.mode = weights, gamma, mode
        self.buffer = config["buffer"]
        self.config = config
        self.initial_target = PORTFOLIOS["balanced"]
        self.variance = np.einsum("ij,jk,ik->i", weights, cov, weights)

    def decide(self, c):
        current = np.asarray(c.holdings) / sum(c.holdings)
        mu = self.mu
        if self.mode == "ridge_half" and len(c.history) >= 2:
            forecast = predict(causal_features(c.history)[None, :], self.predictor)[0, :3]
            mu = .5 * self.mu + .5 * forecast
        gamma = self.gamma
        if self.mode == "drawdown":
            drawdown = max(0.0, 1 - sum(c.holdings) / c.cumulative_inflation / c.real_peak)
            gamma *= 1 + self.config["drawdown_response"] * min(drawdown / self.config["drawdown_scale"], 1.0)
        utility = 1 + self.weights @ mu - .5 * gamma * self.variance
        costs = c.transaction_cost_rate * .5 * np.abs(self.weights - current).sum(axis=1)
        chosen = int(np.argmax((1 - costs) * utility))
        value = utility[chosen]
        current_value = 1 + mu @ current - .5 * gamma * (current @ self.cov @ current)
        if value <= 0 or not np.isfinite(value):
            return Decision(Action.HUMAN_REVIEW, ("INVALID_ESTIMATE",), "The estimate needs human review.")
        benefit = float((value - current_value) / value)
        cost = float(costs[chosen])
        if benefit <= cost + self.buffer:
            return Decision(Action.KEEP, ("BENEFIT_BELOW_COST",), "The estimated benefit does not cover costs.",
                            estimated_benefit=benefit, estimated_cost=cost)
        target = tuple(self.weights[chosen])
        action = Action.REBALANCE if np.allclose(target, c.target_weights, atol=1e-10, rtol=0) else Action.CHANGE_TARGET
        return Decision(action, ("BENEFIT_EXCEEDS_COST",),
                        "The estimated risk-adjusted benefit exceeds costs. Losses remain possible.", target, benefit, cost)


def summarize(frame, groups):
    return frame.groupby(groups, dropna=False).agg(
        n=("scenario_id", "size"), median_growth=("real_cagr", "median"),
        p10_growth=("real_cagr", lambda x: x.quantile(.1)),
        median_loss=("real_max_drawdown", "median"),
        p90_loss=("real_max_drawdown", lambda x: x.quantile(.9)),
        trade_rate=("trade_rate", "mean"), real_cost=("total_real_cost", "mean"),
    ).reset_index()


def check_static(scenarios, weights, cost, config, generated=False):
    gross = np.asarray([[(1 + np.asarray(o.nominal_returns)) / (1 + o.inflation)
                         for o in s.observations] for s in scenarios])
    horizons = [len(scenarios[0].observations)]
    count, max_error = 0, 0.0
    for mode in config["static_modes"]:
        fast = static_paths(gross, weights, cost, mode, config["band"], horizons)[horizons[0]]
        for i, w in enumerate(weights):
            for j, scenario in enumerate(scenarios):
                policy = BandPolicy(w, mode, config["band"])
                result = simulate(scenario, policy, tuple(w), SimulationConfig(horizons[0], 100, cost))
                scalar = compute_metrics(result)
                for name in fast:
                    error = abs(float(fast[name][i, j]) - scalar[name])
                    max_error = max(max_error, error)
                    np.testing.assert_allclose(fast[name][i, j], scalar[name], atol=1e-10, rtol=1e-10)
                count += 1
    return {"paths": count, "max_absolute_error": max_error, "generated": generated}


def paired(frame, config):
    block = frame[(frame.horizon == 30) & np.isclose(frame.cost_rate, config["costs"][0])]
    rng = np.random.default_rng(config["seed"])
    indices = rng.integers(0, 120, (config["bootstrap"], 120))
    rows = []
    for name in sorted(block.policy.unique()):
        refs = ["balanced_annual", "equal_annual"]
        if name.startswith("gate_") and not name.startswith("gate_constant"):
            refs.append("gate_constant_" + name.rsplit("_", 1)[1])
        a = block[block.policy == name].sort_values("scenario_id")
        for ref in refs:
            if ref == name:
                continue
            b = block[block.policy == ref].sort_values("scenario_id")
            for col, q in [("real_cagr", .5), ("real_cagr", .1), ("real_max_drawdown", .9)]:
                av, bv = a[col].to_numpy(), b[col].to_numpy()
                boot = np.quantile(av[indices], q, axis=1) - np.quantile(bv[indices], q, axis=1)
                rows.append(dict(policy=name, reference=ref, metric=col, quantile=q,
                    difference=float(np.quantile(av, q) - np.quantile(bv, q)),
                    lower95=float(np.quantile(boot, .025)), upper95=float(np.quantile(boot, .975))))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="开发解题路线小实验：完整网格与门控诊断")
    parser.add_argument("--out", type=Path, default=ROOT / "benchmark/outputs/strategy_planning_20261016/routes")
    args = parser.parse_args()
    out = args.out.resolve()
    out.relative_to((ROOT / "benchmark/outputs").resolve())
    out.mkdir(parents=True, exist_ok=False)
    config_path = ROOT / "configs/strategy_planning.json"
    config = json.loads(config_path.read_text())
    scenarios, data_hash = load_partition(ROOT / "data/SafeInvest_Synthetic_Data.csv")
    ids = [s.scenario_id for s in scenarios]
    assert ids == list(range(1, 121))
    groups = folds(ids, config["folds"], config["seed"])
    weights = grid(((0., 1.),) * 3, config["grid_step"])
    gross = np.asarray([[(1 + np.asarray(o.nominal_returns)) / (1 + o.inflation)
                         for o in s.observations] for s in scenarios])
    generated = tuple(Scenario(i, tuple(MarketObservation(i, t, r, .015)
                       for t, r in enumerate([(.2, -.05, .01), (-.3, .08, -.02), (.4, -.04, .02)], 1)))
                      for i in [1, 2])
    validation = [check_static(generated, np.asarray([EQUAL, PORTFOLIOS["balanced"], (0., 0., 1.)]),
                               .005, config, True)]
    validation.append(check_static(scenarios[:3], weights[[0, 50, 100, 150, 230]], .0025, config))
    static_rows, selection_rows, raw = [], [], {}
    for mode in config["static_modes"]:
        for cost in config["costs"]:
            result = static_paths(gross, weights, cost, mode, config["band"], config["horizons"])
            for h, metrics in result.items():
                for metric, values in metrics.items():
                    raw[f"{mode}_{cost}_{h}_{metric}"] = values
            for fold, valid_ids in enumerate(groups):
                vi = np.asarray([sid - 1 for sid in valid_ids])
                ti = np.asarray([sid - 1 for sid in ids if sid not in valid_ids])
                for h in config["horizons"]:
                    # Freeze weights selected at the official cost for the doubled-cost sensitivity.
                    official = result[h] if cost == config["costs"][0] else {
                        k: raw[f"{mode}_{config['costs'][0]}_{h}_{k}"] for k in result[h]}
                    risk = np.quantile(official["real_max_drawdown"][:, ti], .9, axis=1)
                    score = np.log1p(official["real_cagr"][:, ti]).mean(axis=1)
                    for budget in config["risk_budgets"]:
                        feasible = risk <= budget
                        chosen = int(np.argmax(np.where(feasible, score, -np.inf))) if feasible.any() else None
                        selection_rows.append(dict(mode=mode, cost_rate=cost, horizon=h, fold=fold,
                            budget=budget, feasible=bool(feasible.any()), weight_index=chosen,
                            weights=weights[chosen].tolist() if chosen is not None else None,
                            train_p90=float(risk[chosen]) if chosen is not None else None))
                        if chosen is None:
                            continue
                        for j in vi:
                            static_rows.append(dict(policy=f"{mode}_risk{budget:g}", mode=mode, budget=budget,
                                horizon=h, cost_rate=cost, fold=fold, scenario_id=ids[j],
                                **{k: float(v[chosen, j]) for k, v in result[h].items()}))
            print("静态网格完成", mode, cost, flush=True)
    static = pd.DataFrame(static_rows)
    static.to_csv(out / "static_oof_metrics.csv", index=False)
    summarize(static, ["policy", "horizon", "cost_rate"]).to_csv(out / "static_summary.csv", index=False)
    dump(out / "static_selection.json", selection_rows)
    np.savez_compressed(out / "static_grid_paths.npz", weights=weights, **raw)

    gate_rows, ledger, models = [], [], []
    for fold, valid_ids in enumerate(groups):
        train = tuple(s for s in scenarios if s.scenario_id not in valid_ids)
        valid = tuple(s for s in scenarios if s.scenario_id in valid_ids)
        mu, cov = moments(real_rows(train))
        x, y, _ = design(train)
        predictor = fit_ridge(x, y)
        models.append(dict(fold=fold, train_ids=[s.scenario_id for s in train], validation_ids=valid_ids,
            mu=mu.tolist(), cov=cov.tolist(), predictor={k: v.tolist() for k, v in predictor.items()}))
        for cost in config["costs"]:
            candidates = [(f"gate_{mode}_{gamma:g}", gamma, mode) for mode in config["gate_modes"] for gamma in config["gammas"]]
            candidates += [(name, None, None) for name in ["equal_annual", "balanced_annual", "balanced_hold"]]
            for name, gamma, mode in candidates:
                for scenario in valid:
                    if gamma is None:
                        anchor = EQUAL if name == "equal_annual" else PORTFOLIOS["balanced"]
                        policy = StaticPolicy(name, anchor, "hold" if name.endswith("hold") else "annual")
                    else:
                        policy = ProbeGate(name, mu, cov, predictor, weights, gamma, mode, config)
                    full = simulate(scenario, policy, policy.initial_target, SimulationConfig(30, 100, cost))
                    for h in config["horizons"]:
                        result = replace(full, config=replace(full.config, horizon=h), records=full.records[:h])
                        gate_rows.append(dict(compute_metrics(result), fold=fold))
                    for record in full.records:
                        ledger.append(dict(scenario_id=scenario.scenario_id, policy=name, cost_rate=cost,
                            period=record.period, action=record.action.value, real_wealth=record.real_wealth,
                            moved_amount=record.moved_amount, transaction_cost=record.transaction_cost,
                            estimated_benefit=record.estimated_benefit, estimated_cost=record.estimated_cost))
        print("门控外折完成", fold + 1, flush=True)
    gated = pd.DataFrame(gate_rows)
    gated.to_csv(out / "gate_oof_metrics.csv", index=False)
    summarize(gated, ["policy", "horizon", "cost_rate"]).to_csv(out / "gate_summary.csv", index=False)
    summarize(gated, ["policy", "horizon", "cost_rate", "fold"]).to_csv(out / "gate_fold_summary.csv", index=False)
    paired(gated, config).to_csv(out / "gate_paired_pointwise.csv", index=False)
    pd.DataFrame(ledger).to_csv(out / "gate_decisions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    dump(out / "models.json", models)
    dump(out / "validation.json", validation)
    paths = [Path(__file__), ROOT / "scripts/probe_predictability.py", config_path,
             ROOT / "benchmark/safebench/engine.py", ROOT / "benchmark/safebench/data.py",
             ROOT / "benchmark/safebench/metrics.py", ROOT / "benchmark/safebench/policies.py",
             ROOT / "src/safeinvest/contracts.py"]
    dump(out / "manifest.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
        phase="development", holdout_used=False, development_sha256=data_hash,
        config=config, code_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        caveats=["探索性方案筛选，未经完整晋级。", "固定年度与带宽交易为诊断参照，不声称满足收益门槛。",
                 "门控保持原Balanced初始配置，动作网格使用官方单纯形；9候选全部保留。",
                 "bootstrap仅逐点且条件于已拟合OOF模型，不校正全部研究迭代。",
                 "固定网格的训练选择与外折评价隔离；费用敏感性不重新选权重。"],
        static_rows=len(static), gate_rows=len(gated), gate_decisions=len(ledger)))
    summary = summarize(gated, ["policy", "horizon", "cost_rate"])
    print(summary[(summary.horizon == 30) & np.isclose(summary.cost_rate, .0025)].to_string(index=False))


if __name__ == "__main__":
    main()
