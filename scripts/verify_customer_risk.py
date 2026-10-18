"""Independent scalar accounting and selection checks for saved customer research."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "benchmark")]
import numpy as np
import pandas as pd
from scipy.stats import beta
from safebench.data import load_partition
from safebench.experiment import dump


def verify(run):
    run = Path(run)
    if (run / "independent_replay.json").exists():
        raise FileExistsError("Verification already exists")
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["phase"] == "development" and not manifest["holdout_used"]
    settings = manifest["settings"]
    scenarios, digest = load_partition(ROOT / "data/SafeInvest_Synthetic_Data.csv")
    assert digest == manifest["development_sha256"]
    market = {s.scenario_id: s for s in scenarios}
    profiles = {p["profile_id"]: p for p in json.loads((ROOT / "configs/profiles.json").read_text())}
    selections = json.loads((run / "selections.json").read_text())
    inner = pd.read_csv(run / "inner_metrics.csv.gz")
    error = 0.
    count = paths = gates = 0

    def check(a, b):
        nonlocal error
        if isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                check(x, y)
        else:
            error = max(error, abs(float(a) - float(b)))
            assert math.isclose(float(a), float(b), rel_tol=2e-12, abs_tol=2e-9), (a, b)

    for chosen in selections:
        block = inner[(inner.seed == chosen["seed"]) & (inner.fold == chosen["fold"]) &
                      (inner.profile_id == chosen["profile_id"])]
        assert set(block.scenario_id) == set(chosen["train_ids"])
        assert not set(chosen["train_ids"]) & set(chosen["validation_ids"])
        assert len(block) == len(chosen["train_ids"]) * 5 * 2
        profile = profiles[chosen["profile_id"]]
        candidates = []
        for fraction in settings["mix_fractions"]:
            violations = []
            for cost in settings["costs"]:
                g = block[(block.fraction == fraction) & (block.cost_rate == cost)]
                original = block[(block.fraction == 1.) & (block.cost_rate == cost)]
                target = max(0., min(.8 * profile["loss_tolerance"],
                                    float(np.quantile(original.real_max_drawdown, .9)) - .03))
                k = int(sum(g.real_max_drawdown > profile["loss_tolerance"]))
                upper = 1. if k == len(g) else float(beta.ppf(.9999, k + 1, len(g) - k))
                violations.extend([max(0., float(np.quantile(g.real_max_drawdown, .9)) - target),
                                   max(0., upper - .2), 0. if profile["liquidity"] != "high" or
                                   g.strategic_bill_min.min() >= .4 - 1e-8 else 1.])
            growth = float(np.median(block[(block.fraction == fraction) & (block.cost_rate == .0025)].real_cagr))
            candidates.append(dict(fraction=fraction, violation=sum(violations), median_growth=growth))
        feasible = [c for c in candidates if c["violation"] <= 1e-10]
        if feasible:
            selected = max(feasible, key=lambda c: (c["median_growth"], c["fraction"]))
        else:
            selected = min(candidates, key=lambda c: (c["violation"], -c["median_growth"], c["fraction"]))
        assert chosen["feasible"] == bool(feasible)
        assert chosen["fraction"] == selected["fraction"]
        check(chosen["violation"], selected["violation"])
    bounds = pd.read_csv(run / "unavoidable_drawdown.csv").set_index(["profile_id", "scenario_id"])
    for seed in settings["seeds"]:
        frame = pd.read_csv(run / str(seed) / "scenario_metrics.csv")
        metrics = {(r.policy, r.profile_id, r.scenario_id, r.cost_rate): r for r in frame.itertuples()}
        assert len(metrics) == len(frame) == 7200
        expected = {(p, pid, sid, cost) for p in ["original", "customer_budget", "customer_early_review",
                     "same_anchor_annual", "bills_hold", "equal_annual"] for pid in profiles
                    for sid in range(1, 121) for cost in settings["costs"]}
        assert set(metrics) == expected
        active, seen = None, set()
        with gzip.open(run / str(seed) / "decisions.jsonl.gz", "rt") as handle:
            for line in handle:
                r = json.loads(line)
                key = (r["policy"], r["profile_id"], r["scenario_id"], r["cost_rate"])
                if key != active:
                    if active is not None:
                        assert previous == profiles[active[1]]["horizon"]
                    assert key not in seen and key in metrics and r["period"] == 1
                    seen.add(key)
                    active = key
                    h = [0., 0., 100.]
                    inflation = 1.
                    peak = 100.
                    mdd = fees = 0.
                    previous = trades = 0
                    pending = False
                    wealths = [100.]
                t = r["period"]
                assert t == previous + 1
                previous = t
                check(r["pre_trade_holdings"], h)
                if pending:
                    assert r["action"] == "keep" and r["review_pending"]
                assert bool(r["review_pending"]) == (pending or r["action"] == "human_review")
                pending = r["review_pending"]
                wealth = sum(h)
                if t == 1 or r["action"] in ["rebalance", "change_target"]:
                    weights = r["target_weights"]
                    check(sum(weights), 1.)
                    assert min(weights) >= 0
                    moved = sum(abs(wealth * w - v) for w, v in zip(weights, h)) / 2
                    fee = moved * r["cost_rate"]
                    post = [(wealth - fee) * w for w in weights]
                else:
                    moved = fee = 0.
                    post = h[:]
                if t > 1 and moved > 1e-8:
                    trades += 1
                    if r["requires_benefit_gate"]:
                        check(r["estimated_cost"], fee / wealth)
                        assert r["estimated_benefit"] > fee / wealth + r["required_buffer"]
                        gates += 1
                check([r["moved_amount"], r["transaction_cost"]], [moved, fee])
                check(r["post_trade_holdings"], post)
                fees += fee / inflation
                mark = sum(post) / inflation
                mdd = max(mdd, 1 - mark / peak)
                peak = max(peak, mark)
                obs = market[r["scenario_id"]].observations[t - 1]
                h = [v * (1 + g) for v, g in zip(post, obs.nominal_returns)]
                inflation *= 1 + obs.inflation
                mark = sum(h) / inflation
                mdd = max(mdd, 1 - mark / peak)
                peak = max(peak, mark)
                check(r["end_holdings"], h)
                check([r["real_wealth"], r["cumulative_inflation"]], [mark, inflation])
                wealths.append(mark)
                count += 1
                if t == r["horizon"]:
                    metric = metrics[key]
                    returns = [b / a - 1 for a, b in zip(wealths[:-1], wealths[1:])]
                    check([metric.real_cagr, metric.real_max_drawdown, metric.total_real_cost, metric.trade_rate,
                           metric.real_return_volatility, metric.terminal_real_wealth, metric.worst_real_return],
                          [(mark / 100) ** (1 / t) - 1, mdd, fees, trades / (t - 1),
                           float(np.std(returns, ddof=1)), mark, min(returns)])
                    assert bounds.loc[(r["profile_id"], r["scenario_id"]), "lower_bound"] <= mdd + 1e-12
                    paths += 1
        assert seen == expected
    assert paths == 14400 and len(selections) == 50
    for name, expected_sha in {**manifest["sources"], **manifest["protected_hashes"]}.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected_sha
    dump(run / "independent_replay.json", dict(passed=True, paths=paths, annual_records=count,
         gate_checks=gates, selection_checks=len(selections), maximum_error=error,
         sources_and_protected_hashes_passed=True, final_market_data_read=False,
         verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(json.loads((run / "independent_replay.json").read_text()), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="独立标量核算客户风险开发账本和全部选择")
    parser.add_argument("run", type=Path)
    verify(parser.parse_args().run)
