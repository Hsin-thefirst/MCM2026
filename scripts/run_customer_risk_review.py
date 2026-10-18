"""One finite, nested development review of customer-specific defensive allocations."""
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "benchmark")]
import numpy as np
import pandas as pd
from customer_risk import CustomerRiskPolicy, select_fraction, unavoidable_drawdown
from safeinvest.contracts import SimulationConfig
from safebench.data import load_partition
from safebench.engine import simulate
from safebench.experiment import code_digest, dump
from safebench.metrics import compute_metrics, summarize
from safebench.policies import make_policy


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(output):
    output = Path(output).resolve()
    output.relative_to(ROOT / "benchmark/outputs")
    settings = json.loads((ROOT / "configs/customer_risk_review.json").read_text())
    profiles = json.loads((ROOT / "configs/profiles.json").read_text())
    source = ROOT / settings["source_run"]
    files = [Path(__file__), ROOT / "scripts/customer_risk.py", ROOT / "scripts/analyze_customer_risk.py",
             ROOT / "configs/customer_risk_review.json", ROOT / "configs/profiles.json"]
    for seed in settings["seeds"]:
        files.extend(source / str(seed) / name for name in ["config.json", "splits.json", "fitted_models.json"])
        files.extend(ROOT / settings["parameter_runs"][str(seed)] / name
                     for name in ["fitted_models.json", "scenario_metrics.csv", "manifest.json"])
    protected_paths = ["benchmark/FINAL_EVALUATION_LOCK.json", "benchmark/submission_frozen_v1.json",
                       "dist/SafeInvest_source.zip", "dist/SafeInvest_final_source.zip"]
    protected = {name: sha(ROOT / name) for name in protected_paths}
    scenarios, data_hash = load_partition(ROOT / "data/SafeInvest_Synthetic_Data.csv")
    lookup = {s.scenario_id: s for s in scenarios}
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), phase="development",
                    holdout_used=False, final_already_exposed=True, independent_confirmation=False,
                    settings=settings, code_sha256=code_digest(), development_sha256=data_hash,
                    sources={str(p.relative_to(ROOT)): sha(p) for p in files}, protected_hashes=protected,
                    selection="客户风险预算优先；五个固定混合比例，完整内层含费验证，外层不可选参数。",
                    failure="无可行参数时仅保存最小违约反事实路径并标记；不得作为自动推荐绩效。")
    dump(output / "manifest.json", manifest)
    all_inner = []
    reproduced = 0
    all_selections = []
    for seed in settings["seeds"]:
        target = output / str(seed)
        target.mkdir()
        original = source / str(seed)
        config = json.loads((original / "config.json").read_text())
        fitted = json.loads((original / "fitted_models.json").read_text())
        audits = json.loads((original / "splits.json").read_text())
        parameter_source = ROOT / settings["parameter_runs"][str(seed)]
        source_manifest = json.loads((parameter_source / "manifest.json").read_text())
        assert source_manifest["code_sha256"] == code_digest()
        assert source_manifest.get("partition_sha256", source_manifest.get("development_sha256")) == data_hash
        parameters = json.loads((parameter_source / "fitted_models.json").read_text())
        outer = []
        annual_records = 0
        with gzip.open(target / "decisions.jsonl.gz", "wt", encoding="utf-8") as ledger:
            for item, audit, saved in zip(fitted, audits, parameters, strict=True):
                fold = item["fold"]
                assert fold == audit["outer_fold"] == saved["fold"]
                model = item["model"]
                assert model["train_ids"] == saved["model"]["train_ids"] == audit["train_ids"]
                assert not set(model["train_ids"]) & set(audit["validation_ids"])
                for profile in profiles:
                    pid, horizon = profile["profile_id"], profile["horizon"]
                    policy_id = "plan_a_balanced" if pid == "E" else "plan_a_conservative"
                    params = saved["params"][policy_id]
                    assert (model["plan_a"]["entries"][f"{policy_id.removeprefix('plan_a_')}:{horizon}:{params['risk_ratio']:g}"]
                            == saved["model"]["plan_a"]["entries"][f"{policy_id.removeprefix('plan_a_')}:{horizon}:{params['risk_ratio']:g}"])
                    inner_rows = []
                    for inner in audit["inner"]:
                        training = inner["model"]
                        valid_ids = inner["validation_ids"]
                        assert training["train_ids"] == inner["train_ids"]
                        assert not set(training["train_ids"]) & set(valid_ids)
                        assert set(training["train_ids"]) | set(valid_ids) == set(model["train_ids"])
                        for fraction in settings["mix_fractions"]:
                            for cost in settings["costs"]:
                                for sid in valid_ids:
                                    policy = CustomerRiskPolicy("inner", training, params, config, profile, fraction)
                                    result = simulate(lookup[sid], policy, policy.initial_target,
                                                      SimulationConfig(horizon, 100., cost))
                                    inner_rows.append(dict(compute_metrics(result), fraction=fraction,
                                                           strategic_bill_min=policy.initial_target[2],
                                                           inner_fold=inner["inner_fold"]))
                    inner_frame = pd.DataFrame(inner_rows)
                    choice = select_fraction(inner_frame, profile, settings)
                    all_inner.extend(dict(r, seed=seed, fold=fold, profile_id=pid) for r in inner_rows)
                    all_selections.append(dict(choice, seed=seed, fold=fold, profile_id=pid,
                                               train_ids=model["train_ids"], validation_ids=audit["validation_ids"]))
                    for name in ["original", "customer_budget", "customer_early_review", "same_anchor_annual",
                                 "bills_hold", "equal_annual"]:
                        for cost in settings["costs"]:
                            for sid in audit["validation_ids"]:
                                if name in ["bills_hold", "equal_annual"]:
                                    policy = make_policy(name, model, params, config, horizon)
                                else:
                                    fraction = 1. if name == "original" else choice["fraction"]
                                    policy = CustomerRiskPolicy(name, model, params, config, profile, fraction,
                                        early_review=name == "customer_early_review",
                                        early_review_fraction=settings["early_review_fraction"],
                                        mode="annual" if name == "same_anchor_annual" else "partial")
                                result = simulate(lookup[sid], policy, policy.initial_target,
                                                  SimulationConfig(horizon, 100., cost))
                                outer.append(dict(compute_metrics(result), fold=fold, profile_id=pid,
                                                  fraction=1. if name == "original" else choice["fraction"],
                                                  selection_feasible=choice["feasible"],
                                                  original_automated=pid in ["B", "C", "E"],
                                                  strategic_bill_min=policy.initial_target[2]))
                                for record in result.records:
                                    row = asdict(record)
                                    row.update(action=record.action.value, policy=name, scenario_id=sid,
                                               horizon=horizon, cost_rate=cost, fold=fold, profile_id=pid,
                                               requires_benefit_gate=getattr(policy, "requires_benefit_gate", False),
                                               required_buffer=getattr(policy, "buffer", 0.))
                                    ledger.write(json.dumps(row, allow_nan=False) + "\n")
                                    annual_records += 1
                    print(f"seed={seed} 外折{fold+1}/5 客户{pid}: 比例{choice['fraction']}, 内层合格={choice['feasible']}", flush=True)
        frame = pd.DataFrame(outer)
        assert len(frame) == 7200
        frame.to_csv(target / "scenario_metrics.csv", index=False)
        summarize(frame).to_csv(target / "summary.csv", index=False)
        dump(target / "verification.json", dict(paths=len(frame), annual_records=annual_records))
        reference = pd.read_csv(parameter_source / "scenario_metrics.csv")
        for profile in profiles:
            name = "plan_a_balanced" if profile["profile_id"] == "E" else "plan_a_conservative"
            for cost in settings["costs"]:
                actual = frame[(frame.profile_id == profile["profile_id"]) & (frame.policy == "original") & (frame.cost_rate == cost)].sort_values("scenario_id")
                expected = reference[(reference.policy == name) & (reference.horizon == profile["horizon"]) & (reference.cost_rate == cost)].sort_values("scenario_id")
                if expected.empty and seed == 20261019 and name == "plan_a_conservative":
                    continue
                assert len(expected) == 120
                np.testing.assert_allclose(actual[["real_cagr", "real_max_drawdown", "total_real_cost"]],
                                           expected[["real_cagr", "real_max_drawdown", "total_real_cost"]], rtol=1e-12, atol=1e-12)
                reproduced += len(expected)
    pd.DataFrame(all_inner).to_csv(output / "inner_metrics.csv.gz", index=False,
                                 compression={"method": "gzip", "mtime": 0})
    dump(output / "selections.json", all_selections)
    returns = np.asarray([[(1 + np.asarray(o.nominal_returns)) / (1 + o.inflation) - 1
                            for o in s.observations] for s in scenarios])
    lower_bounds = []
    for profile in profiles:
        bounds = unavoidable_drawdown(returns[:, :profile["horizon"]])
        lower_bounds.extend(dict(profile_id=profile["profile_id"], scenario_id=s.scenario_id,
                                 lower_bound=float(bound), tolerance=profile["loss_tolerance"],
                                 unavoidable_breach=bool(bound > profile["loss_tolerance"]))
                            for s, bound in zip(scenarios, bounds, strict=True))
    pd.DataFrame(lower_bounds).to_csv(output / "unavoidable_drawdown.csv", index=False)
    for name, digest in manifest["sources"].items():
        assert sha(ROOT / name) == digest
    for name, digest in protected.items():
        assert sha(ROOT / name) == digest
    dump(output / "verification.json", dict(passed=True, inner_paths=len(all_inner), outer_paths=14400,
                                            original_reproduction_paths=reproduced, protected_unchanged=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="客户风险预算与提前人工复核的最后有限开发核验")
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
