"""Fixed development-only, scenario-fold ridge predictability diagnostic."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "benchmark")]

import numpy as np
import pandas as pd

from safeinvest.contracts import ASSETS, MarketObservation, Scenario
from safebench.data import folds, load_partition


from safebench.signals import (SEED, RIDGE, FEATURE_NAMES, TARGET_NAMES, real_returns,
                                causal_features, fit_ridge, predict)

def design(scenarios: tuple[Scenario, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, targets, keys = [], [], []
    for scenario in scenarios:
        for index in range(2, len(scenario.observations)):
            observation = scenario.observations[index]
            row = real_returns(observation)
            features.append(causal_features(scenario.observations[:index]))
            targets.append(np.concatenate([row, row ** 2]))
            keys.append([scenario.scenario_id, observation.period])
    return np.asarray(features), np.asarray(targets), np.asarray(keys, dtype=int)


def synthetic_self_check() -> dict:
    observations = tuple(
        MarketObservation(
            1, period, (0.015 * period, -0.003 * period, 0.002 * period), 0.001 * period
        ) for period in range(1, 9)
    )
    original = Scenario(1, observations)
    full_features, _, full_keys = design((original,))
    checks = 0
    for period in range(3, 9):
        index = period - 1
        prefix = Scenario(1, observations[:period])
        prefix_features, _, _ = design((prefix,))
        np.testing.assert_array_equal(prefix_features[-1], full_features[index - 2])
        mutated = Scenario(1, observations[:index] + tuple(
            replace(o, nominal_returns=(0.7, -0.3, 0.1), inflation=0.12)
            for o in observations[index:]
        ))
        mutated_features, _, mutated_keys = design((mutated,))
        np.testing.assert_array_equal(mutated_keys, full_keys)
        np.testing.assert_array_equal(mutated_features[index - 2], full_features[index - 2])
        checks += 2
    assert full_features.shape == (6, 12)
    assert np.array_equal(full_keys[:, 1], np.arange(3, 9))
    return {"synthetic_prefix_checks": checks, "passed": True}


def metric_row(scope: str, fold: int | None, target: str, actual: np.ndarray,
               estimated: np.ndarray, constant: np.ndarray) -> dict:
    squared_error = (actual - estimated) ** 2
    constant_error = (actual - constant) ** 2
    denominator = float(constant_error.sum())
    return {
        "scope": scope, "fold": fold, "target": target, "n": len(actual),
        "model_mse": float(squared_error.mean()),
        "constant_mse": float(constant_error.mean()),
        "model_sse": float(squared_error.sum()),
        "constant_sse": denominator,
        "oos_r2": 1.0 - float(squared_error.sum()) / denominator if denominator else None,
    }


def json_write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="固定五折 Ridge 开发可预测性诊断")
    parser.add_argument("--out", type=Path, default=ROOT / "benchmark/outputs/strategy_planning_20261016/predictability")
    parser.add_argument("--self-check-only", action="store_true", help="只检查合成数据因果前缀")
    args = parser.parse_args()
    checks = synthetic_self_check()
    if args.self_check_only:
        print(json.dumps(checks, ensure_ascii=False))
        return

    output = args.out.resolve()
    output.relative_to((ROOT / "benchmark/outputs").resolve())
    output.mkdir(parents=True, exist_ok=False)
    scenarios, partition_hash = load_partition(ROOT / "data/SafeInvest_Synthetic_Data.csv", phase="development")
    assert [s.scenario_id for s in scenarios] == list(range(1, 121))
    all_ids = {s.scenario_id for s in scenarios}
    split_ids = folds(all_ids, 5, SEED)
    metric_rows, prediction_rows, model_rows, coefficient_rows = [], [], [], []
    for fold_index, validation_ids in enumerate(split_ids, 1):
        validation_set = set(validation_ids)
        training_ids = sorted(all_ids - validation_set)
        assert not set(training_ids) & validation_set
        training = tuple(s for s in scenarios if s.scenario_id in training_ids)
        validation = tuple(s for s in scenarios if s.scenario_id in validation_set)
        x_train, y_train, _ = design(training)
        x_valid, y_valid, keys = design(validation)
        model = fit_ridge(x_train, y_train)
        estimates = predict(x_valid, model)
        constant = np.broadcast_to(model["target_mean"], y_valid.shape)
        for target_index, target_name in enumerate(TARGET_NAMES):
            metric_rows.append(metric_row("fold", fold_index, target_name,
                                         y_valid[:, target_index], estimates[:, target_index],
                                         constant[:, target_index]))
            for feature_index, feature_name in enumerate(FEATURE_NAMES):
                coefficient_rows.append({
                    "fold": fold_index, "target": target_name, "feature": feature_name,
                    "standardized_feature_coefficient": float(model["coefficients"][feature_index, target_index]),
                })
        for index, (scenario_id, period) in enumerate(keys):
            row = {"fold": fold_index, "scenario_id": int(scenario_id), "period": int(period)}
            for target_index, name in enumerate(TARGET_NAMES):
                row[f"actual_{name}"] = float(y_valid[index, target_index])
                row[f"predicted_{name}"] = float(estimates[index, target_index])
                row[f"constant_{name}"] = float(constant[index, target_index])
            prediction_rows.append(row)
        model_rows.append({
            "fold": fold_index, "training_ids": training_ids,
            "validation_ids": validation_ids, "training_rows": len(y_train),
            "validation_rows": len(y_valid),
            **{key: value.tolist() for key, value in model.items()},
        })

    predictions = pd.DataFrame(prediction_rows).sort_values(["scenario_id", "period"])
    assert len(predictions) == 120 * 28
    assert not predictions.duplicated(["scenario_id", "period"]).any()
    for name in TARGET_NAMES:
        metric_rows.append(metric_row("overall", None, name,
                                     predictions[f"actual_{name}"].to_numpy(),
                                     predictions[f"predicted_{name}"].to_numpy(),
                                     predictions[f"constant_{name}"].to_numpy()))
    metrics = pd.DataFrame(metric_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    predictions.to_csv(output / "oof_predictions.csv", index=False)
    coefficients.to_csv(output / "coefficients.csv", index=False)
    stability = coefficients.groupby(["target", "feature"])["standardized_feature_coefficient"].agg(
        coefficient_mean="mean", coefficient_std="std", coefficient_min="min", coefficient_max="max",
        positive_folds=lambda x: int((x > 0).sum()), negative_folds=lambda x: int((x < 0).sum()),
    ).reset_index()
    stability.to_csv(output / "coefficient_stability.csv", index=False)
    json_write(output / "fitted_models.json", model_rows)
    code_paths = [Path(__file__).resolve(), ROOT / "benchmark/safebench/data.py", ROOT / "benchmark/safebench/signals.py", ROOT / "src/safeinvest/contracts.py"]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(), "phase": "development",
        "seed": SEED, "outer_folds": 5, "ridge": RIDGE,
        "objective": "sum_squared_error + ridge * sum_squared_standardized_feature_coefficients",
        "feature_names": FEATURE_NAMES, "target_names": TARGET_NAMES,
        "history_window": 5, "minimum_target_period": 3, "window_std_ddof": 0,
        "feature_scaler_fit": "training_scenarios_only", "intercept_penalized": False,
        "squared_return_predictions_clipped": False,
        "training_validation_splits": [{key: m[key] for key in ("fold", "training_ids", "validation_ids")} for m in model_rows],
        "development_partition_sha256": partition_hash,
        "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "holdout_used": False, "validation": checks,
        "notes": [
            "仅解释下一年实际收益及其平方的线性可预测性；不等同投资策略收益。",
            "所有样本按完整情景划分；常数基准为各训练折目标均值。",
            "总体 R2 由逐折留出预测误差总和计算，未平均逐折 R2。",
            "平方收益衡量二阶矩，不能直接解释为已扣除条件均值的条件方差。",
            "固定特征和惩罚，无调参；开发结果不是独立最终证据。",
        ],
    }
    json_write(output / "manifest.json", manifest)
    overall = metrics[metrics.scope == "overall"]
    print(overall[["target", "model_mse", "constant_mse", "oos_r2"]].to_string(index=False))
    print(f"已保存 {len(predictions)} 条整情景折外预测；合成因果前缀检查 {checks['synthetic_prefix_checks']} 项通过。")


if __name__ == "__main__":
    main()
