"""Causal training-only ridge signals shared by diagnostics and Plan B."""
import numpy as np
from safeinvest.contracts import ASSETS, MarketObservation

SEED = 20261014
RIDGE = 10.0
FEATURE_NAMES = (
    "last_real_equity", "last_real_bond", "last_real_bill", "last_inflation",
    "window_real_equity_mean", "window_real_bond_mean", "window_real_bill_mean",
    "window_real_equity_std", "window_real_bond_std", "window_real_bill_std",
    "window_inflation_mean", "history_length",
)
TARGET_NAMES = tuple(f"real_return_{a}" for a in ASSETS) + tuple(
    f"squared_real_return_{a}" for a in ASSETS
)


def real_returns(observation: MarketObservation) -> np.ndarray:
    return (1.0 + np.asarray(observation.nominal_returns)) / (1.0 + observation.inflation) - 1.0


def causal_features(history: tuple[MarketObservation, ...]) -> np.ndarray:
    """Receive only observations available immediately before the decision year."""
    if len(history) < 2:
        raise ValueError("At least two observed years are required")
    window = history[-5:]
    rows = np.asarray([real_returns(o) for o in window])
    return np.concatenate([
        real_returns(history[-1]),
        [history[-1].inflation],
        rows.mean(axis=0),
        rows.std(axis=0, ddof=0),
        [np.mean([o.inflation for o in window]), len(history)],
    ])


def fit_ridge(features: np.ndarray, targets: np.ndarray) -> dict:
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0, ddof=0)
    feature_scale = np.where(feature_scale > 0.0, feature_scale, 1.0)
    target_mean = targets.mean(axis=0)
    standardized = (features - feature_mean) / feature_scale
    coefficients = np.linalg.solve(
        standardized.T @ standardized + RIDGE * np.eye(standardized.shape[1]),
        standardized.T @ (targets - target_mean),
    )
    return {
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "target_mean": target_mean,
        "coefficients": coefficients,
    }


def predict(features: np.ndarray, model: dict) -> np.ndarray:
    return (
        (features - model["feature_mean"]) / model["feature_scale"]
    ) @ model["coefficients"] + model["target_mean"]



def fit_signals(scenarios):
    if not scenarios or any(s.scenario_id > 120 for s in scenarios):
        raise PermissionError('Signals accept development scenarios only')
    features, targets = [], []
    for s in scenarios:
        x = np.asarray([causal_features(s.observations[:i]) for i in range(2, len(s.observations))])
        y = np.asarray([real_returns(o)[0]-real_returns(o)[2] for o in s.observations[2:]])
        features.append(x); targets.append(y)
    x, y = np.concatenate(features), np.concatenate(targets)
    models = {}
    for name in ('ridge', 'period', 'shifted'):
        design_x = x[:, -1:] if name == 'period' else x
        target = np.concatenate([np.roll(v, 7) for v in targets]) if name == 'shifted' else y
        models[name] = {k: np.asarray(v).tolist() for k, v in fit_ridge(design_x, target).items()}
    return dict(train_ids=[s.scenario_id for s in scenarios], models=models, ridge=RIDGE,
                feature_names=list(FEATURE_NAMES), minimum_period=3, shift=7)


def forecast(history, fitted, name='ridge', ood_limit=10.):
    if len(history) < 2:
        return 0., False, None
    x = causal_features(history)
    if name == 'period': x = x[-1:]
    m = fitted['models'][name]
    z = (x-np.asarray(m['feature_mean']))/np.asarray(m['feature_scale'])
    if not np.isfinite(z).all() or np.max(np.abs(z)) > ood_limit:
        return 0., True, None
    prediction = float(z @ np.asarray(m['coefficients']) + m['target_mean'])
    if not np.isfinite(prediction): return 0., True, None
    return prediction-float(m['target_mean']), False, prediction
