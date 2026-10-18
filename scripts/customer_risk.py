"""Customer-specific defensive projection, with explicit infeasible selections."""
from __future__ import annotations

import numpy as np
from safeinvest.contracts import validate_weights
from safebench.metrics import binomial_upper
from safebench.plan_a import PlanAPolicy


class CustomerRiskPolicy(PlanAPolicy):
    """Blend a fitted strategic anchor with the same training set's min-variance anchor."""

    def __init__(self, name, model, params, config, profile, fraction, early_review=False,
                 early_review_fraction=.8, mode="partial"):
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("Mix fraction must be in [0,1]")
        if not 0 < early_review_fraction <= 1 or not 0 < profile["loss_tolerance"] < 1:
            raise ValueError("Invalid review fraction or loss tolerance")
        if not model["train_ids"] or any(sid not in range(1, 121) for sid in model["train_ids"]):
            raise PermissionError("Customer research accepts development-trained models only")
        product = "balanced" if profile["profile_id"] == "E" else "conservative"
        super().__init__(name, model, params, config, profile["horizon"],
                         product=product, mode=mode, review=mode == "partial")
        defensive = np.asarray(model["minvar"], dtype=float)
        validate_weights(tuple(defensive))
        self.anchor = fraction * self.anchor + (1 - fraction) * defensive
        self.initial_target = tuple(self.anchor)
        validate_weights(self.initial_target)
        self.anchor_mean = float(np.asarray(model["mu"]) @ self.anchor)
        if early_review:
            self.review_threshold = min(self.review_threshold,
                                        early_review_fraction * profile["loss_tolerance"])


def select_fraction(frame, profile, settings):
    """Use inner OOF execution only; keep a marked diagnostic if all trials fail."""
    if frame.empty:
        raise ValueError("Missing inner execution sample")
    if frame.duplicated(["fraction", "cost_rate", "scenario_id"]).any():
        raise ValueError("Duplicate inner scenario")
    if not frame.scenario_id.between(1, 120).all():
        raise PermissionError("Selection only accepts development scenarios")
    fractions = settings["mix_fractions"]
    costs = settings["costs"]
    expected = set(frame.scenario_id)
    confidence = 1 - (1 - settings["family_confidence"]) / settings["comparison_family"]
    evidence = []
    for fraction in fractions:
        violations = []
        diagnostics = []
        for cost in costs:
            g = frame[(frame.fraction == fraction) & (frame.cost_rate == cost)]
            ref = frame[(frame.fraction == 1.) & (frame.cost_rate == cost)]
            if set(g.scenario_id) != expected or set(ref.scenario_id) != expected:
                raise ValueError("Incomplete paired inner sample")
            target = max(0., min(settings["customer_headroom"] * profile["loss_tolerance"],
                                ref.real_max_drawdown.quantile(.9) - settings["minimum_p90_reduction"]))
            p90 = float(g.real_max_drawdown.quantile(.9))
            breaches = int((g.real_max_drawdown > profile["loss_tolerance"]).sum())
            upper = binomial_upper(breaches, len(g), confidence)
            bill_min = float(g.strategic_bill_min.min())
            liquidity_ok = profile["liquidity"] != "high" or bill_min >= settings["liquidity_floor"] - 1e-8
            violations.extend([max(0., p90 - target),
                               max(0., upper - settings["maximum_breach_probability"]),
                               0. if liquidity_ok else 1.])
            diagnostics.append(dict(cost_rate=cost, p90=p90, target=target, breaches=breaches,
                                    upper_bound=upper, n=len(g), liquidity_ok=liquidity_ok,
                                    strategic_bill_min=bill_min))
        normal = frame[(frame.fraction == fraction) & (frame.cost_rate == costs[0])]
        violation = float(sum(violations))
        evidence.append(dict(fraction=fraction, feasible=violation <= 1e-10,
                             violation=violation, median_growth=float(normal.real_cagr.median()),
                             diagnostics=diagnostics))
    feasible = [r for r in evidence if r["feasible"]]
    if feasible:
        chosen = max(feasible, key=lambda r: (r["median_growth"], r["fraction"]))
    else:
        chosen = min(evidence, key=lambda r: (r["violation"], -r["median_growth"], r["fraction"]))
    return dict(chosen, evidence=evidence, confidence=confidence,
                interpretation="OOF diagnostic; not an independently calibrated risk guarantee")


def unavoidable_drawdown(real_returns):
    """Lower bound on every long-only, fully invested strategy's path MDD, before fees."""
    returns = np.asarray(real_returns, dtype=float)
    if returns.ndim != 3 or returns.shape[2] != 3 or min(returns.shape[:2]) < 1:
        raise ValueError("Expected nonempty scenario by period by three-asset returns")
    gross = 1 + returns
    if not np.isfinite(gross).all() or (gross <= 0).any():
        raise ValueError("Invalid gross returns")
    upper_log = np.concatenate([np.zeros((len(returns), 1)),
                                np.log(gross.max(axis=2)).cumsum(axis=1)], axis=1)
    return 1 - np.exp(-(np.maximum.accumulate(upper_log, axis=1) - upper_log).max(axis=1))
