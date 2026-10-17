"""Convex path-risk allocation and transparent partial rebalancing."""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import linprog

from safeinvest.contracts import Action, Decision, validate_weights
from .metrics import expected_shortfall

PRODUCTS = ("conservative", "balanced", "growth")
REFERENCES = {"conservative": (.2, .3, .5), "balanced": (.5, .35, .15), "growth": (.75, .2, .05)}
POLICIES = ["plan_a_" + p for p in PRODUCTS] + [
    "plan_a_annual", "plan_a_hold", "plan_a_full_gate", "plan_a_band_edge", "plan_a_no_review"]
CONTROLS = POLICIES[3:]


def enabled(config):
    return config.get("plan_a", {}).get("enabled", False)


def parameter_grid(config):
    return [dict(risk_ratio=r, tracking_gamma=g) for r in config["plan_a"]["risk_ratios"]
            for g in config["plan_a"]["tracking_gammas"]]


def log_path_values(returns, weights, alpha=.9):
    """Return negative annual log growth, path log-ES, and valid subgradients."""
    returns = np.asarray(returns, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if returns.ndim != 3 or returns.shape[2] != 3 or min(returns.shape[:2]) == 0 or not 0 <= alpha < 1:
        raise ValueError("Expected nonempty scenario by time by three-asset returns and alpha in [0,1)")
    gross = 1 + returns @ weights
    if not np.isfinite(gross).all() or (gross <= 0).any():
        raise ValueError("Nonpositive or invalid portfolio gross return")
    n, horizon = gross.shape
    cumulative = np.concatenate([np.zeros((n, 1)), np.cumsum(np.log(gross), axis=1)], axis=1)
    derivative = returns / gross[..., None]
    cumulative_derivative = np.concatenate([np.zeros((n, 1, 3)), np.cumsum(derivative, axis=1)], axis=1)
    peak = np.zeros(n)
    peak_index = np.zeros(n, dtype=int)
    worst = np.zeros(n)
    gradients = np.zeros((n, 3))
    rows = np.arange(n)
    for t in range(1, horizon + 1):
        gap = peak - cumulative[:, t]
        update = gap > worst
        gradients[update] = (cumulative_derivative[rows, peak_index] - cumulative_derivative[:, t])[update]
        worst = np.maximum(worst, gap)
        update_peak = cumulative[:, t] > peak
        peak[update_peak] = cumulative[update_peak, t]
        peak_index[update_peak] = t
    order = np.argsort(-worst, kind="stable")
    mass = n * (1 - alpha)
    whole = min(n, int(np.floor(mass)))
    tail_weights = np.zeros(n)
    tail_weights[order[:whole]] = 1 / mass
    if whole < n:
        tail_weights[order[whole]] = (mass - whole) / mass
    return dict(objective=float(-np.mean(np.log(gross))),
                objective_gradient=-derivative.mean(axis=(0, 1)),
                risk=float(tail_weights @ worst), risk_gradient=tail_weights @ gradients,
                log_drawdowns=worst)


def grid_values(returns, weights, alpha):
    gross = 1 + np.einsum("nta,ka->ntk", returns, weights)
    cumulative = np.concatenate([np.zeros((len(returns), 1, len(weights))),
                                 np.cumsum(np.log(gross), axis=1)], axis=1)
    worst = (np.maximum.accumulate(cumulative, axis=1) - cumulative).max(axis=1)
    risks = np.asarray([expected_shortfall(worst[:, i], alpha) for i in range(len(weights))])
    return -np.log(gross).mean(axis=(0, 1)), risks


def solve_allocation(returns, budget, alpha=.9, bill_floor=0., options=None, cached_grid=None):
    """Kelley outer approximation: feasible incumbent minus LP lower bound is a gap."""
    options = options or {}
    tolerance = options.get("objective_tolerance", 1e-6)
    feasibility = options.get("feasibility_tolerance", 1e-8)
    max_iterations = options.get("max_iterations", 300)
    if budget < 0 or not 0 <= bill_floor <= 1 or not 0 <= alpha < 1:
        raise ValueError("Invalid risk budget, alpha or bill floor")
    if cached_grid is None:
        weights = np.array([(i / 20, j / 20, (20-i-j) / 20) for i in range(21) for j in range(21-i)])
        objectives, risks = grid_values(returns, weights, alpha)
    else:
        weights, objectives, risks = cached_grid
    admissible = weights[:, 2] >= bill_floor - 1e-12
    feasible = admissible & (risks <= budget + feasibility)
    best = None
    if feasible.any():
        index = int(np.argmin(np.where(feasible, objectives, np.inf)))
        best = (float(objectives[index]), weights[index].copy())
        grid_objective = best[0]
    else:
        grid_objective = None
    # Tangent planes under-estimate both convex functions over the whole simplex.
    matrix, rhs = [], []
    def add_cut(w):
        nonlocal best
        value = log_path_values(returns, w, alpha)
        for kind, last, bound in [("objective", -1., 0.), ("risk", 0., budget)]:
            gradient = value[kind + "_gradient"]
            matrix.append([*gradient, last])
            rhs.append(float(bound - value[kind] + gradient @ w))
        if value["risk"] <= budget + feasibility and (best is None or value["objective"] < best[0]):
            best = (value["objective"], np.array(w))
        return value
    seeds = [[0., 0., 1.], [1-bill_floor, 0., bill_floor], [0., 1-bill_floor, bill_floor]]
    if best is not None:
        seeds.append(best[1])
    for w in seeds:
        add_cut(np.asarray(w))
    for iteration in range(1, max_iterations + 1):
        lp = linprog([0., 0., 0., 1.], A_ub=matrix, b_ub=rhs,
                     A_eq=[[1., 1., 1., 0.]], b_eq=[1.],
                     bounds=[(0., 1.), (0., 1.), (bill_floor, 1.), (None, None)],
                     method="highs", options={"primal_feasibility_tolerance": 1e-9,
                                              "dual_feasibility_tolerance": 1e-9})
        if lp.status == 2:
            raise RuntimeError("Plan A path-risk budget infeasible (linear relaxation certificate)")
        if not lp.success:
            raise RuntimeError("Plan A LP failed: " + lp.message)
        lower_bound = float(lp.fun)
        w = np.maximum(lp.x[:3], 0.)
        w /= w.sum()
        add_cut(w)
        if best is not None and best[0] - lower_bound <= tolerance:
            if best[0] - lower_bound < -1e-7:
                raise RuntimeError("Plan A lower bound exceeds feasible objective")
            value = log_path_values(returns, best[1], alpha)
            validate_weights(tuple(best[1]))
            return dict(weights=best[1].tolist(), objective=best[0], lower_bound=lower_bound,
                        optimality_gap=max(0., best[0] - lower_bound), risk=value["risk"],
                        risk_budget=float(budget), risk_slack=float(budget - value["risk"]),
                        grid_objective=grid_objective,
                        grid_growth_gain=None if grid_objective is None else grid_objective - best[0],
                        iterations=iteration, cuts=len(rhs), status="optimal_within_tolerance",
                        bill_floor=bill_floor,
                        lp_certificate=dict(matrix=matrix[:-2], rhs=rhs[:-2], primal=lp.x.tolist(),
                            inequality_duals=lp.ineqlin.marginals.tolist(),
                            equality_dual=float(lp.eqlin.marginals[0]),
                            lower_duals=lp.lower.marginals.tolist(), upper_duals=lp.upper.marginals.tolist()))
    raise RuntimeError("Plan A solver exhausted iterations without a certified gap")


def annual_statistics(returns, weight, cost, alpha):
    """Exact real-currency accounting, including initial allocation and fee marks."""
    holdings = np.zeros((len(returns), 3))
    holdings[:, 2] = 100.
    peak = np.full(len(returns), 100.)
    drawdown = np.zeros(len(returns))
    for t in range(returns.shape[1]):
        wealth = holdings.sum(axis=1)
        moved = .5 * np.abs(wealth[:, None] * weight - holdings).sum(axis=1)
        holdings = (wealth - cost * moved)[:, None] * weight
        drawdown = np.maximum(drawdown, 1 - holdings.sum(axis=1) / peak)
        holdings *= 1 + returns[:, t]
        wealth = holdings.sum(axis=1)
        peak = np.maximum(peak, wealth)
        drawdown = np.maximum(drawdown, 1 - wealth / peak)
    growth = (holdings.sum(axis=1) / 100) ** (1 / returns.shape[1]) - 1
    return dict(p90_drawdown=float(np.quantile(drawdown, .9)),
                log_drawdown_es=expected_shortfall(-np.log1p(-drawdown), alpha),
                median_growth=float(np.median(growth)), p10_growth=float(np.quantile(growth, .1)))


def fit_plan_a(scenarios, config):
    started = time.perf_counter()
    if not scenarios or any(s.scenario_id not in range(1, 121) for s in scenarios):
        raise PermissionError("Plan A fit accepts development scenarios only")
    settings = config["plan_a"]
    returns = np.asarray([[(1 + np.asarray(o.nominal_returns)) / (1 + o.inflation) - 1
                          for o in s.observations] for s in scenarios])
    alpha = settings["alpha"]
    weights = np.array([(i / 20, j / 20, (20-i-j) / 20) for i in range(21) for j in range(21-i)])
    entries = {}
    for h in config["horizons"]:
        sample = returns[:, :h]
        objective, risk = grid_values(sample, weights, alpha)
        cached = (weights, objective, risk)
        for product in PRODUCTS:
            reference = log_path_values(sample, np.asarray(REFERENCES[product]), alpha)["risk"]
            for ratio in settings["risk_ratios"]:
                fit = solve_allocation(sample, reference * ratio, alpha,
                                       settings["bill_floors"][product], settings["solver"], cached)
                annual = annual_statistics(sample, np.asarray(fit["weights"]), config["cost_rate"], alpha)
                fit.update(reference_risk=reference, risk_ratio=ratio, horizon=h, product=product,
                           annual_with_fees=annual,
                           review_threshold=min(settings["review_maximum"],
                               max(settings["review_minimum"], settings["review_multiple"] * annual["p90_drawdown"])))
                entries[f"{product}:{h}:{ratio:g}"] = fit
    return dict(train_ids=[s.scenario_id for s in scenarios], entries=entries,
                training_seconds=time.perf_counter() - started)


class PlanAPolicy:
    def __init__(self, name, model, params, config, horizon, product=None, mode=None, review=None):
        if horizon is None:
            raise ValueError("Plan A requires an explicit horizon")
        product = product or (name.removeprefix("plan_a_") if name in ["plan_a_" + p for p in PRODUCTS] else "balanced")
        if set(model["train_ids"]) != set(model["plan_a"]["train_ids"]):
            raise ValueError("Plan A model training IDs differ")
        entry = model["plan_a"]["entries"][f"{product}:{horizon}:{params['risk_ratio']:g}"]
        self.name = name
        self.initial_target = tuple(entry["weights"])
        self.anchor = np.asarray(self.initial_target)
        self.cov = np.asarray(model["cov"])
        self.anchor_mean = float(np.asarray(model["mu"]) @ self.anchor)
        self.gamma = params["tracking_gamma"]
        self.settings = config["plan_a"]
        self.buffer = self.settings["buffer"]
        self.mode = mode or {"plan_a_annual": "annual", "plan_a_hold": "hold",
                            "plan_a_band_edge": "band_edge", "plan_a_full_gate": "full"}.get(name, "partial")
        self.review = review if review is not None else name in ["plan_a_" + p for p in PRODUCTS]
        self.review_threshold = entry["review_threshold"]
        self.requires_benefit_gate = self.mode in ("partial", "full")
        self.decision_calls = 0
        self.inference_seconds = 0.

    def decide(self, context):
        start = time.perf_counter()
        self.decision_calls += 1
        try:
            return self._decide(context)
        finally:
            self.inference_seconds += time.perf_counter() - start

    def _decide(self, c):
        current = np.asarray(c.holdings) / sum(c.holdings)
        drawdown = max(0., 1 - sum(c.holdings) / c.cumulative_inflation / c.real_peak)
        if self.review and drawdown >= self.review_threshold:
            return Decision(Action.HUMAN_REVIEW, ("PLAN_A_RISK_REVIEW",),
                "Your purchasing-power loss has exceeded our review threshold. A human adviser should reassess your plan. Existing investments remain in place while review is pending.")
        if self.mode == "hold":
            return Decision(Action.KEEP, ("PLAN_A_HOLD",), "Hold the existing assets.")
        if self.mode in ("annual", "band_edge"):
            target = self.anchor
            if self.mode == "band_edge":
                deviation = float(np.max(np.abs(current - self.anchor)))
                if deviation <= self.settings["band"]:
                    return Decision(Action.KEEP, ("PLAN_A_INSIDE_BAND",), "The allocation remains inside the diagnostic band.")
                target = self.anchor + (current - self.anchor) * self.settings["band"] / deviation
            return self._trade(c, target, None, None, "PLAN_A_DIAGNOSTIC")
        fractions = [0., 1.] if self.mode == "full" else self.settings["move_fractions"]
        targets = current + np.asarray(fractions)[:, None] * (self.anchor - current)
        gap = targets - self.anchor
        # Inverse-optimized strategic prior: anchor is the unconstrained quadratic optimum.
        values = 1 + self.anchor_mean - .5 * self.gamma * np.einsum("ij,jk,ik->i", gap, self.cov, gap)
        costs = c.transaction_cost_rate * .5 * np.abs(targets - current).sum(axis=1)
        if not np.isfinite(values).all() or (values <= 0).any():
            return Decision(Action.HUMAN_REVIEW, ("PLAN_A_INVALID_VALUE",), "The model estimate needs human review. Existing investments will be held.")
        chosen = int(np.argmax((1 - costs) * values))
        benefit = float((values[chosen] - values[0]) / values[chosen])
        cost = float(costs[chosen])
        if benefit <= cost + self.buffer:
            return Decision(Action.KEEP, ("PLAN_A_BENEFIT_BELOW_COST",),
                "The estimated improvement in strategic risk-adjusted value does not cover trading costs and the safety margin.",
                estimated_benefit=benefit, estimated_cost=cost)
        return self._trade(c, targets[chosen], benefit, cost, f"PLAN_A_MOVE_{fractions[chosen]:g}")

    @staticmethod
    def _trade(c, target, benefit, cost, reason):
        target = np.maximum(target, 0.)
        target /= target.sum()
        action = Action.REBALANCE if np.allclose(target, c.target_weights, atol=1e-10, rtol=0) else Action.CHANGE_TARGET
        explanation = ("Diagnostic adjustment toward the strategic allocation." if benefit is None else
            "This adjustment moves your investments toward the strategic allocation. Its estimated risk-adjusted benefit exceeds trading costs. Investment losses remain possible.")
        return Decision(action, (reason,), explanation, tuple(target), benefit, cost)
