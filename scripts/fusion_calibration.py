"""Training-only selection for long-horizon A and actual fusion execution."""
import numpy as np
import pandas as pd
from safeinvest.contracts import SimulationConfig
from safebench.engine import simulate
from safebench.metrics import compute_metrics
from safebench.plan_a import PlanAPolicy, parameter_grid, allocation_available, selection_constraints
from efficient_fusion import EfficientFusion


def inner_scores(scenarios, model, params, config, horizons, fusion, rule):
    """Score both executions against the identical fee-paying annual anchor."""
    if set(s.scenario_id for s in scenarios) & set(model['train_ids']):
        raise ValueError('Inner validation overlaps fitted training IDs')
    rows = []
    for h in horizons:
        entry = model['plan_a']['entries'][f"balanced:{h}:{params['risk_ratio']:g}"]
        for scenario in scenarios:
            annual = PlanAPolicy('annual', model, params, config, h, mode='annual', review=False)
            cfg = SimulationConfig(h, config['initial_wealth'], config['cost_rate'])
            annual_growth = compute_metrics(simulate(scenario, annual, annual.initial_target, cfg))['real_cagr']
            for family in ('a', 'fusion'):
                policy = (PlanAPolicy('a', model, params, config, h, review=True) if family == 'a' else
                          EfficientFusion('fusion', model, params, config, h, fusion, rule))
                row = compute_metrics(simulate(scenario, policy, policy.initial_target, cfg))
                row.update(family=family, annual_growth=annual_growth,
                           plan_a_risk_budget=entry['risk_budget'],
                           plan_a_validation_budget=entry.get('validation_budget', entry['risk_budget']))
                rows.append(row)
    return rows


def choose_group(scores, model, config, horizons, ids, family, unavailable):
    """Keep A's ranking and fixed constraints; retain every infeasible trial."""
    scoped = dict(config, horizons=list(horizons))
    rows = []
    choices = []
    for trial, params in enumerate(parameter_grid(config)):
        failed = unavailable.get(trial, [])
        refit_ok = allocation_available(model, 'balanced', params, horizons)
        if failed or not refit_ok:
            rows.append(dict(family=family, trial=trial, params=params, selection_feasible=False,
                             allocation_feasible=False, inner_infeasible_folds=failed, refit_feasible=refit_ok))
            continue
        block = scores[(scores.family == family) & (scores.trial == trial)]
        expected = {(sid, h) for sid in ids for h in horizons}
        actual = list(zip(block.scenario_id, block.horizon))
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError('Incomplete or duplicated inner validation grid')
        violation, details = selection_constraints(block, scoped)
        feasible = violation <= 1e-10
        score = float(np.log1p(block.real_cagr).mean())
        choices.append((trial, feasible, violation, score))
        rows.append(dict(family=family, trial=trial, params=params, mean_utility=score,
                         selection_feasible=feasible, allocation_feasible=True, refit_feasible=True,
                         constraint_violation=violation, horizon_constraints=details))
    if not choices:
        raise RuntimeError('No executable long-horizon allocation')
    winner = min(choices, key=lambda x: (not x[1], x[2] if not x[1] else 0., -x[3], x[0]))
    return dict(parameter_grid(config)[winner[0]], selection_feasible=winner[1],
                constraint_violation=winner[2], trial=winner[0]), rows
