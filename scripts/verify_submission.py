"""Independent reconstruction of development selection, fees and matched controls."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'benchmark')]
import numpy as np
import pandas as pd
from safebench.signals import fit_signals
from safebench.data import load_partition

RULES = ['plan_a_balanced', 'submission_a_continuous', 'submission_a_band',
         'submission_c_full', 'submission_c_cap10', 'submission_c_cap20',
         'submission_b_long025', 'submission_b_long05']
ORDER = [RULES[i] for i in (0, 1, 2, 6, 7, 4, 5, 3)]
KEYS = ['policy', 'scenario_id', 'horizon', 'cost_rate']


def tail_mean(values, alpha):
    ordered = sorted(values, reverse=True)
    mass = len(ordered)*(1-alpha)
    return sum(v*max(0., min(1., mass-i)) for i, v in enumerate(ordered))/mass


def verify(run):
    out = Path(run)
    meta = json.loads((out/'manifest.json').read_text())
    if meta['phase'] != 'development' or meta['scenario_ids'] != list(range(1, 121)):
        raise PermissionError('Only development evidence may be reconstructed')
    config = json.loads((out/'config.json').read_text())
    metrics = pd.read_csv(out/'scenario_metrics.csv')
    records = pd.read_csv(out/'decisions.csv.gz')
    fits = json.loads((out/'fitted_models.json').read_text())
    splits = json.loads((out/'splits.json').read_text())
    trials = pd.read_csv(out/'tuning_trials.csv')
    settings = config['plan_a']['selection']
    selections, horizon_checks = [], 0
    for fit, split in zip(fits, splits):
        frames = []
        for inner in split['inner']:
            block = pd.DataFrame(inner['submission_validation_metrics'])
            assert set(block.scenario_id) == set(inner['validation_ids'])
            assert set(block.scenario_id).isdisjoint(inner['train_ids'])
            assert set(block.scenario_id).isdisjoint(split['validation_ids'])
            assert len(block) == len(inner['validation_ids'])*len(config['horizons'])*8
            assert not block.duplicated(['policy', 'scenario_id', 'horizon']).any()
            frames.append(block)
        scores = pd.concat(frames, ignore_index=True)
        feasible = []
        for rule in RULES:
            candidate = scores[scores.policy == rule]
            violation = 0.
            saved = trials[(trials.outer_fold == fit['fold'])&(trials.policy == 'submission_selected')&
                           (trials.execution_rule == rule)].iloc[0]
            diagnostics = json.loads(saved.horizon_constraints)
            for h, g in candidate.groupby('horizon'):
                risk = tail_mean(-np.log1p(-g.real_max_drawdown), config['plan_a']['alpha'])
                budget = g.plan_a_validation_budget.mean()
                median_loss = g.annual_growth.median()-g.real_cagr.median()
                p10_loss = g.annual_growth.quantile(.1)-g.real_cagr.quantile(.1)
                rate = g.review.mean()
                v = sum([max(0., (risk-budget-settings['log_risk_slack'])/max(budget, .01)),
                         max(0., (median_loss-settings['median_growth_loss'])/settings['median_growth_loss']),
                         max(0., (p10_loss-settings['p10_growth_loss'])/settings['p10_growth_loss']),
                         max(0., (rate-settings['maximum_review_rate'])/settings['maximum_review_rate'])])
                d = next(d for d in diagnostics if d['horizon'] == h)
                np.testing.assert_allclose([risk, budget, median_loss, p10_loss, v],
                    [d['log_risk'], d['validation_budget'], d['median_growth_loss'], d['p10_growth_loss'],
                     d['constraint_violation']], atol=2e-12, rtol=2e-12)
                violation += v
                horizon_checks += 1
            score = np.log1p(candidate.real_cagr).mean()
            np.testing.assert_allclose([score, violation], [saved.mean_utility, saved.constraint_violation], atol=2e-12)
            assert bool(saved.selection_feasible) == (violation <= 1e-10)
            if violation <= 1e-10:
                feasible.append((rule, score))
        expected = 'plan_a_balanced'
        if feasible:
            best = max(feasible, key=lambda r: r[1])[0]
            pivot = scores.assign(u=np.log1p(scores.real_cagr)).groupby(['scenario_id', 'policy']).u.mean().unstack()
            near = []
            for rule, _ in feasible:
                diff = pivot[best]-pivot[rule]
                se = diff.std(ddof=1)/np.sqrt(len(diff))
                if diff.mean() <= se+1e-12:
                    near.append(rule)
            expected = min(near, key=ORDER.index)
        params = fit['params']['submission_selected']
        assert params['execution_rule'] == expected
        assert params['execution_selection_feasible'] == bool(feasible)
        selections.append(dict(fold=fit['fold'], rule=expected, feasible=bool(feasible)))
    prior = pd.read_csv(ROOT/'benchmark/outputs/development_plan_a_calibrated_v1/scenario_metrics.csv')
    common = metrics.merge(prior[KEYS], on=KEYS, validate='one_to_one').sort_values(KEYS).reset_index(drop=True)
    prior = prior.sort_values(KEYS).reset_index(drop=True)
    columns = [c for c in prior if c != 'policy_inference_seconds']
    pd.testing.assert_frame_equal(common[columns], prior[columns], check_dtype=False, check_exact=True)
    old_c = pd.read_csv(ROOT/'benchmark/outputs/development_plan_c_screen_v3/scenario_metrics.csv')
    old_c = old_c[old_c.policy == 'plan_c_l1'].copy()
    old_c['policy'] = 'submission_c_full'
    match = metrics.merge(old_c[KEYS], on=KEYS, validate='one_to_one').sort_values(KEYS).reset_index(drop=True)
    old_c = old_c.sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(match[columns], old_c[columns], check_dtype=False, check_exact=True)
    caps = gates = zero = historic_checks = 0
    for _, group in records[records.policy.str.startswith('submission_')].groupby(
            ['policy', 'scenario_id', 'cost_rate', 'simulation_horizon']):
        peak = config['initial_wealth']; historic = 0.; inflation = 1.
        for row in group.sort_values('period').itertuples():
            diagnostics = None if pd.isna(row.policy_diagnostics) else json.loads(row.policy_diagnostics)
            if diagnostics and 'depth' in diagnostics:
                d = diagnostics
                np.testing.assert_allclose(d['historic_max_drawdown'], historic, atol=1e-12)
                np.testing.assert_allclose(d['best_objective'], max(d['first_action_values']), atol=1e-12)
                np.testing.assert_allclose(d['hold_objective'], d['first_action_values'][0], atol=1e-12)
                np.testing.assert_allclose(d['best_objective'], d['mean_log_growth']-d['log_drawdown_es'], atol=1e-12)
                benefit = d['first_cost']+(1-d['first_cost'])*(-np.expm1(d['hold_objective']-d['best_objective']))
                np.testing.assert_allclose([row.estimated_benefit, row.estimated_cost], [benefit, d['first_cost']], atol=1e-12)
                gates += 1; historic_checks += 1
                if 'move_cap' in d:
                    assert row.moved_amount/sum(getattr(row, 'pre_trade_holdings_'+a) for a in ('equity','bond','bill')) <= d['move_cap']+1e-12
                    caps += 1
            post = sum(getattr(row, 'post_trade_holdings_'+a) for a in ('equity', 'bond', 'bill'))/inflation
            historic = max(historic, 1-post/peak); peak = max(peak, post)
            historic = max(historic, 1-row.real_wealth/peak); peak = max(peak, row.real_wealth)
            inflation = row.cumulative_inflation
    for name in ('submission_b_long025', 'submission_b_long05'):
        a = metrics[(metrics.policy == 'plan_a_balanced')&metrics.horizon.isin([5,6])].sort_values(KEYS[1:])
        b = metrics[(metrics.policy == name)&metrics.horizon.isin([5,6])].sort_values(KEYS[1:])
        cols = ['real_cagr', 'real_max_drawdown', 'terminal_real_wealth', 'total_real_cost', 'trade_rate', 'review']
        np.testing.assert_array_equal(a[cols].to_numpy(), b[cols].to_numpy()); zero += len(a)
    scenarios, _ = load_partition(ROOT/'data/SafeInvest_Synthetic_Data.csv')
    by_id = {s.scenario_id:s for s in scenarios}
    for fit in fits:
        signals = fit_signals([by_id[i] for i in fit['model']['train_ids']])
        assert signals == fit['model']['signals']
    result = dict(all_checks_passed=True, inner_selections=selections, inner_horizon_checks=horizon_checks,
        unchanged_a_rows=len(prior), unchanged_c1_rows=len(old_c), short_zero_signal_rows=zero,
        cap_checks=caps, net_value_checks=gates, historic_state_checks=historic_checks,
        reconstructed_signal_fits=len(fits), holdout_used=False)
    with (out/'submission_verification.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('run'); verify(p.parse_args().run)
