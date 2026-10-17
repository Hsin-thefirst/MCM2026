"""Independently audit Plan A path objectives and LP primal/dual certificates."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmark'))

import numpy as np
import pandas as pd
from safebench import ROOT
from safebench.data import load_partition


def path_values(returns, w, alpha):
    growths, losses = [], []
    for path in returns:
        logs = [float(np.log1p(sum(float(x) * float(y) for x, y in zip(row, w)))) for row in path]
        growths.append(sum(logs) / len(logs))
        # Exhaustive intervals, independent of the optimizer's running-peak recurrence.
        losses.append(max([0.] + [-sum(logs[start:end]) for start in range(len(logs))
                                 for end in range(start + 1, len(logs) + 1)]))
    losses = sorted(losses, reverse=True)
    mass = len(losses) * (1 - alpha)
    tail = sum(losses[i] * max(0., min(1., mass - i)) for i in range(len(losses))) / mass
    return -float(np.mean(growths)), tail


def verify_infeasible(entry):
    from scipy.optimize import linprog
    cert=entry['failure_certificate']
    result=linprog([0.,0.,0.,1.],A_ub=cert['matrix'],b_ub=cert['rhs'],
        A_eq=[[1.,1.,1.,0.]],b_eq=[1.],
        bounds=[(0.,1.),(0.,1.),(cert['bill_floor'],1.),(None,None)],method='highs')
    assert result.status==2 and 'weights' not in entry


def verify_entry(returns, entry, alpha):
    w = np.asarray(entry['weights'])
    assert np.all(w >= -1e-12) and abs(w.sum() - 1) < 1e-9
    assert w[2] >= entry['bill_floor'] - 1e-9
    objective, tail = path_values(returns, w, alpha)
    assert abs(objective - entry['objective']) < 1e-10
    assert abs(tail - entry['risk']) < 1e-10
    assert tail <= entry['risk_budget'] + 1e-8
    c = entry['lp_certificate']
    a, b = np.asarray(c['matrix']), np.asarray(c['rhs'])
    x, y = np.asarray(c['primal']), np.asarray(c['inequality_duals'])
    lo, hi = np.asarray(c['lower_duals']), np.asarray(c['upper_duals'])
    eq = np.array([1., 1., 1., 0.])
    bound_lo, bound_hi = np.array([0., 0., entry['bill_floor']]), np.ones(3)
    residual = max(0., float((a @ x - b).max()), abs(eq @ x - 1),
                   float((bound_lo - x[:3]).max()), float((x[:3] - bound_hi).max()))
    stationarity = np.array([0., 0., 0., 1.]) - a.T @ y - c['equality_dual'] * eq - lo - hi
    assert residual < 1e-7 and np.max(np.abs(stationarity)) < 1e-7
    assert y.max() <= 1e-8 and hi.max() <= 1e-8 and lo.min() >= -1e-8
    assert abs(lo[3]) + abs(hi[3]) < 1e-8
    dual = b @ y + c['equality_dual'] + bound_lo @ lo[:3] + bound_hi @ hi[:3]
    assert abs(dual - entry['lower_bound']) < 1e-7
    assert abs(x[3] - dual) < 1e-7
    assert -1e-7 <= objective - dual <= 1.1e-6
    return dict(risk_slack=entry['risk_budget'] - tail, optimality_gap=objective - dual,
                lp_primal_residual=residual, lp_dual_residual=float(np.max(np.abs(stationarity))))


def main():
    parser = argparse.ArgumentParser(description='独立核验计划A开发实验的优化证书与参照一致性')
    parser.add_argument('run', type=Path)
    parser.add_argument('--reference', type=Path, default=ROOT / 'benchmark/outputs/development_consolidated')
    args = parser.parse_args()
    manifest = json.loads((args.run / 'manifest.json').read_text())
    assert manifest['phase'] == 'development' and not manifest['holdout_used']
    config = json.loads((args.run / 'config.json').read_text())
    scenarios, sha = load_partition(ROOT / 'data/SafeInvest_Synthetic_Data.csv')
    assert sha == manifest['partition_sha256']
    by_id = {s.scenario_id: s for s in scenarios}
    fits = json.loads((args.run / 'fitted_models.json').read_text())
    rows = []; infeasible = 0; inner_infeasible = 0
    trial_table=pd.read_csv(args.run/'tuning_trials.csv')
    selected_feasible=0; invalid_trials=0
    references={'conservative':(.2,.3,.5),'balanced':(.5,.35,.15),'growth':(.75,.2,.05)}
    for split in json.loads((args.run/'splits.json').read_text()):
        for inner in split['inner']:
            for entry in inner.get('infeasible_allocations',{}).values():
                verify_infeasible(entry)
                inner_infeasible+=1
    for f in fits:
        model = f['model']
        assert model['train_ids'] == model['plan_a']['train_ids']
        sample = np.array([[(1 + np.array(o.nominal_returns)) / (1 + o.inflation) - 1
                           for o in by_id[sid].observations] for sid in model['train_ids']])
        reference_cache={}
        for key, e in model['plan_a']['entries'].items():
            ref_key=(e['product'],e['horizon'])
            if ref_key not in reference_cache:
                reference_cache[ref_key]=path_values(sample[:,:e['horizon']],references[e['product']],config['plan_a']['alpha'])[1]
            reference=reference_cache[ref_key]
            assert abs(e['reference_risk']-reference)<1e-10
            assert abs(e['risk_budget']-reference*e['risk_ratio'])<1e-10
            if 'validation_risk_ratio' in config['plan_a']:
                assert abs(e['validation_budget']-e['reference_risk']*config['plan_a']['validation_risk_ratio'])<1e-12
            if e['status']=='infeasible':
                verify_infeasible(e)
                infeasible+=1
                continue
            checks = verify_entry(sample[:, :e['horizon']], e, config['plan_a']['alpha'])
            rows.append(dict(fold=f['fold'], entry=key, **checks))
        for name in ['plan_a_annual', 'plan_a_hold', 'plan_a_full_gate', 'plan_a_band_edge', 'plan_a_no_review']:
            assert f['params'][name] == f['params']['plan_a_balanced']
        if 'horizon_constraints' in trial_table:
            grid=[(r,g) for r in config['plan_a']['risk_ratios'] for g in config['plan_a']['tracking_gammas']]
            for product in references:
                name='plan_a_'+product
                block=trial_table[(trial_table.outer_fold==f['fold'])&(trial_table.policy==name)]
                assert len(block)==len(grid) and not block.trial.duplicated().any()
                eligible=[]; budgets={}
                for row in block.itertuples():
                    if not row.allocation_feasible:
                        assert not row.selection_feasible
                        invalid_trials+=1
                        continue
                    details=json.loads(row.horizon_constraints)
                    assert {x['horizon'] for x in details}==set(config['horizons'])
                    violation=0.; settings=config['plan_a']['selection']
                    for x in details:
                        h=x['horizon']
                        if 'validation_risk_ratio' in config['plan_a']:
                            assert abs(budgets.setdefault(h,x['validation_budget'])-x['validation_budget'])<1e-12
                        v=sum([max(0.,(x['log_risk']-x['validation_budget']-settings['log_risk_slack'])/max(x['validation_budget'],.01)),
                            max(0.,(x['median_growth_loss']-settings['median_growth_loss'])/settings['median_growth_loss']),
                            max(0.,(x['p10_growth_loss']-settings['p10_growth_loss'])/settings['p10_growth_loss']),
                            max(0.,(x['review_rate']-settings['maximum_review_rate'])/settings['maximum_review_rate'])])
                        assert abs(v-x['constraint_violation'])<1e-10
                        assert x['feasible']==(v<=1e-10)
                        violation+=v
                    assert abs(violation-row.constraint_violation)<1e-10
                    assert row.selection_feasible==(violation<=1e-10)
                    eligible.append(row)
                chosen=min(eligible,key=lambda row:(not row.selection_feasible,
                    row.constraint_violation if not row.selection_feasible else 0.,-row.mean_utility,row.trial))
                params=f['params'][name]
                assert grid[chosen.trial]==(params['risk_ratio'],params['tracking_gamma'])
                assert params['selection_feasible']==chosen.selection_feasible
                selected_feasible+=int(chosen.selection_feasible)
    pd.DataFrame(rows).to_csv(args.run / 'allocation_certificates.csv', index=False)
    legacy_rows = None
    if (args.reference / 'scenario_metrics.csv').exists():
        old = pd.read_csv(args.reference / 'scenario_metrics.csv')
        new = pd.read_csv(args.run / 'scenario_metrics.csv')
        keys = ['policy', 'scenario_id', 'horizon', 'cost_rate']
        common = new.merge(old[keys], on=keys, validate='one_to_one')
        pd.testing.assert_frame_equal(old.sort_values(keys).reset_index(drop=True),
                                      common[old.columns].sort_values(keys).reset_index(drop=True),
                                      check_exact=True)
        legacy_rows = len(common)
    result = dict(all_checks_passed=True, certified_allocations=len(rows), certified_infeasible=infeasible,
                  certified_inner_infeasible=inner_infeasible, unavailable_trials=invalid_trials,
                  selected_inner_feasible=selected_feasible if 'horizon_constraints' in trial_table else None,
                  legacy_rows_identical=legacy_rows,
                  maximum_optimality_gap=max(x['optimality_gap'] for x in rows),
                  maximum_lp_primal_residual=max(x['lp_primal_residual'] for x in rows),
                  maximum_lp_dual_residual=max(x['lp_dual_residual'] for x in rows), holdout_used=False)
    (args.run / 'independent_plan_a.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
