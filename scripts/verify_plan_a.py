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


def verify_entry(returns, entry, alpha):
    w = np.asarray(entry['weights'])
    assert np.all(w >= -1e-12) and abs(w.sum() - 1) < 1e-9
    assert w[2] >= entry['bill_floor'] - 1e-9
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
    objective = -float(np.mean(growths))
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
    rows = []
    for f in fits:
        model = f['model']
        assert model['train_ids'] == model['plan_a']['train_ids']
        sample = np.array([[(1 + np.array(o.nominal_returns)) / (1 + o.inflation) - 1
                           for o in by_id[sid].observations] for sid in model['train_ids']])
        for key, e in model['plan_a']['entries'].items():
            checks = verify_entry(sample[:, :e['horizon']], e, config['plan_a']['alpha'])
            rows.append(dict(fold=f['fold'], entry=key, **checks))
        for name in ['plan_a_annual', 'plan_a_hold', 'plan_a_full_gate', 'plan_a_band_edge', 'plan_a_no_review']:
            assert f['params'][name] == f['params']['plan_a_balanced']
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
    result = dict(all_checks_passed=True, certified_allocations=len(rows), legacy_rows_identical=legacy_rows,
                  maximum_optimality_gap=max(x['optimality_gap'] for x in rows),
                  maximum_lp_primal_residual=max(x['lp_primal_residual'] for x in rows),
                  maximum_lp_dual_residual=max(x['lp_dual_residual'] for x in rows), holdout_used=False)
    (args.run / 'independent_plan_a.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
