#!/usr/bin/env python3
"""Recompute fitted path budgets with the independent scalar ledger engine."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmark'))
from safebench import ROOT
import numpy as np
from safeinvest.contracts import SimulationConfig
from safebench.data import load_partition
from safebench.engine import simulate
from safebench.experiment import code_digest, dump
from safebench.metrics import compute_metrics, expected_shortfall
from safebench.policies import StaticPolicy, PORTFOLIOS, EQUAL


def verify(run):
    config=json.loads((run/'config.json').read_text())
    info=json.loads((run/'manifest.json').read_text())
    if info['phase']!='development' or info['code_sha256']!=code_digest():
        raise ValueError('Budget verification requires unchanged development code')
    paths,sha=load_partition(ROOT/'data/SafeInvest_Synthetic_Data.csv')
    if sha!=info['partition_sha256']: raise ValueError('Development data changed')
    by_id={s.scenario_id:s for s in paths}
    maximum_error=0.; checked=0; infeasible=0
    for f in json.loads((run/'fitted_models.json').read_text()):
        ids=f['model']['train_ids']
        bundle=f['model']['frontier']
        assert set(bundle['train_ids'])==set(ids)
        cache={}
        def calculate(weights,h):
            key=(tuple(weights),h)
            if key not in cache:
                p=StaticPolicy('budget_check',tuple(weights))
                rows=[compute_metrics(simulate(by_id[sid],p,tuple(weights),
                    SimulationConfig(h,config['initial_wealth'],config['cost_rate']))) for sid in ids]
                growth=np.array([r['real_cagr'] for r in rows]); loss=np.array([r['real_max_drawdown'] for r in rows])
                cache[key]=dict(median_growth=float(np.median(growth)),p10=float(np.quantile(growth,.1)),
                                p90=float(np.quantile(loss,.9)),tail=expected_shortfall(loss))
            return cache[key]
        for trial,fit in bundle['fits'].items():
            setting=bundle['settings']['trials'][int(trial)]
            for diag in fit['diagnostics']:
                h=diag['horizon']; actual=calculate(fit['allocations'][str(h)],h)
                is_balanced=setting.get('reference')=='balanced'
                ref=calculate(PORTFOLIOS['balanced'] if is_balanced else EQUAL,h)
                budget=max(0,ref['p90']-setting['minimum_p90_reduction']) if is_balanced else ref['tail']*setting['risk_budget_ratio']
                risk=actual['p90'] if is_balanced else actual['tail']
                differences=[actual['median_growth']-diag['training_median_growth'],
                    actual['p10']-diag['training_p10_growth'],actual['tail']-diag['training_tail_drawdown'],
                    budget-diag['risk_budget']]
                maximum_error=max(maximum_error,*[abs(x) for x in differences])
                assert max(abs(x) for x in differences)<1e-10
                feasible=risk<=budget+1e-10 and actual['p10']>=ref['p10']-bundle['settings']['maximum_p10_loss']-1e-10
                assert bool(diag['feasible'])==feasible
                infeasible+=not feasible; checked+=1
    result=dict(all_checks_passed=True,checked_training_budgets=checked,infeasible_budgets=infeasible,
                maximum_numeric_discrepancy=maximum_error,holdout_used=False,
                scope='Training allocations, horizon-specific costs, P90/tail-mean budgets, P10 constraints and feasibility, recomputed with scalar ledger.')
    dump(run/'frontier_budget_verification.json',result); print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description='独立标量重算每个训练配置及期限的风险预算。')
    p.add_argument('run',type=Path); a=p.parse_args(); verify(a.run.resolve())
