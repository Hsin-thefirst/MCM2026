"""Explain fixed Plan A selections and stress joint annual return order without retuning."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmark'))

import numpy as np
import pandas as pd
from safebench import ROOT
from safebench.data import load_partition
from safebench.engine import simulate
from safebench.experiment import dump, evaluate, code_digest
from safebench.metrics import compute_metrics, expected_shortfall, summarize
from safebench.plan_a import PlanAPolicy, PRODUCTS, annual_statistics, grid_values, REFERENCES
from safebench.policies import fit_model
from safeinvest.contracts import SimulationConfig


def main():
    parser = argparse.ArgumentParser(description='固定已选参数，定位内层约束失败并做联合年度顺序压力诊断')
    parser.add_argument('run', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to((ROOT / 'benchmark/outputs').resolve()):
        raise ValueError('Diagnostic output must remain under benchmark/outputs')
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.run / 'manifest.json').read_text())
    assert manifest['phase'] == 'development' and not manifest['holdout_used']
    config = json.loads((args.run / 'config.json').read_text())
    scenarios, sha = load_partition(ROOT / 'data/SafeInvest_Synthetic_Data.csv')
    assert sha == manifest['partition_sha256']
    splits = json.loads((args.run / 'splits.json').read_text())
    fits = json.loads((args.run / 'fitted_models.json').read_text())
    by_id = {s.scenario_id: s for s in scenarios}
    dump(args.out / 'manifest.json', dict(parent_run=str(args.run), parent_manifest=manifest,
         script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), holdout_used=False,
         diagnostic_code_sha256=code_digest(),
         purpose='Reproduce selected inner-validation constraints; fixed-fit order stress, no tuning',
         order_seeds=[20261017], orders=['original', 'reverse', 'shuffle']))
    constraints, stress, decomposition, floors = [], [], [], []
    settings = config['plan_a']['selection']
    for split, fit in zip(splits, fits):
        assert split['outer_fold'] == fit['fold']
        training_sample=np.array([[(1+np.array(o.nominal_returns))/(1+o.inflation)-1
            for o in by_id[sid].observations] for sid in split['train_ids']])
        weights=np.array([(i/20,j/20,(20-i-j)/20) for i in range(21) for j in range(21-i)])
        for h in config['horizons']:
            _, risks=grid_values(training_sample[:,:h],weights,config['plan_a']['alpha'])
            for product in PRODUCTS:
                floor=config['plan_a']['bill_floors'][product]
                minimum=float(risks[weights[:,2]>=floor-1e-12].min())
                entry=next(e for e in fit['model']['plan_a']['entries'].values()
                           if e['product']==product and e['horizon']==h)
                reference=entry['reference_risk']
                floors.append(dict(fold=fit['fold'],product=product,horizon=h,min_grid_risk=minimum,
                    reference_risk=reference,minimum_ratio=minimum/reference if reference>0 else None))
        pieces = []
        for inner in split['inner']:
            model = fit_model([by_id[s] for s in inner['train_ids']], config)
            valid = [by_id[s] for s in inner['validation_ids']]
            for product in PRODUCTS:
                name = 'plan_a_' + product
                params = fit['params'][name]
                metrics, _ = evaluate(valid, model, {name: params}, config, names=[name], keep_records=False)
                annual = []
                for s in valid:
                    for h in config['horizons']:
                        p = PlanAPolicy(name, model, params, config, h, product=product, mode='annual', review=False)
                        r = simulate(s, p, p.initial_target, SimulationConfig(h, config['initial_wealth'], config['cost_rate']))
                        annual.append(dict(scenario_id=s.scenario_id, horizon=h, annual_growth=compute_metrics(r)['real_cagr']))
                pieces.append(metrics.merge(pd.DataFrame(annual), on=['scenario_id', 'horizon'], validate='one_to_one'))
        for name, block in pd.concat(pieces, ignore_index=True).groupby('policy'):
            violation = 0.
            for h, g in block.groupby('horizon'):
                risk = expected_shortfall(-np.log1p(-g.real_max_drawdown.to_numpy()), config['plan_a']['alpha'])
                budget = g.plan_a_validation_budget.mean() if 'plan_a_validation_budget' in g else g.plan_a_risk_budget.mean()
                median_loss = g.annual_growth.median() - g.real_cagr.median()
                p10_loss = g.annual_growth.quantile(.1) - g.real_cagr.quantile(.1)
                rate = g.review.mean()
                violation += sum([max(0., (risk-budget-settings['log_risk_slack'])/max(budget, .01)),
                    max(0., (median_loss-settings['median_growth_loss'])/settings['median_growth_loss']),
                    max(0., (p10_loss-settings['p10_growth_loss'])/settings['p10_growth_loss']),
                    max(0., (rate-settings['maximum_review_rate'])/settings['maximum_review_rate'])])
                constraints.append(dict(fold=fit['fold'], policy=name, horizon=h, log_risk=risk,
                    training_budget=g.plan_a_risk_budget.mean(), validation_budget=budget,
                    risk_limit=budget+settings['log_risk_slack'], median_growth_loss=median_loss,
                    p10_growth_loss=p10_loss, review_rate=rate,
                    risk_pass=risk<=budget+settings['log_risk_slack']+1e-10,
                    median_pass=median_loss<=settings['median_growth_loss']+1e-10,
                    p10_pass=p10_loss<=settings['p10_growth_loss']+1e-10,
                    review_pass=rate<=settings['maximum_review_rate']+1e-10))
            assert abs(violation-fit['params'][name]['constraint_violation']) < 1e-8
        valid = [by_id[s] for s in split['validation_ids']]
        for order in ['original', 'reverse', 'shuffle']:
            altered = []
            for s in valid:
                index = np.arange(len(s.observations))
                if order == 'reverse': index = index[::-1]
                if order == 'shuffle': index = np.random.default_rng(20261017+s.scenario_id).permutation(index)
                altered.append(replace(s, observations=tuple(replace(s.observations[int(i)], period=t+1)
                                                             for t, i in enumerate(index))))
            names = ['equal_annual', 'balanced_annual', 'plan_a_annual', 'plan_a_balanced',
                     'plan_a_no_review', 'plan_a_conservative', 'plan_a_growth']
            metrics, _ = evaluate(altered, fit['model'], fit['params'], config, names=names,
                                  horizons=[30], fold=fit['fold'], keep_records=False)
            metrics['order'] = order
            stress.append(metrics)
            if order == 'original':
                for product in PRODUCTS:
                    name='plan_a_'+product; params=fit['params'][name]
                    entry=fit['model']['plan_a']['entries'][f"{product}:30:{params['risk_ratio']:g}"]
                    training_drawdowns=[]
                    for sid in split['train_ids']:
                        p=PlanAPolicy(name,fit['model'],params,config,30)
                        r=simulate(by_id[sid],p,p.initial_target,SimulationConfig(30,config['initial_wealth'],config['cost_rate']))
                        training_drawdowns.append(compute_metrics(r)['real_max_drawdown'])
                    sample=np.array([[(1+np.array(o.nominal_returns))/(1+o.inflation)-1
                                      for o in s.observations] for s in valid])
                    annual=annual_statistics(sample,np.array(entry['weights']),config['cost_rate'],config['plan_a']['alpha'])
                    train_risk=expected_shortfall(-np.log1p(-np.array(training_drawdowns)),config['plan_a']['alpha'])
                    valid_risk=expected_shortfall(-np.log1p(-metrics[metrics.policy==name].real_max_drawdown.to_numpy()),config['plan_a']['alpha'])
                    decomposition.append(dict(fold=fit['fold'],policy=name,horizon=30,
                        training_budget=entry['risk_budget'],validation_budget=entry.get('validation_budget',entry['risk_budget']),
                        train_strategic_risk=entry['risk'],train_annual_fee_risk=entry['annual_with_fees']['log_drawdown_es'],
                        train_execution_risk=train_risk,validation_annual_fee_risk=annual['log_drawdown_es'],
                        validation_execution_risk=valid_risk,
                        training_execution_gap=train_risk-entry['annual_with_fees']['log_drawdown_es'],
                        generalization_gap=valid_risk-train_risk))
        print('Diagnosed fold', fit['fold'] + 1, flush=True)
    pd.DataFrame(constraints).to_csv(args.out / 'selected_constraints.csv', index=False)
    pd.DataFrame(decomposition).to_csv(args.out / 'risk_decomposition.csv', index=False)
    pd.DataFrame(floors).to_csv(args.out / 'training_risk_floors.csv', index=False)
    stress = pd.concat(stress, ignore_index=True)
    stress.to_csv(args.out / 'order_stress_metrics.csv', index=False)
    pd.concat([summarize(g).assign(order=order) for order, g in stress.groupby('order')],
              ignore_index=True).to_csv(args.out / 'order_stress_summary.csv', index=False)
    original = pd.read_csv(args.run / 'scenario_metrics.csv')
    cols = ['policy', 'scenario_id', 'horizon', 'cost_rate', 'real_cagr', 'real_max_drawdown', 'trade_rate']
    keys = cols[:4]
    baseline = stress[stress.order == 'original'][cols].sort_values(keys).reset_index(drop=True)
    original = original.merge(baseline[keys], on=keys, validate='one_to_one')[cols].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(baseline, original, atol=1e-12, rtol=1e-12)
    dump(args.out / 'verification.json', dict(all_checks_passed=True, reproduced_selections=15,
         original_metrics_reproduced=len(baseline), stress_metric_rows=len(stress), holdout_used=False,
         note='Counterfactual annual order stress is a mechanism diagnostic, not an independent sample or probability estimate.'))


if __name__ == '__main__':
    main()
