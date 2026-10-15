"""Predeclared paired ablations for the lifecycle budget research candidate."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import binomial_upper

CANDIDATE='budget_lifecycle_balanced'
REFERENCES=[CANDIDATE+'_static',CANDIDATE+'_annual',CANDIDATE+'_hold']


def incremental_analysis(out, config):
    out=Path(out)
    metrics=pd.read_csv(out/'scenario_metrics.csv')
    if CANDIDATE not in set(metrics.policy):
        return
    spec=config['budget_value_screen']
    primary=metrics[np.isclose(metrics.cost_rate,config['cost_rate'])]
    horizons=sorted(primary[primary.policy==CANDIDATE].horizon.unique())
    family_size=len(horizons)*len(REFERENCES)
    tail=(1-spec['confidence'])/(2*family_size)
    rng=np.random.default_rng(config['seed']+704)
    comparisons=[]
    screens=[]
    bands=[]
    endpoints=[('real_cagr',.5,'median_growth'),('real_cagr',.1,'p10_growth'),
               ('real_max_drawdown',.9,'p90_loss')]
    for horizon in horizons:
        block=primary[primary.horizon==horizon]
        candidate=block[block.policy==CANDIDATE].set_index('scenario_id').sort_index()
        n=len(candidate)
        indices=rng.integers(0,n,(config['bootstrap'],n))
        for reference in REFERENCES:
            ref=block[block.policy==reference].set_index('scenario_id').sort_index()
            if not candidate.index.equals(ref.index):
                raise ValueError('Incremental comparisons require exactly paired scenarios')
            estimates={}
            for column,q,label in endpoints:
                a=candidate[column].to_numpy(); b=ref[column].to_numpy()
                boot=np.quantile(a[indices],q,axis=1)-np.quantile(b[indices],q,axis=1)
                row=dict(policy=CANDIDATE,reference=reference,horizon=int(horizon),
                         cost_rate=config['cost_rate'],metric=label,n=n,
                         difference=float(np.quantile(a,q)-np.quantile(b,q)),
                         lower95=float(np.quantile(boot,.025)),upper95=float(np.quantile(boot,.975)),
                         family_lower=float(np.quantile(boot,tail)),
                         family_upper=float(np.quantile(boot,1-tail)),family_size=family_size,
                         bootstrap=config['bootstrap'])
                comparisons.append(row)
                estimates[label]=row
            folds=[]
            for fold in sorted(candidate.fold.unique()):
                a=candidate[candidate.fold==fold].real_max_drawdown
                b=ref[ref.fold==fold].real_max_drawdown
                folds.append(float(a.quantile(.9)-b.quantile(.9)))
            consistent=sum(value<0 for value in folds)
            growth_ok=estimates['median_growth']['family_lower']>=-spec['maximum_median_growth_loss']
            p10_ok=estimates['p10_growth']['family_lower']>=-spec['maximum_p10_growth_loss']
            risk_ok=(estimates['p90_loss']['difference']<=-spec['minimum_p90_reduction']
                     and estimates['p90_loss']['family_upper']<0)
            double_cost_ok=True
            if int(horizon)==30:
                robust=metrics[(metrics.horizon==30)&np.isclose(metrics.cost_rate,2*config['cost_rate'])]
                a=robust[robust.policy==CANDIDATE].set_index('scenario_id').sort_index()
                b=robust[robust.policy==reference].set_index('scenario_id').sort_index()
                if not a.index.equals(candidate.index) or not b.index.equals(candidate.index):
                    raise ValueError('Missing paired double-cost robustness scenarios')
                double_cost_ok=bool(
                    a.real_cagr.median()-b.real_cagr.median()>=-spec['maximum_median_growth_loss']
                    and a.real_cagr.quantile(.1)-b.real_cagr.quantile(.1)>=-spec['maximum_p10_growth_loss']
                    and a.real_max_drawdown.quantile(.9)-b.real_max_drawdown.quantile(.9)<=-spec['minimum_p90_reduction'])
            breaches=int((candidate.real_max_drawdown>.25).sum())
            upper=binomial_upper(breaches,n,spec['confidence'])
            review=float(candidate.review.mean())
            service_ok=bool(upper<=spec['maximum_breach_upper'] and review<=spec['maximum_review_rate'])
            passed=bool(growth_ok and p10_ok and risk_ok and consistent>=spec['minimum_consistent_folds'] and double_cost_ok)
            screens.append(dict(policy=CANDIDATE,reference=reference,horizon=int(horizon),
                risk_reduction_pass=bool(risk_ok),median_growth_noninferior=bool(growth_ok),
                p10_growth_noninferior=bool(p10_ok),risk_consistent_folds=consistent,
                fold_p90_differences=json.dumps(folds),double_cost_checked=int(horizon)==30,
                double_cost_pass=bool(double_cost_ok) if int(horizon)==30 else None,
                incremental_value_established=passed,breaches=breaches,n=n,breach_upper=float(upper),
                review_rate=review,service_screen_pass=service_ok,
                decision='research_screen_pass' if passed else 'not_established'))
        zero=block[block.policy==CANDIDATE+'_zero_band'].set_index('scenario_id').sort_index()
        if not zero.index.equals(candidate.index):
            raise ValueError('Missing zero-band paired ablation')
        bands.append(dict(horizon=int(horizon),cost_rate=config['cost_rate'],n=n,
            trade_rate_difference=float(candidate.trade_rate.mean()-zero.trade_rate.mean()),
            mean_real_cost_difference=float(candidate.total_real_cost.mean()-zero.total_real_cost.mean()),
            median_growth_difference=float(candidate.real_cagr.median()-zero.real_cagr.median()),
            p90_loss_difference=float(candidate.real_max_drawdown.quantile(.9)-zero.real_max_drawdown.quantile(.9))))
    pd.DataFrame(comparisons).to_csv(out/'incremental_comparisons.csv',index=False)
    screen_frame=pd.DataFrame(screens)
    screen_frame.to_csv(out/'incremental_screen.csv',index=False)
    conclusions=[]
    for horizon,group in screen_frame.groupby('horizon'):
        conclusions.append(dict(horizon=int(horizon),
            matched_references=len(group),
            all_matched_references_pass=bool(group.incremental_value_established.all()),
            service_screen_pass=bool(group.service_screen_pass.all()),
            decision='research_screen_pass' if group.incremental_value_established.all() else 'not_established'))
    (out/'incremental_conclusion.json').write_text(json.dumps(conclusions,indent=2)+'\n',encoding='utf-8')
    pd.DataFrame(bands).to_csv(out/'band_ablation.csv',index=False)
    metadata=dict(family_size=family_size,confidence=spec['confidence'],bootstrap=config['bootstrap'],
        family_scope='Three matched references across all declared horizons; separately for each endpoint.',
        scope='Exploratory conditional OOF scenario bootstrap; overlapping training folds and repeated development reuse remain.',
        deployment_note='Research risk reduction does not imply customer suitability or permission to bypass a benefit gate.',
        holdout_used=False)
    (out/'incremental_protocol.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
