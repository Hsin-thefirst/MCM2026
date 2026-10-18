"""Fixed efficiency endpoints plus conditional paired uncertainty from saved OOF paths."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description='迭代开发数值核验，不选最终模型')
    parser.add_argument('run',type=Path)
    args=parser.parse_args();run=args.run
    if (run/'efficiency_screen.csv').exists():raise FileExistsError('Existing analysis')
    manifest=json.loads((run/'manifest.json').read_text());settings=manifest['settings']
    m=pd.read_csv(run/'scenario_metrics.csv');pairs=pd.read_csv(run/'paired_comparisons.csv')
    rng=np.random.default_rng(settings['seed']);idx=rng.integers(0,120,(settings['bootstrap'],120))
    family_tail=.05/(2*max(1,len(settings['rules'])))
    rows=[]
    for name in settings['rules']:
        for ref in ['fusion_drawdown','plan_a_balanced']:
            for h in settings['horizons']:
                block=m[(m.horizon==h)&(m.cost_rate==.0025)]
                a=block[block.policy==name].sort_values('scenario_id');b=block[block.policy==ref].sort_values('scenario_id')
                assert a.scenario_id.tolist()==b.scenario_id.tolist()==list(range(1,121))
                ca=a.total_real_cost.to_numpy();cb=b.total_real_cost.to_numpy()
                ratios=ca[idx].mean(axis=1)/cb[idx].mean(axis=1)-1
                cost_ratio=ca.mean()/cb.mean()-1
                table=pairs[(pairs.policy==name)&(pairs.reference==ref)&(pairs.horizon==h)&(pairs.cost_rate==.0025)].set_index('metric')
                growth,p10,risk=[table.loc[x] for x in ['median_growth','p10_growth','p90_loss']]
                fold_cost=fold_growth=fold_risk=0
                for fold in range(5):
                    af=a[a.fold==fold];bf=b[b.fold==fold]
                    fold_cost+=int(af.total_real_cost.mean()<bf.total_real_cost.mean())
                    fold_growth+=int(af.real_cagr.median()>bf.real_cagr.median())
                    fold_risk+=int(af.real_max_drawdown.quantile(.9)<bf.real_max_drawdown.quantile(.9))
                high=m[(m.horizon==h)&(m.cost_rate==.005)];ah=high[high.policy==name];bh=high[high.policy==ref]
                high_g=ah.real_cagr.median()-bh.real_cagr.median();high_r=ah.real_max_drawdown.quantile(.9)-bh.real_max_drawdown.quantile(.9)
                high_c=ah.total_real_cost.mean()/bh.total_real_cost.mean()-1
                point_pass=bool(cost_ratio<=-.3 and growth.difference>=-.001 and risk.difference<=.01 and fold_cost>=4 and high_c<=-.3 and high_g>=-.001 and high_r<=.01)
                uncertainty_pass=bool(growth.family_lower>=-.001 and p10.family_lower>=-.002 and risk.family_upper<=.01 and np.quantile(ratios,1-family_tail)<=-.3)
                review_ok=a.review.mean()<=.1
                growth_pass=bool(growth.difference>=.001 and growth.family_lower>0 and p10.family_lower>=-.001 and risk.family_upper<=.01 and fold_growth>=4 and high_g>0 and review_ok)
                risk_pass=bool(risk.difference<=-.03 and risk.family_upper<0 and growth.family_lower>=-.005 and p10.family_lower>=-.002 and fold_risk>=4 and high_r<0 and high_g>=-.005 and review_ok)
                rows.append(dict(policy=name,reference=ref,horizon=h,cost_change=cost_ratio,cost_lower=np.quantile(ratios,family_tail),cost_upper=np.quantile(ratios,1-family_tail),
                    growth_change=growth.difference,growth_lower=growth.family_lower,growth_upper=growth.family_upper,
                    p10_change=p10.difference,p10_lower=p10.family_lower,p10_upper=p10.family_upper,
                    p90_change=risk.difference,p90_lower=risk.family_lower,p90_upper=risk.family_upper,
                    cheaper_folds=fold_cost,growth_positive_folds=fold_growth,risk_positive_folds=fold_risk,
                    doubled_cost_change=high_c,doubled_growth_change=high_g,doubled_p90_change=high_r,
                    efficiency_point_pass=point_pass,efficiency_interval_pass=point_pass and uncertainty_pass,
                    growth_screen=growth_pass,risk_screen=risk_pass))
    screen=pd.DataFrame(rows);screen.to_csv(run/'efficiency_screen.csv',index=False)
    print(screen.query("horizon==30 and reference=='fusion_drawdown'").to_string(index=False))


if __name__=='__main__':main()
