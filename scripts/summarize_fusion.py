"""Describe saved fusion results with existing screens; never select a new policy."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description='从已保存融合结果生成数值诊断表')
    parser.add_argument('run',type=Path)
    args=parser.parse_args(); run=args.run
    destinations=[run/'screen.csv',run/'action_counts.csv']
    if any(p.exists() for p in destinations): raise FileExistsError('Analysis output exists')
    frame=pd.read_csv(run/'scenario_metrics.csv')
    comparisons=pd.read_csv(run/'paired_comparisons.csv')
    manifest=json.loads((run/'manifest.json').read_text())
    names=manifest['settings']['policies']; rows=[]
    for name in names:
        for ref in ['plan_a_balanced','fusion_mpc','equal_annual','balanced_annual']:
            if name==ref: continue
            for h in manifest['settings']['horizons']:
                table=comparisons[(comparisons.policy==name)&(comparisons.reference==ref)&
                    (comparisons.horizon==h)&(comparisons.cost_rate==.0025)].set_index('metric')
                growth,p10,loss=[table.loc[k] for k in ('median_growth','p10_growth','p90_loss')]
                sub=frame[(frame.horizon==h)&(frame.cost_rate==.0025)]
                gf=rf=0
                for _,g in sub.groupby('fold'):
                    a=g[g.policy==name]; b=g[g.policy==ref]
                    gf+=int(a.real_cagr.median()>b.real_cagr.median())
                    rf+=int(a.real_max_drawdown.quantile(.9)<b.real_max_drawdown.quantile(.9))
                high=frame[(frame.horizon==h)&(frame.cost_rate==.005)]
                a=high[high.policy==name]; b=high[high.policy==ref]
                dg=a.real_cagr.median()-b.real_cagr.median()
                dr=a.real_max_drawdown.quantile(.9)-b.real_max_drawdown.quantile(.9)
                review_ok=sub[sub.policy==name].review.mean()<=.1
                growth_checks=dict(gain=growth.difference>=.001,positive_ci=growth.family_lower>0,
                    p10=p10.family_lower>=-.001,p90=loss.family_upper<=.01,folds=gf>=4,
                    doubled_cost=dg>0,review=review_ok)
                risk_checks=dict(reduction=loss.difference<=-.03,negative_ci=loss.family_upper<0,
                    growth=growth.family_lower>=-.005,p10=p10.family_lower>=-.002,folds=rf>=4,
                    doubled_risk=dr<0,doubled_growth=dg>=-.005,review=review_ok)
                rows.append(dict(policy=name,reference=ref,horizon=h,
                    growth_difference=growth.difference,growth_lower=growth.family_lower,growth_upper=growth.family_upper,
                    p90_difference=loss.difference,p90_lower=loss.family_lower,p90_upper=loss.family_upper,
                    growth_positive_folds=gf,risk_positive_folds=rf,doubled_growth_difference=dg,doubled_p90_difference=dr,
                    growth_screen=all(growth_checks.values()),risk_screen=all(risk_checks.values()),
                    growth_checks=json.dumps({k:bool(v) for k,v in growth_checks.items()}),
                    risk_checks=json.dumps({k:bool(v) for k,v in risk_checks.items()})))
    pd.DataFrame(rows).to_csv(destinations[0],index=False)
    ledger=pd.read_csv(run/'decisions.csv.gz'); actions=[]
    for (name,h,cost),g in ledger[ledger.policy.isin(names)].groupby(['policy','simulation_horizon','cost_rate']):
        after=g[g.period>1]; counts={i:0 for i in range(5)}
        for raw in after.policy_diagnostics.dropna():
            diag=json.loads(raw)
            if diag is not None: counts[diag['chosen']]+=1
        actions.append(dict(policy=name,horizon=h,cost_rate=cost,review_periods=len(after),
            **{f'choice_{k}':v for k,v in counts.items()},
            cai_choice_rate=counts[4]/len(after)))
    pd.DataFrame(actions).to_csv(destinations[1],index=False)
    print(pd.DataFrame(rows).query("horizon==30 and reference=='plan_a_balanced'").to_string(index=False))


if __name__=='__main__': main()
