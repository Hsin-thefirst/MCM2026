"""Predeclared matched development comparisons for signal and planning extensions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import paired_comparisons


def extension_comparisons(out, config):
    out = Path(out)
    metrics = pd.read_csv(out/'scenario_metrics.csv')
    main = metrics[(metrics.horizon==30)&np.isclose(metrics.cost_rate,config['cost_rate'])]
    names = set(main.policy)
    comparisons = []
    if 'plan_b_narrow' in names:
        comparisons += [(f'plan_b_{f}',f'plan_b_{f}_0',True) for f in ('narrow','wide')]
        comparisons += [(p,'plan_a_balanced',False) for p in sorted(names) if p.startswith('plan_b_')]
    if 'plan_c_l3' in names:
        comparisons += [('plan_c_l3','plan_c_l1',True),('plan_c_l3_no_risk','plan_c_l1_no_risk',False),
                        ('plan_c_l1','plan_a_balanced',False),('plan_c_l3','plan_a_balanced',False)]
    if not comparisons: return
    tables = {}
    for ref in sorted({r for _,r,_ in comparisons}):
        table=paired_comparisons(main,config['seed'],config['bootstrap'],baseline=ref)
        table['reference']=ref; tables[ref]=table
    pd.concat(tables.values(),ignore_index=True).to_csv(out/'extension_comparisons.csv',index=False)
    high=metrics[(metrics.horizon==30)&np.isclose(metrics.cost_rate,.005)]
    rows=[]; settings=config['plan_a']['risk_promotion']
    fitted=json.loads((out/'fitted_models.json').read_text())
    for candidate,ref,primary in comparisons:
        t=tables[ref][tables[ref].policy==candidate].set_index('metric')
        growth,p10,loss=[t.loc[k] for k in ('median_growth','p10_growth','p90_loss')]
        folds=[]; risk_folds=[]
        for _,g in main[main.policy.isin([candidate,ref])].groupby('fold'):
            g=g.set_index('policy')
            folds.append(g.loc[candidate,'real_cagr'].median()>g.loc[ref,'real_cagr'].median())
            risk_folds.append(g.loc[candidate,'real_max_drawdown'].quantile(.9)<g.loc[ref,'real_max_drawdown'].quantile(.9))
        hc=high[high.policy==candidate]; hr=high[high.policy==ref]
        inner_ok=all(f['params'].get(candidate,{}).get('signal_selection_feasible',True) for f in fitted)
        review_ok=bool(main[main.policy==candidate].review.mean()<=settings['maximum_review_rate'])
        growth_checks=[growth.difference>=config['minimum_growth_improvement'],growth.family_lower>0,
            p10.family_lower>=-.001,loss.family_upper<=config['maximum_p90_loss_increase'],
            sum(folds)>=4,hc.real_cagr.median()>hr.real_cagr.median(),review_ok,inner_ok]
        risk_checks=[loss.difference<=-settings['minimum_p90_reduction'],loss.family_upper<0,
            growth.family_lower>=-settings['maximum_growth_loss'],p10.family_lower>=-settings['maximum_p10_loss'],
            sum(risk_folds)>=4,hc.real_max_drawdown.quantile(.9)<hr.real_max_drawdown.quantile(.9),
            hc.real_cagr.median()>=hr.real_cagr.median()-settings['maximum_growth_loss'],review_ok,inner_ok]
        rows.append(dict(policy=candidate,reference=ref,primary=primary,
            growth_difference=growth.difference,p90_difference=loss.difference,
            growth_lower=growth.family_lower,growth_upper=growth.family_upper,
            p90_lower=loss.family_lower,p90_upper=loss.family_upper,
            positive_growth_folds=sum(folds),positive_risk_folds=sum(risk_folds),inner_feasible=inner_ok,
            review_ok=review_ok,growth_route=all(growth_checks),risk_route=all(risk_checks),
            growth_checks=json.dumps([bool(x) for x in growth_checks]),risk_checks=json.dumps([bool(x) for x in risk_checks]),
            status='investigate_for_promotion' if all(growth_checks) or all(risk_checks) else 'not_established'))
    pd.DataFrame(rows).to_csv(out/'extension_screen.csv',index=False)
