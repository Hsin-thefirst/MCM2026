"""Independent accounting audit and explicit exploratory promotion screens."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import paired_comparisons

def comparisons_and_screen(out,config):
    out=Path(out)
    metrics=pd.read_csv(out/'scenario_metrics.csv')
    main=metrics[(metrics.horizon==30)&np.isclose(metrics.cost_rate,config['cost_rate'])]
    all_pairs=[]; screens=[]
    candidates=['balanced_gate','cost_aware','adaptive','adaptive_review']
    if config.get('study_kind')=='scenario_efficiency':
        from .scenario_utility import CE_POLICY_NAMES
        candidates+=list(CE_POLICY_NAMES)
    if config.get('study_kind')=='counterfactual_fqi':
        from .learning_core import LEARNING_NAMES
        candidates+=list(LEARNING_NAMES)
    if config.get('study_kind')=='neural_path_control':
        from .neural_core import NEURAL_NAMES
        candidates+=list(NEURAL_NAMES)
    for ref in ['balanced_annual','balanced_hold']:
        pairs=paired_comparisons(main,config['seed'],config['bootstrap'],baseline=ref)
        pairs['reference']=ref
        all_pairs.append(pairs)
    official=pd.read_csv(out/'paired_comparisons.csv')
    official=official[(official.horizon==30)&np.isclose(official.cost_rate,config['cost_rate'])].copy()
    official['reference']='equal_annual'; all_pairs.append(official)
    pairs=pd.concat(all_pairs,ignore_index=True)
    pairs.to_csv(out/'matched_comparisons.csv',index=False)
    for ref in ['equal_annual','balanced_annual','balanced_hold']:
        for candidate in candidates:
            if candidate not in set(main.policy): continue
            table=pairs[(pairs.policy==candidate)&(pairs.reference==ref)].set_index('metric')
            growth=table.loc['median_growth']; p10=table.loc['p10_growth']; loss=table.loc['p90_loss']
            deltas=[]
            for _,g in main[main.policy.isin([ref,candidate])].groupby('fold'):
                med=g.groupby('policy').real_cagr.median()
                deltas.append(float(med[candidate]-med[ref]))
            high=metrics[(metrics.horizon==30)&np.isclose(metrics.cost_rate,.005)]
            highmed=high.groupby('policy').real_cagr.median()
            checks=dict(practical_growth=bool(growth.difference>=config['minimum_growth_improvement']),
                growth_lower_positive=bool(growth.family_lower>0),
                p10_noninferior=bool(p10.family_lower>=-.001),
                p90_noninferior=bool(loss.family_upper<=config['maximum_p90_loss_increase']),
                fold_stability=bool(sum(d>0 for d in deltas)>=4),
                doubled_cost_growth=bool(highmed[candidate]>highmed[ref]))
            screens.append(dict(policy=candidate,reference=ref,**checks,
                positive_folds=sum(d>0 for d in deltas),status='investigate_for_promotion' if all(checks.values()) else 'not_established',
                note='Exploratory growth-route screen only; service suitability and independent final evaluation still required.'))
    pd.DataFrame(screens).to_csv(out/'promotion_screen.csv',index=False)

def audit_outputs(out):
    """Recompute wealth and metrics with independent scalar recurrences for all saved paths."""
    out=Path(out); c=json.loads((out/'config.json').read_text())
    manifest=json.loads((out/'manifest.json').read_text())
    m=pd.read_csv(out/'scenario_metrics.csv'); r=pd.read_csv(out/'decisions.csv.gz')
    expected=set(manifest['scenario_ids'])
    assert set(m.scenario_id)==expected
    assert not m.duplicated(['policy','scenario_id','horizon','cost_rate']).any()
    conditioned='evaluation_horizon' in r.columns
    path_keys=['policy','scenario_id','cost_rate']+(['evaluation_horizon'] if conditioned else [])
    assert not r.duplicated(path_keys+['period']).any()
    audit_paths=0; maximum_error=0.
    assets=['equity','bond','bill']
    r=r.sort_values(path_keys+['period']).reset_index(drop=True)
    groups=path_keys
    pre=r[['pre_trade_holdings_'+a for a in assets]].to_numpy()
    post=r[['post_trade_holdings_'+a for a in assets]].to_numpy()
    end=r[['end_holdings_'+a for a in assets]].to_numpy()
    target=r[['target_weights_'+a for a in assets]].to_numpy()
    previous=r.groupby(groups)[['end_holdings_'+a for a in assets]].shift().to_numpy()
    previous[r.period.to_numpy()==1]=[0,0,c['initial_wealth']]
    np.testing.assert_allclose(pre,previous,atol=1e-9,rtol=1e-10)
    assert min(pre.min(),post.min(),end.min())>=-1e-10
    np.testing.assert_allclose(target.sum(axis=1),1,atol=1e-9)
    trading=r.action.isin(['rebalance','change_target']).to_numpy()
    q=np.where(trading,np.abs(pre.sum(axis=1)[:,None]*target-pre).sum(axis=1)/2,0)
    fees=q*r.cost_rate.to_numpy()
    expected_post=np.where(trading[:,None],(pre.sum(axis=1)-fees)[:,None]*target,pre)
    np.testing.assert_allclose(post,expected_post,atol=1e-9)
    np.testing.assert_allclose(fees,r.transaction_cost,atol=1e-9)
    np.testing.assert_allclose(q,r.moved_amount,atol=1e-8)
    np.testing.assert_allclose(post.sum(axis=1)+fees,pre.sum(axis=1),atol=1e-8)
    metric_groups=['policy','scenario_id','cost_rate']+(['horizon'] if conditioned else [])
    lookup={key:g for key,g in m.groupby(metric_groups)}
    for key,g in r.groupby(groups):
        policy,sid,cost=key[:3]
        initial=c['initial_wealth']; peak=initial; mdd=0.; fees=0.; realfees=0.; infl=1.; wealths=[initial]; trades=0
        metric_by_h={row.horizon:row for row in lookup[key].itertuples()}
        assert g.period.tolist()==list(range(1,(int(key[3]) if conditioned else int(g.period.max()))+1))
        for row in g.itertuples():
            fee=row.transaction_cost
            if row.period>1 and row.moved_amount>1e-10*initial: trades+=1
            fees+=fee; realfees+=fee/infl
            postreal=sum(getattr(row,'post_trade_holdings_'+a) for a in assets)/infl
            endreal=sum(getattr(row,'end_holdings_'+a) for a in assets)/row.cumulative_inflation
            for z in [postreal,endreal]:
                peak=max(peak,z); mdd=max(mdd,1-z/peak)
            wealths.append(endreal); infl=row.cumulative_inflation
            if row.period not in metric_by_h: continue
            metric=metric_by_h[row.period]
            growth=(endreal/initial)**(1/row.period)-1
            error=max(abs(growth-metric.real_cagr),abs(mdd-metric.real_max_drawdown),abs(endreal-metric.terminal_real_wealth))
            maximum_error=max(maximum_error,error)
            assert error<1e-8
            assert abs(metric.total_real_cost-realfees)<1e-8
            assert abs(metric.total_nominal_cost-fees)<1e-8
            assert abs(metric.trade_rate-trades/max(1,row.period-1))<1e-10
            annual=[wealths[i]/wealths[i-1]-1 for i in range(1,len(wealths))]
            vol=np.std(annual,ddof=1) if len(annual)>1 else 0
            assert abs(vol-metric.real_return_volatility)<1e-10
        audit_paths+=1
    if (out/'splits.json').exists():
        splits=json.loads((out/'splits.json').read_text()); seen=[]
        for f in splits:
            train=set(f['train_ids']); valid=set(f['validation_ids']); seen.extend(valid)
            assert train.isdisjoint(valid) and train|valid==expected
            inner_seen=[]
            for inner in f['inner']:
                it=set(inner['train_ids']); iv=set(inner['validation_ids']); inner_seen.extend(iv)
                assert it.isdisjoint(iv) and it|iv==train and (it|iv).isdisjoint(valid)
            assert sorted(inner_seen)==sorted(train)
        assert sorted(seen)==sorted(expected)
        fitted=json.loads((out/'fitted_models.json').read_text())
        assert all(f['model']['cvar_feasible'] and f['model']['cvar_estimate']<=c['cvar_limit']+1e-10 for f in fitted)
    result=dict(phase=manifest['phase'],audited_paths=audit_paths,metric_rows=len(m),decision_rows=len(r),
                maximum_numeric_discrepancy=maximum_error,all_checks_passed=True,
                scope='All saved accounting paths, metric recurrences, uniqueness, CV splits and outer CVaR feasibility; market returns checked separately in replay tests.')
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
