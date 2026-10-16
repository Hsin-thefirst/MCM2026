"""Independent accounting audit and explicit exploratory promotion screens."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import paired_comparisons
from .experts import EXPERT_NAMES, enabled
from .frontier import FRONTIER_NAMES, MATCHED_NAMES, enabled as frontier_enabled
from .cushion import CUSHION_NAMES, CALIBRATED_NAMES, enabled as cushion_enabled

def comparisons_and_screen(out,config):
    out=Path(out)
    metrics=pd.read_csv(out/'scenario_metrics.csv')
    main=metrics[(metrics.horizon==30)&np.isclose(metrics.cost_rate,config['cost_rate'])]
    all_pairs=[]; screens=[]
    candidates=['balanced_gate','cost_aware','adaptive','adaptive_review']+(list(EXPERT_NAMES) if enabled(config) else [])
    if frontier_enabled(config): candidates+=list(FRONTIER_NAMES)
    if cushion_enabled(config): candidates+=list(CUSHION_NAMES)
    if config.get('calibrated_research',{}).get('enabled',False): candidates+=list(CALIBRATED_NAMES)
    references=['equal_annual','balanced_annual','balanced_hold']
    if 'expert_greedy' in set(main.policy): references.append('expert_greedy')
    references += [n for n in ('frontier_annual','frontier_hold','frontier_gate') if n in set(main.policy)]
    references += [n for n in MATCHED_NAMES if n in set(main.policy)]
    if 'cushion_static' in set(main.policy): references.append('cushion_static')
    if 'calibrated_fixed' in set(main.policy): references.append('calibrated_fixed')
    for ref in references[1:]:
        pairs=paired_comparisons(main,config['seed'],config['bootstrap'],baseline=ref)
        pairs['reference']=ref
        all_pairs.append(pairs)
    official=pd.read_csv(out/'paired_comparisons.csv')
    official=official[(official.horizon==30)&np.isclose(official.cost_rate,config['cost_rate'])].copy()
    official['reference']='equal_annual'; all_pairs.append(official)
    pairs=pd.concat(all_pairs,ignore_index=True)
    pairs.to_csv(out/'matched_comparisons.csv',index=False)
    for ref in references:
        for candidate in candidates:
            if candidate not in set(main.policy) or candidate==ref: continue
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
    if enabled(config) or frontier_enabled(config) or cushion_enabled(config):
        risk_route_screen(metrics,pairs,config).to_csv(out/'risk_promotion_screen.csv',index=False)


def risk_route_screen(metrics,pairs,config):
    settings=dict(primary_reference='balanced_annual',minimum_p90_reduction=.03,
                  maximum_growth_loss=.005,maximum_p10_loss=.002)
    settings.update(config.get('expert_research',{})); rows=[]
    main=metrics[(metrics.horizon==30)&np.isclose(metrics.cost_rate,config['cost_rate'])]
    high=metrics[(metrics.horizon==30)&np.isclose(metrics.cost_rate,2*config['cost_rate'])]
    for ref in [settings['primary_reference'],'equal_annual','expert_greedy','frontier_annual','frontier_gate']:
        for name in EXPERT_NAMES+FRONTIER_NAMES+CUSHION_NAMES+CALIBRATED_NAMES:
            if name==ref or name not in set(main.policy) or ref not in set(main.policy): continue
            table=pairs[(pairs.policy==name)&(pairs.reference==ref)].set_index('metric')
            growth,p10,loss=[table.loc[x] for x in ['median_growth','p10_growth','p90_loss']]
            positive=0
            for _,g in main[main.policy.isin([ref,name])].groupby('fold'):
                tail=g.groupby('policy').real_max_drawdown.quantile(.9)
                positive+=int(tail[name]<tail[ref])
            g=high[high.policy==name]; r=high[high.policy==ref]
            candidate=main[main.policy==name]
            checks=dict(practical_risk_reduction=bool(loss.difference<=-settings['minimum_p90_reduction']),
                risk_upper_negative=bool(loss.family_upper<0),
                growth_noninferior=bool(growth.family_lower>=-settings['maximum_growth_loss']),
                p10_noninferior=bool(p10.family_lower>=-settings['maximum_p10_loss']),
                fold_stability=positive>=4,
                doubled_cost_robust=bool(len(g)>0 and len(r)>0 and
                    g.real_cagr.median()-r.real_cagr.median()>=-settings['maximum_growth_loss'] and
                    g.real_max_drawdown.quantile(.9)-r.real_max_drawdown.quantile(.9)<=-settings['minimum_p90_reduction']),
                no_execution_fallback=bool(candidate.fallback_count.sum()==0))
            rows.append(dict(policy=name,reference=ref,**checks,positive_folds=positive,
                growth_difference=growth.difference,p90_difference=loss.difference,
                status='investigate_for_promotion' if all(checks.values()) else 'not_established'))
    return pd.DataFrame(rows)

def audit_outputs(out):
    """Recompute wealth and metrics with independent scalar recurrences for all saved paths."""
    out=Path(out); c=json.loads((out/'config.json').read_text())
    manifest=json.loads((out/'manifest.json').read_text())
    m=pd.read_csv(out/'scenario_metrics.csv'); r=pd.read_csv(out/'decisions.csv.gz')
    expected=set(manifest['scenario_ids'])
    assert set(m.scenario_id)==expected
    assert not m.duplicated(['policy','scenario_id','horizon','cost_rate']).any()
    if 'evaluated_policies' in manifest:
        expected_grid={(p,s,h,round(c['cost_rate'],10)) for p in manifest['evaluated_policies']
                       for s in expected for h in c['horizons']}
        expected_grid|={(p,s,30,round(cost,10)) for p in manifest['evaluated_policies']
                        for s in expected for cost in c['sensitivity_costs']}
        actual_grid={(x.policy,int(x.scenario_id),int(x.horizon),round(x.cost_rate,10)) for x in m.itertuples()}
        assert actual_grid==expected_grid, 'Incomplete or unexpected evaluation grid'
    horizon_key=['simulation_horizon'] if 'simulation_horizon' in r else []
    assert not r.duplicated(['policy','scenario_id','period','cost_rate']+horizon_key).any()
    audit_paths=0; maximum_error=0.
    assets=['equity','bond','bill']
    groups=['policy','scenario_id','cost_rate']+horizon_key
    r=r.sort_values(groups+['period']).reset_index(drop=True)
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
    metric_frame=m.assign(simulation_horizon=m.horizon) if horizon_key else m
    lookup={key:g for key,g in metric_frame.groupby(groups)}
    assert set(lookup)==set(r.groupby(groups).groups), 'Metric/decision path mismatch'
    gated=r.policy.isin(['balanced_gate','cost_aware','adaptive','adaptive_review',*EXPERT_NAMES,
                        *FRONTIER_NAMES[2:],'frontier_matched_gate',*CUSHION_NAMES,*CALIBRATED_NAMES]) & (r.period>1) & trading
    buffers=np.zeros(len(r))
    buffers[r.policy.isin(EXPERT_NAMES)]=c.get('expert_research',{}).get('buffer',0.)
    buffers[r.policy.isin(FRONTIER_NAMES+MATCHED_NAMES)]=c.get('frontier_research',{}).get('buffer',0.)
    buffers[r.policy.isin(CUSHION_NAMES+CALIBRATED_NAMES)]=c.get('cushion_research',{}).get('buffer',0.)
    source=out/'fitted_models.json'
    if source.exists():
        for f in json.loads(source.read_text()):
            for name,params in f['params'].items():
                if name in ('balanced_gate','cost_aware','adaptive','adaptive_review'):
                    buffers[(r.policy==name)&(r.fold==f['fold'])]=params['buffer']
    elif (out/'frozen_bundle.json').exists():
        frozen=json.loads((out/'frozen_bundle.json').read_text())
        for name,params in frozen['params'].items():
            if name in ('balanced_gate','cost_aware','adaptive','adaptive_review'):
                buffers[r.policy==name]=params['buffer']
    assert (r.loc[gated,'estimated_benefit'].to_numpy()>r.loc[gated,'estimated_cost'].to_numpy()+buffers[gated]).all()
    np.testing.assert_allclose(r.loc[gated,'estimated_cost'],fees[gated]/pre[gated].sum(axis=1),atol=1e-10)
    for key,g in r.groupby(groups):
        assert g.period.tolist()==list(range(1,len(g)+1))
        if horizon_key: assert len(g)==key[-1]
        initial=c['initial_wealth']; peak=initial; mdd=0.; fees=0.; realfees=0.; infl=1.; wealths=[initial]; trades=0
        metric_by_h={row.horizon:row for row in lookup[key].itertuples()}
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
        for f in fitted:
            if 'experts' in f['model']:
                assert set(f['model']['experts']['train_ids'])==set(f['model']['train_ids'])
    result=dict(phase=manifest['phase'],audited_paths=audit_paths,metric_rows=len(m),decision_rows=len(r),
                maximum_numeric_discrepancy=maximum_error,all_checks_passed=True,
                scope='All saved accounting paths, metric recurrences, uniqueness, CV splits and outer CVaR feasibility; market returns checked separately in replay tests.')
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
