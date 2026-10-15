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

def exempt_direction_violations(records):
    """Gate-exempt trades (period>1) must strictly reduce equity exposure; the
    benefit-gate exemption is only valid for protective de-risking."""
    r=records
    trading=(r.action.isin(['rebalance','change_target']) & r.gate_exempt.astype(bool)
             & (r.period>1)).to_numpy()
    if not trading.any(): return []
    pre=r.loc[trading,['pre_trade_holdings_equity','pre_trade_holdings_bond','pre_trade_holdings_bill']].to_numpy()
    target_equity=r.loc[trading,'target_weights_equity'].to_numpy()
    current_equity=pre[:,0]/pre.sum(axis=1)
    return [int(i) for i in np.where(target_equity>=current_equity-1e-12)[0]]

def _audit_benefit_gates(records, config, out):
    """Check saved gate evidence without imposing a gate on diagnostic comparators."""
    fitted = {}
    if (out/'fitted_models.json').exists():
        fitted = {item['fold']: item['params'] for item in
                  json.loads((out/'fitted_models.json').read_text())}
    frozen = None
    if (out/'frozen_bundle.json').exists():
        frozen = json.loads((out/'frozen_bundle.json').read_text())['params']

    # Estimates and explicit flags identify new policies as well as legacy gates.
    # Missing estimates on one trade cannot hide it if another row identifies the policy.
    evidence = records[['estimated_benefit', 'estimated_cost']].notna().any(axis=1) | records.gate_exempt
    if 'requires_benefit_gate' in records:
        evidence |= records.requires_benefit_gate.fillna(False).astype(bool)
    # Do not infer gate membership from historical policy names: old artifacts do
    # not carry enough evidence to verify their buffer, and diagnostics must remain
    # outside this check. New runs identify the gate explicitly per decision row.
    gate_policies = set(records.loc[evidence, 'policy'])
    checked = cost_checked = exempt = missing_buffer = unclassified = 0
    for row in records.itertuples():
        if row.period <= 1 or row.action not in ('rebalance', 'change_target'):
            continue
        if row.policy not in gate_policies:
            unclassified += 1
            continue
        explicit = getattr(row, 'requires_benefit_gate', None)
        if explicit is not None and not pd.isna(explicit) and not bool(explicit):
            unclassified += 1
            continue
        actual_cost = row.transaction_cost / sum(getattr(row, 'pre_trade_holdings_'+a)
                                                for a in ('equity', 'bond', 'bill'))
        label = (row.policy, row.scenario_id, row.horizon, row.period, row.cost_rate)
        assert np.isfinite(row.estimated_cost) and np.isclose(
            row.estimated_cost, actual_cost, atol=1e-10, rtol=0), f'Gate cost estimate mismatch: {label}'
        cost_checked += 1
        if row.gate_exempt:
            exempt += 1
            continue
        assert np.isfinite(row.estimated_benefit), f'Missing or invalid gate benefit: {label}'

        selected = fitted.get(getattr(row, 'fold', None), frozen)
        if selected is None:
            selected = config.get('params')
        params = selected.get(row.policy) if selected is not None else None
        # evaluate() uses the first general-grid entry only when a saved selection
        # explicitly omits this policy. Never substitute it for missing CV artifacts.
        if params is None and selected is not None:
            params = config.get('tuning_grid', [{}])[0]
        buffer = next((getattr(row, field) for field in ('gate_buffer', 'buffer')
                       if pd.notna(getattr(row, field, None))), None)
        if buffer is None:
            buffer = (params or {}).get('buffer', config.get('buffer'))
        if buffer is None:
            # A necessary, not sufficient, condition for a nonnegative safety margin.
            missing_buffer += 1
            buffer = 0.
        assert np.isfinite(buffer) and buffer >= 0, f'Invalid gate buffer: {label}'
        assert row.estimated_benefit > actual_cost + buffer, f'Trade fails benefit/cost gate: {label}'
        checked += 1
    return dict(gate_cost_checked_trades=cost_checked, gate_benefit_checked_trades=checked,
                gate_full_buffer_checked_trades=checked-missing_buffer,
                gate_exempt_checked_trades=exempt, gate_missing_buffer_trades=missing_buffer,
                gate_unclassified_or_diagnostic_trades=unclassified,
                gate_check_limitations=(
                    'Gate membership uses saved estimates, exemptions or requires_benefit_gate; '
                    'trades without gate evidence are not assumed to be gated. '
                    'For missing buffers only benefit > independently audited cost is checked '
                    '(necessary for nonnegative buffers); the full safety margin is not verified.'))


def audit_outputs(out):
    """Recompute wealth and metrics with independent scalar recurrences for all saved paths."""
    out=Path(out); c=json.loads((out/'config.json').read_text())
    manifest=json.loads((out/'manifest.json').read_text())
    m=pd.read_csv(out/'scenario_metrics.csv'); r=pd.read_csv(out/'decisions.csv.gz')
    expected=set(manifest['scenario_ids'])
    assert set(m.scenario_id)==expected
    assert not m.duplicated(['policy','scenario_id','horizon','cost_rate']).any()
    assert not r.duplicated(['policy','scenario_id','horizon','period','cost_rate']).any()
    assert not m[['policy','scenario_id','horizon','cost_rate']].isna().any().any()
    assert not r[['policy','scenario_id','horizon','period','cost_rate']].isna().any().any()
    for key,g in m.groupby(['policy','horizon','cost_rate']):
        assert set(g.scenario_id)==expected, f'Missing scenario metrics: {key}'
    if 'horizons' in c and 'cost_rate' in c:
        primary=m[np.isclose(m.cost_rate,c['cost_rate'])]
        for policy in primary.policy.unique():
            assert set(primary.loc[primary.policy==policy,'horizon'])==set(c['horizons']), f'Missing configured metric horizons: {policy}'
    audit_paths=0; maximum_error=0.
    assets=['equity','bond','bill']
    r=r.sort_values(['policy','scenario_id','horizon','cost_rate','period']).reset_index(drop=True)
    groups=['policy','scenario_id','horizon','cost_rate']
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
    exempt_violations=exempt_direction_violations(r)
    assert not exempt_violations, f'Gate-exempt trades that do not reduce equity exposure: {exempt_violations[:10]}'
    gate_checks=_audit_benefit_gates(r,c,out)
    paths=dict(tuple(r.groupby(groups)))
    path_horizons={key:set(g.horizon) for key,g in r.groupby(['policy','scenario_id','cost_rate'])}
    independent_policies={key[0] for key,hs in path_horizons.items() if len(hs)>1}
    # Legacy lifecycle paths are independent even if all but one horizon went missing.
    independent_policies.update(p for p in m.policy.unique()
                                if p.startswith(('drawdown_lifecycle_', 'budget_lifecycle_')))
    metrics_by_path={key:{} for key in paths}
    for metric in m.itertuples():
        key=(metric.policy,metric.scenario_id,metric.horizon,metric.cost_rate)
        if key not in paths:
            base=(metric.policy,metric.scenario_id,metric.cost_rate)
            hs=path_horizons.get(base,set())
            assert metric.policy not in independent_policies and len(hs)==1, f'Missing metric path: {key}'
            saved_horizon=next(iter(hs))
            assert 1<=metric.horizon<=saved_horizon, f'Missing metric path: {key}'
            key=(metric.policy,metric.scenario_id,saved_horizon,metric.cost_rate)
        metrics_by_path[key][metric.horizon]=metric
    metric_coverage=np.zeros(len(m),dtype=int)
    for (policy,sid,horizon_val,cost),g in paths.items():
        key=(policy,sid,horizon_val,cost)
        assert g.period.tolist()==list(range(1,int(horizon_val)+1)), f'Incomplete record path: {key}'
        initial=c['initial_wealth']; peak=initial; mdd=0.; fees=0.; realfees=0.; infl=1.; wealths=[initial]; trades=0
        turnovers=[]; all_trades=0; review_first=0
        metric_by_h=metrics_by_path[key]
        assert horizon_val in metric_by_h, f'Missing terminal metric for record path: {key}'
        for row in g.itertuples():
            fee=row.transaction_cost
            moved=row.moved_amount>1e-10*initial
            all_trades+=int(moved)
            if row.period>1 and moved: trades+=1
            turnovers.append(row.moved_amount/sum(getattr(row,'pre_trade_holdings_'+a) for a in assets))
            if row.review_pending and not review_first: review_first=row.period
            fees+=fee; realfees+=fee/infl
            postreal=sum(getattr(row,'post_trade_holdings_'+a) for a in assets)/infl
            endreal=sum(getattr(row,'end_holdings_'+a) for a in assets)/row.cumulative_inflation
            for z in [postreal,endreal]:
                peak=max(peak,z); mdd=max(mdd,1-z/peak)
            wealths.append(endreal); infl=row.cumulative_inflation
            if row.period not in metric_by_h: continue
            metric=metric_by_h[row.period]
            growth=(endreal/initial)**(1/row.period)-1
            assert np.isfinite([metric.real_cagr,metric.real_max_drawdown,metric.terminal_real_wealth]).all(), f'Nonfinite metric: {key}, {row.period}'
            error=max(abs(growth-metric.real_cagr),abs(mdd-metric.real_max_drawdown),abs(endreal-metric.terminal_real_wealth))
            maximum_error=max(maximum_error,error)
            assert error<1e-8
            assert abs(metric.total_real_cost-realfees)<1e-8
            assert abs(metric.total_nominal_cost-fees)<1e-8
            assert abs(metric.trade_rate-trades/max(1,row.period-1))<1e-10
            annual=[wealths[i]/wealths[i-1]-1 for i in range(1,len(wealths))]
            vol=np.std(annual,ddof=1) if len(annual)>1 else 0
            assert abs(vol-metric.real_return_volatility)<1e-10
            # Independent fractional empirical-tail integration, without compute_metrics.
            losses=sorted((-x for x in annual),reverse=True)
            mass=len(losses)*.1
            tail=sum(x*max(0.,min(1.,mass-i)) for i,x in enumerate(losses))/mass
            extra=dict(trade_rate_including_initial=all_trades/row.period,
                       mean_turnover=float(np.mean(turnovers)),worst_real_return=min(annual),
                       annual_return_cvar90=tail,terminal_loss=float(endreal<initial),
                       review=float(bool(review_first)),review_first_period=review_first)
            for field,value in extra.items():
                assert abs(getattr(metric,field)-value)<1e-10, f'Metric mismatch: {key}, {row.period}, {field}'
            metric_coverage[metric.Index]+=1
        audit_paths+=1
    assert (metric_coverage==1).all(), 'Every metric row must be audited exactly once'
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
                audited_metric_rows=int(metric_coverage.sum()),
                maximum_numeric_discrepancy=maximum_error,all_checks_passed=True,**gate_checks,
                gate_exempt_trades=int((r.gate_exempt & r.action.isin(['rebalance','change_target'])).sum()),
                exempt_direction_violations=len(exempt_violations),
                scope='All saved accounting paths, metric recurrences, uniqueness, CV splits and outer CVaR feasibility; market returns checked separately in replay tests.')
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
