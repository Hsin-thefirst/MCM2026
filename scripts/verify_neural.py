"""Independent NumPy path simulation and model/constraint/deployment verification."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'benchmark'),str(ROOT/'src')]
NAMES=('neural_linear','neural_mlp','neural_portfolio')


def market_arrays(scenarios):
    nominal=np.array([[o.nominal_returns for o in s.observations] for s in scenarios],dtype=float)
    inflation=np.array([[o.inflation for o in s.observations] for s in scenarios],dtype=float)
    history=np.zeros((len(scenarios),30,6))
    real=(1+nominal)/(1+inflation[:,:,None])-1
    for t in range(1,30):
        equity=real[:,max(0,t-3):t,0]
        history[:,t,:]=np.column_stack((real[:,t-1,:],inflation[:,t-1],equity.mean(axis=1),equity.std(axis=1)))
    return dict(scenario_id=np.array([s.scenario_id for s in scenarios],dtype=np.int64),nominal=nominal,inflation=inflation,history=history)


def expected_normalizer(data,config):
    mean=np.array([.4,.3,.15,.25,0,0,0,0,0,0,config['cost_rate']*1.5,0,.5])
    scale=np.array([.25,.2,.15,.25,1,1,1,1,1,1,config['cost_rate'],1,.5])
    h=data['history'][:,1:,:].reshape(-1,6);mean[4:10]=h.mean(axis=0);scale[4:10]=np.maximum(h.std(axis=0),.01)
    return mean,scale


def targets(fit,checkpoint,x):
    z=np.clip((x-np.array(fit['mean']))/np.array(fit['scale']),-5,5)[:,fit['features']]
    values=[]
    for member in fit['checkpoints'][str(checkpoint)]['members']:
        value=z
        for i,layer in enumerate(member['layers']):
            value=value.dot(np.array(layer['weight']).T)+np.array(layer['bias'])
            if i+1<len(member['layers']):value=np.tanh(value)
        prob=np.exp(value-value.max(axis=1,keepdims=True));prob/=prob.sum(axis=1,keepdims=True)
        values.append(prob.dot(np.array(fit['targets'])))
    return np.mean(values,axis=0)


def numpy_rollout(fit,checkpoint,model,data,config,horizon,cost,reference=False):
    n=len(data['scenario_id']);h=np.zeros((n,3));h[:,2]=config['initial_wealth']
    peak=np.full(n,config['initial_wealth']);mdd=np.zeros(n);pending=np.zeros(n,dtype=bool)
    goal=np.tile([.5,.35,.15],(n,1));fees=np.zeros(n);trades=np.zeros(n)
    mu=np.array(model['mu']);cov=np.array(model['cov']);actions=[];benefits=[];fractions=[];goals=[];wealths=[]
    for t in range(horizon):
        v=h.sum(axis=1);p=h/v[:,None];dd=1-v/peak;old_pending=pending.copy()
        if t==0 or reference:
            proposed=np.tile([.5,.35,.15],(n,1));trade=np.ones(n,dtype=bool);benefit=np.full(n,np.nan)
        else:
            x=np.column_stack((p[:,:2],dd,mdd,data['history'][:,t,:],np.full(n,cost),pending,np.full(n,(horizon-t)/30)))
            proposed=targets(fit,checkpoint,x)
            uq=1+proposed.dot(mu)-config['neural_gate_gamma']*np.sum((proposed.dot(cov))*proposed,axis=1)/2
            up=1+p.dot(mu)-config['neural_gate_gamma']*np.sum((p.dot(cov))*p,axis=1)/2
            benefit=(uq-up)/uq
            fraction=cost*np.abs(proposed-p).sum(axis=1)/2
            pending|=(dd>=config['review_drawdown'])|(~np.isfinite(benefit))|(uq<=0)
            trade=(~pending)&(benefit>fraction+config['neural_buffer'])
        moved=np.abs(v[:,None]*proposed-h).sum(axis=1)/2
        fee=np.where(trade,cost*moved,0.)
        action=np.full(n,'keep',dtype=object)
        if t==0 or reference:action[trade]='rebalance'
        else:
            action[(~old_pending)&pending]='human_review'
            equal=np.max(np.abs(proposed-goal),axis=1)<=1e-10
            action[trade&equal]='rebalance';action[trade&~equal]='change_target'
        h=np.where(trade[:,None],(v-fee)[:,None]*proposed,h)
        goal=np.where(trade[:,None],proposed,goal)
        if t:trades+=trade&(moved>1e-10*config['initial_wealth'])
        fees+=fee;post=h.sum(axis=1);peak=np.maximum(peak,post);mdd=np.maximum(mdd,1-post/peak)
        h=h*(1+data['nominal'][:,t,:])/(1+data['inflation'][:,t,None])
        wealth=h.sum(axis=1);peak=np.maximum(peak,wealth);mdd=np.maximum(mdd,1-wealth/peak)
        actions.append(action);benefits.append(np.where(old_pending|pending,np.nan,benefit));fractions.append(fee/v);goals.append(goal.copy());wealths.append(wealth)
    return dict(wealth=wealth,log_growth=np.log(wealth/config['initial_wealth'])/horizon,mdd=mdd,
        total_real_cost=fees,trade_rate=trades/max(1,horizon-1),review=pending,
        actions=np.array(actions).T,benefits=np.array(benefits).T,fees=np.array(fractions).T,
        targets=np.stack(goals,axis=1),wealth_path=np.array(wealths).T)


def upper_tail(values,alpha=.9):
    ordered=sorted(map(float,values),reverse=True);mass=len(ordered)*(1-alpha);whole=int(np.floor(mass))
    return (sum(ordered[:whole])+(mass-whole)*ordered[whole])/mass


def verify_neural(out,scenarios):
    out=Path(out);manifest=json.loads((out/'manifest.json').read_text());config=json.loads((out/'config.json').read_text())
    if manifest['phase']!='development' or manifest['holdout_used'] or any(not 1<=i<=120 for i in manifest['scenario_ids']):
        raise PermissionError('Neural verification is development-only')
    by_id={s.scenario_id:s for s in scenarios};outer=json.loads((out/'fitted_models.json').read_text())
    splits=json.loads((out/'splits.json').read_text());datasets=constraints=networks=0;max_constraint_error=0.
    for fold in splits:
        number=fold['outer_fold'];fits=[(f'outer_{number}',outer[number]['model'],fold['train_ids'],fold['validation_ids'])]
        for inner in fold['inner']:
            stem=f"outer_{number}_inner_{inner['inner_fold']}"
            fits.append((stem,json.loads((out/'tuning_models'/(stem+'.json')).read_text()),inner['train_ids'],
                         inner['validation_ids']+fold['validation_ids']))
        for stem,model,train_ids,valid_ids in fits:
            assert set(train_ids).isdisjoint(valid_ids)
            with np.load(out/'training_data'/(stem+'.npz'),allow_pickle=False) as archive:data={key:archive[key] for key in archive.files}
            assert data['scenario_id'].tolist()==train_ids
            expected=market_arrays([by_id[i] for i in train_ids])
            for key in data:np.testing.assert_allclose(data[key],expected[key],atol=1e-14,rtol=1e-14)
            digest=hashlib.sha256()
            for key,array in sorted(data.items()):
                digest.update(key.encode());digest.update(str((array.shape,array.dtype.str)).encode());digest.update(np.ascontiguousarray(array).tobytes())
            center,scale=expected_normalizer(data,config)
            for name,fit in model['neural'].items():
                assert fit['dataset_sha256']==digest.hexdigest();assert sorted(fit['train_ids'])==sorted(train_ids)
                assert fit['seeds']==config['neural_seeds']
                np.testing.assert_allclose(fit['mean'],center,atol=1e-14);np.testing.assert_allclose(fit['scale'],scale,atol=1e-14)
                assert fit['features']==([0,1,2,3,10,11,12] if name=='neural_portfolio' else list(range(13)))
                for trace in fit['trace']:
                    assert np.isfinite([trace['loss'],trace['gradient_norm'],trace['risk_dual'],trace['growth_dual']]).all()
                    assert 0<=trace['risk_dual']<=config['neural_dual_cap'] and 0<=trace['growth_dual']<=config['neural_dual_cap']
                for checkpoint,ensemble in fit['checkpoints'].items():
                    assert len(ensemble['members'])==len(config['neural_seeds']);networks+=len(ensemble['members'])
                    for member in ensemble['members']:
                        count=0
                        for layer in member['layers']:
                            weight=np.array(layer['weight']);bias=np.array(layer['bias']);assert np.isfinite(weight).all() and np.isfinite(bias).all()
                            count+=weight.size+bias.size
                        assert count==fit['parameter_count']
                    for row in ensemble['training_constraints']:
                        value=numpy_rollout(fit,int(checkpoint),model,data,config,row['horizon'],row['cost_rate'])
                        ref=numpy_rollout(fit,int(checkpoint),model,data,config,row['horizon'],row['cost_rate'],reference=True)
                        risk=upper_tail(value['mdd']);weak=-upper_tail(-value['log_growth'])
                        risk_bound=upper_tail(ref['mdd'])-config['neural_risk_improvement']
                        weak_bound=-upper_tail(-ref['log_growth'])-config['neural_weak_growth_margin']
                        error=max(abs(risk-row['cvar_drawdown']),abs(weak-row['weak_log_growth']),
                                  abs(risk_bound-row['risk_bound']),abs(weak_bound-row['weak_growth_bound']))
                        assert error<1e-10;max_constraint_error=max(max_constraint_error,error)
                        assert row['feasible']==bool(risk<=risk_bound+1e-10 and weak>=weak_bound-1e-10)
                        constraints+=1
                    assert ensemble['training_feasible']==all(row['feasible'] for row in ensemble['training_constraints'])
            datasets+=1
    metrics=pd.read_csv(out/'scenario_metrics.csv',float_precision='round_trip')
    records=pd.read_csv(out/'decisions.csv.gz',float_precision='round_trip');decision_rows=gates=0;max_gate_error=max_metric_error=0.
    for fold in splits:
        f=fold['outer_fold'];wrapper=outer[f];model=wrapper['model'];ids=fold['validation_ids'];data=market_arrays([by_id[i] for i in ids])
        for name in NAMES:
            fit=model['neural'][name];checkpoint=wrapper['params'][name]['checkpoint']
            for cost,horizons in [(config['cost_rate'],config['horizons'])]+[(v,[30]) for v in config['sensitivity_costs']]:
                for horizon in horizons:
                    replay=numpy_rollout(fit,checkpoint,model,data,config,horizon,cost)
                    for slot,sid in enumerate(ids):
                        metric=metrics[(metrics.fold==f)&(metrics.policy==name)&(metrics.scenario_id==sid)&(metrics.horizon==horizon)&np.isclose(metrics.cost_rate,cost)].iloc[0]
                        error=max(abs(metric.terminal_real_wealth-replay['wealth'][slot]),abs(metric.real_max_drawdown-replay['mdd'][slot]),
                                  abs(metric.total_real_cost-replay['total_real_cost'][slot]),abs(metric.trade_rate-replay['trade_rate'][slot]))
                        assert error<1e-8;max_metric_error=max(max_metric_error,error)
                        group=records[(records.fold==f)&(records.policy==name)&(records.scenario_id==sid)&
                            (records.evaluation_horizon==horizon)&np.isclose(records.cost_rate,cost)].sort_values('period')
                        assert group.period.tolist()==list(range(1,horizon+1))
                        for row in group.itertuples():
                            t=row.period-1;assert row.action==replay['actions'][slot,t];decision_rows+=1
                            if row.period>1 and row.action in ('rebalance','change_target','keep') and not row.review_pending:
                                estimate=replay['benefits'][slot,t]
                                assert np.isfinite(estimate) and abs(row.estimated_benefit-estimate)<1e-10
                                max_gate_error=max(max_gate_error,abs(row.estimated_benefit-estimate));gates+=1
                            if row.action in ('rebalance','change_target'):
                                np.testing.assert_allclose([getattr(row,'target_weights_'+a) for a in ('equity','bond','bill')],replay['targets'][slot,t],atol=1e-12)
                                if row.period>1:assert row.estimated_benefit>row.estimated_cost+config['neural_buffer']
                                if row.period>1:assert abs(row.estimated_cost-replay['fees'][slot,t])<1e-10
    result=dict(all_checks_passed=True,holdout_used=False,training_datasets_checked=datasets,network_checkpoints_checked=networks,
        training_constraints_checked=constraints,decision_rows_checked=decision_rows,value_gates_checked=gates,
        maximum_training_constraint_error=max_constraint_error,maximum_metric_error=max_metric_error,maximum_gate_error=max_gate_error,
        verification_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Independent NumPy network, source history/normalization, path constraints and all selected horizon/cost deployment paths.')
    (out/'neural_replay.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    from safebench.data import load_partition
    out=Path(sys.argv[1])
    if json.loads((out/'manifest.json').read_text())['phase']!='development':raise PermissionError('Development only')
    scenarios,_=load_partition(ROOT/'data/SafeInvest_Synthetic_Data.csv')
    print(json.dumps(verify_neural(out,scenarios),indent=2))
