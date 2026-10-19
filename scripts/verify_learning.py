"""Independent scalar counterfactual/Bellman/decision replay; development partition only."""
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'benchmark'),str(ROOT/'src')]


def tree_values(tree,x):
    x=np.atleast_2d(x)
    result=np.empty((len(x),4))
    stack=[(tree,np.arange(len(x)))]
    while stack:
        node,rows=stack.pop()
        if 'feature' not in node:
            result[rows]=node['value']
        else:
            mask=x[rows,node['feature']]<=node['threshold']
            stack.append((node['left'],rows[mask]));stack.append((node['right'],rows[~mask]))
    return result


def decoded(tree,x,config):
    q=tree_values(tree,x)
    q[:,1:]=q[:,:1]+q[:,1:]/config['learning_advantage_scale']
    return q


def select(q,x,config):
    x=np.atleast_2d(x)
    p=np.column_stack((x[:,:2],1-x[:,:2].sum(axis=1)))
    fees=np.column_stack((np.zeros(len(x)),x[:,10,None]*np.abs(
        p[:,None,:]-np.array(config['learning_targets'])[None,:,:]).sum(axis=2)/2))
    forced=(x[:,2]>=config['review_drawdown'])|(x[:,11]>.5)
    fees[forced]=0
    benefit=1-np.exp(np.clip(-(q-q[:,:1]),-700,700))
    eligible=benefit>fees+config['learning_buffer'];eligible[:,0]=True
    net=q+np.log1p(-fees);net[~eligible]=-np.inf
    action=np.argmax(net,axis=1);action[forced]=0
    return action,net[np.arange(len(x)),action],benefit,fees


def features(holdings,inflation,peak,mdd,history,cost,pending):
    total=sum(holdings)
    last=history[-1]
    reals=[(1+r)/(1+last.inflation)-1 for r in last.nominal_returns]
    equity=[(1+o.nominal_returns[0])/(1+o.inflation)-1 for o in history[-3:]]
    mean=sum(equity)/len(equity)
    vol=math.sqrt(sum((v-mean)**2 for v in equity)/len(equity))
    return np.array([holdings[0]/total,holdings[1]/total,1-total/inflation/peak,mdd,
        *reals,last.inflation,mean,vol,cost,float(pending)])


def verify_transitions(data,scenarios,config):
    rows=0;maximum_error=0.
    for sid in sorted(set(data['scenario_id'])):
        scenario=scenarios[int(sid)]
        for cost in (config['cost_rate'],2*config['cost_rate']):
            for behavior in range(4):
                indices=np.flatnonzero((data['scenario_id']==sid)&(data['behavior']==behavior)&np.isclose(data['x'][:,10],cost))
                assert data['period'][indices].tolist()==list(range(2,len(scenario.observations)+1))
                first=scenario.observations[0]
                fee=100*cost*.85
                h=[(100-fee)*w*(1+r) for w,r in zip((.5,.35,.15),first.nominal_returns)]
                inflation=1+first.inflation
                peak=max(100.,sum(h)/inflation);mdd=max(fee/100,1-sum(h)/inflation/peak);pending=False
                for index in indices:
                    period=int(data['period'][index]);o=scenario.observations[period-1]
                    x=features(h,inflation,peak,mdd,scenario.observations[:period-1],cost,pending)
                    np.testing.assert_allclose(data['x'][index],x,atol=2e-12,rtol=2e-12)
                    assert data['available'][index]==len(scenario.observations)-period+1
                    forced=pending or x[2]>=config['review_drawdown']
                    choices=[]
                    for action in range(4):
                        v=sum(h)
                        target=config['learning_targets'][action-1]
                        moved=0 if action==0 or forced else sum(abs(v*w-z) for w,z in zip(target,h))/2
                        fee=cost*moved
                        post=h[:] if action==0 or forced else [(v-fee)*w for w in target]
                        end=[z*(1+r) for z,r in zip(post,o.nominal_returns)]
                        next_inflation=inflation*(1+o.inflation)
                        next_peak=max(peak,sum(post)/inflation,sum(end)/next_inflation)
                        next_mdd=max(mdd,1-sum(post)/inflation/peak,1-sum(end)/next_inflation/next_peak)
                        nx=features(end,next_inflation,next_peak,next_mdd,scenario.observations[:period],cost,forced)
                        reward=math.log(sum(end)/sum(post)/(1+o.inflation))
                        breach=float(mdd<=config['learning_loss_tolerance']<next_mdd)
                        np.testing.assert_allclose(data['next_x'][index,action],nx,atol=2e-12,rtol=2e-12)
                        assert abs(data['gross_log'][index,action]-reward)<2e-12
                        assert abs(data['fees'][index,action]-fee/v)<2e-12
                        assert data['breach'][index,action]==breach
                        maximum_error=max(maximum_error,float(np.max(np.abs(data['next_x'][index,action]-nx))))
                        choices.append((end,next_inflation,next_peak,next_mdd,forced));rows+=1
                    h,inflation,peak,mdd,pending=choices[behavior]
    assert rows==4*len(data['x'])
    return rows,maximum_error


def verify_node_means(tree,x,y,allowed):
    error=0.;stack=[(tree,np.arange(len(x)))];nodes=0
    while stack:
        node,rows=stack.pop();nodes+=1
        assert len(rows)==node['n']
        error=max(error,float(np.max(np.abs(np.mean(y[rows],axis=0)-node['value']))))
        if 'feature' in node:
            assert node['feature'] in allowed
            mask=x[rows,node['feature']]<=node['threshold']
            stack.extend([(node['left'],rows[mask]),(node['right'],rows[~mask])])
    assert error<1e-9
    return nodes,error


def verify_bellman(model,data,config):
    count=0;maximum_error=0.;nodes=0
    for family,penalties in model['fqi'].items():
        for penalty,fitted in penalties.items():
            assert sorted(set(data['scenario_id'].tolist()))==sorted(fitted['train_ids'])==sorted(model['train_ids'])
            assert fitted['feature_indices']==(list(range(12)) if family=='full' else [0,1,2,3,10,11])
            for k in range(1,fitted['maximum_remaining']+1):
                mask=data['available']>=k;x=data['x'][mask]
                target=data['gross_log'][mask]-float(penalty)*data['breach'][mask]
                if k>1:
                    nx=data['next_x'][mask].reshape(-1,12)
                    q=decoded(fitted['models'][str(k-1)]['teacher'],nx,config)
                    target+=select(q,nx,config)[1].reshape(-1,4)
                tree=fitted['models'][str(k)]['teacher']
                encoded=target.copy();encoded[:,1:]=(target[:,1:]-target[:,:1])*config['learning_advantage_scale']
                node_count,error=verify_node_means(tree,x,encoded,fitted['feature_indices'])
                teaching=tree_values(tree,x)
                student=fitted['models'][str(k)]['student']
                student_count,student_error=verify_node_means(student,x,teaching,fitted['feature_indices'])
                trace=fitted['trace'][k-1]
                np.testing.assert_allclose(target.sum(axis=0),trace['target_sum'],atol=1e-8,rtol=1e-10)
                np.testing.assert_allclose((target**2).sum(axis=0),trace['target_square_sum'],atol=1e-8,rtol=1e-10)
                maximum_error=max(maximum_error,error,student_error);nodes+=node_count+student_count;count+=1
    return count,nodes,maximum_error


def verify_learning(out,scenarios):
    out=Path(out)
    manifest=json.loads((out/'manifest.json').read_text())
    if manifest['phase']!='development' or manifest['holdout_used'] or any(not 1<=sid<=120 for sid in manifest['scenario_ids']):
        raise PermissionError('FQI replay is development-only')
    config=json.loads((out/'config.json').read_text())
    if config['study_kind']!='counterfactual_fqi':raise ValueError('Wrong replay kind')
    by_id={s.scenario_id:s for s in scenarios}
    outer=json.loads((out/'fitted_models.json').read_text())
    splits=json.loads((out/'splits.json').read_text())
    checks=transitions=bellman=nodes=0;transition_error=bellman_error=0.
    for fold in splits:
        index=fold['outer_fold']
        fits=[(out/'training_data'/f'outer_{index}.npz',outer[index]['model'],fold['train_ids'],fold['validation_ids'])]
        for inner in fold['inner']:
            stem=f"outer_{index}_inner_{inner['inner_fold']}"
            fits.append((out/'training_data'/(stem+'.npz'),json.loads((out/'tuning_models'/(stem+'.json')).read_text()),
                         inner['train_ids'],inner['validation_ids']+fold['validation_ids']))
        for path,model,train_ids,validation_ids in fits:
            with np.load(path,allow_pickle=False) as archive:data={key:archive[key] for key in archive.files}
            assert set(data['scenario_id'])==set(train_ids)
            assert set(train_ids).isdisjoint(validation_ids)
            digest=hashlib.sha256()
            for key,array in sorted(data.items()):
                digest.update(key.encode());digest.update(str((array.shape,array.dtype.str)).encode());digest.update(np.ascontiguousarray(array).tobytes())
            assert all(m['dataset_sha256']==digest.hexdigest() for penalties in model['fqi'].values() for m in penalties.values())
            n,error=verify_transitions(data,by_id,config);transitions+=n;transition_error=max(transition_error,error)
            n,node_count,error=verify_bellman(model,data,config);bellman+=n;nodes+=node_count;bellman_error=max(bellman_error,error);checks+=1
    # Default CSV conversion can shift a saved double across a tree threshold.
    # Restore the original IEEE value before testing deterministic decisions.
    records=pd.read_csv(out/'decisions.csv.gz',float_precision='round_trip')
    records=records[records.policy.isin(['fqi_student','fqi_teacher','fqi_portfolio'])]
    decision_count=gate_count=0;gate_error=0.
    for (fold,name,sid,horizon,cost),group in records.groupby(['fold','policy','scenario_id','evaluation_horizon','cost_rate']):
        fitted=outer[int(fold)]
        family='portfolio' if name=='fqi_portfolio' else 'full';kind='teacher' if name=='fqi_teacher' else 'student'
        value_model=fitted['model']['fqi'][family][str(fitted['params'][name]['penalty'])]
        peak=config['initial_wealth'];mdd=0.;inflation=1.;pending=False
        for record in group.sort_values('period').itertuples():
            period=record.period;decision_count+=1
            h=[getattr(record,'pre_trade_holdings_'+a) for a in ('equity','bond','bill')]
            if period>1:
                x=features(h,inflation,peak,mdd,by_id[int(sid)].observations[:period-1],cost,pending)
                if pending:assert record.action=='keep'
                elif x[2]>=config['review_drawdown']:assert record.action=='human_review';pending=True
                else:
                    tree=value_model['models'][str(int(horizon)-period+1)][kind]
                    q=decoded(tree,x,config)
                    action,_,benefit,fees=select(q,x,config);action=int(action[0]);gate_count+=1
                    if action==0:assert record.action=='keep'
                    else:
                        target=config['learning_targets'][action-1]
                        np.testing.assert_allclose([getattr(record,'target_weights_'+a) for a in ('equity','bond','bill')],target,atol=1e-12)
                        assert record.action in ('rebalance','change_target')
                        gate_error=max(gate_error,abs(record.estimated_benefit-benefit[0,action]),abs(record.estimated_cost-fees[0,action]))
                        assert abs(record.estimated_benefit-benefit[0,action])<1e-10
                        assert abs(record.estimated_cost-fees[0,action])<1e-10
                        assert record.estimated_benefit>record.estimated_cost+config['learning_buffer']
            post=sum(getattr(record,'post_trade_holdings_'+a) for a in ('equity','bond','bill'))/inflation
            peak=max(peak,post);mdd=max(mdd,1-post/peak)
            inflation=record.cumulative_inflation;peak=max(peak,record.real_wealth);mdd=max(mdd,1-record.real_wealth/peak)
    result=dict(all_checks_passed=True,holdout_used=False,training_datasets_checked=checks,
        counterfactual_transitions_checked=transitions,bellman_stages_checked=bellman,tree_nodes_checked=nodes,
        maximum_transition_error=transition_error,maximum_tree_mean_error=bellman_error,
        decision_rows_checked=decision_count,value_gates_checked=gate_count,maximum_gate_error=gate_error,
        verification_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Independent nominal scalar transitions, source isolation, Bellman target/node means, distillation and all horizon-conditioned deployment decisions.')
    (out/'learning_replay.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    from safebench.data import load_partition
    out=Path(sys.argv[1])
    if json.loads((out/'manifest.json').read_text())['phase']!='development':raise PermissionError('Development only')
    scenarios,_=load_partition(ROOT/'data/SafeInvest_Synthetic_Data.csv')
    print(json.dumps(verify_learning(out,scenarios),indent=2))
