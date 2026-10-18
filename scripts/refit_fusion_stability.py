"""Repeat the declared nested A fit on a new development split, without new search."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'benchmark')]
from efficient_fusion import EfficientFusion
from fusion_policy import FusionPolicy
from run_fusion_research import scalar_check
from safeinvest.contracts import SimulationConfig
from safebench.data import load_partition,folds
from safebench.engine import simulate
from safebench.experiment import tune,dump,code_digest
from safebench.policies import fit_model,make_policy
from safebench.plan_c import fit_control
from safebench.metrics import compute_metrics,summarize,paired_comparisons


def main():
    parser=argparse.ArgumentParser(description='固定候选新开发划分重拟合稳定性检查')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve();out.relative_to(ROOT/'benchmark/outputs')
    settings=json.loads((ROOT/'configs/fusion_efficiency_round3.json').read_text())
    candidate=settings['stability_candidate'];rule=settings['rules'][candidate]
    config=json.loads((ROOT/settings['source_run']/'config.json').read_text())
    config['submission']['enabled']=False;config['plan_b']['enabled']=False;config['plan_c']['enabled']=False
    config['seed']=settings['stability_seed']
    base=json.loads((ROOT/'configs/fusion_research.json').read_text())
    scenarios,sha=load_partition(ROOT/'data/SafeInvest_Synthetic_Data.csv','development')
    lookup={s.scenario_id:s for s in scenarios};assert set(lookup)==set(range(1,121))
    protected={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [ROOT/'benchmark/FINAL_EVALUATION_LOCK.json',ROOT/'benchmark/submission_frozen_v1.json']}
    out.mkdir(parents=True,exist_ok=False)
    settings=dict(settings,seed=config['seed'],rules={candidate:rule},policies=[candidate])
    files=[Path(__file__),ROOT/'scripts/efficient_fusion.py',ROOT/'scripts/fusion_policy.py',ROOT/'scripts/run_fusion_research.py',
        ROOT/'configs/fusion_efficiency_round3.json',ROOT/'configs/fusion_research.json']
    dump(out/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),phase='development',holdout_used=False,
        final_already_exposed=True,independent_confirmation=False,development_sha256=sha,code_sha256=code_digest(),
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        settings=settings,protected_hashes=protected,selection='候选在本次划分执行前固定；重新嵌套选择原A参数，不按结果换候选。'))
    dump(out/'config.json',config)
    rows=[];models=[];splits=[];trials=[];errors=[];records=gates=caps=0;started=time.perf_counter()
    with gzip.open(out/'decisions.jsonl.gz','wt',encoding='utf-8') as log:
        for fold,valid_ids in enumerate(folds(lookup,5,config['seed'])):
            train=[s for s in scenarios if s.scenario_id not in valid_ids]
            model=fit_model(train,config)
            params,trial,audit=tune(train,config,config['seed']+fold+1,refit_model=model)
            model['scenario_control']=fit_control(train,config)
            assert not set(model['train_ids'])&set(valid_ids)
            models.append(dict(fold=fold,model=model,params=params))
            splits.append(dict(outer_fold=fold,train_ids=model['train_ids'],validation_ids=valid_ids,inner=audit))
            trials.extend(dict(x,outer_fold=fold) for x in trial)
            dump(out/'fitted_models.json',models);dump(out/'splits.json',splits)
            print(f'重新拟合及内层调参 {fold+1}/5 完成，A参数 {params["plan_a_balanced"]}',flush=True)
            for name in [candidate,'fusion_drawdown','plan_a_balanced','plan_a_annual','equal_annual','balanced_annual']:
                for cost in settings['costs']:
                    for h in settings['horizons']:
                        for sid in valid_ids:
                            if name==candidate:p=EfficientFusion(name,model,params['plan_a_balanced'],config,h,base,rule)
                            elif name=='fusion_drawdown':p=FusionPolicy(name,model,params['plan_a_balanced'],config,h,base)
                            else:p=make_policy(name,model,params.get(name,config['tuning_grid'][0]),config,h)
                            result=simulate(lookup[sid],p,p.initial_target,SimulationConfig(h,100.,cost))
                            # This verifier checks gated paths; annual controls retain the tested engine.
                            if name in (candidate,'fusion_drawdown','plan_a_balanced'):
                                err,ng,_=scalar_check(result,lookup[sid]);errors.append(err);gates+=ng
                            rows.append(dict(compute_metrics(result),fold=fold))
                            for r in result.records:
                                d=r.policy_diagnostics
                                if d and name==candidate and r.moved_amount>1e-8:
                                    assert r.estimated_benefit>r.estimated_cost+p.buffer
                                    w=np.array(r.post_trade_holdings)/sum(r.post_trade_holdings)
                                    before=np.array(r.pre_trade_holdings)/sum(r.pre_trade_holdings)
                                    assert .5*np.abs(w-before).sum()<=.1+1e-10
                                    assert not d['reversal_blocked'][d['chosen']];caps+=1
                                record=asdict(r);record['action']=r.action.value
                                record.update(policy=name,scenario_id=sid,cost_rate=cost,horizon=h,fold=fold)
                                log.write(json.dumps(record,allow_nan=False)+'\n');records+=1
    frame=pd.DataFrame(rows);frame.to_csv(out/'scenario_metrics.csv',index=False)
    summarize(frame).to_csv(out/'summary.csv',index=False);pd.DataFrame(trials).to_csv(out/'tuning_trials.csv',index=False)
    paired=[]
    for ref in ['plan_a_balanced','fusion_drawdown']:
        table=paired_comparisons(frame,settings['seed'],settings['bootstrap'],ref);table['reference']=ref;paired.append(table)
    pd.concat(paired,ignore_index=True).to_csv(out/'paired_comparisons.csv',index=False)
    for path,sha in protected.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha
    dump(out/'verification.json',dict(passed=True,paths=len(frame),annual_records=records,scalar_paths=3600,
        gate_checks=gates,cap_checks=caps,maximum_scalar_error=max(errors),protected_unchanged=True,
        seconds=time.perf_counter()-started,all_a_selected_feasible=all(m['params']['plan_a_balanced']['selection_feasible'] for m in models)))
    print(summarize(frame).query('horizon==30 and cost_rate==0.0025').to_string(index=False),flush=True)


if __name__=='__main__':main()
