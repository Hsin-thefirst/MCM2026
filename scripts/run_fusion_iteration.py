"""Finite registered execution experiments on development OOF models only."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
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
from run_fusion_research import scalar_check
from safeinvest.contracts import SimulationConfig
from safebench.data import load_partition
from safebench.engine import simulate
from safebench.experiment import code_digest,dump
from safebench.metrics import compute_metrics,summarize,paired_comparisons


def main():
    parser=argparse.ArgumentParser(description='固定预算融合迭代，仅开发分区')
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); out=args.out.resolve();out.relative_to(ROOT/'benchmark/outputs')
    settings=json.loads(args.config.read_text())
    base=json.loads((ROOT/'configs/fusion_research.json').read_text())
    source=ROOT/settings['source_run'];fusion_source=ROOT/settings['fusion_run']
    config=json.loads((source/'config.json').read_text())
    models=json.loads((source/'fitted_models.json').read_text());splits=json.loads((source/'splits.json').read_text())
    scenarios,sha=load_partition(ROOT/'data/SafeInvest_Synthetic_Data.csv','development')
    original=json.loads((source/'manifest.json').read_text())
    assert original['partition_sha256']==sha and original['code_sha256']==code_digest()
    lookup={s.scenario_id:s for s in scenarios};assert set(lookup)==set(range(1,121))
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    protected={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'benchmark/FINAL_EVALUATION_LOCK.json',ROOT/'benchmark/submission_frozen_v1.json']}
    files=[args.config.resolve(),ROOT/'configs/fusion_research.json',Path(__file__),ROOT/'scripts/efficient_fusion.py',
        ROOT/'scripts/fusion_policy.py',ROOT/'scripts/run_fusion_research.py',source/'fitted_models.json',source/'splits.json',
        source/'scenario_metrics.csv',fusion_source/'scenario_metrics.csv']
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),phase='development',holdout_used=False,
        final_already_exposed=True,independent_confirmation=False,development_sha256=sha,source_core_sha256=code_digest(),
        settings=dict(settings,policies=list(settings['rules'])),sources={str(p.relative_to(ROOT)):digest(p) for p in files},
        protected_hashes=protected,selection='全部固定规则，探索性开发；不替换冻结提交。'))
    rows=[]; errors=[];gate_count=cap_count=records=0; started=time.perf_counter()
    with gzip.open(out/'decisions.jsonl.gz','wt',encoding='utf-8') as log:
        for fitted,split in zip(models,splits):
            model=fitted['model'];params=fitted['params']['plan_a_balanced']
            assert fitted['fold']==split['outer_fold'] and model['train_ids']==split['train_ids']
            assert not set(model['train_ids'])&set(split['validation_ids'])
            assert set(model['train_ids'])|set(split['validation_ids'])==set(lookup)
            for name,rule in settings['rules'].items():
                for cost in settings['costs']:
                    for h in settings['horizons']:
                        for sid in split['validation_ids']:
                            p=EfficientFusion(name,model,params,config,h,dict(base,**rule.get('valuation',{})),rule)
                            result=simulate(lookup[sid],p,p.initial_target,SimulationConfig(h,100.,cost))
                            err,ng,_=scalar_check(result,lookup[sid]);errors.append(err);gate_count+=ng
                            rows.append(dict(compute_metrics(result),fold=fitted['fold']))
                            for r in result.records:
                                d=r.policy_diagnostics
                                if d and r.action in (r.action.REBALANCE,r.action.CHANGE_TARGET):
                                    assert r.estimated_benefit > r.estimated_cost+p.buffer
                                    assert .5*np.abs(np.array(r.post_trade_holdings)/sum(r.post_trade_holdings)-np.array(r.pre_trade_holdings)/sum(r.pre_trade_holdings)).sum()<=base['move_cap']+1e-10
                                    assert not d['reversal_blocked'][d['chosen']]
                                    cap_count+=1
                                row=asdict(r);row['action']=r.action.value
                                row.update(policy=name,scenario_id=sid,cost_rate=cost,horizon=h,fold=fitted['fold'])
                                log.write(json.dumps(row,allow_nan=False)+'\n');records+=1
                print(f'外折 {fitted["fold"]+1}/5：{name}',flush=True)
    frame=pd.DataFrame(rows)
    old=pd.read_csv(source/'scenario_metrics.csv');previous=pd.read_csv(fusion_source/'scenario_metrics.csv')
    refs=old[old.policy.isin(['plan_a_balanced','equal_annual','balanced_annual','plan_a_annual','submission_c_cap10'])&old.cost_rate.isin(settings['costs'])]
    refs=pd.concat([refs[frame.columns],previous[previous.policy.isin(['fusion_mpc','fusion_drawdown'])&previous.cost_rate.isin(settings['costs'])][frame.columns]])
    combined=pd.concat([frame,refs],ignore_index=True)
    combined.to_csv(out/'scenario_metrics.csv',index=False);summarize(combined).to_csv(out/'summary.csv',index=False)
    comparisons=[]
    for ref in ['plan_a_balanced','fusion_drawdown','fusion_mpc','balanced_annual']:
        table=paired_comparisons(combined,settings['seed'],settings['bootstrap'],ref);table['reference']=ref;comparisons.append(table)
    pd.concat(comparisons,ignore_index=True).to_csv(out/'paired_comparisons.csv',index=False)
    reproduction=0
    if 'efficient_base' in settings['rules']:
        a=frame[frame.policy=='efficient_base'].drop(columns=['policy']).set_index(['scenario_id','horizon','cost_rate']).sort_index()
        b=previous[previous.policy=='fusion_drawdown'].drop(columns=['policy']).set_index(['scenario_id','horizon','cost_rate']).sort_index()
        b=b.loc[a.index,a.columns]
        pd.testing.assert_frame_equal(a,b,check_exact=False,atol=1e-12,rtol=1e-12,check_dtype=False);reproduction=len(a)
    for path,d in protected.items():assert digest(ROOT/path)==d
    dump(out/'verification.json',dict(passed=True,paths=len(frame),annual_records=records,base_reproduced_rows=reproduction,
        gate_checks=gate_count,cap_checks=cap_count,maximum_scalar_error=max(errors),protected_unchanged=True,seconds=time.perf_counter()-started))
    print(summarize(combined).query('horizon==30 and cost_rate==0.0025').to_string(index=False),flush=True)


if __name__=='__main__':main()
