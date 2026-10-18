"""Evaluate the two fixed value rules after the calibration gate has passed."""
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
from fusion_value_extensions import ValueFusion
from fusion_calibration import choose_group
from safeinvest.contracts import SimulationConfig
from safebench.data import load_partition
from safebench.engine import simulate
from safebench.experiment import dump,code_digest
from safebench.metrics import compute_metrics,summarize
from safebench.plan_a import PlanAPolicy,parameter_grid,allocation_available


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description='固定A延续与不确定性惩罚开发扩展')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve();out.relative_to(ROOT/'benchmark/outputs')
    settings=json.loads((ROOT/'configs/fusion_value_extensions.json').read_text())
    source=ROOT/settings['source_run']
    gate=json.loads((source/'decision.json').read_text())
    assert gate['continue_extensions'] and gate['all_primary_groups_feasible'] and gate['primary_cost_protection']
    assert settings['bootstrap_training_draws']==200 and settings['bootstrap_training_seed']==20261021 and settings['uncertainty_coefficient']==1.
    fusion=json.loads((ROOT/'configs/fusion_research.json').read_text())
    scenarios,data_hash=load_partition(ROOT/'data/SafeInvest_Synthetic_Data.csv','development')
    lookup={s.scenario_id:s for s in scenarios};assert set(lookup)==set(range(1,121))
    parent=json.loads((source/'manifest.json').read_text())
    assert parent['development_sha256']==data_hash and parent['code_sha256']==code_digest()
    protected={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'benchmark/FINAL_EVALUATION_LOCK.json',ROOT/'benchmark/submission_frozen_v1.json',ROOT/'dist/SafeInvest_source.zip']}
    files=[Path(__file__),ROOT/'scripts/fusion_value_extensions.py',ROOT/'scripts/fusion_calibration.py',
           ROOT/'scripts/analyze_fusion_calibration.py',ROOT/'scripts/verify_calibration_ledger.py',
           ROOT/'scripts/efficient_fusion.py',ROOT/'scripts/fusion_policy.py',
           ROOT/'configs/fusion_value_extensions.json',ROOT/'configs/fusion_research.json',source/'decision.json',source/'manifest.json']
    for seed in [20261014,20261019]:
        files.extend(source/str(seed)/name for name in ['config.json','splits.json','fitted_models.json','scenario_metrics.csv','decisions.jsonl.gz','independent_replay.json'])
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),phase='development',holdout_used=False,
        final_already_exposed=True,independent_confirmation=False,development_sha256=data_hash,code_sha256=code_digest(),
        settings=settings,sources={str(p.relative_to(ROOT)):sha(p) for p in files},protected_hashes=protected,
        selection='两个固定独立规则分别按实际路径校准原六参数；不组合，全部完成后停止。',policy_count=16)
    out.mkdir(parents=True,exist_ok=False);dump(out/'manifest.json',manifest)
    all_inner=[];started=time.perf_counter()
    references=['efficient_horizon','calibration_group_fusion','plan_a_balanced','fusion_drawdown',
                'equal_annual','balanced_annual','plan_a_annual','plan_a_hold']
    for seed in [20261014,20261019]:
        original=source/str(seed);target=out/str(seed);target.mkdir()
        config=json.loads((original/'config.json').read_text());dump(target/'config.json',config)
        fitted=json.loads((original/'fitted_models.json').read_text())
        audits=json.loads((original/'splits.json').read_text())
        old_metrics=pd.read_csv(original/'scenario_metrics.csv')
        metrics=old_metrics[old_metrics.policy.isin(references)].to_dict('records')
        models=[];trials=[];annual_records=0
        with gzip.open(target/'decisions.jsonl.gz','wt',encoding='utf-8') as ledger:
            # Copy audited reference paths verbatim; do not resimulate them.
            with gzip.open(original/'decisions.jsonl.gz','rt') as saved:
                for line in saved:
                    row=json.loads(line)
                    if row['policy'] in references: ledger.write(line);annual_records+=1
            for item,audit in zip(fitted,audits):
                fold=item['fold'];model=item['model'];ids=audit['train_ids'];valid_ids=audit['validation_ids']
                assert model['train_ids']==ids and not set(ids)&set(valid_ids)
                inner_rows=[];unavailable={}
                for inner in audit['inner']:
                    training=inner['model'];validation=inner['validation_ids']
                    assert training['train_ids']==training['scenario_control']['train_ids']==inner['train_ids']
                    assert not set(training['train_ids'])&set(validation)
                    assert set(training['train_ids'])|set(validation)==set(ids)
                    for trial,params in enumerate(parameter_grid(config)):
                        if not allocation_available(training,'balanced',params,settings['long_horizons']):
                            unavailable.setdefault(trial,[]).append(inner['inner_fold']);continue
                        for h in settings['long_horizons']:
                            entry=training['plan_a']['entries'][f"balanced:{h}:{params['risk_ratio']:g}"]
                            for sid in validation:
                                cfg=SimulationConfig(h,100.,config['cost_rate'])
                                annual=PlanAPolicy('annual',training,params,config,h,mode='annual',review=False)
                                growth=compute_metrics(simulate(lookup[sid],annual,annual.initial_target,cfg))['real_cagr']
                                for program,family in settings['families']:
                                    policy=ValueFusion(program,training,params,config,h,fusion,settings['rule'],family)
                                    result=simulate(lookup[sid],policy,policy.initial_target,cfg)
                                    row=compute_metrics(result)
                                    row.update(family=family,annual_growth=growth,trial=trial,inner_fold=inner['inner_fold'],
                                        plan_a_risk_budget=entry['risk_budget'],plan_a_validation_budget=entry.get('validation_budget',entry['risk_budget']))
                                    inner_rows.append(row)
                    print(f'seed={seed} 外折{fold+1}/5 内折{inner["inner_fold"]+1}/3 估值规则完成',flush=True)
                frame=pd.DataFrame(inner_rows);selected={}
                for program,family in settings['families']:
                    params,evidence=choose_group(frame,model,config,settings['long_horizons'],ids,family,unavailable)
                    selected[program]=params
                    trials.extend(dict(r,outer_fold=fold,program=program) for r in evidence)
                all_inner.extend(dict(r,seed=seed,outer_fold=fold) for r in inner_rows)
                models.append(dict(fold=fold,model=model,original=item['original'],long_params=selected))
                dump(target/'fitted_models.json',models)
                print(f'seed={seed} 外折{fold+1} 估值选择 {selected}',flush=True)
                for program,family in settings['families']:
                    prefix='k2' if program==settings['primary'] else 'k1'
                    for name in [program]+[prefix+'_anchor_'+mode for mode in ['a','annual','hold']]:
                        for cost in settings['costs']:
                            for h in settings['horizons']:
                                params=item['original'] if h<=6 else selected[program]
                                for sid in valid_ids:
                                    if name==program: policy=ValueFusion(name,model,params,config,h,fusion,settings['rule'],family)
                                    else:
                                        mode=name.rsplit('_',1)[1]
                                        policy=PlanAPolicy(name,model,params,config,h,mode='partial' if mode=='a' else mode,review=mode=='a')
                                    result=simulate(lookup[sid],policy,policy.initial_target,SimulationConfig(h,100.,cost))
                                    metrics.append(dict(compute_metrics(result),fold=fold))
                                    for record in result.records:
                                        row=asdict(record)
                                        row.update(action=record.action.value,policy=name,scenario_id=sid,horizon=h,cost_rate=cost,
                                            fold=fold,required_buffer=getattr(policy,'buffer',0.),requires_benefit_gate=policy.requires_benefit_gate)
                                        ledger.write(json.dumps(row,allow_nan=False)+'\n');annual_records+=1
                print(f'seed={seed} 外折{fold+1}/5 估值外层完成',flush=True)
        frame=pd.DataFrame(metrics);assert len(frame)==19200
        frame.to_csv(target/'scenario_metrics.csv',index=False);summarize(frame).to_csv(target/'summary.csv',index=False)
        dump(target/'splits.json',audits);dump(target/'tuning_trials.json',trials)
        dump(target/'manifest.json',dict(manifest,seed=seed))
        dump(target/'verification.json',dict(passed=True,paths=len(frame),annual_records=annual_records,
             reused_reference_paths=9600,newly_evaluated_paths=9600))
    pd.DataFrame(all_inner).to_csv(out/'inner_metrics.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    for p,digest in protected.items(): assert sha(ROOT/p)==digest
    dump(out/'verification.json',dict(passed=True,protected_unchanged=True,inner_paths=len(all_inner),seconds=time.perf_counter()-started))


if __name__=='__main__': main()
