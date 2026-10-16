#!/usr/bin/env python3
"""Extend a verified development run without retraining unchanged old models."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmark'))
from safebench import ROOT
import numpy as np
import pandas as pd
from safebench.cushion import CUSHION_NAMES,CALIBRATED_NAMES
from safebench.data import load_partition
from safebench.experiment import (digest,dump,manifest,evaluate,policy_names,tune_cushion,tune_calibrated,suitability)
from safebench.metrics import summarize,pareto_flags,paired_comparisons
from safebench.validation import comparisons_and_screen,audit_outputs


def run(source,out):
    started=time.perf_counter(); config=json.loads((ROOT/'configs/benchmark.json').read_text())
    prior=json.loads((source/'manifest.json').read_text()); previous=json.loads((source/'config.json').read_text())
    comparison=dict(config); comparison.pop('cushion_research',None)
    prior_comparison=dict(previous); prior_comparison.pop('cushion_research',None)
    comparison.pop('calibrated_research',None); prior_comparison.pop('calibrated_research',None)
    if comparison!=prior_comparison or not json.loads((source/'validation.json').read_text())['all_checks_passed']:
        raise ValueError('Only the declared cushion extension may differ from the verified source configuration')
    snap=source/'source_snapshot'
    files=sorted((snap/'benchmark/safebench').glob('*.py'))+[snap/'benchmark/run.py',snap/'src/safeinvest/contracts.py',snap/'src/safeinvest/__init__.py']
    if digest({str(p.relative_to(snap)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})!=prior['code_sha256']:
        raise ValueError('Source snapshot does not reproduce its historical code hash')
    scenarios,sha=load_partition(ROOT/'data/SafeInvest_Synthetic_Data.csv')
    if prior['phase']!='development' or sha!=prior['partition_sha256']:
        raise ValueError('Source is not the same development partition')
    out.mkdir(exist_ok=False)
    info=manifest(config,sha,[s.scenario_id for s in scenarios],'development')
    info.update(evaluated_policies=policy_names(config),reused_model_run=str(source),
                reused_model_sha256=hashlib.sha256((source/'fitted_models.json').read_bytes()).hexdigest(),
                reused_source_code_sha256=prior['code_sha256'])
    dump(out/'manifest.json',info); dump(out/'config.json',config)
    fitted=json.loads((source/'fitted_models.json').read_text()); splits=json.loads((source/'splits.json').read_text())
    calibrating=config.get('calibrated_research',{}).get('enabled',False)
    revised_names=CALIBRATED_NAMES if calibrating else CUSHION_NAMES
    all_metrics=[]; all_records=[]; trials=[]; inner=[]
    for f in fitted:
        fold=f['fold']; model=f['model']; params=f['params']
        train=[s for s in scenarios if s.scenario_id in set(model['train_ids'])]
        valid=[s for s in scenarios if s.scenario_id not in set(model['train_ids'])]
        tuner=tune_calibrated if calibrating else tune_cushion
        added,rows,audits=tuner(train,config,config['seed']+fold+1)
        for actual,expected in zip(audits,splits[fold]['inner']):
            assert actual['train_ids']==expected['train_ids'] and actual['validation_ids']==expected['validation_ids']
        params.update(added); trials.extend(dict(outer_fold=fold,**r) for r in rows)
        inner.append(dict(fold=fold,inner=audits))
        print('Outer fold',fold+1,'new selected',added,flush=True)
        fold_metrics=[]
        for cost,horizons in [(config['cost_rate'],config['horizons']),*[(c,[30]) for c in config['sensitivity_costs']]]:
            m,r=evaluate(valid,model,params,config,cost=cost,horizons=horizons,fold=fold)
            all_metrics.append(m); all_records.append(r); fold_metrics.append(m)
        summarize(pd.concat(fold_metrics,ignore_index=True)).to_csv(out/f'fold_{fold}_summary.csv',index=False)
        dump(out/f'fold_{fold}_checkpoint.json',dict(model=model,params=params,inner=audits,
            code_sha256=info['code_sha256'],source_model_sha256=info['reused_model_sha256']))
    metrics=pd.concat(all_metrics,ignore_index=True); records=pd.concat(all_records,ignore_index=True)
    original=pd.read_csv(source/'scenario_metrics.csv')
    original=original[~original.policy.isin(revised_names)]
    key=['policy','scenario_id','horizon','cost_rate']
    columns=[c for c in original if c not in key and c not in ['simulation_seconds','policy_inference_seconds','policy_decision_calls']]
    actual=metrics[metrics.policy.isin(original.policy.unique())].set_index(key).sort_index()
    expected=original.set_index(key).sort_index()
    assert actual.index.equals(expected.index)
    difference=float(np.max(np.abs(actual[columns].to_numpy(dtype=float)-expected[columns].to_numpy(dtype=float))))
    assert difference<1e-11,'Cached old-model financial outcomes changed'
    metrics.to_csv(out/'scenario_metrics.csv',index=False)
    records.to_csv(out/'decisions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    dump(out/'fitted_models.json',fitted); dump(out/'splits.json',splits); dump(out/'cushion_tuning_splits.json',inner)
    prior_trials=pd.read_csv(source/'tuning_trials.csv')
    prior_trials=prior_trials[~prior_trials.policy.isin(revised_names)]
    pd.concat([prior_trials,pd.DataFrame(trials)],ignore_index=True).to_csv(out/'tuning_trials.csv',index=False)
    for name in ['seed_metrics.csv','seed_summary.csv','training_trace.csv','frontier_training_diagnostics.csv','forecast_diagnostics.csv']:
        if (source/name).exists(): (out/name).write_bytes((source/name).read_bytes())
    pareto_flags(summarize(metrics)).to_csv(out/'summary.csv',index=False)
    paired_comparisons(metrics,config['seed'],config['bootstrap']).to_csv(out/'paired_comparisons.csv',index=False)
    assigned,risk=suitability(metrics,config)
    assigned.to_csv(out/'suitability.csv',index=False); risk.to_csv(out/'suitability_risk.csv',index=False)
    comparisons_and_screen(out,config); audit_outputs(out)
    dump(out/'extension_verification.json',dict(old_policy_count=original.policy.nunique(),
        old_metric_rows=len(original),maximum_financial_difference=difference,unchanged_old_outcomes=True,
        source_model_sha256=info['reused_model_sha256'],new_family_fit_inside_inner_folds=True,holdout_used=False))
    dump(out/'runtime.json',dict(seconds=time.perf_counter()-started,training_reused_from=str(source)))
    print('Completed extension; old financial outcomes verified:',difference,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description='复用已核验外层模型，仅新增平滑风险偏好策略及内层选择；不读取最终留出集。')
    p.add_argument('--source',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); run(a.source.resolve(),a.out.resolve())
