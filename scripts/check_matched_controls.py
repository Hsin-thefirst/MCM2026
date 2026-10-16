#!/usr/bin/env python3
"""Evaluate strict controls using each candidate's already-selected parameters.

No refitting or parameter selection occurs here. Pointwise intervals are
diagnostic; they do not replace the preregistered multi-candidate screens.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmark'))
import numpy as np
import pandas as pd
from safebench.data import load_partition
from safebench.experiment import evaluate, dump, manifest, code_digest, digest
from safebench.metrics import summarize, paired_comparisons
from safebench.frontier import FRONTIER_NAMES
from safebench.validation import audit_outputs


def check(source):
    info=json.loads((source/'manifest.json').read_text())
    config=json.loads((source/'config.json').read_text())
    scenarios,sha=load_partition(ROOT/'data/SafeInvest_Synthetic_Data.csv')
    if info['phase']!='development' or info['partition_sha256']!=sha or info['code_sha256']!=code_digest():
        raise ValueError('Matched controls require the unchanged development program and data')
    fitted=json.loads((source/'fitted_models.json').read_text())
    original=pd.read_csv(source/'scenario_metrics.csv')
    cfg=dict(config,horizons=[30],sensitivity_costs=[2*config['cost_rate']])
    out=source/'matched_controls'; out.mkdir(exist_ok=False)
    frames=[]; records=[]; names=[]
    modes=['frontier_annual','frontier_hold','frontier_gate']
    for f in fitted:
        training=set(f['model']['train_ids'])
        valid=[s for s in scenarios if s.scenario_id not in training]
        for trial in range(len(config['frontier_research']['trials'])):
            for cost in [config['cost_rate'],2*config['cost_rate']]:
                m,r=evaluate(valid,f['model'],{n:{'frontier_trial':trial} for n in modes},cfg,
                    names=modes,horizons=[30],cost=cost,fold=f['fold'])
                replacements={n:n+'__trial'+str(trial) for n in modes}
                m['policy']=m.policy.replace(replacements); r['policy']=r.policy.replace(replacements)
                frames.append(m); records.append(r)
                names.extend(replacements.values())
    m=pd.concat(frames,ignore_index=True); r=pd.concat(records,ignore_index=True)
    m.to_csv(out/'scenario_metrics.csv',index=False)
    r.to_csv(out/'decisions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    dump(out/'config.json',cfg)
    run=manifest(cfg,sha,list(range(1,121)),'development')
    run.update(evaluated_policies=sorted(set(names)),source_run=str(source),
               purpose='Strict fixed-parameter controls; no model selection or refitting.')
    dump(out/'manifest.json',run)
    summarize(m).to_csv(out/'summary.csv',index=False)
    audit_outputs(out)
    traded=r[(r.period>1)&r.policy.str.startswith('frontier_gate')&r.action.isin(['rebalance','change_target'])]
    assert (traded.estimated_benefit>traded.estimated_cost+config['frontier_research']['buffer']).all()
    # The market replay reads only the existing development partition.
    rows=[]
    for name in FRONTIER_NAMES:
        candidate=original[(original.policy==name)&(original.horizon==30)&
            original.cost_rate.isin([config['cost_rate'],2*config['cost_rate']])]
        for mode in modes:
            controls=[]
            for f in fitted:
                trial=f['params'][name]['frontier_trial']
                controls.append(m[(m.fold==f['fold'])&(m.policy==mode+'__trial'+str(trial))])
            control=pd.concat(controls,ignore_index=True).copy(); control['policy']='matched_reference'
            block=pd.concat([candidate,control],ignore_index=True)
            p=paired_comparisons(block,config['seed'],config['bootstrap'],baseline='matched_reference')
            p=p.drop(columns=['family_lower','family_upper'])
            p['reference']=mode; p['interval_scope']='pointwise_only_exploratory'
            rows.append(p)
    pd.concat(rows,ignore_index=True).to_csv(source/'strict_matched_comparisons.csv',index=False)
    dump(out/'verification.json',dict(all_checks_passed=True,training_reused=True,
        source_manifest_sha256=digest(info),source_code_sha256=info['code_sha256'],
        exact_parameter_matching=True,gate_buffer_checked=True,holdout_used=False))
    print('Strict matched controls verified:',len(m),'paths;',len(r),'annual decisions.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description='按候选已选参数核验严格匹配参照，不重新训练或挑参数。')
    p.add_argument('run',type=Path); a=p.parse_args(); check(a.run.resolve())
