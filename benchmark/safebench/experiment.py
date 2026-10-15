"""Scenario-level nested CV, audit artifacts, suitability and frozen final workflow."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import pandas as pd
from safeinvest.contracts import SimulationConfig
from . import ROOT
from .data import load_partition, folds, real_rows
from .engine import simulate
from .metrics import compute_metrics, summarize, paired_comparisons, pareto_flags, binomial_upper
from .policies import fit_model, make_policy, POLICY_NAMES, PORTFOLIOS

BASE=ROOT/'benchmark'
TUNED=('balanced_gate','cost_aware','adaptive','adaptive_review',
       'drawdown_conservative','drawdown_balanced','drawdown_aggressive')

def dump(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,allow_nan=False).encode()).hexdigest()

def code_digest():
    files=sorted((BASE/'safebench').glob('*.py'))+[BASE/'run.py',ROOT/'src/safeinvest/contracts.py',ROOT/'src/safeinvest/__init__.py']
    return digest({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})

def versions():
    return {p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','matplotlib']}

def manifest(config,data_hash,ids,phase):
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),phase=phase,scenario_ids=ids,
                partition_sha256=data_hash,config_sha256=digest(config),code_sha256=code_digest(),
                python=platform.python_version(),dependencies=versions(),seed=config['seed'],
                git_commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True).stdout.strip(),
                holdout_used=phase=='final')

def evaluate(scenarios,model,params,config,names=None,cost=None,horizons=None,fold=0,keep_records=True):
    names=names or POLICY_NAMES
    cost=config['cost_rate'] if cost is None else cost
    horizons=horizons or config['horizons']
    output,records=[],[]
    for name in names:
        selected=params.get(name,config['tuning_grid'][0])
        for s in scenarios:
            if s.scenario_id in model['train_ids']:
                raise ValueError('Evaluation scenario overlaps fitted training scenarios')
            p=make_policy(name,model,selected,config)
            cfg=SimulationConfig(max(horizons),config['initial_wealth'],cost)
            full=simulate(s,p,p.initial_target,cfg)
            # All current policies are horizon-independent. Future horizon-conditioned policies must rerun each H.
            for h in horizons:
                result=replace(full,config=replace(cfg,horizon=h),records=full.records[:h])
                row=compute_metrics(result); row['fold']=fold
                output.append(row)
            if keep_records:
                for r in full.records:
                    row=asdict(r)
                    for field in ('target_weights','pre_trade_holdings','post_trade_holdings','end_holdings'):
                        values=row.pop(field)
                        row.update({field+'_'+asset:v for asset,v in zip(('equity','bond','bill'),values)})
                    row.update(scenario_id=s.scenario_id,policy=name,cost_rate=cost,fold=fold)
                    row['action']=r.action.value
                    row['reason_codes']=json.dumps(row['reason_codes'])
                    records.append(row)
    return pd.DataFrame(output),pd.DataFrame(records)

def tune(scenarios,config,seed):
    by_id={s.scenario_id:s for s in scenarios}
    groups=folds(by_id,config['inner_folds'],seed)
    collected=[]; audits=[]
    for inner,valid_ids in enumerate(groups):
        train=[s for sid,s in by_id.items() if sid not in valid_ids]
        valid=[by_id[sid] for sid in valid_ids]
        model=fit_model(train,config)
        audits.append(dict(inner_fold=inner,train_ids=model['train_ids'],validation_ids=valid_ids))
        # Separate tuning grids for different policy families
        drawdown_names=[n for n in TUNED if n.startswith('drawdown_')]
        gated_names=[n for n in TUNED if not n.startswith('drawdown_')]
        for trial,params in enumerate(config['tuning_grid']):
            metrics,_=evaluate(valid,model,{n:params for n in gated_names},config,names=list(gated_names),
                                horizons=[30],keep_records=False)
            metrics['trial']=trial
            collected.append(metrics)
        for trial,params in enumerate(config.get('drawdown_tuning_grid',[])):
            metrics,_=evaluate(valid,model,{n:params for n in drawdown_names},config,names=list(drawdown_names),
                                horizons=[30],keep_records=False)
            metrics['trial']=trial
            collected.append(metrics)
    scores=pd.concat(collected,ignore_index=True)
    # Risk-matched tuning objective for drawdown policies
    for name in drawdown_names:
        if name not in scores.policy.values: continue
        tolerance={'conservative':0.15,'balanced':0.25,'aggressive':0.35}[name.split('_',1)[1]]
        mask=scores.policy==name
        scores.loc[mask,'utility']=np.where(
            scores.loc[mask,'real_max_drawdown']<=tolerance+0.05,
            scores.loc[mask,'real_cagr'],
            -np.inf)
    # Standard utility for gated policies
    gated_mask=scores.policy.isin(gated_names)
    scores.loc[gated_mask,'utility']=scores.loc[gated_mask,'real_cagr']-config['selection_drawdown_penalty']*scores.loc[gated_mask,'real_max_drawdown']
    rows=[]; chosen={}
    for name,g in scores.groupby('policy'):
        means=g.groupby('trial').utility.mean()
        best=int(means.idxmax())
        # Paired one-standard-error rule: prefer higher buffer, then higher gamma among near-best candidates.
        pivot=g.pivot(index='scenario_id',columns='trial',values='utility')
        eligible=[]
        for trial in means.index:
            diff=pivot[best]-pivot[trial]
            se=float(diff.std(ddof=1)/np.sqrt(len(diff)))
            ok=float(diff.mean()) <= se+1e-12
            if ok: eligible.append(int(trial))
            grid=config.get('drawdown_tuning_grid' if name.startswith('drawdown_') else 'tuning_grid',[])
            rows.append(dict(policy=name,trial=int(trial),mean_utility=float(means[trial]),
                             best_trial=best,paired_gap_se=se,one_se_eligible=ok))
        grid=config.get('drawdown_tuning_grid' if name.startswith('drawdown_') else 'tuning_grid',[])
        if name.startswith('drawdown_'):
            selected=max(eligible,key=lambda i:(grid[i]['buffer'],grid[i].get('k',1.0),-i))
        else:
            selected=max(eligible,key=lambda i:(grid[i]['buffer'],grid[i]['gamma'],-i))
        chosen[name]=grid[selected]
    return chosen,rows,audits

def suitability(metrics,config):
    profiles=json.loads((ROOT/'configs/profiles.json').read_text())
    rows=[]; risk=[]
    for p in profiles:
        candidates=[]
        for portfolio in PORTFOLIOS:
            g=metrics[(metrics.policy==portfolio+'_annual')&(metrics.horizon==p['horizon'])&
                      np.isclose(metrics.cost_rate,config['cost_rate'])]
            if len(g)==0: raise ValueError('Missing suitability horizon')
            k=int((g.real_max_drawdown>p['loss_tolerance']).sum())
            # Bonferroni over 3 candidate portfolios for each profile, to allow portfolio selection.
            confidence=1-(1-config['suitability_confidence'])/3
            upper=binomial_upper(k,len(g),confidence)
            liquidity_ok=p['liquidity']!='high' or portfolio=='conservative'
            horizon_ok=portfolio!='growth' or p['horizon']>=15
            eligible=upper<=config['suitability_max_breach_probability'] and liquidity_ok and horizon_ok
            risk.append(dict(profile=p['profile_id'],portfolio=portfolio,horizon=p['horizon'],n=len(g),
                breaches=k,breach_probability=k/len(g),upper_bound=upper,confidence=confidence,
                tolerance=p['loss_tolerance'],eligible=eligible,liquidity_ok=liquidity_ok,horizon_ok=horizon_ok,
                median_growth=g.real_cagr.median()))
            if eligible: candidates.append((g.real_cagr.median(),portfolio))
        choice=max(candidates)[1] if candidates else 'human_review'
        reason=('Choose the eligible portfolio with highest median real growth; loss tolerance is not guaranteed.'
                if candidates else 'No portfolio passes the conservative development risk screen and liquidity/horizon rules.')
        rows.append(dict(**p,portfolio=choice,explanation=reason))
    return pd.DataFrame(rows),pd.DataFrame(risk)

def run_development(config,data_path,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    scenarios,data_hash=load_partition(data_path)
    dump(out/'config.json',config)
    ids=[s.scenario_id for s in scenarios]
    dump(out/'manifest.json',manifest(config,data_hash,ids,'development'))
    rows=real_rows(scenarios)
    pd.DataFrame(rows,columns=['equity_real','bond_real','bill_real']).describe(percentiles=[.01,.1,.5,.9,.99]).to_csv(out/'data_summary.csv')
    pd.DataFrame(rows,columns=['equity_real','bond_real','bill_real']).corr().to_csv(out/'data_correlations.csv')
    all_metrics=[]; all_records=[]; audit=[]; trial_rows=[]; fitted=[]
    for outer,valid_ids in enumerate(folds(ids,config['outer_folds'],config['seed'])):
        train=[s for s in scenarios if s.scenario_id not in valid_ids]
        valid=[s for s in scenarios if s.scenario_id in valid_ids]
        params,trials,inner_audit=tune(train,config,config['seed']+outer+1)
        model=fit_model(train,config)
        print('Outer fold',outer+1,'selected',params,flush=True)
        fitted.append(dict(fold=outer,model=model,params=params))
        audit.append(dict(outer_fold=outer,train_ids=model['train_ids'],validation_ids=valid_ids,inner=inner_audit))
        trial_rows.extend(dict(outer_fold=outer,**r) for r in trials)
        metrics,records=evaluate(valid,model,params,config,fold=outer)
        all_metrics.append(metrics); all_records.append(records)
        for cost in config['sensitivity_costs']:
            metrics,records=evaluate(valid,model,params,config,cost=cost,horizons=[30],fold=outer)
            all_metrics.append(metrics); all_records.append(records)
    metrics=pd.concat(all_metrics,ignore_index=True)
    records=pd.concat(all_records,ignore_index=True)
    metrics.to_csv(out/'scenario_metrics.csv',index=False)
    records.to_csv(out/'decisions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    dump(out/'splits.json',audit); dump(out/'fitted_models.json',fitted)
    pd.DataFrame(trial_rows).to_csv(out/'tuning_trials.csv',index=False)
    summary=pareto_flags(summarize(metrics)); summary.to_csv(out/'summary.csv',index=False)
    paired=paired_comparisons(metrics,config['seed'],config['bootstrap']); paired.to_csv(out/'paired_comparisons.csv',index=False)
    assignments,risks=suitability(metrics,config)
    assignments.to_csv(out/'suitability.csv',index=False); risks.to_csv(out/'suitability_risk.csv',index=False)
    from .validation import comparisons_and_screen, audit_outputs
    comparisons_and_screen(out,config)
    audit_outputs(out)
    dump(out/'runtime.json',dict(seconds=time.perf_counter()-started))
    from .report import generate_report
    generate_report(out)
    return out

def freeze(config,data_path,policy,output,development_run):
    if policy not in POLICY_NAMES:
        raise ValueError('Unknown candidate')
    prior=json.loads((Path(development_run)/'manifest.json').read_text())
    if prior['phase']!='development' or prior['code_sha256']!=code_digest() or prior['config_sha256']!=digest(config):
        raise ValueError('Development run must match current code and configuration')
    scenarios,data_hash=load_partition(data_path)
    if data_hash!=prior['partition_sha256']:
        raise ValueError('Development data changed')
    params,trials,audit=tune(scenarios,config,config['seed']+1000)
    model=fit_model(scenarios,config)
    bundle=dict(schema=1,selected_policy=policy,config=config,model=model,params=params,
                code_sha256=code_digest(),dependencies=versions(),development_sha256=data_hash,
                development_run=str(Path(development_run).resolve()),final_scenarios=list(range(121,181)),
                final_comparators=['equal_annual','conservative_annual','balanced_annual','growth_annual','balanced_hold'],
                frozen_suitability=pd.read_csv(Path(development_run)/'suitability.csv').to_dict('records'),
                tuning_trials=trials,tuning_splits=audit,
                protocol='One final evaluation; no adjustments after exposure; failed executions also consume the lock.')
    bundle['bundle_sha256']=digest(bundle)
    with Path(output).open('x') as f: json.dump(bundle,f,indent=2,sort_keys=True)

def verify_bundle(bundle):
    b=dict(bundle); claimed=b.pop('bundle_sha256')
    if digest(b)!=claimed or b['code_sha256']!=code_digest() or b['dependencies']!=versions():
        raise ValueError('Frozen bundle integrity, code or environment mismatch')
    if b['final_scenarios']!=list(range(121,181)):
        raise ValueError('Invalid final partition')
    return b

def claim_final_lock(path,bundle_hash):
    # O_EXCL semantics: a different output directory cannot bypass this project-wide guard.
    with Path(path).open('x') as f:
        json.dump(dict(bundle_sha256=bundle_hash,started=datetime.now(timezone.utc).isoformat()),f)

def run_final(bundle_path,data_path,out,acknowledge):
    if not acknowledge:
        raise PermissionError('Explicit --acknowledge-final required')
    raw=json.loads(Path(bundle_path).read_text()); b=verify_bundle(raw)
    _,dev_hash=load_partition(data_path)
    if dev_hash!=b['development_sha256']: raise ValueError('Development data changed')
    out=Path(out)
    if out.exists(): raise FileExistsError(out)
    claim_final_lock(BASE/'FINAL_EVALUATION_LOCK.json',raw['bundle_sha256'])
    out.mkdir(parents=True)
    scenarios,final_hash=load_partition(data_path,'final',authorized=True)
    config=b['config']
    names=list(dict.fromkeys(b['final_comparators']+[b['selected_policy']]))
    metrics,records=evaluate(scenarios,b['model'],b['params'],config,names=names)
    # Predeclared cost robustness run, all generated in the same one-shot operation.
    all_metrics=[metrics]; all_records=[records]
    for cost in config['sensitivity_costs']:
        m,r=evaluate(scenarios,b['model'],b['params'],config,names=names,cost=cost,horizons=[30])
        all_metrics.append(m); all_records.append(r)
    metrics=pd.concat(all_metrics,ignore_index=True); records=pd.concat(all_records,ignore_index=True)
    metrics.to_csv(out/'scenario_metrics.csv',index=False)
    records.to_csv(out/'decisions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pareto_flags(summarize(metrics)).to_csv(out/'summary.csv',index=False)
    paired_comparisons(metrics,config['seed'],config['bootstrap']).to_csv(out/'paired_comparisons.csv',index=False)
    dump(out/'manifest.json',manifest(config,final_hash,[s.scenario_id for s in scenarios],'final'))
    dump(out/'config.json',config); dump(out/'frozen_bundle.json',raw)
    # Evaluate the development-frozen assignments; do not reassign using final outcomes.
    assignment_rows=[]
    for profile in b['frozen_suitability']:
        row=dict(profile)
        if profile['portfolio']!='human_review':
            group=metrics[(metrics.policy==profile['portfolio']+'_annual') &
                          (metrics.horizon==profile['horizon']) & np.isclose(metrics.cost_rate,config['cost_rate'])]
            row['final_breach_probability']=float((group.real_max_drawdown>profile['loss_tolerance']).mean())
            row['final_median_growth']=float(group.real_cagr.median())
        assignment_rows.append(row)
    pd.DataFrame(assignment_rows).to_csv(out/'suitability.csv',index=False)
    from .validation import audit_outputs
    audit_outputs(out)
    from .report import generate_report
    generate_report(out)
