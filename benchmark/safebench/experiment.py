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
       'drawdown_conservative','drawdown_balanced','drawdown_aggressive',
       'drawdown_lifecycle_conservative','drawdown_lifecycle_balanced','drawdown_lifecycle_aggressive',
       'budget_lifecycle_balanced')
BUDGET_POLICY='budget_lifecycle_balanced'

def json_safe(obj):
    """Normalize numerical artifacts before strict JSON serialization and hashing."""
    if isinstance(obj,np.ndarray): return json_safe(obj.tolist())
    if isinstance(obj,np.generic): return json_safe(obj.item())
    if isinstance(obj,float) and not np.isfinite(obj): return None
    if isinstance(obj,dict): return {str(k):json_safe(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [json_safe(v) for v in obj]
    return obj

def dump(path,obj):
    Path(path).write_text(json.dumps(json_safe(obj),indent=2,sort_keys=True,allow_nan=False)+'\n')

def digest(obj):
    # Keep integrity hashing strict: malformed non-finite values must be rejected,
    # not normalized to the same digest as an explicit JSON null.
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

def evaluate(scenarios,model,params,config,names=None,cost=None,horizons=None,fold=0,keep_records=True,allow_train=False):
    names=names or POLICY_NAMES
    cost=config['cost_rate'] if cost is None else cost
    horizons=horizons or config['horizons']
    output,records=[],[]
    for name in names:
        # Ablations inherit the dynamic candidate's fitted parameters, never tune independently.
        source=BUDGET_POLICY if name.startswith('budget_lifecycle_') else name
        default_grid=config.get('budget_value_tuning_grid',config['tuning_grid']) if source==BUDGET_POLICY else config['tuning_grid']
        selected=params.get(source,default_grid[0])
        horizon_dependent=name.startswith(('drawdown_lifecycle_','budget_lifecycle_'))
        for s in scenarios:
            if s.scenario_id in model['train_ids'] and not allow_train:
                raise ValueError('Evaluation scenario overlaps fitted training scenarios')
            if horizon_dependent:
                # Run each horizon separately with its own policy instance
                for h in horizons:
                    horizon_params=selected['by_horizon'][str(h)] if 'by_horizon' in selected else selected
                    p=make_policy(name,model,horizon_params,config,horizon=h)
                    cfg=SimulationConfig(h,config['initial_wealth'],cost)
                    result=simulate(s,p,p.initial_target,cfg)
                    row=compute_metrics(result); row['fold']=fold
                    output.append(row)
                    # Save records for all horizons since each simulation is independent
                    if keep_records:
                        for r in result.records:
                            row=asdict(r)
                            for field in ('target_weights','pre_trade_holdings','post_trade_holdings','end_holdings'):
                                values=row.pop(field)
                                row.update({field+'_'+asset:v for asset,v in zip(('equity','bond','bill'),values)})
                            row.update(scenario_id=s.scenario_id,policy=name,horizon=h,cost_rate=cost,fold=fold)
                            row['requires_benefit_gate']=bool(getattr(p,'requires_benefit_gate',False))
                            row['buffer']=float(getattr(p,'buffer',0.0))
                            row['action']=r.action.value
                            row['reason_codes']=json.dumps(row['reason_codes'])
                            records.append(row)
            else:
                # Horizon-independent: run once at max horizon, truncate for shorter horizons
                p=make_policy(name,model,selected,config)
                cfg=SimulationConfig(max(horizons),config['initial_wealth'],cost)
                full=simulate(s,p,p.initial_target,cfg)
                for h in horizons:
                    result=replace(full,config=replace(cfg,horizon=h),records=full.records[:h])
                    row=compute_metrics(result); row['fold']=fold
                    output.append(row)
                if keep_records:
                    max_h=max(horizons)
                    for r in full.records:
                        row=asdict(r)
                        for field in ('target_weights','pre_trade_holdings','post_trade_holdings','end_holdings'):
                            values=row.pop(field)
                            row.update({field+'_'+asset:v for asset,v in zip(('equity','bond','bill'),values)})
                        row.update(scenario_id=s.scenario_id,policy=name,horizon=max_h,cost_rate=cost,fold=fold)
                        row['requires_benefit_gate']=bool(getattr(p,'requires_benefit_gate',False))
                        row['buffer']=float(getattr(p,'buffer',0.0))
                        row['action']=r.action.value
                        row['reason_codes']=json.dumps(row['reason_codes'])
                        records.append(row)
    return pd.DataFrame(output),pd.DataFrame(records)

def select_drawdown_params(valid_scores,train_risk,name,tolerance,grid,config):
    """Constrained selection for a drawdown policy: maximize the median validation
    CAGR among trials whose training-scenario breach-probability upper bound meets the
    suitability bar. When no trial qualifies, fall back to the least-violating trial
    and record it, instead of degenerating to an arbitrary argmax over -inf utilities."""
    g=valid_scores[valid_scores.policy==name]
    r=train_risk[train_risk.policy==name]
    stats={}
    for trial in sorted(set(g.trial.unique())|set(r.trial.unique())):
        v=g[g.trial==trial]; t=r[r.trial==trial]
        breaches=int((t.real_max_drawdown>tolerance).sum()) if len(t) else 0
        n=len(t)
        upper=binomial_upper(breaches,n,config['suitability_confidence']) if n else 1.0
        stats[int(trial)]=dict(mean_cagr=float(v.real_cagr.mean()),median_cagr=float(v.real_cagr.median()),
                               breaches=breaches,n=n,upper=float(upper),
                               feasible=bool(upper<=config['suitability_max_breach_probability']))
    feasible=sorted(i for i,s in stats.items() if s['feasible'])
    fallback=not feasible
    # The objective is the median throughout; use complexity only for exact ties.
    tie=lambda i:(grid[i].get('k',1.0),-grid[i].get('derisk_threshold',0.0),i)
    if feasible:
        selected=min(feasible,key=lambda i:(-stats[i]['median_cagr'],*tie(i)))
    else:
        selected=min(stats,key=lambda i:(stats[i]['upper'],-stats[i]['median_cagr'],*tie(i)))
    rows=[dict(policy=name,trial=trial,mean_utility=stats[trial]['mean_cagr'],
               best_trial=int(selected),selected_trial=int(selected),selected=trial==selected,
               params=json.dumps(json_safe(grid[trial]),sort_keys=True,allow_nan=False),
               selection_rule='min_breach_upper_then_max_median' if fallback else 'max_median_cagr_feasible',
               paired_gap_se=None,
               one_se_eligible=False,
               breach_count=stats[trial]['breaches'],train_paths=stats[trial]['n'],
               breach_upper_bound=stats[trial]['upper'],
               constraint_satisfied=stats[trial]['feasible'],
               constraint_fallback=fallback,
               median_cagr_valid=stats[trial]['median_cagr']) for trial in sorted(stats)]
    rows=json_safe(rows)
    return int(selected),rows

def drawdown_families(config):
    """Per-family drawdown tuning grids. Lifecycle policies declare their own search
    space: a global equity_shift tuned on the H=30 constraint double-shrinks the
    horizon-scaled equity cap and can pin short-horizon policies on the e_min floor.
    Missing lifecycle grid falls back to the shared one for backward compatibility."""
    plain=[n for n in TUNED if n.startswith('drawdown_') and not n.startswith('drawdown_lifecycle_')]
    lifecycle=[n for n in TUNED if n.startswith('drawdown_lifecycle_')]
    return [(plain,config.get('drawdown_tuning_grid',[])),
            (lifecycle,config.get('drawdown_lifecycle_tuning_grid',config.get('drawdown_tuning_grid',[])))]

def tune(scenarios,config,seed):
    by_id={s.scenario_id:s for s in scenarios}
    groups=folds(by_id,config['inner_folds'],seed)
    collected=[]; audits=[]
    gated_names=[n for n in TUNED if not n.startswith(('drawdown_','budget_lifecycle_'))]
    families=drawdown_families(config)
    budget_grid=config.get('budget_value_tuning_grid',[])
    if budget_grid and len(budget_grid)!=4:
        raise ValueError('Budget value search requires exactly four predeclared candidates')
    # Train-side risk constraint per family: each trial is evaluated on the full
    # outer-training set under the pooled training model. Breach statistics feed the
    # suitability-style constraint; explicitly authorized, never used for the objective.
    train_risk=pd.DataFrame()
    train_model=fit_model(scenarios,config)
    for names,grid in families:
        if not names or not grid: continue
        for trial,params in enumerate(grid):
            metrics,_=evaluate(scenarios,train_model,{n:params for n in names},config,
                               names=list(names),horizons=[30],keep_records=False,allow_train=True)
            metrics['trial']=trial
            train_risk=pd.concat([train_risk,metrics],ignore_index=True)
    for inner,valid_ids in enumerate(groups):
        train=[s for sid,s in by_id.items() if sid not in valid_ids]
        valid=[by_id[sid] for sid in valid_ids]
        model=fit_model(train,config)
        audits.append(dict(inner_fold=inner,train_ids=model['train_ids'],validation_ids=valid_ids))
        for trial,params in enumerate(config['tuning_grid']):
            metrics,_=evaluate(valid,model,{n:params for n in gated_names},config,names=list(gated_names),
                                horizons=[30],keep_records=False)
            metrics['trial']=trial
            collected.append(metrics)
        for names,grid in families:
            if not names or not grid: continue
            for trial,params in enumerate(grid):
                metrics,_=evaluate(valid,model,{n:params for n in names},config,names=list(names),
                                    horizons=[30],keep_records=False)
                metrics['trial']=trial
                collected.append(metrics)
        # Both risk screening and growth selection use inner out-of-fold paths at H.
        # Do not replay the training set or substitute H=30 for shorter clients.
        for trial,params in enumerate(budget_grid):
            metrics,_=evaluate(valid,model,{BUDGET_POLICY:params},config,names=[BUDGET_POLICY],
                                horizons=config['horizons'],keep_records=False)
            metrics['trial']=trial
            collected.append(metrics)
    scores=pd.concat(collected,ignore_index=True)
    # Standard utility for gated policies
    gated_mask=scores.policy.isin(gated_names)
    scores.loc[gated_mask,'utility']=scores.loc[gated_mask,'real_cagr']-config['selection_drawdown_penalty']*scores.loc[gated_mask,'real_max_drawdown']
    rows=[]; chosen={}
    for name in gated_names:
        if name not in scores.policy.values: continue
        g=scores[scores.policy==name]
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
            rows.append(dict(policy=name,trial=int(trial),mean_utility=float(means[trial]),
                             best_trial=best,paired_gap_se=se,one_se_eligible=ok))
        grid=config['tuning_grid']
        selected=max(eligible,key=lambda i:(grid[i]['buffer'],grid[i]['gamma'],-i))
        chosen[name]=grid[selected]
    tolerances={'conservative':0.15,'balanced':0.25,'aggressive':0.35}
    for names,grid in families:
        for name in names:
            if name not in scores.policy.values: continue
            if name.startswith('drawdown_lifecycle_'):
                suffix=name.split('_',2)[2]
            else:
                suffix=name.split('_',1)[1]
            selected,trial_rows=select_drawdown_params(scores,train_risk,name,tolerances[suffix],grid,config)
            rows.extend(trial_rows)
            chosen[name]=grid[selected]
    if budget_grid:
        by_horizon={}
        for h in config['horizons']:
            oof=scores[(scores.policy==BUDGET_POLICY)&(scores.horizon==h)]
            selected,trial_rows=select_drawdown_params(oof,oof,BUDGET_POLICY,.25,budget_grid,config)
            for row in trial_rows:
                row.update(horizon=int(h),risk_source='inner_oof',oof_paths=row['train_paths'])
            rows.extend(trial_rows)
            by_horizon[str(h)]=dict(budget_grid[selected])
        chosen[BUDGET_POLICY]={'by_horizon':by_horizon}
    return chosen,json_safe(rows),audits

def suitability(metrics,config):
    profiles=json.loads((ROOT/'configs/profiles.json').read_text())
    rows=[]; risk=[]
    policies=[(portfolio,portfolio+'_annual') for portfolio in PORTFOLIOS]
    if BUDGET_POLICY in set(metrics.policy): policies.append(('balanced',BUDGET_POLICY))
    for p in profiles:
        candidates=[]
        for portfolio,policy in policies:
            g=metrics[(metrics.policy==policy)&(metrics.horizon==p['horizon'])&
                      np.isclose(metrics.cost_rate,config['cost_rate'])]
            if len(g)==0: raise ValueError('Missing suitability horizon')
            k=int((g.real_max_drawdown>p['loss_tolerance']).sum())
            confidence=1-(1-config['suitability_confidence'])/len(policies)
            upper=binomial_upper(k,len(g),confidence)
            liquidity_ok=p['liquidity']!='high' or portfolio=='conservative'
            horizon_ok=portfolio!='growth' or p['horizon']>=15
            review_rate=float(g.review.mean()) if 'review' in g else 0.0
            service_ok=policy!=BUDGET_POLICY or review_rate<=config.get('budget_value_screen',{}).get('maximum_review_rate',.20)
            eligible=upper<=config['suitability_max_breach_probability'] and liquidity_ok and horizon_ok and service_ok
            risk.append(dict(profile=p['profile_id'],portfolio=portfolio,policy=policy,horizon=p['horizon'],n=len(g),
                breaches=k,breach_probability=k/len(g),upper_bound=upper,confidence=confidence,
                tolerance=p['loss_tolerance'],eligible=eligible,liquidity_ok=liquidity_ok,horizon_ok=horizon_ok,
                review_rate=review_rate,service_ok=service_ok,median_growth=g.real_cagr.median()))
            if eligible: candidates.append((g.real_cagr.median(),portfolio,policy))
        _,choice,policy=max(candidates) if candidates else (None,'human_review','human_review')
        reason=('Choose the eligible policy with highest median real growth; loss tolerance is not guaranteed.'
                if candidates else 'No portfolio passes the conservative development risk screen and liquidity/horizon rules.')
        rows.append(dict(**p,portfolio=choice,policy=policy,explanation=reason))
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
    if BUDGET_POLICY in set(metrics.policy) and 'budget_value_screen' in config:
        from .incremental import incremental_analysis
        incremental_analysis(out,config)
    dump(out/'runtime.json',dict(seconds=time.perf_counter()-started))
    from .report import generate_report
    generate_report(out)
    return out

def assignment_policy(profile):
    """Resolve exact frozen policy ids while accepting legacy portfolio-only bundles."""
    policy=profile.get('policy')
    if isinstance(policy,str) and policy: return policy
    portfolio=profile['portfolio']
    return 'human_review' if portfolio=='human_review' else portfolio+'_annual'

def selection_diagnostics(policy,assignments,trials,config):
    policies=[policy]+[assignment_policy(p) for p in assignments]
    uses_budget=any(p.startswith('budget_lifecycle_') for p in policies)
    horizons=[]
    if uses_budget:
        for h in config['horizons']:
            rows=[r for r in trials if r.get('policy')==BUDGET_POLICY and r.get('horizon')==h]
            fallback=any(r.get('constraint_fallback',False) for r in rows)
            horizons.append(dict(horizon=int(h),constraint_fallback=fallback,
                                 selected_trial=rows[0].get('selected_trial') if rows else None,
                                 diagnostics_available=bool(rows)))
    fallback=any(r['constraint_fallback'] for r in horizons)
    complete=all(r['diagnostics_available'] for r in horizons)
    status=('research_only_if_fallback' if fallback else
            'research_only_missing_diagnostics' if not complete else 'no_declared_horizon_fallback')
    return dict(status=status,uses_budget_candidate=uses_budget,by_horizon=horizons,
                deployment_established=False,
                note='Freezing preserves a research selection; it does not establish deployment suitability.')

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
    assignments=pd.read_csv(Path(development_run)/'suitability.csv').to_dict('records')
    for profile in assignments: profile['policy']=assignment_policy(profile)
    bundle=dict(schema=1,selected_policy=policy,config=config,model=model,params=params,
                code_sha256=code_digest(),dependencies=versions(),development_sha256=data_hash,
                development_run=str(Path(development_run).resolve()),final_scenarios=list(range(121,181)),
                final_comparators=['equal_annual','conservative_annual','balanced_annual','growth_annual','balanced_hold'],
                frozen_suitability=assignments,
                tuning_trials=trials,tuning_splits=audit,
                selection_diagnostics=selection_diagnostics(policy,assignments,trials,config),
                protocol='One final evaluation; no adjustments after exposure; failed executions also consume the lock.')
    bundle=json_safe(bundle)
    bundle['bundle_sha256']=digest(bundle)
    with Path(output).open('x') as f: json.dump(bundle,f,indent=2,sort_keys=True,allow_nan=False)

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
    frozen_policies=[assignment_policy(p) for p in b.get('frozen_suitability',[])]
    names=[name for name in dict.fromkeys(b['final_comparators']+[b['selected_policy']]+frozen_policies)
           if name!='human_review']
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
        assigned=assignment_policy(profile)
        if assigned!='human_review':
            group=metrics[(metrics.policy==assigned) &
                          (metrics.horizon==profile['horizon']) & np.isclose(metrics.cost_rate,config['cost_rate'])]
            if len(group)==0:
                raise ValueError('Missing final metrics for frozen suitability policy')
            row['final_breach_probability']=float((group.real_max_drawdown>profile['loss_tolerance']).mean())
            row['final_median_growth']=float(group.real_cagr.median())
        row['policy']=assigned
        assignment_rows.append(row)
    pd.DataFrame(assignment_rows).to_csv(out/'suitability.csv',index=False)
    from .validation import audit_outputs
    audit_outputs(out)
    from .report import generate_report
    generate_report(out)
