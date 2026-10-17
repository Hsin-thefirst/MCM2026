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
from .metrics import compute_metrics, summarize, paired_comparisons, pareto_flags, binomial_upper, expected_shortfall
from .policies import fit_model, make_policy, POLICY_NAMES, PORTFOLIOS, policy_names
from . import plan_a, plan_b, plan_c

BASE=ROOT/'benchmark'
TUNED=('balanced_gate','cost_aware','adaptive','adaptive_review')

def dump(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,allow_nan=False).encode()).hexdigest()

def code_digest():
    files=sorted((BASE/'safebench').glob('*.py'))+[BASE/'run.py',ROOT/'src/safeinvest/contracts.py',ROOT/'src/safeinvest/__init__.py']
    return digest({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})

def versions():
    return {p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','matplotlib']}

def manifest(config,data_hash,ids,phase,names=None):
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),phase=phase,scenario_ids=ids,
                partition_sha256=data_hash,config_sha256=digest(config),code_sha256=code_digest(),
                python=platform.python_version(),dependencies=versions(),seed=config['seed'],
                git_commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True).stdout.strip(),
                holdout_used=phase=='final',evaluated_policies=names or policy_names(config))

def evaluate(scenarios,model,params,config,names=None,cost=None,horizons=None,fold=0,keep_records=True):
    names=names or policy_names(config)
    cost=config['cost_rate'] if cost is None else cost
    horizons=horizons or config['horizons']
    output,records=[],[]
    for name in names:
        research = name in plan_a.POLICIES+plan_b.POLICIES+plan_c.POLICIES
        active_horizons = [h for h in horizons if name not in plan_c.POLICIES or h in config.get('plan_c', {}).get('evaluation_horizons', config['horizons'])]
        default=plan_a.parameter_grid(config)[0] if research else config['tuning_grid'][0]
        selected=params.get(name,default)
        for s in scenarios:
            if s.scenario_id in model['train_ids']:
                raise ValueError('Evaluation scenario overlaps fitted training scenarios')
            simulation_horizons = active_horizons if research else [max(horizons)]
            for simulation_horizon in simulation_horizons:
                p=make_policy(name,model,selected,config,horizon=simulation_horizon)
                cfg=SimulationConfig(simulation_horizon,config['initial_wealth'],cost)
                full=simulate(s,p,p.initial_target,cfg)
                for h in ([simulation_horizon] if research else horizons):
                    result=replace(full,config=replace(cfg,horizon=h),records=full.records[:h])
                    row=compute_metrics(result)
                    row.update(fold=fold,simulation_horizon=simulation_horizon,
                               policy_decision_calls=getattr(p,'decision_calls',0),
                               policy_inference_seconds=getattr(p,'inference_seconds',0.))
                    if research:
                        product=name.removeprefix('plan_a_') if name in ['plan_a_'+x for x in plan_a.PRODUCTS] else 'balanced'
                        entry=model['plan_a']['entries'][f"{product}:{h}:{selected['risk_ratio']:g}"]
                        row['plan_a_risk_budget']=entry['risk_budget']
                        row['plan_a_validation_budget']=entry.get('validation_budget',entry['risk_budget'])
                    output.append(row)
                if keep_records:
                    for r in full.records:
                        row=asdict(r)
                        for field in ('target_weights','pre_trade_holdings','post_trade_holdings','end_holdings'):
                            values=row.pop(field)
                            row.update({field+'_'+asset:v for asset,v in zip(('equity','bond','bill'),values)})
                        row.update(scenario_id=s.scenario_id,policy=name,cost_rate=cost,fold=fold,simulation_horizon=simulation_horizon,
                                   required_buffer=getattr(p,'buffer',0.),requires_benefit_gate=getattr(p,'requires_benefit_gate',False))
                        row['action']=r.action.value
                        row['reason_codes']=json.dumps(row['reason_codes'])
                        row['policy_diagnostics']=json.dumps(row['policy_diagnostics'],allow_nan=False)
                        records.append(row)
    return pd.DataFrame(output),pd.DataFrame(records)

def tune(scenarios,config,seed,refit_model=None):
    by_id={s.scenario_id:s for s in scenarios}
    groups=folds(by_id,config['inner_folds'],seed)
    collected=[]; audits=[]; plan_scores=[]; unavailable={}; inner_models=[]
    if plan_a.enabled(config) and refit_model is None:
        refit_model=fit_model(scenarios,config)
    if refit_model is not None and set(refit_model['train_ids'])!=set(by_id):
        raise ValueError('Refit model must use exactly the tuning population')
    for inner,valid_ids in enumerate(groups):
        train=[s for sid,s in by_id.items() if sid not in valid_ids]
        valid=[by_id[sid] for sid in valid_ids]
        model=fit_model(train,config)
        inner_models.append((model,valid))
        audits.append(dict(inner_fold=inner,train_ids=model['train_ids'],validation_ids=valid_ids))
        if plan_a.enabled(config):
            audits[-1]['infeasible_allocations']={k:v for k,v in model['plan_a']['entries'].items() if v['status']=='infeasible'}
        for trial,params in enumerate(config['tuning_grid']):
            metrics,_=evaluate(valid,model,{n:params for n in TUNED},config,names=list(TUNED),
                                horizons=[30],keep_records=False)
            metrics['trial']=trial
            collected.append(metrics)
        if plan_a.enabled(config):
            # Reuse only this inner-training fit. Every product is evaluated at its actual horizons.
            for trial, params in enumerate(plan_a.parameter_grid(config)):
                for product in plan_a.PRODUCTS:
                    name='plan_a_'+product
                    if not plan_a.allocation_available(model,product,params,config['horizons']):
                        unavailable.setdefault((name,trial),[]).append(inner)
                        continue
                    metrics,_=evaluate(valid,model,{name:params},config,names=[name],keep_records=False)
                    controls=[]
                    for s in valid:
                        for h in config['horizons']:
                            p=plan_a.PlanAPolicy(name,model,params,config,h,product=product,mode='annual',review=False)
                            r=simulate(s,p,p.initial_target,SimulationConfig(h,config['initial_wealth'],config['cost_rate']))
                            controls.append(dict(scenario_id=s.scenario_id,horizon=h,
                                                 annual_growth=compute_metrics(r)['real_cagr']))
                    metrics=metrics.merge(pd.DataFrame(controls),on=['scenario_id','horizon'],validate='one_to_one')
                    metrics['trial']=trial
                    plan_scores.append(metrics)
    scores=pd.concat(collected,ignore_index=True)
    scores['utility']=scores.real_cagr-config['selection_drawdown_penalty']*scores.real_max_drawdown
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
            rows.append(dict(policy=name,trial=int(trial),mean_utility=float(means[trial]),
                             best_trial=best,paired_gap_se=se,one_se_eligible=ok))
        selected=max(eligible,key=lambda i:(config['tuning_grid'][i]['buffer'],config['tuning_grid'][i]['gamma'],-i))
        chosen[name]=config['tuning_grid'][selected]
    if plan_a.enabled(config):
        trials=plan_a.parameter_grid(config)
        scores=pd.concat(plan_scores,ignore_index=True) if plan_scores else pd.DataFrame()
        for product in plan_a.PRODUCTS:
            name='plan_a_'+product
            choices=[]
            for trial, trial_params in enumerate(trials):
                failed=unavailable.get((name,trial),[])
                refit_ok=plan_a.allocation_available(refit_model,product,trial_params,config['horizons'])
                if failed or not refit_ok:
                    rows.append(dict(policy=name,trial=trial,selection_feasible=False,allocation_feasible=False,
                        inner_infeasible_folds=json.dumps(failed),refit_feasible=refit_ok))
                    continue
                block=scores[(scores.policy==name)&(scores.trial==trial)]
                if len(block)!=len(scenarios)*len(config['horizons']):
                    raise ValueError('Incomplete inner validation; candidate cannot be ranked')
                violation, details=plan_a.selection_constraints(block,config)
                feasible=violation<=1e-10
                score=float(np.log1p(block.real_cagr).mean())
                choices.append((int(trial),feasible,violation,score))
                rows.append(dict(policy=name,trial=int(trial),mean_utility=score,
                                 selection_feasible=feasible,constraint_violation=violation,
                                 allocation_feasible=True,refit_feasible=True,horizon_constraints=json.dumps(details)))
            if not choices:
                raise RuntimeError('No feasible strategic allocation across inner fits and refit: '+name)
            winner=min(choices,key=lambda x:(not x[1],x[2] if not x[1] else 0.,-x[3],x[0]))
            chosen[name]=dict(trials[winner[0]],selection_feasible=winner[1],constraint_violation=winner[2])
        for name in plan_a.CONTROLS:
            chosen[name]=dict(chosen['plan_a_balanced'])
    if plan_b.enabled(config):
        anchor_params = chosen['plan_a_balanced']
        for name in plan_b.POLICIES: chosen[name] = dict(anchor_params)
        for family in ('narrow', 'wide'):
            blocks = []
            for inner_model, valid in inner_models:
                names = [f'plan_b_{family}_{label}' for label in ('0', '025', '05')]
                block, _ = evaluate(valid, inner_model, {n:anchor_params for n in names}, config, names=names, keep_records=False)
                reference = block[block.policy == names[0]][['scenario_id','horizon','real_cagr']].rename(columns={'real_cagr':'annual_growth'})
                blocks.append(block.merge(reference,on=['scenario_id','horizon'],validate='many_to_one'))
            scores = pd.concat(blocks, ignore_index=True)
            choices = []
            for index, eta in enumerate((0., .25, .5)):
                name = f'plan_b_{family}_{("0", "025", "05")[index]}'
                block = scores[scores.policy == name]
                violation, details = plan_a.selection_constraints(block, config)
                score = float(np.log1p(block.real_cagr).mean())
                choices.append((index, violation <= 1e-10, violation, score))
                rows.append(dict(policy='plan_b_'+family, trial=index, eta=eta, mean_utility=score,
                    selection_feasible=violation<=1e-10, constraint_violation=violation,
                    horizon_constraints=json.dumps(details)))
            feasible = [x for x in choices if x[1]]
            best = min(choices, key=lambda x:(not x[1], x[2] if not x[1] else 0., -x[3], x[0]))
            # Paired one-SE prefers the smallest tilt among feasible near-best choices.
            if feasible:
                pivot = scores.assign(log_growth=np.log1p(scores.real_cagr)).pivot(index=['scenario_id','horizon'], columns='policy', values='log_growth')
                eligible = []
                for candidate in feasible:
                    a=f'plan_b_{family}_{("0", "025", "05")[best[0]]}'
                    b=f'plan_b_{family}_{("0", "025", "05")[candidate[0]]}'
                    # Scenario, not year/horizon, is the independent resampling unit.
                    diff=(pivot[a]-pivot[b]).groupby('scenario_id').mean()
                    if diff.mean() <= diff.std(ddof=1)/np.sqrt(len(diff))+1e-12: eligible.append(candidate)
                best=min(eligible,key=lambda x:x[0])
            chosen['plan_b_'+family]=dict(anchor_params,eta=(0.,.25,.5)[best[0]],
                signal_selection_feasible=best[1],signal_constraint_violation=best[2])
    if plan_c.enabled(config):
        for name in plan_c.POLICIES: chosen[name]=dict(chosen['plan_a_balanced'])
    return chosen,rows,audits

def suitability(metrics,config,policy_map=None):
    profiles=json.loads((ROOT/'configs/profiles.json').read_text())
    rows=[]; risk=[]
    if policy_map is None:
        use_plan=plan_a.enabled(config) and all('plan_a_'+p in set(metrics.policy) for p in PORTFOLIOS)
        policy_map={p:('plan_a_'+p if use_plan else p+'_annual') for p in PORTFOLIOS}
    for p in profiles:
        candidates=[]
        for portfolio in PORTFOLIOS:
            policy=policy_map[portfolio]
            g=metrics[(metrics.policy==policy)&(metrics.horizon==p['horizon'])&
                      np.isclose(metrics.cost_rate,config['cost_rate'])]
            if len(g)==0: raise ValueError('Missing suitability horizon')
            k=int((g.real_max_drawdown>p['loss_tolerance']).sum())
            # Bonferroni over 3 candidate portfolios for each profile, to allow portfolio selection.
            confidence=1-(1-config['suitability_confidence'])/3
            upper=binomial_upper(k,len(g),confidence)
            liquidity_ok=p['liquidity']!='high' or portfolio=='conservative'
            horizon_ok=portfolio!='growth' or p['horizon']>=15
            eligible=upper<=config['suitability_max_breach_probability'] and liquidity_ok and horizon_ok
            evidence='OOF diagnostic; refit coverage not guaranteed' if policy.startswith(('plan_a_', 'plan_b_', 'plan_c_')) else 'Fixed-policy scenario sample'
            risk.append(dict(profile=p['profile_id'],portfolio=portfolio,policy=policy,evidence=evidence,horizon=p['horizon'],n=len(g),
                breaches=k,breach_probability=k/len(g),upper_bound=upper,confidence=confidence,
                tolerance=p['loss_tolerance'],eligible=eligible,liquidity_ok=liquidity_ok,horizon_ok=horizon_ok,
                median_growth=g.real_cagr.median()))
            if eligible: candidates.append((g.real_cagr.median(),portfolio))
        choice=max(candidates)[1] if candidates else 'human_review'
        reason=('Choose the eligible portfolio with highest median real growth; loss tolerance is not guaranteed.'
                if candidates else 'No portfolio passes the conservative development risk screen and liquidity/horizon rules.')
        rows.append(dict(**p,portfolio=choice,policy=policy_map.get(choice,'human_review'),explanation=reason))
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
    all_metrics=[]; all_records=[]; audit=[]; trial_rows=[]; fitted=[]; forecasts=[]
    for outer,valid_ids in enumerate(folds(ids,config['outer_folds'],config['seed'])):
        train=[s for s in scenarios if s.scenario_id not in valid_ids]
        valid=[s for s in scenarios if s.scenario_id in valid_ids]
        model=fit_model(train,config)
        params,trials,inner_audit=tune(train,config,config['seed']+outer+1,refit_model=model)
        print('Outer fold',outer+1,'selected',params,flush=True)
        if plan_b.enabled(config):
            from .signals import forecast, real_returns
            for scenario in valid:
                for i in range(2, len(scenario.observations)):
                    for signal in ('ridge','period','shifted'):
                        delta, fallback, prediction=forecast(scenario.observations[:i],model['signals'],signal,config['plan_b']['ood_limit'])
                        forecasts.append(dict(fold=outer,scenario_id=scenario.scenario_id,period=i+1,signal_model=signal,
                            actual=float(real_returns(scenario.observations[i])[0]-real_returns(scenario.observations[i])[2]),
                            prediction=prediction,constant=model['signals']['models'][signal]['target_mean'],delta=delta,fallback=fallback))
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
    if forecasts: pd.DataFrame(forecasts).to_csv(out/'oof_predictions.csv',index=False)
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
    if config.get('generate_report',True):
        from .report import generate_report
        generate_report(out)
    return out

def freeze(config,data_path,policy,output,development_run):
    if policy not in policy_names(config):
        raise ValueError('Unknown candidate')
    if policy in plan_c.POLICIES and set(config['plan_c'].get('evaluation_horizons', config['horizons'])) != set(config['horizons']):
        raise ValueError('Plan C screening is incomplete; full-horizon development required before freeze')
    prior=json.loads((Path(development_run)/'manifest.json').read_text())
    if prior['phase']!='development' or prior['code_sha256']!=code_digest() or prior['config_sha256']!=digest(config):
        raise ValueError('Development run must match current code and configuration')
    scenarios,data_hash=load_partition(data_path)
    if data_hash!=prior['partition_sha256']:
        raise ValueError('Development data changed')
    model=fit_model(scenarios,config)
    params,trials,audit=tune(scenarios,config,config['seed']+1000,refit_model=model)
    assignments=pd.read_csv(Path(development_run)/'suitability.csv').to_dict('records')
    if plan_a.enabled(config) and (Path(development_run)/'scenario_metrics.csv').exists():
        mapping={p:'plan_a_'+p for p in PORTFOLIOS}
        if policy in plan_a.CONTROLS+plan_b.POLICIES+plan_c.POLICIES or policy=='plan_a_balanced': mapping['balanced']=policy
        assignments=suitability(pd.read_csv(Path(development_run)/'scenario_metrics.csv'),config,mapping)[0].to_dict('records')
    comparators=['equal_annual','conservative_annual','balanced_annual','growth_annual','balanced_hold']
    if plan_a.enabled(config): comparators+=plan_a.POLICIES
    if plan_b.enabled(config): comparators+=plan_b.POLICIES
    if plan_c.enabled(config) and set(config['plan_c'].get('evaluation_horizons',config['horizons']))==set(config['horizons']):
        comparators+=plan_c.POLICIES
    bundle=dict(schema=1,selected_policy=policy,config=config,model=model,params=params,
                code_sha256=code_digest(),dependencies=versions(),development_sha256=data_hash,
                development_run=str(Path(development_run).resolve()),final_scenarios=list(range(121,181)),
                final_comparators=comparators,
                frozen_suitability=assignments,
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
    dump(out/'manifest.json',manifest(config,final_hash,[s.scenario_id for s in scenarios],'final',names))
    dump(out/'config.json',config); dump(out/'frozen_bundle.json',raw)
    # Evaluate the development-frozen assignments; do not reassign using final outcomes.
    assignment_rows=[]
    for profile in b['frozen_suitability']:
        row=dict(profile)
        if profile['portfolio']!='human_review':
            group=metrics[(metrics.policy==profile.get('policy',profile['portfolio']+'_annual')) &
                          (metrics.horizon==profile['horizon']) & np.isclose(metrics.cost_rate,config['cost_rate'])]
            row['final_breach_probability']=float((group.real_max_drawdown>profile['loss_tolerance']).mean())
            row['final_median_growth']=float(group.real_cagr.median())
        assignment_rows.append(row)
    pd.DataFrame(assignment_rows).to_csv(out/'suitability.csv',index=False)
    from .validation import audit_outputs
    audit_outputs(out)
    if config.get('generate_report',True):
        from .report import generate_report
        generate_report(out)
