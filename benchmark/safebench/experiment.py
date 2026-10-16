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
from .experts import EXPERT_NAMES, LEARNED_NAMES, SEED_NAMES, enabled, fit_expert_bundle
from .frontier import FRONTIER_NAMES, MATCHED_NAMES, enabled as frontier_enabled
from .cushion import CUSHION_NAMES, CALIBRATED_NAMES, enabled as cushion_enabled

BASE=ROOT/'benchmark'
TUNED=('balanced_gate','cost_aware','adaptive','adaptive_review')

def policy_names(config):
    return POLICY_NAMES+(list(EXPERT_NAMES) if enabled(config) else [])+(list(FRONTIER_NAMES+MATCHED_NAMES) if frontier_enabled(config) else [])+(list(CUSHION_NAMES) if cushion_enabled(config) else [])+(list(CALIBRATED_NAMES) if config.get('calibrated_research',{}).get('enabled',False) else [])

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
    names=names or policy_names(config)
    cost=config['cost_rate'] if cost is None else cost
    horizons=horizons or config['horizons']
    output,records=[],[]
    for name in names:
        selected=params.get(name,config['tuning_grid'][0])
        if name in MATCHED_NAMES:
            selected=params.get(name,params.get('frontier_mpc',{'frontier_trial':0}))
        for s in scenarios:
            if s.scenario_id in model['train_ids']:
                raise ValueError('Evaluation scenario overlaps fitted training scenarios')
            for h in horizons:
                p=make_policy(name,model,selected,config,horizon=h)
                cfg=SimulationConfig(h,config['initial_wealth'],cost)
                began=time.perf_counter()
                result=simulate(s,p,p.initial_target,cfg)
                elapsed=time.perf_counter()-began
                row=compute_metrics(result); row['fold']=fold
                row['simulation_seconds']=elapsed
                row['policy_inference_seconds']=getattr(p,'inference_seconds',None)
                row['policy_decision_calls']=getattr(p,'decision_calls',None)
                row['fallback_count']=sum(any('INVALID' in code for code in r.reason_codes) for r in result.records)
                output.append(row)
                if keep_records:
                    for r in result.records:
                        row=asdict(r)
                        for field in ('target_weights','pre_trade_holdings','post_trade_holdings','end_holdings'):
                            values=row.pop(field)
                            row.update({field+'_'+asset:v for asset,v in zip(('equity','bond','bill'),values)})
                        row.update(scenario_id=s.scenario_id,policy=name,cost_rate=cost,fold=fold,simulation_horizon=h)
                        row['action']=r.action.value
                        row['reason_codes']=json.dumps(row['reason_codes'])
                        records.append(row)
    return pd.DataFrame(output),pd.DataFrame(records)

def tune(scenarios,config,seed):
    by_id={s.scenario_id:s for s in scenarios}
    groups=folds(by_id,config['inner_folds'],seed)
    collected=[]; audits=[]; expert_scores=[]; frontier_scores=[]
    for inner,valid_ids in enumerate(groups):
        train=[s for sid,s in by_id.items() if sid not in valid_ids]
        valid=[by_id[sid] for sid in valid_ids]
        model=fit_model(train,config)
        audits.append(dict(inner_fold=inner,train_ids=model['train_ids'],validation_ids=valid_ids))
        if frontier_enabled(config):
            for trial in range(len(config['frontier_research']['trials'])):
                metrics,_=evaluate(valid,model,{n:{'frontier_trial':trial} for n in FRONTIER_NAMES},config,
                    names=list(FRONTIER_NAMES)+['balanced_annual'],keep_records=False)
                metrics['trial']=trial; frontier_scores.append(metrics)
        for trial,params in enumerate(config['tuning_grid']):
            metrics,_=evaluate(valid,model,{n:params for n in TUNED},config,names=list(TUNED),
                                horizons=[30],keep_records=False)
            metrics['trial']=trial
            collected.append(metrics)
        if enabled(config):
            expert_model=dict(model,experts=fit_expert_bundle(train,model,config))
            audits[-1]['expert_training_ids']=expert_model['experts']['train_ids']
            audits[-1]['expert_seeds']=config['expert_research']['seeds']
            for trial,penalty in enumerate(config['expert_research']['drawdown_penalties']):
                expert_params={n:{'drawdown_penalty':penalty} for n in EXPERT_NAMES}
                metrics,_=evaluate(valid,expert_model,expert_params,config,
                    names=list(EXPERT_NAMES)+['balanced_annual'],horizons=[30],keep_records=False)
                metrics['trial']=trial
                expert_scores.append(metrics)
            print('  Inner fold',inner+1,'expert training complete',flush=True)
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
    if expert_scores:
        expert_chosen,expert_rows=select_experts(pd.concat(expert_scores,ignore_index=True),config)
        chosen.update(expert_chosen); rows.extend(expert_rows)
    if frontier_scores:
        fs=pd.concat(frontier_scores,ignore_index=True)
        for name in FRONTIER_NAMES:
            candidates=[]; records=[]
            for trial in range(len(config['frontier_research']['trials'])):
                g=fs[(fs.policy==name)&(fs.trial==trial)]
                main=g[g.horizon==30]
                ref=fs[(fs.policy=='balanced_annual')&(fs.trial==trial)&(fs.horizon==30)]
                dg=float(main.real_cagr.median()-ref.real_cagr.median())
                dp=float(main.real_cagr.quantile(.1)-ref.real_cagr.quantile(.1))
                dd=float(main.real_max_drawdown.quantile(.9)-ref.real_max_drawdown.quantile(.9))
                settings=dict(maximum_growth_loss=.005,maximum_p10_loss=.002,minimum_p90_reduction=.03)
                settings.update(config.get('expert_research',{}))
                violation=(max(0.,-dg-settings['maximum_growth_loss'])/settings['maximum_growth_loss']
                           +max(0.,-dp-settings['maximum_p10_loss'])/settings['maximum_p10_loss']
                           +max(0.,dd+settings['minimum_p90_reduction'])/settings['minimum_p90_reduction'])
                utility=float((g.real_cagr-config['selection_drawdown_penalty']*g.real_max_drawdown).mean())
                candidates.append((violation,-utility,trial))
                records.append(dict(policy=name,trial=trial,constraint_violation=violation,
                    risk_selection_feasible=violation<=1e-12,all_horizon_utility=utility,
                    median_growth_difference=dg,p10_growth_difference=dp,p90_loss_difference=dd))
            selected=min(candidates)[-1]
            chosen[name]={'frontier_trial':selected}
            rows.extend(dict(**r,selected=r['trial']==selected) for r in records)
    if cushion_enabled(config):
        added,extra,_=tune_cushion(scenarios,config,seed)
        chosen.update(added); rows.extend(extra)
    if config.get('calibrated_research',{}).get('enabled',False):
        added,extra,_=tune_calibrated(scenarios,config,seed)
        chosen.update(added); rows.extend(extra)
    return chosen,rows,audits


def tune_calibrated(scenarios,config,seed):
    by_id={s.scenario_id:s for s in scenarios}; frames=[]; audits=[]
    basic=dict(config,frontier_research={'enabled':False},expert_research={'enabled':False})
    for inner,ids in enumerate(folds(by_id,config['inner_folds'],seed)):
        train=[s for sid,s in by_id.items() if sid not in ids]; valid=[by_id[sid] for sid in ids]
        model=fit_model(train,basic)
        audits.append(dict(inner_fold=inner,train_ids=model['train_ids'],validation_ids=ids))
        for trial,gamma in enumerate(config['calibrated_research']['gammas']):
            scores,_=evaluate(valid,model,{'calibrated_gate':{'gamma':gamma}},config,
                names=['calibrated_gate','balanced_annual'],keep_records=False)
            scores['trial']=trial; frames.append(scores)
    scores=pd.concat(frames,ignore_index=True); settings=dict(maximum_growth_loss=.005,
        maximum_p10_loss=.002,minimum_p90_reduction=.03)
    settings.update(config.get('expert_research',{})); rows=[]; candidates=[]
    for trial,gamma in enumerate(config['calibrated_research']['gammas']):
        g=scores[(scores.policy=='calibrated_gate')&(scores.trial==trial)]
        m=g[g.horizon==30]; ref=scores[(scores.policy=='balanced_annual')&(scores.trial==trial)&(scores.horizon==30)]
        dg=float(m.real_cagr.median()-ref.real_cagr.median())
        dp=float(m.real_cagr.quantile(.1)-ref.real_cagr.quantile(.1))
        dd=float(m.real_max_drawdown.quantile(.9)-ref.real_max_drawdown.quantile(.9))
        violation=(max(0,-dg-settings['maximum_growth_loss'])/settings['maximum_growth_loss']
            +max(0,-dp-settings['maximum_p10_loss'])/settings['maximum_p10_loss']
            +max(0,dd+settings['minimum_p90_reduction'])/settings['minimum_p90_reduction'])
        utility=float((g.real_cagr-config['selection_drawdown_penalty']*g.real_max_drawdown).mean())
        candidates.append((violation,-utility,-gamma,trial))
        rows.append(dict(policy='calibrated_gate',trial=trial,gamma=gamma,constraint_violation=violation,
            risk_selection_feasible=violation<=1e-12,all_horizon_utility=utility,
            median_growth_difference=dg,p10_growth_difference=dp,p90_loss_difference=dd))
    selected=min(candidates)[-1]
    for r in rows: r['selected']=r['trial']==selected
    return {'calibrated_fixed':{'gamma':4.},'calibrated_gate':{'gamma':config['calibrated_research']['gammas'][selected]}},rows,audits


def tune_cushion(scenarios,config,seed):
    """New-family tuning only; expensive legacy fits need not be repeated."""
    by_id={s.scenario_id:s for s in scenarios}; frames=[]; audits=[]
    basic=dict(config,frontier_research={'enabled':False},expert_research={'enabled':False})
    for inner,ids in enumerate(folds(by_id,config['inner_folds'],seed)):
        train=[s for sid,s in by_id.items() if sid not in ids]; valid=[by_id[sid] for sid in ids]
        model=fit_model(train,basic)
        audits.append(dict(inner_fold=inner,train_ids=model['train_ids'],validation_ids=ids))
        for trial,budget in enumerate(config['cushion_research']['loss_budgets']):
            scores,_=evaluate(valid,model,{name:{'loss_budget':budget} for name in CUSHION_NAMES},config,
                names=list(CUSHION_NAMES)+['balanced_annual'],keep_records=False)
            scores['trial']=trial; frames.append(scores)
    scores=pd.concat(frames,ignore_index=True)
    settings=dict(maximum_growth_loss=.005,maximum_p10_loss=.002,minimum_p90_reduction=.03)
    settings.update(config.get('expert_research',{})); rows=[]; chosen={}
    for name in CUSHION_NAMES:
        candidates=[]; records=[]
        for trial,budget in enumerate(config['cushion_research']['loss_budgets']):
            g=scores[(scores.policy==name)&(scores.trial==trial)]
            main=g[g.horizon==30]; ref=scores[(scores.policy=='balanced_annual')&(scores.trial==trial)&(scores.horizon==30)]
            dg=float(main.real_cagr.median()-ref.real_cagr.median())
            dp=float(main.real_cagr.quantile(.1)-ref.real_cagr.quantile(.1))
            dd=float(main.real_max_drawdown.quantile(.9)-ref.real_max_drawdown.quantile(.9))
            violation=(max(0,-dg-settings['maximum_growth_loss'])/settings['maximum_growth_loss']
                +max(0,-dp-settings['maximum_p10_loss'])/settings['maximum_p10_loss']
                +max(0,dd+settings['minimum_p90_reduction'])/settings['minimum_p90_reduction'])
            utility=float((g.real_cagr-config['selection_drawdown_penalty']*g.real_max_drawdown).mean())
            candidates.append((violation,-utility,-budget,trial))
            records.append(dict(policy=name,trial=trial,loss_budget=budget,constraint_violation=violation,
                risk_selection_feasible=violation<=1e-12,all_horizon_utility=utility,
                median_growth_difference=dg,p10_growth_difference=dp,p90_loss_difference=dd))
        selected=min(candidates)[-1]
        chosen[name]={'loss_budget':config['cushion_research']['loss_budgets'][selected]}
        rows.extend(dict(**r,selected=r['trial']==selected) for r in records)
    return chosen,rows,audits


def select_experts(scores,config):
    """Declared risk-first inner selection; violations remain visible in artifacts."""
    settings=config['expert_research']; chosen={}; rows=[]
    for name in EXPERT_NAMES:
        candidates=[]
        for trial,penalty in enumerate(settings['drawdown_penalties']):
            block=scores[scores.trial==trial]
            g=block[block.policy==name]; ref=block[block.policy==settings['primary_reference']]
            growth=float(g.real_cagr.median()-ref.real_cagr.median())
            p10=float(g.real_cagr.quantile(.1)-ref.real_cagr.quantile(.1))
            loss=float(g.real_max_drawdown.quantile(.9)-ref.real_max_drawdown.quantile(.9))
            violation=(max(0.,-growth-settings['maximum_growth_loss'])/settings['maximum_growth_loss']
                       +max(0.,-p10-settings['maximum_p10_loss'])/settings['maximum_p10_loss']
                       +max(0.,loss+settings['minimum_p90_reduction'])/settings['minimum_p90_reduction'])
            row=dict(policy=name,trial=trial,drawdown_penalty=penalty,
                median_growth_difference=growth,p10_growth_difference=p10,p90_loss_difference=loss,
                constraint_violation=violation,risk_selection_feasible=violation<=1e-12)
            rows.append(row)
            candidates.append((violation,loss,-growth,-penalty,trial))
        selected=min(candidates)[-1]
        chosen[name]={'drawdown_penalty':settings['drawdown_penalties'][selected]}
        for row in rows:
            if row['policy']==name: row['selected']=row['trial']==selected
    return chosen,rows


def fit_selected(scenarios,config,params):
    model=fit_model(scenarios,config)
    if enabled(config):
        penalties=[params[n]['drawdown_penalty'] for n in LEARNED_NAMES]
        model['experts']=fit_expert_bundle(scenarios,model,config,penalties)
    return model

def suitability(metrics,config,balanced_policy=None):
    profiles=json.loads((ROOT/'configs/profiles.json').read_text())
    rows=[]; risk=[]
    balanced_policy=balanced_policy or config.get('balanced_policy','balanced_annual')
    for p in profiles:
        candidates=[]
        for portfolio in PORTFOLIOS:
            actual_policy=balanced_policy if portfolio=='balanced' else portfolio+'_annual'
            g=metrics[(metrics.policy==actual_policy)&(metrics.horizon==p['horizon'])&
                      np.isclose(metrics.cost_rate,config['cost_rate'])]
            if len(g)==0: raise ValueError('Missing suitability horizon')
            k=int((g.real_max_drawdown>p['loss_tolerance']).sum())
            # Bonferroni over 3 candidate portfolios for each profile, to allow portfolio selection.
            confidence=1-(1-config['suitability_confidence'])/3
            upper=binomial_upper(k,len(g),confidence)
            liquidity_ok=p['liquidity']!='high' or portfolio=='conservative'
            horizon_ok=portfolio!='growth' or p['horizon']>=15
            eligible=upper<=config['suitability_max_breach_probability'] and liquidity_ok and horizon_ok
            fixed={'equal_annual','equal_hold','bills_hold','conservative_annual','balanced_annual','growth_annual',
                   'balanced_hold','balanced_periodic','balanced_threshold'}
            evidence='fixed_policy_iid_screen' if actual_policy in fixed else 'overlapping_oof_diagnostic'
            risk.append(dict(profile=p['profile_id'],portfolio=portfolio,policy=actual_policy,evidence=evidence,horizon=p['horizon'],n=len(g),
                breaches=k,breach_probability=k/len(g),upper_bound=upper,confidence=confidence,
                tolerance=p['loss_tolerance'],eligible=eligible,liquidity_ok=liquidity_ok,horizon_ok=horizon_ok,
                median_growth=g.real_cagr.median()))
            if eligible: candidates.append((g.real_cagr.median(),portfolio,actual_policy))
        choice=max(candidates)[1] if candidates else 'human_review'
        actual=max(candidates)[2] if candidates else 'human_review'
        reason=('Choose the eligible portfolio with highest median real growth; loss tolerance is not guaranteed.'
                if candidates else 'No portfolio passes the conservative development risk screen and liquidity/horizon rules.')
        rows.append(dict(**p,portfolio=choice,policy=actual,explanation=reason,
            evidence='Development screen only; learned-policy OOF bounds are diagnostic, not certified coverage.'))
    return pd.DataFrame(rows),pd.DataFrame(risk)

def run_development(config,data_path,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    scenarios,data_hash=load_partition(data_path)
    dump(out/'config.json',config)
    ids=[s.scenario_id for s in scenarios]
    info=manifest(config,data_hash,ids,'development')
    info['evaluated_policies']=policy_names(config)
    dump(out/'manifest.json',info)
    rows=real_rows(scenarios)
    pd.DataFrame(rows,columns=['equity_real','bond_real','bill_real']).describe(percentiles=[.01,.1,.5,.9,.99]).to_csv(out/'data_summary.csv')
    pd.DataFrame(rows,columns=['equity_real','bond_real','bill_real']).corr().to_csv(out/'data_correlations.csv')
    all_metrics=[]; all_records=[]; audit=[]; trial_rows=[]; fitted=[]; seed_metrics=[]
    for outer,valid_ids in enumerate(folds(ids,config['outer_folds'],config['seed'])):
        train=[s for s in scenarios if s.scenario_id not in valid_ids]
        valid=[s for s in scenarios if s.scenario_id in valid_ids]
        params,trials,inner_audit=tune(train,config,config['seed']+outer+1)
        model=fit_selected(train,config,params)
        print('Outer fold',outer+1,'selected',params,flush=True)
        fitted.append(dict(fold=outer,model=model,params=params))
        audit.append(dict(outer_fold=outer,train_ids=model['train_ids'],validation_ids=valid_ids,inner=inner_audit))
        trial_rows.extend(dict(outer_fold=outer,**r) for r in trials)
        metrics,records=evaluate(valid,model,params,config,fold=outer)
        all_metrics.append(metrics); all_records.append(records)
        fold_metrics=[metrics]
        if enabled(config):
            seed_params={n:params['moe_rl'] for n in SEED_NAMES}
            sm,_=evaluate(valid,model,seed_params,config,names=list(SEED_NAMES),horizons=[30],fold=outer,keep_records=False)
            seed_metrics.append(sm)
        for cost in config['sensitivity_costs']:
            metrics,records=evaluate(valid,model,params,config,cost=cost,horizons=[30],fold=outer)
            all_metrics.append(metrics); all_records.append(records)
            fold_metrics.append(metrics)
        summarize(pd.concat(fold_metrics,ignore_index=True)).to_csv(out/f'fold_{outer}_summary.csv',index=False)
        dump(out/f'fold_{outer}_checkpoint.json',dict(model=model,params=params,split=audit[-1],
            trials=trials,code_sha256=info['code_sha256'],config_sha256=info['config_sha256']))
    metrics=pd.concat(all_metrics,ignore_index=True)
    records=pd.concat(all_records,ignore_index=True)
    metrics.to_csv(out/'scenario_metrics.csv',index=False)
    records.to_csv(out/'decisions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    dump(out/'splits.json',audit); dump(out/'fitted_models.json',fitted)
    diagnostics=[]
    for f in fitted:
        for trial,fit in f['model'].get('frontier',{}).get('fits',{}).items():
            diagnostics.extend(dict(fold=f['fold'],trial=trial,**row) for row in fit['diagnostics'])
    if diagnostics:
        pd.DataFrame(diagnostics).to_csv(out/'frontier_training_diagnostics.csv',index=False)
    diagnostic_records=records[records.period>1].copy()
    diagnostic_records['legal_actions']=pd.to_numeric(diagnostic_records.reason_codes.str.extract(r'LEGAL_ACTIONS_(\d+)',expand=False),errors='coerce')
    diagnostic_records.groupby(['policy','simulation_horizon','cost_rate']).agg(
        review_periods=('period','size'),mean_legal_actions=('legal_actions','mean'),
        forced_hold_fraction=('legal_actions',lambda x:float((x==1).mean())),
        trading_fraction=('action',lambda x:float(x.isin(['rebalance','change_target']).mean()))
    ).to_csv(out/'action_diagnostics.csv')
    pd.DataFrame(trial_rows).to_csv(out/'tuning_trials.csv',index=False)
    if seed_metrics:
        seed_frame=pd.concat(seed_metrics,ignore_index=True)
        seed_frame.to_csv(out/'seed_metrics.csv',index=False)
        summarize(seed_frame).to_csv(out/'seed_summary.csv',index=False)
        trace=[]
        for f in fitted:
            for penalty,fit in f['model']['experts']['fits'].items():
                trace.extend(dict(fold=f['fold'],drawdown_penalty=penalty,**row) for row in fit['trace'])
        pd.DataFrame(trace).to_csv(out/'training_trace.csv',index=False)
    summary=pareto_flags(summarize(metrics)); summary.to_csv(out/'summary.csv',index=False)
    paired=paired_comparisons(metrics,config['seed'],config['bootstrap']); paired.to_csv(out/'paired_comparisons.csv',index=False)
    assignments,risks=suitability(metrics,config)
    assignments.to_csv(out/'suitability.csv',index=False); risks.to_csv(out/'suitability_risk.csv',index=False)
    if enabled(config):
        customer_risk=[]
        for profile in json.loads((ROOT/'configs/profiles.json').read_text()):
            for name in EXPERT_NAMES:
                g=metrics[(metrics.policy==name)&(metrics.horizon==profile['horizon'])&np.isclose(metrics.cost_rate,config['cost_rate'])]
                k=int((g.real_max_drawdown>profile['loss_tolerance']).sum())
                upper=binomial_upper(k,len(g),1-(1-config['suitability_confidence'])/len(EXPERT_NAMES))
                customer_risk.append(dict(profile=profile['profile_id'],policy=name,horizon=profile['horizon'],
                    tolerance=profile['loss_tolerance'],breaches=k,n=len(g),breach_probability=k/len(g),
                    upper_bound=upper,risk_screen_pass=upper<=config['suitability_max_breach_probability'],
                    median_growth=float(g.real_cagr.median()),
                    note='仅期限风险诊断，不替代流动性筛选，不自动更改入门推荐。'))
        pd.DataFrame(customer_risk).to_csv(out/'expert_customer_risk.csv',index=False)
    from .validation import comparisons_and_screen, audit_outputs
    comparisons_and_screen(out,config)
    audit_outputs(out)
    dump(out/'runtime.json',dict(seconds=time.perf_counter()-started))
    from .report import generate_report
    if config.get('generate_report',True): generate_report(out)
    return out

def freeze(config,data_path,policy,output,development_run):
    if policy not in policy_names(config):
        raise ValueError('Unknown candidate')
    prior=json.loads((Path(development_run)/'manifest.json').read_text())
    if prior['phase']!='development' or prior['code_sha256']!=code_digest() or prior['config_sha256']!=digest(config):
        raise ValueError('Development run must match current code and configuration')
    scenarios,data_hash=load_partition(data_path)
    if data_hash!=prior['partition_sha256']:
        raise ValueError('Development data changed')
    params,trials,audit=tune(scenarios,config,config['seed']+1000)
    model=fit_selected(scenarios,config,params)
    metrics_file=Path(development_run)/'scenario_metrics.csv'
    if 'evaluated_policies' in prior:
        assignments,risks=suitability(pd.read_csv(metrics_file),config,balanced_policy=policy)
    else:
        assignments=pd.read_csv(Path(development_run)/'suitability.csv')
        risks=pd.DataFrame()
    comparators=['equal_annual','conservative_annual','balanced_annual','growth_annual','balanced_hold']
    if enabled(config): comparators.append('expert_greedy')
    if cushion_enabled(config): comparators+=list(CUSHION_NAMES)
    if config.get('calibrated_research',{}).get('enabled',False): comparators+=list(CALIBRATED_NAMES)
    if frontier_enabled(config):
        comparators+=list(FRONTIER_NAMES[:2])+['frontier_gate']+list(MATCHED_NAMES)
        source=policy if policy in FRONTIER_NAMES else 'frontier_mpc'
        params.update({name:dict(params[source]) for name in MATCHED_NAMES})
    bundle=dict(schema=2,selected_policy=policy,config=config,model=model,params=params,
                code_sha256=code_digest(),dependencies=versions(),development_sha256=data_hash,
                development_run=str(Path(development_run).resolve()),final_scenarios=list(range(121,181)),
                final_comparators=list(dict.fromkeys(comparators)),
                frozen_suitability=assignments.to_dict('records'),
                frozen_suitability_risk=risks.to_dict('records'),
                selection_basis=config.get('selection_basis','Development-only selection; report all limitations.'),
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
    info=manifest(config,final_hash,[s.scenario_id for s in scenarios],'final')
    info['evaluated_policies']=names
    dump(out/'manifest.json',info)
    dump(out/'config.json',config); dump(out/'frozen_bundle.json',raw)
    # Evaluate the development-frozen assignments; do not reassign using final outcomes.
    assignment_rows=[]
    for profile in b['frozen_suitability']:
        row=dict(profile)
        if profile['portfolio']!='human_review':
            actual_policy=profile.get('policy',profile['portfolio']+'_annual')
            group=metrics[(metrics.policy==actual_policy) &
                          (metrics.horizon==profile['horizon']) & np.isclose(metrics.cost_rate,config['cost_rate'])]
            row['final_breach_probability']=float((group.real_max_drawdown>profile['loss_tolerance']).mean())
            row['final_median_growth']=float(group.real_cagr.median())
        assignment_rows.append(row)
    pd.DataFrame(assignment_rows).to_csv(out/'suitability.csv',index=False)
    from .validation import comparisons_and_screen
    comparisons_and_screen(out,config)
    from .validation import audit_outputs
    audit_outputs(out)
    from .report import generate_report
    if config.get('generate_report',True): generate_report(out)
