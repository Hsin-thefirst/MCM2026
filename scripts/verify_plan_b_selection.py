"""Rebuild every Plan B inner-validation candidate on development-only training IDs."""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'benchmark')]
import numpy as np
import pandas as pd
from safebench.data import load_partition
from safebench.policies import fit_model
from safebench.experiment import evaluate
from safebench.plan_a import selection_constraints


def main(run):
    out=Path(run); config=json.loads((out/'config.json').read_text())
    manifest=json.loads((out/'manifest.json').read_text())
    if manifest['phase']!='development' or manifest['scenario_ids']!=list(range(1,121)):
        raise PermissionError('Development scenarios only')
    scenarios,_=load_partition(ROOT/'data/SafeInvest_Synthetic_Data.csv')
    by_id={s.scenario_id:s for s in scenarios}
    models=json.loads((out/'fitted_models.json').read_text())
    splits=json.loads((out/'splits.json').read_text())
    saved=pd.read_csv(out/'tuning_trials.csv')
    checks=[]; details=[]
    for fitted,split in zip(models,splits):
        fold=fitted['fold']; params=fitted['params']['plan_a_balanced']
        fit_config=copy.deepcopy(config)
        # Refit the selected strategic margin only; the solver itself is unchanged.
        fit_config['plan_a']['risk_ratios']=[params['risk_ratio']]
        frames=[]
        for inner in split['inner']:
            train=[by_id[i] for i in inner['train_ids']]
            valid=[by_id[i] for i in inner['validation_ids']]
            model=fit_model(train,fit_config)
            names=[f'plan_b_{family}_{label}' for family in ('narrow','wide') for label in ('0','025','05')]
            block,_=evaluate(valid,model,{n:params for n in names},config,names=names,keep_records=False)
            frames.append(block)
        scores=pd.concat(frames,ignore_index=True)
        for family in ('narrow','wide'):
            candidate_names=[f'plan_b_{family}_{label}' for label in ('0','025','05')]
            ref=scores[scores.policy==candidate_names[0]][['scenario_id','horizon','real_cagr']].rename(columns={'real_cagr':'annual_growth'})
            block=scores[scores.policy.isin(candidate_names)].merge(ref,on=['scenario_id','horizon'],validate='many_to_one')
            ranks=[]
            for trial,name in enumerate(candidate_names):
                b=block[block.policy==name]
                violation,per_horizon=selection_constraints(b,config)
                score=float(np.log1p(b.real_cagr).mean())
                row=saved[(saved.outer_fold==fold)&(saved.policy=='plan_b_'+family)&(saved.trial==trial)].iloc[0]
                np.testing.assert_allclose([violation,score],[row.constraint_violation,row.mean_utility],atol=2e-13)
                self_saved=json.loads(row.horizon_constraints)
                for actual,expected in zip(per_horizon,self_saved):
                    for key in actual:
                        if isinstance(actual[key],(float,int)): np.testing.assert_allclose(actual[key],expected[key],atol=2e-13)
                details.extend(dict(fold=fold,family=family,eta=(0.,.25,.5)[trial],**d) for d in per_horizon)
                ranks.append((trial,violation<=1e-10,violation,score))
            feasible=[r for r in ranks if r[1]]
            if feasible:
                winner=max(feasible,key=lambda r:(r[3],-r[0]))
                means=block.assign(log_growth=np.log1p(block.real_cagr)).groupby(['scenario_id','policy']).log_growth.mean().unstack()
                eligible=[]
                for r in feasible:
                    difference=means[candidate_names[winner[0]]]-means[candidate_names[r[0]]]
                    if difference.mean() <= difference.std(ddof=1)/np.sqrt(len(difference))+1e-12: eligible.append(r)
                winner=min(eligible,key=lambda r:r[0])
            else: winner=min(ranks,key=lambda r:(r[2],-r[3],r[0]))
            chosen=fitted['params']['plan_b_'+family]
            assert chosen['eta']==(0.,.25,.5)[winner[0]]
            assert chosen['signal_selection_feasible']==winner[1]
            checks.append(dict(fold=fold,family=family,eta=chosen['eta'],feasible=winner[1],verified=True))
        print('Reconstructed inner fold',fold+1,flush=True)
    with (out/'signal_selection_verification.json').open('x') as file:
        json.dump(dict(all_checks_passed=True,selections=checks,reconstructed_horizon_checks=len(details)),file,indent=2)
    pd.DataFrame(details).to_csv(out/'signal_inner_constraints.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run');main(parser.parse_args().run)
