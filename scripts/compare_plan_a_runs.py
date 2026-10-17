"""Compare saved development runs using a common risk screen, without retuning."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmark'))

import numpy as np
import pandas as pd
from safebench import ROOT
from safebench.metrics import expected_shortfall


def main():
    parser=argparse.ArgumentParser(description='按相同风险线比较计划A历史与当前开发结果，不重新拟合')
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--previous-diagnostics',type=Path,required=True)
    parser.add_argument('--current',type=Path,required=True)
    parser.add_argument('--current-diagnostics',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if not args.out.resolve().is_relative_to((ROOT/'benchmark/outputs').resolve()):
        raise ValueError('Output must remain under benchmark/outputs')
    runs=[args.previous,args.current]; diagnostic=[args.previous_diagnostics,args.current_diagnostics]
    manifests=[json.loads((p/'manifest.json').read_text()) for p in runs]
    assert all(m['phase']=='development' and not m['holdout_used'] for m in manifests)
    assert manifests[0]['partition_sha256']==manifests[1]['partition_sha256']
    configs=[json.loads((p/'config.json').read_text()) for p in runs]
    for key in ['seed','outer_folds','inner_folds','horizons','cost_rate','sensitivity_costs']:
        assert configs[0][key]==configs[1][key]
    config=configs[1]; settings=config['plan_a']['selection']
    target_ratio=config['plan_a']['validation_risk_ratio']
    inner=[]; outer=[]
    for label,run,diag in zip(['previous','current'],runs,diagnostic):
        fits=json.loads((run/'fitted_models.json').read_text())
        ratio={(f['fold'],name):p['risk_ratio'] for f in fits for name,p in f['params'].items() if name.startswith('plan_a_')}
        c=pd.read_csv(diag/'selected_constraints.csv')
        assert len(c)==config['outer_folds']*3*len(config['horizons'])
        assert not c.duplicated(['fold','policy','horizon']).any()
        c['common_limit']=[r.training_budget/ratio[(r.fold,r.policy)]*target_ratio+settings['log_risk_slack']
                           for r in c.itertuples()]
        c['common_risk_pass']=c.log_risk<=c.common_limit+1e-10
        c['common_screen_pass']=c.common_risk_pass & (c.median_growth_loss<=settings['median_growth_loss']+1e-10) & \
            (c.p10_growth_loss<=settings['p10_growth_loss']+1e-10) & (c.review_rate<=settings['maximum_review_rate']+1e-10)
        c['run']=label; inner.append(c)
        m=pd.read_csv(run/'scenario_metrics.csv')
        m=m[np.isclose(m.cost_rate,config['cost_rate']) & m.policy.isin(['plan_a_'+p for p in ['conservative','balanced','growth']])].copy()
        m['common_budget']=[r.plan_a_risk_budget/ratio[(r.fold,r.policy)]*target_ratio for r in m.itertuples()]
        for (policy,h),g in m.groupby(['policy','horizon']):
            risk=expected_shortfall(-np.log1p(-g.real_max_drawdown.to_numpy()),config['plan_a']['alpha'])
            limit=float(g.common_budget.mean()+settings['log_risk_slack'])
            outer.append(dict(run=label,policy=policy,horizon=int(h),log_risk=risk,
                              common_limit=limit,risk_pass=risk<=limit+1e-10))
    keys=['fold','policy','horizon']
    aligned=inner[0][keys+['common_limit']].merge(inner[1][keys+['common_limit']],on=keys,validate='one_to_one')
    np.testing.assert_allclose(aligned.common_limit_x,aligned.common_limit_y,atol=1e-10,rtol=0)
    args.out.mkdir(parents=True,exist_ok=False)
    combined=pd.concat(inner,ignore_index=True)
    combined.to_csv(args.out/'common_inner_risk_screen.csv',index=False)
    pd.DataFrame(outer).to_csv(args.out/'common_outer_risk.csv',index=False)
    summary_keys=['policy','horizon','cost_rate']
    measures=['median_growth','p10_growth','median_loss','p90_loss','mean_trade_rate','mean_real_cost']
    a,b=[pd.read_csv(p/'summary.csv')[summary_keys+measures] for p in runs]
    financial=a.merge(b,on=summary_keys,suffixes=('_previous','_current'),validate='one_to_one')
    assert len(financial)==len(a)==len(b)
    for name in measures:
        financial[name+'_delta']=financial[name+'_current']-financial[name+'_previous']
    financial.to_csv(args.out/'financial_comparison.csv',index=False)
    counts={label:int(g.groupby(['policy','fold']).common_screen_pass.all().sum())
            for label,g in combined.groupby('run')}
    result=dict(all_checks_passed=True,inner_product_fold_passes=counts,
        common_target_ratio=target_ratio,outer_risk_checks=outer,financial_rows=len(financial),
        previous=str(args.previous),current=str(args.current),holdout_used=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Descriptive common-target comparison on reused development data; no refitting or new selection.')
    (args.out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='outer_risk_checks'},indent=2))


if __name__=='__main__':
    main()
