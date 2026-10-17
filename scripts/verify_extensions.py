"""Reconstruct development signals, matched controls and net planning gate diagnostics."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'benchmark')]
import numpy as np
import pandas as pd
from safebench.data import load_partition
from safebench.signals import fit_signals, forecast, real_returns


def verify(run):
    out=Path(run); config=json.loads((out/'config.json').read_text())
    manifest=json.loads((out/'manifest.json').read_text())
    if manifest['phase']!='development' or manifest['scenario_ids']!=list(range(1,121)):
        raise PermissionError('Development partition only')
    models=json.loads((out/'fitted_models.json').read_text())
    metrics=pd.read_csv(out/'scenario_metrics.csv')
    baseline=pd.read_csv(ROOT/'benchmark/outputs/development_plan_a_calibrated_v1/scenario_metrics.csv')
    keys=['policy','scenario_id','horizon','cost_rate']
    old=metrics[metrics.policy.isin(baseline.policy.unique())]
    reference=baseline.merge(old[keys],on=keys,validate='one_to_one').sort_values(keys).reset_index(drop=True)
    old=old.sort_values(keys).reset_index(drop=True)
    columns=[c for c in baseline if c not in keys+['policy_inference_seconds']]
    np.testing.assert_array_equal(old[columns].to_numpy(),reference[columns].to_numpy())
    records=pd.read_csv(out/'decisions.csv.gz')
    forecast_checks=gate_checks=zero_checks=0
    if config.get('plan_b',{}).get('enabled'):
        scenarios,_=load_partition(ROOT/'data/SafeInvest_Synthetic_Data.csv')
        byid={s.scenario_id:s for s in scenarios}
        predictions=pd.read_csv(out/'oof_predictions.csv')
        for f in models:
            fitted=fit_signals([byid[sid] for sid in f['model']['train_ids']])
            for name in fitted['models']:
                for key in fitted['models'][name]:
                    np.testing.assert_allclose(fitted['models'][name][key],f['model']['signals']['models'][name][key],atol=1e-13)
            for row in predictions[predictions.fold==f['fold']].itertuples():
                s=byid[row.scenario_id]
                delta,fallback,prediction=forecast(s.observations[:row.period-1],fitted,row.signal_model,config['plan_b']['ood_limit'])
                assert fallback==row.fallback
                np.testing.assert_allclose(delta,row.delta,atol=1e-14)
                if prediction is not None: np.testing.assert_allclose(prediction,row.prediction,atol=1e-14)
                else: assert pd.isna(row.prediction)
                actual=real_returns(s.observations[row.period-1])
                np.testing.assert_allclose(actual[0]-actual[2],row.actual,atol=1e-14)
                assert row.scenario_id not in fitted['train_ids']
                forecast_checks+=1
        diagnostics=[]
        for signal,group in predictions.groupby('signal_model'):
            denominator=((group.actual-group.constant)**2).sum()
            improvement=1-((group.actual-group.prediction.fillna(group.constant))**2).sum()/denominator
            positive=0
            for _,fold in group.groupby('fold'):
                positive+=int(((fold.actual-fold.prediction.fillna(fold.constant))**2).sum()<((fold.actual-fold.constant)**2).sum())
            diagnostics.append(dict(signal_model=signal,oos_r2=improvement,positive_folds=positive,
                                    fallback_rate=float(group.fallback.mean()),n=len(group)))
        pd.DataFrame(diagnostics).to_csv(out/'signal_diagnostics.csv',index=False)
        a=records[records.policy=='plan_a_balanced'].sort_values(['scenario_id','cost_rate','simulation_horizon','period']).reset_index(drop=True)
        b=records[records.policy=='plan_b_narrow_0'].sort_values(['scenario_id','cost_rate','simulation_horizon','period']).reset_index(drop=True)
        columns=[c for c in records if c not in ('policy','reason_codes','explanation','policy_diagnostics')]
        np.testing.assert_array_equal(a[columns].fillna('NA').to_numpy(),b[columns].fillna('NA').to_numpy()); zero_checks=len(a)
    for row in records[records.policy.str.startswith('plan_c_')].itertuples():
        if pd.isna(row.policy_diagnostics) or row.policy_diagnostics=='null': continue
        d=json.loads(row.policy_diagnostics)
        assert d['scenarios']==len(models[row.fold]['model']['train_ids'])
        assert d['sequence_count']==4**min(d['depth'],row.simulation_horizon-row.period+1)
        np.testing.assert_allclose(d['best_objective'],max(d['first_action_values']),atol=1e-12)
        np.testing.assert_allclose(d['hold_objective'],d['first_action_values'][0],atol=1e-12)
        np.testing.assert_allclose(d['best_objective'],d['mean_log_growth']-d['kappa']*d['log_drawdown_es'],atol=1e-12)
        expected=d['first_cost']+(1-d['first_cost'])*(-np.expm1(d['hold_objective']-d['best_objective']))
        np.testing.assert_allclose(expected,row.estimated_benefit,atol=1e-13)
        np.testing.assert_allclose(d['first_cost'],row.estimated_cost,atol=1e-13)
        if row.action in ('rebalance','change_target'):
            wealth=sum(getattr(row,'pre_trade_holdings_'+a) for a in ('equity','bond','bill'))
            np.testing.assert_allclose(row.transaction_cost/wealth,d['first_cost'],atol=1e-12)
        gate_checks+=1
    state_checks=0
    controls=records[records.policy.str.startswith('plan_c_')]
    for _,group in controls.groupby(['policy','scenario_id','cost_rate','simulation_horizon']):
        peak=config['initial_wealth']; historic=0.; inflation=1.
        for row in group.sort_values('period').itertuples():
            if not pd.isna(row.policy_diagnostics) and row.policy_diagnostics!='null':
                d=json.loads(row.policy_diagnostics)
                np.testing.assert_allclose(d['historic_max_drawdown'],historic,atol=1e-12)
                state_checks+=1
            post=sum(getattr(row,'post_trade_holdings_'+a) for a in ('equity','bond','bill'))/inflation
            historic=max(historic,1-post/peak);peak=max(peak,post)
            historic=max(historic,1-row.real_wealth/peak);peak=max(peak,row.real_wealth)
            inflation=row.cumulative_inflation
    result=dict(all_checks_passed=True,historic_drawdown_state_checks=state_checks,unchanged_baseline_financial_rows=len(old),zero_signal_annual_records=zero_checks,
                reconstructed_oof_predictions=forecast_checks,net_planning_gate_checks=gate_checks,holdout_used=False)
    target=out/'extension_verification.json'
    with target.open('x') as file: json.dump(result,file,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('run'); verify(parser.parse_args().run)
